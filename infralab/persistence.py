# persistence.py
import logging
import os
import json
import re
import shutil
from datetime import datetime
from typing import Optional, Dict, List
import io
import zipfile

logger = logging.getLogger(__name__)

import pandas as pd
from filelock import FileLock
try:
    import streamlit as st
except ImportError:
    class _StreamlitStub:
        @staticmethod
        def error(*args, **kwargs):
            return None
    st = _StreamlitStub()

from infralab.settings.config import NumpyJSONEncoder
from infralab.core.environment import BridgeEnv

DATA_DIR = "data"
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
FREE_PLAY_DIR = os.path.join(DATA_DIR, "free_play")
DEFAULT_SESSION_ID = "default_free_play_session"
SCENARIOS_FILE = os.path.join(DATA_DIR, "scenarios.json")
# Radice del repo: il file scenari versionato sta un livello sopra il pacchetto.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_SEED_FILE = os.path.join(_REPO_ROOT, "scenarios.json")
PARTICIPANTS_FILE = os.path.join(DATA_DIR, "participants.json")


def ensure_scenarios_file() -> None:
    """Copia il file scenari versionato nella cartella runtime se assente.

    `scenarios.json` alla radice del repo e' la sorgente versionata; l'app
    legge sempre la copia runtime in `data/scenarios.json` (modificabile
    dall'admin senza toccare il file sotto controllo versione).
    """
    if os.path.exists(SCENARIOS_FILE):
        return
    if os.path.exists(SCENARIOS_SEED_FILE):
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copyfile(SCENARIOS_SEED_FILE, SCENARIOS_FILE)


ensure_scenarios_file()

GAME_STATE_FILENAME = "game_state.json"
PLAYER_ACTIONS_FILENAME = "player_actions.json"
ACTION_LOG_FILENAME = "log_actions.csv"
BUDGET_LOG_FILENAME = "log_budgets.csv"
KPI_LOG_FILENAME = "log_kpis.csv"
METADATA_FILENAME = "metadata.json"

LOG_FILENAMES = { 'action': ACTION_LOG_FILENAME, 'budget': BUDGET_LOG_FILENAME, 'kpi': KPI_LOG_FILENAME }

# Colonne del log azioni indipendenti dalla taxonomy dei componenti.
BASE_ACTION_LOG_COLS = [ "quarter", "agent", "bridge_id", "bridge_name", "requested_action", "action", "action_status", "action_block_reason", "cost", "min_condition", "bridge_reliability_index", "bridge_failure_probability", "bridge_annual_reliability_index", "bridge_annual_failure_probability", "traffic_fragility_eta", "traffic_fragility_beta", "traffic_annual_failure_rate", "seismic_failure_probability", "proof_load_alpha", "proof_load_passed" ]

DEFAULT_COMPONENT_NAMES = ("impalcato", "pile", "spalle", "appoggi")


def action_log_cols(component_names=DEFAULT_COMPONENT_NAMES) -> List[str]:
    """Schema colonne del log azioni per una data taxonomy di componenti."""
    names = list(component_names)
    return (
        BASE_ACTION_LOG_COLS
        + [f"{c}_cond" for c in names]
        + [f"{c}_obs_{m}" for c in names for m in ("min", "max")]
        + [f"{c}_eff_age" for c in names]
    )


# Schema legacy (asset type "bridge"), mantenuto come costante di modulo.
ACTION_LOG_COLS = action_log_cols()
BUDGET_LOG_COLS = ["quarter", "agent", "budget", "revenue", "costs"]
KPI_LOG_COLS = ["quarter", "agent", "health_index", "backlog_count", "total_asset_value", "network_reliability_index", "network_failure_probability", "network_annual_reliability_index", "network_annual_failure_probability", "proof_loads_passed", "inspection_coverage", "knowledge_gap_count", "active_monitoring_count", "blocked_action_count"]

REQUIRED_BRIDGE_KEYS = ("id", "name", "length", "width", "adt")


class ScenarioValidationError(ValueError):
    """Scenario malformato (chiavi mancanti o riferimenti inesistenti)."""


def validate_scenario(scenario: Dict) -> List[str]:
    """Valida uno scenario e restituisce la lista dei problemi NON fatali.

    Problemi fatali (chiavi obbligatorie mancanti) sollevano
    ScenarioValidationError; problemi recuperabili (es. azione sconosciuta in
    allowed_actions) vengono restituiti come warning e loggati.
    """
    from dataclasses import fields as dc_fields

    from infralab.settings.config import MaintOpsConfig

    warnings_list: List[str] = []
    sid = scenario.get("scenario_id")
    if not sid:
        raise ScenarioValidationError("Scenario senza 'scenario_id'.")
    for bridge in scenario.get("bridges", []):
        missing = [k for k in REQUIRED_BRIDGE_KEYS if k not in bridge]
        if missing:
            raise ScenarioValidationError(
                f"Scenario '{sid}': ponte {bridge.get('name', bridge.get('id', '?'))} "
                f"senza chiavi obbligatorie {missing}."
            )
    known_actions = {f.name for f in dc_fields(MaintOpsConfig)}
    try:
        from infralab.assets import get_asset_type

        spec = get_asset_type(scenario.get("asset_type", "bridge"))
        known_actions |= set(spec.maintenance_ops)
    except KeyError:
        raise ScenarioValidationError(
            f"Scenario '{sid}': asset_type sconosciuto {scenario.get('asset_type')!r}."
        ) from None
    unknown = sorted(set(scenario.get("allowed_actions") or []) - known_actions)
    if unknown:
        msg = f"Scenario '{sid}': azioni sconosciute in allowed_actions {unknown} (ignorate)."
        warnings_list.append(msg)
        logger.warning(msg)
    # Validazione guidata dallo schema proprieta' del tipo (se dichiarato).
    schema = getattr(spec, "asset_property_schema", None) or {}
    for bridge in scenario.get("bridges", []):
        bname = bridge.get("name", bridge.get("id", "?"))
        for key, meta in schema.items():
            value = bridge.get(key)
            if value is None:
                if meta.get("required") and meta.get("default") is None:
                    raise ScenarioValidationError(
                        f"Scenario '{sid}': asset {bname} senza campo obbligatorio '{key}'."
                    )
                continue
            options = meta.get("options")
            if options and str(value) not in [str(o) for o in options]:
                msg = f"Scenario '{sid}': asset {bname}, '{key}'={value!r} fuori dalle opzioni {options}."
                warnings_list.append(msg)
                logger.warning(msg)
            if meta.get("type") == "number":
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    raise ScenarioValidationError(
                        f"Scenario '{sid}': asset {bname}, '{key}'={value!r} non numerico."
                    ) from None
                lo, hi = meta.get("min"), meta.get("max")
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    msg = f"Scenario '{sid}': asset {bname}, '{key}'={v} fuori range [{lo}, {hi}]."
                    warnings_list.append(msg)
                    logger.warning(msg)
    return warnings_list


def load_scenarios(validate: bool = True) -> List[Dict]:
    """Carica gli scenari: file runtime (bootstrap dal seed) + scenari custom.

    Gli scenari custom (creati dal costruttore asset nella UI) vivono in
    `data/custom_scenarios/*.json`, uno per file, con `"source": "custom"`;
    un id custom in collisione con uno scenario base viene ignorato con warning.
    """
    ensure_scenarios_file()
    scenarios: List[Dict] = []
    if os.path.exists(SCENARIOS_FILE):
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    base_ids = {s.get("scenario_id") for s in scenarios}
    for custom in load_custom_scenarios():
        if custom.get("scenario_id") in base_ids:
            logger.warning("Scenario custom '%s' ignorato: id in collisione con uno scenario base.",
                           custom.get("scenario_id"))
            continue
        scenarios.append(custom)
    if validate:
        for scenario in scenarios:
            validate_scenario(scenario)
    return scenarios


# ---------------------------------------------------------------------------
# Scenari custom (costruttore asset)
# ---------------------------------------------------------------------------

CUSTOM_SCENARIOS_DIR = os.path.join(DATA_DIR, "custom_scenarios")

_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _custom_scenario_path(scenario_id: str) -> str:
    if not _CUSTOM_ID_RE.match(scenario_id or ""):
        raise ScenarioValidationError(
            f"scenario_id non valido: {scenario_id!r} (ammessi lettere, numeri, '-', '_'; max 64)."
        )
    return os.path.join(CUSTOM_SCENARIOS_DIR, f"{scenario_id}.json")


def load_custom_scenarios() -> List[Dict]:
    """Tutti gli scenari custom su disco (marcati source=custom), ordinati per id."""
    if not os.path.isdir(CUSTOM_SCENARIOS_DIR):
        return []
    out: List[Dict] = []
    for fname in sorted(os.listdir(CUSTOM_SCENARIOS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CUSTOM_SCENARIOS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            scenario["source"] = "custom"
            out.append(scenario)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Scenario custom illeggibile %s: %s", path, e)
    return out


def base_scenario_ids() -> set:
    """Id degli scenari base (runtime file), per l'anti-collisione."""
    ensure_scenarios_file()
    if not os.path.exists(SCENARIOS_FILE):
        return set()
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        return {s.get("scenario_id") for s in json.load(f)}


def save_custom_scenario(scenario: Dict) -> List[str]:
    """Valida e salva uno scenario custom; ritorna i warning non fatali.

    Solleva ScenarioValidationError per: id malformato, collisione con uno
    scenario base, chiavi obbligatorie mancanti, asset_type ignoto.
    """
    sid = scenario.get("scenario_id")
    warnings_list = validate_scenario(scenario)
    if sid in base_scenario_ids():
        raise ScenarioValidationError(
            f"scenario_id '{sid}' gia' usato da uno scenario base: scegli un altro id."
        )
    path = _custom_scenario_path(sid)
    os.makedirs(CUSTOM_SCENARIOS_DIR, exist_ok=True)
    payload = {k: v for k, v in scenario.items() if k != "source"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    return warnings_list


def delete_custom_scenario(scenario_id: str) -> bool:
    """Elimina uno scenario custom; True se esisteva."""
    path = _custom_scenario_path(scenario_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def get_session_path(session_id: str) -> str:
    if session_id.endswith("_ARCHIVED"):
        return os.path.join(SESSIONS_DIR, session_id)
    if session_id == DEFAULT_SESSION_ID:
        return FREE_PLAY_DIR
    return os.path.join(SESSIONS_DIR, session_id)

def initialize_session_storage(session_id: str):
    os.makedirs(get_session_path(session_id), exist_ok=True)


def session_lock(session_id: str) -> FileLock:
    """Lock di file per gli accessi concorrenti alla stessa sessione.

    Necessario per le partite multi-giocatore sullo stesso Space: stato di
    gioco, azioni dei giocatori e append CSV passano tutti da qui.
    """
    initialize_session_storage(session_id)
    return FileLock(os.path.join(get_session_path(session_id), ".session.lock"), timeout=15)


# --- Versioning dello stato di gioco -----------------------------------------
GAME_STATE_SCHEMA_VERSION = 1


def migrate_game_state(data: Dict) -> Dict:
    """Migra un game_state.json a GAME_STATE_SCHEMA_VERSION (idempotente)."""
    version = data.get("schema_version", 0)
    if version < 1:
        # v0 (pre-InfraLab): niente asset_type ne' schema_version.
        data.setdefault("asset_type", "bridge")
        data["schema_version"] = 1
    return data

def create_zip_archive(session_ids: List[str]) -> bytes:
    """Crea un archivio ZIP in memoria contenente i dati delle sessioni specificate."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for session_id in session_ids:
            session_path = get_session_path(session_id)
            if os.path.isdir(session_path):
                for root, _, files in os.walk(session_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Crea un percorso relativo per l'archivio
                        archive_path = os.path.join(session_id, os.path.relpath(file_path, session_path))
                        zip_file.write(file_path, archive_path)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def save_participants(participants: List[str]):
    """Salva la lista di nomi dei partecipanti in un file JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PARTICIPANTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(participants, f, indent=2)

def load_participants() -> List[str]:
    """Carica la lista di nomi dei partecipanti, se esiste."""
    if not os.path.exists(PARTICIPANTS_FILE):
        return []
    try:
        with open(PARTICIPANTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning("File partecipanti non leggibile (%s): restituisco lista vuota.", e)
        return []

def archive_session(session_id: str):
    """Archivia una sessione rinominando la sua cartella."""
    if session_id.endswith("_ARCHIVED"):
        return 
    session_path = get_session_path(session_id)
    archived_path = session_path + "_ARCHIVED"
    if os.path.exists(session_path):
        shutil.move(session_path, archived_path)

def find_session(scenario_id: str, player_name: str) -> Optional[str]:
    """Trova la sessione di un giocatore per uno scenario.

    Il nome sessione ha la forma ``{scenario}_{player}_{YYYYMMDD-HHMMSS}``:
    il match richiede il timestamp subito dopo il nome, così "Anna" non
    cattura la sessione di "Anna Maria".
    """
    if not os.path.exists(SESSIONS_DIR):
        return None

    prefix = f"{scenario_id}_{player_name.replace(' ', '_')}"
    pattern = re.compile(re.escape(prefix) + r"_\d{8}-\d{6}(_ARCHIVED)?$")
    for session_name in sorted(os.listdir(SESSIONS_DIR)):
        if pattern.fullmatch(session_name):
            return session_name
    return None

def clear_and_initialize_log_files(session_id: str, component_names=DEFAULT_COMPONENT_NAMES):
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    for filename in LOG_FILENAMES.values():
        filepath = os.path.join(session_path, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    pd.DataFrame(columns=action_log_cols(component_names)).to_csv(os.path.join(session_path, ACTION_LOG_FILENAME), index=False)
    pd.DataFrame(columns=BUDGET_LOG_COLS).to_csv(os.path.join(session_path, BUDGET_LOG_FILENAME), index=False)
    pd.DataFrame(columns=KPI_LOG_COLS).to_csv(os.path.join(session_path, KPI_LOG_FILENAME), index=False)

def save_game_state(env: BridgeEnv, session_id: str):
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    file_path = os.path.join(session_path, GAME_STATE_FILENAME)
    temp_file_path = file_path + ".tmp"
    with session_lock(session_id):
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            json.dump(env.to_dict(), f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)
        os.replace(temp_file_path, file_path)

def load_game_state(session_id: str) -> Optional[BridgeEnv]:
    file_path = os.path.join(get_session_path(session_id), GAME_STATE_FILENAME)
    if not os.path.exists(file_path):
        return None
    try:
        with session_lock(session_id):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return BridgeEnv.from_dict(migrate_game_state(data))
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
        logger.exception("Caricamento stato di gioco fallito per la sessione %s", session_id)
        st.error(f"Errore nel caricamento del file di stato per la sessione {session_id}: {e}.")
        return None

def log_history(env: BridgeEnv, session_id: str, action_costs_by_agent: Dict, revenues: Dict):
    current_quarter_state = env.quarter - 1
    session_path = get_session_path(session_id)
    
    all_action_logs, all_budget_logs, all_kpi_logs = [], [], []
    # Taxonomy dei componenti dall'asset type dell'ambiente (fallback legacy bridge).
    spec = getattr(env, "spec", None)
    component_names = list(spec.component_names) if spec is not None else list(DEFAULT_COMPONENT_NAMES)
    action_cols = action_log_cols(component_names)

    for agent in env.params.agent_names:
        my_bridges = {b.id: b for b in env.bridges.values() if b.manager == agent}
        actions_on_bridges = {ac['bridge_id']: ac for ac in action_costs_by_agent.get(agent, [])}
        diagnostics_by_bridge = {item['bridge_id']: item for item in env.last_plan_diagnostics.get(agent, []) if 'bridge_id' in item}
        for bridge_id, bridge in my_bridges.items():
            action_info = actions_on_bridges.get(bridge_id)
            diagnostic = diagnostics_by_bridge.get(bridge_id, {})
            action_name = action_info['action'] if action_info else "None"
            log_data = {"quarter": current_quarter_state, "agent": agent, "bridge_id": bridge.id, "bridge_name": bridge.name,
                        "requested_action": diagnostic.get('requested_action'),
                        "action": action_name, "action_status": diagnostic.get('status', "not_requested"),
                        "action_block_reason": diagnostic.get('reason'),
                        "cost": action_info['cost'] if action_info else 0, "min_condition": bridge.min_condition,
                        "bridge_reliability_index": bridge.reliability_index, "bridge_failure_probability": bridge.failure_probability,
                        "bridge_annual_reliability_index": bridge.annual_reliability_index, "bridge_annual_failure_probability": bridge.annual_failure_probability,
                        "traffic_fragility_eta": bridge.traffic_fragility.get('eta') if bridge.traffic_fragility else None,
                        "traffic_fragility_beta": bridge.traffic_fragility.get('beta') if bridge.traffic_fragility else None,
                        "traffic_annual_failure_rate": bridge.traffic_fragility.get('annual_failure_rate') if bridge.traffic_fragility else None,
                        "seismic_failure_probability": bridge.seismic_fragility.get('failure_probability') if bridge.seismic_fragility else None,
                        "proof_load_alpha": bridge.last_proof_load_report.get('alpha') if bridge.last_proof_load_report else None,
                        "proof_load_passed": bridge.last_proof_load_report.get('passed') if bridge.last_proof_load_report and bridge.last_proof_load_report.get('quarter') == current_quarter_state else None}
            is_inspected_this_turn = env.maint_ops.get(action_name, {}).get('type') in ['analysis', 'monitoring', 'proof_load']
            for comp_name in component_names:
                comp_obj = bridge.components.get(comp_name)
                if comp_obj:
                    log_data[f'{comp_name}_cond'] = comp_obj.current_condition
                    log_data[f'{comp_name}_eff_age'] = comp_obj.effective_age_years
                    if is_inspected_this_turn and bridge.last_inspection_quarter == current_quarter_state:
                        log_data[f'{comp_name}_obs_min'], log_data[f'{comp_name}_obs_max'] = comp_obj.observed_condition_min, comp_obj.observed_condition_max
                    else:
                        log_data[f'{comp_name}_obs_min'], log_data[f'{comp_name}_obs_max'] = None, None
                else: 
                    for suffix in ['_cond', '_eff_age', '_obs_min', '_obs_max']: log_data[f'{comp_name}{suffix}'] = None
            all_action_logs.append([log_data.get(col) for col in action_cols])
        
        if my_bridges and agent in env.active_agents:
            total_adt = sum(b.adt for b in my_bridges.values())
            reliability_values = [b.posterior_reliability_index if b.posterior_reliability_index is not None else b.reliability_index for b in my_bridges.values()]
            failure_probs = [b.posterior_failure_probability if b.posterior_failure_probability is not None else b.failure_probability for b in my_bridges.values()]
            kpi_data = {"quarter": current_quarter_state, "agent": agent, 
                        "health_index": sum(b.min_condition * b.adt for b in my_bridges.values()) / total_adt if total_adt > 0 else 0,
                        "backlog_count": sum(1 for b in my_bridges.values() if env.params.enforce_guidelines and b.next_inspection_quarter < env.quarter),
                        "total_asset_value": sum(b.current_asset_value for b in my_bridges.values()),
                        "network_reliability_index": min(reliability_values) if reliability_values else 0,
                        "network_failure_probability": max(failure_probs) if failure_probs else 0,
                        "network_annual_reliability_index": min((b.posterior_annual_reliability_index if b.posterior_annual_reliability_index is not None else b.annual_reliability_index) for b in my_bridges.values()) if my_bridges else 0,
                        "network_annual_failure_probability": max((b.posterior_annual_failure_probability if b.posterior_annual_failure_probability is not None else b.annual_failure_probability) for b in my_bridges.values()) if my_bridges else 0,
                        "proof_loads_passed": sum(1 for b in my_bridges.values() if b.last_proof_load_report and b.last_proof_load_report.get('passed') and b.last_proof_load_report.get('quarter') == current_quarter_state),
                        "inspection_coverage": sum(1 for b in my_bridges.values() if b.min_observed_condition() is not None) / len(my_bridges),
                        "knowledge_gap_count": sum(1 for b in my_bridges.values() if b.min_observed_condition() is None),
                        "active_monitoring_count": sum(1 for b in my_bridges.values() if b.monitoring_ends_quarter >= env.quarter),
                        "blocked_action_count": sum(1 for item in env.last_plan_diagnostics.get(agent, []) if item.get('status') == 'blocked')}
            all_kpi_logs.append([kpi_data.get(col) for col in KPI_LOG_COLS])
        
        budget_data = {"quarter": current_quarter_state, "agent": agent, "budget": env.budgets.get(agent, 0), "revenue": revenues.get(agent, 0),
                       "costs": sum(item['cost'] for item in action_costs_by_agent.get(agent, []))}
        all_budget_logs.append([budget_data.get(col) for col in BUDGET_LOG_COLS])
    
    action_log_path = os.path.join(session_path, ACTION_LOG_FILENAME)
    with session_lock(session_id):
        if all_action_logs: pd.DataFrame(all_action_logs, columns=action_cols).to_csv(action_log_path, mode='a', header=not os.path.exists(action_log_path), index=False)
        if all_budget_logs: pd.DataFrame(all_budget_logs, columns=BUDGET_LOG_COLS).to_csv(os.path.join(session_path, BUDGET_LOG_FILENAME), mode='a', header=False, index=False)
        if all_kpi_logs: pd.DataFrame(all_kpi_logs, columns=KPI_LOG_COLS).to_csv(os.path.join(session_path, KPI_LOG_FILENAME), mode='a', header=False, index=False)

def load_history(session_id: str) -> Dict[str, pd.DataFrame]:
    session_path = get_session_path(session_id)
    history_dfs = {}
    for name, filename in LOG_FILENAMES.items():
        file_path = os.path.join(session_path, filename)
        if os.path.exists(file_path):
            try:
                history_dfs[name] = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                history_dfs[name] = pd.DataFrame()
        else:
            history_dfs[name] = pd.DataFrame()
    return history_dfs

def save_player_action(session_id: str, agent_name: str, plan: Dict):
    session_path = get_session_path(session_id)
    file_path = os.path.join(session_path, PLAYER_ACTIONS_FILENAME)
    with session_lock(session_id):
        actions = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f: actions = json.load(f)
            except json.JSONDecodeError:
                logger.warning("player_actions.json corrotto in %s: verra' sovrascritto.", file_path)
        actions[agent_name] = plan
        with open(file_path, 'w') as f: json.dump(actions, f, indent=2)

def load_player_action(session_id: str, agent_name: str):
    file_path = os.path.join(get_session_path(session_id), PLAYER_ACTIONS_FILENAME)
    try:
        with open(file_path, 'r') as f:
            return json.load(f).get(agent_name, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Experiment support — metadata, questionnaire, turn timing
# ---------------------------------------------------------------------------

QUESTIONNAIRE_FILENAME = "questionnaire.json"
TURN_LOG_FILENAME = "log_turns.csv"


def save_experiment_metadata(session_id: str, **kwargs):
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    file_path = os.path.join(session_path, METADATA_FILENAME)
    with session_lock(session_id):
        metadata: Dict = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Metadata esperimento non leggibili (%s): ricreati da zero.", e)
        metadata.update(kwargs)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_experiment_metadata(session_id: str) -> Dict:
    file_path = os.path.join(get_session_path(session_id), METADATA_FILENAME)
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_questionnaire(session_id: str, data: Dict):
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    with open(os.path.join(session_path, QUESTIONNAIRE_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_questionnaire(session_id: str) -> Optional[Dict]:
    file_path = os.path.join(get_session_path(session_id), QUESTIONNAIRE_FILENAME)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


COPILOT_LOG_FILENAME = "copilot_dialogues.json"
RATIONALE_LOG_FILENAME = "decision_rationales.jsonl"


def save_copilot_log(session_id: str, dialogues: Dict):
    """Salva l'intera storia dei dialoghi copilota (explainability Uomo+LLM).

    dialogues: {agent_name: [{'role','content'}, ...]}. Riscritto per intero a
    ogni scambio (piccolo, leggibile per l'analisi qualitativa)."""
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    with open(os.path.join(session_path, COPILOT_LOG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)


def load_copilot_log(session_id: str) -> Optional[Dict]:
    file_path = os.path.join(get_session_path(session_id), COPILOT_LOG_FILENAME)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def append_decision_rationale(session_id: str, quarter: int, agent: str,
                              plan: Dict, rationale: Optional[str] = None):
    """Aggiunge una riga JSONL col razionale della decisione di un trimestre.

    Per gli agenti AI il razionale è la spiegazione prodotta dal modello; per
    l'umano è la motivazione libera opzionale. Dato grezzo per l'explainability.
    """
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    row = {"quarter": quarter, "agent": agent, "timestamp": datetime.now().isoformat(),
           "plan": plan, "rationale": rationale}
    with open(os.path.join(session_path, RATIONALE_LOG_FILENAME), 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, cls=NumpyJSONEncoder) + "\n")


def load_decision_rationales(session_id: str) -> List[Dict]:
    file_path = os.path.join(get_session_path(session_id), RATIONALE_LOG_FILENAME)
    if not os.path.exists(file_path):
        return []
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return rows


def log_turn_time(session_id: str, quarter: int, agent: str, elapsed_seconds: float):
    session_path = get_session_path(session_id)
    initialize_session_storage(session_id)
    file_path = os.path.join(session_path, TURN_LOG_FILENAME)
    row = pd.DataFrame(
        [[quarter, agent, datetime.now().isoformat(), round(elapsed_seconds, 1)]],
        columns=["quarter", "agent", "timestamp_submit", "elapsed_seconds"],
    )
    row.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
