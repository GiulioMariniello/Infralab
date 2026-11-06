"""Sandbox delle policy (policy_sandbox=True): integrita' del benchmark.

Con il sandbox attivo una policy sottoposta al benchmark (scritta a mano, LLM,
ottimizzatore, umano via UI) non riceve mai l'oggetto motore. Riceve un
`PlanningHandle` autosufficiente, costruito dal harness a ogni decisione:

- `get_agent_observations(nome)`: osservazione SANITIZZATA del solo proprietario
  (copia profonda, JSON puro). Contiene cio' che il gestore puo' sapere nella
  modalita' di osservazione dichiarata:
    * observation_mode="full"      -> stato vero (per dichiarazione del benchmark);
    * oracle_observation=True      -> stato vero (concessione esplicita dell'oracolo);
    * observation_mode="inspected" -> solo stime derivate dall'evidenza; in piu'
      rispetto alla vista stimata del motore sono chiusi: danno idraulico e evento
      "danno_idraulico" finche' un'ispezione successiva non lo ha rilevato; referto
      della prova di carico con beta dallo stato stimato (non dal vero); curva di
      monitoraggio proiettata dallo stato stimato; moltiplicatori di processo,
      parametri latenti, vita utile campionata e rischio della prova (calcolato sul
      vero) tolti dai componenti.
  In tutte le modalita' il seme (`game_settings.seed`) e' rimosso.
- `planning_clone()` / `belief_clone()`: copie del "mondo del gestore" per
  pianificatori (GA, memetico, MCTS, tool agentici). Sono costruite da una
  fotografia gia' ripulita (nessun riferimento al motore) con seme proprio: il
  futuro casuale e' indipendente da quello reale anche con rng_streams="keyed".
  Con il sandbox la copia e' sempre quella "belief" (planner_information e'
  irrilevante): in "inspected" senza oracolo lo stato nascosto e' riestratto dalla
  credenza a ogni copia; con oracolo o "full" resta quello vero. In "inspected" la
  fotografia non contiene i danni improvvisi non rilevati (`sudden_pending`) ne' i
  contatori veri del canale (`sudden_stats`); degli eventi regionali resta solo la
  parte pubblica (`regional_events`), non i contatori veri (`regional_stats`); degli
  eventi progressivi restano i soli difetti NOTI del gestore (tipo, trimestre di
  scoperta), senza inizio vero ne' data di fine: via i difetti latenti non rilevati, i
  contatori veri (`progressive_stats`) e il registro dei collassi (`collapse_log`).
- `params`: vista in sola lettura delle regole (senza seme); `maint_ops`: copia.

Qualunque altro accesso (bridges, budgets, to_dict, rng, attributi privati...) solleva
`SandboxViolation` e viene registrato in `violations` (diagnostica best-effort).

Segnali derivati dal vero DENTRO il motore (incertezza epistemica che cresce a un danno
non osservato, alpha e componenti della prova di carico scelti sul vero, vita utile
campionata dei componenti mai osservati, classificazione del regime sul beta vero) non
sono ricalcolati qui: li chiude `strict_information=True`, raccomandato per ogni run di
benchmark con il sandbox (non forzato, per non cambiare le run storiche). Prima di una
campagna: `python -m infralab.lab.leak_check` (controllo "due mondi, stessa evidenza").

Limite dichiarato: l'isolamento e' in-process. Il handle non contiene riferimenti al
motore, ma codice Python arbitrario nello stesso processo puo' comunque risalire agli
oggetti del harness (gc, frame). Per sottomissioni non fidate l'osservazione e' JSON
proprio per poter eseguire la policy fuori processo.
"""
from __future__ import annotations

import copy
import json
import random as _random
from typing import Any, Dict, List, Optional

import numpy as np

from infralab.settings.config import NumpyJSONEncoder


class SandboxViolation(PermissionError):
    """Accesso di una policy a stato non concesso dalla configurazione."""


# Chiavi dei parametri che non devono mai raggiungere una policy sandboxata.
SECRET_PARAM_KEYS = ("seed",)

# Campi dei componenti tolti dalla vista "inspected" sanitizzata: stato nascosto,
# parametri latenti (nella vista stimata sono le medie del profilo, ma il nome del
# campo resta quello del parametro vero), vita utile campionata, moltiplicatore di
# processo dell'ultimo trimestre (vero), rischio della prova di carico (vero).
_LATENT = ("degradation_scale", "degradation_shape_factor", "traffic_sensitivity",
           "environment_sensitivity", "resistance_bias", "demand_bias", "crack_bias")
INSPECTED_DROPPED_COMPONENT_FIELDS = (
    "accumulated_damage", "last_process_multiplier", "vita_utile_anni", "last_proof_load_risk", *_LATENT,
)

_RELIABILITY_CORE = ("reliability_index", "failure_probability", "annual_reliability_index",
                     "annual_failure_probability", "posterior_reliability_index", "posterior_failure_probability",
                     "posterior_annual_reliability_index", "posterior_annual_failure_probability",
                     "reliability_details", "posterior_reliability_details")
_COMPONENT_RELIABILITY_FIELDS = _RELIABILITY_CORE
_BRIDGE_RELIABILITY_FIELDS = (*_RELIABILITY_CORE, "traffic_fragility", "seismic_fragility")

# Tipi d'azione che producono un'osservazione dello stato del ponte.
_OBSERVING_TYPES = ("analysis", "monitoring", "proof_load")

def sandbox_enabled(env) -> bool:
    return bool(getattr(getattr(env, "params", None), "policy_sandbox", False))


def _hide_truth(env) -> bool:
    """True se la policy deve vedere solo la stima del gestore."""
    p = env.params
    return p.observation_mode == "inspected" and not bool(getattr(p, "oracle_observation", False))


def information_mode(env) -> str:
    if bool(getattr(env.params, "oracle_observation", False)):
        return "oracle"
    return "inspected" if env.params.observation_mode == "inspected" else "full"


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, cls=NumpyJSONEncoder))


# ---------------------------------------------------------------------------
# Sanitizzazione di un ponte nella vista "inspected"
# ---------------------------------------------------------------------------

def _op_type(env, action: Optional[str]) -> Optional[str]:
    if not action:
        return None
    op = env.maint_ops.get(action) or getattr(env, "_authority_catalogue", {}).get(action) or {}
    return op.get("type")


def _observed_after(env, bridge_dict: Dict[str, Any], quarter: int) -> bool:
    """Il ponte e' stato osservato (ispezione, monitoraggio, prova di carico,
    rilievo post-intervento) in un trimestre successivo a `quarter`?

    Nel trimestre dell'evento le azioni precedono il campionamento dei pericoli:
    un'ispezione dello stesso trimestre non vede il danno."""
    if int(bridge_dict.get("last_inspection_quarter") or 0) > quarter:
        return True
    for ev in bridge_dict.get("action_history") or []:
        q = int(ev.get("quarter", -1))
        if q <= quarter:
            continue
        status = ev.get("status")
        if status == "completed":            # rilievo post-intervento (_post_maintenance_feedback)
            return True
        if status in ("executed", "imposed_by_regime") and _op_type(env, ev.get("action")) in _OBSERVING_TYPES:
            return True
        if ev.get("action") in ("collasso_e_ricostruzione", "crollo_idraulico", "crollo_progressivo"):
            return True                      # ricostruito: stato di nuovo noto
    return False


def _sanitize_history(env, b: Dict[str, Any]) -> None:
    """Toglie gli eventi di danno idraulico non ancora rilevati e azzera il trimestre."""
    hist = b.get("action_history") or []
    kept = []
    for ev in hist:
        if ev.get("action") == "danno_idraulico" and not _observed_after(env, b, int(ev.get("quarter", 0))):
            continue
        kept.append(ev)
    b["action_history"] = kept
    q = int(b.get("hydraulic_damage_quarter") or 0)
    if q and not _observed_after(env, b, q):
        b["hydraulic_damage_quarter"] = 0


def _sanitize_proof_load_report(report: Optional[Dict[str, Any]], est) -> Optional[Dict[str, Any]]:
    """Referto della prova di carico: esito osservato (passata/scartata, carico
    applicato) + beta ricalcolati sullo stato STIMATO del gestore. Le probabilita'
    di rottura/scarto durante la prova e i beta del referto originale sono calcolati
    sullo stato vero: tolti."""
    if not report:
        return report
    out = {k: report.get(k) for k in ("quarter", "alpha", "step_count", "passed", "screened", "conditioning_mode")
           if k in report}
    comps = []
    for c in report.get("components") or []:
        name = c.get("component")
        ec = est.components.get(name)
        entry = {"component": name, "proof_load_alpha": c.get("proof_load_alpha"),
                 "passed": c.get("passed"), "screened": c.get("screened")}
        if ec is not None:
            entry.update({
                "prior_beta": ec.reliability_index,
                "prior_pf": ec.failure_probability,
                "prior_annual_beta": ec.annual_reliability_index,
                "posterior_beta": ec.posterior_reliability_index if c.get("passed") else None,
                "posterior_annual_beta": ec.posterior_annual_reliability_index if c.get("passed") else None,
            })
        comps.append(entry)
    out["components"] = comps
    out.update({
        "posterior_beta": est.posterior_reliability_index,
        "posterior_pf": est.posterior_failure_probability,
        "posterior_annual_beta": est.posterior_annual_reliability_index,
        "posterior_annual_pf": est.posterior_annual_failure_probability,
        "basis": "assessed",
    })
    return out


def _monitoring_curve(env, est) -> Dict[str, list]:
    op = env.maint_ops.get("monitoring", {}) or {}
    return env.predict_future_degradation(est, op.get("error_sigma", 0.005))


def _sanitize_inspected_bridge(env, bridge, b: Dict[str, Any]) -> Dict[str, Any]:
    est = env._assessed_bridge(bridge)
    _sanitize_history(env, b)
    b["last_proof_load_report"] = _sanitize_proof_load_report(b.get("last_proof_load_report"), est)
    if b.get("future_curve_range") is not None:
        b["future_curve_range"] = _monitoring_curve(env, est)
    for comp in (b.get("components") or {}).values():
        for f in INSPECTED_DROPPED_COMPONENT_FIELDS:
            comp.pop(f, None)
    return b


# ---------------------------------------------------------------------------
# Osservazione sanitizzata
# ---------------------------------------------------------------------------

def sanitized_observation(env, agent_name: str) -> Dict[str, Any]:
    """Osservazione del proprietario ridotta a cio' che la configurazione concede.

    JSON puro (nessun riferimento a oggetti del motore), sempre una copia nuova."""
    obs = env.get_agent_observations(agent_name)
    obs["maintenance_options"] = copy.deepcopy(obs.get("maintenance_options") or {})
    settings = dict(obs.get("game_settings") or {})
    for k in SECRET_PARAM_KEYS:
        settings.pop(k, None)
    obs["game_settings"] = settings
    # Contabilita' propria nota al gestore (le policy sandboxate non leggono il motore).
    obs["game_rules"]["my_emissions_tco2"] = float((getattr(env, "emissions_tco2", {}) or {}).get(agent_name, 0.0))
    if _hide_truth(env):
        by_id = {b.id: b for b in env.bridges.values()}
        obs["my_bridges"] = [_sanitize_inspected_bridge(env, by_id[b["id"]], b) if b.get("id") in by_id else b
                             for b in obs["my_bridges"]]
    obs["information"] = {"policy_sandbox": True, "mode": information_mode(env),
                          "observation_mode": env.params.observation_mode,
                          "oracle_observation": bool(getattr(env.params, "oracle_observation", False))}
    return _jsonable(obs)


# ---------------------------------------------------------------------------
# Fotografia del mondo del gestore (base delle copie dei pianificatori)
# ---------------------------------------------------------------------------

def belief_snapshot(env, agent_name: str, seed: int) -> Dict[str, Any]:
    """Stato serializzato da cui il handle genera le copie dei pianificatori.

    Non contiene il seme del motore (sostituito da `seed`), ne' i segreti della
    configurazione degli agenti; in "inspected" senza oracolo i ponti sono quelli
    STIMATI (stato nascosto a valori di credenza, parametri latenti alle medie del
    profilo), con le stesse chiusure dell'osservazione sanitizzata, e per gli asset
    degli altri gestori anche l'evidenza privata e' tolta."""
    data = env.to_dict()
    data["params"] = dict(data["params"])
    data["params"]["seed"] = int(seed)
    data["agent_config"] = {n: {"type": (c or {}).get("type")} for n, c in (data.get("agent_config") or {}).items()}
    data["failure_stream_salt"] = 0
    data.pop("stream_salt", None)
    data.pop("inspection_counters", None)
    data["last_plan_diagnostics"] = {agent_name: data.get("last_plan_diagnostics", {}).get(agent_name, [])}
    if _hide_truth(env):
        from infralab.core.environment import Component
        exponent = env.params.deterioration_exponent
        data["last_expected_loss"] = {}
        data["earthquake_events"] = [{k: v for k, v in (e or {}).items() if k != "damage_reports"}
                                     for e in data.get("earthquake_events") or []]
        # Danno locale improvviso (RT009): la lista dei danni NON ancora rilevati
        # (componente, trimestre, perdita vera) e i contatori veri del canale (eventi
        # avvenuti, perdita totale, danni riparati prima della scoperta) sono verita'.
        # Tolti: la copia li reinizializza (vuoti / a zero) come un motore nuovo.
        data.pop("sudden_pending", None)
        data.pop("sudden_stats", None)
        # Eventi regionali: l'elenco degli eventi (trimestre, centro, raggio, ponti
        # nell'area) e' pubblico e resta; i danni per ponte stanno nei pendenti (tolti
        # sopra) e i contatori veri del canale (ponti colpiti, perdita) sono verita'.
        data.pop("regional_stats", None)
        # Eventi progressivi: i difetti latenti NON rilevati, l'inizio vero e la data di
        # fine di quelli noti, i contatori veri e il registro dei collassi (causa, danni
        # nascosti al collasso) sono verita'. Restano i difetti noti del gestore (tipo,
        # trimestre di scoperta): la copia riestrae la loro durata residua.
        from infralab.core.hazards import progressive_events as _progressive
        data.pop("progressive_stats", None)
        data.pop("collapse_log", None)
        known = _progressive.redact_for_owner(
            getattr(env, "progressive_defects", None) or {},
            keep_bridge=lambda bid: bid in env.bridges and env.bridges[bid].manager == agent_name)
        data.pop("progressive_defects", None)
        if known:
            data["progressive_defects"] = known
        bridges = {}
        for bid, bridge in env.bridges.items():
            est = env._assessed_bridge(bridge)
            d = est.to_dict()
            d.pop("_is_assessment", None)
            d["id"] = bridge.id
            d["manager"] = bridge.manager
            if bridge.manager == agent_name:
                _sanitize_history(env, d)
                d["last_proof_load_report"] = _sanitize_proof_load_report(d.get("last_proof_load_report"), est)
            else:
                # evidenza privata di un altro gestore: credenza da solo inventario
                # (eta' nominale, parametri medi), niente storico ne' referti.
                d["action_history"] = []
                d["last_proof_load_report"] = None
                d["last_management_feedback"] = None
                d["hydraulic_damage_quarter"] = 0
                d["future_curve_range"] = None
                elapsed = (env.quarter - 1) / 4.0
                for name, comp in d["components"].items():
                    nominal = Component.from_dict(comp, exponent)
                    nominal.observed_condition_min = nominal.observed_condition_max = None
                    nominal.effective_age_years = float(nominal.initial_age_years) + elapsed
                    nominal.update_condition(exponent)
                    comp.update({"current_condition": nominal.current_condition,
                                 "accumulated_damage": nominal.accumulated_damage,
                                 "effective_age_years": nominal.effective_age_years,
                                 "observed_condition_min": None, "observed_condition_max": None})
                    for k in _COMPONENT_RELIABILITY_FIELDS:
                        comp[k] = None
                for k in _BRIDGE_RELIABILITY_FIELDS:
                    d[k] = None
            for comp in d["components"].values():
                comp["last_process_multiplier"] = 1.0
                comp["last_proof_load_risk"] = None
            bridges[bid] = d
        data["bridges"] = bridges
    return _jsonable(data)


# ---------------------------------------------------------------------------
# Handle consegnato alle policy
# ---------------------------------------------------------------------------

class _ParamsView:
    """Vista in sola lettura dei parametri (senza seme)."""

    __slots__ = ("_values", "_handle")

    def __init__(self, values: Dict[str, Any], handle: "PlanningHandle"):
        object.__setattr__(self, "_values", dict(values))
        object.__setattr__(self, "_handle", handle)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in SECRET_PARAM_KEYS:
            object.__getattribute__(self, "_handle")._violation(f"params.{name}")
        if name in values:
            return copy.deepcopy(values[name])
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_handle")._violation(f"params.{name} (scrittura)")

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(object.__getattribute__(self, "_values"))


class PlanningHandle:
    """Interfaccia ristretta che sostituisce il motore nelle policy sandboxate.

    Autosufficiente: costruita dal harness con `PlanningHandle.build(env, agente)`,
    non conserva riferimenti al motore. Espone osservazione, catalogo, regole e copie
    del mondo del gestore."""

    _ALLOWED = {"agent_name", "quarter", "params", "maint_ops", "information", "violations",
                "get_agent_observations", "observation", "planning_clone", "belief_clone",
                "feasible_actions", "rules", "sandboxed"}

    sandboxed = True

    def __init__(self, agent_name: str, observation: Dict[str, Any], snapshot: Dict[str, Any]):
        self.agent_name = agent_name
        self._obs = observation
        self._snapshot = snapshot
        self._base = None
        self.violations: List[str] = []
        self.quarter = int(observation["game_rules"]["quarter"])
        self.params = _ParamsView(observation.get("game_settings") or {}, self)
        self.information = copy.deepcopy(observation.get("information") or {})

    @classmethod
    def build(cls, env, agent_name: str, seed: Optional[int] = None) -> "PlanningHandle":
        """Costruisce il handle dal motore (lato harness). `seed`: seme del mondo del
        gestore; default dal random globale (seedato per run -> riproducibile)."""
        if seed is None:
            seed = _random.getrandbits(31)
        return cls(agent_name, sanitized_observation(env, agent_name), belief_snapshot(env, agent_name, seed))

    # -- diagnostica ---------------------------------------------------------
    def _violation(self, what: str):
        msg = f"policy_sandbox: accesso non consentito a '{what}' (agente {self.agent_name})"
        self.violations.append(what)
        raise SandboxViolation(msg)

    def __getattr__(self, name: str) -> Any:
        # Chiamato solo per attributi assenti: tutto cio' che non e' esposto e' vietato.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._violation(name)

    # -- osservazione e regole ------------------------------------------------
    def get_agent_observations(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        if agent_name is not None and agent_name != self.agent_name:
            self._violation(f"get_agent_observations({agent_name!r})")
        return copy.deepcopy(self._obs)

    observation = get_agent_observations

    @property
    def maint_ops(self) -> Dict[str, Any]:
        return copy.deepcopy(self._obs.get("maintenance_options") or {})

    def rules(self) -> Dict[str, Any]:
        o = self._obs
        return copy.deepcopy({k: o.get(k) for k in ("game_rules", "game_settings", "management_constraints",
                                                    "cda_config", "regime", "information")})

    def feasible_actions(self) -> Dict[str, Dict[str, Any]]:
        """{asset_id: fattibilita' per azione} (dall'osservazione)."""
        return {str(b["id"]): copy.deepcopy(b.get("action_feasibility") or {}) for b in self._obs["my_bridges"]}

    # -- mondo del gestore ------------------------------------------------------
    def _base_env(self):
        if self._base is None:
            from infralab.core.environment import BridgeEnv
            self._base = BridgeEnv.from_dict(copy.deepcopy(self._snapshot), refresh=False)
        return self._base

    def belief_clone(self, salt: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        """Copia del mondo del gestore (vedi BridgeEnv.belief_clone) con futuro casuale
        proprio. Ogni chiamata e' una nuova estrazione."""
        return self._base_env().belief_clone(salt=salt, rng=rng)

    planning_clone = belief_clone


def policy_env(env, agent_name: str):
    """Cio' che una policy riceve al posto del motore: il motore stesso (default,
    comportamento storico) o un PlanningHandle (policy_sandbox=True)."""
    if env is None or isinstance(env, PlanningHandle) or not sandbox_enabled(env):
        return env
    return PlanningHandle.build(env, agent_name)


def is_sandboxed(env) -> bool:
    return isinstance(env, PlanningHandle)


def owner_view(env, asset):
    """Oggetto ponte da mostrare a chi gioca (UI/API) al posto del vero: con il
    sandbox e la sola evidenza ("inspected" senza oracolo) e' la stima del gestore
    (_assessed_bridge), altrimenti il ponte stesso. Sola lettura."""
    if sandbox_enabled(env) and _hide_truth(env):
        return env._assessed_bridge(asset)
    return asset
