"""API FastAPI del laboratorio InfraLab.

Espone il motore `infralab` a un frontend web (React). Le sessioni di
simulazione vivono in memoria nel processo del server; ogni sessione è un
AssetEnv più un agente (adapter dell'agente legacy) per l'avanzamento
automatico dei trimestri.

Avvio dev:
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import os
import statistics
import threading
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

import infralab
import infralab.persistence as persistence
from infralab.agents.adapters import make_agent
from infralab.agents.copilot import Copilot
from infralab.agents.sandbox import owner_view
from infralab.assets import available_asset_types, get_asset_type
from infralab.codes import available_code_frameworks, get_code_framework
from infralab.core.hazards.sudden_damage import UI_FIELDS as SUDDEN_DAMAGE_UI_FIELDS
from infralab.core.hazards.regional_events import UI_FIELDS as REGIONAL_EVENTS_UI_FIELDS
from infralab.core.hazards import progressive_events as _progressive
from infralab.lab.build import build_env_from_scenario
from infralab.lab.runner import run_experiment
from infralab.lab.spec import SpecError, resolve_scenario, spec_from_dict

from backend import research as research_module

app = FastAPI(title="InfraLab API", version=infralab.__version__)
app.include_router(research_module.router)

# Config provider LLM da variabili d'ambiente (Google richiede configure esplicito).
if os.environ.get("GOOGLE_API_KEY"):
    try:
        import google.generativeai as _genai

        _genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except Exception:  # pragma: no cover - dipendenza opzionale
        pass

# Origini CORS configurabili (default aperto per sviluppo).
_cors_origins = [o.strip() for o in os.environ.get("INFRALAB_CORS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tipi di agente AI ammessi.
SESSION_AGENT_TYPES = ("Euristico", "GA Orizzonte Variabile", "Algoritmo Memetico", "MCTS", "RL Q-Learning", "LLM", "LLM Agentico", "Custom")
LLM_AGENT_TYPES = ("LLM", "LLM Agentico")
# "Umano": agente manuale — decide solo tramite i piani inviati dalla UI.
HUMAN_AGENT_TYPE = "Umano"
LLM_PROVIDERS = ("OpenAI", "Google", "HuggingFace", "Locale")
MAX_AGENTS_PER_SESSION = 6

# Limiti degli store in-memory (evicting FIFO del piu' vecchio).
MAX_SESSIONS = 20
MAX_CAMPAIGN_JOBS = 10
MAX_HISTORY_POINTS = 2000


# ---------------------------------------------------------------------------
# Store sessioni in-memory
# ---------------------------------------------------------------------------

class Session:
    """Sessione multi-agente: piu' manager (AI o umani) sulla stessa rete.

    agents: {nome: adapter | None} — None = agente umano, decide solo via piani
    manuali dalla UI; gli adapter AI decidono da soli ad ogni step.
    """

    def __init__(self, sid: str, env, agents: Dict[str, Any], agent_types: Dict[str, str], scenario_id: str,
                 persist_id: Optional[str] = None, player_name: Optional[str] = None,
                 condition: Optional[str] = None):
        self.sid = sid
        self.env = env
        self.agents = agents
        self.agent_types = agent_types
        self.scenario_id = scenario_id
        # Sessione di ricerca: id della cartella persistita in data/sessions/
        # (stato + log CSV salvati ad ogni avanzamento). None = sessione effimera.
        self.persist_id = persist_id
        self.player_name = player_name
        # Condizione sperimentale del giocatore ("Uomo"/"Uomo+LLM"); gate del copilota.
        self.condition = condition
        # Batteria di appartenenza (within-subjects): lista partite + id, se la
        # sessione è una partita di batteria; altrimenti None.
        self.battery: Optional[List[Dict[str, Any]]] = None
        self.battery_id: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        # Dialogo copilota (Uomo+LLM): {agent_name: [{'role','content'}, ...]}
        self.copilot_history: Dict[str, List[Dict[str, str]]] = {}
        # Le richieste FastAPI girano in un threadpool: due chiamate concorrenti
        # sulla stessa sessione non devono mutare l'ambiente insieme.
        self.lock = threading.Lock()

    def persist(self, action_costs: Optional[Dict[str, Any]] = None, revenues: Optional[Dict[str, Any]] = None) -> None:
        """Salva stato e log su disco per le sessioni di ricerca."""
        if not self.persist_id:
            return
        persistence.save_game_state(self.env, self.persist_id)
        if action_costs is not None and revenues is not None:
            persistence.log_history(self.env, self.persist_id, action_costs, revenues)

    @property
    def agent_names(self) -> List[str]:
        return list(self.agents)

    def record_snapshot(self) -> None:
        row = dict(self.aggregate_kpis())
        row["agents"] = {name: self.snapshot_kpis(name) for name in self.agents}
        self.history.append(row)
        if len(self.history) > MAX_HISTORY_POINTS:
            del self.history[: len(self.history) - MAX_HISTORY_POINTS]

    def _asset_kpis(self, assets) -> Dict[str, Any]:
        env = self.env
        # policy_sandbox: chi gioca vede la stima del gestore, non lo stato vero.
        assets = [owner_view(env, a) for a in assets]
        total_adt = sum(a.adt for a in assets) or 1
        betas = [a.reliability_index for a in assets if a.reliability_index is not None]
        ann_pfs = [a.annual_failure_probability for a in assets if a.annual_failure_probability is not None]
        return {
            "quarter": env.quarter,
            "health_index": round(sum(env.get_health_index(a) * a.adt for a in assets) / total_adt, 4) if assets else None,
            "net_asset_value": round(sum(env.get_net_asset_value(a) for a in assets), 2),
            "min_reliability_index": round(min(betas), 3) if betas else None,
            "max_annual_pf": max(ann_pfs) if ann_pfs else None,
        }

    def snapshot_kpis(self, name: str) -> Dict[str, Any]:
        env = self.env
        assets = [a for a in env.bridges.values() if a.manager == name]
        kpis = self._asset_kpis(assets)
        kpis["budget"] = round(float(env.budgets.get(name, 0.0)), 2)
        kpis["active"] = name in env.active_agents
        kpis["asset_count"] = len(assets)
        kpis["emissions_tco2"] = round(float(getattr(env, "emissions_tco2", {}).get(name, 0.0)), 3)
        return kpis

    def aggregate_kpis(self) -> Dict[str, Any]:
        """KPI a livello di rete (tutti gli agenti insieme)."""
        env = self.env
        kpis = self._asset_kpis(list(env.bridges.values()))
        kpis["budget"] = round(sum(float(env.budgets.get(n, 0.0)) for n in self.agents), 2)
        kpis["active"] = any(n in env.active_agents for n in self.agents)
        kpis["active_agents"] = sum(1 for n in self.agents if n in env.active_agents)
        kpis["emissions_tco2"] = round(sum(float(getattr(env, "emissions_tco2", {}).get(n, 0.0)) for n in self.agents), 3)
        return kpis


SESSIONS: Dict[str, Session] = {}


# ---------------------------------------------------------------------------
# Modelli richiesta/risposta
# ---------------------------------------------------------------------------

class AgentSpecModel(BaseModel):
    name: str
    type: str
    # Solo per type == "LLM":
    provider: Optional[str] = None    # OpenAI | Google | Locale
    profile: Optional[str] = None     # stile decisionale
    model: Optional[str] = None       # override del modello
    plugin: Optional[str] = None      # solo type == "Custom": nome del plugin agente


class CreateSessionRequest(BaseModel):
    scenario_id: str
    agent_type: str = "Euristico"          # compat: sessione a singolo agente
    agents: List[AgentSpecModel] = Field(default_factory=list)  # multi-agente (Editor)
    seed: Optional[int] = 42
    seismic_enabled: Optional[bool] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class SeismicToggleRequest(BaseModel):
    enabled: bool


# Campi di configurazione esposti all'UI (curati, con metadati per il form).
CONFIG_FIELDS: List[Dict[str, Any]] = [
    {"key": "initial_budget", "label": "Budget iniziale (€)", "group": "Economia", "type": "number", "step": 100000},
    {"key": "years", "label": "Durata (anni)", "group": "Economia", "type": "number", "min": 1, "max": 100, "step": 1},
    {"key": "unallocated_budget_return_annual", "label": "Rendimento budget annuo", "group": "Economia", "type": "number", "min": 0, "max": 0.2, "step": 0.005},
    {"key": "debt_interest_rate_annual", "label": "Tasso debito annuo", "group": "Economia", "type": "number", "min": 0, "max": 0.3, "step": 0.005},
    {"key": "degradation_model", "label": "Modello di degrado", "group": "Degrado", "type": "select", "options": ["weibull", "gamma"]},
    {"key": "deterioration_exponent", "label": "Esponente curva degrado", "group": "Degrado", "type": "number", "min": 1, "max": 5, "step": 0.1},
    {"key": "deterioration_process_cov", "label": "CoV processo degrado", "group": "Degrado", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
    {"key": "gamma_process_theta", "label": "theta processo gamma", "group": "Degrado", "type": "number", "min": 0.005, "max": 0.5, "step": 0.005},
    {"key": "env_degradation_std", "label": "Dev.std fattore ambientale", "group": "Degrado", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
    {"key": "traffic_load_model", "label": "Modello di carico", "group": "Traffico", "type": "select", "options": ["NTC", "GL_heavy", "GL_medium", "GL_light"]},
    {"key": "traffic_reference_adt", "label": "ADT di riferimento", "group": "Traffico", "type": "number", "min": 100, "step": 1000},
    {"key": "traffic_degradation_factor", "label": "Fattore degrado traffico", "group": "Traffico", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
    {"key": "traffic_growth_annual", "label": "Crescita traffico annua", "group": "Traffico", "type": "number", "min": -0.2, "max": 0.5, "step": 0.01},
    {"key": "code_framework", "label": "Normativa", "group": "Affidabilità", "type": "select", "options": available_code_frameworks()},
    {"key": "reliability_consequence_class", "label": "Classe di conseguenza", "group": "Affidabilità", "type": "select", "options": ["CC1", "CC2", "CC3"]},
    {"key": "reliability_target_beta", "label": "beta target", "group": "Affidabilità", "type": "number", "min": 1.5, "max": 6, "step": 0.1},
    {"key": "reliability_projection_years", "label": "Anni di proiezione", "group": "Affidabilità", "type": "number", "min": 1, "max": 50, "step": 1},
    {"key": "reliability_mc_samples", "label": "Campioni Monte Carlo", "group": "Affidabilità", "type": "number", "min": 64, "max": 2048, "step": 64},
    {"key": "inspection_bias", "label": "Bias ispezione", "group": "Ispezioni", "type": "number", "min": -0.2, "max": 0.2, "step": 0.01},
    {"key": "inspection_pod_enabled", "label": "POD abilitata", "group": "Ispezioni", "type": "bool"},
    {"key": "inspection_pod_exponent", "label": "Esponente POD", "group": "Ispezioni", "type": "number", "min": 1, "max": 20, "step": 1},
    {"key": "climate_degradation_trend_annual", "label": "Trend climatico annuo", "group": "Ispezioni", "type": "number", "min": -0.2, "max": 0.5, "step": 0.01},
    {"key": "earthquake_enabled", "label": "Sismica abilitata", "group": "Sismica", "type": "bool"},
    {"key": "seismic_hazard_peak_pga", "label": "PGA picco (g)", "group": "Sismica", "type": "number", "min": 0, "max": 1, "step": 0.01},
    {"key": "seismic_hazard_far_pga", "label": "PGA fondo (g)", "group": "Sismica", "type": "number", "min": 0, "max": 0.5, "step": 0.01},
    {"key": "seismic_hazard_decay_km", "label": "Decadimento (km)", "group": "Sismica", "type": "number", "min": 5, "max": 300, "step": 5},
    {"key": "seismic_hazard_center_lat", "label": "Centro hazard lat", "group": "Sismica", "type": "number", "step": 0.1},
    {"key": "seismic_hazard_center_lon", "label": "Centro hazard lon", "group": "Sismica", "type": "number", "step": 0.1},
    {"key": "earthquake_annual_prob", "label": "Prob. sisma annua", "group": "Sismica", "type": "number", "min": 0, "max": 1, "step": 0.01},
    # Danno locale improvviso (idraulico, urto, natante, fatica, tipi utente): metadati
    # condivisi con la UI Streamlit, definiti accanto al modello.
    *SUDDEN_DAMAGE_UI_FIELDS,
    # Eventi estremi regionali (piena di bacino, tempesta): stessi metadati della UI.
    *REGIONAL_EVENTS_UI_FIELDS,
    # Eventi progressivi (latenti), registro dei collassi e regola di fine partita.
    *_progressive.UI_FIELDS,
    # Informazione e integrita' del benchmark (tutti opt-in; default = storico).
    {"key": "observation_mode", "label": "Modalità di osservazione", "group": "Benchmark", "type": "select", "options": ["full", "inspected"]},
    {"key": "oracle_observation", "label": "Oracolo (stato vero agli agenti)", "group": "Benchmark", "type": "bool"},
    {"key": "assessment_epistemic_mode", "label": "Incertezza nella stima del gestore", "group": "Benchmark", "type": "select", "options": ["manager", "reference"]},
    {"key": "posting_entry_rate_q", "label": "Declassamento di portata: ingressi/trim.", "group": "Benchmark", "type": "number", "min": 0, "max": 1, "step": 1e-4},
    {"key": "posting_severity_floor", "label": "Declassamento di portata: severita' minima", "group": "Benchmark", "type": "number", "min": 0, "max": 2, "step": 0.05},
    {"key": "posting_lift_on_intervention", "label": "Declassamento revocato dai lavori strutturali", "group": "Benchmark", "type": "bool"},
    {"key": "temporary_closure_condition_rate_q", "label": "Chiusura temporanea: ingressi/trim. legati allo stato", "group": "Benchmark", "type": "number", "min": 0, "max": 0.5, "step": 1e-5},
    {"key": "temporary_closure_rate_q", "label": "Chiusura temporanea: ingressi/trim.", "group": "Benchmark", "type": "number", "min": 0, "max": 1, "step": 1e-5},
    {"key": "temporary_closure_mean_duration_quarters", "label": "Chiusura temporanea: durata media (trimestri)", "group": "Benchmark", "type": "number", "min": 1, "max": 80, "step": 1},
    {"key": "planner_information", "label": "Mondo dei pianificatori", "group": "Benchmark", "type": "select", "options": ["truth", "belief"]},
    {"key": "policy_sandbox", "label": "Sandbox delle policy", "group": "Benchmark", "type": "bool"},
    {"key": "strict_information", "label": "Informazione stretta (nessun segnale dal vero)", "group": "Benchmark", "type": "bool"},
    {"key": "strict_uncertainty_drop", "label": "Calo osservato letto come danno (informazione stretta)", "group": "Benchmark", "type": "number", "min": 0, "max": 1, "step": 0.01},
    {"key": "revenue_health_source", "label": "Salute che riduce i ricavi", "group": "Benchmark", "type": "select", "options": ["truth", "assessed", "none"]},
    {"key": "detour_km_cap", "label": "Tetto ai km di deviazione (0 = nessuno)", "group": "Benchmark", "type": "number", "min": 0},
    {"key": "detour_partial_diversion", "label": "Limitazione parziale: chi devia", "group": "Benchmark", "type": "select", "options": ["all", "trucks"]},
    {"key": "detour_works_diversion", "label": "I lavori deviano il traffico", "group": "Benchmark", "type": "bool"},
    {"key": "planner_objective", "label": "Obiettivo dei pianificatori", "group": "Benchmark", "type": "select", "options": ["asset_value", "risk_adjusted"]},
    {"key": "planner_safety_weight", "label": "Peso della sicurezza (lambda)", "group": "Benchmark", "type": "number"},
    {"key": "planner_terminal_quarters", "label": "Orizzonte terminale dei pianificatori (trimestri)", "group": "Benchmark", "type": "number"},
]


class CampaignAgentSpec(BaseModel):
    id: str                              # etichetta della condizione (agent_id nei run)
    type: str                            # Euristico | LLM | LLM Agentico | Custom | ...
    provider: Optional[str] = None       # LLM: OpenAI | Google | Locale
    profile: Optional[str] = None        # LLM: stile decisionale
    model: Optional[str] = None          # LLM: override modello
    plugin: Optional[str] = None         # Custom: nome plugin agente
    max_rounds: Optional[int] = None     # LLM Agentico: round di codice


class CampaignRequest(BaseModel):
    scenario_id: str
    # Retrocompat: lista di stringhe (tipo=etichetta) oppure spec ricche con config.
    agents: List[Any] = Field(default_factory=lambda: ["Euristico"])
    horizon_quarters: int = Field(default=16, ge=2, le=60)
    replicates: int = Field(default=3, ge=1, le=15)
    seed: int = 20260702
    # Sweep opzionale di un parametro: {"param": "env_degradation_std", "values": [0.08, 0.16]}
    grid_param: Optional[str] = None
    grid_values: List[float] = Field(default_factory=list)
    # Confronto normative: lista di code_framework da confrontare (categoriale).
    compare_frameworks: List[str] = Field(default_factory=list)


# Parametri ammessi per lo sweep (evita input arbitrari sul motore).
CAMPAIGN_SWEEP_PARAMS = {
    "env_degradation_std", "deterioration_exponent", "deterioration_process_cov",
    "traffic_degradation_factor", "reliability_target_beta", "gamma_process_theta",
    "traffic_growth_annual", "climate_degradation_trend_annual",
}
MAX_CAMPAIGN_RUNS = 60


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=200)
    # Piani manuali per agente: {agent_name: {asset_id: action}}. Applicati solo
    # al primo trimestre del batch; gli agenti AI decidono da soli ad ogni step.
    plans: Dict[str, Dict[str, str]] = Field(default_factory=dict)


class ActRequest(BaseModel):
    # Piano manuale: {asset_id (str): action_name}. Vuoto = nessun intervento.
    # Compat single-view: ogni azione viene instradata al manager dell'asset.
    plan: Dict[str, str] = Field(default_factory=dict)
    # Motivazione libera opzionale della decisione (raccolta per explainability).
    rationale: Optional[str] = None


class CopilotRequest(BaseModel):
    message: str
    agent_name: Optional[str] = None      # default: primo agente umano
    provider: str = "Locale"
    model: Optional[str] = None


def _seismic_view(asset) -> Optional[Dict[str, Any]]:
    sf = getattr(asset, "seismic_fragility", None)
    if not sf:
        return None
    return {
        "site_pga": _round(sf.get("site_pga"), 4),
        "distance_to_hazard_km": _round(sf.get("distance_to_hazard_km"), 1),
        "controlling_pga": _round(sf.get("controlling_pga"), 4),
        "failure_probability": sf.get("failure_probability"),
        "enabled": sf.get("enabled", False),
    }


def _asset_view(env, real_asset) -> Dict[str, Any]:
    # La fattibilità delle azioni è valutata rispetto al manager dell'asset
    # (budget, cooldown, occupato): corretto anche in multi-agente.
    feas = env.get_feasible_actions_for_bridge(real_asset.manager, real_asset)
    # policy_sandbox: condizioni, beta e Pf mostrati sono quelli che il gestore conosce.
    asset = owner_view(env, real_asset)
    actions = [
        {"action": name, "type": info.get("type"), "cost": round(info.get("cost", 0.0), 2),
         "feasible": info.get("feasible", False), "reason": info.get("reason")}
        for name, info in feas.items()
    ]
    # Difetti progressivi NOTI (tipo, trimestre di scoperta): chiave presente solo con il canale acceso.
    known = ({"known_progressive_defects": _progressive.public_view(env, real_asset)}
             if _progressive.enabled(env.params) else {})
    return {
        **known,
        "id": real_asset.id,
        "name": asset.name,
        "manager": asset.manager,
        "status": asset.status,
        "busy": asset.active_maintenance is not None,
        "min_condition": round(asset.min_condition, 4),
        "health_index": round(env.get_health_index(asset), 4),
        "reliability_index": None if asset.reliability_index is None else round(asset.reliability_index, 3),
        "annual_failure_probability": asset.annual_failure_probability,
        "net_asset_value": round(env.get_net_asset_value(asset), 2),
        "risk_priority_index": round(getattr(asset, "risk_priority_index", 0.0) or 0.0, 6),
        "network_importance": round(getattr(asset, "network_importance_factor", 1.0) or 1.0, 3),
        "cda": getattr(asset, "cda", None),
        "seismic": _seismic_view(asset),
        "lat": asset.lat,
        "lon": asset.lon,
        "components": {
            n: {"condition": round(c.current_condition, 4),
                "effective_age_years": round(c.effective_age_years, 2)}
            for n, c in asset.components.items()
        },
        "actions": actions,
    }


def _asset_detail(env, real_asset) -> Dict[str, Any]:
    """Dettaglio affidabilità: livello ponte + per componente + proiezione degrado."""
    asset = owner_view(env, real_asset)   # policy_sandbox: stima del gestore
    details = asset.reliability_details or {}
    lcp = (details.get("life_cycle_profile") or {}).get("profile", [])
    projection = [
        {"year": p.get("year"),
         "beta": _round(p.get("structural_annual_beta")),
         "pf": p.get("structural_annual_pf")}
        for p in lcp
    ]
    components = []
    for name, comp in asset.components.items():
        cd = comp.reliability_details or {}
        cprofile = (cd.get("life_cycle_profile") or {}).get("profile", [])
        limit_states = {
            ls_name: {
                "beta_annual": _round(ls.get("annual_beta")),
                "annual_pf": ls.get("annual_failure_probability"),
                "margin_annual": _round(ls.get("beta_margin_annual")),
                "category": ls.get("category"),
            }
            for ls_name, ls in (cd.get("limit_state_results") or {}).items()
        }
        components.append({
            "name": name,
            "condition": round(comp.current_condition, 4),
            "effective_age_years": round(comp.effective_age_years, 2),
            "beta_annual": _round(comp.annual_reliability_index),
            "annual_pf": comp.annual_failure_probability,
            "posterior_beta_annual": _round(comp.posterior_annual_reliability_index),
            "governing_limit_state": cd.get("governing_limit_state"),
            "limit_states": limit_states,
            "degradation": [
                {"year": p.get("year"), "condition": _round(p.get("condition")),
                 "beta": _round(p.get("structural_annual_beta"))}
                for p in cprofile
            ],
        })
    return {
        "id": real_asset.id,
        "name": asset.name,
        "beta_annual": _round(asset.annual_reliability_index),
        "annual_pf": asset.annual_failure_probability,
        "posterior_beta_annual": _round(asset.posterior_annual_reliability_index),
        "target_beta_annual": _round((details or {}).get("target_beta_annual")),
        "projection": projection,
        "components": components,
    }


def _round(v, d: int = 3):
    return None if v is None else round(float(v), d)


def _action_catalog(env) -> List[Dict[str, Any]]:
    """Elenco delle azioni di manutenzione disponibili nella sessione."""
    return [
        {"action": name, "type": op.get("type"),
         "duration_quarters": op.get("duration_quarters", 0),
         "revenue_impact": op.get("revenue_impact", 0.0)}
        for name, op in env.maint_ops.items()
    ]


def _session_horizon(env) -> int:
    """Orizzonte fisso della partita in trimestri (max_quarters o years·4)."""
    return int(env.params.max_quarters or env.params.years * 4)


def _state(session: Session) -> Dict[str, Any]:
    env = session.env
    horizon = _session_horizon(env)
    any_active = any(n in env.active_agents for n in session.agents)
    # Partita conclusa: orizzonte raggiunto (truncation) o nessun agente attivo
    # (fallimento). Uniforme per tutte le condizioni.
    finished = (env.quarter > horizon) or (not any_active)
    state = {
        "session_id": session.sid,
        "scenario_id": session.scenario_id,
        "asset_type": env.asset_type,
        "condition": session.condition,
        "copilot_enabled": session.condition == "Uomo+LLM",
        "horizon_quarters": horizon,
        "cadence_quarters": max(1, int(getattr(env.params, "human_cadence_quarters", 1) or 1)),
        "finished": finished,
        "agents": [
            {
                "name": name,
                "type": session.agent_types.get(name, "?"),
                "is_human": session.agents.get(name) is None,
                "kpis": session.snapshot_kpis(name),
            }
            for name in session.agents
        ],
        "action_catalog": _action_catalog(env),
        "kpis": session.aggregate_kpis(),
        "assets": [_asset_view(env, a) for a in sorted(env.bridges.values(), key=lambda b: b.id)],
        "history": session.history,
    }
    block = _battery_block(session)
    if block is not None:
        state["battery"] = block
    if getattr(env.params, "collapse_game_over", "none") != "none":
        # regola di fine partita al collasso: motivo dell'uscita del gestore
        state["failure_info"] = env.last_failure_info
    return state


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": infralab.__version__, "sessions": len(SESSIONS)}


@app.get("/api/asset-types")
def asset_types() -> List[Dict[str, Any]]:
    out = []
    for name in available_asset_types():
        spec = get_asset_type(name)
        out.append({
            "name": spec.name,
            "display_name": spec.display_name,
            "components": list(spec.component_names),
            "structural_components": sorted(spec.structural_components),
            "ui_meta": dict(spec.ui_meta),
        })
    return out


@app.get("/api/asset-types/{name}/schema")
def asset_type_schema(name: str) -> Dict[str, Any]:
    """Schema completo di un tipo: guida il form del costruttore asset."""
    try:
        spec = get_asset_type(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Asset type '{name}' sconosciuto")
    ops = {
        op_name: {k: v for k, v in op.items() if k != "gains"} | {"gains": op.get("gains", {})}
        for op_name, op in spec.maintenance_ops.items()
    }
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "components": list(spec.component_names),
        "structural_components": sorted(spec.structural_components),
        "property_schema": dict(spec.asset_property_schema),
        "ui_meta": dict(spec.ui_meta),
        "hazard_sensitivities": dict(spec.hazard_sensitivities),
        "component_default_lifetimes": dict(spec.component_default_lifetimes),
        "maintenance_ops": ops,
    }


# ---------------------------------------------------------------------------
# Scenari custom (costruttore asset)
# ---------------------------------------------------------------------------

class CustomScenarioRequest(BaseModel):
    scenario_id: str
    name: str = ""
    description: str = ""
    asset_type: str = "bridge"
    params: Dict[str, Any] = Field(default_factory=dict)
    allowed_actions: Optional[List[str]] = None
    bridges: List[Dict[str, Any]] = Field(default_factory=list)

    def to_scenario(self) -> Dict[str, Any]:
        scenario = {
            "scenario_id": self.scenario_id.strip(),
            "name": self.name.strip() or self.scenario_id.strip(),
            "description": self.description,
            "asset_type": self.asset_type,
            "params": self.params,
            "bridges": self.bridges,
        }
        if self.allowed_actions is not None:
            scenario["allowed_actions"] = self.allowed_actions
        return scenario


MAX_CUSTOM_SCENARIO_ASSETS = 50


def _check_custom_scenario(req: CustomScenarioRequest) -> Dict[str, Any]:
    scenario = req.to_scenario()
    if not 1 <= len(scenario["bridges"]) <= MAX_CUSTOM_SCENARIO_ASSETS:
        raise HTTPException(status_code=400,
                            detail=f"Numero asset fuori range (1–{MAX_CUSTOM_SCENARIO_ASSETS}).")
    ids = [b.get("id") for b in scenario["bridges"]]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="Id asset mancanti o duplicati.")
    return scenario


MAX_IMPORT_BYTES = 4 * 1024 * 1024   # 4 MB


@app.get("/api/asset-types/{name}/template.xlsx")
def asset_type_template(name: str) -> Response:
    """Modello Excel precompilato per l'import del parco asset di un tipo."""
    from infralab.data.excel_import import build_template_xlsx

    try:
        spec = get_asset_type(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Asset type '{name}' sconosciuto")
    data = build_template_xlsx(spec)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="infralab_{name}_template.xlsx"'},
    )


@app.post("/api/asset-import")
async def asset_import(asset_type: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parsa un Excel/CSV compilato in asset per il tipo dato (nessun salvataggio)."""
    from infralab.data.excel_import import parse_asset_workbook

    try:
        spec = get_asset_type(asset_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Asset type '{asset_type}' sconosciuto")
    raw = await file.read()
    if len(raw) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="File troppo grande (max 4 MB).")
    try:
        result = parse_asset_workbook(raw, file.filename or "", spec)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File non leggibile: {e}")
    return {"asset_type": asset_type, **result}


# ---------------------------------------------------------------------------
# Plugin utente (agenti + modelli di degrado in codice Python)
# ---------------------------------------------------------------------------

PLUGIN_KINDS = ("agent", "degradation")
MAX_PLUGIN_SOURCE_BYTES = 64 * 1024


class PluginRequest(BaseModel):
    name: str
    source: str


class PluginValidateRequest(BaseModel):
    kind: str
    source: str


def _check_plugin_kind(kind: str) -> None:
    if kind not in PLUGIN_KINDS:
        raise HTTPException(status_code=404, detail=f"Tipo plugin sconosciuto: {kind!r} (validi: {list(PLUGIN_KINDS)}).")


@app.get("/api/plugins/templates")
def plugin_templates() -> Dict[str, str]:
    from infralab.plugins.templates import TEMPLATES

    return dict(TEMPLATES)


@app.post("/api/plugins/validate")
def validate_plugin(req: PluginValidateRequest) -> Dict[str, Any]:
    from infralab.plugins.loader import validate_agent, validate_degradation

    _check_plugin_kind(req.kind)
    validator = validate_agent if req.kind == "agent" else validate_degradation
    ok, message = validator(req.source)
    return {"ok": ok, "message": message}


@app.get("/api/plugins/{kind}")
def list_plugins(kind: str) -> List[Dict[str, str]]:
    from infralab.plugins import registry

    _check_plugin_kind(kind)
    return registry.list_plugins(kind)


@app.get("/api/plugins/{kind}/{name}")
def get_plugin(kind: str, name: str) -> Dict[str, str]:
    from infralab.plugins import registry

    _check_plugin_kind(kind)
    try:
        return registry.get_plugin(kind, name)
    except registry.PluginError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/plugins/{kind}")
def save_plugin(kind: str, req: PluginRequest) -> Dict[str, str]:
    from infralab.plugins import registry

    _check_plugin_kind(kind)
    if len(req.source.encode("utf-8")) > MAX_PLUGIN_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Codice troppo lungo (max 64 KB).")
    try:
        return registry.save_plugin(kind, req.name.strip(), req.source)
    except registry.PluginError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/plugins/{kind}/{name}")
def delete_plugin(kind: str, name: str) -> Dict[str, str]:
    from infralab.plugins import registry

    _check_plugin_kind(kind)
    try:
        existed = registry.delete_plugin(kind, name)
    except registry.PluginError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not existed:
        raise HTTPException(status_code=404, detail=f"Plugin '{name}' ({kind}) non trovato.")
    return {"deleted": name, "kind": kind}


@app.get("/api/custom-scenarios")
def list_custom_scenarios() -> List[Dict[str, Any]]:
    return persistence.load_custom_scenarios()


@app.post("/api/custom-scenarios")
def save_custom_scenario(req: CustomScenarioRequest) -> Dict[str, Any]:
    """Crea o aggiorna uno scenario custom (upsert per scenario_id)."""
    scenario = _check_custom_scenario(req)
    try:
        warnings_list = persistence.save_custom_scenario(scenario)
    except persistence.ScenarioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"scenario_id": scenario["scenario_id"], "warnings": warnings_list}


@app.post("/api/custom-scenarios/validate")
def validate_custom_scenario(req: CustomScenarioRequest) -> Dict[str, Any]:
    """Valida senza salvare: errori fatali come 400, warning nel payload."""
    scenario = _check_custom_scenario(req)
    try:
        warnings_list = persistence.validate_scenario(scenario)
    except persistence.ScenarioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"scenario_id": scenario["scenario_id"], "warnings": warnings_list}


@app.delete("/api/custom-scenarios/{scenario_id}")
def delete_custom_scenario(scenario_id: str) -> Dict[str, Any]:
    try:
        existed = persistence.delete_custom_scenario(scenario_id)
    except persistence.ScenarioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not existed:
        raise HTTPException(status_code=404, detail=f"Scenario custom '{scenario_id}' non trovato")
    return {"deleted": scenario_id}


@app.get("/api/config")
def config() -> Dict[str, Any]:
    """Schema dei parametri configurabili + valori di default.

    Le opzioni di `degradation_model` includono i plugin di degrado salvati
    (`custom:<nome>`), così l'utente può selezionare il proprio modello.
    """
    from dataclasses import asdict as _asdict

    from infralab.plugins import registry
    from infralab.settings.config import SimParameters
    defaults = _asdict(SimParameters())
    fields = [dict(f) for f in CONFIG_FIELDS]
    custom_models = [f"custom:{p['name']}" for p in registry.list_plugins("degradation")]
    if custom_models:
        for f in fields:
            if f["key"] == "degradation_model":
                f["options"] = list(f.get("options", [])) + custom_models
    return {
        "fields": fields,
        "defaults": {f["key"]: defaults.get(f["key"]) for f in fields},
    }


@app.get("/api/codes")
def codes() -> List[Dict[str, Any]]:
    """Normative disponibili (target β per classe di conseguenza)."""
    out = []
    for name in available_code_frameworks():
        fw = get_code_framework(name)
        out.append({
            "name": fw.name,
            "display_name": fw.display_name,
            "source": fw.source,
            "beta_reference_by_cc": fw.beta_reference_by_cc,
            "reference_period_years": fw.reference_period_years,
        })
    return out


@app.get("/api/scenarios")
def scenarios() -> List[Dict[str, Any]]:
    out = []
    for s in persistence.load_scenarios(validate=False):
        out.append({
            "scenario_id": s.get("scenario_id"),
            "name": s.get("name", s.get("scenario_id")),
            "description": s.get("description", ""),
            "asset_type": s.get("asset_type", "bridge"),
            "asset_count": len(s.get("bridges", [])),
            "allowed_actions": s.get("allowed_actions"),
            "source": s.get("source", "base"),
        })
    return out


def _normalize_agent_specs(req: CreateSessionRequest) -> Dict[str, Dict[str, Any]]:
    """Valida e normalizza la composizione agenti: {nome: config}."""
    if req.agents:
        specs = [(a.name.strip(), a) for a in req.agents]
    else:
        specs = [("Agent", AgentSpecModel(name="Agent", type=req.agent_type))]
    if not 1 <= len(specs) <= MAX_AGENTS_PER_SESSION:
        raise HTTPException(status_code=400, detail=f"Numero agenti fuori range (1–{MAX_AGENTS_PER_SESSION}).")
    names = [n for n, _ in specs]
    if any(not n for n in names):
        raise HTTPException(status_code=400, detail="Nome agente vuoto.")
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="Nomi agente duplicati.")
    valid_types = set(SESSION_AGENT_TYPES) | {HUMAN_AGENT_TYPE}
    result: Dict[str, Dict[str, Any]] = {}
    for name, spec in specs:
        if spec.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Tipo agente non ammesso: {spec.type!r}. Validi: {sorted(valid_types)}")
        config: Dict[str, Any] = {"type": spec.type}
        if spec.type in LLM_AGENT_TYPES:
            provider = spec.provider or "Locale"
            if provider not in LLM_PROVIDERS:
                raise HTTPException(status_code=400, detail=f"Provider LLM non ammesso: {provider!r}. Validi: {list(LLM_PROVIDERS)}")
            if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
                raise HTTPException(status_code=400, detail="OPENAI_API_KEY non configurata sul server.")
            if provider == "Google" and not os.environ.get("GOOGLE_API_KEY"):
                raise HTTPException(status_code=400, detail="GOOGLE_API_KEY non configurata sul server.")
            config.update({"provider": provider, "profile": spec.profile or "Bilanciato"})
            if spec.model:
                config["model"] = spec.model
        elif spec.type == "Custom":
            if not spec.plugin:
                raise HTTPException(status_code=400, detail=f"Agente '{name}': tipo Custom richiede 'plugin'.")
            from infralab.plugins.registry import PluginError, load_agent_decider
            try:
                load_agent_decider(spec.plugin)   # verifica esistenza + validità
            except PluginError as e:
                raise HTTPException(status_code=400, detail=f"Agente '{name}': plugin non valido — {e}")
            config["plugin"] = spec.plugin
        result[name] = config
    return result


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    agent_configs = _normalize_agent_specs(req)
    scenario = next(
        (s for s in persistence.load_scenarios(validate=False) if s.get("scenario_id") == req.scenario_id),
        None,
    )
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' non trovato")

    allowed_keys = {f["key"] for f in CONFIG_FIELDS}
    overrides = {k: v for k, v in (req.params or {}).items() if k in allowed_keys}
    if req.seismic_enabled is not None:
        overrides["earthquake_enabled"] = bool(req.seismic_enabled)
    try:
        env = build_env_from_scenario(
            scenario, seed=req.seed,
            param_overrides=overrides or None,
            agents=agent_configs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parametri non validi: {e}")

    # Adapter AI per gli agenti non umani; gli umani decidono via piani manuali.
    agents: Dict[str, Any] = {}
    for name, config in agent_configs.items():
        if config["type"] == HUMAN_AGENT_TYPE:
            agents[name] = None
        else:
            adapter = make_agent(name, config)
            adapter.reset(env.get_agent_observations(name), {"env": env})
            agents[name] = adapter

    agent_types = {name: config["type"] for name, config in agent_configs.items()}
    sid = uuid.uuid4().hex[:12]
    session = Session(sid, env, agents, agent_types, req.scenario_id)
    session.record_snapshot()

    # Evict FIFO oltre il limite (le sessioni sono effimere, in-memory).
    while len(SESSIONS) >= MAX_SESSIONS:
        oldest = next(iter(SESSIONS))
        del SESSIONS[oldest]
    SESSIONS[sid] = session
    return _state(session)


# ---------------------------------------------------------------------------
# Batteria di partite (within-subjects)
# ---------------------------------------------------------------------------

def _battery_id(token: str) -> str:
    return hashlib.sha1(token.encode()).hexdigest()[:12]


def _battery_persist_id(battery_id: str, index: int) -> str:
    return f"BATTERY_{battery_id}_g{index}"


def _game_is_finished(persist_id: str, horizon: int) -> bool:
    env = persistence.load_game_state(persist_id)
    if env is None:
        return False
    active = [n for n in env.params.agent_names if n in env.active_agents]
    return (env.quarter > horizon) or (not active)


def _battery_current_index(battery_id: str, battery: List[Dict[str, Any]]) -> int:
    """Indice della prima partita non ancora marcata 'done' (len = completata)."""
    for g in battery:
        meta = persistence.load_experiment_metadata(_battery_persist_id(battery_id, g["index"]))
        if not meta.get("game_done"):
            return g["index"]
    return len(battery)


def _battery_games_public(battery: List[Dict[str, Any]], battery_id: str) -> List[Dict[str, Any]]:
    out = []
    for g in battery:
        meta = persistence.load_experiment_metadata(_battery_persist_id(battery_id, g["index"]))
        out.append({"index": g["index"], "length_class": g["length_class"],
                    "condition": g["condition"], "done": bool(meta.get("game_done"))})
    return out


def _open_battery_game(battery_id: str, player_name: str, game: Dict[str, Any]) -> "Session":
    """Crea o riprende la sessione persistita di una partita della batteria."""
    persist_id = _battery_persist_id(battery_id, game["index"])
    # Riusa una sessione web già aperta per questa partita.
    for existing in SESSIONS.values():
        if existing.persist_id == persist_id:
            return existing
    env = persistence.load_game_state(persist_id)
    if env is None:
        scenario = next((s for s in persistence.load_scenarios(validate=False)
                         if s.get("scenario_id") == game["scenario_id"]), None)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"Scenario '{game['scenario_id']}' non trovato")
        env = build_env_from_scenario(scenario, agents={player_name: {"type": HUMAN_AGENT_TYPE}})
        persistence.clear_and_initialize_log_files(persist_id, component_names=env.spec.component_names)
        persistence.save_game_state(env, persist_id)
        persistence.save_experiment_metadata(
            persist_id, experiment_phase="playing", scenario_id=game["scenario_id"],
            condition=game["condition"], player_name=player_name, battery_id=battery_id,
            game_index=game["index"], length_class=game["length_class"], game_done=False)
    sid = uuid.uuid4().hex[:12]
    session = Session(sid, env, {player_name: None}, {player_name: HUMAN_AGENT_TYPE},
                      game["scenario_id"], persist_id=persist_id, player_name=player_name,
                      condition=game["condition"])
    session.record_snapshot()
    while len(SESSIONS) >= MAX_SESSIONS:
        del SESSIONS[next(iter(SESSIONS))]
    SESSIONS[sid] = session
    return session


def _battery_block(session: "Session") -> Optional[Dict[str, Any]]:
    """Blocco batteria pubblico per _state (None se non è una partita di batteria)."""
    if not session.battery or not session.battery_id:
        return None
    cur = _battery_current_index(session.battery_id, session.battery)
    return {
        "battery_id": session.battery_id, "total": len(session.battery),
        "current_index": cur, "complete": cur >= len(session.battery),
        "games": _battery_games_public(session.battery, session.battery_id),
    }


def _battery_response(token: str, battery: List[Dict[str, Any]], player_name: str) -> Dict[str, Any]:
    battery_id = _battery_id(token)
    cur = _battery_current_index(battery_id, battery)
    game = battery[min(cur, len(battery) - 1)]
    session = _open_battery_game(battery_id, player_name, game)
    # Collega la sessione alla batteria: così ogni _state (anche da act/step)
    # riporta il blocco batteria.
    session.battery = battery
    session.battery_id = battery_id
    consent = persistence.load_experiment_metadata(session.persist_id).get("consent_given", False)
    return {**_state(session), "player_name": player_name, "consent_given": bool(consent)}


class BatteryAdvanceRequest(BaseModel):
    token: str


@app.post("/api/battery/advance")
def battery_advance(req: BatteryAdvanceRequest) -> Dict[str, Any]:
    """Chiude la partita corrente (se conclusa) e passa alla successiva."""
    payload = research_module.resolve_player(req.token)
    if payload.get("kind") != "battery":
        raise HTTPException(status_code=400, detail="Token non di tipo batteria.")
    battery, player_name = payload["battery"], payload["player_name"]
    battery_id = _battery_id(req.token)
    cur = _battery_current_index(battery_id, battery)
    if cur < len(battery):
        persist_id = _battery_persist_id(battery_id, cur)
        if not _game_is_finished(persist_id, battery[cur]["horizon_quarters"]):
            raise HTTPException(status_code=409, detail="Partita non ancora conclusa.")
        persistence.save_experiment_metadata(persist_id, game_done=True)
        # rimuove dalla memoria la sessione conclusa
        for s_id, s in list(SESSIONS.items()):
            if s.persist_id == persist_id:
                del SESSIONS[s_id]
    return _battery_response(req.token, battery, player_name)


@app.post("/api/player/join")
def player_join(req: research_module.PlayerJoinRequest) -> Dict[str, Any]:
    """Ingresso giocatore via token: crea o riprende una sessione PERSISTITA."""
    payload = research_module.resolve_player(req.token)
    if payload.get("kind") == "battery":
        return _battery_response(req.token, payload["battery"], payload["player_name"])
    scenario_id, player_name = payload["scenario_id"], payload["player_name"]
    condition = payload.get("condition", "Uomo")

    # Sessione web gia' aperta per questa partita? Riusala.
    for existing in SESSIONS.values():
        if existing.player_name == player_name and existing.scenario_id == scenario_id and existing.persist_id:
            return {**_state(existing), "player_name": player_name}

    persist_id = persistence.find_session(scenario_id, player_name)
    if persist_id:
        env = persistence.load_game_state(persist_id)
        if env is None:
            raise HTTPException(status_code=500, detail="Sessione salvata non caricabile")
        meta = persistence.load_experiment_metadata(persist_id)
        condition = meta.get("condition", condition)
    else:
        scenario = next(
            (s for s in persistence.load_scenarios(validate=False) if s.get("scenario_id") == scenario_id), None)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' non trovato")
        env = build_env_from_scenario(scenario, agents={player_name: {"type": HUMAN_AGENT_TYPE}})
        persist_id = research_module.new_research_session_id(scenario_id, player_name)
        persistence.clear_and_initialize_log_files(persist_id, component_names=env.spec.component_names)
        persistence.save_game_state(env, persist_id)
        persistence.save_experiment_metadata(persist_id, experiment_phase="playing",
                                             scenario_id=scenario_id, condition=condition,
                                             player_name=player_name)

    consent = persistence.load_experiment_metadata(persist_id).get("consent_given", False)
    sid = uuid.uuid4().hex[:12]
    session = Session(
        sid, env,
        agents={player_name: None},
        agent_types={player_name: HUMAN_AGENT_TYPE},
        scenario_id=scenario_id,
        persist_id=persist_id,
        player_name=player_name,
        condition=condition,
    )
    session.record_snapshot()
    while len(SESSIONS) >= MAX_SESSIONS:
        del SESSIONS[next(iter(SESSIONS))]
    SESSIONS[sid] = session
    return {**_state(session), "player_name": player_name, "consent_given": bool(consent)}


class ConsentRequest(BaseModel):
    agreed: bool = True


@app.post("/api/sessions/{sid}/consent")
def record_consent(sid: str, req: ConsentRequest) -> Dict[str, Any]:
    """Registra il consenso informato del partecipante prima di giocare."""
    from datetime import datetime, timezone

    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    if not req.agreed:
        raise HTTPException(status_code=400, detail="Consenso non fornito.")
    if session.persist_id:
        persistence.save_experiment_metadata(
            session.persist_id, consent_given=True,
            consent_timestamp=datetime.now(timezone.utc).isoformat())
    return {"consent_given": True}


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    return _state(session)


@app.post("/api/sessions/{sid}/step")
def step_session(sid: str, req: StepRequest) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    env = session.env
    with session.lock:
        try:
            for iteration in range(req.steps):
                active = [n for n in session.agents if n in env.active_agents]
                if not active:
                    break
                plans: Dict[str, Any] = {}
                for name in active:
                    manual = req.plans.get(name) if iteration == 0 else None
                    if manual is not None:
                        # Piano manuale (umano o override di un agente AI).
                        plans[name] = {"maintenance": {str(k): v for k, v in manual.items() if v}}
                    elif session.agents[name] is not None:
                        obs = env.get_agent_observations(name)
                        plans[name] = session.agents[name].act(obs) or {}
                    else:
                        plans[name] = {}
                quarter_before = env.quarter
                action_costs, revenues = env.step(plans)
                session.record_snapshot()
                session.persist(action_costs, revenues)
                # Explainability: razionale degli agenti AI (spiegazione del
                # modello) per trimestre, sulle sessioni di ricerca.
                if session.persist_id:
                    for name in active:
                        adapter = session.agents.get(name)
                        rationale = getattr(adapter, "last_rationale", None) if adapter is not None else None
                        maint = plans.get(name, {}).get("maintenance", {})
                        if rationale is not None or maint:
                            persistence.append_decision_rationale(
                                session.persist_id, quarter_before, name, maint, _rationale_text(rationale))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Errore dell'agente durante l'avanzamento: {e}")
    return _state(session)


def _rationale_text(rationale: Any) -> Optional[str]:
    """Riduce il razionale eterogeneo degli adapter a testo leggibile."""
    if rationale is None:
        return None
    if isinstance(rationale, str):
        return rationale[:2000]
    if isinstance(rationale, tuple):
        # Agentico: (context, (data, transcript)) o simili; prende la spiegazione se c'è.
        for part in rationale:
            if isinstance(part, dict) and part.get("explanation"):
                return str(part["explanation"])[:2000]
        return str(rationale[-1])[:2000]
    if isinstance(rationale, dict):
        return str(rationale.get("explanation") or rationale)[:2000]
    return str(rationale)[:2000]


@app.get("/api/sessions/{sid}/seismic")
def seismic_state(sid: str) -> Dict[str, Any]:
    """Mappa di pericolosità (griglia PGA) + sismica per asset."""
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    env = session.env
    hazard = env._seismic_hazard()
    points = [(a.lat, a.lon) for a in env.bridges.values() if a.lat is not None]
    lat0, lon0, lat1, lon1 = hazard.bounds_for(points or [(hazard.center_lat, hazard.center_lon)])
    grid = hazard.field_grid(lat0, lon0, lat1, lon1, n=26)
    assets = [
        {"id": a.id, "name": a.name, "lat": a.lat, "lon": a.lon, **(_seismic_view(a) or {})}
        for a in sorted(env.bridges.values(), key=lambda b: b.id)
    ]
    return {
        "enabled": bool(env.params.earthquake_enabled),
        "hazard_center": {"lat": hazard.center_lat, "lon": hazard.center_lon},
        "peak_pga": hazard.peak_pga,
        "far_pga": hazard.far_pga,
        "decay_km": hazard.decay_km,
        "bounds": {"lat0": lat0, "lon0": lon0, "lat1": lat1, "lon1": lon1},
        "grid": grid,
        "assets": assets,
    }


@app.post("/api/sessions/{sid}/seismic")
def seismic_toggle(sid: str, req: SeismicToggleRequest) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    with session.lock:
        session.env.params.earthquake_enabled = bool(req.enabled)
        session.env.refresh_network_reliability()
    return _state(session)


@app.get("/api/sessions/{sid}/assets/{aid}")
def asset_detail(sid: str, aid: int) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    asset = session.env.bridges.get(aid)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset non trovato")
    return _asset_detail(session.env, asset)


@app.post("/api/sessions/{sid}/act")
def act_session(sid: str, req: ActRequest) -> Dict[str, Any]:
    """Applica un piano manuale (azioni scelte dall'utente) e avanza un trimestre."""
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    env = session.env
    # Orizzonte fisso: la partita non avanza oltre max_quarters (uniforme fra
    # condizioni; a fine orizzonte il giocatore compila il questionario).
    if env.quarter > _session_horizon(env):
        raise HTTPException(status_code=409, detail="Partita conclusa (orizzonte raggiunto).")
    with session.lock:
        # Instrada ogni azione al manager dell'asset; gli agenti senza azioni
        # restano fermi per questo trimestre (nessuna decisione AI).
        plans: Dict[str, Any] = {name: {"maintenance": {}} for name in session.agents if name in env.active_agents}
        for asset_id, action in req.plan.items():
            if not action:
                continue
            asset = env.bridges.get(int(asset_id))
            if asset is not None and asset.manager in plans:
                plans[asset.manager]["maintenance"][str(asset_id)] = action
        if plans:
            quarter_before = env.quarter
            # Cadenza: il piano si applica al primo trimestre, poi il motore
            # avanza fino a completare la cadenza (es. 4 = decisione annuale).
            cadence = max(1, int(getattr(env.params, "human_cadence_quarters", 1) or 1))
            horizon = _session_horizon(env)
            action_costs, revenues = env.step(plans)
            session.record_snapshot()
            session.persist(action_costs, revenues)
            empty = {name: {"maintenance": {}} for name in session.agents if name in env.active_agents}
            for _ in range(cadence - 1):
                if env.quarter > horizon or not any(n in env.active_agents for n in session.agents):
                    break
                ac, rv = env.step({n: dict(empty[n]) for n in empty if n in env.active_agents} or empty)
                session.record_snapshot()
                session.persist(ac, rv)
            # Explainability: motivazione della mossa umana per questo ciclo.
            if session.persist_id and (req.rationale or any(p["maintenance"] for p in plans.values())):
                for name, p in plans.items():
                    if p["maintenance"] or req.rationale:
                        persistence.append_decision_rationale(
                            session.persist_id, quarter_before, name, p["maintenance"], req.rationale)
    return _state(session)


MAX_COPILOT_MESSAGES = 60          # per agente: oltre, si scarta il piu' vecchio
MAX_COPILOT_MESSAGE_CHARS = 2000


@app.post("/api/sessions/{sid}/copilot")
def copilot_chat(sid: str, req: CopilotRequest) -> Dict[str, Any]:
    """Un turno di dialogo con il copilota (condizione Uomo+LLM).

    Il copilota vede lo stato tecnico degli asset dell'agente, discute con
    l'ingegnere e puo' proporre un piano — che l'umano applica, modifica o
    ignora dalla Console. La decisione resta all'umano.
    """
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    # Gate sperimentale: il copilota è disponibile SOLO nella condizione
    # Uomo+LLM. Le sessioni non-ricerca (condition None) restano libere.
    if session.condition is not None and session.condition != "Uomo+LLM":
        raise HTTPException(status_code=403,
                            detail="Copilota non disponibile in questa condizione sperimentale.")
    message = (req.message or "").strip()[:MAX_COPILOT_MESSAGE_CHARS]
    if not message:
        raise HTTPException(status_code=400, detail="Messaggio vuoto.")
    if req.provider not in LLM_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Provider non ammesso: {req.provider!r}. Validi: {list(LLM_PROVIDERS)}")
    if req.provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY non configurata sul server.")
    if req.provider == "Google" and not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY non configurata sul server.")

    # Default: consiglia il primo agente umano; senza umani, il primo agente.
    agent_name = req.agent_name
    if agent_name is None:
        humans = [n for n, t in session.agent_types.items() if t == HUMAN_AGENT_TYPE]
        agent_name = humans[0] if humans else next(iter(session.agents))
    if agent_name not in session.agents:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_name}' non presente nella sessione")

    copilot = Copilot(provider=req.provider, model=req.model)
    with session.lock:
        history = session.copilot_history.setdefault(agent_name, [])
        result = copilot.respond(session.env, agent_name, history, message)
        history.append({"role": "user", "content": message})
        history.append({"role": "copilot", "content": result["reply"]})
        if len(history) > MAX_COPILOT_MESSAGES:
            del history[: len(history) - MAX_COPILOT_MESSAGES]
        # Explainability: i dialoghi copilota sono il dato della condizione
        # Uomo+LLM — persistiti su disco per le sessioni di ricerca.
        if session.persist_id:
            persistence.save_copilot_log(session.persist_id, session.copilot_history)
    return {
        "agent_name": agent_name,
        "reply": result["reply"],
        "proposed_plan": result.get("proposed_plan"),
        "history": history,
    }


@app.get("/api/sessions/{sid}/copilot")
def copilot_history(sid: str, agent_name: Optional[str] = None) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    if agent_name is None:
        humans = [n for n, t in session.agent_types.items() if t == HUMAN_AGENT_TYPE]
        agent_name = humans[0] if humans else next(iter(session.agents))
    return {"agent_name": agent_name, "history": session.copilot_history.get(agent_name, [])}


# ---------------------------------------------------------------------------
# Studio umano: questionario post-partita + analisi statistica
# ---------------------------------------------------------------------------

class QuestionnaireRequest(BaseModel):
    responses: Dict[str, Any] = Field(default_factory=dict)
    agent_name: Optional[str] = None


@app.get("/api/questionnaire/schema")
def questionnaire_schema() -> Dict[str, Any]:
    from infralab.study.questionnaire import QUESTIONNAIRE_SCHEMA

    return {"sections": QUESTIONNAIRE_SCHEMA}


@app.post("/api/sessions/{sid}/questionnaire")
def submit_questionnaire(sid: str, req: QuestionnaireRequest) -> Dict[str, Any]:
    """Salva le risposte del questionario + i punteggi derivati (TLX, fiducia).

    Richiede una sessione di ricerca (persistita); i dati finiscono nella
    cartella della sessione insieme a stato/log."""
    from datetime import datetime, timezone

    from infralab.study.questionnaire import score_questionnaire

    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    if not session.persist_id:
        raise HTTPException(status_code=400, detail="Il questionario è disponibile solo nelle sessioni di ricerca.")
    scores = score_questionnaire(req.responses)
    payload = {
        "session_id": session.persist_id,
        "scenario_id": session.scenario_id,
        "player_name": session.player_name,
        "agent_name": req.agent_name or session.player_name,
        "responses": req.responses,
        "scores": scores,
    }
    persistence.save_questionnaire(session.persist_id, payload)
    persistence.save_experiment_metadata(session.persist_id, questionnaire_submitted=True)
    return {"saved": True, "scores": scores}


@app.get("/api/sessions/{sid}/questionnaire")
def get_questionnaire(sid: str) -> Dict[str, Any]:
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Sessione non trovata")
    data = persistence.load_questionnaire(session.persist_id) if session.persist_id else None
    return {"submitted": data is not None, "data": data}


class CompareRequest(BaseModel):
    # {condizione: [valori dell'esito]} — es. reward/β per LLM, Uomo, ...
    groups: Dict[str, List[float]]
    alpha: float = 0.05


@app.post("/api/analysis/compare")
def analysis_compare(req: CompareRequest) -> Dict[str, Any]:
    """Confronto statistico fra condizioni (CI bootstrap, effect size, Mann-Whitney)."""
    from infralab.study.stats import compare_conditions

    clean = {k: [float(x) for x in v] for k, v in req.groups.items() if v}
    if len(clean) < 1:
        raise HTTPException(status_code=400, detail="Servono almeno un gruppo con valori.")
    return compare_conditions(clean, alpha=req.alpha)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0


CAMPAIGNS: Dict[str, Dict[str, Any]] = {}


def _aggregate_runs(runs, grid_key):
    def group_key(r):
        return (r["agent_id"], r.get(grid_key) if grid_key else None)

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(group_key(r), []).append(r)

    aggregates = []
    for (agent_id, gval), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else 0)):
        aggregates.append({
            "agent": agent_id,
            "grid_value": gval,
            "runs": len(rows),
            "mean_reward": _mean([r["total_reward"] for r in rows]),
            "mean_health": _mean([r.get("final_health_index") for r in rows]),
            "mean_beta_margin": _mean([r.get("final_min_beta_margin") for r in rows]),
            "std_beta_margin": _std([r.get("final_min_beta_margin") for r in rows]),
            "mean_quarters": _mean([r["quarters_survived"] for r in rows]),
            "mean_emissions_tco2": _mean([r.get("final_emissions_tco2") for r in rows]),
            "failure_rate": round(sum(1 for r in rows if r.get("terminated")) / len(rows), 3),
        })
    return aggregates


_STAT_METRICS = [
    ("total_reward", "Reward", "total_reward"),
    ("final_min_beta_margin", "Margine β minimo", "final_min_beta_margin"),
    ("final_health_index", "Health index", "final_health_index"),
    ("final_emissions_tco2", "Emissioni (tCO₂)", "final_emissions_tco2"),
    ("quarters_survived", "Trimestri sopravvissuti", "quarters_survived"),
]


def _condition_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confronto statistico fra agenti (= condizioni) per ogni esito chiave.

    Raggruppa i run per agent_id; per ogni metrica calcola medie+CI bootstrap e
    i confronti a coppie (Mann-Whitney, effect size). Vuoto se < 2 condizioni."""
    from infralab.study.stats import compare_conditions

    agents = sorted({r["agent_id"] for r in runs})
    if len(agents) < 2:
        return {"metrics": [], "note": "Servono almeno 2 agenti/condizioni per il confronto statistico."}
    metrics_out = []
    for key, label, field in _STAT_METRICS:
        groups = {a: [r.get(field) for r in runs if r["agent_id"] == a and r.get(field) is not None]
                  for a in agents}
        groups = {a: v for a, v in groups.items() if v}
        if len(groups) < 2:
            continue
        metrics_out.append({"key": key, "label": label, **compare_conditions(groups)})
    return {"metrics": metrics_out}


def _run_campaign_job(job_id: str, spec, scenario, grid_key, grid_param):
    job = CAMPAIGNS[job_id]

    def progress(completed, total):
        job["done"] = completed
        job["total"] = total

    try:
        results = run_experiment(spec, workers=1, scenario=scenario, progress_callback=progress)
        job["result"] = {
            "scenario_id": scenario.get("scenario_id"),
            "grid_param": grid_param,
            "run_count": spec.run_count,
            "completed": len(results["runs"]),
            "failed": len(results["errors"]),
            "aggregates": _aggregate_runs(results["runs"], grid_key),
            "stats": _condition_stats(results["runs"]),
            "runs": results["runs"],
        }
        job["status"] = "done"
    except Exception:
        job["status"] = "error"
        job["error"] = traceback.format_exc()


def _normalize_campaign_agents(agents: List[Any]) -> List[Dict[str, Any]]:
    """Normalizza gli agenti di campagna (stringhe o spec) in agent-spec del lab.

    Valida i tipi, i provider/chiavi LLM e l'esistenza dei plugin Custom. Ogni
    voce diventa {id, type, params}, dove params confluisce nell'agent_config
    passato a agent_factory dal runner."""
    if not agents:
        agents = ["Euristico"]
    valid_types = set(SESSION_AGENT_TYPES)
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    for a in agents:
        if isinstance(a, str):
            spec = CampaignAgentSpec(id=a, type=a)
        elif isinstance(a, dict):
            spec = CampaignAgentSpec(**a)
        else:
            raise HTTPException(status_code=400, detail=f"Agente campagna non valido: {a!r}")
        if spec.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Tipo agente non ammesso in campagna: {spec.type!r}. Validi: {sorted(valid_types)}")
        if spec.id in seen_ids:
            raise HTTPException(status_code=400, detail=f"Id agente/condizione duplicato: {spec.id!r}")
        seen_ids.add(spec.id)
        params: Dict[str, Any] = {}
        if spec.type in LLM_AGENT_TYPES:
            provider = spec.provider or ("Locale" if spec.type == "LLM Agentico" else "Google")
            if provider not in LLM_PROVIDERS:
                raise HTTPException(status_code=400, detail=f"Provider LLM non ammesso: {provider!r}")
            if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
                raise HTTPException(status_code=400, detail=f"'{spec.id}': OPENAI_API_KEY non configurata.")
            if provider == "Google" and not os.environ.get("GOOGLE_API_KEY"):
                raise HTTPException(status_code=400, detail=f"'{spec.id}': GOOGLE_API_KEY non configurata.")
            params = {"provider": provider, "profile": spec.profile or "Bilanciato"}
            if spec.model:
                params["model"] = spec.model
            if spec.type == "LLM Agentico" and spec.max_rounds:
                params["max_rounds"] = int(spec.max_rounds)
        elif spec.type == "Custom":
            if not spec.plugin:
                raise HTTPException(status_code=400, detail=f"'{spec.id}': tipo Custom richiede 'plugin'.")
            from infralab.plugins.registry import PluginError, load_agent_decider
            try:
                load_agent_decider(spec.plugin)
            except PluginError as e:
                raise HTTPException(status_code=400, detail=f"'{spec.id}': plugin non valido — {e}")
            params = {"plugin": spec.plugin}
        out.append({"id": spec.id, "type": spec.type, "params": params})
    return out


@app.post("/api/campaigns")
def start_campaign(req: CampaignRequest) -> Dict[str, Any]:
    """Avvia una campagna in background (confronto agenti ± sweep). Restituisce job_id."""
    agent_specs = _normalize_campaign_agents(req.agents)
    grid: Dict[str, List[Any]] = {}
    grid_param = None
    if req.compare_frameworks:
        unknown = [f for f in req.compare_frameworks if f not in available_code_frameworks()]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Normative sconosciute: {unknown}")
        grid["params.code_framework"] = list(req.compare_frameworks)
        grid_param = "code_framework"
    elif req.grid_param and req.grid_values:
        if req.grid_param not in CAMPAIGN_SWEEP_PARAMS:
            raise HTTPException(status_code=400, detail=f"Parametro non ammesso: {req.grid_param}")
        grid[f"params.{req.grid_param}"] = list(req.grid_values)
        grid_param = req.grid_param

    raw_spec = {
        "name": "ui-campaign",
        "scenario": req.scenario_id,
        "horizon_quarters": req.horizon_quarters,
        "agents": agent_specs,
        "grid": grid,
        "replicates": req.replicates,
        "seed": req.seed,
    }
    try:
        spec = spec_from_dict(raw_spec)
        scenario = resolve_scenario(req.scenario_id)
    except SpecError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if spec.run_count > MAX_CAMPAIGN_RUNS:
        raise HTTPException(status_code=400, detail=f"Troppe run ({spec.run_count} > {MAX_CAMPAIGN_RUNS}).")

    grid_key = f"params.{grid_param}" if grid else None
    job_id = uuid.uuid4().hex[:12]
    # Evict FIFO dei job conclusi oltre il limite (i risultati includono tutte le run).
    finished = [k for k, v in CAMPAIGNS.items() if v.get("status") != "running"]
    while len(CAMPAIGNS) >= MAX_CAMPAIGN_JOBS and finished:
        del CAMPAIGNS[finished.pop(0)]
    CAMPAIGNS[job_id] = {"status": "running", "done": 0, "total": spec.run_count, "result": None, "error": None}
    t = threading.Thread(
        target=_run_campaign_job, args=(job_id, spec, scenario, grid_key, grid_param if grid else None), daemon=True
    )
    t.start()
    return {"job_id": job_id, "run_count": spec.run_count}


@app.get("/api/campaigns/{job_id}")
def campaign_status(job_id: str) -> Dict[str, Any]:
    job = CAMPAIGNS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Campagna non trovata")
    return {
        "status": job["status"],
        "done": job.get("done", 0),
        "total": job.get("total", 0),
        "result": job.get("result"),
        "error": job.get("error"),
    }



@app.delete("/api/sessions/{sid}")
def delete_session(sid: str) -> Dict[str, Any]:
    if sid in SESSIONS:
        del SESSIONS[sid]
    return {"deleted": sid}


# ---------------------------------------------------------------------------
# Frontend statico (deploy single-process): se esiste frontend/dist, servilo.
# In sviluppo si continua a usare `npm run dev` (proxy /api -> :8000).
# ---------------------------------------------------------------------------

_DIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_DIST_DIR):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
