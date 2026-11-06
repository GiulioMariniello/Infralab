import math
import copy
import pickle
import logging
import warnings
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from infralab.settings.config import SimParameters, CdaConfig, DEFAULT_COMPONENTS, TRAFFIC_LOAD_MODELS
from infralab.core.degradation import (
    condition_from_damage,
    damage_from_condition,
    gamma_increment,
)
from infralab.core.inspection import inspect_component
from infralab.core.hazards import sudden_damage as _sudden
from infralab.core.hazards import regional_events as _regional
from infralab.core.hazards import progressive_events as _progressive
from infralab.core.reliability import (
    assess_proof_load,
    beta_from_pf,
    combine_independent_probabilities,
    get_target_reliability,
    series_system_reliability,
    evaluate_limit_states,
)


TRAFFIC_FRAGILITY_BASE = {
    "NTC": {"eta": 1.68, "beta": 0.33, "pf_code": 5.7e-2},
    "GL_heavy": {"eta": 2.33, "beta": 0.33, "pf_code": 4.9e-3},
    "GL_medium": {"eta": 3.19, "beta": 0.36, "pf_code": 2.25e-4},
    "GL_light": {"eta": 7.17, "beta": 0.40, "pf_code": 8.0e-7},
}

# La lista dei modelli validi vive in config (validazione SimParameters):
# fallire subito se le due definizioni divergono.
assert set(TRAFFIC_FRAGILITY_BASE) == set(TRAFFIC_LOAD_MODELS), (
    "TRAFFIC_FRAGILITY_BASE ed elenco config.TRAFFIC_LOAD_MODELS disallineati"
)


# Taxonomy dei componenti: vive nel plugin dell'asset type (Fase B2).
# COMPONENT_PROFILES resta esportato per retrocompatibilita' (= profili bridge).
from infralab.assets.bridges.profiles import (  # noqa: E402
    BRIDGE_COMPONENT_PROFILES as COMPONENT_PROFILES,
    BRIDGE_STRUCTURAL_COMPONENTS,
)


def _apply_code_framework(params: SimParameters) -> SimParameters:
    """Applica la normativa selezionata risolvendo i target di affidabilità.

    Con code_framework='custom' (default) non cambia nulla: il comportamento
    storico è preservato. Con una normativa registrata, i target β vengono
    dettati dal codice per la classe di conseguenza corrente.
    """
    import dataclasses

    from infralab.codes import get_code_framework

    name = getattr(params, "code_framework", "custom") or "custom"
    try:
        framework = get_code_framework(name)
    except KeyError as e:
        raise ValueError(str(e)) from None
    overrides = framework.resolve(params.reliability_consequence_class)
    if not overrides:
        return params
    return dataclasses.replace(params, **overrides)


def _resolve_regime(params: SimParameters):
    """Regime normativo selezionato (default 'none' = passthrough)."""
    from infralab.regimes import get_regime
    name = getattr(params, "regulatory_regime", "none") or "none"
    try:
        return get_regime(name)
    except KeyError as e:
        raise ValueError(str(e)) from None


def _apply_regime_overrides(params: SimParameters, regime) -> SimParameters:
    """La legge fissa parametri (es. escalation L4) e, se dichiara un tier, la normativa dei target."""
    import dataclasses
    overrides = dict(regime.param_overrides)
    if regime.target_tier:
        overrides["code_framework"] = regime.target_tier
    if not overrides:
        return params
    try:
        return dataclasses.replace(params, **overrides)
    except TypeError as e:
        raise ValueError(f"Regime '{regime.id}': param_overrides non validi ({e})") from None


def _new_regime_stats() -> Dict[str, Any]:
    return {"forced_actions": 0, "forced_cost": 0.0, "forced_deferred": 0,
            "sanction_bridge_quarters": 0, "closure_bridge_quarters": 0, "fines_total": 0.0,
            "rejected_actions": 0, "breach_bridge_quarters": 0, "forced_unexecutable": 0,
            "overridden_owner_actions": 0, "forced_unfunded": 0, "forced_unfunded_cost": 0.0,
            "state_funded_cost": 0.0, "grant_total": 0.0}


def _copy_optional_exposure(src: Dict[str, Any], bridge) -> None:
    """Attributi di esposizione facoltativi (urti): copiati solo se presenti."""
    if src.get("navigable_waterway") is not None:
        bridge.navigable_waterway = bool(src["navigable_waterway"])
    if src.get("service_under") not in (None, ""):
        bridge.service_under = str(src["service_under"])
    # Versante instabile (eventi progressivi geotecnici): solo se lo scenario lo porta.
    if src.get("on_unstable_slope") is not None:
        bridge.on_unstable_slope = bool(src["on_unstable_slope"])


def _new_floor_stats() -> Dict[str, Any]:
    return {"interventions": 0, "spend": 0.0, "closures": 0, "closure_bridge_quarters": 0,
            "capacity_blocked": 0, "budget_blocked": 0}


def _calculate_cost(op: Dict, bridge_area: float) -> float:
    cost_fixed = op.get('cost', 0)
    cost_per_m2 = op.get('cost_m2', 0)
    if pd.notna(cost_fixed) and cost_fixed > 0:
        return cost_fixed
    elif pd.notna(cost_per_m2) and cost_per_m2 > 0:
        return cost_per_m2 * bridge_area
    return 0.0


def _calculate_emissions_tco2(op: Dict, bridge_area: float) -> float:
    """Emissioni dell'azione in tonnellate di CO2e (fattori kgCO2e in config)."""
    co2_fixed = op.get('co2', 0) or 0
    co2_per_m2 = op.get('co2_m2', 0) or 0
    kg = co2_fixed if co2_fixed > 0 else co2_per_m2 * bridge_area
    return kg / 1000.0


def _detached(value: Any) -> Any:
    """Copia profonda e indipendente di un valore di stato (dict/list/array annidati).

    Round-trip pickle al posto di copy.deepcopy: stessi valori (float e scalari
    numpy compresi), costo circa 5 volte minore.
    """
    return pickle.loads(pickle.dumps(value, pickle.HIGHEST_PROTOCOL))


# Alberi dei risultati di affidabilita' (per stato limite, profilo anno per anno):
# sono la parte voluminosa dello stato (~85% di un ponte). Il motore non li modifica
# mai sul posto: a ogni aggiornamento li sostituisce con alberi nuovi.
_DETAIL_FIELDS = ("reliability_details", "posterior_reliability_details")


def _details_copy(details: Any) -> Any:
    """Copia a due livelli di un albero di dettagli di affidabilita'.

    Copiati il dizionario e i suoi contenitori diretti (life_cycle_profile,
    limit_state_results...): un agente che riassegna o aggiunge una voce non tocca
    il motore. Le foglie piu' profonde (risultato del singolo stato limite, punto
    del singolo anno) restano condivise in sola lettura: copiarle raddoppierebbe
    il costo delle osservazioni.
    """
    if not isinstance(details, dict):
        return details
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
            for k, v in details.items()}


def _state_copy(attrs: Dict[str, Any], components: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Copia indipendente del __dict__ di un componente o di un ponte (con i suoi
    componenti): tutto in profondita' tranne i dettagli di affidabilita' (_details_copy).
    L'ordine delle chiavi resta quello dell'oggetto."""
    def light(d):
        return {k: v for k, v in d.items() if k not in _DETAIL_FIELDS}
    payload = {"attrs": light(attrs)}
    if components is not None:
        payload["components"] = {n: light(c) for n, c in components.items()}
    copied = _detached(payload)

    def merge(orig, dup):
        return {k: (dup[k] if k in dup else _details_copy(v)) for k, v in orig.items()}
    out = merge(attrs, copied["attrs"])
    if components is not None:
        out["components"] = {n: merge(c, copied["components"][n]) for n, c in components.items()}
    return out


class Component:
    def __init__(self, name: str, eta_iniziale: float, vita_utile: int, exponent: float,
                 rng: Optional[np.random.Generator] = None,
                 profile: Optional[Dict[str, float]] = None):
        self.name, self.vita_utile_anni = name, vita_utile
        if profile is None:
            # Fallback legacy: lookup nella taxonomy bridge.
            if name not in COMPONENT_PROFILES:
                warnings.warn(
                    f"Profilo componente sconosciuto '{name}': uso 'impalcato' come fallback. "
                    f"Profili noti: {sorted(COMPONENT_PROFILES)}.",
                    stacklevel=2,
                )
            profile = COMPONENT_PROFILES.get(name, COMPONENT_PROFILES["impalcato"])
        self.profile = copy.deepcopy(profile)
        self.condition_init, self.condition_final = 1.0, 0.0
        self.current_condition: float = 1.0
        self.observed_condition_min: Optional[float] = None
        self.observed_condition_max: Optional[float] = None
        self.effective_age_years = float(eta_iniziale)
        self.initial_age_years = float(eta_iniziale)
        # Parametri aleatori per-istanza: campionati dal generatore dell'ambiente
        # (riproducibilita' controllata dal seed di simulazione), non da stato globale.
        sampler = rng if rng is not None else np.random.default_rng()
        for key, value in self.draw_latent_parameters(self.profile, sampler).items():
            setattr(self, key, value)
        self.epistemic_uncertainty = 0.22
        self.last_process_multiplier = 1.0
        self.reliability_index: Optional[float] = None
        self.failure_probability: Optional[float] = None
        self.annual_reliability_index: Optional[float] = None
        self.annual_failure_probability: Optional[float] = None
        self.posterior_reliability_index: Optional[float] = None
        self.posterior_failure_probability: Optional[float] = None
        self.posterior_annual_reliability_index: Optional[float] = None
        self.posterior_annual_failure_probability: Optional[float] = None
        self.reliability_details: Optional[Dict[str, Any]] = None
        self.posterior_reliability_details: Optional[Dict[str, Any]] = None
        self.last_proof_load_quarter: int = 0
        self.last_proof_load_alpha: Optional[float] = None
        self.last_proof_load_risk: Optional[float] = None
        self.last_proof_load_passed: Optional[bool] = None
        self.update_condition(exponent)

    # Parametri latenti del componente (mai osservati dal gestore): distribuzione a
    # priori del profilo. Stesso ordine di estrazione del costruttore storico.
    LATENT_PARAMETERS = ("degradation_scale", "degradation_shape_factor", "traffic_sensitivity",
                         "environment_sensitivity", "resistance_bias", "demand_bias", "crack_bias")

    @staticmethod
    def draw_latent_parameters(profile: Dict[str, float], sampler: np.random.Generator) -> Dict[str, float]:
        """Un'estrazione dei parametri latenti dalla loro distribuzione a priori."""
        return {
            "degradation_scale": max(0.30, float(sampler.normal(profile["scale_mean"], 0.10))),
            "degradation_shape_factor": max(0.75, float(sampler.normal(profile["shape_factor"], 0.08))),
            "traffic_sensitivity": max(0.25, float(sampler.normal(profile["traffic_sensitivity"], 0.08))),
            "environment_sensitivity": max(0.25, float(sampler.normal(profile["environment_sensitivity"], 0.08))),
            "resistance_bias": max(0.60, float(sampler.normal(profile["resistance_bias"], 0.05))),
            "demand_bias": max(0.50, float(sampler.normal(profile["demand_bias"], 0.05))),
            "crack_bias": max(0.50, float(sampler.normal(profile["crack_bias"], 0.05))),
        }

    def _shape(self, exponent: float) -> float:
        return max(0.8, exponent * self.degradation_shape_factor)

    def _scale(self) -> float:
        return max(0.05, self.degradation_scale)

    def update_condition(self, exponent: float):
        if self.effective_age_years <= 0 or self.vita_utile_anni <= 0:
            self.current_condition = 1.0
            self.accumulated_damage = 0.0
            return

        try:
            age_ratio = self.effective_age_years / self.vita_utile_anni
            degradation = self._scale() * (max(age_ratio, 0.0) ** self._shape(exponent))
            self.current_condition = max(0.0, min(1.0, math.exp(-degradation)))
        except (ZeroDivisionError, OverflowError):
            self.current_condition = 0.0
        self.accumulated_damage = damage_from_condition(self.current_condition)

    def apply_maintenance(self, gain: float, exponent: float):
        current_age_before_maint = self.effective_age_years
        if gain >= 1.0:
            self.current_condition = self.condition_init
            self.effective_age_years = 0.0
        else:
            self.current_condition = min(self.condition_init, self.current_condition + gain)
            if self.current_condition < 1.0 and exponent > 0 and self.vita_utile_anni > 0 and self._scale() > 0:
                degradation = -math.log(max(self.current_condition, 1e-6))
                age_ratio = max(degradation / self._scale(), 0.0) ** (1.0 / self._shape(exponent))
                self.effective_age_years = self.vita_utile_anni * age_ratio
            else:
                self.effective_age_years = 0.0

        self.accumulated_damage = damage_from_condition(self.current_condition)
        self.observed_condition_min = self.current_condition
        self.observed_condition_max = self.current_condition
        self.epistemic_uncertainty = max(0.06, self.epistemic_uncertainty * 0.90)
        return current_age_before_maint - self.effective_age_years

    def apply_damage(self, loss: float, exponent: float, raise_uncertainty: bool = True):
        self.current_condition = max(0.0, self.current_condition - max(0.0, loss))
        if self.current_condition <= 0:
            self.effective_age_years = float(self.vita_utile_anni)
        else:
            degradation = -math.log(max(self.current_condition, 1e-6))
            age_ratio = max(degradation / self._scale(), 0.0) ** (1.0 / self._shape(exponent))
            self.effective_age_years = self.vita_utile_anni * age_ratio
        self.accumulated_damage = damage_from_condition(self.current_condition)
        # raise_uncertainty=False (strict_information): il danno non osservato non
        # cambia la conoscenza del gestore; la cambia l'evidenza (_inspect_bridge).
        if raise_uncertainty:
            self.epistemic_uncertainty = min(0.40, self.epistemic_uncertainty * 1.10)

    def to_dict(self):
        # Copia, non il __dict__ vivo: chi riceve il dizionario (agente, copia del
        # pianificatore) non deve poter modificare il componente del motore.
        return _state_copy(self.__dict__)

    @classmethod
    def from_dict(cls, data, exponent: float):
        # I parametri campionati in __init__ vengono sovrascritti dai dati serializzati:
        # il generatore usato qui e' irrilevante per il risultato.
        comp = cls(data.get('name', ''), data.get('initial_age_years', 0.0), data.get('vita_utile_anni', 100), exponent,
                   rng=np.random.default_rng(0), profile=data.get('profile'))
        for key, value in data.items():
            setattr(comp, key, value)
        comp.current_condition = data.get('current_condition', comp.current_condition)
        comp.observed_condition_min = data.get('observed_condition_min')
        comp.observed_condition_max = data.get('observed_condition_max')
        comp.effective_age_years = data.get('effective_age_years', data.get('initial_age_years', 0.0))
        if not hasattr(comp, "profile") or comp.profile is None:
            comp.profile = copy.deepcopy(COMPONENT_PROFILES.get(comp.name, COMPONENT_PROFILES["impalcato"]))
        if 'accumulated_damage' not in data:
            # Sessioni salvate prima del modello gamma: ricava dal valore di condizione.
            comp.accumulated_damage = damage_from_condition(comp.current_condition)
        return comp


class Bridge:
    DEFAULT_STRUCTURAL_COMPONENTS = tuple(BRIDGE_STRUCTURAL_COMPONENTS)

    def __init__(self, id: int, name: str, length: float, width: float, adt: int, manager: str,
                 component_data: Dict[str, Dict], quarterly_revenue: float, initial_asset_value: float,
                 exponent: float, rng: Optional[np.random.Generator] = None,
                 component_profiles: Optional[Dict[str, Dict[str, float]]] = None,
                 structural_components: Optional[Tuple[str, ...]] = None):
        self.id, self.name, self.length, self.width, self.adt, self.manager = id, name, length, width, adt, manager
        self.quarterly_revenue = quarterly_revenue
        self.initial_asset_value = initial_asset_value
        profiles = component_profiles or {}
        self.components = {name: Component(name, data['eta_iniziale_anni'], data['vita_utile_anni'], exponent,
                                           rng=rng, profile=profiles.get(name))
                           for name, data in component_data.items()}
        self.structural_component_names = tuple(structural_components) if structural_components else self.DEFAULT_STRUCTURAL_COMPONENTS
        self.next_inspection_quarter, self.cda, self.last_inspection_quarter = 4, "Sconosciuta", 0
        # Linee Guida Ponti: censimento (L0), tracciamento valutazione L4,
        # restrizione d'uso (misure operative) e manutenzione preventiva.
        self.censused = True                       # default True: scenari legaci già "gestibili"
        self.last_L4_quarter = 0
        self.quarters_in_cda_alta_without_L4 = 0
        self.traffic_restriction_factor = 1.0      # 1.0 = nessuna limitazione d'uso
        # Stato del regime normativo (vuoto senza regime): trimestre di prima
        # attivazione di ciascun trigger e sanzione operativa attiva.
        self.regime_trigger_since: Dict[str, int] = {}
        # ultimo trimestre in cui ciascuna azione e' stata eseguita sull'asset
        # (serve agli obblighi che si assolvono eseguendo un'azione periodica)
        self.regime_action_done: Dict[str, int] = {}
        self.posting_until_quarter: int = 0        # limitazione di portata in corso
        self.hydraulic_damage_quarter: int = 0
        self.regime_sanction: Optional[Dict[str, Any]] = None
        self.maintenance_rate_multiplier = 1.0     # <1 = degrado rallentato (manut. preventiva)
        self.rate_multiplier_until_quarter = 0
        self.monitoring_ends_quarter = 0
        self.future_curve_range: Optional[Dict[str, list]] = None
        self.active_maintenance: Optional[Dict] = None
        self.last_action_name: Optional[str] = None
        self.last_action_quarter: int = 0
        self.reliability_index: Optional[float] = None
        self.failure_probability: Optional[float] = None
        self.annual_reliability_index: Optional[float] = None
        self.annual_failure_probability: Optional[float] = None
        self.posterior_reliability_index: Optional[float] = None
        self.posterior_failure_probability: Optional[float] = None
        self.posterior_annual_reliability_index: Optional[float] = None
        self.posterior_annual_failure_probability: Optional[float] = None
        self.reliability_details: Optional[Dict[str, Any]] = None
        self.posterior_reliability_details: Optional[Dict[str, Any]] = None
        self.last_proof_load_report: Optional[Dict[str, Any]] = None
        self.traffic_fragility: Optional[Dict[str, Any]] = None
        self.seismic_fragility: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.last_management_feedback: Optional[Dict[str, Any]] = None
        self.network_centrality: float = 0.0
        self.network_importance_factor: float = 1.0
        self.risk_priority_index: float = 0.0
        # Coordinate geografiche (GIS)
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None

    @property
    def area(self) -> float:
        return self.length * self.width

    @property
    def min_condition(self) -> float:
        return min((c.current_condition for c in self.components.values()), default=1.0)

    @property
    def structural_min_condition(self) -> float:
        """Condizione minima dei soli componenti portanti (dall'asset type)."""
        structural = set(getattr(self, 'structural_component_names', self.DEFAULT_STRUCTURAL_COMPONENTS))
        vals = [c.current_condition for n, c in self.components.items() if n in structural]
        return min(vals) if vals else self.min_condition

    @property
    def current_asset_value(self) -> float:
        return self.initial_asset_value * self.min_condition

    @property
    def status(self) -> str:
        if self.active_maintenance:
            return f"In Manutenzione (fino a T{self.active_maintenance['ends_at_quarter']})"
        return "Operativo"

    def min_observed_condition(self) -> Optional[float]:
        known_conditions = [c.observed_condition_min for c in self.components.values() if
                            c.observed_condition_min is not None]
        return min(known_conditions) if known_conditions else None

    def update_cda_and_due_date(self, current_quarter: int, cda_config: CdaConfig):
        c = self.min_observed_condition()
        if c is None:
            self.cda = "Sconosciuta"
            return
        if c > cda_config.bassa_threshold:
            self.cda, self.next_inspection_quarter = "Bassa", current_quarter + cda_config.bassa_due_quarters
        elif c > cda_config.medio_bassa_threshold:
            self.cda, self.next_inspection_quarter = "Medio-Bassa", current_quarter + cda_config.medio_bassa_due_quarters
        elif c > cda_config.media_threshold:
            self.cda, self.next_inspection_quarter = "Media", current_quarter + cda_config.media_due_quarters
        elif c > cda_config.medio_alta_threshold:
            self.cda, self.next_inspection_quarter = "Medio-Alta", current_quarter + cda_config.medio_alta_due_quarters
        else:
            self.cda, self.next_inspection_quarter = "Alta", current_quarter + cda_config.alta_due_quarters

    def to_dict(self, with_components: bool = True, with_details: bool = True) -> Dict[str, Any]:
        """Stato del ponte come dizionario INDIPENDENTE dal motore.

        Contenitori copiati (storico azioni, stato del regime, componenti, dettagli
        di affidabilita' fino al secondo livello, vedi _details_copy): chi riceve il
        dizionario (agente, copia del pianificatore) non modifica il ponte.
        `with_components=False` / `with_details=False` servono solo a
        get_agent_observations quando la vista (stimata/oracolo) sostituisce comunque
        quei campi: si evita di copiarli due volte (le chiavi restano, i componenti
        vuoti e i dettagli a None fino alla sostituzione).
        """
        attrs = {k: v for k, v in self.__dict__.items() if k != 'components'}
        if not with_details:
            for k in _DETAIL_FIELDS:
                if k in attrs:
                    attrs[k] = None
        state = _state_copy(attrs, {name: comp.__dict__ for name, comp in self.components.items()}
                            if with_components else {})
        min_obs_cond = self.min_observed_condition()
        components = state.pop("components")
        return {**state, "area": self.area, "status": self.status, "min_condition": self.min_condition,
                "current_asset_value": self.current_asset_value, "min_observed_condition": min_obs_cond,
                "lat": self.lat, "lon": self.lon,
                "components": components}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], exponent: float) -> 'Bridge':
        init_keys = ['id', 'name', 'length', 'width', 'adt', 'manager', 'quarterly_revenue', 'initial_asset_value']
        init_data = {k: data.get(k, 0) for k in init_keys}
        bridge = cls(**init_data, component_data={}, exponent=exponent)
        bridge.components = {name: Component.from_dict(comp_data, exponent) for name, comp_data in
                             data.get('components', {}).items()}
        bridge.next_inspection_quarter = data.get('next_inspection_quarter', 4)
        bridge.cda = data.get('cda', "Sconosciuta")
        bridge.last_inspection_quarter = data.get('last_inspection_quarter', 0)
        bridge.censused = data.get('censused', True)
        bridge.last_L4_quarter = data.get('last_L4_quarter', 0)
        bridge.quarters_in_cda_alta_without_L4 = data.get('quarters_in_cda_alta_without_L4', 0)
        bridge.traffic_restriction_factor = data.get('traffic_restriction_factor', 1.0)
        bridge.regime_trigger_since = {str(k): int(v) for k, v in (data.get('regime_trigger_since') or {}).items()}
        bridge.regime_action_done = {str(k): int(v) for k, v in (data.get('regime_action_done') or {}).items()}
        bridge.regime_sanction = copy.deepcopy(data.get('regime_sanction'))
        # Declassamento di portata in corso e ultimo danno idraulico: senza, nelle copie
        # dei pianificatori e nelle sessioni ricaricate il ponte declassato non torna
        # mai al traffico pieno (_sample_posting_events non vede la scadenza).
        bridge.posting_until_quarter = int(data.get('posting_until_quarter', 0) or 0)
        bridge.hydraulic_damage_quarter = int(data.get('hydraulic_damage_quarter', 0) or 0)
        # Chiusura per soglia etica: l'attributo esiste solo se la soglia l'ha creato
        # (con i default il dizionario del ponte resta quello storico).
        if data.get('floor_closure') is not None:
            bridge.floor_closure = copy.deepcopy(data.get('floor_closure'))
        if data.get('temporary_closure') is not None:
            bridge.temporary_closure = copy.deepcopy(data.get('temporary_closure'))
        bridge.maintenance_rate_multiplier = data.get('maintenance_rate_multiplier', 1.0)
        bridge.rate_multiplier_until_quarter = data.get('rate_multiplier_until_quarter', 0)
        bridge.monitoring_ends_quarter = data.get('monitoring_ends_quarter', 0)
        bridge.future_curve_range = data.get('future_curve_range')
        bridge.active_maintenance = data.get('active_maintenance')
        bridge.last_action_name = data.get('last_action_name')
        bridge.last_action_quarter = data.get('last_action_quarter', 0)
        bridge.reliability_index = data.get('reliability_index')
        bridge.failure_probability = data.get('failure_probability')
        bridge.annual_reliability_index = data.get('annual_reliability_index')
        bridge.annual_failure_probability = data.get('annual_failure_probability')
        bridge.posterior_reliability_index = data.get('posterior_reliability_index')
        bridge.posterior_failure_probability = data.get('posterior_failure_probability')
        bridge.posterior_annual_reliability_index = data.get('posterior_annual_reliability_index')
        bridge.posterior_annual_failure_probability = data.get('posterior_annual_failure_probability')
        bridge.reliability_details = data.get('reliability_details')
        bridge.posterior_reliability_details = data.get('posterior_reliability_details')
        bridge.last_proof_load_report = data.get('last_proof_load_report')
        bridge.traffic_fragility = data.get('traffic_fragility')
        bridge.seismic_fragility = data.get('seismic_fragility')
        bridge.action_history = data.get('action_history', [])
        bridge.last_management_feedback = data.get('last_management_feedback')
        # Attributi di esposizione letti dal NBI (item 019, 109, 071, 113). Con i
        # valori di default lo scenario si comporta come prima.
        bridge.detour_km = float(data.get('detour_km', 0.0) or 0.0)
        bridge.truck_fraction = float(data.get('truck_fraction', 0.0) or 0.0)
        bridge.crosses_water = bool(data.get('crosses_water', False))
        bridge.scour_critical = bool(data.get('scour_critical', False))
        _copy_optional_exposure(data, bridge)
        bridge.network_relevance = float(data.get('network_relevance', 0.0) or 0.0)
        bridge.network_centrality = data.get('network_centrality', 0.0)
        bridge.network_importance_factor = data.get('network_importance_factor', 1.0)
        bridge.risk_priority_index = data.get('risk_priority_index', 0.0)
        bridge.lat = data.get('lat')
        bridge.lon = data.get('lon')
        bridge.seismic_sensitivity = data.get('seismic_sensitivity')
        bridge.structural_component_names = tuple(
            data.get('structural_component_names', cls.DEFAULT_STRUCTURAL_COMPONENTS)
        )
        return bridge


class BridgeEnv:
    def __init__(self, params: SimParameters, maint_ops: Dict, agent_config: Dict, cda_config: CdaConfig,
                 component_weights: Dict, asset_type: str = "bridge"):
        from infralab.assets import get_asset_type

        self.regime = _resolve_regime(params)
        params = _apply_regime_overrides(params, self.regime)
        params = _apply_code_framework(params)
        self.params = params
        self.regime_stats: Dict[str, Dict[str, Any]] = {a: _new_regime_stats() for a in params.agent_names}
        self._forced_this_step: Dict[str, Dict[str, Any]] = {}
        self.maint_ops = maint_ops
        self.agent_config = agent_config
        self.cda_config = cda_config
        if getattr(self.regime, "cda_config", None):
            import dataclasses as _dc
            try:
                self.cda_config = _dc.replace(cda_config, **self.regime.cda_config)
            except TypeError as e:
                raise ValueError(f"Regime '{self.regime.id}': cda_config non valida ({e})") from None
        self.asset_type = asset_type
        self.spec = get_asset_type(asset_type)
        self.component_weights = component_weights or dict(self.spec.default_component_weights)
        # Generatore per-istanza: seed intero (0 compreso) => run deterministico;
        # seed None => entropia di sistema. Nessuno stato RNG globale condiviso.
        self.rng = np.random.default_rng(self.params.seed)
        # Base per i seed dell'MC di affidabilita' (stimatore numerico: sempre
        # deterministico, anche quando la simulazione gira con seed=None).
        self._reliability_seed_base = self.params.seed if self.params.seed is not None else 0
        # Cache delle proiezioni life-cycle per (ponte, componente, stato):
        # refresh_network_reliability viene chiamata piu' volte per turno.
        self._life_cycle_cache: Dict[tuple, Dict[str, Any]] = {}
        # Stato stimato (observation_mode="inspected"): cache per chiave di conoscenza.
        self._assessed_cache: Dict[tuple, Any] = {}
        self._assessment_obs_quarter: Dict[tuple, Any] = {}   # solo con assessment_state_mode="projected"
        # Catalogo completo per le azioni eseguite dall'autorita' (anche se lo scenario
        # restringe il catalogo del gestore).
        self._authority_catalogue = {**self._full_catalogue(), **copy.deepcopy(getattr(self.spec, "maintenance_ops", {}) or {})}
        self._validate_regime_actions()
        # Collassi realizzati (failure_model="reliability").
        self.failure_stats: Dict[str, Dict[str, float]] = {a: {"failures": 0, "realised_loss": 0.0, "realised_loss_c0": 0.0}
                                                            for a in params.agent_names}
        # Soglia etica (ethical_floor_enabled): contabilita' separata dalle sanzioni del regime.
        self.floor_stats: Dict[str, Dict[str, Any]] = {a: _new_floor_stats() for a in params.agent_names}
        # Danno locale improvviso (hazards/sudden_damage.py): danni non ancora noti al
        # gestore per ponte e contatori per gestore. Vuoti se il canale e' spento.
        self.sudden_pending: Dict[int, List[Dict[str, Any]]] = {}
        self.sudden_stats: Dict[str, Dict[str, Any]] = (
            {a: _sudden.new_stats(params) for a in params.agent_names} if _sudden.channel_active(params) else {})
        # Eventi estremi regionali (hazards/regional_events.py): elenco PUBBLICO degli
        # eventi avvenuti e contatori VERI per gestore. Vuoti se il canale e' spento; i
        # danni per ponte non ancora noti stanno in sudden_pending (tipo "regional").
        self.regional_events: List[Dict[str, Any]] = []
        self.regional_stats: Dict[str, Dict[str, Any]] = (
            {a: _regional.new_stats() for a in params.agent_names} if _regional.enabled(params) else {})
        # Eventi progressivi (hazards/progressive_events.py): difetti in sviluppo per ponte
        # (stato VERO: noti e non noti, con inizio e fine) e contatori veri per gestore.
        # Vuoti se il canale e' spento. Con il canale acceso il catalogo riceve l'azione
        # dedicata di arresto.
        self.progressive_defects: Dict[int, List[Dict[str, Any]]] = {}
        self.progressive_stats: Dict[str, Dict[str, Any]] = (
            {a: _progressive.new_stats(params) for a in params.agent_names} if _progressive.enabled(params) else {})
        if _progressive.enabled(params) and _progressive.ARREST_ACTION not in self.maint_ops:
            self.maint_ops = {**self.maint_ops, _progressive.ARREST_ACTION: _progressive.arrest_op(self)}
        # Registro dei collassi (uno per collasso, qualunque causa): causa, stato noto al
        # gestore all'inizio del trimestre, danni nascosti, limitazioni. E' VERITA' del
        # harness: non entra nelle copie dei pianificatori in modalita' "inspected".
        self.collapse_log: List[Dict[str, Any]] = []
        self._collapse_game_over: Dict[str, Dict[str, Any]] = {}
        self._owner_before: Dict[int, Dict[str, Any]] = {}
        self.last_expected_loss: Dict[str, Dict[str, float]] = {}
        # Sale del flusso dei collassi: 0 per l'ambiente reale. Le copie usate dai
        # pianificatori per le simulazioni interne ne ricevono uno diverso
        # (planning_clone), cosi' non conoscono i sorteggi futuri reali.
        self.failure_stream_salt: int = 0
        # Sale dei flussi a chiave (_keyed_stream): 0 per l'ambiente reale; le copie
        # "belief" dei pianificatori ne ricevono uno proprio.
        self.stream_salt: int = 0
        self.active_agents = list(agent_config.keys())
        self.quarter = 1
        self.bridges: Dict[int, Bridge] = {}
        self.budgets = {a: self.params.initial_budget for a in self.active_agents}
        # Sostenibilita': emissioni cumulate degli interventi (tCO2e) per agente.
        self.emissions_tco2 = {a: 0.0 for a in self.active_agents}
        self.negative_budget_streak = {a: 0 for a in self.active_agents}
        self.last_failure_info: Optional[Dict] = None
        # Azioni rinviate per mancanza di squadre, per agente.
        self.crew_stats: Dict[str, int] = {}
        self.last_plan_diagnostics: Dict[str, List[Dict[str, Any]]] = {a: [] for a in self.params.agent_names}
        # Storico eventi sismici
        self.earthquake_events: List[Dict] = []

    def initialize_bridges(self, bridges_data: List[Dict]):
        num_agents = len(self.params.agent_names)
        if num_agents == 0:
            raise ValueError("Impossibile inizializzare i ponti: la lista dei nomi degli agenti è vuota.")

        for i, b_data in enumerate(bridges_data):
            manager = b_data.get("manager", self.params.agent_names[i % num_agents])
            vita_utile_media = b_data.get('vita_utile_anni', 100)
            vita_utile_ponte = round(float(self.rng.normal(vita_utile_media, vita_utile_media * 0.10)))
            component_info = {}
            nominal_lives = {}
            for comp_name in self.component_weights.keys():
                # Vita utile esplicita per componente (scenari data-driven, es.
                # import NBI): nessuna perturbazione, cosi' l'inversione
                # condizione->eta' efficace resta consistente. Altrimenti:
                # perturbazione stocastica storica (realismo del sandbox).
                explicit_life = b_data.get(f'{comp_name}_vita_utile_anni')
                # Vita media dal tipo di asset: adapter (dipendente dalle
                # proprieta', es. classe d'uso NTC) o tabella di default.
                type_life = None
                if self.spec.lifetime_adapter is not None:
                    type_life = self.spec.lifetime_adapter(b_data, comp_name)
                if type_life is None:
                    type_life = self.spec.component_default_lifetimes.get(comp_name)
                if explicit_life is not None:
                    vita_utile_componente = round(float(explicit_life))
                    vita_nominale = vita_utile_componente
                elif type_life is not None:
                    vita_nominale = round(float(type_life))
                    variazione = float(self.rng.uniform(0.9, 1.1))
                    vita_utile_componente = round(float(type_life) * variazione)
                else:
                    vita_nominale = round(float(vita_utile_media))
                    variazione = float(self.rng.uniform(0.9, 1.1))
                    vita_utile_componente = round(vita_utile_ponte * variazione)
                nominal_lives[comp_name] = vita_nominale
                component_info[comp_name] = {'eta_iniziale_anni': b_data.get(f'{comp_name}_eta_iniziale_anni',
                                                                             b_data.get('eta_iniziale_anni', 0)),
                                             'vita_utile_anni': vita_utile_componente}
            bridge = Bridge(id=b_data['id'], name=b_data['name'], length=b_data['length'],
                            width=b_data['width'], adt=b_data['adt'], manager=manager,
                            component_data=component_info,
                            quarterly_revenue=b_data.get('incasso_trimestrale', 0),
                            initial_asset_value=b_data.get('valore_asset', 0),
                            exponent=self.params.deterioration_exponent,
                            rng=self.rng,
                            component_profiles=self.spec.component_profiles,
                            structural_components=tuple(sorted(self.spec.structural_components)))
            # Override per-asset della sensibilita' sismica (None = default del tipo).
            if self._strict_information():
                # Vita utile NOMINALE (inventario/tipo, senza la perturbazione
                # campionata): e' quella che il gestore conosce. Salvata solo con
                # strict_information, cosi' lo stato storico non cambia.
                for _cn, _life in nominal_lives.items():
                    if _cn in bridge.components:
                        bridge.components[_cn].nominal_service_life = float(_life)
            _sens = b_data.get('seismic_sensitivity')
            bridge.seismic_sensitivity = float(_sens) if _sens is not None else None
            # Attributi di esposizione dal NBI (item 019 deviazione, 109 mezzi pesanti,
            # 071 corso d'acqua, 113 erosione). Assenti negli scenari storici: default neutri.
            bridge.detour_km = float(b_data.get('detour_km', 0.0) or 0.0)
            bridge.truck_fraction = float(b_data.get('truck_fraction', 0.0) or 0.0)
            bridge.crosses_water = bool(b_data.get('crosses_water', False))
            bridge.scour_critical = bool(b_data.get('scour_critical', False))
            # Esposizione agli urti (item NBI 038 via navigabile, 042B servizio sotto):
            # attributi creati solo se lo scenario li porta (dizionario storico invariato).
            _copy_optional_exposure(b_data, bridge)
            bridge.network_relevance = 0.0
            # Assegna coordinate GIS: usa quelle nel dataset oppure genera casualmente
            if b_data.get('lat') is not None and b_data.get('lon') is not None:
                bridge.lat = float(b_data['lat'])
                bridge.lon = float(b_data['lon'])
            else:
                # Coordinate casuali nella regione configurata
                angle = float(self.rng.uniform(0, 2 * math.pi))
                dist = float(self.rng.uniform(0, self.params.bridge_region_radius_km))
                bridge.lat = round(
                    self.params.bridge_region_lat + (dist * math.cos(angle)) / 111.0, 4)
                bridge.lon = round(
                    self.params.bridge_region_lon + (dist * math.sin(angle)) / (
                        111.0 * math.cos(math.radians(self.params.bridge_region_lat))), 4)
            self.bridges[b_data['id']] = bridge
        _progressive.assign_exposure(self)
        self._update_network_importance()
        if getattr(self.params, "initial_inventory_known", False):
            self._initial_inventory_inspection()
        self.refresh_network_reliability()

    def _initial_inventory_inspection(self) -> None:
        """Il rating dell'inventario nazionale e' un'ispezione ordinaria: ogni ponte
        parte con una condizione osservata come dopo un'ispezione L1 al trimestre 0.

        Stesso percorso di `_inspect_bridge` (rumore, bias, POD, guadagno di
        conoscenza, classe e prossima scadenza), ma a costo zero, non registrata come
        azione del gestore e con un flusso casuale dedicato sempre a chiave: con
        rng_streams="shared" l'RNG storico non viene toccato, con "keyed" i flussi
        degli altri processi restano quelli di prima.
        """
        op = self.maint_ops.get("L1_inspect") or self._authority_catalogue.get("L1_inspect") or {"error_sigma": 0.10}
        saved_quarter = self.quarter
        self.quarter = 0
        try:
            for bid in sorted(self.bridges):
                bridge = self.bridges[bid]
                self._inspect_bridge(bridge, op, dedicated_purpose=4210)
                self._record_action_event(bridge, "inventario_iniziale", "inventory")
        finally:
            self.quarter = saved_quarter

    def _distance_km(self, a: Bridge, b: Bridge) -> float:
        if a.lat is None or a.lon is None or b.lat is None or b.lon is None:
            return 0.0
        lat1, lon1, lat2, lon2 = map(math.radians, [a.lat, a.lon, b.lat, b.lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(hav)))

    def _update_network_relevance(self):
        """Rilevanza di rete = percentile dell'esposizione da deviazione (ADT x km di
        deviazione), lo stesso costrutto del termine S3 della scheda FHWA e della
        componente di esposizione della classe di attenzione delle Linee Guida
        (traffico e presenza di alternativa stradale).

        Sostituisce il punteggio di prossimita', che sull'inventario di studio risulta
        scorrelato dall'esposizione reale (rho 0.07, p 0.43) e peggiore del solo
        traffico (rho 0.34). Attivo solo se lo scenario porta i km di deviazione.
        """
        bridges = list(self.bridges.values())
        if not bridges or not any(getattr(b, "detour_km", 0.0) > 0 for b in bridges):
            return False
        exposure = [(float(b.adt) * float(getattr(b, "detour_km", 0.0)), b) for b in bridges]
        order = sorted(exposure, key=lambda t: t[0])
        n = len(order)
        for rank, (_, bridge) in enumerate(order):
            bridge.network_relevance = (rank + 0.5) / n
            bridge.network_importance_factor = max(0.25, bridge.network_relevance)
        return True

    def _update_network_importance(self):
        if getattr(self.params, "network_relevance_mode", "proximity") == "nbi_detour" and self._update_network_relevance():
            return
        bridges = list(self.bridges.values())
        if not bridges:
            return
        max_adt = max((b.adt for b in bridges), default=1)
        centrality_scores = []
        for bridge in bridges:
            distances = sorted(
                [self._distance_km(bridge, other) for other in bridges if other.id != bridge.id and other.lat is not None and other.lon is not None]
            )
            nearest = distances[: min(3, len(distances))]
            if nearest:
                centrality = 1.0 / max(1.0, float(sum(nearest) / len(nearest)))
            else:
                centrality = 1.0
            centrality_scores.append(centrality)
        max_centrality = max(centrality_scores) if centrality_scores else 1.0
        adt_weight = min(1.0, max(0.0, self.params.network_importance_adt_weight))
        for bridge, centrality in zip(bridges, centrality_scores):
            adt_score = bridge.adt / max(1.0, max_adt)
            centrality_score = centrality / max(1e-9, max_centrality)
            bridge.network_centrality = centrality_score
            bridge.network_importance_factor = max(0.25, adt_weight * adt_score + (1.0 - adt_weight) * centrality_score)

    def _component_observation_sigma(self, component: Component, base_sigma: float) -> float:
        scale = float(getattr(self.params, "inspection_sigma_scale", 1.0))
        return max(0.005, scale * base_sigma * (0.65 + component.epistemic_uncertainty))

    def _years_elapsed(self) -> float:
        return (self.quarter - 1) / 4.0

    def _traffic_growth_multiplier(self) -> float:
        """Crescita non-stazionaria del traffico: (1+g)^anni trascorsi."""
        growth = self.params.traffic_growth_annual
        if growth == 0.0:
            return 1.0
        return (1.0 + growth) ** self._years_elapsed()

    def _climate_trend_multiplier(self) -> float:
        """Trend climatico sul tasso di invecchiamento: (1+c)^anni trascorsi."""
        trend = self.params.climate_degradation_trend_annual
        if trend == 0.0:
            return 1.0
        return (1.0 + trend) ** self._years_elapsed()

    def _traffic_multiplier(self, bridge: Bridge) -> float:
        reference = max(1, self.params.traffic_reference_adt)
        # Le misure operative (limitazione carico/chiusura) riducono il traffico
        # effettivo → meno domanda, degrado e fragilità da traffico più bassi.
        restriction = getattr(bridge, 'traffic_restriction_factor', 1.0)
        if getattr(self.params, "traffic_demand_mode", "network_relative") == "self_designed":
            return 1.0 + self.params.traffic_degradation_factor * (self._traffic_growth_multiplier() * restriction)
        effective_adt = bridge.adt * self._traffic_growth_multiplier() * restriction
        return 1.0 + self.params.traffic_degradation_factor * (effective_adt / reference)

    def _traffic_fragility_metrics(self, bridge: Bridge) -> Dict[str, float]:
        base = TRAFFIC_FRAGILITY_BASE.get(self.params.traffic_load_model, TRAFFIC_FRAGILITY_BASE["GL_heavy"]).copy()
        lane_penalty = 1.0
        if self.params.traffic_lane_count == 2:
            lane_penalty = 1.10
        elif self.params.traffic_lane_count == 1:
            lane_penalty = 1.35

        offset_bonus = 1.0 + 0.22 * max(0.0, self.params.traffic_first_lane_offset_m)
        remaining_area_bonus = 1.03 if not self.params.traffic_remaining_area_active else 1.0
        age_penalty = 1.0 - 0.18 * max(0.0, 1.0 - bridge.structural_min_condition)

        eta = max(1.05, base["eta"] * lane_penalty * offset_bonus * remaining_area_bonus * age_penalty)
        beta_disp = min(0.50, max(0.28, base["beta"] + 0.03 * max(0.0, 1.0 - bridge.min_condition)))
        alpha_code = 1.0
        z = math.log(max(alpha_code, 1e-6) / eta) / beta_disp
        code_failure = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        overload_rate = self.params.traffic_monthly_overload_probability
        if self.params.traffic_load_model == "NTC":
            annual_rate = code_failure * 0.001
        else:
            annual_rate = code_failure * overload_rate
            if self.params.traffic_load_model == "GL_heavy":
                annual_rate *= max(self.params.traffic_congestion_probability, 1e-4)

        return {
            "traffic_model": self.params.traffic_load_model,
            "eta": eta,
            "beta": beta_disp,
            "code_failure_probability": max(1e-9, min(1.0 - 1e-9, code_failure)),
            "annual_failure_rate": max(1e-9, annual_rate),
            "lane_count": self.params.traffic_lane_count,
            "first_lane_offset_m": self.params.traffic_first_lane_offset_m,
        }

    def _seismic_hazard(self):
        from infralab.core.hazards.seismic_hazard import hazard_from_params
        return hazard_from_params(self.params)

    def _seismic_fragility_metrics(self, bridge: Bridge) -> Dict[str, float]:
        from infralab.core.hazards.earthquake import get_code_conforming_seismic_fragility

        hazard = self._seismic_hazard()
        # PGA di sito dalla mappa di pericolosità (dipende dalla posizione).
        site_pga = hazard.site_pga(bridge.lat, bridge.lon)
        distance_km = hazard.distance_km(bridge.lat, bridge.lon)
        # Penalità di condizione: una struttura degradata è più vulnerabile.
        condition_factor = 1.0 + 0.20 * max(0.0, 1.0 - bridge.structural_min_condition)
        # Sensibilita' sismica: default calibrabile per tipo di asset
        # (hazard_sensitivities della spec), override per-asset dallo scenario.
        _asset_sens = getattr(bridge, "seismic_sensitivity", None)
        sensitivity = float(_asset_sens if _asset_sens is not None
                            else self.spec.hazard_sensitivities.get("seismic", 1.0))
        controlling_pga = site_pga * condition_factor * sensitivity
        fragility = get_code_conforming_seismic_fragility(
            pga=controlling_pga,
            bridge_class=self.params.seismic_bridge_class,
            limit_state=self.params.seismic_limit_state,
        )
        fragility["site_pga"] = site_pga
        fragility["distance_to_hazard_km"] = None if distance_km == float("inf") else distance_km
        fragility["controlling_pga"] = controlling_pga
        fragility["annual_exceedance_proxy"] = fragility["failure_probability"] * self.params.earthquake_annual_prob
        fragility["enabled"] = bool(self.params.earthquake_enabled)
        return fragility

    def _reliability_targets(self):
        return get_target_reliability(
            consequence_class=self.params.reliability_consequence_class,
            reference_period_years=self.params.reliability_reference_period_years,
            beta_reference_override=self.params.reliability_target_beta,
            beta_annual_override=self.params.reliability_target_beta_annual,
        )

    def _component_limit_states(self, bridge: Bridge, component: Component) -> Dict[str, Dict[str, float]]:
        # Delegato all'adapter dell'asset type registrato (plugin).
        return self.spec.limit_state_adapter(self, bridge, component)

    # -- Degrado: dispatch weibull/gamma/custom -----------------------------
    def _degradation_plugin(self):
        """Callable del modello di degrado utente se `degradation_model` è
        'custom:<nome>', altrimenti None (risolta una volta e cachata)."""
        model = self.params.degradation_model or ""
        if not model.startswith("custom:"):
            return None
        if getattr(self, "_degradation_plugin_fn", "unset") == "unset":
            try:
                from infralab.plugins.registry import load_degradation_model
                self._degradation_plugin_fn = load_degradation_model(model.split(":", 1)[1])
            except Exception as e:  # noqa: BLE001 - fallback al modello di default
                logger.warning("Modello di degrado custom non caricabile (%s): %s", model, e)
                self._degradation_plugin_fn = None
        return self._degradation_plugin_fn

    def _degradation_ctx(self, component: Component, age_before: float, rand) -> Dict[str, Any]:
        exponent = self.params.deterioration_exponent
        return {
            "condition": component.current_condition,
            "previous_age_years": age_before,
            "effective_age_years": component.effective_age_years,
            "age_increment_years": component.effective_age_years - age_before,
            "vita_utile_anni": component.vita_utile_anni,
            "shape": component._shape(exponent),
            "scale": component._scale(),
            "deterioration_exponent": exponent,
            "accumulated_damage": component.accumulated_damage,
            "component_name": component.name,
            "quarter": self.quarter,
            "rand": rand,
        }

    def _advance_component_condition(self, component: Component, age_before: float) -> None:
        """Aggiorna la condizione nello STEP reale. Preserva esattamente
        weibull/gamma; usa il plugin utente se `degradation_model` è custom."""
        plugin = self._degradation_plugin()
        if plugin is not None:
            _g = getattr(self, "_deg_rng", None) or self.rng
            ctx = self._degradation_ctx(component, age_before, lambda: float(_g.random()))
            try:
                new_cond = float(plugin(ctx))
            except Exception as e:  # noqa: BLE001 - fallback robusto al default
                logger.warning("Modello di degrado custom in errore su %s: %s", component.name, e)
                component.update_condition(self.params.deterioration_exponent)
                return
            component.current_condition = max(0.0, min(1.0, new_cond))
            component.accumulated_damage = damage_from_condition(component.current_condition)
            return
        if self.params.degradation_model == "gamma":
            increment = gamma_increment(
                component, self.params.deterioration_exponent, age_before,
                component.effective_age_years, self.params.gamma_process_theta, getattr(self, "_deg_rng", None) or self.rng,
            )
            component.accumulated_damage = max(0.0, component.accumulated_damage) + increment
            component.current_condition = condition_from_damage(component.accumulated_damage)
        else:
            component.update_condition(self.params.deterioration_exponent)

    def _advance_projection_condition(self, component: Component, age_before: float) -> None:
        """Aggiorna la condizione nelle PROIEZIONI (traiettoria media). Senza
        plugin resta il Weibull medio deterministico (invariato anche in modalità
        gamma); col plugin usa il modello utente con rand deterministico (0.5)."""
        plugin = self._degradation_plugin()
        if plugin is not None:
            ctx = self._degradation_ctx(component, age_before, lambda: 0.5)
            try:
                component.current_condition = max(0.0, min(1.0, float(plugin(ctx))))
                component.accumulated_damage = damage_from_condition(component.current_condition)
                return
            except Exception:  # noqa: BLE001
                pass
        component.update_condition(self.params.deterioration_exponent)

    def _component_reliability(self, bridge: Bridge, component: Component):
        targets = self._reliability_targets()
        limit_states = self._component_limit_states(bridge, component)
        component_seed = sum(ord(ch) for ch in component.name)
        prior = evaluate_limit_states(
            limit_states,
            targets=targets,
            sample_count=self.params.reliability_mc_samples,
            seed=self._reliability_seed_base + getattr(bridge, "_mc_seed_id", bridge.id) * 17 + component_seed,
            sls_beta_reference=self.params.reliability_sls_target_beta,
            brittle_beta_increment=self.params.reliability_brittle_failure_beta_increment,
        )

        posterior = prior
        if component.last_proof_load_passed and component.last_proof_load_alpha:
            proof_assessment = assess_proof_load(
                limit_states,
                targets=targets,
                alpha=component.last_proof_load_alpha,
                sample_count=self.params.reliability_mc_samples,
                seed=self._reliability_seed_base + getattr(bridge, "_mc_seed_id", bridge.id) * 31 + component_seed,
                step_count=self.params.proof_load_step_count,
                sls_beta_reference=self.params.reliability_sls_target_beta,
                brittle_beta_increment=self.params.reliability_brittle_failure_beta_increment,
            )
            posterior = proof_assessment["posterior_result"]
        return prior, posterior

    def _project_component_life_cycle(self, bridge: Bridge, component: Component) -> Dict[str, Any]:
        # La proiezione dipende solo dallo stato corrente del componente (usa il
        # fattore ambientale MEDIO, senza rumore di processo): cache per evitare
        # di rieseguire evaluate_limit_states (anni x componenti) ad ogni refresh
        # ripetuto nello stesso turno.
        rate_adapter = self.spec.degradation_rate_adapter
        rate_mult = (
            max(0.0, float(rate_adapter(self, bridge, component))) if rate_adapter is not None else None
        )
        cache_key = (
            bridge.id,
            component.name,
            round(component.effective_age_years, 9),
            round(component.current_condition, 12),
            round(component.epistemic_uncertainty, 12),
            # con crescita traffico non-stazionaria il multiplier varia nel tempo
            round(self._traffic_growth_multiplier(), 9),
            # con accoppiamento sul rate la proiezione dipende anche dallo
            # stato degli altri componenti (es. involucro)
            round(rate_mult, 9) if rate_mult is not None else None,
        )
        cached = self._life_cycle_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        if len(self._life_cycle_cache) > 20000:     # la proiezione dipende dallo stato: le vecchie non servono piu'
            self._life_cycle_cache.clear()

        years = max(1, int(self.params.reliability_projection_years))
        targets = self._reliability_targets()
        component_seed = sum(ord(ch) for ch in component.name)
        temp_component = Component.from_dict(component.to_dict(), self.params.deterioration_exponent)
        profile_samples = max(64, self.params.reliability_mc_samples // 2)
        profile = []
        for year in range(years + 1):
            limit_states = self._component_limit_states(bridge, temp_component)
            result = evaluate_limit_states(
                limit_states,
                targets=targets,
                sample_count=profile_samples,
                seed=self._reliability_seed_base + getattr(bridge, "_mc_seed_id", bridge.id) * 101 + component_seed + year,
                sls_beta_reference=self.params.reliability_sls_target_beta,
                brittle_beta_increment=self.params.reliability_brittle_failure_beta_increment,
            )
            profile.append(
                {
                    "year": year,
                    "structural_annual_beta": result.structural_annual_beta,
                    "structural_annual_pf": result.structural_annual_failure_probability,
                    "serviceability_annual_beta": result.serviceability_annual_beta,
                    "serviceability_annual_pf": result.serviceability_annual_failure_probability,
                    "condition": temp_component.current_condition,
                }
            )
            if year < years:
                env_factor = self.params.env_degradation_mean
                age_increase_factor = (
                    self._traffic_multiplier(bridge) ** temp_component.traffic_sensitivity
                ) * (
                    env_factor ** temp_component.environment_sensitivity
                )
                if rate_mult is not None:
                    age_increase_factor *= rate_mult
                _age_before = temp_component.effective_age_years
                temp_component.effective_age_years += 1.0 * age_increase_factor
                self._advance_projection_condition(temp_component, _age_before)
        min_point = min(profile, key=lambda item: item["structural_annual_beta"] if item["structural_annual_beta"] is not None else 999.0)
        # Componenti con soli stati limite SLS (es. drenaggio) non hanno pf strutturale.
        structural_pfs = [item["structural_annual_pf"] for item in profile if item["structural_annual_pf"] is not None]
        result_profile = {
            "projection_years": years,
            "profile": profile,
            "min_remaining_structural_annual_beta": min_point["structural_annual_beta"],
            "max_remaining_structural_annual_pf": max(structural_pfs) if structural_pfs else None,
            "year_of_min_structural_beta": min_point["year"],
        }
        if len(self._life_cycle_cache) > 4096:
            self._life_cycle_cache.clear()
        self._life_cycle_cache[cache_key] = copy.deepcopy(result_profile)
        return result_profile

    def _proof_load_relevant_components(self, bridge: Bridge) -> List[Component]:
        relevant = [
            component for component in bridge.components.values()
            if component.traffic_sensitivity >= 0.75
        ]
        return relevant or list(bridge.components.values())

    def _evaluate_bridge_proof_load_option(self, bridge: Bridge, alpha: float, step_count: int, targets,
                                           component_names: Optional[List[str]] = None):
        component_assessments = []
        posterior_results = []
        if component_names is None:
            evaluated_components = self._proof_load_relevant_components(bridge)
        else:
            evaluated_components = [bridge.components[n] for n in component_names if n in bridge.components]
        for component in evaluated_components:
            component_seed = sum(ord(ch) for ch in component.name)
            limit_states = self._component_limit_states(bridge, component)
            assessment = assess_proof_load(
                limit_states,
                targets=targets,
                alpha=alpha,
                sample_count=self.params.reliability_mc_samples,
                seed=self._reliability_seed_base + getattr(bridge, "_mc_seed_id", bridge.id) * 31 + component_seed,
                step_count=step_count,
                sls_beta_reference=self.params.reliability_sls_target_beta,
                brittle_beta_increment=self.params.reliability_brittle_failure_beta_increment,
            )
            component_assessments.append((component, assessment))
            posterior_results.append(assessment["posterior_result"])

        failure_probability = max(
            (assessment["failure_probability_during_test"] for _, assessment in component_assessments),
            default=1e-12,
        )
        screening_probability = max(
            (assessment["screening_probability"] for _, assessment in component_assessments),
            default=0.0,
        )
        pass_probability = max(1e-12, 1.0 - failure_probability - screening_probability)

        extra_annual_pf = 0.0
        if self.params.earthquake_enabled and bridge.seismic_fragility:
            extra_annual_pf = bridge.seismic_fragility.get("annual_exceedance_proxy", 0.0) or 0.0
        for component in bridge.components.values():
            if component in evaluated_components:
                continue
            _, posterior = self._component_reliability(bridge, component)
            posterior_results.append(posterior)
        bridge_posterior = series_system_reliability(posterior_results, extra_annual_failure_probability=extra_annual_pf)
        targets_satisfied = all(assessment["targets_satisfied"] for _, assessment in component_assessments)
        return {
            "alpha": alpha,
            "step_count": step_count,
            "component_assessments": component_assessments,
            "failure_probability_during_test": failure_probability,
            "screening_probability": screening_probability,
            "pass_probability": pass_probability,
            "bridge_posterior": bridge_posterior,
            "targets_satisfied": targets_satisfied,
        }

    def _select_proof_load_option(self, bridge: Bridge, op: Dict[str, Any], targets):
        max_failure_probability = op.get("max_failure_probability")
        if not max_failure_probability:
            alpha = op.get("alpha", self.params.proof_load_base_alpha)
            step_count = op.get("step_count", self.params.proof_load_step_count)
            option = self._evaluate_bridge_proof_load_option(bridge, alpha, step_count, targets)
            option["risk_constraint_satisfied"] = True
            option["used_fallback_option"] = False
            option["feasible_option_count"] = 1
            option["max_failure_probability"] = None
            return option

        alpha_min = float(op.get("alpha_min", 0.45))
        alpha_max = float(op.get("alpha_max", op.get("alpha", self.params.proof_load_base_alpha)))
        alpha_grid_size = int(op.get("alpha_grid_size", 12))
        step_count = int(op.get("step_count", self.params.proof_load_step_count))
        candidate_alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
        feasible_options = []
        fallback_options = []
        for alpha in candidate_alphas:
            option = self._evaluate_bridge_proof_load_option(bridge, float(alpha), step_count, targets)
            fallback_options.append(option)
            if option["failure_probability_during_test"] <= max_failure_probability:
                feasible_options.append(option)

        if feasible_options:
            feasible_options.sort(
                key=lambda item: (
                    item["targets_satisfied"],
                    item["bridge_posterior"].annual_beta,
                    item["alpha"],
                ),
                reverse=True,
            )
            selected = feasible_options[0]
            selected["risk_constraint_satisfied"] = True
            selected["used_fallback_option"] = False
            selected["feasible_option_count"] = len(feasible_options)
            selected["max_failure_probability"] = max_failure_probability
            return selected

        fallback_options.sort(key=lambda item: item["failure_probability_during_test"])
        selected = fallback_options[0]
        selected["risk_constraint_satisfied"] = False
        selected["used_fallback_option"] = True
        selected["feasible_option_count"] = 0
        selected["max_failure_probability"] = max_failure_probability
        return selected

    # ------------------------------------------------------------------
    # Informazione stretta (strict_information)
    # ------------------------------------------------------------------
    def _strict_information(self) -> bool:
        return bool(getattr(self.params, "strict_information", False))

    def _owner_knows_truth(self) -> bool:
        return self.params.observation_mode == "full" or bool(getattr(self.params, "oracle_observation", False))

    def _strict_proof_load(self) -> bool:
        return self._strict_information() and not self._owner_knows_truth()

    @staticmethod
    def _nominal_life(comp: "Component") -> float:
        """Vita utile che il gestore conosce: nominale se salvata (strict_information),
        altrimenti quella del componente (storico)."""
        life = getattr(comp, "nominal_service_life", None)
        return float(life) if life is not None else float(comp.vita_utile_anni)

    def _apply_hidden_damage(self, comp: "Component", loss: float) -> None:
        """Danno fisico non osservato (idraulico, sismico): con strict_information la
        conoscenza del gestore (incertezza epistemica) non cambia."""
        comp.apply_damage(loss, self.params.deterioration_exponent,
                          raise_uncertainty=not self._strict_information())

    def _strict_uncertainty_update(self, component: "Component", outcome) -> None:
        """Aggiornamento dell'incertezza guidato dalla sola evidenza: un calo del
        centro dell'intervallo osservato, rispetto all'osservazione precedente, oltre
        `strict_uncertainty_drop` e' letto come danno accertato (x1.10, tetto 0.40)."""
        centre = 0.5 * (float(outcome.observed_min) + float(outcome.observed_max))
        prev = getattr(component, "strict_last_observed_centre", None)
        if prev is not None and float(prev) - centre > float(getattr(self.params, "strict_uncertainty_drop", 0.10)):
            component.epistemic_uncertainty = min(0.40, component.epistemic_uncertainty * 1.10)
        component.strict_last_observed_centre = centre

    def _strict_proof_plan(self, bridge: Bridge, op: Dict[str, Any], targets) -> Dict[str, Any]:
        """strict_information: alpha, passi e componenti da provare decisi sullo stato
        STIMATO dal gestore (mai sul vero)."""
        est = self._assessed_bridge(bridge)
        names = [c.name for c in self._proof_load_relevant_components(est)]
        if op.get("max_failure_probability"):
            plan = self._select_proof_load_option(est, op, targets)
            return {"alpha": float(plan["alpha"]), "step_count": int(plan["step_count"]), "components": names,
                    "meta": {k: plan.get(k) for k in ("risk_constraint_satisfied", "used_fallback_option",
                                                       "feasible_option_count", "max_failure_probability")}}
        return {"alpha": op.get("alpha", self.params.proof_load_base_alpha),
                "step_count": op.get("step_count", self.params.proof_load_step_count), "components": names,
                "meta": {"risk_constraint_satisfied": True, "used_fallback_option": False,
                         "feasible_option_count": 1, "max_failure_probability": None}}

    def _strict_proof_load_option(self, bridge: Bridge, op: Dict[str, Any], targets) -> Dict[str, Any]:
        """Piano sullo stato stimato; rischi, esiti e posteriori della prova sul ponte vero."""
        plan = self._strict_proof_plan(bridge, op, targets)
        option = self._evaluate_bridge_proof_load_option(bridge, plan["alpha"], plan["step_count"], targets,
                                                         component_names=plan["components"])
        option.update(plan["meta"])
        option["planned_on"] = "assessed"
        return option

    def _proof_load_plan(self, bridge: Bridge, op: Dict[str, Any]) -> Dict[str, Any]:
        """Decisione di prova (alpha, passi, componenti) che il motore prenderebbe ora,
        senza effetti sullo stato. Diagnostica (infralab.lab.leak_check)."""
        targets = self._reliability_targets()
        if self._strict_proof_load():
            plan = self._strict_proof_plan(bridge, op, targets)
            return {"alpha": float(plan["alpha"]), "step_count": int(plan["step_count"]),
                    "components": list(plan["components"])}
        option = self._select_proof_load_option(bridge, op, targets)
        return {"alpha": float(option["alpha"]), "step_count": int(option["step_count"]),
                "components": [c.name for c, _ in option["component_assessments"]]}

    def refresh_bridge_reliability(self, bridge: Bridge):
        # Pf fisica: sul ponte VERO (non sulla stima del gestore) l'incertezza epistemica
        # e' fissata al riferimento per tutto il calcolo, poi ripristinata esattamente.
        # Con assessment_epistemic_mode="reference" lo stesso vale per la STIMA del gestore.
        _is_assessment = getattr(bridge, "_is_assessment", False)
        if ((getattr(self.params, "true_reliability_mode", "knowledge") == "physical" and not _is_assessment)
                or (_is_assessment and getattr(self.params, "assessment_epistemic_mode", "manager") == "reference")):
            ref = float(self.params.physical_epistemic_reference)
            saved = {n: c.epistemic_uncertainty for n, c in bridge.components.items()}
            for c in bridge.components.values():
                c.epistemic_uncertainty = ref
            try:
                return self._refresh_bridge_reliability_impl(bridge)
            finally:
                for n, c in bridge.components.items():
                    c.epistemic_uncertainty = saved[n]
        return self._refresh_bridge_reliability_impl(bridge)

    def _refresh_bridge_reliability_impl(self, bridge: Bridge):
        priors = []
        posteriors = []
        component_profiles = {}
        for component in bridge.components.values():
            prior, posterior = self._component_reliability(bridge, component)
            component.reliability_index = prior.beta
            component.failure_probability = prior.failure_probability
            component.annual_reliability_index = prior.annual_beta
            component.annual_failure_probability = prior.annual_failure_probability
            component.posterior_reliability_index = posterior.beta
            component.posterior_failure_probability = posterior.failure_probability
            component.posterior_annual_reliability_index = posterior.annual_beta
            component.posterior_annual_failure_probability = posterior.annual_failure_probability
            component.reliability_details = prior.to_dict()
            component.posterior_reliability_details = posterior.to_dict()
            component_profiles[component.name] = self._project_component_life_cycle(bridge, component)
            component.reliability_details["life_cycle_profile"] = component_profiles[component.name]
            component.posterior_reliability_details["life_cycle_profile"] = component_profiles[component.name]
            priors.append(prior)
            posteriors.append(posterior)

        bridge.traffic_fragility = self._traffic_fragility_metrics(bridge)
        # Structural-only series (no seismic) needed to drive the seismic fragility heuristic.
        _struct_prior = series_system_reliability(priors)
        bridge.annual_failure_probability = _struct_prior.annual_failure_probability

        bridge.seismic_fragility = self._seismic_fragility_metrics(bridge)
        extra_annual_pf = 0.0
        if self.params.earthquake_enabled and bridge.seismic_fragility:
            extra_annual_pf = bridge.seismic_fragility.get("annual_exceedance_proxy", 0.0) or 0.0

        bridge_prior = series_system_reliability(priors, extra_annual_failure_probability=extra_annual_pf)
        bridge_posterior = series_system_reliability(posteriors, extra_annual_failure_probability=extra_annual_pf)
        bridge.reliability_index = bridge_prior.beta
        bridge.failure_probability = bridge_prior.failure_probability
        bridge.annual_reliability_index = bridge_prior.annual_beta
        bridge.annual_failure_probability = bridge_prior.annual_failure_probability
        bridge.posterior_reliability_index = bridge_posterior.beta
        bridge.posterior_failure_probability = bridge_posterior.failure_probability
        bridge.posterior_annual_reliability_index = bridge_posterior.annual_beta
        bridge.posterior_annual_failure_probability = bridge_posterior.annual_failure_probability
        bridge.reliability_details = bridge_prior.to_dict()
        bridge.posterior_reliability_details = bridge_posterior.to_dict()
        years = max(1, int(self.params.reliability_projection_years))
        life_cycle_profile = []
        for year in range(years + 1):
            component_year_pfs = []
            for profile in component_profiles.values():
                if year < len(profile["profile"]):
                    year_pf = profile["profile"][year]["structural_annual_pf"]
                    if year_pf is not None:
                        component_year_pfs.append(year_pf)
            annual_pf_year = float(np.clip(1.0 - np.prod([1.0 - pf for pf in component_year_pfs]), 1e-12, 1.0 - 1e-12)) if component_year_pfs else 1e-12
            if extra_annual_pf > 0.0:
                annual_pf_year = 1.0 - (1.0 - annual_pf_year) * (1.0 - extra_annual_pf)
            life_cycle_profile.append({
                "year": year,
                "structural_annual_pf": annual_pf_year,
                "structural_annual_beta": beta_from_pf(annual_pf_year),
            })
        min_bridge_point = min(life_cycle_profile, key=lambda item: item["structural_annual_beta"])
        bridge.reliability_details["life_cycle_profile"] = {
            "projection_years": years,
            "profile": life_cycle_profile,
            "min_remaining_structural_annual_beta": min_bridge_point["structural_annual_beta"],
            "max_remaining_structural_annual_pf": max(item["structural_annual_pf"] for item in life_cycle_profile),
            "year_of_min_structural_beta": min_bridge_point["year"],
        }
        bridge.posterior_reliability_details["life_cycle_profile"] = bridge.reliability_details["life_cycle_profile"]

        # Time-variant reference-period reliability.
        # Replaces the stationary approximation pf(T) = 1−(1−pf_ann)^T with the
        # exact non-stationary product over the projected annual failure probabilities.
        # Ref: Rackwitz (2001), Frangopol et al. (2004).
        cumulative_survival = 1.0
        for item in life_cycle_profile:
            cumulative_survival *= max(1e-12, 1.0 - item["structural_annual_pf"])
        tv_pf = float(np.clip(1.0 - cumulative_survival, 1e-12, 1.0 - 1e-12))
        bridge.failure_probability = tv_pf
        bridge.reliability_index = beta_from_pf(tv_pf)

        posterior_pf = bridge.posterior_annual_failure_probability if bridge.posterior_annual_failure_probability is not None else bridge.annual_failure_probability or 0.0
        bridge.risk_priority_index = bridge.network_importance_factor * posterior_pf * (0.5 + 0.5 * (1.0 - bridge.min_condition))

    def refresh_network_reliability(self):
        for bridge in self.bridges.values():
            self.refresh_bridge_reliability(bridge)

    def _inspect_bridge(self, bridge: Bridge, op: Dict[str, Any], dedicated_purpose: Optional[int] = None):
        # Un'ispezione di sorveglianza intermedia osserva ma non fa ripartire la
        # cadenza ordinaria: e' aggiuntiva, non sostitutiva (Linee Guida, IO 7.4.1.2).
        keep_cadence = not bool(op.get("resets_cadence", True))
        prev_due, prev_last = bridge.next_inspection_quarter, bridge.last_inspection_quarter
        bridge.last_inspection_quarter = self.quarter
        strict = self._strict_information()
        progressive_on = _progressive.enabled(self.params)
        pod_detected = []
        for component in bridge.components.values():
            sigma = self._component_observation_sigma(component, op.get('error_sigma', 0.1))
            _k = (bridge.id, self._comp_key(component.name), self.quarter)
            if dedicated_purpose is not None:
                # Flusso dedicato sempre a chiave (es. inventario iniziale): non consuma
                # l'RNG condiviso ne' i contatori delle ispezioni ordinarie.
                _insp_rng = self._keyed_stream(dedicated_purpose, *_k)
            else:
                self._insp_counter = getattr(self, "_insp_counter", {})
                self._insp_counter[_k] = self._insp_counter.get(_k, 0) + 1
                _insp_rng = self._stream(5, *_k, self._insp_counter[_k])
            outcome = inspect_component(
                component,
                sigma,
                _insp_rng,
                bias=self.params.inspection_bias,
                pod_enabled=self.params.inspection_pod_enabled,
                pod_exponent=self.params.inspection_pod_exponent,
            )
            component.observed_condition_min = outcome.observed_min
            component.observed_condition_max = outcome.observed_max
            if getattr(self, "sudden_pending", None):
                _sudden.on_inspection(self, bridge, component.name, outcome.detected)
            if progressive_on and outcome.detected:
                pod_detected.append(component.name)
            if strict:
                self._strict_uncertainty_update(component, outcome)
            if outcome.detected or strict:
                # strict: il gestore non sa se la POD ha rilevato il danno; il
                # guadagno di conoscenza non puo' dipendere dall'esito vero.
                component.epistemic_uncertainty = max(0.04, component.epistemic_uncertainty * (1.0 - op.get('knowledge_gain', 0.15)))
            # Danno non rilevato (POD): nessun guadagno di conoscenza reale.
        if progressive_on:
            # difetti in sviluppo: l'ispezione puo' scoprirli (fattore per tipo e livello)
            _progressive.on_inspection(self, bridge, op, pod_detected)
        self.classify_bridge(bridge)
        if keep_cadence:
            bridge.next_inspection_quarter, bridge.last_inspection_quarter = prev_due, prev_last

    def _post_maintenance_feedback(self, bridge: Bridge):
        feedback_op = {
            "error_sigma": self.params.post_maintenance_observation_sigma,
            "knowledge_gain": self.params.post_maintenance_knowledge_gain,
        }
        self._inspect_bridge(bridge, feedback_op)
        bridge.last_management_feedback = {
            "quarter": self.quarter,
            "source": "post_maintenance_feedback",
            "note": "La manutenzione aggiorna in modo grossolano la conoscenza sullo stato del ponte.",
        }

    def _record_action_event(self, bridge: Bridge, action_name: str, status: str, detail: Optional[Dict[str, Any]] = None):
        event = {"quarter": self.quarter, "action": action_name, "status": status}
        if detail:
            event.update(detail)
        bridge.action_history.append(event)
        if self._strict_information() and any(e.get("action") == "danno_idraulico" for e in bridge.action_history):
            # strict_information: un danno idraulico NASCOSTO (canale storico) non deve
            # togliere dallo storico un evento che il gestore vedeva: i 12 posti contano
            # i soli eventi visibili (altrimenti la lunghezza dello storico ripulito
            # rivela il danno).
            visible = [i for i, e in enumerate(bridge.action_history) if e.get("action") != "danno_idraulico"]
            bridge.action_history = bridge.action_history[(visible[-12] if len(visible) > 12 else 0):]
        else:
            bridge.action_history = bridge.action_history[-12:]
        if status in ("executed", "imposed_by_regime", "started"):
            bridge.regime_action_done[action_name] = self.quarter

    def _action_block_reason(self, agent: str, bridge: Bridge, action_name: str) -> Optional[str]:
        op = self.maint_ops.get(action_name)
        if bridge.manager != agent:
            return "wrong_manager"
        if not op:
            return "unknown_action"
        # Linee Guida: senza censimento (L0) si può solo ispezionare/censire.
        if not getattr(bridge, "censused", True) and op.get("type") != "analysis":
            return "not_censused"
        if bridge.active_maintenance:
            return "bridge_busy"
        if action_name == "monitoring" and bridge.monitoring_ends_quarter > self.quarter:
            return "monitoring_active"
        if action_name != "L1_inspect" and bridge.last_action_name == action_name and bridge.last_action_quarter == self.quarter - 1:
            return "same_action_cooldown"
        if action_name == "proof_load_test" and bridge.last_proof_load_report and bridge.last_proof_load_report.get("quarter") == self.quarter - 1:
            return "proof_load_cooldown"
        # Regime: con una sanzione operativa attiva il gestore non puo' modificare
        # l'uso dell'opera (riapertura, cambio limitazione) finche' non adempie.
        if bridge.regime_sanction and op.get("type") == "operational":
            return "regime_sanction_active"
        # Soglia etica: il ponte chiuso perche' pericoloso non si riapre per scelta del
        # gestore; lo riapre la soglia quando lo stato noto torna sopra.
        if getattr(bridge, "floor_closure", None) and op.get("type") == "operational":
            return "ethical_floor_closure"
        # Arresto di un difetto progressivo: solo su un difetto NOTO (mai su quelli nascosti).
        if op.get("type") == _progressive.ARREST_TYPE and not _progressive.known_defects(self, bridge):
            return "no_known_defect"
        return None

    def _action_cost(self, op: Dict[str, Any], bridge: Bridge) -> float:
        """Costo di un'azione sul ponte. L'arresto di un difetto progressivo dipende dai
        difetti noti del ponte, non solo dal catalogo."""
        if op.get("type") == _progressive.ARREST_TYPE:
            return _progressive.arrest_cost(self, bridge)
        return _calculate_cost(op, bridge.area)

    def get_feasible_actions_for_bridge(self, agent: str, bridge: Bridge) -> Dict[str, Dict[str, Any]]:
        feasibility = {}
        for action_name, op in self.maint_ops.items():
            reason = self._action_block_reason(agent, bridge, action_name)
            cost = self._action_cost(op, bridge)
            feasible = reason is None and self.budgets.get(agent, 0.0) >= cost
            if not feasible and reason is None:
                reason = "insufficient_budget"
            feasibility[action_name] = {
                "feasible": feasible,
                "reason": reason,
                "cost": cost,
                "type": op.get("type"),
            }
        return feasibility

    def _sanitize_agent_plan(self, agent: str, plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        sanitized = {"maintenance": {}}
        diagnostics: List[Dict[str, Any]] = []
        available_budget = self.budgets.get(agent, 0.0)
        # Regime: quota di budget riservata alle azioni di analisi (ispezioni,
        # valutazioni, censimento). Le altre azioni non possono intaccarla.
        earmark = max(0.0, float(getattr(self.regime, "earmarked_fraction", 0.0) or 0.0)) * max(0.0, available_budget)
        for bridge_id_str, action_name in plan.get("maintenance", {}).items():
            bridge = self.bridges.get(int(bridge_id_str))
            if bridge is None:
                diagnostics.append({"bridge_id": int(bridge_id_str), "requested_action": action_name, "status": "blocked", "reason": "unknown_bridge"})
                continue
            op = self.maint_ops.get(action_name)
            cost = self._action_cost(op, bridge) if op else None
            reason = self._action_block_reason(agent, bridge, action_name)
            if reason is None and cost is not None and available_budget < cost:
                reason = "insufficient_budget"
            if reason is None and cost is not None and earmark > 0 and op.get("type") != "analysis" \
                    and available_budget - cost < earmark:
                reason = "earmarked_budget"
            if reason in ("regime_sanction_active", "earmarked_budget"):
                self.regime_stats.setdefault(agent, _new_regime_stats())["rejected_actions"] += 1
            if reason is not None:
                diagnostics.append({"bridge_id": bridge.id, "requested_action": action_name, "status": "blocked", "reason": reason})
                continue
            sanitized["maintenance"][str(bridge.id)] = action_name
            available_budget -= cost or 0.0
            diagnostics.append({"bridge_id": bridge.id, "requested_action": action_name, "status": "accepted", "reason": None})
        return sanitized, diagnostics

    def _apply_proof_load(self, bridge: Bridge, op: Dict[str, Any]):
        uncertainty_reduction = op.get("uncertainty_reduction", 0.35)
        damage_limit = op.get("damage_limit", self.params.proof_load_damage_limit)
        component_reports = []
        passed = True
        screened = False
        targets = self._reliability_targets()
        if self._strict_proof_load():
            selected_option = self._strict_proof_load_option(bridge, op, targets)
        else:
            selected_option = self._select_proof_load_option(bridge, op, targets)
        alpha = selected_option["alpha"]
        step_count = selected_option["step_count"]

        for component, assessment in selected_option["component_assessments"]:
            prior = component.reliability_details or {}
            failure_risk = assessment["failure_probability_during_test"]
            screening_risk = assessment["screening_probability"]
            pass_probability = assessment["pass_probability"]
            draw = float(self._stream(7, bridge.id, self._comp_key(component.name), self.quarter).random())
            if draw < failure_risk:
                success = False
                outcome = "failure"
            elif draw < failure_risk + screening_risk:
                success = False
                outcome = "screened"
            else:
                success = True
                outcome = "pass"

            if success:
                posterior = assessment["posterior_result"]
                component.last_proof_load_passed = True
                component.last_proof_load_quarter = self.quarter
                component.last_proof_load_alpha = alpha
                component.last_proof_load_risk = failure_risk
                component.posterior_reliability_index = posterior.beta
                component.posterior_failure_probability = posterior.failure_probability
                component.posterior_annual_reliability_index = posterior.annual_beta
                component.posterior_annual_failure_probability = posterior.annual_failure_probability
                component.posterior_reliability_details = posterior.to_dict()
            else:
                passed = False
                if outcome == "failure":
                    damage = min(0.35, max(damage_limit, failure_risk * 0.60))
                    component.apply_damage(damage, self.params.deterioration_exponent)
                else:
                    screened = True
                component.last_proof_load_passed = False
                component.last_proof_load_quarter = self.quarter
                component.last_proof_load_alpha = alpha
                component.last_proof_load_risk = failure_risk

            component_reports.append({
                "component": component.name,
                "proof_load_alpha": alpha,
                "proof_load_failure_probability": failure_risk,
                "proof_load_screening_probability": screening_risk,
                "proof_load_pass_probability": pass_probability,
                "passed": success,
                "screened": outcome == "screened",
                "targets_satisfied_if_passed": assessment["targets_satisfied"],
                "prior_beta": prior.get("beta"),
                "prior_pf": prior.get("failure_probability"),
                "prior_annual_beta": prior.get("annual_beta"),
                "posterior_beta_if_passed": assessment["posterior_result"].beta,
                "posterior_annual_beta_if_passed": assessment["posterior_result"].annual_beta,
                "step_reports": assessment["step_reports"],
            })

        self._inspect_bridge(bridge, op)
        self.refresh_bridge_reliability(bridge)
        bridge.last_proof_load_report = {
            "quarter": self.quarter,
            "alpha": alpha,
            "step_count": step_count,
            "passed": passed,
            "screened": screened,
            "components": component_reports,
            "posterior_beta": bridge.posterior_reliability_index,
            "posterior_pf": bridge.posterior_failure_probability,
            "posterior_annual_beta": bridge.posterior_annual_reliability_index,
            "posterior_annual_pf": bridge.posterior_annual_failure_probability,
            "selected_failure_probability_during_test": selected_option["failure_probability_during_test"],
            "selected_screening_probability": selected_option["screening_probability"],
            "selected_pass_probability": selected_option["pass_probability"],
            "selected_targets_satisfied": selected_option["targets_satisfied"],
            "conditioning_mode": selected_option.get("conditioning_mode", "survival_conditioning_random_test_demand"),
            "accepted_sample_ratio": selected_option.get("accepted_sample_ratio"),
            "risk_constraint_satisfied": selected_option.get("risk_constraint_satisfied", True),
            "used_fallback_option": selected_option.get("used_fallback_option", False),
            "feasible_option_count": selected_option.get("feasible_option_count", 1),
            "max_failure_probability": selected_option.get("max_failure_probability"),
        }
        bridge.last_management_feedback = {
            "quarter": self.quarter,
            "source": "proof_load",
            "note": "La prova di carico ha aggiornato beta/Pf tramite truncation della distribuzione di capacita sulle realizzazioni compatibili con il superamento del test.",
        }

    def _get_revenue_multiplier(self, bridge: Bridge) -> float:
        # Accoppiamento condizione -> produzione dell'asset type (es. pale
        # erose o drivetrain fermo riducono l'energia prodotta da una turbina).
        # revenue_health_source: da quale stato si legge la salute che riduce i ricavi.
        #  "truth" = stato vero (storico); "assessed" = stima del gestore (nessun segnale
        #  dal vero nel budget); "none" = i ricavi non dipendono dalla condizione.
        source = getattr(self.params, "revenue_health_source", "truth")
        if source == "assessed" and self.params.observation_mode == "inspected":
            health_bridge = self._assessed_bridge(bridge)
        else:
            health_bridge = bridge
        condition_mult = 1.0
        if self.spec.revenue_condition_adapter is not None and source != "none":
            condition_mult = max(0.0, min(1.0, float(self.spec.revenue_condition_adapter(self, health_bridge))))
        # Misura operativa persistente (limitazione/chiusura): i ricavi seguono
        # il traffico effettivamente ammesso (un ponte chiuso non incassa).
        condition_mult *= max(0.0, min(1.0, getattr(bridge, "traffic_restriction_factor", 1.0)))
        if bridge.active_maintenance:
            action_name = bridge.active_maintenance['action_name']
            op = self.maint_ops.get(action_name, {})
            return (1.0 - op.get('revenue_impact', 0.0)) * condition_mult
        if not self.params.enforce_guidelines or source == "none":
            return condition_mult
        hi = self.get_health_index(health_bridge)
        if hi > 0.5: return condition_mult
        if hi <= 0.2: return 0.0
        return (hi - 0.2) / (0.5 - 0.2) * condition_mult

    def get_health_index(self, bridge: Bridge) -> float:
        """Weighted average condition across all components (0–1)."""
        total_w = sum(self.component_weights.get(n, 0.0) for n in bridge.components)
        if total_w <= 0:
            return bridge.min_condition
        return sum(
            self.component_weights.get(n, 0.0) * c.current_condition
            for n, c in bridge.components.items()
        ) / total_w

    def get_net_asset_value(self, bridge: Bridge) -> float:
        """
        Net economic value of bridge asset.

        NAV = Valore_Operativo − Passivo_Manutenzione_Differita − Mancato_Guadagno_Intervento

        Valore_Operativo:
            initial_asset_value × health_index  (condizione ponderata, non il minimo)

        Passivo_Manutenzione_Differita:
            Costo stimato per riportare il ponte a condizione=1.0, proporzionale al gap.
            Reference: costo unitario di ripristino_completo; fallback = 35 % del valore iniziale.

        Mancato_Guadagno_Intervento:
            Somma non-attualizzata delle entrate perse nei trimestri restanti dell'intervento corrente.
        """
        health = self.get_health_index(bridge)

        operational_value = bridge.initial_asset_value * health

        rc_op = self.maint_ops.get('ripristino_completo')
        full_restoration_cost = (
            _calculate_cost(rc_op, bridge.area) if rc_op
            else bridge.initial_asset_value * 0.35
        )
        deferred_maintenance = full_restoration_cost * (1.0 - health)

        revenue_loss = 0.0
        if bridge.active_maintenance:
            op = self.maint_ops.get(bridge.active_maintenance['action_name'], {})
            remaining = max(0, bridge.active_maintenance['ends_at_quarter'] - self.quarter)
            revenue_loss = bridge.quarterly_revenue * op.get('revenue_impact', 0.0) * remaining

        return max(0.0, operational_value - deferred_maintenance - revenue_loss)

    def get_agent_revenue_preview(self, agent_name: str) -> float:
        if agent_name not in self.active_agents: return 0.0
        agent_bridges = [b for b in self.bridges.values() if b.manager == agent_name]
        return sum(b.quarterly_revenue * self._get_revenue_multiplier(b) for b in agent_bridges)

    def step(self, plans: Dict[str, Dict[str, Any]]) -> Tuple[Dict, Dict]:
        bridges_actioned_this_turn = set()
        self.last_plan_diagnostics = {agent: [] for agent in self.params.agent_names}
        # Stato noto al gestore all'inizio del trimestre (per il registro dei collassi).
        self._owner_before = self._owner_state_snapshot()
        self._collapse_game_over = {}

        for bridge in self.bridges.values():
            if bridge.active_maintenance and bridge.active_maintenance['ends_at_quarter'] == self.quarter:
                action_name = bridge.active_maintenance['action_name']
                op_details = self.maint_ops.get(action_name, {})
                if op_details.get('type') == _progressive.ARREST_TYPE:
                    # Fine dei lavori di arresto di un difetto progressivo: nessun guadagno
                    # di condizione e nessun rilievo post-intervento.
                    bridge.active_maintenance = None
                    continue
                gains = op_details.get('gains', {})
                for comp_name, gain_value in gains.items():
                    if (comp := bridge.components.get(comp_name)):
                        actual_gain = max(0.0, float(self._stream(6, bridge.id, self._comp_key(comp_name), self.quarter)
                                                     .normal(gain_value, op_details.get('gain_std_dev', 0))))
                        comp.apply_maintenance(actual_gain, self.params.deterioration_exponent)

                # Intervento sismico: porta la sensibilità sismica al target
                # (miglioramento riduce, adeguamento riporta al livello di norma).
                target = op_details.get('seismic_sensitivity_target')
                if target is not None:
                    current = getattr(bridge, 'seismic_sensitivity', None)
                    base = current if current is not None else self.spec.hazard_sensitivities.get("seismic", 1.0)
                    bridge.seismic_sensitivity = min(base, float(target))

                bridge.active_maintenance = None
                if getattr(self, "sudden_pending", None):
                    _sudden.on_repair(self, bridge, [c for c in gains if c in bridge.components])
                self._post_maintenance_feedback(bridge)
                self._record_action_event(bridge, action_name, "completed")
                self.classify_bridge(bridge)
                bridges_actioned_this_turn.add(bridge.id)

        action_costs_this_turn = {agent: [] for agent in self.params.agent_names}
        crew_used: Dict[str, Dict[str, float]] = {}
        self._regime_refresh_triggers()
        self._forced_this_step = {}
        for agent, plan in plans.items():
            if agent not in self.active_agents:
                continue
            sanitized_plan, diagnostics = self._sanitize_agent_plan(agent, plan)
            forced_map = self._regime_forced_actions(agent, sanitized_plan, diagnostics)
            self._forced_this_step[agent] = forced_map
            self.last_plan_diagnostics[agent] = diagnostics
            for bridge_id_str, action_name in sanitized_plan.get("maintenance", {}).items():
                forced_ob = forced_map.get(bridge_id_str)
                bridge, op = self.bridges.get(int(bridge_id_str)), self.maint_ops.get(action_name)
                if op is None and forced_ob is not None:
                    op = self._authority_catalogue.get(action_name)
                if not bridge or not op or bridge.manager != agent or bridge.active_maintenance:
                    continue

                # Capacita' operativa: ispezioni e lavori attingono alle stesse squadre.
                # Superata la capacita', l'azione slitta al trimestre successivo; gli
                # obblighi di legge hanno la precedenza (sono inseriti prima nel piano).
                if not self._crew_admits(agent, op, crew_used):
                    diagnostics.append({"bridge_id": bridge_id_str, "action": action_name,
                                        "status": "deferred", "reason": "capacity"})
                    self.crew_stats[agent] = self.crew_stats.get(agent, 0) + 1
                    continue

                cost = self._action_cost(op, bridge)
                if forced_ob is None and self.budgets[agent] < cost:
                    continue
                if forced_ob is not None:
                    funding = getattr(self.params, "forced_funding", "owner_unlimited")
                    if funding == "matched":
                        # Contributo in quota parte (23 U.S.C. 120: quota federale
                        # 80-90%): lo Stato paga `forced_matching_share`, il gestore
                        # il resto e solo se ha la cassa, altrimenti l'obbligo resta
                        # pendente come in "owner_budget".
                        share = float(getattr(self.params, "forced_matching_share", 0.8))
                        share = min(max(share, 0.0), 1.0)
                        owner_part = cost * (1.0 - share)
                        if self.budgets[agent] < owner_part:
                            st = self.regime_stats.setdefault(agent, _new_regime_stats())
                            st["forced_unfunded"] += 1
                            st["forced_unfunded_cost"] += cost
                            continue
                        st = self.regime_stats.setdefault(agent, _new_regime_stats())
                        st["state_funded_cost"] += cost - owner_part
                        cost = owner_part
                    if funding == "owner_budget" and self.budgets[agent] < cost:
                        # Obbligo non finanziabile: resta pendente (arretrato),
                        # le eventuali sanzioni del regime continuano a valere.
                        st = self.regime_stats.setdefault(agent, _new_regime_stats())
                        st["forced_unfunded"] += 1
                        st["forced_unfunded_cost"] += cost
                        continue
                    if funding == "state" or forced_ob.charged_to == "state":
                        st = self.regime_stats.setdefault(agent, _new_regime_stats())
                        st["state_funded_cost"] += cost
                        cost = 0.0
                    # Esecuzione d'ufficio: la legge costa anche a budget esaurito
                    # (charged_to='owner'); 'state' = a carico pubblico.
                    st = self.regime_stats.setdefault(agent, _new_regime_stats())
                    st["forced_actions"] += 1
                    st["forced_cost"] += cost

                self.budgets[agent] -= cost
                emissions = _calculate_emissions_tco2(op, bridge.area)
                self.emissions_tco2[agent] = self.emissions_tco2.get(agent, 0.0) + emissions
                action_costs_this_turn[agent].append(
                    {"bridge_id": int(bridge_id_str), "action": action_name, "cost": cost,
                     "emissions_tco2": round(emissions, 4)})
                bridges_actioned_this_turn.add(bridge.id)
                bridge.last_action_name = action_name
                bridge.last_action_quarter = self.quarter

                duration = op.get('duration_quarters', 0)
                if op.get('type') == 'intervention' and duration > 0:
                    bridge.active_maintenance = {"action_name": action_name, "ends_at_quarter": self.quarter + duration}
                    bridge.last_management_feedback = {
                        "quarter": self.quarter,
                        "source": "intervention_started",
                        "note": f"Azione {action_name} avviata; ponte temporaneamente indisponibile.",
                    }
                    self._record_action_event(bridge, action_name, "started", {"duration_quarters": duration})
                    if getattr(self, "progressive_defects", None):
                        _progressive.on_intervention_started(self, bridge, op)
                elif op.get('type') == _progressive.ARREST_TYPE:
                    # Arresto dei difetti progressivi NOTI del ponte: efficace dall'avvio
                    # dei lavori, che occupano il ponte per la durata dell'azione.
                    stopped = _progressive.arrest_known(self, bridge, "action")
                    if duration > 0:
                        bridge.active_maintenance = {"action_name": action_name, "ends_at_quarter": self.quarter + duration}
                    bridge.last_management_feedback = {
                        "quarter": self.quarter, "source": "progressive_arrest",
                        "note": f"Azione {action_name}: difetto in sviluppo arrestato ({', '.join(stopped)}).",
                    }
                    self._record_action_event(bridge, action_name, "executed", {"event_types": stopped})
                elif op.get('type') in ['analysis', 'monitoring']:
                    self._inspect_bridge(bridge, op)
                    if op.get('census'):
                        bridge.censused = True   # L0: l'asset diventa gestibile
                    if op.get('is_assessment') or op.get('inspection_level') == 4:
                        # Valutazione accurata L4: "chiude" l'obbligo di CdA Alta.
                        bridge.last_L4_quarter = self.quarter
                        bridge.quarters_in_cda_alta_without_L4 = 0
                    if op.get('type') == 'monitoring':
                        bridge.monitoring_ends_quarter = self.quarter + (self.params.monitoring_duration_years * 4)
                        bridge.future_curve_range = self.predict_future_degradation(bridge,
                                                                                    op.get('error_sigma', 0.005))
                    else:
                        bridge.future_curve_range = None
                    bridge.last_management_feedback = {
                        "quarter": self.quarter,
                        "source": op.get('type'),
                        "note": f"Azione {action_name} eseguita con aggiornamento osservativo.",
                    }
                    self._record_action_event(bridge, action_name, "executed")
                elif op.get('type') == 'operational':
                    # Misura operativa / limitazione d'uso: modifica la DOMANDA
                    # (traffico), non la condizione fisica. Effetto immediato.
                    bridge.traffic_restriction_factor = float(op.get('traffic_restriction_factor', 1.0))
                    bridge.last_management_feedback = {
                        "quarter": self.quarter, "source": "operational",
                        "note": f"Misura operativa {action_name}: traffico effettivo al {int(bridge.traffic_restriction_factor*100)}%.",
                    }
                    self._record_action_event(bridge, action_name, "executed")
                elif op.get('type') == 'preventive':
                    # Manutenzione preventiva: rallenta il tasso di degrado per un
                    # periodo, senza recuperare condizione.
                    bridge.maintenance_rate_multiplier = float(op.get('decay_rate_multiplier', 1.0))
                    bridge.rate_multiplier_until_quarter = self.quarter + int(op.get('rate_duration_quarters', 0))
                    bridge.last_management_feedback = {
                        "quarter": self.quarter, "source": "preventive",
                        "note": f"Manutenzione preventiva {action_name}: degrado rallentato.",
                    }
                    self._record_action_event(bridge, action_name, "executed")
                elif op.get('type') == 'proof_load':
                    self._apply_proof_load(bridge, op)
                    self._record_action_event(
                        bridge,
                        action_name,
                        "executed",
                        {"passed": bridge.last_proof_load_report.get("passed") if bridge.last_proof_load_report else None},
                    )

        floor_touched: set = set()
        if getattr(self.params, "ethical_floor_enabled", False):
            floor_touched = self._apply_ethical_floor(action_costs_this_turn, bridges_actioned_this_turn, crew_used)

        environmental_factor = max(0,
                                   float(self._stream(1, self.quarter).normal(self.params.env_degradation_mean, self.params.env_degradation_std)))
        if getattr(self.params, "degradation_pause_on", "any_action") == "interventions":
            # Solo gli interventi (e le manutenzioni in corso) sospendono il degrado:
            # ispezioni, monitoraggio, misure operative e prove di carico no.
            _paused = {b.id for b in self.bridges.values()
                       if b.id in bridges_actioned_this_turn
                       and self.maint_ops.get(getattr(b, "last_action_name", None) or "", {}).get("type") in ("intervention", "preventive")
                       and getattr(b, "last_action_quarter", None) == self.quarter}
        else:
            _paused = set(bridges_actioned_this_turn)
        climate_multiplier = self._climate_trend_multiplier()
        rate_adapter = self.spec.degradation_rate_adapter
        for bridge in self.bridges.values():
            if bridge.id in _paused or bridge.active_maintenance:
                continue
            for component in bridge.components.values():
                traffic_multiplier = self._traffic_multiplier(bridge)
                _crng = self._stream(2, bridge.id, self._comp_key(component.name), self.quarter)
                self._deg_rng = self._stream(3, bridge.id, self._comp_key(component.name), self.quarter) \
                    if getattr(self.params, "rng_streams", "shared") == "keyed" else None
                process_multiplier = max(
                    0.35,
                    float(_crng.lognormal(mean=0.0, sigma=self.params.deterioration_process_cov))
                )
                age_increase_factor = (
                    traffic_multiplier ** component.traffic_sensitivity
                ) * (
                    environmental_factor ** component.environment_sensitivity
                ) * process_multiplier * climate_multiplier
                if rate_adapter is not None:
                    # Accoppiamento sul rate (es. involucro degradato accelera
                    # il degrado di struttura e solai di un edificio).
                    age_increase_factor *= max(0.0, float(rate_adapter(self, bridge, component)))
                # Manutenzione preventiva attiva: rallenta il degrado (rate < 1).
                if self.quarter <= getattr(bridge, 'rate_multiplier_until_quarter', 0):
                    age_increase_factor *= max(0.0, float(getattr(bridge, 'maintenance_rate_multiplier', 1.0)))
                component.last_process_multiplier = age_increase_factor
                age_before = component.effective_age_years
                component.effective_age_years += 0.25 * age_increase_factor
                # Dispatch weibull/gamma/custom (preserva i modelli di default).
                self._advance_component_condition(component, age_before)
                component.epistemic_uncertainty = min(0.40, component.epistemic_uncertainty * 1.01)

        # Refresh degradation predictions for actively monitored bridges
        _monitoring_op = self.maint_ops.get('monitoring', {})
        for bridge in self.bridges.values():
            if bridge.monitoring_ends_quarter > self.quarter and bridge.future_curve_range is not None:
                bridge.future_curve_range = self.predict_future_degradation(
                    bridge, _monitoring_op.get('error_sigma', 0.005)
                )

        # Evento sismico stocastico (se abilitato)
        if self.params.earthquake_enabled:
            from infralab.core.hazards.earthquake import should_earthquake_occur, generate_random_earthquake, apply_earthquake_damage
            _eq = self._stream(8, self.quarter)
            if should_earthquake_occur(self.params.earthquake_annual_prob, rng=_eq):
                event = generate_random_earthquake(
                    quarter=self.quarter,
                    region_lat=self.params.bridge_region_lat,
                    region_lon=self.params.bridge_region_lon,
                    region_radius_km=self.params.bridge_region_radius_km * 1.5,
                    min_magnitude=self.params.earthquake_min_mag,
                    max_magnitude=self.params.earthquake_max_mag,
                    rng=_eq,
                )
                reports = apply_earthquake_damage(self.bridges, event, self.params.deterioration_exponent, rng=_eq,
                                                  raise_uncertainty=not self._strict_information())
                event.damage_reports = reports
                self.earthquake_events.append(event.to_dict())
                self.refresh_network_reliability()

        grant_fraction = float(getattr(self.params, "annual_budget_grant_fraction", 0.0) or 0.0)
        if grant_fraction > 0:
            for agent in self.active_agents:
                base = sum(float(getattr(b, "initial_asset_value", 0.0) or 0.0)
                           for b in self.bridges.values() if b.manager == agent)
                grant = grant_fraction / 4.0 * base
                self.budgets[agent] += grant
                self.regime_stats.setdefault(agent, _new_regime_stats())["grant_total"] += grant
        quarterly_revenues = {agent: 0.0 for agent in self.params.agent_names}
        _q_deposit = (1 + self.params.unallocated_budget_return_annual) ** 0.25 - 1
        _q_debt    = (1 + self.params.debt_interest_rate_annual) ** 0.25 - 1
        for agent in self.active_agents:
            if self.budgets[agent] >= 0:
                self.budgets[agent] *= (1 + _q_deposit)
            else:
                # Budget negativo: matura interessi passivi sul debito (tasso > deposito)
                self.budgets[agent] *= (1 + _q_debt)

            quarterly_revenues[agent] = self.get_agent_revenue_preview(agent)
            self.budgets[agent] += quarterly_revenues[agent]

        agents_to_remove = set()
        # Perdita attesa del trimestre, valutata sullo STESSO stato su cui si campiona
        # il collasso (dopo il degrado, prima del campionamento): attesa e realizzata
        # sono cosi' coerenti in media.
        self.last_expected_loss = {a: self.expected_quarterly_failure_loss(a) for a in self.params.agent_names}
        if self.params.failure_model == "reliability":
            self._sample_reliability_failures()
        _sudden.sample_events(self)
        _regional.sample_events(self)
        self._sample_hydraulic_events()
        _progressive.step(self)
        self._sample_posting_events()
        self._sample_temporary_closures()
        for agent in self.active_agents:
            if self.params.failure_model == "reliability":
                failing_bridges = []   # i collassi sono eventi campionati: la partita continua
            else:
                failing_bridges = [b for b in self.bridges.values() if
                                   b.manager == agent and b.structural_min_condition <= self.params.failure_threshold]
            game_over = self._collapse_game_over.get(agent)
            if game_over is not None:
                # collapse_game_over ("any" | "known_critical"): il gestore esce dalla
                # partita come nel modello storico failure_model="condition".
                agents_to_remove.add(agent)
                self.last_failure_info = {
                    "agent_name": agent, "reason": "BRIDGE_COLLAPSE", "bridge_id": game_over["bridge_id"],
                    "bridge_name": game_over["bridge_name"], "cause": game_over["cause"],
                    "known_critical": bool(game_over["known_critical"]),
                    "rule": self.params.collapse_game_over,
                    "message": f"L'agente {agent} è fallito a causa del crollo del ponte "
                               f"'{game_over['bridge_name']}' (ID: {game_over['bridge_id']})."}
            if failing_bridges:
                agents_to_remove.add(agent)
                failing_bridge = failing_bridges[0]
                self._log_collapse(failing_bridge, "condition_threshold", game_over=False)
                self.last_failure_info = {"agent_name": agent, "reason": "BRIDGE_COLLAPSE",
                                          "bridge_id": failing_bridge.id, "bridge_name": failing_bridge.name,
                                          "message": f"L'agente {agent} è fallito a causa del crollo del ponte '{failing_bridge.name}' (ID: {failing_bridge.id})."}

            self.negative_budget_streak[agent] = self.negative_budget_streak.get(agent, 0) + 1 if self.budgets.get(
                agent, 0) < 0 else 0
            if (getattr(self.params, "bankruptcy_enabled", True)
                    and self.negative_budget_streak.get(agent, 0) >= 4 and agent not in agents_to_remove):
                agents_to_remove.add(agent)
                self.last_failure_info = {"agent_name": agent, "reason": "BANKRUPTCY",
                                          "message": f"L'agente {agent} è fallito per bancarotta (budget negativo per 1 anno)."}

        every = max(1, int(getattr(self.params, "reliability_refresh_every_quarters", 1) or 1))
        if every == 1 or self.quarter % every == 0:
            self.refresh_network_reliability()
        else:
            # rivalutazione parziale: solo gli asset toccati da un'azione in questo trimestre
            for _bid in bridges_actioned_this_turn | floor_touched:
                _b = self.bridges.get(_bid)
                if _b is not None:
                    self.refresh_bridge_reliability(_b)
        self._apply_linee_guida_workflow()
        self._apply_regime_enforcement()
        self.active_agents = [a for a in self.active_agents if a not in agents_to_remove]
        self.quarter += 1
        return action_costs_this_turn, quarterly_revenues

    # -- Workflow Linee Guida Ponti (CdA multi-rischio + escalation L4) ------
    def _risk_based_cda(self, bridge: Bridge) -> str:
        """CdA da rischio combinato: condizione + sismico + esposizione, regola
        max() (un difetto grave porta CdA Alta a prescindere dal traffico)."""
        cfg = self.cda_config
        # Componente condizione (come la CdA classica).
        c = bridge.min_observed_condition()
        if c is None:
            cond_level = 0
        else:
            cond_level = (5 if c <= cfg.medio_alta_threshold else 4 if c <= cfg.media_threshold
                          else 3 if c <= cfg.medio_bassa_threshold else 2 if c <= cfg.bassa_threshold else 1)
        # Componente sismica × esposizione (se abilitata la sismica).
        seis = (bridge.seismic_fragility or {}).get("failure_probability", 0.0) if self.params.earthquake_enabled else 0.0
        exposure = min(1.0, getattr(bridge, "network_importance_factor", 1.0))
        seismic_risk = seis * (0.5 + 0.5 * exposure)
        seis_level = (5 if seismic_risk > 0.30 else 4 if seismic_risk > 0.15
                      else 3 if seismic_risk > 0.06 else 2 if seismic_risk > 0.02 else 1)
        level = max(cond_level, seis_level)
        return {5: "Alta", 4: "Medio-Alta", 3: "Media", 2: "Medio-Bassa", 1: "Bassa"}.get(level, "Bassa")

    def _apply_linee_guida_workflow(self):
        """Sovrascrive la CdA con quella multi-rischio (se abilitata) e traccia
        l'escalation: CdA Alta senza valutazione L4 recente diventa 'overdue',
        con penalità sul rischio (indicatore di conformità alle Linee Guida)."""
        multi = self.params.cda_multi_hazard
        esc = self.params.l4_escalation_quarters
        for bridge in self.bridges.values():
            if multi and bridge.cda != "Sconosciuta":
                risk_cda = self._risk_based_cda(bridge)
                order = {"Bassa": 1, "Medio-Bassa": 2, "Media": 3, "Medio-Alta": 4, "Alta": 5}
                if order.get(risk_cda, 0) > order.get(bridge.cda, 0):
                    bridge.cda = risk_cda
            if esc > 0:
                if bridge.cda == "Alta" and bridge.last_L4_quarter < self.quarter - esc:
                    bridge.quarters_in_cda_alta_without_L4 += 1
                else:
                    bridge.quarters_in_cda_alta_without_L4 = 0
                if bridge.quarters_in_cda_alta_without_L4 > esc:
                    # Inadempienza: amplifica il rischio percepito (KPI conformità).
                    bridge.risk_priority_index *= max(1.0, self.params.l4_escalation_risk_penalty)

    # ------------------------------------------------------------------
    # Regime normativo: trigger, esecuzione d'ufficio, sanzioni
    # ------------------------------------------------------------------
    def _regime_trigger_active(self, bridge: Bridge, ob) -> bool:
        t = ob.trigger
        if t == "inspection_overdue":
            grace = self.params.inspection_overdue_grace_quarters
            return self.quarter > bridge.next_inspection_quarter + grace
        if t == "network_relevance_above_threshold":
            # La legge puo' trattare diversamente i ponti piu' rilevanti per la rete
            # (esposizione: traffico x deviazione), come fa l'esposizione della classe
            # di attenzione e come fa il metodo a base di rischio del 23 CFR 650.311.
            th = 0.8 if ob.threshold is None else float(ob.threshold)
            return float(getattr(bridge, "network_relevance", 0.0)) > th
        if t == "l4_escalation_overdue":
            esc = self.params.l4_escalation_quarters
            return bool(esc > 0 and bridge.cda == "Alta" and bridge.quarters_in_cda_alta_without_L4 > esc)
        # In modalita' "inspected" la legge vede lo stato stimato del gestore,
        # non quello vero (la norma si applica a cio' che e' stato accertato).
        src = self._assessed_bridge(bridge) if self.params.observation_mode == "inspected" else bridge
        if t == "risk_above_threshold":
            return ob.threshold is not None and float(getattr(src, "risk_priority_index", 0.0) or 0.0) > float(ob.threshold)
        if t == "beta_below_target":
            beta = (src.posterior_annual_reliability_index
                    if src.posterior_annual_reliability_index is not None else src.annual_reliability_index)
            target = (src.reliability_details or {}).get("target_beta_annual")
            return beta is not None and target is not None and float(beta) < float(target)
        if t == "pf_above_threshold":
            pf = (src.posterior_annual_failure_probability
                  if src.posterior_annual_failure_probability is not None else src.annual_failure_probability)
            return ob.threshold is not None and pf is not None and float(pf) > float(ob.threshold)
        if t == "condition_below_threshold":
            c = bridge.min_observed_condition()
            return ob.threshold is not None and c is not None and float(c) < float(ob.threshold)
        if t == "not_censused":
            return not getattr(bridge, "censused", True)
        if t == _regional.TRIGGER:
            # Ponte esposto nell'area di un evento regionale (pubblico) non ancora seguito
            # da un'ispezione; threshold = intensita' locale minima (None = tutta l'area).
            return _regional.area_status(self, bridge, ob.threshold) is not None
        return False

    def _regime_refresh_triggers(self) -> None:
        """Aggiorna, per ogni asset, il trimestre di prima attivazione di ciascun trigger."""
        if not self.regime.obligations:
            return
        for bridge in self.bridges.values():
            for ob in self.regime.obligations:
                key = self._obligation_key(ob)
                if self._regime_trigger_active(bridge, ob) and not self._regime_obligation_discharged(bridge, ob):
                    bridge.regime_trigger_since.setdefault(key, self.quarter)
                else:
                    bridge.regime_trigger_since.pop(key, None)

    @staticmethod
    def _obligation_key(ob) -> str:
        """Chiave di stato di un obbligo. Deve distinguere due obblighi che condividono
        il presupposto ma impongono azioni diverse (per esempio: sotto soglia si ripara,
        e sotto un'altra soglia si sorveglia). Indicizzare sul solo presupposto li faceva
        sovrascrivere a vicenda."""
        return f"{ob.trigger}|{ob.action}|{'' if ob.threshold is None else ob.threshold}"

    def _regime_obligation_discharged(self, bridge: Bridge, ob) -> bool:
        """Obblighi con discharge='action': assolti per `recurrence_quarters` trimestri
        dall'ultima esecuzione dell'azione dovuta, anche se il trigger resta vero.

        E' il regime di transitabilita' sotto sorveglianza: il ponte ammalorato resta in
        esercizio finche' viene ispezionato con la cadenza prescritta. Senza questa
        distinzione un obbligo di sorveglianza non e' mai assolvibile, perche' l'azione
        non cambia lo stato che ha fatto scattare il trigger.
        """
        if getattr(ob, "discharge", "state") != "action":
            return False
        last = bridge.regime_action_done.get(ob.action)
        if last is None:
            return False
        return (self.quarter - int(last)) < int(ob.recurrence_quarters)

    def _regime_due_obligations(self, bridge: Bridge) -> List[Any]:
        """Obblighi attivi e oltre la tolleranza (grace) per l'asset."""
        due = []
        for ob in self.regime.obligations:
            since = bridge.regime_trigger_since.get(self._obligation_key(ob))
            if since is None:
                continue
            if self.quarter - since >= ob.grace_quarters:
                due.append(ob)
        return due

    def _regime_forced_actions(self, agent: str, sanitized_plan: Dict[str, Any],
                               diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Esecuzione d'ufficio: inserisce nel piano le azioni dovute non proposte dal gestore."""
        forced: Dict[str, Any] = {}
        if not self.regime.forces:
            return forced
        maint = sanitized_plan.setdefault("maintenance", {})
        legacy_override = bool(getattr(self.params, "regime_override_unfunded", False))
        committed = 0.0   # quota del gestore gia' impegnata dalle azioni imposte del trimestre
        for bridge in self.bridges.values():
            if bridge.manager != agent:
                continue
            key = str(bridge.id)
            for ob in self._regime_due_obligations(bridge):
                if not self.regime.obligation_forced(ob):
                    continue  # obbligo fatto valere con le sole sanzioni: nessuna esecuzione d'ufficio
                if maint.get(key) == ob.action:
                    break  # il gestore ha gia' programmato l'azione dovuta
                if key in maint and self._action_satisfies(maint[key], ob.action):
                    # Il gestore sta facendo qualcosa di equivalente o superiore
                    # (es. un intervento al posto dell'intervento dovuto, o
                    # un'ispezione di livello non inferiore): obbligo assolto.
                    break
                if not legacy_override:
                    owner_part, total = self._forced_owner_cost(bridge, ob)
                    if owner_part is not None and self.budgets.get(agent, 0.0) - committed < owner_part:
                        # Chi paga non ha la cassa: l'autorita' non puo' eseguire, quindi
                        # non annulla l'azione del gestore e passa all'obbligo successivo
                        # (un'ispezione dovuta, economica, resta imponibile). Prima
                        # l'azione del gestore veniva sostituita da un'azione poi
                        # rifiutata per mancanza di fondi: sul ponte non accadeva nulla,
                        # anche per molti trimestri di fila.
                        st = self.regime_stats.setdefault(agent, _new_regime_stats())
                        st["forced_unfunded"] += 1
                        st["forced_unfunded_cost"] += total
                        diagnostics.append({"bridge_id": bridge.id, "requested_action": ob.action,
                                            "status": "deferred", "reason": "regime_forced_unfunded"})
                        continue
                if key in maint:
                    # Il gestore ha programmato altro: l'obbligo prevale (un'azione per asset).
                    diagnostics.append({"bridge_id": bridge.id, "requested_action": maint[key],
                                        "status": "overridden_by_regime", "reason": "regime_forced_override"})
                    self.regime_stats.setdefault(agent, _new_regime_stats())["overridden_owner_actions"] += 1
                reason = self._action_block_reason(agent, bridge, ob.action)
                if reason == "unknown_action" and ob.action in self._authority_catalogue:
                    reason = None   # l'autorita' esegue anche azioni fuori dal catalogo del gestore
                if reason is not None:
                    self.regime_stats.setdefault(agent, _new_regime_stats())["forced_deferred"] += 1
                    if reason == "unknown_action":
                        self.regime_stats[agent]["forced_unexecutable"] += 1
                    diagnostics.append({"bridge_id": bridge.id, "requested_action": ob.action,
                                        "status": "deferred", "reason": f"regime_forced_{reason}"})
                    break
                maint[key] = ob.action
                forced[key] = ob
                if not legacy_override:
                    committed += self._forced_owner_cost(bridge, ob)[0] or 0.0
                diagnostics.append({"bridge_id": bridge.id, "requested_action": ob.action,
                                    "status": "forced_by_regime", "reason": ob.trigger})
                break  # una sola azione forzata per asset e trimestre
        return forced

    def _forced_owner_cost(self, bridge: Bridge, ob: Any) -> Tuple[Optional[float], float]:
        """(quota che il gestore deve avere in cassa, costo totale) di un'azione imposta.

        Quota None = sempre eseguibile (owner_unlimited, state). Stesse regole del
        controllo in step(): owner_budget -> costo intero; matched -> quota del gestore.
        """
        op = self.maint_ops.get(ob.action) or self._authority_catalogue.get(ob.action) or {}
        cost = _calculate_cost(op, bridge.area)
        funding = getattr(self.params, "forced_funding", "owner_unlimited")
        if funding == "owner_budget":
            return cost, cost
        if funding == "matched":
            share = min(max(float(getattr(self.params, "forced_matching_share", 0.8)), 0.0), 1.0)
            return cost * (1.0 - share), cost
        return None, cost

    _ACTION_RANK = {"analysis": 1, "monitoring": 2, "proof_load": 3, "preventive": 4, "intervention": 5}

    def _action_satisfies(self, planned: str, due: str) -> bool:
        """L'azione del gestore assolve l'obbligo se e' della stessa famiglia e non inferiore."""
        cat = self._authority_catalogue
        p, d = cat.get(planned) or self.maint_ops.get(planned), cat.get(due) or self.maint_ops.get(due)
        if not p or not d:
            return False
        rp, rd = self._ACTION_RANK.get(p.get("type"), 0), self._ACTION_RANK.get(d.get("type"), 0)
        if rd == 1:      # obbligo di ispezione: vale un'ispezione di livello non inferiore
            return (rp == 1 and int(p.get("inspection_level", 0) or 0) >= int(d.get("inspection_level", 0) or 0)) or rp >= 4
        return rp >= rd

    def _apply_regime_enforcement(self) -> None:
        """Post-step: inadempienze, sanzioni operative progressive, multe, revoca a conformita' raggiunta."""
        if not self.regime.obligations:
            return
        self._regime_refresh_triggers()   # dopo le azioni del trimestre: un'ispezione eseguita chiude il trigger
        enf = self.regime.enforcement
        for bridge in self.bridges.values():
            agent = bridge.manager
            if agent not in self.active_agents:
                continue
            st = self.regime_stats.setdefault(agent, _new_regime_stats())
            due = self._regime_due_obligations(bridge)
            if due:
                st["breach_bridge_quarters"] += 1
                if enf.fine_per_quarter_m2 > 0 and self.regime.enforces:
                    fine = float(enf.fine_per_quarter_m2) * float(bridge.area)
                    self.budgets[agent] -= fine
                    st["fines_total"] += fine
            if self.regime.sanctions:
                # solo gli obblighi fatti valere con le sanzioni muovono la scala e la revoca
                due = [ob for ob in due if self.regime.obligation_sanctioned(ob)]
                if due and enf.sanction_ladder:
                    duration = max(self.quarter - bridge.regime_trigger_since[self._obligation_key(ob)] + 1 for ob in due)
                    step = None
                    soglia = getattr(enf, "closure_condition_threshold", None)
                    vista = bridge.min_observed_condition()
                    for cand in enf.sanction_ladder:   # ordinata per after_quarters crescente
                        if duration < cand.after_quarters:
                            continue
                        if soglia is not None:
                            op_cand = (self.maint_ops.get(cand.action)
                                       or self._authority_catalogue.get(cand.action) or {})
                            chiude = float(op_cand.get("traffic_restriction_factor", 1.0)) <= 0.0
                            if chiude and (vista is None or float(vista) >= float(soglia)):
                                continue   # chiusura solo se l'evidenza la giustifica
                        step = cand
                    if step is not None:
                        # La misura imposta dall'autorita' esiste anche se lo scenario
                        # ha un catalogo azioni ridotto: si legge dal catalogo completo.
                        op = self.maint_ops.get(step.action) or self._authority_catalogue.get(step.action)
                        if op is None:
                            raise ValueError(f"Regime '{self.regime.id}': misura sanzionatoria '{step.action}' non nel catalogo azioni")
                        factor = float(op.get("traffic_restriction_factor", 1.0))
                        if not bridge.regime_sanction:
                            # Sotto chiusura per soglia etica il fattore da ripristinare e'
                            # quello precedente alla chiusura, non lo zero della chiusura.
                            _fc = getattr(bridge, "floor_closure", None)
                            _restored = (float(_fc.get("restored_factor", 1.0)) if _fc
                                         else float(bridge.traffic_restriction_factor))
                            bridge.regime_sanction = {"action": step.action, "since_quarter": self.quarter,
                                                      "restored_factor": _restored,
                                                      "trigger": due[0].trigger}
                        bridge.regime_sanction["action"] = step.action
                        bridge.regime_sanction["level"] = enf.sanction_ladder.index(step) + 1
                        bridge.traffic_restriction_factor = min(float(bridge.traffic_restriction_factor), factor)
                        bridge.last_management_feedback = {
                            "quarter": self.quarter, "source": "regime_sanction",
                            "note": f"Sanzione '{step.action}' imposta dal regime {self.regime.id} (inadempienza: {due[0].trigger}).",
                        }
                        self._record_action_event(bridge, step.action, "imposed_by_regime",
                                                  {"trigger": due[0].trigger, "duration_quarters": duration})
                if bridge.regime_sanction:
                    st["sanction_bridge_quarters"] += 1
                    if float(bridge.traffic_restriction_factor) <= 0.0:
                        st["closure_bridge_quarters"] += 1
                    if not due and enf.lift_on_compliance:
                        _restored = float(bridge.regime_sanction.get("restored_factor", 1.0))
                        if getattr(bridge, "floor_closure", None):
                            # la revoca della sanzione non riapre un ponte chiuso dalla soglia etica
                            bridge.floor_closure["restored_factor"] = _restored
                        else:
                            bridge.traffic_restriction_factor = _restored
                        bridge.last_management_feedback = {
                            "quarter": self.quarter, "source": "regime_sanction_lifted",
                            "note": f"Sanzione revocata: obbligo assolto (regime {self.regime.id}).",
                        }
                        self._record_action_event(bridge, bridge.regime_sanction.get("action", "sanction"), "lifted_by_regime")
                        bridge.regime_sanction = None

    def detour_user_cost_quarter(self, agent_name: str) -> float:
        """Costo per gli utenti, in un trimestre, del traffico deviato.

        Somma su ogni asset di (1 - disponibilita') x ADT x km di deviazione x costo
        unitario per veicolo-km x 91.25 giorni. La disponibilita' tiene conto sia delle
        misure operative (limitazione, senso unico, chiusura) sia della riduzione di
        servizio durante i lavori, altrimenti una riabilitazione di cinque trimestri
        risulterebbe per la collettivita' meno cara di una chiusura di un trimestre.

        Grandezza CONTABILIZZATA, non inclusa nel punteggio: con le tariffe del manuale
        europeo dei costi esterni un ponte mediano chiuso costa agli utenti piu' di
        quanto il gestore incassi in un anno, quindi sommarla trasformerebbe uno studio
        sulla regolazione in uno studio sul traffico.
        """
        if not getattr(self.params, "detour_accounting", False):
            return 0.0
        c_car = float(getattr(self.params, "detour_cost_car_eur_km", 0.41))
        c_truck = float(getattr(self.params, "detour_cost_truck_eur_km", 1.15))
        # Opzioni del modello di costo utenti (default = storico):
        #  detour_km_cap: tetto ai km di deviazione (0 = nessuno). Il codice NBI 199
        #    dell'item 019 significa "nessuna deviazione possibile", non 199 km.
        #  detour_partial_diversion: "all" = una limitazione parziale devia la quota
        #    (1 - disponibilita') di tutto il traffico (storico); "trucks" = devia solo i
        #    mezzi pesanti (una limitazione di portata ferma i carichi, non le auto).
        #  detour_works_diversion: True = i lavori in corso deviano la quota
        #    revenue_impact del traffico (storico); False = solo la chiusura vera devia.
        km_cap = float(getattr(self.params, "detour_km_cap", 0.0) or 0.0)
        partial = getattr(self.params, "detour_partial_diversion", "all")
        works_div = bool(getattr(self.params, "detour_works_diversion", True))
        total = 0.0
        for bridge in self.bridges.values():
            if bridge.manager != agent_name:
                continue
            km = float(getattr(bridge, "detour_km", 0.0))
            if km <= 0.0:
                continue
            if km_cap > 0.0:
                km = min(km, km_cap)
            availability = float(bridge.traffic_restriction_factor)
            active = getattr(bridge, "active_maintenance", None) or {}
            op = self.maint_ops.get(active.get("action_name", ""), {}) if isinstance(active, dict) else {}
            if op and works_div:
                availability *= max(0.0, 1.0 - float(op.get("revenue_impact", 0.0)))
            diverted = max(0.0, 1.0 - availability)
            if diverted <= 0.0:
                continue
            trk = float(getattr(bridge, "truck_fraction", 0.0))
            if partial == "trucks" and availability > 0.0:
                # limitazione parziale: deviano i soli mezzi pesanti
                total += trk * float(bridge.adt) * km * c_truck * 91.25
                continue
            c_blend = (1.0 - trk) * c_car + trk * c_truck
            total += diverted * float(bridge.adt) * km * c_blend * 91.25
        return total

    def expected_quarterly_failure_loss(self, agent_name: str) -> Dict[str, float]:
        """Perdita attesa da collasso nel trimestre per gli asset dell'agente.

        Per asset: p_q = 1 - (1 - Pf annuale)^(1/4) (posterior se c'e', serie componenti
        + sisma) x C0 x (1 + H/C0), con C0 = valore iniziale dell'asset (costo di
        sostituzione) e H/C0 = params.failure_consequence_ratio. Somma sugli asset.
        E' una lettura dello stato: non modifica budget, stato o RNG.
        """
        ratio = max(0.0, float(getattr(self.params, "failure_consequence_ratio", 3.0) or 0.0))
        total, replacement, consequence = 0.0, 0.0, 0.0
        for b in self.bridges.values():
            if b.manager != agent_name:
                continue
            pf = (b.posterior_annual_failure_probability
                  if b.posterior_annual_failure_probability is not None else b.annual_failure_probability)
            if pf is None:
                continue
            # stessa probabilita' trimestrale del campionatore dei collassi
            pq = 1.0 - (1.0 - min(1.0, max(0.0, float(pf)))) ** 0.25
            c0 = max(0.0, float(getattr(b, "initial_asset_value", 0.0) or 0.0))
            replacement += pq * c0
            consequence += pq * c0 * ratio
        total = replacement + consequence
        return {"total": total, "replacement": replacement, "consequence": consequence}

    # ------------------------------------------------------------------
    # (c) Validazione del regime
    # ------------------------------------------------------------------
    def _validate_regime_actions(self) -> None:
        """Ogni azione imposta dalla legge deve esistere nel catalogo completo."""
        missing = []
        for ob in self.regime.obligations:
            if ob.action not in self._authority_catalogue:
                missing.append(f"obbligo {ob.trigger} -> {ob.action}")
        for st in self.regime.enforcement.sanction_ladder:
            if st.action not in self._authority_catalogue:
                missing.append(f"sanzione -> {st.action}")
        if missing:
            raise ValueError(f"Regime '{self.regime.id}': azioni inesistenti nel catalogo: {missing}")

    # ------------------------------------------------------------------
    # (a) Stato stimato del gestore (observation_mode="inspected")
    # ------------------------------------------------------------------
    def _assessment_key(self, bridge: Bridge) -> tuple:
        comps = tuple(sorted(
            (n, None if c.observed_condition_min is None else round(float(c.observed_condition_min), 9),
             None if c.observed_condition_max is None else round(float(c.observed_condition_max), 9),
             round(float(c.epistemic_uncertainty), 2), c.last_proof_load_quarter)
            for n, c in bridge.components.items()))
        # l'eta' nominale conta solo per i componenti mai osservati: aggiornata per anno
        nominal_year = (self.quarter - 1) // 4
        if getattr(self.params, "assessment_state_mode", "last_observation") == "projected":
            nominal_year = self.quarter                  # la proiezione avanza ogni trimestre
        return (bridge.id, nominal_year, comps, getattr(bridge, "censused", True))

    def _assessed_bridge(self, bridge: Bridge) -> Bridge:
        """Copia del ponte con lo stato che il gestore puo' conoscere.

        - componente osservato: condizione = centro dell'intervallo osservato;
        - componente mai osservato: credenza nominale (eta' iniziale + tempo trascorso)
          con i parametri MEDI del profilo, non quelli campionati (latenti);
        - parametri di resistenza/domanda: medi del profilo per tutti i componenti;
        - incertezza epistemica: quella del gestore (stato di conoscenza, legittimo);
        - beta, Pf, proiezione e indice di rischio ricalcolati su questo stato.
        """
        key = self._assessment_key(bridge)
        # Una sola stima per asset (quella corrente): la memoria non cresce col tempo.
        hit = self._assessed_cache.get(bridge.id)
        if hit is not None and hit[0] == key:
            return hit[1]
        exponent = self.params.deterioration_exponent
        clone = Bridge.from_dict(bridge.to_dict(), exponent)   # to_dict e' gia' una copia
        clone.id = 10 ** 6 + int(bridge.id)          # seed MC distinti da quelli del ponte vero
        if getattr(self.params, "assessment_seed_mode", "distinct") == "shared":
            clone._mc_seed_id = int(bridge.id)         # stessi campioni MC del ponte vero
        clone._is_assessment = True                    # la stima usa l'incertezza reale del gestore
        clone.manager = bridge.manager
        elapsed_years = (self.quarter - 1) / 4.0
        strict = self._strict_information()
        mean_params = getattr(self.params, "assessment_parameter_mode", "mean") == "mean"
        projected = getattr(self.params, "assessment_state_mode", "last_observation") == "projected"
        for name, comp in clone.components.items():
            prof = comp.profile or {}
            if mean_params:
                if strict:
                    comp.vita_utile_anni = self._nominal_life(comp)   # non la vita campionata
                    comp.last_process_multiplier = 1.0                # moltiplicatore vero: ignoto
                comp.degradation_scale = float(prof.get("scale_mean", comp.degradation_scale))
                comp.degradation_shape_factor = float(prof.get("shape_factor", comp.degradation_shape_factor))
                for attr in ("traffic_sensitivity", "environment_sensitivity", "resistance_bias", "demand_bias", "crack_bias"):
                    if attr in prof:
                        setattr(comp, attr, float(prof[attr]))
            lo, hi = comp.observed_condition_min, comp.observed_condition_max
            if lo is not None and hi is not None:
                comp.current_condition = max(1e-6, min(1.0, 0.5 * (float(lo) + float(hi))))
                comp.apply_maintenance(0.0, exponent)           # inverte l'eta' efficace dalla condizione
                if projected:
                    # trimestre dell'osservazione: il primo in cui la stima vede questo intervallo
                    obs = self._assessment_obs_quarter.get((bridge.id, name))
                    if obs is None or obs[0] != (lo, hi):
                        obs = ((lo, hi), self.quarter)
                        self._assessment_obs_quarter[(bridge.id, name)] = obs
                    dt = max(0, self.quarter - obs[1]) / 4.0
                    if dt > 0:
                        comp.effective_age_years += dt
                        comp.update_condition(exponent)
                comp.observed_condition_min, comp.observed_condition_max = lo, hi
                comp.epistemic_uncertainty = bridge.components[name].epistemic_uncertainty
            else:
                comp.effective_age_years = float(comp.initial_age_years) + elapsed_years
                comp.update_condition(exponent)
        self.refresh_bridge_reliability(clone)
        self._assessed_cache[bridge.id] = (key, clone)
        return clone

    _TRUTH_ONLY_FIELDS = ("accumulated_damage",)

    def _assessed_view(self, bridge: Bridge, b_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sostituisce nel dict d'osservazione i campi derivati dallo stato vero."""
        est = self._assessed_bridge(bridge)
        est_dict = est.to_dict()
        for k in ("min_condition", "current_asset_value", "reliability_index", "failure_probability",
                  "annual_reliability_index", "annual_failure_probability",
                  "posterior_reliability_index", "posterior_failure_probability",
                  "posterior_annual_reliability_index", "posterior_annual_failure_probability",
                  "reliability_details", "posterior_reliability_details", "risk_priority_index",
                  "traffic_fragility", "seismic_fragility", "structural_min_condition"):
            if k in est_dict:
                b_dict[k] = est_dict[k]
        b_dict["components"] = est_dict.get("components", {})
        for comp in b_dict["components"].values():
            for f in self._TRUTH_ONLY_FIELDS:
                comp.pop(f, None)
        b_dict["health_index"] = self.get_health_index(est)
        b_dict["net_asset_value"] = self.get_net_asset_value(est)
        b_dict["observation_mode"] = "inspected"
        return b_dict

    def _oracle_view(self, bridge: Bridge, b_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Oracolo (oracle_observation): l'agente riceve lo stato VERO del ponte.

        Condizione minima osservata = condizione minima vera; per ogni componente
        l'intervallo osservato collassa sulla condizione vera; beta, Pf e indici sono
        gia' quelli del ponte vero. Solo il dizionario consegnato all'agente cambia:
        trigger di legge, soglia etica e punteggio continuano a leggere lo stato che
        prescrive observation_mode. I componenti sono copiati: il dizionario di un
        componente e' il suo __dict__ e non va modificato.
        """
        b_dict["min_observed_condition"] = float(bridge.min_condition)
        b_dict["structural_min_condition"] = float(bridge.structural_min_condition)
        comps = {}
        for name, comp in bridge.components.items():
            d = comp.to_dict()
            d["observed_condition_min"] = float(comp.current_condition)
            d["observed_condition_max"] = float(comp.current_condition)
            comps[name] = d
        b_dict["components"] = comps
        b_dict["observation_mode"] = "oracle"
        return b_dict

    # ------------------------------------------------------------------
    # Soglia etica (ethical_floor_enabled)
    # ------------------------------------------------------------------
    def _floor_triggered(self, bridge: Bridge) -> bool:
        """La soglia etica deve agire sul ponte? Stato NOTO sotto soglia
        (_floor_condition_triggered) oppure, con gli eventi progressivi, un difetto in
        sviluppo NOTO (progressive_floor_trigger). Mai lo stato vero."""
        if _progressive.floor_trigger(self, bridge):
            return True
        return self._floor_condition_triggered(bridge)

    def _floor_condition_triggered(self, bridge: Bridge) -> bool:
        """Lo stato NOTO del ponte e' sotto la soglia etica? Mai lo stato vero: la
        condizione e' quella minima osservata, il beta quello annuo stimato dal gestore
        (stato stimato con observation_mode="inspected", altrimenti quello del motore)."""
        seen = bridge.min_observed_condition()
        if seen is not None and float(seen) < float(self.params.ethical_floor_condition):
            return True
        beta_th = getattr(self.params, "ethical_floor_beta", None)
        if beta_th is None:
            return False
        src = self._assessed_bridge(bridge) if self.params.observation_mode == "inspected" else bridge
        beta = (src.posterior_annual_reliability_index
                if src.posterior_annual_reliability_index is not None else src.annual_reliability_index)
        return beta is not None and float(beta) < float(beta_th)

    def _floor_intervention_choice(self, agent: str, bridge: Bridge) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """Intervento strutturale piu' economico che rimette in sicurezza lo stato noto.

        Preferenza, a parita' di livello per costo crescente: (0) interventi il cui
        guadagno atteso riporta sopra soglia ogni componente osservato sotto soglia
        (con il solo trigger sul beta: che agiscono sul componente osservato peggiore);
        (1) interventi che agiscono almeno su uno di quei componenti; (2) gli altri.
        Solo interventi con durata (quelli senza durata non applicano guadagni).
        """
        th = float(self.params.ethical_floor_condition)
        observed = {n: float(c.observed_condition_min) for n, c in bridge.components.items()
                    if c.observed_condition_min is not None}
        critical = [n for n, v in observed.items() if v < th]
        if not critical and observed:
            critical = [min(observed, key=lambda n: (observed[n], n))]
        ranked = []
        for name, op in self.maint_ops.items():
            if op.get("type") != "intervention" or int(op.get("duration_quarters", 0) or 0) <= 0:
                continue
            if self._action_block_reason(agent, bridge, name) is not None:
                continue
            gains = op.get("gains") or {}
            g = {c: float(gains.get(c, 0.0) or 0.0) for c in critical}
            if all(g[c] > 0.0 and (g[c] >= 1.0 or observed[c] + g[c] >= th) for c in critical):
                tier = 0
            elif any(v > 0.0 for v in g.values()):
                tier = 1
            else:
                tier = 2
            ranked.append((tier, _calculate_cost(op, bridge.area), name, op))
        if not ranked:
            return None
        tier, cost, name, op = min(ranked, key=lambda t: (t[0], t[1], t[2]))
        return name, op, cost

    def _floor_close(self, bridge: Bridge) -> None:
        """Chiusura al traffico con la stessa meccanica della chiusura sanzionatoria
        (fattore di traffico a zero: domanda, ricavi e costo di deviazione la seguono)."""
        factor = 0.0
        if str(getattr(self.params, "ethical_floor_response", "repair_or_close")) == "restrict_only":
            # solo limitazione di carico: il ponte resta aperto col fattore del declassamento
            factor = min(float(bridge.traffic_restriction_factor),
                         float(getattr(self.params, "posting_restriction_factor", 0.6)))
        bridge.floor_closure = {"since_quarter": self.quarter, "factor": factor,
                                "restored_factor": float(bridge.traffic_restriction_factor)}
        bridge.traffic_restriction_factor = factor
        bridge.last_management_feedback = {
            "quarter": self.quarter, "source": "ethical_floor_closure",
            "note": "Stato noto sotto la soglia etica e intervento non eseguibile: ponte chiuso al traffico.",
        }
        self._record_action_event(bridge, "chiusura_ponte", "imposed_by_ethical_floor")
        self.floor_stats.setdefault(bridge.manager, _new_floor_stats())["closures"] += 1

    def _floor_reopen(self, bridge: Bridge) -> None:
        """Riapertura: torna il fattore precedente alla chiusura, salvo le limitazioni
        ancora in vigore (sanzione del regime, declassamento di portata)."""
        fc = bridge.floor_closure or {}
        factor = float(fc.get("restored_factor", 1.0))
        if bridge.regime_sanction:
            s_op = (self.maint_ops.get(bridge.regime_sanction.get("action"))
                    or self._authority_catalogue.get(bridge.regime_sanction.get("action")) or {})
            factor = min(factor, float(s_op.get("traffic_restriction_factor", 1.0)))
        if int(getattr(bridge, "posting_until_quarter", 0) or 0) > self.quarter:
            factor = min(factor, float(getattr(self.params, "posting_restriction_factor", 0.6)))
        bridge.traffic_restriction_factor = factor
        bridge.floor_closure = None
        bridge.last_management_feedback = {
            "quarter": self.quarter, "source": "ethical_floor_reopening",
            "note": "Stato noto di nuovo sopra la soglia etica: ponte riaperto.",
        }
        self._record_action_event(bridge, "riapertura", "lifted_by_ethical_floor")

    def _apply_ethical_floor(self, action_costs: Dict[str, List[Dict[str, Any]]], actioned: set,
                             crew_used: Dict[str, Dict[str, float]]) -> set:
        """Non si puo' lasciar cadere consapevolmente un ponte; al piu', senza cassa, lo
        si chiude con tutte le conseguenze.

        Dopo il piano del gestore e le azioni imposte: per ogni ponte il cui stato noto
        e' sotto soglia, senza lavori in corso ne' avviati nel trimestre, si esegue a
        carico del gestore l'intervento scelto da `_floor_intervention_choice` se la
        cassa lo copre e le squadre lo ammettono; altrimenti il ponte viene chiuso. Un
        ponte chiuso resta chiuso finche' ci sono lavori in corso e viene riaperto
        quando lo stato noto torna sopra soglia. Restituisce gli asset toccati.

        Difetto progressivo NOTO (progressive_floor_trigger): stessa regola agisci-o-
        chiudi. Si esegue l'azione di arresto (un trimestre), a meno che l'intervento
        scelto per lo stato sotto soglia ripristini gia' i componenti bersaglio e quindi
        arresti anche il difetto; senza cassa o squadre il ponte viene chiuso e la soglia
        riprova ogni trimestre. Contatori propri in progressive_stats (floor_*).
        """
        touched: set = set()
        candidates: List[Bridge] = []
        for bridge in self.bridges.values():
            if bridge.manager not in self.active_agents or bridge.active_maintenance:
                continue   # lavori in corso: l'eventuale chiusura resta fino al completamento
            if self._floor_triggered(bridge):
                candidates.append(bridge)
            elif getattr(bridge, "floor_closure", None):
                self._floor_reopen(bridge)
                touched.add(bridge.id)

        def _seen(b: Bridge) -> float:
            v = b.min_observed_condition()
            return 1.0 if v is None else float(v)

        # i ponti peggiori per primi: sono loro ad avere la precedenza su cassa e squadre
        for bridge in sorted(candidates, key=lambda b: (_seen(b), b.id)):
            agent = bridge.manager
            st = self.floor_stats.setdefault(agent, _new_floor_stats())
            choice = self._floor_intervention_choice(agent, bridge)
            defect = _progressive.floor_trigger(self, bridge)
            arrest = False
            pst = _progressive._stats(self, agent) if defect else {}
            if defect:
                known = _progressive.known_defects(self, bridge)
                covered = (choice is not None and self._floor_condition_triggered(bridge)
                           and all(_progressive.intervention_arrests(self, bridge, choice[1], d) for d in known))
                if not covered:
                    # prima il difetto in sviluppo (ha una scadenza ignota al gestore)
                    arrest = True
                    a_op = self.maint_ops.get(_progressive.ARREST_ACTION)
                    choice = None
                    if a_op is not None and self._action_block_reason(agent, bridge, _progressive.ARREST_ACTION) is None:
                        choice = (_progressive.ARREST_ACTION, a_op, _progressive.arrest_cost(self, bridge))
            if str(getattr(self.params, "ethical_floor_response", "repair_or_close")) != "repair_or_close":
                choice = None   # la soglia non ripara: resta la sola misura operativa (chiusura o limitazione)
            executed = False
            if choice is not None:
                name, op, cost = choice
                if self.budgets.get(agent, 0.0) < cost:
                    st["budget_blocked"] += 1
                    if defect:
                        pst["floor_budget_blocked"] += 1
                elif not self._crew_admits(agent, op, crew_used):
                    st["capacity_blocked"] += 1
                    if defect:
                        pst["floor_capacity_blocked"] += 1
                elif arrest:
                    self.budgets[agent] -= cost
                    emissions = _calculate_emissions_tco2(op, bridge.area)
                    self.emissions_tco2[agent] = self.emissions_tco2.get(agent, 0.0) + emissions
                    action_costs.setdefault(agent, []).append(
                        {"bridge_id": int(bridge.id), "action": name, "cost": cost,
                         "emissions_tco2": round(emissions, 4), "ethical_floor": True})
                    actioned.add(bridge.id)
                    bridge.last_action_name = name
                    bridge.last_action_quarter = self.quarter
                    duration = int(op.get("duration_quarters", 0) or 0)
                    stopped = _progressive.arrest_known(self, bridge, "floor")
                    if duration > 0:
                        bridge.active_maintenance = {"action_name": name, "ends_at_quarter": self.quarter + duration}
                    bridge.last_management_feedback = {
                        "quarter": self.quarter, "source": "ethical_floor_arrest",
                        "note": f"Difetto in sviluppo noto ({', '.join(stopped)}): arresto d'ufficio a carico del gestore.",
                    }
                    self._record_action_event(bridge, name, "executed",
                                              {"event_types": stopped, "ethical_floor": True})
                    st["interventions"] += 1
                    st["spend"] += cost
                    pst["floor_arrests"] += 1
                    pst["floor_arrest_spend"] += cost
                    executed = True
                else:
                    self.budgets[agent] -= cost
                    emissions = _calculate_emissions_tco2(op, bridge.area)
                    self.emissions_tco2[agent] = self.emissions_tco2.get(agent, 0.0) + emissions
                    action_costs.setdefault(agent, []).append(
                        {"bridge_id": int(bridge.id), "action": name, "cost": cost,
                         "emissions_tco2": round(emissions, 4), "ethical_floor": True})
                    actioned.add(bridge.id)
                    bridge.last_action_name = name
                    bridge.last_action_quarter = self.quarter
                    duration = int(op.get("duration_quarters", 0) or 0)
                    bridge.active_maintenance = {"action_name": name, "ends_at_quarter": self.quarter + duration}
                    bridge.last_management_feedback = {
                        "quarter": self.quarter, "source": "ethical_floor_intervention",
                        "note": f"Stato noto sotto la soglia etica: intervento {name} avviato d'ufficio a carico del gestore.",
                    }
                    self._record_action_event(bridge, name, "started",
                                              {"duration_quarters": duration, "ethical_floor": True})
                    if getattr(self, "progressive_defects", None):
                        _progressive.on_intervention_started(self, bridge, op)
                    st["interventions"] += 1
                    st["spend"] += cost
                    executed = True
            touched.add(bridge.id)
            if not executed and not getattr(bridge, "floor_closure", None):
                self._floor_close(bridge)
                if defect:
                    pst["floor_closures"] += 1

        for bridge in self.bridges.values():
            if getattr(bridge, "floor_closure", None) and bridge.manager in self.active_agents:
                # la chiusura prevale su ogni altra misura operativa finche' dura
                bridge.traffic_restriction_factor = min(float(bridge.traffic_restriction_factor),
                                                        float(bridge.floor_closure.get("factor", 0.0)))
                self.floor_stats.setdefault(bridge.manager, _new_floor_stats())["closure_bridge_quarters"] += 1
        return touched

    # ------------------------------------------------------------------
    # (b) Collasso campionato dalla Pf (failure_model="reliability")
    # ------------------------------------------------------------------
    def _sample_reliability_failures(self) -> None:
        ratio = max(0.0, float(getattr(self.params, "failure_consequence_ratio", 3.0) or 0.0))
        exponent = self.params.deterioration_exponent
        for bridge in self.bridges.values():
            if bridge.manager not in self.active_agents:
                continue
            pf = (bridge.posterior_annual_failure_probability
                  if bridge.posterior_annual_failure_probability is not None else bridge.annual_failure_probability)
            if pf is None:
                continue
            pq = 1.0 - (1.0 - min(1.0, max(0.0, float(pf)))) ** 0.25
            # Flusso dedicato (seme, asset, trimestre): due strategie diverse sullo stesso
            # seme affrontano lo stesso "destino" (numeri casuali comuni), e il campionamento
            # non consuma l'RNG della simulazione.
            base = int(self.params.seed) if self.params.seed is not None else 0
            u = float(np.random.default_rng([base & 0x7FFFFFFF, int(bridge.id), int(self.quarter), 7919,
                                             int(self.failure_stream_salt) & 0x7FFFFFFF]).random())
            if u >= pq:
                continue
            c0 = max(0.0, float(getattr(bridge, "initial_asset_value", 0.0) or 0.0))
            st = self.failure_stats.setdefault(bridge.manager, {"failures": 0, "realised_loss": 0.0, "realised_loss_c0": 0.0})
            st["failures"] += 1
            st["realised_loss"] += c0 * (1.0 + ratio)
            st["realised_loss_c0"] += c0
            self._log_collapse(bridge, "pf", annual_pf=float(pf), quarterly_p=pq)
            if getattr(self, "sudden_pending", None):
                _sudden.on_failure(self, bridge)
            # Ricostruzione: l'opera torna nuova (costo contabilizzato nella perdita
            # realizzata, non nel budget del gestore, come la perdita attesa).
            for comp in bridge.components.values():
                comp.apply_maintenance(1.0, exponent)
            bridge.active_maintenance = None
            self.classify_bridge(bridge)
            self._record_action_event(bridge, "collasso_e_ricostruzione", "failure",
                                      {"annual_pf": float(pf), "quarterly_p": pq})
            self.last_failure_info = {"agent_name": bridge.manager, "reason": "SAMPLED_COLLAPSE",
                                      "bridge_id": bridge.id, "bridge_name": bridge.name,
                                      "message": f"Collasso del ponte '{bridge.name}' (Pf annua {float(pf):.2e}); ricostruito."}
            self.refresh_bridge_reliability(bridge)   # solo l'opera ricostruita

    def _crew_load(self, op: Dict[str, Any]) -> float:
        """Squadre-trimestre richieste da un'azione. Le misure operative (limitazione,
        chiusura) non impegnano squadre di manutenzione."""
        t = op.get("type")
        if t in ("intervention", _progressive.ARREST_TYPE):
            return float(getattr(self.params, "crew_load_intervention", 1.0))
        if t == "preventive":
            return float(getattr(self.params, "crew_load_preventive", 0.5))
        if t == "monitoring":
            return float(getattr(self.params, "crew_load_monitoring", 0.1))
        if t in ("analysis", "proof_load"):
            return float(getattr(self.params, "crew_load_analysis", 0.25))
        return 0.0

    def _classifier_values(self, bridge: Bridge) -> Dict[str, float]:
        """Grandezze osservabili su cui una legge puo' classificare l'opera. Sono tutte
        ricavabili dall'evidenza ispettiva e dall'inventario: nessuna legge legge lo
        stato vero.

        Con strict_information e observation_mode="inspected" beta ed eta' sono quelli
        dello stato stimato (_assessed_bridge); storicamente erano letti sul ponte vero."""
        seen = bridge.min_observed_condition()
        if self._strict_information() and self.params.observation_mode == "inspected":
            bridge = self._assessed_bridge(bridge)
        det = (bridge.posterior_reliability_details or bridge.reliability_details or {})
        target = det.get("target_beta_annual")
        beta = (bridge.posterior_annual_reliability_index
                if bridge.posterior_annual_reliability_index is not None else bridge.annual_reliability_index)
        margin = (float(beta) - float(target)) if (beta is not None and target is not None) else None
        age = max((c.effective_age_years for c in bridge.components.values()), default=0.0)
        return {
            "observed_condition": None if seen is None else float(seen),
            "observed_beta_margin": margin,
            "quarters_since_inspection": float(self.quarter - int(bridge.last_inspection_quarter or 0)),
            "network_relevance": float(getattr(bridge, "network_relevance", 0.0)),
            "adt": float(bridge.adt),
            "detour_km": float(getattr(bridge, "detour_km", 0.0)),
            "age_years": float(age),
            "area_m2": float(bridge.area),
            "crosses_water": 1.0 if getattr(bridge, "crosses_water", False) else 0.0,
            "scour_critical": 1.0 if getattr(bridge, "scour_critical", False) else 0.0,
            "restricted": 1.0 if float(bridge.traffic_restriction_factor) < 1.0 else 0.0,
        }

    def classify_bridge(self, bridge: Bridge) -> None:
        """Attribuisce classe di attenzione e prossima scadenza. Se la legge in vigore
        porta una propria tabella di classificazione, decide lei; altrimenti vale la
        regola storica a soglie sulla sola condizione osservata."""
        cls = getattr(self.regime, "classification", None) if self.regime else None
        if cls is None:
            bridge.update_cda_and_due_date(self.quarter, self.cda_config)
            return
        if bridge.min_observed_condition() is None:
            bridge.cda = "Sconosciuta"
            return
        values = self._classifier_values(bridge)
        bridge.cda = cls.classify(values)
        bridge.next_inspection_quarter = self.quarter + cls.interval(bridge.cda, 8)

    def _crew_admits(self, agent: str, op: Dict[str, Any], used: Dict[str, Dict[str, float]]) -> bool:
        """Capacita' operativa. Due riserve separate, perche' ispettori e squadre di
        lavoro sono persone diverse: `inspection_capacity` conta le ispezioni avviate
        nel trimestre, `works_capacity` i cantieri contemporaneamente aperti (un
        intervento occupa la squadra per tutta la durata). `crew_capacity`, se
        impostata, le sostituisce con un'unica riserva condivisa.
        """
        t = op.get("type")
        shared = float(getattr(self.params, "crew_capacity", 0.0) or 0.0)
        bucket = used.setdefault(agent, {})
        if shared > 0.0:
            need = self._crew_load(op)
            if need <= 0.0:
                return True
            cur = bucket.get("shared")
            if cur is None:
                cur = bucket["shared"] = self._crew_in_use(agent)
            if cur + need > shared:
                return False
            bucket["shared"] = cur + need
            return True
        if t in ("analysis", "proof_load", "monitoring"):
            cap = float(getattr(self.params, "inspection_capacity", 0.0) or 0.0)
            if cap <= 0.0:
                return True
            cur = bucket.get("insp", 0.0)
            if cur + 1.0 > cap:
                return False
            bucket["insp"] = cur + 1.0
            return True
        if t in ("intervention", "preventive", _progressive.ARREST_TYPE):
            cap = float(getattr(self.params, "works_capacity", 0.0) or 0.0)
            if cap <= 0.0:
                return True
            cur = bucket.get("works")
            if cur is None:
                cur = bucket["works"] = float(sum(
                    1 for b in self.bridges.values() if b.manager == agent and b.active_maintenance))
            if cur + 1.0 > cap:
                return False
            bucket["works"] = cur + 1.0
            return True
        return True

    def _crew_in_use(self, agent: str) -> float:
        """Capacita' gia' impegnata da lavori in corso, che occupano la squadra fino
        alla fine della durata dell'intervento."""
        used = 0.0
        for bridge in self.bridges.values():
            if bridge.manager != agent or not bridge.active_maintenance:
                continue
            op = self.maint_ops.get(bridge.active_maintenance.get("action_name"))                 or self._authority_catalogue.get(bridge.active_maintenance.get("action_name"), {})
            used += self._crew_load(op)
        return used

    def _sample_hydraulic_events(self) -> None:
        """Piena ed erosione al fondo: l'evento localizzato che manca al modello.

        Colpisce solo i ponti su corso d'acqua, con tasso piu' alto per quelli a rischio
        erosione. Due esiti: crollo (raro) oppure danno improvviso alle fondazioni, che
        e' cio' che una cadenza di ispezione breve puo' intercettare e una lunga no.
        Flusso di numeri casuali dedicato, quindi due strategie sullo stesso seme
        affrontano gli stessi eventi.

        hydraulic_legacy_mode=True (default): canale storico, invariato. False: stessa
        esposizione e stessi tassi, ma sorteggio da un flusso sempre a chiave (non sposta
        l'RNG condiviso) e danno trattato come tipo del canale improvviso
        (hazards/sudden_damage.py): bersagli e intensita' configurabili, nascosto fino
        all'ispezione salvo autodenuncia, contato nelle metriche.
        """
        if not getattr(self.params, "hydraulic_hazard_enabled", False):
            return
        ratio = max(0.0, float(getattr(self.params, "failure_consequence_ratio", 3.0) or 0.0))
        exponent = self.params.deterioration_exponent
        mult = float(getattr(self.params, "hydraulic_damage_rate_multiplier", 10.0))
        drop = float(getattr(self.params, "hydraulic_damage_condition_drop", 0.20))
        legacy = bool(getattr(self.params, "hydraulic_legacy_mode", True))
        spec = None if legacy else _sudden.hydraulic_spec(self.params)
        # Quota dei crolli diretti riassegnata agli eventi progressivi (0 = storico): il
        # crollo diretto avviene per u < tasso x (1 - quota); la fascia tolta non produce
        # nulla qui (la sostituisce il percorso latente), il danno resta com'era.
        share = _progressive.effective_share(self.params)
        for bridge in self.bridges.values():
            if bridge.manager not in self.active_agents or not getattr(bridge, "crosses_water", False):
                continue
            rate = (float(self.params.hydraulic_collapse_rate_scour_q)
                    if getattr(bridge, "scour_critical", False)
                    else float(self.params.hydraulic_collapse_rate_water_q))
            if legacy:
                u = float(self._stream(4201, bridge.id, self.quarter).random())
            else:
                ev_rng = self._keyed_stream(_sudden.PURPOSE["hydraulic"], bridge.id, self.quarter)
                u = float(ev_rng.random())
            rate_c = rate * (1.0 - share) if share > 0.0 else rate
            if u < rate_c:
                self._log_collapse(bridge, "hydraulic_direct", quarterly_p=rate_c)
                if getattr(self, "sudden_pending", None):
                    _sudden.on_failure(self, bridge)
                c0 = max(0.0, float(getattr(bridge, "initial_asset_value", 0.0) or 0.0))
                st = self.failure_stats.setdefault(bridge.manager,
                                                   {"failures": 0, "realised_loss": 0.0, "realised_loss_c0": 0.0})
                st["failures"] += 1
                st["realised_loss"] += c0 * (1.0 + ratio)
                st["realised_loss_c0"] += c0
                for comp in bridge.components.values():
                    comp.apply_maintenance(1.0, exponent)
                bridge.active_maintenance = None
                self.classify_bridge(bridge)
                self._record_action_event(bridge, "crollo_idraulico", "failure", {"quarterly_p": rate_c})
                self.last_failure_info = {"agent_name": bridge.manager, "reason": "HYDRAULIC_COLLAPSE",
                                          "bridge_id": bridge.id, "bridge_name": bridge.name,
                                          "message": f"Crollo per piena o erosione del ponte '{bridge.name}'; ricostruito."}
                self.refresh_bridge_reliability(bridge)
            elif u < rate:
                pass    # crollo diretto riassegnato al canale progressivo (share > 0)
            elif u < rate * mult and not legacy:
                _sudden.apply_event(self, bridge, spec, ev_rng)
            elif u < rate * mult:
                # Danno improvviso alle fondazioni: scalzamento, erosione localizzata.
                for name in ("pile", "spalle"):
                    comp = bridge.components.get(name)
                    if comp is not None:
                        self._apply_hidden_damage(comp, drop)
                bridge.hydraulic_damage_quarter = self.quarter
                self._record_action_event(bridge, "danno_idraulico", "hazard", {"condition_drop": drop})
                self.refresh_bridge_reliability(bridge)

    def _progressive_collapse(self, bridge: Bridge, defect: Dict[str, Any], spec: Dict[str, Any]) -> None:
        """Crollo a fine finestra di un difetto progressivo non arrestato: trattato come
        un crollo diretto (perdita realizzata, ricostruzione, evento pubblico)."""
        ratio = max(0.0, float(getattr(self.params, "failure_consequence_ratio", 3.0) or 0.0))
        exponent = self.params.deterioration_exponent
        known_since = defect.get("known_since")
        self._log_collapse(bridge, f"progressive_{spec['name']}", quarterly_p=float(spec["p_collapse"]),
                           extra={"defect_start_quarter": int(defect.get("start_quarter") or 0),
                                  "defect_known_since": None if known_since is None else int(known_since)})
        if getattr(self, "sudden_pending", None):
            _sudden.on_failure(self, bridge)
        c0 = max(0.0, float(getattr(bridge, "initial_asset_value", 0.0) or 0.0))
        st = self.failure_stats.setdefault(bridge.manager,
                                           {"failures": 0, "realised_loss": 0.0, "realised_loss_c0": 0.0})
        st["failures"] += 1
        st["realised_loss"] += c0 * (1.0 + ratio)
        st["realised_loss_c0"] += c0
        for comp in bridge.components.values():
            comp.apply_maintenance(1.0, exponent)
        bridge.active_maintenance = None
        self.classify_bridge(bridge)
        self._record_action_event(bridge, "crollo_progressivo", "failure", {"event_type": spec["name"]})
        self.last_failure_info = {"agent_name": bridge.manager, "reason": "PROGRESSIVE_COLLAPSE",
                                  "bridge_id": bridge.id, "bridge_name": bridge.name,
                                  "message": f"Crollo del ponte '{bridge.name}' per un difetto in sviluppo "
                                             f"non arrestato ({spec['name']}); ricostruito."}
        self.refresh_bridge_reliability(bridge)

    # ------------------------------------------------------------------
    # Registro dei collassi e regola di fine partita
    # ------------------------------------------------------------------
    def _owner_state_snapshot(self) -> Dict[int, Dict[str, Any]]:
        """Che cosa sapeva il gestore di ogni ponte all'inizio del trimestre (prima
        delle azioni): e' lo stato "del trimestre prima" su cui ha deciso. Solo letture.

        Con observation_mode="inspected" beta e condizione stimata vengono dalla stima
        gia' calcolata per l'osservazione (cache); se non e' aggiornata (motore usato
        senza osservazioni) restano None, tranne quando servono alla regola
        collapse_game_over="known_critical" con ethical_floor_beta."""
        p = self.params
        inspected = p.observation_mode == "inspected"
        th = float(getattr(p, "ethical_floor_condition", 0.0))
        beta_th = getattr(p, "ethical_floor_beta", None)
        need_beta = beta_th is not None and getattr(p, "collapse_game_over", "none") == "known_critical"
        floor_on = bool(getattr(p, "ethical_floor_enabled", False))
        defect_floor = bool(getattr(p, "progressive_floor_trigger", True))
        snap: Dict[int, Dict[str, Any]] = {}
        for b in self.bridges.values():
            seen = b.min_observed_condition()
            src: Optional[Bridge] = b
            if inspected:
                hit = self._assessed_cache.get(b.id)
                if hit is not None and hit[0] == self._assessment_key(b):
                    src = hit[1]
                else:
                    src = self._assessed_bridge(b) if need_beta else None
            beta = est_min = None
            if src is not None:
                beta = (src.posterior_annual_reliability_index
                        if src.posterior_annual_reliability_index is not None else src.annual_reliability_index)
                est_min = float(src.min_condition)
            known = _progressive.public_view(self, b)
            below = bool((seen is not None and float(seen) < th)
                         or (beta_th is not None and beta is not None and float(beta) < float(beta_th)))
            snap[b.id] = {
                "observed_min_condition": None if seen is None else float(seen),
                "assessed_min_condition": est_min,
                "assessed_annual_beta": None if beta is None else float(beta),
                "last_inspection_quarter": int(b.last_inspection_quarter or 0),
                "known_defects": known,
                "floor_trigger_active": bool(floor_on and (below or (bool(known) and defect_floor))),
                "known_critical": bool(below or known),
            }
        return snap

    def _log_collapse(self, bridge: Bridge, cause: str, annual_pf: Optional[float] = None,
                      quarterly_p: Optional[float] = None, extra: Optional[Dict[str, Any]] = None,
                      game_over: bool = True) -> Dict[str, Any]:
        """Registra un collasso in env.collapse_log (prima della ricostruzione) e applica
        la regola collapse_game_over. Cause: "pf" (campionato dalla Pf),
        "hydraulic_direct", "progressive_<tipo>", "condition_threshold" (modello storico).

        Stato noto al gestore: quello dell'inizio del trimestre (_owner_state_snapshot).
        Danni nascosti, limitazioni e lavori: quelli al momento del collasso."""
        pre = (self._owner_before or {}).get(bridge.id) or {}
        pending = (getattr(self, "sudden_pending", None) or {}).get(bridge.id) or []
        defects = (getattr(self, "progressive_defects", None) or {}).get(bridge.id) or []
        closed = float(bridge.traffic_restriction_factor) <= 0.0
        active = bridge.active_maintenance or {}
        rec: Dict[str, Any] = {
            "quarter": int(self.quarter), "bridge_id": int(bridge.id), "bridge_name": bridge.name,
            "agent": bridge.manager, "cause": cause,
            "annual_pf": None if annual_pf is None else float(annual_pf),
            "quarterly_p": None if quarterly_p is None else float(quarterly_p),
            "observed_min_condition": pre.get("observed_min_condition"),
            "assessed_min_condition": pre.get("assessed_min_condition"),
            "assessed_annual_beta": pre.get("assessed_annual_beta"),
            "last_inspection_quarter": pre.get("last_inspection_quarter"),
            "known_defects": copy.deepcopy(pre.get("known_defects") or []),
            "floor_trigger_active": bool(pre.get("floor_trigger_active", False)),
            "known_critical": bool(pre.get("known_critical", False)),
            "hidden_damage_pending": len(pending),
            "hidden_defects_pending": sum(1 for d in defects if d.get("known_since") is None),
            "true_min_condition": float(bridge.min_condition),
            "traffic_restriction_factor": float(bridge.traffic_restriction_factor),
            "closed": closed,
            "floor_closure": bool(getattr(bridge, "floor_closure", None)),
            "regime_sanction": (bridge.regime_sanction or {}).get("action") if bridge.regime_sanction else None,
            "posting_active": int(getattr(bridge, "posting_until_quarter", 0) or 0) > self.quarter,
            "works_active": active.get("action_name") if isinstance(active, dict) else None,
        }
        if extra:
            rec.update(extra)
        self.collapse_log.append(rec)
        rule = getattr(self.params, "collapse_game_over", "none")
        if game_over and (rule == "any" or (
                rule == "known_critical" and rec["known_critical"]
                and not (closed and bool(getattr(self.params, "collapse_game_over_exempt_closed", True))))):
            self._collapse_game_over.setdefault(bridge.manager, rec)
        # l'opera e' ricostruita: i difetti progressivi in sviluppo sono cancellati
        if getattr(self, "progressive_defects", None):
            _progressive.on_failure(self, bridge)
        return rec

    def _sample_posting_events(self) -> None:
        """Declassamento di portata che nasce dalla verifica di capacita', non
        dall'inadempienza. Senza questo canale nessun regime finanziato mostra mai un
        ponte limitato, mentre l'inventario reale ne ha il 2.5% in Delaware e il 10%
        negli Stati Uniti."""
        rate = float(getattr(self.params, "posting_entry_rate_q", 0.0) or 0.0)
        if rate <= 0.0:
            return
        dur = max(1, int(getattr(self.params, "posting_mean_duration_quarters", 20)))
        factor = float(getattr(self.params, "posting_restriction_factor", 0.6))
        sev_floor = float(getattr(self.params, "posting_severity_floor", 0.25))
        lift = bool(getattr(self.params, "posting_lift_on_intervention", False))
        for bridge in self.bridges.values():
            if bridge.manager not in self.active_agents:
                continue
            until = int(getattr(bridge, "posting_until_quarter", 0) or 0)
            if lift and bridge.active_maintenance and (self.maint_ops.get(
                    bridge.active_maintenance.get("action_name")) or {}).get("type") == "intervention":
                # lavori strutturali: il declassamento e' revocato e non ne nasce uno nuovo
                if until:
                    bridge.posting_until_quarter = 0
                    if (not bridge.regime_sanction and not getattr(bridge, "floor_closure", None)
                            and not getattr(bridge, "temporary_closure", None)):
                        bridge.traffic_restriction_factor = 1.0
                    self._record_action_event(bridge, "declassamento_portata", "lifted_by_works")
                continue
            if until:
                if self.quarter >= until:
                    bridge.posting_until_quarter = 0
                    if (not bridge.regime_sanction and not getattr(bridge, "floor_closure", None)
                            and not getattr(bridge, "temporary_closure", None)):
                        bridge.traffic_restriction_factor = 1.0
                continue
            # piu' probabile su un ponte in cattivo stato: il declassamento segue la capacita'
            severity = max(sev_floor, 2.0 * (1.0 - float(bridge.min_condition)))
            u = float(self._stream(4202, bridge.id, self.quarter).random())
            if u < rate * severity:
                length = int(self._stream(4203, bridge.id, self.quarter).geometric(1.0 / dur))
                bridge.posting_until_quarter = self.quarter + max(1, length)
                bridge.traffic_restriction_factor = min(float(bridge.traffic_restriction_factor), factor)
                self._record_action_event(bridge, "declassamento_portata", "hazard",
                                          {"until_quarter": bridge.posting_until_quarter})

    def _sample_temporary_closures(self) -> None:
        """Chiusura temporanea (opt-in): una parte indipendente dallo stato dell'opera e, con
        temporary_closure_condition_rate_q, una parte che cresce sui ponti peggiori. Finche' dura
        il fattore di traffico e' zero; alla scadenza torna quello del gestore, con le
        limitazioni ancora in vigore (sanzione del regime, declassamento, soglia etica)."""
        rate = float(getattr(self.params, "temporary_closure_rate_q", 0.0) or 0.0)
        rate_c = float(getattr(self.params, "temporary_closure_condition_rate_q", 0.0) or 0.0)
        if rate <= 0.0 and rate_c <= 0.0:
            return
        dur = max(1, int(getattr(self.params, "temporary_closure_mean_duration_quarters", 4)))
        post_factor = float(getattr(self.params, "posting_restriction_factor", 0.6))
        for bridge in self.bridges.values():
            if bridge.manager not in self.active_agents:
                continue
            posting = int(getattr(bridge, "posting_until_quarter", 0) or 0) > self.quarter
            tc = getattr(bridge, "temporary_closure", None)
            if tc:
                if self.quarter < int(tc["until_quarter"]):
                    bridge.traffic_restriction_factor = 0.0
                    continue
                own = float(tc.get("restored_factor", 1.0))
                since = int(tc.get("since_quarter", 0))
                factor = own
                if bridge.regime_sanction:
                    # sanzione nata durante la chiusura: aveva memorizzato lo zero
                    if int(bridge.regime_sanction.get("since_quarter", 0)) >= since:
                        bridge.regime_sanction["restored_factor"] = own
                    s_op = (self.maint_ops.get(bridge.regime_sanction.get("action"))
                            or self._authority_catalogue.get(bridge.regime_sanction.get("action")) or {})
                    factor = min(factor, float(s_op.get("traffic_restriction_factor", 1.0)))
                if posting:
                    factor = min(factor, post_factor)
                fc = getattr(bridge, "floor_closure", None)
                if fc:
                    if int(fc.get("since_quarter", 0)) >= since:
                        fc["restored_factor"] = factor
                else:
                    bridge.traffic_restriction_factor = factor
                bridge.temporary_closure = None
                self._record_action_event(bridge, "chiusura_temporanea", "ended")
                continue
            u = float(self._stream(4204, bridge.id, self.quarter).random())
            p_in = rate + (rate_c * 2.0 * (1.0 - float(bridge.min_condition)) if rate_c > 0.0 else 0.0)
            if u >= p_in or getattr(bridge, "floor_closure", None):
                continue
            length = int(self._stream(4205, bridge.id, self.quarter).geometric(1.0 / dur))
            if bridge.regime_sanction:
                own = float(bridge.regime_sanction.get("restored_factor", 1.0))
            elif posting:
                own = 1.0
            else:
                own = float(bridge.traffic_restriction_factor)
            bridge.temporary_closure = {"since_quarter": self.quarter,
                                        "until_quarter": self.quarter + max(1, length),
                                        "restored_factor": own}
            bridge.traffic_restriction_factor = 0.0
            self._record_action_event(bridge, "chiusura_temporanea", "hazard",
                                      {"until_quarter": bridge.temporary_closure["until_quarter"]})

    def _stream(self, purpose: int, *keys) -> np.random.Generator:
        """Generatore per una fonte di casualita'. "shared": l'RNG unico storico.
        "keyed": flusso dedicato a (seme, fonte, chiavi) indipendente da cio' che e'
        stato estratto prima (numeri casuali comuni fra strategie e leggi)."""
        if getattr(self.params, "rng_streams", "shared") != "keyed":
            return self.rng
        return self._keyed_stream(purpose, *keys)

    def _keyed_stream(self, purpose: int, *keys) -> np.random.Generator:
        """Flusso a chiave (seme, fonte, chiavi), indipendente dalla modalita' rng_streams."""
        base = int(self.params.seed) if self.params.seed is not None else 0
        key = [base & 0x7FFFFFFF, 1000 + int(purpose)]
        # Copie dei pianificatori in modalita' "belief": sale proprio, cosi' il futuro
        # casuale simulato non coincide con quello reale. Sale 0 (ambiente reale e
        # default) = chiave storica.
        salt = int(getattr(self, "stream_salt", 0) or 0)
        if salt:
            key += [0x5A17, salt & 0x7FFFFFFF]
        return np.random.default_rng(key + [int(k) & 0x7FFFFFFF for k in keys])

    @staticmethod
    def _comp_key(name: str) -> int:
        return sum((i + 1) * ord(ch) for i, ch in enumerate(name))

    def planning_clone(self, salt: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None) -> "BridgeEnv":
        """Copia per simulazioni interne dei pianificatori (GA, memetico, MCTS).

        planner_information="truth" (storico): stesso stato VERO, stesse regole (regime,
        modalita' di osservazione, modello di collasso), ma flusso dei collassi diverso
        da quello reale: il pianificatore conosce il modello, non i sorteggi futuri. Il
        sale deriva dal random globale (seedato per run), quindi la run resta
        riproducibile. `rng` e' ignorato. Attenzione: con rng_streams="keyed" la copia
        ha lo stesso seme del motore e rivive gli stessi flussi di degrado, ispezione,
        esiti degli interventi, eventi idraulici e declassamenti del futuro reale.
        planner_information="belief": vedi belief_clone.
        """
        if getattr(self.params, "planner_information", "truth") == "belief":
            return self.belief_clone(salt=salt, rng=rng)
        import random as _random
        clone = BridgeEnv.from_dict(self.to_dict())
        clone.failure_stream_salt = int(salt) if salt is not None else _random.getrandbits(31) + 1
        return clone

    # Campi del componente che il gestore conosce: inventario (nome, profilo del tipo,
    # vita utile, eta' iniziale), evidenza (intervalli osservati, stato di conoscenza),
    # esiti delle prove di carico. Lo stato nascosto qui elencato viene riestratto nella
    # copia "belief"; beta, Pf, dettagli e proiezioni sono ricalcolati dal motore sullo
    # stato estratto.
    BELIEF_HIDDEN_COMPONENT_FIELDS = (
        "current_condition", "accumulated_damage", "effective_age_years",
        *Component.LATENT_PARAMETERS, "last_process_multiplier",
    )

    def _draw_belief_component(self, comp: Component, rng: np.random.Generator, elapsed_years: float) -> None:
        """Sostituisce lo stato nascosto del componente con un'estrazione dalla
        credenza del gestore. Legge solo campi noti (profilo, eta' iniziale, vita utile,
        intervalli osservati, incertezza epistemica), mai quelli nascosti.

        - parametri latenti: estrazione dalla priori del profilo (stessa legge del
          costruttore), per tutti i componenti;
        - componente osservato: condizione ~ N(centro, ((max-min)/3.92)^2) troncata a
          [1e-6, 1] (l'intervallo osservato letto come intervallo al 95%, centro come in
          _assessed_bridge), eta' efficace invertita dalla condizione con i parametri
          estratti; intervalli osservati e incertezza epistemica invariati;
        - componente mai osservato: eta' nominale (eta' iniziale + tempo trascorso, come in
          _assessed_bridge), condizione dalla curva con i parametri estratti.
        """
        exponent = self.params.deterioration_exponent
        profile = comp.profile or COMPONENT_PROFILES.get(comp.name, COMPONENT_PROFILES["impalcato"])
        for key, value in Component.draw_latent_parameters(profile, rng).items():
            setattr(comp, key, value)
        comp.last_process_multiplier = 1.0
        if self._strict_information():
            comp.vita_utile_anni = self._nominal_life(comp)       # non la vita campionata
        lo, hi = comp.observed_condition_min, comp.observed_condition_max
        if lo is not None and hi is not None:
            lo, hi = float(lo), float(hi)
            centre = 0.5 * (lo + hi)
            sd = max(0.0, hi - lo) / 3.92
            draw = float(rng.normal(centre, sd)) if sd > 0.0 else centre
            eps = comp.epistemic_uncertainty
            comp.current_condition = max(1e-6, min(1.0, draw))
            comp.apply_maintenance(0.0, exponent)       # inverte l'eta' efficace dalla condizione
            comp.observed_condition_min, comp.observed_condition_max = lo, hi
            comp.epistemic_uncertainty = eps
        else:
            comp.effective_age_years = float(comp.initial_age_years) + elapsed_years
            comp.update_condition(exponent)

    def belief_clone(self, salt: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None) -> "BridgeEnv":
        """Copia del pianificatore costruita sulla CONOSCENZA del gestore
        (planner_information="belief").

        Stato corrente: con observation_mode="inspected" e senza oracolo, per ogni
        componente di ogni ponte lo stato nascosto (BELIEF_HIDDEN_COMPONENT_FIELDS) e'
        riestratto dalla credenza del gestore (_draw_belief_component); a livello di ponte
        l'ultimo danno idraulico (hydraulic_damage_quarter, evento "danno_idraulico"
        nello storico) e' nascosto, e beta, Pf, fragilita', indice di rischio, proiezioni
        e curve di monitoraggio sono ricalcolati sullo stato estratto; la perdita attesa
        dell'ultimo trimestre (calcolata sul vero) e' azzerata. Con oracle_observation=True
        (oracolo = pianificatore sul vero) o observation_mode="full" (gli agenti vedono gia'
        il vero) lo stato corrente resta quello vero.
        Futuro casuale: sempre indipendente da quello reale (sale proprio dei flussi a
        chiave, generatore condiviso nuovo, sale dei collassi), anche per l'oracolo.

        Tutte le estrazioni vengono da `rng` (il generatore del pianificatore; default:
        derivato dal random globale, seedato per run). Nessun flusso del motore reale
        viene consumato. Ogni chiamata e' una nuova estrazione.
        """
        import random as _random
        if rng is None:
            rng = np.random.default_rng(_random.getrandbits(63))
        hide = (self.params.observation_mode == "inspected"
                and not bool(getattr(self.params, "oracle_observation", False)))
        data = self.to_dict()
        if hide:
            data["last_expected_loss"] = {}
        clone = BridgeEnv.from_dict(data, refresh=False)
        clone.failure_stream_salt = int(salt) if salt is not None else int(rng.integers(1, 2 ** 31))
        clone.stream_salt = int(rng.integers(1, 2 ** 31))
        clone.rng = np.random.default_rng(int(rng.integers(0, 2 ** 63)))
        clone.planner_view = "belief" if hide else "truth"
        if hide:
            elapsed_years = (clone.quarter - 1) / 4.0
            for bid in sorted(clone.bridges):
                bridge = clone.bridges[bid]
                for name in sorted(bridge.components):
                    clone._draw_belief_component(bridge.components[name], rng, elapsed_years)
                bridge.hydraulic_damage_quarter = 0
                bridge.action_history = [e for e in bridge.action_history if e.get("action") != "danno_idraulico"]
            # danni improvvisi non ancora rilevati: il gestore non li conosce; nemmeno i
            # contatori veri del canale (eventi avvenuti, perdita totale...): la copia
            # riparte da zero come un motore nuovo (RT009)
            clone.sudden_pending = {}
            clone.sudden_stats = ({a: _sudden.new_stats(clone.params) for a in clone.params.agent_names}
                                  if _sudden.channel_active(clone.params) else {})
            # eventi regionali: l'elenco degli eventi e' pubblico e resta; i contatori
            # veri (ponti colpiti, perdita) no
            clone.regional_stats = ({a: _regional.new_stats() for a in clone.params.agent_names}
                                    if _regional.enabled(clone.params) else {})
            # eventi progressivi: restano i soli difetti NOTI, con data di fine riestratta
            # (il gestore non la conosce); via difetti nascosti, contatori veri e
            # registro dei collassi (cause e danni nascosti sono verita')
            _progressive.reset_belief(clone, rng)
        clone.refresh_network_reliability()
        if hide:
            _monitoring_op = clone.maint_ops.get('monitoring', {})
            for bridge in clone.bridges.values():
                if bridge.future_curve_range is not None:
                    bridge.future_curve_range = clone.predict_future_degradation(
                        bridge, _monitoring_op.get('error_sigma', 0.005))
        return clone

    def _full_catalogue(self) -> Dict[str, Any]:
        from dataclasses import asdict as _asdict
        from infralab.settings.config import MaintOpsConfig
        return _asdict(MaintOpsConfig())

    def regime_summary(self) -> Dict[str, Any]:
        return {"id": self.regime.id, "display_name": self.regime.display_name,
                "hypothetical": self.regime.hypothetical, "enforcement": self.regime.enforcement.mode,
                "target_tier": self.regime.target_tier,
                "obligations": [{"trigger": o.trigger, "action": o.action, "grace_quarters": o.grace_quarters}
                                for o in self.regime.obligations],
                "earmarked_fraction": self.regime.earmarked_fraction}

    def get_agent_observations(self, agent_name: str):
        obs = {"game_rules": {"quarter": self.quarter, "my_budget": self.budgets.get(agent_name, 0)}, "my_bridges": [],
               "maintenance_options": self.maint_ops, "cda_config": asdict(self.cda_config),
               "game_settings": asdict(self.params),
               "management_constraints": {
                   "single_action_if_bridge_busy": True,
                   "no_same_action_consecutive_quarters_except_L1": True,
                   "monitoring_overlap_forbidden": True,
                   "budget_feasibility_checked_before_execution": True,
                   "objective_mode": self.params.management_objective_mode,
               },
               "last_plan_diagnostics": self.last_plan_diagnostics.get(agent_name, []),
               "regime": self.regime_summary(),
               "regime_stats": dict(self.regime_stats.get(agent_name, {}))}
        grace = self.params.inspection_overdue_grace_quarters
        esc = self.params.l4_escalation_quarters
        regional_on = _regional.enabled(self.params)
        if regional_on:
            obs["regional_events"] = _regional.public_events(self)
        progressive_on = _progressive.enabled(self.params)
        for b in self.bridges.values():
            if b.manager == agent_name:
                # Le viste oracolo/stimata rifanno i componenti (e la stimata anche i
                # dettagli di affidabilita'): non li si copia due volte.
                oracle = bool(getattr(self.params, "oracle_observation", False))
                assessed = not oracle and self.params.observation_mode == "inspected"
                b_dict = b.to_dict(with_components=not (oracle or assessed), with_details=not assessed)
                b_dict["action_feasibility"] = self.get_feasible_actions_for_bridge(agent_name, b)
                b_dict["health_index"] = self.get_health_index(b)
                b_dict["net_asset_value"] = self.get_net_asset_value(b)
                # Segnali di conformità Linee Guida per l'agente.
                b_dict["inspection_overdue"] = self.quarter > b.next_inspection_quarter + grace
                b_dict["overdue_L4_assessment"] = bool(
                    esc > 0 and b.cda == "Alta" and b.quarters_in_cda_alta_without_L4 > esc)
                b_dict["not_censused"] = not getattr(b, "censused", True)
                due = self._regime_due_obligations(b) if self.regime.obligations else []
                b_dict["regime_mandatory_action"] = due[0].action if due else None
                b_dict["regime_mandatory_trigger"] = due[0].trigger if due else None
                b_dict["regime_sanction"] = copy.deepcopy(b.regime_sanction)
                if regional_on:
                    # parte pubblica degli eventi regionali (area, intensita' locale)
                    b_dict.update(_regional.observation_fields(self, b))
                if progressive_on:
                    # difetti in sviluppo NOTI: tipo e trimestre di scoperta, mai la fine
                    b_dict["known_progressive_defects"] = _progressive.public_view(self, b)
                if oracle:
                    b_dict = self._oracle_view(b, b_dict)
                elif assessed:
                    b_dict = self._assessed_view(b, b_dict)
                obs["my_bridges"].append(b_dict)
        return obs

    def apply_manual_earthquake(self, magnitude: float, epicenter_lat: float, epicenter_lon: float) -> Dict:
        """Applica un evento sismico manuale (da UI) e restituisce il report."""
        from infralab.core.hazards.earthquake import EarthquakeEvent, apply_earthquake_damage
        event = EarthquakeEvent(
            quarter=self.quarter,
            magnitude=magnitude,
            epicenter_lat=epicenter_lat,
            epicenter_lon=epicenter_lon,
            name=f"M{magnitude:.1f} Manuale - T{self.quarter}",
        )
        reports = apply_earthquake_damage(self.bridges, event, self.params.deterioration_exponent, rng=self.rng,
                                          raise_uncertainty=not self._strict_information())
        event.damage_reports = reports
        self.earthquake_events.append(event.to_dict())
        self.refresh_network_reliability()
        return event.to_dict()

    def predict_future_degradation(self, bridge: Bridge, error_sigma: float) -> Dict[str, list]:
        future_data_range = {name: [] for name in bridge.components.keys()}

        for comp_name, initial_comp in bridge.components.items():
            for i in range(self.params.monitoring_duration_years * 4 + 1):
                future_quarter = self.quarter + i

                temp_comp = Component.from_dict(initial_comp.to_dict(), self.params.deterioration_exponent)

                _age_before = temp_comp.effective_age_years
                for _ in range(i):
                    env_factor = self.params.env_degradation_mean
                    age_increase_factor = (
                        self._traffic_multiplier(bridge) ** temp_comp.traffic_sensitivity
                    ) * (
                        env_factor ** temp_comp.environment_sensitivity
                    )
                    if self.spec.degradation_rate_adapter is not None:
                        age_increase_factor *= max(0.0, float(self.spec.degradation_rate_adapter(self, bridge, temp_comp)))
                    temp_comp.effective_age_years += 0.25 * age_increase_factor

                self._advance_projection_condition(temp_comp, _age_before)

                half_width = self._component_observation_sigma(temp_comp, error_sigma) * 1.96
                predicted_min = np.clip(temp_comp.current_condition - half_width, 0, 1)
                predicted_max = np.clip(temp_comp.current_condition + half_width, 0, 1)
                future_data_range[comp_name].append((future_quarter, predicted_min, predicted_max))

        return future_data_range

    def to_dict(self):
        return {"schema_version": 1,
                "asset_type": self.asset_type,
                "params": asdict(self.params), "maint_ops": self.maint_ops, "agent_config": self.agent_config,
                "cda_config": asdict(self.cda_config), "component_weights": self.component_weights,
                "active_agents": self.active_agents, "quarter": self.quarter, "budgets": self.budgets,
                "emissions_tco2": self.emissions_tco2,
                "negative_budget_streak": self.negative_budget_streak, "last_plan_diagnostics": self.last_plan_diagnostics,
                "last_failure_info": self.last_failure_info,
                "earthquake_events": self.earthquake_events,
                "regime_stats": self.regime_stats,
                "failure_stats": self.failure_stats,
                "floor_stats": self.floor_stats,
                "crew_stats": dict(self.crew_stats),
                "last_expected_loss": self.last_expected_loss,
                "failure_stream_salt": self.failure_stream_salt,
                # Contatori delle ispezioni del trimestre corrente (chiave del flusso
                # casuale con rng_streams="keyed"); quelli dei trimestri passati non
                # vengono piu' letti. Lista di [ponte, componente, trimestre, n] (JSON).
                "inspection_counters": [[int(k[0]), int(k[1]), int(k[2]), int(n)]
                                        for k, n in (getattr(self, "_insp_counter", None) or {}).items()
                                        if int(k[2]) >= self.quarter],
                "bridges": {bid: b.to_dict() for bid, b in self.bridges.items()},
                # sale dei flussi a chiave: solo nelle copie "belief" (chiave assente = 0)
                **({"stream_salt": int(self.stream_salt)} if getattr(self, "stream_salt", 0) else {}),
                # danno improvviso: chiavi presenti solo con il canale acceso
                **({"sudden_stats": copy.deepcopy(self.sudden_stats)} if getattr(self, "sudden_stats", None) else {}),
                **({"sudden_pending": [[int(bid), copy.deepcopy(evs)] for bid, evs in self.sudden_pending.items()]}
                   if getattr(self, "sudden_pending", None) else {}),
                # eventi regionali: chiavi presenti solo con il canale acceso
                **({"regional_events": copy.deepcopy(self.regional_events)}
                   if getattr(self, "regional_events", None) else {}),
                **({"regional_stats": copy.deepcopy(self.regional_stats)}
                   if getattr(self, "regional_stats", None) else {}),
                # eventi progressivi e registro dei collassi: chiavi presenti solo se non vuoti
                **({"progressive_defects": [[int(bid), copy.deepcopy(defs)]
                                            for bid, defs in self.progressive_defects.items()]}
                   if getattr(self, "progressive_defects", None) else {}),
                **({"progressive_stats": copy.deepcopy(self.progressive_stats)}
                   if getattr(self, "progressive_stats", None) else {}),
                **({"collapse_log": copy.deepcopy(self.collapse_log)}
                   if getattr(self, "collapse_log", None) else {})}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], refresh: bool = True) -> 'BridgeEnv':
        params_obj = SimParameters(**copy.deepcopy(data.get('params', {})))
        cda_conf = CdaConfig(**copy.deepcopy(data.get('cda_config', {})))
        comp_weights = copy.deepcopy(data.get('component_weights', DEFAULT_COMPONENTS))
        maint_ops = copy.deepcopy(data.get('maint_ops', {}))
        agent_config = copy.deepcopy(data.get('agent_config', {}))

        env = cls(params_obj, maint_ops, agent_config, cda_conf, comp_weights,
                  asset_type=data.get('asset_type', 'bridge'))

        env.active_agents = copy.deepcopy(data.get('active_agents', []))
        env.quarter = data.get('quarter', 1)
        env.budgets = copy.deepcopy(data.get('budgets', {}))
        env.emissions_tco2 = copy.deepcopy(data.get('emissions_tco2', {a: 0.0 for a in env.active_agents}))
        env.negative_budget_streak = copy.deepcopy(data.get('negative_budget_streak', {}))
        env.last_plan_diagnostics = copy.deepcopy(data.get('last_plan_diagnostics', {a: [] for a in env.params.agent_names}))
        env.last_failure_info = copy.deepcopy(data.get('last_failure_info'))
        env.earthquake_events = copy.deepcopy(data.get('earthquake_events', []))
        env.regime_stats = {a: {**_new_regime_stats(), **v} for a, v in (data.get('regime_stats') or {}).items()} \
            or {a: _new_regime_stats() for a in env.params.agent_names}
        env.failure_stats = copy.deepcopy(data.get('failure_stats')) or {
            a: {"failures": 0, "realised_loss": 0.0, "realised_loss_c0": 0.0} for a in env.params.agent_names}
        env.floor_stats = {a: {**_new_floor_stats(), **v} for a, v in (data.get('floor_stats') or {}).items()} \
            or {a: _new_floor_stats() for a in env.params.agent_names}
        env.crew_stats = {str(a): int(n) for a, n in (data.get('crew_stats') or {}).items()}
        env.last_expected_loss = copy.deepcopy(data.get('last_expected_loss') or {})
        env.failure_stream_salt = int(data.get('failure_stream_salt', 0) or 0)
        counters = data.get('inspection_counters') or []
        if counters:
            env._insp_counter = {(int(b), int(c), int(q)): int(n) for b, c, q, n in counters}

        env.bridges = {int(bid): Bridge.from_dict(b_data, env.params.deterioration_exponent) for bid, b_data in
                       data.get("bridges", {}).items()}
        env.stream_salt = int(data.get('stream_salt', 0) or 0)
        if data.get('sudden_stats'):
            env.sudden_stats = copy.deepcopy(data['sudden_stats'])
        env.sudden_pending = {int(bid): copy.deepcopy(evs) for bid, evs in (data.get('sudden_pending') or [])}
        env.regional_events = copy.deepcopy(data.get('regional_events') or [])
        if data.get('regional_stats'):
            env.regional_stats = copy.deepcopy(data['regional_stats'])
        env.progressive_defects = _progressive.load_defects(data.get('progressive_defects'))
        if data.get('progressive_stats'):
            env.progressive_stats = copy.deepcopy(data['progressive_stats'])
        env.collapse_log = copy.deepcopy(data.get('collapse_log') or [])
        # refresh=False: il chiamante modifica lo stato e ricalcola lui (belief_clone).
        if refresh:
            env.refresh_network_reliability()
        return env


# Alias in avanti: il motore e' asset-agnostico, il nome storico resta valido.
AssetEnv = BridgeEnv
