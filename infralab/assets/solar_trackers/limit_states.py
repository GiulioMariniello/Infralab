"""Adapter stati limite per gli inseguitori fotovoltaici.

Stati limite:
- struttura_torsione:   vento estremo/galloping (ULS FRAGILE), fatica del
                        tubo di torsione (ULS)
- fondazioni_micropali: sfilamento micropalo (ULS fragile)
- attuatori:            guasto (SLS funzionale)
- moduli:               degrado di potenza (SLS, revenue)

ACCOPPIAMENTO STOW (il piu' forte del laboratorio): lo stow orizzontale
riduce il momento alla base del 60-90% in evento di vento estremo; un
attuatore degradato non porta il tracker in stow → domanda vento
1 + 2.5·(1−cond_attuatori), CAP 3.5x, su struttura e fondazioni. E' il
motivo per cui gli standard tracker impongono lo stow automatico come
mitigazione primaria.
"""
from typing import Any, Dict

# Mancato stow: fino a +250% di domanda vento (cap totale 3.5x).
STOW_DEMAND_FACTOR_MAX = 2.5
DEAD_SHARE = 0.20
LIVE_SHARE = 0.80          # domanda quasi tutta da vento
RESISTANCE_BASE = 2.60     # progetto governato dal vento con stow: riserva alta


def tracker_component_limit_states(env: Any, tracker: Any, component: Any) -> Dict[str, Dict[str, float]]:
    actuators = tracker.components.get("attuatori")
    actuator_condition = actuators.current_condition if actuators is not None else 1.0
    stow_factor = 1.0 + STOW_DEMAND_FACTOR_MAX * max(0.0, 1.0 - actuator_condition)

    demand_base = component.demand_bias * 0.55
    if component.name in ("struttura_torsione", "fondazioni_micropali"):
        demand_base *= stow_factor
    dead_mean = DEAD_SHARE * demand_base
    live_mean = LIVE_SHARE * demand_base
    dead_cov = min(0.10, max(0.05, env.params.demand_cov * 0.50))
    live_cov = min(0.38, max(env.params.demand_cov * 1.3, 0.22))   # vento estremo
    code_live_effect = live_mean * 1.35

    condition = max(component.current_condition, 0.02)
    base_resistance = component.resistance_bias * condition * (
        RESISTANCE_BASE + 0.20 * max(0.0, 1.0 - component.epistemic_uncertainty)
    )
    material_cov = min(0.20, 0.05 + 0.30 * env.params.resistance_cov + 0.05 * (1.0 - condition))
    geometry_cov = min(0.08, env.params.reliability_geometry_cov)
    model_resistance_cov = min(
        0.18,
        env.params.reliability_model_uncertainty_cov * (1.0 + 0.80 * component.epistemic_uncertainty),
    )
    model_demand_cov = min(0.18, 0.06 + 0.30 * env.params.demand_cov + 0.05 * component.epistemic_uncertainty)

    common = {
        "material_cov": material_cov,
        "geometry_cov": geometry_cov,
        "model_resistance_cov": model_resistance_cov,
        "dead_load_cov": dead_cov,
        "live_load_cov": live_cov,
        "model_demand_cov": model_demand_cov,
        "proof_load_cov": 0.05,
        "live_load_distribution": "lognormal",
    }

    if component.name == "fondazioni_micropali":
        return {
            "micropile_pullout": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",
                "resistance_mean": base_resistance * 1.05,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
        }
    if component.name == "attuatori":
        return {
            "actuator_failure": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.72,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.85,
                "code_live_effect": code_live_effect * 0.85,
            },
        }
    if component.name == "moduli":
        return {
            "power_degradation": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.80,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.80,
                "code_live_effect": code_live_effect * 0.80,
            },
        }
    # struttura_torsione (e fallback)
    return {
        "extreme_wind_galloping": {
            **common,
            "category": "ULS",
            "failure_mode_type": "brittle",   # instabilita' aeroelastica: improvvisa
            "resistance_mean": base_resistance * 1.02,
            "dead_load_mean": dead_mean,
            "live_load_mean": live_mean,
            "code_live_effect": code_live_effect,
        },
        "torque_tube_fatigue": {
            **common,
            "category": "ULS",
            "failure_mode_type": "ductile",
            "resistance_mean": base_resistance * 1.12,
            "dead_load_mean": dead_mean,
            "live_load_mean": live_mean * 0.95,
            "code_live_effect": code_live_effect * 0.95,
        },
    }


def tracker_revenue_multiplier(env: Any, tracker: Any) -> float:
    """La produzione segue la condizione dei moduli (soiling/degrado potenza)."""
    modules = tracker.components.get("moduli")
    if modules is None:
        return 1.0
    return max(0.2, modules.current_condition)
