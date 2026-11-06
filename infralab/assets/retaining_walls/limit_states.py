"""Adapter stati limite per i muri di sostegno.

Stati limite geotecnici classici come coppie R-S lognormali per il motore
FOSM (EC7-like, semplificati):
- paramento:  scorrimento (ULS duttile), ribaltamento (ULS fragile),
              fessurazione (SLS)
- fondazione: capacita' portante (ULS duttile), scorrimento base (ULS duttile)
- drenaggio:  intasamento (SLS funzionale)

La domanda e' dominata dalla spinta delle terre (quota "dead") con una piccola
quota variabile da sovraccarico stradale ("live"). Il drenaggio degradato
aumenta la spinta idraulica: moltiplicatore di domanda per TUTTI i componenti.
"""
from typing import Any, Dict

# Aumento massimo di spinta con drenaggio completamente intasato (+35%).
WATER_PRESSURE_FACTOR_MAX = 0.35
# Quota statica/variabile della domanda (spinta terre vs sovraccarico).
DEAD_SHARE = 0.80
LIVE_SHARE = 0.20
# Base di resistenza (analogo del 2.24 dei ponti: scala relativa, non assoluta).
RESISTANCE_BASE = 2.10


def wall_component_limit_states(env: Any, wall: Any, component: Any) -> Dict[str, Dict[str, float]]:
    drainage = wall.components.get("drenaggio")
    drainage_condition = drainage.current_condition if drainage is not None else 1.0
    water_factor = 1.0 + WATER_PRESSURE_FACTOR_MAX * max(0.0, 1.0 - drainage_condition)

    demand_base = component.demand_bias * 0.62 * water_factor
    dead_mean = DEAD_SHARE * demand_base
    live_mean = LIVE_SHARE * demand_base
    dead_cov = min(0.10, max(0.06, env.params.demand_cov * 0.60))
    live_cov = min(0.30, max(env.params.demand_cov, 0.15))
    code_live_effect = live_mean * 1.50

    condition = max(component.current_condition, 0.02)
    base_resistance = component.resistance_bias * condition * (
        RESISTANCE_BASE + 0.20 * max(0.0, 1.0 - component.epistemic_uncertainty)
    )
    material_cov = min(0.24, 0.05 + 0.40 * env.params.resistance_cov + 0.05 * (1.0 - condition))
    geometry_cov = min(0.10, env.params.reliability_geometry_cov * (1.0 + 0.10 * component.profile["shape_factor"]))
    model_resistance_cov = min(
        0.18,
        env.params.reliability_model_uncertainty_cov * (1.0 + 0.80 * component.epistemic_uncertainty),
    )
    model_demand_cov = min(0.16, 0.05 + 0.30 * env.params.demand_cov + 0.05 * component.epistemic_uncertainty)

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

    if component.name == "fondazione":
        return {
            "bearing_capacity": {
                **common,
                "category": "ULS",
                "failure_mode_type": "ductile",
                "resistance_mean": base_resistance * 1.10,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
            "base_sliding": {
                **common,
                "category": "ULS",
                "failure_mode_type": "ductile",
                "resistance_mean": base_resistance * 0.98,
                "dead_load_mean": dead_mean * 0.92,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
        }
    if component.name == "drenaggio":
        return {
            "clogging": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.80,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.80,
                "code_live_effect": code_live_effect * 0.85,
            },
        }
    # paramento (e fallback)
    return {
        "sliding": {
            **common,
            "category": "ULS",
            "failure_mode_type": "ductile",
            "resistance_mean": base_resistance * 1.02,
            "dead_load_mean": dead_mean,
            "live_load_mean": live_mean,
            "code_live_effect": code_live_effect,
        },
        "overturning": {
            **common,
            "category": "ULS",
            "failure_mode_type": "brittle",   # perdita di equilibrio: improvvisa
            "resistance_mean": base_resistance * 1.12,
            "dead_load_mean": dead_mean * 0.95,
            "live_load_mean": live_mean * 1.10,
            "code_live_effect": code_live_effect * 1.10,
        },
        "cracking": {
            **common,
            "category": "SLS",
            "failure_mode_type": "serviceability",
            "resistance_mean": base_resistance * component.crack_bias * 0.78,
            "dead_load_mean": dead_mean * 0.95,
            "live_load_mean": live_mean * 0.90,
            "code_live_effect": code_live_effect * 0.90,
        },
    }
