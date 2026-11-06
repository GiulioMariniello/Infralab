"""Adapter stati limite per gli aerogeneratori.

Coppie R-S lognormali per il motore FOSM. La domanda e' dominata dal vento
(quota variabile alta). Stati limite per componente:
- fondazione:  rottura grouted connection (ULS FRAGILE — guasto storico
               documentato DNV-GL sulle turbine anni 2000), ribaltamento
               (ULS fragile), capacita' (ULS duttile)
- torre:       fatica connessioni — giunti flangiati imbullonati e saldature
               (ULS fragile), instabilita' (ULS fragile)
- pale:        fatica alla radice (ULS duttile: composito, progressiva),
               erosione bordo d'attacco (SLS)
- drivetrain:  guasto cuscinetti/gearbox (SLS funzionale — non porta al
               collasso strutturale ma ferma la produzione, con impatto
               economico e di sicurezza operativa alto)

ACCOPPIAMENTO (squilibrio aerodinamico): pale degradate generano carichi
ciclici sbilanciati su torre e drivetrain — moltiplicatore di domanda
1 + 0.30·(1−cond_pale), cap 1.30.
"""
from typing import Any, Dict

# Aumento massimo della domanda a fatica per squilibrio aerodinamico (+30%).
AERO_IMBALANCE_FACTOR_MAX = 0.30
# Domanda dominata dal vento: quota variabile alta.
DEAD_SHARE = 0.30
LIVE_SHARE = 0.70
RESISTANCE_BASE = 2.15


def turbine_component_limit_states(env: Any, turbine: Any, component: Any) -> Dict[str, Dict[str, float]]:
    blades = turbine.components.get("pale")
    blades_condition = blades.current_condition if blades is not None else 1.0
    imbalance = 1.0 + AERO_IMBALANCE_FACTOR_MAX * max(0.0, 1.0 - blades_condition)

    demand_base = component.demand_bias * 0.60
    if component.name in ("torre", "drivetrain"):
        demand_base *= imbalance
    dead_mean = DEAD_SHARE * demand_base
    live_mean = LIVE_SHARE * demand_base
    dead_cov = min(0.10, max(0.05, env.params.demand_cov * 0.50))
    live_cov = min(0.35, max(env.params.demand_cov * 1.2, 0.20))   # vento: variabilita' alta
    code_live_effect = live_mean * 1.40

    condition = max(component.current_condition, 0.02)
    base_resistance = component.resistance_bias * condition * (
        RESISTANCE_BASE + 0.20 * max(0.0, 1.0 - component.epistemic_uncertainty)
    )
    material_cov = min(0.22, 0.05 + 0.35 * env.params.resistance_cov + 0.05 * (1.0 - condition))
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

    if component.name == "fondazione":
        return {
            "grouted_connection": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",
                "resistance_mean": base_resistance * 1.00,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
            "overturning": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",
                "resistance_mean": base_resistance * 1.15,
                "dead_load_mean": dead_mean * 0.95,
                "live_load_mean": live_mean * 1.05,
                "code_live_effect": code_live_effect * 1.05,
            },
            "bearing_capacity": {
                **common,
                "category": "ULS",
                "failure_mode_type": "ductile",
                "resistance_mean": base_resistance * 1.20,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
        }
    if component.name == "torre":
        return {
            "flange_weld_fatigue": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",   # rottura per fatica: improvvisa
                "resistance_mean": base_resistance * 1.02,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
            "buckling": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",
                "resistance_mean": base_resistance * 1.18,
                "dead_load_mean": dead_mean * 0.95,
                "live_load_mean": live_mean * 1.05,
                "code_live_effect": code_live_effect * 1.05,
            },
        }
    if component.name == "drivetrain":
        return {
            "bearing_gearbox_failure": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.78,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.90,
                "code_live_effect": code_live_effect * 0.90,
            },
        }
    # pale (e fallback)
    return {
        "root_fatigue": {
            **common,
            "category": "ULS",
            "failure_mode_type": "ductile",   # composito: danno progressivo
            "resistance_mean": base_resistance * 1.05,
            "dead_load_mean": dead_mean,
            "live_load_mean": live_mean,
            "code_live_effect": code_live_effect,
        },
        "leading_edge_erosion": {
            **common,
            "category": "SLS",
            "failure_mode_type": "serviceability",
            "resistance_mean": base_resistance * component.crack_bias * 0.80,
            "dead_load_mean": dead_mean * 0.90,
            "live_load_mean": live_mean * 0.85,
            "code_live_effect": code_live_effect * 0.85,
        },
    }


def turbine_revenue_multiplier(env: Any, turbine: Any) -> float:
    """Produzione limitata dal componente 'produttivo' peggiore (curtailment)."""
    conds = [
        c.current_condition
        for name, c in turbine.components.items()
        if name in ("pale", "drivetrain")
    ]
    if not conds:
        return 1.0
    return max(0.2, min(conds))
