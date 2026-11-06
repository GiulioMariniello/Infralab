"""Adapter stato componente -> stati limite per l'asset type "bridge".

Estratto verbatim da BridgeEnv._component_limit_states (Fase B2): tre stati
limite (flessione ULS duttile, taglio ULS fragile, fessurazione SLS) con
domanda e resistenza funzioni algebriche di condizione, traffico e incertezza
epistemica. I coefficienti sono i tuning documentati in docs/MODEL_REPORT.md.
"""
from typing import Any, Dict


def confidence_factor(levels: Any, epistemic_uncertainty: float) -> float:
    """Fattore di confidenza del valutatore prudente: FC del primo gradino [soglia, FC]
    la cui soglia e' <= incertezza epistemica (gradini in ordine di soglia decrescente)."""
    for soglia, fc in levels:
        if float(epistemic_uncertainty) >= float(soglia):
            return float(fc)
    return 1.0


def bridge_component_limit_states(env: Any, bridge: Any, component: Any) -> Dict[str, Dict[str, float]]:
    traffic_metrics = env._traffic_fragility_metrics(bridge)
    if getattr(env.params, "traffic_demand_mode", "network_relative") == "self_designed":
        adt_ratio = env._traffic_growth_multiplier() * getattr(bridge, "traffic_restriction_factor", 1.0)
    else:
        adt_ratio = bridge.adt / max(1, env.params.traffic_reference_adt)
    demand_base = component.demand_bias * (
        0.54 + 0.18 * env._traffic_multiplier(bridge) + 0.10 * adt_ratio
    )
    demand_base *= 1.0 + 0.12 * max(0.0, 1.0 / traffic_metrics["eta"])

    dead_mean = 0.38 * demand_base
    live_mean = 0.62 * demand_base
    dead_cov = min(0.08, max(0.05, env.params.demand_cov * 0.50))
    live_cov = min(0.18, max(env.params.demand_cov, 0.04 + 0.22 * traffic_metrics["beta"] + 0.06 * (1.0 - component.current_condition)))
    code_live_multiplier = min(1.95, max(1.35, 0.72 * traffic_metrics["eta"]))
    code_live_effect = max(live_mean * code_live_multiplier, live_mean * 1.25)

    k = float(getattr(env.params, "capacity_condition_sensitivity", 1.0))
    capacity_factor = max(0.02, 1.0 - k * (1.0 - component.current_condition))
    base_resistance = float(getattr(env.params, "reliability_resistance_scale", 1.0)) * component.resistance_bias * capacity_factor * (
        2.24 + 0.20 * max(0.0, 1.0 - component.epistemic_uncertainty)
    )
    if getattr(bridge, "_is_assessment", False) and getattr(env.params, "assessment_confidence_mode", "none") == "knowledge_level":
        base_resistance /= confidence_factor(env.params.assessment_confidence_levels, component.epistemic_uncertainty)
    material_cov = min(0.24, 0.04 + 0.40 * env.params.resistance_cov + 0.05 * (1.0 - component.current_condition))
    geometry_cov = min(0.10, env.params.reliability_geometry_cov * (1.0 + 0.12 * component.profile["shape_factor"]))
    model_resistance_cov = min(
        0.16,
        env.params.reliability_model_uncertainty_cov * (1.0 + 0.75 * component.epistemic_uncertainty),
    )
    model_demand_cov = min(0.14, 0.04 + 0.30 * env.params.demand_cov + 0.05 * component.epistemic_uncertainty)

    return {
        "flexure": {
            "category": "ULS",
            "failure_mode_type": "ductile",
            "resistance_mean": base_resistance * 1.08,
            "material_cov": material_cov,
            "geometry_cov": geometry_cov,
            "model_resistance_cov": model_resistance_cov,
            "dead_load_mean": dead_mean,
            "dead_load_cov": dead_cov,
            "live_load_mean": live_mean,
            "live_load_cov": live_cov,
            "model_demand_cov": model_demand_cov,
            "code_live_effect": code_live_effect,
            "proof_load_cov": 0.035,
            "live_load_distribution": "gumbel",
        },
        "shear": {
            "category": "ULS",
            "failure_mode_type": "brittle",
            "resistance_mean": base_resistance * (0.92 + 0.06 * component.traffic_sensitivity),
            "material_cov": min(0.34, material_cov * 1.08),
            "geometry_cov": min(0.22, geometry_cov * 1.05),
            "model_resistance_cov": min(0.38, model_resistance_cov * 1.05),
            "dead_load_mean": dead_mean * 0.88,
            "dead_load_cov": min(0.20, dead_cov * 1.10),
            "live_load_mean": live_mean * 1.10,
            "live_load_cov": min(0.72, live_cov * 1.10),
            "model_demand_cov": min(0.30, model_demand_cov * 1.05),
            "code_live_effect": code_live_effect * 1.12,
            "proof_load_cov": 0.040,
            "live_load_distribution": "gumbel",
        },
        "cracking": {
            "category": "SLS",
            "failure_mode_type": "serviceability",
            "resistance_mean": base_resistance * component.crack_bias * 0.76,
            "material_cov": min(0.26, material_cov * 0.90),
            "geometry_cov": geometry_cov,
            "model_resistance_cov": min(0.30, model_resistance_cov * 0.90),
            "dead_load_mean": dead_mean * 0.95,
            "dead_load_cov": dead_cov,
            "live_load_mean": live_mean * 0.86,
            "live_load_cov": min(0.65, live_cov * 0.90),
            "model_demand_cov": model_demand_cov,
            "code_live_effect": code_live_effect * 0.92,
            "proof_load_cov": 0.030,
            "live_load_distribution": "gumbel",
        },
    }
