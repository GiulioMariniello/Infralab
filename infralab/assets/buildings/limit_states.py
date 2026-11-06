"""Adapter stati limite + accoppiamento sul rate per gli edifici.

Stati limite per componente:
- struttura_verticale: capacita' sismica/taglio (ULS FRAGILE), deriva (SLS)
- solai:               flessione (ULS duttile), PUNZONAMENTO (ULS FRAGILE —
                       meccanismo distinto e improvviso: Sampoong, Pipers Row),
                       fessurazione/vibrazioni (SLS)
- involucro:           infiltrazioni (SLS)
- impianti:            fuori servizio (SLS funzionale)

ACCOPPIAMENTO SUL RATE (fib Model Code 2010): l'involucro degradato lascia
entrare acqua → carbonatazione/ingresso cloruri → innesco corrosione
anticipato. E' un'accelerazione del degrado nel tempo, non un carico
aggiuntivo: moltiplicatore del rate 1 + 0.8·(1−cond_involucro), cap 1.8,
applicato ai soli componenti strutturali (struttura e solai).
"""
from typing import Any, Dict

# Accelerazione massima del degrado strutturale con involucro azzerato (cap x1.8).
ENVELOPE_RATE_FACTOR_MAX = 0.8
DEAD_SHARE = 0.65
LIVE_SHARE = 0.35
RESISTANCE_BASE = 2.20


def building_degradation_rate(env: Any, building: Any, component: Any) -> float:
    """Moltiplicatore del rate di degrado (hook degradation_rate_adapter)."""
    if component.name not in ("struttura_verticale", "solai"):
        return 1.0
    envelope = building.components.get("involucro")
    envelope_condition = envelope.current_condition if envelope is not None else 1.0
    return 1.0 + ENVELOPE_RATE_FACTOR_MAX * max(0.0, 1.0 - envelope_condition)


def building_lifetime(asset_data: Dict[str, Any], component_name: str):
    """Vita media per componente in funzione della classe d'uso NTC.

    Classe II (ordinaria, VN 50): vite base. Classe IV (strategica, VN 100):
    componenti strutturali x2. La classe si dichiara per-asset nello scenario
    (chiave 'classe_uso', default 'II').
    """
    from infralab.assets.buildings.profiles import (
        BUILDING_DEFAULT_LIFETIMES,
        BUILDING_STRUCTURAL_COMPONENTS,
    )

    base = BUILDING_DEFAULT_LIFETIMES.get(component_name)
    if base is None:
        return None
    classe = str(asset_data.get("classe_uso", "II")).strip().upper()
    if classe == "IV" and component_name in BUILDING_STRUCTURAL_COMPONENTS:
        return base * 2.0
    return base


def building_component_limit_states(env: Any, building: Any, component: Any) -> Dict[str, Dict[str, float]]:
    demand_base = component.demand_bias * 0.60
    dead_mean = DEAD_SHARE * demand_base
    live_mean = LIVE_SHARE * demand_base
    dead_cov = min(0.10, max(0.05, env.params.demand_cov * 0.50))
    live_cov = min(0.30, max(env.params.demand_cov, 0.15))
    code_live_effect = live_mean * 1.50

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

    if component.name == "solai":
        return {
            "flexure": {
                **common,
                "category": "ULS",
                "failure_mode_type": "ductile",
                "resistance_mean": base_resistance * 1.08,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean,
                "code_live_effect": code_live_effect,
            },
            "punching_shear": {
                **common,
                "category": "ULS",
                "failure_mode_type": "brittle",   # collasso improvviso, senza preavviso
                "resistance_mean": base_resistance * 1.15,
                "dead_load_mean": dead_mean,
                "live_load_mean": live_mean * 1.05,
                "code_live_effect": code_live_effect * 1.05,
            },
            "cracking_vibration": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.80,
                "dead_load_mean": dead_mean * 0.95,
                "live_load_mean": live_mean * 0.90,
                "code_live_effect": code_live_effect * 0.90,
            },
        }
    if component.name == "involucro":
        return {
            "water_ingress": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.78,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.85,
                "code_live_effect": code_live_effect * 0.85,
            },
        }
    if component.name == "impianti":
        return {
            "out_of_service": {
                **common,
                "category": "SLS",
                "failure_mode_type": "serviceability",
                "resistance_mean": base_resistance * component.crack_bias * 0.75,
                "dead_load_mean": dead_mean * 0.90,
                "live_load_mean": live_mean * 0.90,
                "code_live_effect": code_live_effect * 0.90,
            },
        }
    # struttura_verticale (e fallback)
    return {
        "seismic_shear_capacity": {
            **common,
            "category": "ULS",
            "failure_mode_type": "brittle",
            "resistance_mean": base_resistance * 1.05,
            "dead_load_mean": dead_mean,
            "live_load_mean": live_mean,
            "code_live_effect": code_live_effect,
        },
        "drift": {
            **common,
            "category": "SLS",
            "failure_mode_type": "serviceability",
            "resistance_mean": base_resistance * component.crack_bias * 0.82,
            "dead_load_mean": dead_mean * 0.95,
            "live_load_mean": live_mean * 0.95,
            "code_live_effect": code_live_effect * 0.95,
        },
    }
