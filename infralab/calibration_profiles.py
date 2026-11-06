from copy import deepcopy
from dataclasses import asdict
from typing import Dict, Any

from infralab.settings.config import SimParameters, MaintOpsConfig


CALIBRATION_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline_demo": {
        "description": "Profilo base dimostrativo con calibratura moderata.",
        "params": {
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 3,
            "traffic_first_lane_offset_m": 0.5,
            "traffic_congestion_probability": 0.01,
            "traffic_monthly_overload_probability": 0.09,
            "deterioration_process_cov": 0.12,
            "demand_cov": 0.12,
            "resistance_cov": 0.15,
            "proof_load_base_alpha": 0.95,
            "proof_load_damage_limit": 0.05,
            "seismic_bridge_class": "standard",
            "seismic_limit_state": "SD",
        },
        "maint_ops": {
            "proof_load_test": {
                "alpha": 0.95,
                "proof_load_cov": 0.05,
                "uncertainty_reduction": 0.45,
            }
        },
    },
    "miluccio_traffic_1970_1980": {
        "description": "Coerente con fragility analysis di impalcati PC italiani esistenti sotto traffico.",
        "params": {
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 3,
            "traffic_first_lane_offset_m": 0.0,
            "traffic_remaining_area_active": True,
            "traffic_congestion_probability": 0.01,
            "traffic_monthly_overload_probability": 0.09,
            "deterioration_process_cov": 0.15,
            "demand_cov": 0.14,
            "resistance_cov": 0.17,
            "reliability_target_beta": 3.8,
            "traffic_reference_adt": 10000,
        },
        "maint_ops": {},
    },
    "miluccio_restricted_use": {
        "description": "Profilo traffico con limitazioni d'uso: offset corsia e riduzione corsie.",
        "params": {
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 1,
            "traffic_first_lane_offset_m": 1.5,
            "traffic_remaining_area_active": False,
            "traffic_congestion_probability": 0.01,
            "traffic_monthly_overload_probability": 0.09,
            "deterioration_process_cov": 0.15,
            "demand_cov": 0.13,
            "resistance_cov": 0.17,
        },
        "maint_ops": {},
    },
    "addonizio_proof_load": {
        "description": "Profilo proof load coerente con approccio prior/posterior reliability.",
        "params": {
            "proof_load_base_alpha": 0.95,
            "proof_load_damage_limit": 0.05,
            "demand_cov": 0.10,
            "resistance_cov": 0.14,
            "reliability_target_beta": 3.8,
            "reliability_sls_target_beta": 1.5,
        },
        "maint_ops": {
            "proof_load_test": {
                "alpha": 0.95,
                "proof_load_cov": 0.05,
                "uncertainty_reduction": 0.50,
                "damage_limit": 0.05,
                "knowledge_gain": 0.55,
            },
            "monitoring": {
                "knowledge_gain": 0.35,
            },
        },
    },
    "franchin_tv_high_seismic": {
        "description": "Profilo sismico per tall viaduct code-conforming in sito ad alta sismicità.",
        "params": {
            "earthquake_enabled": True,
            "earthquake_annual_prob": 0.05,
            "earthquake_min_mag": 4.5,
            "earthquake_max_mag": 6.8,
            "seismic_bridge_class": "TV",
            "seismic_limit_state": "SD",
            "bridge_region_radius_km": 80.0,
        },
        "maint_ops": {},
    },
    "paper_combined_bridge_manager_v1": {
        "description": "Profilo combinato traffico-proof load-sisma per esperimenti del nuovo paper.",
        "params": {
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 3,
            "traffic_first_lane_offset_m": 0.5,
            "traffic_remaining_area_active": True,
            "traffic_congestion_probability": 0.01,
            "traffic_monthly_overload_probability": 0.09,
            "deterioration_process_cov": 0.14,
            "demand_cov": 0.12,
            "resistance_cov": 0.15,
            "proof_load_base_alpha": 0.95,
            "proof_load_damage_limit": 0.05,
            "reliability_target_beta": 4.0,
            "earthquake_enabled": True,
            "earthquake_annual_prob": 0.04,
            "earthquake_min_mag": 4.5,
            "earthquake_max_mag": 6.8,
            "seismic_bridge_class": "TV",
            "seismic_limit_state": "SD",
        },
        "maint_ops": {
            "proof_load_test": {
                "alpha": 0.95,
                "proof_load_cov": 0.05,
                "uncertainty_reduction": 0.50,
                "damage_limit": 0.05,
                "knowledge_gain": 0.55,
            },
            "monitoring": {
                "knowledge_gain": 0.35,
            },
        },
    },
    "paper1_proof_load_value_of_information": {
        "description": "Profilo dedicato al paper sulla proof load come azione di gestione sotto informazione imperfetta.",
        "params": {
            "management_objective_mode": "safety_driven",
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 3,
            "traffic_first_lane_offset_m": 0.5,
            "traffic_remaining_area_active": True,
            "traffic_congestion_probability": 0.01,
            "traffic_monthly_overload_probability": 0.09,
            "traffic_reference_adt": 10000,
            "deterioration_process_cov": 0.14,
            "demand_cov": 0.10,
            "resistance_cov": 0.12,
            "reliability_model_uncertainty_cov": 0.08,
            "reliability_geometry_cov": 0.04,
            "reliability_consequence_class": "CC2",
            "reliability_target_beta": 3.8,
            "reliability_sls_target_beta": 1.5,
            "reliability_brittle_failure_beta_increment": 0.3,
            "reliability_projection_years": 10,
            "proof_load_base_alpha": 0.95,
            "proof_load_damage_limit": 0.05,
            "proof_load_step_count": 4,
            "post_maintenance_observation_sigma": 0.12,
            "post_maintenance_knowledge_gain": 0.10,
            "earthquake_enabled": False,
            "seismic_bridge_class": "standard",
            "seismic_limit_state": "SD",
        },
        "maint_ops": {
            "L1_inspect": {
                "cost_m2": 1.0,
                "error_sigma": 0.10,
            },
            "L4_verify": {
                "cost_m2": 100.0,
                "error_sigma": 0.01,
            },
            "monitoring": {
                "cost_m2": 70.0,
                "error_sigma": 0.005,
                "knowledge_gain": 0.35,
            },
            "proof_load_test": {
                "cost_m2": 25.0,
                "alpha": 0.95,
                "alpha_min": 0.45,
                "alpha_max": 1.05,
                "alpha_grid_size": 13,
                "proof_load_cov": 0.05,
                "uncertainty_reduction": 0.50,
                "damage_limit": 0.05,
                "knowledge_gain": 0.55
            },
        },
    },
    "ssrn_llm_explainable_2025": {
        "description": "Profilo per campagne LLM zero-shot con memoria breve, profili di rischio e portafogli scalabili.",
        "params": {
            "traffic_load_model": "GL_heavy",
            "traffic_lane_count": 3,
            "traffic_first_lane_offset_m": 0.5,
            "traffic_remaining_area_active": True,
            "deterioration_process_cov": 0.14,
            "demand_cov": 0.12,
            "resistance_cov": 0.15,
            "reliability_target_beta": 4.0,
            "proof_load_base_alpha": 0.95,
        },
        "maint_ops": {
            "monitoring": {
                "knowledge_gain": 0.35,
            },
            "proof_load_test": {
                "knowledge_gain": 0.50,
                "uncertainty_reduction": 0.45,
            },
        },
    },
    "fhwa_unit_costs_2025": {
        "description": (
            "Costi unitari degli interventi allineati alle fonti pubbliche americane e italiane. "
            "Valore di ricostruzione 3000 EUR/m2 (FHWA Bridge Replacement Unit Costs 2025: mediana "
            "di stato 3296, media non-NHS pesata 3060; Italia 3600, Contardi & La Fortezza 2025). "
            "Riabilitazione completa = 68% della ricostruzione, la quota usata dalla FHWA in ogni "
            "tabella annuale dal 2013 al 2025 e coerente con la soglia VDOT del 65% e con il "
            "rapporto italiano 2500/3600: 2040 EUR/m2 invece di 3500 (che era il 117% della "
            "ricostruzione e rendeva la riabilitazione dominata dalla sostituzione). "
            "Sostituzione dell'impalcato riportata al costo di ricostruzione (3000) perche' "
            "nessuna azione puo' costare piu' del rifacimento. Il resto del catalogo resta "
            "invariato. Vedi papers/infralab-benchmark-aei-2026/positioning/validation-engineering.md."
        ),
        "params": {},
        "maint_ops": {
            "ripristino_completo": {"cost_m2": 2040.0},
            "replace_impalcato": {"cost_m2": 3000.0},
        },
    },
    "fhwa_unit_costs_2025_b": {
        "description": (
            "Come fhwa_unit_costs_2025, piu' la sostituzione degli appoggi prezzata a metro quadro. "
            "Nel catalogo base replace_appoggi costa 100000 EUR a intervento su qualunque ponte: "
            "sopra circa 33 m2 costa meno di replace_impalcato con lo stesso guadagno totale, "
            "quindi ogni regola che rompe i pareggi sul costo la sceglie quasi ovunque (sul ponte "
            "da 58000 m2 del campione Delaware costa 0.1 M contro 175 M). 190 EUR/m2 conserva i "
            "0.1 M sul ponte mediano (520 m2): valore PROVVISORIO, da sostituire con un costo "
            "unitario di fonte per appoggio o per metro di linea di appoggio. I guadagni laterali "
            "su impalcato, pile e spalle scendono da 0.15 a 0.05: sostituire gli appoggi non "
            "ripristina le altre parti."
        ),
        "params": {},
        "maint_ops": {
            "ripristino_completo": {"cost_m2": 2040.0},
            "replace_impalcato": {"cost_m2": 3000.0},
            "replace_appoggi": {
                "cost": 0, "cost_m2": 190.0, "co2": 0.0, "co2_m2": 15.2,
                "gains": {"impalcato": 0.05, "pile": 0.05, "spalle": 0.05, "appoggi": 1.0},
            },
        },
    },
}


def list_calibration_profiles() -> list[str]:
    return list(CALIBRATION_PROFILES.keys())


def get_calibration_profile(name: str) -> Dict[str, Any]:
    return deepcopy(CALIBRATION_PROFILES[name])


def apply_calibration_profile(
    params: SimParameters,
    maint_ops: Dict[str, Any],
    profile_name: str,
) -> tuple[SimParameters, Dict[str, Any]]:
    profile = get_calibration_profile(profile_name)
    params.calibration_profile = profile_name
    for key, value in profile.get("params", {}).items():
        setattr(params, key, value)

    updated_ops = deepcopy(maint_ops)
    for op_name, op_values in profile.get("maint_ops", {}).items():
        if op_name not in updated_ops:
            updated_ops[op_name] = {}
        updated_ops[op_name].update(op_values)
    return params, updated_ops


def default_maint_ops_dict() -> Dict[str, Any]:
    return asdict(MaintOpsConfig())
