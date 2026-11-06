"""Profili componenti e catalogo manutenzioni dell'asset type "building".

Quattro componenti: struttura verticale, solai (portanti), involucro,
impianti. Le vite base sono riferite alla classe d'uso NTC II (VN 50 anni,
gestione su orizzonte lungo); per classe IV (opere strategiche, VN 100)
il lifetime_adapter raddoppia le vite dei componenti strutturali.

Accoppiamento sul RATE (non sulla domanda): involucro degradato →
infiltrazioni → carbonatazione/cloruri → corrosione armature accelerata
(fib Model Code 2010). Vedi limit_states.building_degradation_rate.
"""

BUILDING_COMPONENT_PROFILES = {
    "struttura_verticale": {
        "shape_factor": 1.20,
        "scale_mean": 1.20,
        "traffic_sensitivity": 0.30,   # 'traffico' = intensita' d'uso/occupazione
        "environment_sensitivity": 1.00,
        "resistance_bias": 1.10,
        "demand_bias": 0.95,
        "crack_bias": 1.00,
    },
    "solai": {
        "shape_factor": 1.30,
        "scale_mean": 1.05,
        "traffic_sensitivity": 0.50,
        "environment_sensitivity": 0.95,
        "resistance_bias": 1.05,
        "demand_bias": 1.00,
        "crack_bias": 0.90,
    },
    "involucro": {
        "shape_factor": 1.60,
        "scale_mean": 0.75,
        "traffic_sensitivity": 0.10,
        "environment_sensitivity": 1.30,
        "resistance_bias": 0.95,
        "demand_bias": 0.80,
        "crack_bias": 0.85,
    },
    "impianti": {
        "shape_factor": 1.50,
        "scale_mean": 0.55,
        "traffic_sensitivity": 0.60,
        "environment_sensitivity": 0.90,
        "resistance_bias": 0.90,
        "demand_bias": 0.75,
        "crack_bias": 0.80,
    },
}

BUILDING_STRUCTURAL_COMPONENTS = ("struttura_verticale", "solai")

BUILDING_DEFAULT_WEIGHTS = {
    "struttura_verticale": 0.35, "solai": 0.30, "involucro": 0.20, "impianti": 0.15,
}

# Vite base (classe d'uso II). Classe IV: componenti strutturali x2 (adapter).
BUILDING_DEFAULT_LIFETIMES = {
    "struttura_verticale": 100.0, "solai": 80.0, "involucro": 40.0, "impianti": 25.0,
}

BUILDING_MAINTENANCE_OPS = {
    "consolidamento_sismico": {
        "cost_m2": 1200.0, "co2_m2": 150.0, "type": "intervention", "duration_quarters": 5, "revenue_impact": 0.8,
        "gains": {"struttura_verticale": 0.6, "solai": 0.2}, "gain_std_dev": 0.05,
    },
    "ripristino_solai": {
        "cost_m2": 400.0, "co2_m2": 55.0, "type": "intervention", "duration_quarters": 3, "revenue_impact": 0.4,
        "gains": {"solai": 0.7}, "gain_std_dev": 0.05,
    },
    "rifacimento_involucro": {
        "cost_m2": 350.0, "co2_m2": 40.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.2,
        "gains": {"involucro": 0.9}, "gain_std_dev": 0.04,
    },
    "revisione_impianti": {
        "cost_m2": 150.0, "co2_m2": 12.0, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.1,
        "gains": {"impianti": 0.8}, "gain_std_dev": 0.05,
    },
}
