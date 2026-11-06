"""Profili componenti e manutenzioni dell'asset type "solar_tracker".

Un asset = un blocco di inseguitori monoassiali di un impianto FV
utility-scale (vita PPA 25-30 anni). Contrasto didattico con ponti/edifici:
componenti economici, guasti attuatori quasi-random (k~1.1), e il piu'
forte accoppiamento domanda-condizione del laboratorio — attuatore guasto →
mancato stow in evento di vento → momento alla base 2.5-4x (lo stow riduce
il momento del 60-90%: e' la mitigazione primaria degli standard tracker).
"""

TRACKER_COMPONENT_PROFILES = {
    "fondazioni_micropali": {
        "shape_factor": 1.40,
        "scale_mean": 1.10,
        "traffic_sensitivity": 0.20,   # 'traffico' = ore di inseguimento (cicli)
        "environment_sensitivity": 1.10,
        "resistance_bias": 1.10,
        "demand_bias": 0.90,
        "crack_bias": 1.00,
    },
    "struttura_torsione": {
        "shape_factor": 2.00,          # fatica del tubo di torsione
        "scale_mean": 1.00,
        "traffic_sensitivity": 0.55,
        "environment_sensitivity": 1.05,
        "resistance_bias": 1.00,
        "demand_bias": 1.00,
        "crack_bias": 0.95,
    },
    "attuatori": {
        "shape_factor": 1.10,          # quasi esponenziale: guasti random
        "scale_mean": 0.55,
        "traffic_sensitivity": 0.75,
        "environment_sensitivity": 1.00,
        "resistance_bias": 0.90,
        "demand_bias": 0.80,
        "crack_bias": 0.80,
    },
    "moduli": {
        "shape_factor": 1.70,
        "scale_mean": 0.95,
        "traffic_sensitivity": 0.10,
        "environment_sensitivity": 1.15,
        "resistance_bias": 0.95,
        "demand_bias": 0.75,
        "crack_bias": 0.85,
    },
}

TRACKER_STRUCTURAL_COMPONENTS = ("fondazioni_micropali", "struttura_torsione")

TRACKER_DEFAULT_WEIGHTS = {
    "fondazioni_micropali": 0.25, "struttura_torsione": 0.30, "attuatori": 0.20, "moduli": 0.25,
}

# Vite medie (anni): nessun componente oltre la vita di progetto dell'impianto.
TRACKER_DEFAULT_LIFETIMES = {
    "fondazioni_micropali": 30.0, "struttura_torsione": 30.0, "attuatori": 11.0, "moduli": 28.0,
}

TRACKER_MAINTENANCE_OPS = {
    "revisione_attuatori": {
        "cost": 40000.0, "co2": 3.0, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.1,
        "gains": {"attuatori": 1.0}, "gain_std_dev": 0.02,
    },
    "serraggio_struttura": {
        "cost": 25000.0, "co2": 2.0, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.1,
        "gains": {"struttura_torsione": 0.3}, "gain_std_dev": 0.04,
    },
    "ripristino_micropali": {
        "cost": 120000.0, "co2": 30.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.5,
        "gains": {"fondazioni_micropali": 0.7}, "gain_std_dev": 0.05,
    },
    "pulizia_moduli": {
        # Qui l'area ha senso fisico (superficie captante).
        "cost_m2": 2.0, "co2_m2": 0.1, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.0,
        "gains": {"moduli": 0.3}, "gain_std_dev": 0.03,
    },
}
