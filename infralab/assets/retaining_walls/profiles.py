"""Profili componenti e catalogo manutenzioni dell'asset type "retaining_wall".

Tre componenti: paramento (fusto del muro), fondazione, drenaggio. Il
drenaggio degrada rapidamente (intasamento) e, tramite l'adapter degli stati
limite, un drenaggio degradato AUMENTA la spinta sul muro (accoppiamento
domanda-condizione tipico delle opere di sostegno).
"""

WALL_COMPONENT_PROFILES = {
    "paramento": {
        "shape_factor": 1.10,
        "scale_mean": 1.00,
        "traffic_sensitivity": 0.30,
        "environment_sensitivity": 1.05,
        "resistance_bias": 1.05,
        "demand_bias": 0.95,
        "crack_bias": 0.95,
    },
    "fondazione": {
        "shape_factor": 1.25,
        "scale_mean": 1.15,
        "traffic_sensitivity": 0.25,
        "environment_sensitivity": 0.95,
        "resistance_bias": 1.15,
        "demand_bias": 0.85,
        "crack_bias": 1.05,
    },
    "drenaggio": {
        "shape_factor": 0.95,
        "scale_mean": 0.75,   # degrada presto: intasamento
        "traffic_sensitivity": 0.20,
        "environment_sensitivity": 1.25,
        "resistance_bias": 0.90,
        "demand_bias": 0.70,
        "crack_bias": 0.85,
    },
}

# Il collasso e' determinato da paramento e fondazione; il drenaggio degrada
# la funzionalita' (e aumenta la spinta) ma non causa collasso diretto.
WALL_STRUCTURAL_COMPONENTS = ("paramento", "fondazione")

WALL_DEFAULT_WEIGHTS = {"paramento": 0.45, "fondazione": 0.35, "drenaggio": 0.20}

# Interventi specifici (si aggiungono alle azioni di analisi del catalogo globale).
WALL_MAINTENANCE_OPS = {
    "ripristino_paramento": {
        "cost_m2": 900.0, "co2_m2": 120.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.2,
        "gains": {"paramento": 0.7, "drenaggio": 0.2}, "gain_std_dev": 0.05,
    },
    "consolidamento_fondazione": {
        "cost_m2": 2200.0, "co2_m2": 260.0, "type": "intervention", "duration_quarters": 4, "revenue_impact": 0.4,
        "gains": {"fondazione": 0.8, "paramento": 0.15}, "gain_std_dev": 0.05,
    },
    "pulizia_drenaggio": {
        "cost_m2": 60.0, "co2_m2": 2.0, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.0,
        "gains": {"drenaggio": 1.0}, "gain_std_dev": 0.0,
    },
}
