"""Profili componenti e catalogo manutenzioni dell'asset type "wind_turbine".

Un asset = un aerogeneratore onshore. Vita di progetto IEC 61400-1 ~20-25
anni: le vite dei componenti (tabella concordata col dominio) riflettono la
gerarchia reale di criticita' — il drivetrain e' il piu' corto (12a) con k
basso (guasti anche early-life, evidenza ReliaWind/SPARTA), le pale hanno
fatica marcata (k 2.5), la torre fatica su giunti flangiati/saldature (k 2.2).

I costi degli interventi sono FISSI per unita' (non €/m²: gearbox, pala,
grouted connection sono asset unitari) e includono la mobilitazione della
gru principale dove serve — spesso la voce dominante (100-300 k€).
"""

TURBINE_COMPONENT_PROFILES = {
    "fondazione": {
        "shape_factor": 1.30,
        "scale_mean": 1.20,
        "traffic_sensitivity": 0.35,   # 'traffico' = ore equivalenti vento (proxy)
        "environment_sensitivity": 1.00,
        "resistance_bias": 1.15,
        "demand_bias": 0.85,
        "crack_bias": 1.00,
    },
    "torre": {
        "shape_factor": 2.20,          # fatica: degrado accelerato a fine vita
        "scale_mean": 1.00,
        "traffic_sensitivity": 0.70,
        "environment_sensitivity": 1.00,
        "resistance_bias": 1.05,
        "demand_bias": 1.00,
        "crack_bias": 0.95,
    },
    "pale": {
        "shape_factor": 2.50,
        "scale_mean": 0.90,
        "traffic_sensitivity": 0.80,
        "environment_sensitivity": 1.20,  # erosione bordo d'attacco: pioggia/particolato
        "resistance_bias": 1.00,
        "demand_bias": 1.05,
        "crack_bias": 0.90,
    },
    "drivetrain": {
        "shape_factor": 1.30,          # k basso: guasti anche early-life
        "scale_mean": 0.60,
        "traffic_sensitivity": 0.85,
        "environment_sensitivity": 0.90,
        "resistance_bias": 0.95,
        "demand_bias": 0.90,
        "crack_bias": 0.85,
    },
}

TURBINE_STRUCTURAL_COMPONENTS = ("fondazione", "torre", "pale")

TURBINE_DEFAULT_WEIGHTS = {"fondazione": 0.25, "torre": 0.30, "pale": 0.25, "drivetrain": 0.20}

# Vite utili medie (anni) — tabella concordata (IEC 61400 + dati di settore).
TURBINE_DEFAULT_LIFETIMES = {"fondazione": 30.0, "torre": 25.0, "pale": 22.0, "drivetrain": 12.0}

TURBINE_MAINTENANCE_OPS = {
    "ispezione_droni": {
        "cost": 8000.0, "co2": 0.5, "type": "analysis", "error_sigma": 0.05, "duration_quarters": 0,
        "revenue_impact": 0.0,
    },
    "riparazione_bordo_attacco": {
        # Rope access / piattaforma: nessuna gru principale.
        "cost": 60000.0, "co2": 6.0, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.1,
        "gains": {"pale": 0.4}, "gain_std_dev": 0.05,
    },
    "sostituzione_pala": {
        # ~130 k€ pala + ~250 k€ mobilitazione gru principale.
        "cost": 380000.0, "co2": 45.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.8,
        "gains": {"pale": 1.0}, "gain_std_dev": 0.02,
    },
    "sostituzione_gearbox": {
        # ~320 k€ gearbox + ~200 k€ gru.
        "cost": 520000.0, "co2": 60.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 1.0,
        "gains": {"drivetrain": 0.9}, "gain_std_dev": 0.03,
    },
    "ripristino_grouted_connection": {
        "cost": 120000.0, "co2": 25.0, "type": "intervention", "duration_quarters": 2, "revenue_impact": 0.3,
        "gains": {"fondazione": 0.6}, "gain_std_dev": 0.05,
    },
    "serraggio_flange_torre": {
        "cost": 25000.0, "co2": 1.5, "type": "intervention", "duration_quarters": 1, "revenue_impact": 0.1,
        "gains": {"torre": 0.15}, "gain_std_dev": 0.03,
    },
}
