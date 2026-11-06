"""Profili dei componenti dell'asset type "bridge".

Fattori di tuning per componente (media dei parametri campionati per-istanza):
- shape_factor / scale_mean: forma e scala della curva di degrado Weibull
- traffic_sensitivity / environment_sensitivity: esponenti dei moltiplicatori
  di invecchiamento
- resistance_bias / demand_bias / crack_bias: bias del mapping condizione ->
  stati limite (vedi limit_states.py)
"""

BRIDGE_COMPONENT_PROFILES = {
    "impalcato": {
        "shape_factor": 1.05,
        "scale_mean": 0.95,
        "traffic_sensitivity": 0.95,
        "environment_sensitivity": 0.85,
        "resistance_bias": 1.08,
        "demand_bias": 1.00,
        "crack_bias": 0.95,
    },
    "pile": {
        "shape_factor": 1.20,
        "scale_mean": 1.10,
        "traffic_sensitivity": 0.70,
        "environment_sensitivity": 0.95,
        "resistance_bias": 1.15,
        "demand_bias": 0.90,
        "crack_bias": 1.05,
    },
    "spalle": {
        "shape_factor": 1.10,
        "scale_mean": 1.00,
        "traffic_sensitivity": 0.60,
        "environment_sensitivity": 1.00,
        "resistance_bias": 1.00,
        "demand_bias": 0.82,
        "crack_bias": 1.00,
    },
    "appoggi": {
        "shape_factor": 1.35,
        "scale_mean": 1.12,
        "traffic_sensitivity": 0.75,
        "environment_sensitivity": 1.10,
        "resistance_bias": 0.92,
        "demand_bias": 0.76,
        "crack_bias": 0.88,
    },
}

# Componenti portanti: il loro collasso determina il fallimento del ponte
# (gli appoggi degradati riducono la funzionalita' ma non causano crollo).
BRIDGE_STRUCTURAL_COMPONENTS = ("impalcato", "pile", "spalle")

BRIDGE_DEFAULT_WEIGHTS = {"impalcato": 0.5, "pile": 0.3, "spalle": 0.15, "appoggi": 0.05}
