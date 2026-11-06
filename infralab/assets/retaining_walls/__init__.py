"""Asset type "retaining_wall": muro di sostegno a 3 componenti.

Secondo asset type del laboratorio: dimostra che il motore (degrado, FOSM,
proof load, economia, agenti) è agnostico rispetto al tipo di opera. Stati
limite geotecnici classici (scorrimento, ribaltamento, capacità portante) come
coppie R-S lognormali; il drenaggio intasato aumenta la spinta (domanda).
"""
from infralab.assets.base import AssetTypeSpec
from infralab.assets.registry import register_asset_type
from infralab.assets.retaining_walls.limit_states import wall_component_limit_states
from infralab.assets.retaining_walls.profiles import (
    WALL_COMPONENT_PROFILES,
    WALL_DEFAULT_WEIGHTS,
    WALL_MAINTENANCE_OPS,
    WALL_STRUCTURAL_COMPONENTS,
)

RETAINING_WALL_SPEC = register_asset_type(
    AssetTypeSpec(
        name="retaining_wall",
        display_name="Muro di sostegno",
        component_profiles=WALL_COMPONENT_PROFILES,
        structural_components=frozenset(WALL_STRUCTURAL_COMPONENTS),
        default_component_weights=dict(WALL_DEFAULT_WEIGHTS),
        limit_state_adapter=wall_component_limit_states,
        fallback_component="paramento",
        labels={
            "asset": "Muro di sostegno",
            "network": "Rete di opere di sostegno",
        },
        maintenance_ops=dict(WALL_MAINTENANCE_OPS),
        asset_property_schema={
            "length": {"label": "Sviluppo", "unit": "m", "type": "number", "default": 60, "min": 5, "max": 1000, "required": True},
            "width": {"label": "Altezza fuori terra", "unit": "m", "type": "number", "default": 6, "min": 1, "max": 25, "required": True},
            "adt": {"label": "Traffico a tergo", "unit": "veicoli/g", "type": "number", "default": 4000, "min": 0, "max": 100000, "required": True},
            "vita_utile_anni": {"label": "Vita utile media", "unit": "anni", "type": "number", "default": 100, "min": 20, "max": 200},
            "incasso_trimestrale": {"label": "Ricavo trimestrale", "unit": "€", "type": "number", "default": 0, "min": 0, "max": 1000000},
            "valore_asset": {"label": "Valore asset", "unit": "€", "type": "number", "default": 800000, "min": 0, "max": 50000000},
            "eta_iniziale_anni": {"label": "Età iniziale", "unit": "anni", "type": "number", "default": 0, "min": 0, "max": 120},
            "seismic_sensitivity": {"label": "Sensibilità sismica (override)", "unit": "-", "type": "number", "min": 0, "max": 2},
        },
        ui_meta={
            "icon": "🧱",
            "demand_label": "Traffico a tergo",
            "demand_unit": "veicoli/giorno",
            "area_label": "Superficie paramento",
        },
        hazard_sensitivities={"seismic": 1.0},
    )
)
