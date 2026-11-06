"""Asset type "bridge": taxonomy a 4 componenti (impalcato, pile, spalle, appoggi)."""
from infralab.assets.base import AssetTypeSpec
from infralab.assets.bridges.limit_states import bridge_component_limit_states
from infralab.assets.bridges.profiles import (
    BRIDGE_COMPONENT_PROFILES,
    BRIDGE_DEFAULT_WEIGHTS,
    BRIDGE_STRUCTURAL_COMPONENTS,
)
from infralab.assets.registry import register_asset_type

BRIDGE_SPEC = register_asset_type(
    AssetTypeSpec(
        name="bridge",
        display_name="Ponte",
        component_profiles=BRIDGE_COMPONENT_PROFILES,
        structural_components=frozenset(BRIDGE_STRUCTURAL_COMPONENTS),
        default_component_weights=dict(BRIDGE_DEFAULT_WEIGHTS),
        limit_state_adapter=bridge_component_limit_states,
        fallback_component="impalcato",
        labels={
            "asset": "Ponte",
            "network": "Rete di ponti",
        },
        asset_property_schema={
            "length": {"label": "Lunghezza", "unit": "m", "type": "number", "default": 120, "min": 5, "max": 3000, "required": True},
            "width": {"label": "Larghezza", "unit": "m", "type": "number", "default": 12, "min": 3, "max": 60, "required": True},
            "adt": {"label": "Traffico giornaliero medio", "unit": "veicoli/g", "type": "number", "default": 8000, "min": 0, "max": 300000, "required": True},
            "vita_utile_anni": {"label": "Vita utile media", "unit": "anni", "type": "number", "default": 100, "min": 20, "max": 200},
            "incasso_trimestrale": {"label": "Ricavo trimestrale", "unit": "€", "type": "number", "default": 50000, "min": 0, "max": 10000000},
            "valore_asset": {"label": "Valore asset", "unit": "€", "type": "number", "default": 5000000, "min": 0, "max": 500000000},
            "eta_iniziale_anni": {"label": "Età iniziale", "unit": "anni", "type": "number", "default": 0, "min": 0, "max": 120},
            "seismic_sensitivity": {"label": "Sensibilità sismica (override)", "unit": "-", "type": "number", "min": 0, "max": 2},
        },
        ui_meta={
            "icon": "🌉",
            "demand_label": "Traffico giornaliero medio",
            "demand_unit": "veicoli/giorno",
            "area_label": "Superficie impalcato",
        },
        # Default calibrabile: un ponte isolato sismicamente puo' scendere
        # (override per-asset 'seismic_sensitivity'), uno ad appoggi fissi salire.
        hazard_sensitivities={"seismic": 1.0},
    )
)
