"""Asset type "solar_tracker": blocco di inseguitori FV a 4 componenti.

Accoppiamento stow (attuatori→domanda vento, cap 3.5x), ricavi legati alla
condizione dei moduli. Carico di progetto governante: VENTO — la sensibilita'
sismica bassa (0.15) e' una scelta motivata, non una dimenticanza.
"""
from infralab.assets.base import AssetTypeSpec
from infralab.assets.registry import register_asset_type
from infralab.assets.solar_trackers.limit_states import (
    tracker_component_limit_states,
    tracker_revenue_multiplier,
)
from infralab.assets.solar_trackers.profiles import (
    TRACKER_COMPONENT_PROFILES,
    TRACKER_DEFAULT_LIFETIMES,
    TRACKER_DEFAULT_WEIGHTS,
    TRACKER_MAINTENANCE_OPS,
    TRACKER_STRUCTURAL_COMPONENTS,
)

SOLAR_TRACKER_SPEC = register_asset_type(
    AssetTypeSpec(
        name="solar_tracker",
        display_name="Inseguitore fotovoltaico",
        component_profiles=TRACKER_COMPONENT_PROFILES,
        structural_components=frozenset(TRACKER_STRUCTURAL_COMPONENTS),
        default_component_weights=dict(TRACKER_DEFAULT_WEIGHTS),
        limit_state_adapter=tracker_component_limit_states,
        fallback_component="struttura_torsione",
        labels={"asset": "Blocco tracker", "network": "Impianto fotovoltaico"},
        maintenance_ops=dict(TRACKER_MAINTENANCE_OPS),
        asset_property_schema={
            "length": {"label": "Lunghezza blocco", "unit": "m", "type": "number", "default": 120, "min": 20, "max": 500, "required": True},
            "width": {"label": "Larghezza captante", "unit": "m", "type": "number", "default": 4, "min": 2, "max": 12, "required": True},
            "adt": {"label": "Ore di inseguimento", "unit": "h/anno", "type": "number", "default": 3500, "min": 500, "max": 4500, "required": True},
            "incasso_trimestrale": {"label": "Ricavo trimestrale", "unit": "€", "type": "number", "default": 25000, "min": 0, "max": 5000000},
            "valore_asset": {"label": "Valore asset", "unit": "€", "type": "number", "default": 900000, "min": 0, "max": 50000000},
            "eta_iniziale_anni": {"label": "Età iniziale", "unit": "anni", "type": "number", "default": 0, "min": 0, "max": 30},
            "seismic_sensitivity": {"label": "Sensibilità sismica (override)", "unit": "-", "type": "number", "min": 0, "max": 2},
        },
        ui_meta={
            "icon": "☀️",
            "demand_label": "Ore di inseguimento",
            "demand_unit": "h/anno",
            "area_label": "Superficie captante",
            "demand_proxy_note": "Il carico governante è il vento (stow): la sismica pesa poco per scelta motivata.",
        },
        hazard_sensitivities={"seismic": 0.15},
        component_default_lifetimes=dict(TRACKER_DEFAULT_LIFETIMES),
        revenue_condition_adapter=tracker_revenue_multiplier,
    )
)
