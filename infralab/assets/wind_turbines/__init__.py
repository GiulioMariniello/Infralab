"""Asset type "wind_turbine": aerogeneratore onshore a 4 componenti.

Vita di progetto IEC 61400-1 (~20-25 anni); accoppiamento squilibrio
aerodinamico pale→torre/drivetrain; ricavi legati alla condizione dei
componenti produttivi (curtailment). NOTA sul proxy di domanda: 'adt' e'
riletto come ore equivalenti vento/anno — proxy monotono dell'intensita' di
carico a fatica, NON un modello aeroelastico (limite dichiarato).
"""
from infralab.assets.base import AssetTypeSpec
from infralab.assets.registry import register_asset_type
from infralab.assets.wind_turbines.limit_states import (
    turbine_component_limit_states,
    turbine_revenue_multiplier,
)
from infralab.assets.wind_turbines.profiles import (
    TURBINE_COMPONENT_PROFILES,
    TURBINE_DEFAULT_LIFETIMES,
    TURBINE_DEFAULT_WEIGHTS,
    TURBINE_MAINTENANCE_OPS,
    TURBINE_STRUCTURAL_COMPONENTS,
)

WIND_TURBINE_SPEC = register_asset_type(
    AssetTypeSpec(
        name="wind_turbine",
        display_name="Aerogeneratore",
        component_profiles=TURBINE_COMPONENT_PROFILES,
        structural_components=frozenset(TURBINE_STRUCTURAL_COMPONENTS),
        default_component_weights=dict(TURBINE_DEFAULT_WEIGHTS),
        limit_state_adapter=turbine_component_limit_states,
        fallback_component="torre",
        labels={"asset": "Aerogeneratore", "network": "Parco eolico"},
        maintenance_ops=dict(TURBINE_MAINTENANCE_OPS),
        asset_property_schema={
            "length": {"label": "Altezza mozzo", "unit": "m", "type": "number", "default": 100, "min": 40, "max": 180, "required": True},
            "width": {"label": "Diametro rotore", "unit": "m", "type": "number", "default": 120, "min": 40, "max": 220, "required": True},
            "adt": {"label": "Ore equivalenti vento", "unit": "h/anno", "type": "number", "default": 2500, "min": 800, "max": 5000, "required": True},
            "incasso_trimestrale": {"label": "Ricavo trimestrale", "unit": "€", "type": "number", "default": 90000, "min": 0, "max": 10000000},
            "valore_asset": {"label": "Valore asset", "unit": "€", "type": "number", "default": 3500000, "min": 0, "max": 100000000},
            "eta_iniziale_anni": {"label": "Età iniziale", "unit": "anni", "type": "number", "default": 0, "min": 0, "max": 30},
            "seismic_sensitivity": {"label": "Sensibilità sismica (override)", "unit": "-", "type": "number", "min": 0, "max": 2},
        },
        ui_meta={
            "icon": "🌬️",
            "demand_label": "Ore equivalenti vento",
            "demand_unit": "h/anno",
            "area_label": "Area spazzata equivalente",
            "demand_proxy_note": (
                "Le ore equivalenti vento sono un proxy semplificato dell'intensità "
                "di carico a fatica, non un modello aeroelastico."
            ),
        },
        hazard_sensitivities={"seismic": 0.5},
        component_default_lifetimes=dict(TURBINE_DEFAULT_LIFETIMES),
        revenue_condition_adapter=turbine_revenue_multiplier,
    )
)
