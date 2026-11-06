"""Asset type "building": edificio a 4 componenti, classe d'uso NTC esplicita.

Sismica ad alta sensibilita' (hazard esistente via lat/lon); accoppiamento
sul RATE di degrado: involucro degradato accelera il degrado di struttura e
solai (infiltrazioni → corrosione, fib MC2010), cap x1.8.
"""
from infralab.assets.base import AssetTypeSpec
from infralab.assets.buildings.limit_states import (
    building_component_limit_states,
    building_degradation_rate,
    building_lifetime,
)
from infralab.assets.buildings.profiles import (
    BUILDING_COMPONENT_PROFILES,
    BUILDING_DEFAULT_LIFETIMES,
    BUILDING_DEFAULT_WEIGHTS,
    BUILDING_MAINTENANCE_OPS,
    BUILDING_STRUCTURAL_COMPONENTS,
)
from infralab.assets.registry import register_asset_type

BUILDING_SPEC = register_asset_type(
    AssetTypeSpec(
        name="building",
        display_name="Edificio",
        component_profiles=BUILDING_COMPONENT_PROFILES,
        structural_components=frozenset(BUILDING_STRUCTURAL_COMPONENTS),
        default_component_weights=dict(BUILDING_DEFAULT_WEIGHTS),
        limit_state_adapter=building_component_limit_states,
        fallback_component="struttura_verticale",
        labels={"asset": "Edificio", "network": "Patrimonio edilizio"},
        maintenance_ops=dict(BUILDING_MAINTENANCE_OPS),
        asset_property_schema={
            "length": {"label": "Lunghezza in pianta", "unit": "m", "type": "number", "default": 40, "min": 5, "max": 300, "required": True},
            "width": {"label": "Larghezza in pianta", "unit": "m", "type": "number", "default": 20, "min": 5, "max": 200, "required": True},
            "adt": {"label": "Occupanti equivalenti/giorno", "unit": "persone", "type": "number", "default": 500, "min": 0, "max": 100000, "required": True},
            "classe_uso": {"label": "Classe d'uso NTC", "unit": "-", "type": "select", "options": ["II", "IV"], "default": "II", "required": True},
            "incasso_trimestrale": {"label": "Ricavo trimestrale", "unit": "€", "type": "number", "default": 60000, "min": 0, "max": 10000000},
            "valore_asset": {"label": "Valore asset", "unit": "€", "type": "number", "default": 8000000, "min": 0, "max": 500000000},
            "eta_iniziale_anni": {"label": "Età iniziale", "unit": "anni", "type": "number", "default": 0, "min": 0, "max": 120},
            "seismic_sensitivity": {"label": "Sensibilità sismica (override)", "unit": "-", "type": "number", "min": 0, "max": 2},
        },
        ui_meta={
            "icon": "🏢",
            "demand_label": "Occupanti equivalenti",
            "demand_unit": "persone/giorno",
            "area_label": "Superficie in pianta",
        },
        hazard_sensitivities={"seismic": 1.0},
        component_default_lifetimes=dict(BUILDING_DEFAULT_LIFETIMES),
        degradation_rate_adapter=building_degradation_rate,
        lifetime_adapter=building_lifetime,
    )
)
