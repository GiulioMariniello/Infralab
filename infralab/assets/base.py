"""Contratto di un asset type: cosa serve al motore per simulare una classe di opere.

Il motore di simulazione (`infralab.core.environment`) è agnostico rispetto al
tipo di asset: tutta la conoscenza specifica (taxonomy dei componenti, mapping
condizione → stati limite, componenti portanti) vive in una AssetTypeSpec
registrata nel registry. Aggiungere un nuovo tipo di opera = registrare una
nuova spec, senza toccare il motore.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

# Firma dell'adapter: (env, asset, component) -> Dict[str, Dict[str, float]]
# dove il risultato è il dizionario di stati limite consumato da
# infralab.core.reliability.evaluate_limit_states.
LimitStateAdapter = Callable[[Any, Any, Any], Dict[str, Dict[str, float]]]
# (env, asset) -> moltiplicatore dei ricavi trimestrali in [0, 1]
# (accoppiamento condizione -> produzione: turbine, tracker).
RevenueConditionAdapter = Callable[[Any, Any], float]
# (env, asset, component) -> moltiplicatore della velocita' di degrado (>= 0).
# Accoppiamenti sul RATE anziche' sulla domanda (es. involucro degradato ->
# infiltrazioni -> corrosione accelerata di struttura e solai, fib MC2010).
DegradationRateAdapter = Callable[[Any, Any, Any], float]
# (asset_data, component_name) -> vita utile media in anni, o None per il
# default. Permette vite dipendenti dalle proprieta' dell'asset (es. classe
# d'uso NTC di un edificio).
LifetimeAdapter = Callable[[Dict[str, Any], str], Optional[float]]


@dataclass(frozen=True)
class AssetTypeSpec:
    name: str
    display_name: str
    # Profili dei componenti: chiave = nome componente, valore = dict di fattori
    # (shape_factor, scale_mean, traffic_sensitivity, environment_sensitivity,
    #  resistance_bias, demand_bias, crack_bias).
    component_profiles: Dict[str, Dict[str, float]]
    # Componenti portanti: il loro collasso determina il fallimento dell'asset.
    structural_components: FrozenSet[str]
    # Pesi di default dei componenti (usati per l'health index e la creazione).
    default_component_weights: Dict[str, float]
    # Mapping stato componente -> stati limite R-S per il motore FOSM.
    limit_state_adapter: LimitStateAdapter
    # Componente il cui profilo fa da fallback per nomi sconosciuti.
    fallback_component: str
    # Etichette UI opzionali (italiano).
    labels: Dict[str, str] = field(default_factory=dict)
    # Operazioni di manutenzione specifiche dell'asset type, aggiunte al
    # catalogo globale (MaintOpsConfig) in fase di costruzione dell'ambiente.
    maintenance_ops: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # --- Estensioni opzionali (default neutri: nessun effetto sui tipi esistenti) ---
    # Schema delle proprieta' per-asset: {key: {label, unit, type, default,
    # min, max, required, options}}. Guida il form del costruttore asset nella
    # UI e la validazione degli scenari custom.
    asset_property_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Metadati UI: {icon, demand_label, demand_unit, area_label, demand_proxy_note}.
    # 'adt' e 'area' hanno semantica diversa per tipo (traffico/ore vento/...).
    ui_meta: Dict[str, Any] = field(default_factory=dict)
    # Sensibilita' agli hazard, default CALIBRABILI per tipo (override per-asset
    # via chiave 'seismic_sensitivity' nello scenario). Assente = 1.0.
    hazard_sensitivities: Dict[str, float] = field(default_factory=dict)
    # Vite utili medie di default per componente (anni). Lo scenario puo'
    # sempre sovrascrivere con '{comp}_vita_utile_anni'.
    component_default_lifetimes: Dict[str, float] = field(default_factory=dict)
    # Hook opzionali (None = comportamento invariato).
    revenue_condition_adapter: Optional[RevenueConditionAdapter] = None
    degradation_rate_adapter: Optional[DegradationRateAdapter] = None
    lifetime_adapter: Optional[LifetimeAdapter] = None

    def __post_init__(self):
        if self.fallback_component not in self.component_profiles:
            raise ValueError(
                f"AssetTypeSpec '{self.name}': fallback_component "
                f"'{self.fallback_component}' non presente nei profili."
            )
        unknown_structural = self.structural_components - set(self.component_profiles)
        if unknown_structural:
            raise ValueError(
                f"AssetTypeSpec '{self.name}': componenti strutturali ignoti {sorted(unknown_structural)}."
            )

    @property
    def component_names(self) -> Tuple[str, ...]:
        return tuple(self.component_profiles.keys())
