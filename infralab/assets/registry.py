"""Registry globale degli asset type disponibili."""
from typing import Dict, List

from infralab.assets.base import AssetTypeSpec

_REGISTRY: Dict[str, AssetTypeSpec] = {}


def register_asset_type(spec: AssetTypeSpec, overwrite: bool = False) -> AssetTypeSpec:
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(f"Asset type '{spec.name}' gia' registrato.")
    _REGISTRY[spec.name] = spec
    return spec


def get_asset_type(name: str) -> AssetTypeSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Asset type '{name}' non registrato. Disponibili: {sorted(_REGISTRY)}."
        ) from None


def available_asset_types() -> List[str]:
    return sorted(_REGISTRY)
