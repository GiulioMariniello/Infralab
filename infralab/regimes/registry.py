"""Registro dei regimi normativi: built-in (YAML nel package) + caricati dall'utente."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from infralab.regimes.base import NULL_REGIME, RegulatoryRegime, regime_from_dict

_REGISTRY: Dict[str, RegulatoryRegime] = {NULL_REGIME.id: NULL_REGIME}
_BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "builtin")
_BUILTIN_LOADED = False


def _load_file(path: str) -> RegulatoryRegime:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return regime_from_dict(data)


def _ensure_builtin():
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True
    if not os.path.isdir(_BUILTIN_DIR):
        return
    for fn in sorted(os.listdir(_BUILTIN_DIR)):
        if fn.lower().endswith((".yaml", ".yml", ".json")):
            regime = _load_file(os.path.join(_BUILTIN_DIR, fn))
            _REGISTRY.setdefault(regime.id, regime)


def register_regime(regime: RegulatoryRegime, overwrite: bool = False) -> RegulatoryRegime:
    _ensure_builtin()
    if regime.id in _REGISTRY and not overwrite:
        raise ValueError(f"Regime '{regime.id}' gia' registrato.")
    _REGISTRY[regime.id] = regime
    return regime


def load_regime_file(path: str, overwrite: bool = True) -> RegulatoryRegime:
    """Carica un regime da YAML/JSON e lo registra (per leggi scritte dall'utente)."""
    return register_regime(_load_file(path), overwrite=overwrite)


def get_regime(name: Optional[str]) -> RegulatoryRegime:
    """Risolve un regime per id, per path di file (.yaml/.json) o None -> regime nullo."""
    _ensure_builtin()
    if not name or str(name).lower() in ("none", "null", ""):
        return NULL_REGIME
    name = str(name)
    if name in _REGISTRY:
        return _REGISTRY[name]
    if os.path.isfile(name):
        return load_regime_file(name)
    raise KeyError(f"Regime '{name}' non registrato. Disponibili: {sorted(_REGISTRY)} (o path a YAML/JSON).")


def available_regimes() -> List[str]:
    _ensure_builtin()
    return sorted(_REGISTRY)
