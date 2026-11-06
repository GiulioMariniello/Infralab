"""Registry delle normative disponibili."""
from typing import Dict, List

from infralab.codes.base import CodeFramework

_REGISTRY: Dict[str, CodeFramework] = {}


def register_code_framework(fw: CodeFramework, overwrite: bool = False) -> CodeFramework:
    if fw.name in _REGISTRY and not overwrite:
        raise ValueError(f"Normativa '{fw.name}' gia' registrata.")
    _REGISTRY[fw.name] = fw
    return fw


def get_code_framework(name: str) -> CodeFramework:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Normativa '{name}' non registrata. Disponibili: {sorted(_REGISTRY)}.") from None


def available_code_frameworks() -> List[str]:
    return list(_REGISTRY)
