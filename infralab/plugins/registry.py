"""Persistenza e caricamento dei plugin utente (agenti, modelli di degrado).

I plugin sono file .py in `data/custom_plugins/{agents,degradation}/`. Il nome
del plugin è lo slug del file. Il caricamento passa sempre dalla validazione
(loader) prima di restituire la callable, con una piccola cache in memoria.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Dict, List

from infralab.plugins.loader import (
    PluginError,
    load_entrypoint,
    validate_agent,
    validate_degradation,
)

# kind -> (sottocartella, entrypoint, validatore)
PLUGIN_KINDS: Dict[str, Dict] = {
    "agent": {"subdir": "agents", "entrypoint": "decide", "validate": validate_agent},
    "degradation": {"subdir": "degradation", "entrypoint": "degrade", "validate": validate_degradation},
}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")

# Cache: (kind, name, mtime) -> callable
_CACHE: Dict[tuple, Callable] = {}


def _plugins_root() -> str:
    # Import differito: il DATA_DIR di persistence è monkeypatchato nei test.
    import infralab.persistence as persistence

    return os.path.join(persistence.DATA_DIR, "custom_plugins")


def _kind_dir(kind: str) -> str:
    if kind not in PLUGIN_KINDS:
        raise PluginError(f"Tipo di plugin sconosciuto: {kind!r} (validi: {sorted(PLUGIN_KINDS)}).")
    return os.path.join(_plugins_root(), PLUGIN_KINDS[kind]["subdir"])


def _path(kind: str, name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise PluginError(f"Nome plugin non valido: {name!r} (lettere/numeri/'-'/'_', max 48).")
    return os.path.join(_kind_dir(kind), f"{name}.py")


def list_plugins(kind: str) -> List[Dict[str, str]]:
    """Elenco dei plugin salvati per un tipo: [{name, entrypoint}]."""
    d = _kind_dir(kind)
    if not os.path.isdir(d):
        return []
    out = []
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".py"):
            out.append({"name": fname[:-3], "entrypoint": PLUGIN_KINDS[kind]["entrypoint"]})
    return out


def get_plugin(kind: str, name: str) -> Dict[str, str]:
    """Sorgente di un plugin. Solleva PluginError se assente."""
    path = _path(kind, name)
    if not os.path.exists(path):
        raise PluginError(f"Plugin '{name}' ({kind}) non trovato.")
    with open(path, "r", encoding="utf-8") as f:
        return {"name": name, "kind": kind, "source": f.read()}


def save_plugin(kind: str, name: str, source: str) -> Dict[str, str]:
    """Valida e salva un plugin (upsert). Solleva PluginError se non valido."""
    path = _path(kind, name)
    ok, message = PLUGIN_KINDS[kind]["validate"](source)
    if not ok:
        raise PluginError(message)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    _CACHE.clear()
    return {"name": name, "kind": kind, "message": message}


def delete_plugin(kind: str, name: str) -> bool:
    path = _path(kind, name)
    if os.path.exists(path):
        os.remove(path)
        _CACHE.clear()
        return True
    return False


def _load_callable(kind: str, name: str) -> Callable:
    path = _path(kind, name)
    if not os.path.exists(path):
        raise PluginError(f"Plugin '{name}' ({kind}) non trovato.")
    key = (kind, name, os.path.getmtime(path))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    fn = load_entrypoint(source, PLUGIN_KINDS[kind]["entrypoint"])
    _CACHE[key] = fn
    return fn


def load_degradation_model(name: str) -> Callable:
    """Callable `degrade(ctx) -> float` del modello di degrado salvato."""
    return _load_callable("degradation", name)


def load_agent_decider(name: str) -> Callable:
    """Callable `decide(obs) -> dict` dell'agente salvato."""
    return _load_callable("agent", name)
