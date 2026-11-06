"""Caricamento ed esecuzione sicura (best-effort) del codice-plugin utente.

Riusa i builtins ristretti e la whitelist di import della sandbox agentica.
Il codice viene eseguito con `__builtins__` limitati (niente open/exec/eval/
import fuori whitelist). La validazione esegue una chiamata di prova con
timeout per intercettare errori e loop evidenti; a runtime il degrado viene
chiamato molte volte e i call-site gestiscono le eccezioni con fallback.
"""
from __future__ import annotations

import builtins as _builtins
import threading
from typing import Any, Callable, Dict, Tuple

from infralab.agents.agentic import ALLOWED_IMPORTS, _SAFE_BUILTIN_NAMES, _safe_import

VALIDATION_TIMEOUT_S = 5.0


class PluginError(ValueError):
    """Codice-plugin non valido (sintassi, entrypoint mancante, errore a runtime)."""


def _safe_globals() -> Dict[str, Any]:
    safe = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES}
    safe["__import__"] = _safe_import
    # Utili e innocui per la modellazione numerica.
    for extra in ("abs", "round", "min", "max", "sum", "pow"):
        safe.setdefault(extra, getattr(_builtins, extra))
    return {"__builtins__": safe}


def load_entrypoint(source: str, entrypoint: str) -> Callable:
    """Compila ed esegue `source` in una sandbox e ritorna la funzione `entrypoint`.

    Solleva PluginError su sintassi non valida, errore di esecuzione del modulo
    o entrypoint mancante/non chiamabile.
    """
    if not source or not source.strip():
        raise PluginError("Codice vuoto.")
    try:
        compiled = compile(source, f"<plugin:{entrypoint}>", "exec")
    except SyntaxError as e:
        raise PluginError(f"Errore di sintassi (riga {e.lineno}): {e.msg}") from None
    namespace: Dict[str, Any] = _safe_globals()
    try:
        exec(compiled, namespace)  # noqa: S102 - sandboxata
    except Exception as e:  # noqa: BLE001
        raise PluginError(f"Errore durante il caricamento del modulo: {type(e).__name__}: {e}") from None
    fn = namespace.get(entrypoint)
    if not callable(fn):
        raise PluginError(f"Il codice deve definire una funzione `{entrypoint}(...)`.")
    return fn


def call_with_timeout(fn: Callable, arg: Any, timeout_s: float) -> Any:
    """Esegue fn(arg) in un thread con timeout. Solleva PluginError su timeout/errore."""
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn(arg)
        except Exception as e:  # noqa: BLE001
            box["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise PluginError(f"Timeout: la funzione ha superato {timeout_s}s (loop infinito?).")
    if "error" in box:
        raise PluginError(f"Errore di esecuzione: {box['error']}")
    return box.get("result")


# -- contesti di prova per la validazione ------------------------------------

def _sample_degradation_ctx() -> Dict[str, Any]:
    return {
        "condition": 0.80, "previous_age_years": 10.0, "effective_age_years": 10.25,
        "age_increment_years": 0.25, "vita_utile_anni": 100.0, "shape": 2.0, "scale": 1.0,
        "deterioration_exponent": 2.0, "accumulated_damage": 0.223, "component_name": "prova",
        "quarter": 1, "rand": lambda: 0.5,
    }


def _sample_agent_obs() -> Dict[str, Any]:
    return {
        "quarter": 1, "budget": 500000.0, "emissions_tco2": 0.0,
        "assets": [{
            "id": 1, "name": "Asset di prova", "condition_min": 0.42, "beta": 3.1,
            "annual_pf": 1e-4, "health_index": 0.6, "net_asset_value": 1_000_000.0,
            "components": {"a": 0.42, "b": 0.7},
            "feasible_actions": {"ispezione": 1000.0, "ripara": 50000.0},
        }],
    }


def validate_degradation(source: str) -> Tuple[bool, str]:
    """Carica e prova un modello di degrado. Ritorna (ok, messaggio)."""
    try:
        fn = load_entrypoint(source, "degrade")
        result = call_with_timeout(fn, _sample_degradation_ctx(), VALIDATION_TIMEOUT_S)
    except PluginError as e:
        return False, str(e)
    try:
        v = float(result)
    except (TypeError, ValueError):
        return False, f"`degrade` deve restituire un numero (condizione), non {type(result).__name__}."
    if not (0.0 <= v <= 1.5):  # tolleranza; il motore limita a [0,1]
        return False, f"La condizione restituita ({v:.3f}) è fuori da un intervallo plausibile [0,1]."
    return True, "Modello di degrado valido."


def validate_agent(source: str) -> Tuple[bool, str]:
    """Carica e prova un agente. Ritorna (ok, messaggio)."""
    try:
        fn = load_entrypoint(source, "decide")
        result = call_with_timeout(fn, _sample_agent_obs(), VALIDATION_TIMEOUT_S)
    except PluginError as e:
        return False, str(e)
    if not isinstance(result, dict):
        return False, f"`decide` deve restituire un dict {{id_asset: azione}}, non {type(result).__name__}."
    for k, val in result.items():
        if not isinstance(val, str):
            return False, f"Il piano deve mappare id -> nome azione (stringa); '{k}' -> {type(val).__name__}."
    return True, "Agente valido."
