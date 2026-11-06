"""Shim di compatibilita': il modulo e' stato spostato in infralab.core.reliability.

Mantiene funzionanti i vecchi import top-level (script legacy, sessioni
salvate, paper_materials). L'intero modulo viene sostituito in sys.modules
con quello reale, cosi' anche il monkeypatching degli attributi funziona.
"""
import sys
import warnings

import infralab.core.reliability as _mod

warnings.warn(
    "Il modulo top-level 'reliability_model' e' deprecato: usa 'infralab.core.reliability'.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _mod
