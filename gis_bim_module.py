"""Shim di compatibilita': il modulo e' stato spostato in infralab.ui.gis_bim.

Mantiene funzionanti i vecchi import top-level (script legacy, sessioni
salvate, paper_materials). L'intero modulo viene sostituito in sys.modules
con quello reale, cosi' anche il monkeypatching degli attributi funziona.
"""
import sys
import warnings

import infralab.ui.gis_bim as _mod

warnings.warn(
    "Il modulo top-level 'gis_bim_module' e' deprecato: usa 'infralab.ui.gis_bim'.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _mod
