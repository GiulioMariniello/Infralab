"""Normative testabili: framework di codice selezionabili (target di affidabilità).

Selezionando la normativa (EN1990, fib_MC2010, AASHTO_LRFD, …) i target β
vengono dettati dal codice; 'custom' lascia i target manuali dei parametri.
Permette di valutare l'effetto della normativa con agenti neutri/ottimizzazione.
"""
from infralab.codes.base import CodeFramework  # noqa: F401
from infralab.codes.registry import (  # noqa: F401
    available_code_frameworks,
    get_code_framework,
    register_code_framework,
)

import infralab.codes.frameworks  # noqa: E402,F401  (registra i built-in)
