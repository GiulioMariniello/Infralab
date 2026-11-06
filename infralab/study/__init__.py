"""Apparato di studio umano: questionario post-partita e statistica sperimentale.

Serve gli esperimenti del Paper A (LLM / Uomo / Uomo+LLM / Euristico):
strumenti validati (NASA-TLX per il carico cognitivo, scala di fiducia
nell'automazione), raccolta del razionale decisionale, e confronto statistico
fra condizioni (intervalli bootstrap, effect size, test non parametrici).
"""
from infralab.study.questionnaire import (  # noqa: F401
    QUESTIONNAIRE_SCHEMA,
    score_questionnaire,
)
from infralab.study.stats import compare_conditions  # noqa: F401
