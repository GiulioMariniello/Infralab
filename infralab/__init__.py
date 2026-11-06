"""InfraLab — laboratorio virtuale per la gestione di asset infrastrutturali.

Pacchetto nato dal refactoring di BridgeManager: motore di affidabilità
FOSM/Bayesiano generico, simulatore a turni con degrado stocastico, agenti
decisionali (euristici, ottimizzazione, LLM) e UI Streamlit per esperimenti
con soggetti umani e campagne computazionali.
"""

__version__ = "1.0.0"

# API pubblica della libreria (infralab.api): import pigri, cosi' `import infralab` resta leggero.
_API = ("make", "run", "actions", "compare_laws", "scenarios", "load_scenario", "laws", "describe_law", "configurations",
        "configuration", "degradation_laws", "resolve_degradation", "policies", "basic_policies", "policy_config", "child_seeds", "Env")


def __getattr__(name):
    if name in _API:
        from infralab import api
        return getattr(api, name)
    raise AttributeError(f"module 'infralab' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_API))
