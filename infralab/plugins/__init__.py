"""Plugin utente: modelli di degrado e agenti scritti in Python.

L'utente definisce una funzione con un contratto input/output fisso
(`degrade(ctx)` per il degrado, `decide(obs)` per l'agente), la salva dal
laboratorio codice e il motore la esegue in una sandbox best-effort (builtins
ristretti, import in whitelist, timeout in validazione). I template documentano
il contratto e sono costruibili anche esternamente.
"""
from infralab.plugins.registry import (  # noqa: F401
    PLUGIN_KINDS,
    PluginError,
    delete_plugin,
    get_plugin,
    list_plugins,
    load_agent_decider,
    load_degradation_model,
    save_plugin,
)
from infralab.plugins.templates import TEMPLATES  # noqa: F401
