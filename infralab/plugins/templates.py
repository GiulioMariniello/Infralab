"""Template Python dei plugin: definiscono il contratto input/output.

Sono il punto di partenza mostrato nel laboratorio codice e possono essere
scritti/estesi anche esternamente: basta rispettare la firma della funzione.
"""

DEGRADATION_TEMPLATE = '''\
# ── Modello di degrado personalizzato InfraLab ────────────────────────────────
# Definisci `degrade(ctx)` che restituisce la NUOVA condizione del componente
# per il trimestre corrente: un float in [0, 1] (1 = come nuovo, 0 = fuori uso).
# Viene chiamata per OGNI componente a OGNI trimestre; sostituisce il modello
# Weibull/gamma di default. Il motore avanza già l'età efficace (il "clock")
# prima di chiamarti.
#
# INPUT — ctx (dict, sola lettura):
#   ctx["condition"]              float  condizione attuale (prima dello step)
#   ctx["previous_age_years"]     float  età efficace prima dello step
#   ctx["effective_age_years"]    float  età efficace dopo l'avanzamento del clock
#   ctx["age_increment_years"]    float  avanzamento del clock in questo trimestre
#   ctx["vita_utile_anni"]        float  vita utile del componente
#   ctx["shape"]                  float  esponente di forma (k) del componente
#   ctx["scale"]                  float  scala di danno del componente
#   ctx["deterioration_exponent"] float  parametro globale della simulazione
#   ctx["accumulated_damage"]     float  danno D accumulato (condizione = exp(-D))
#   ctx["component_name"]         str
#   ctx["quarter"]                int
#   ctx["rand"]()                 -> float in [0,1)  (stocastico ma riproducibile col seed)
#
# OUTPUT: float — la nuova condizione in [0, 1] (il motore la limita comunque).
# Import consentiti: math, statistics, random, itertools, functools, collections, heapq, json.

import math


def degrade(ctx):
    L = ctx["vita_utile_anni"]
    if L <= 0 or ctx["effective_age_years"] <= 0:
        return 1.0
    # Traiettoria Weibull di base: D = scale * (t/L)^shape, condizione = exp(-D).
    age_ratio = ctx["effective_age_years"] / L
    damage = ctx["scale"] * age_ratio ** ctx["shape"]
    # Esempio: piccolo shock casuale di manutenzione mancata (togli per il deterministico).
    damage *= 1.0 + 0.10 * ctx["rand"]()
    return math.exp(-damage)
'''


AGENT_TEMPLATE = '''\
# ── Agente personalizzato InfraLab ────────────────────────────────────────────
# Definisci `decide(obs)` che restituisce il piano di manutenzione del trimestre:
# un dict {ID_ASSET (str): NOME_AZIONE (str)}. Gli asset non citati restano fermi.
# Viene chiamata una volta per trimestre.
#
# INPUT — obs (dict, sola lettura):
#   obs["quarter"]         int    trimestre corrente
#   obs["budget"]          float  budget disponibile dell'agente
#   obs["emissions_tco2"]  float  emissioni cumulate dell'agente
#   obs["assets"]          list di dict, uno per asset gestito:
#       a["id"]               int
#       a["name"]             str
#       a["condition_min"]    float       condizione minima tra i componenti (0..1)
#       a["beta"]             float|None  indice di affidabilità
#       a["annual_pf"]        float|None  probabilità di rottura annua
#       a["health_index"]     float       salute media pesata (0..1)
#       a["net_asset_value"]  float
#       a["components"]       dict {nome: condizione}
#       a["feasible_actions"] dict {nome_azione: costo}  azioni disponibili ORA
#
# OUTPUT: dict {str(id_asset): nome_azione}. Usa SOLO azioni presenti in feasible_actions.
# Import consentiti: math, statistics, random, itertools, functools, collections, heapq, json.


def decide(obs):
    plan = {}
    budget = obs["budget"]
    # Ordina per criticità (condizione peggiore prima) e spendi finché c'è budget.
    for a in sorted(obs["assets"], key=lambda x: x["condition_min"]):
        if a["condition_min"] < 0.55 and a["feasible_actions"]:
            cheapest = min(a["feasible_actions"], key=a["feasible_actions"].get)
            cost = a["feasible_actions"][cheapest]
            if cost <= budget:
                plan[str(a["id"])] = cheapest
                budget -= cost
    return plan
'''


TEMPLATES = {"agent": AGENT_TEMPLATE, "degradation": DEGRADATION_TEMPLATE}
