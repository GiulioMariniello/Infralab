"""Specifiche della campagna Z (control/NOTA-campagna-Z.md): le partite di Y con la taratura v3 e il limite operativo.

taratura_beta_nbi_v3.py (phi_c tolto dal fattore di valutazione, ancoraggio operativo 2,5 su 5 anni) ->
capacity_condition_sensitivity 0,4 e reliability_resistance_scale 1,25. Soglia etica sul beta annuo stimato 3,02
(= 2,5 su 5 anni, livello operativo di NCHRP 454). Nelle partite principali (princ, off) si aggiunge la norma N24
(ispezione ogni 24 mesi, 12 per voto <= 3, sul modello NBIS 2022). Tutto il resto identico a Y (stessi semi, giocatori).
Uso: python crea_specifiche_Z.py
"""
import os

import yaml

QUI = os.path.dirname(os.path.abspath(__file__))
NUOVA = {"params.capacity_condition_sensitivity": [0.4], "params.reliability_resistance_scale": [1.25],
         "params.ethical_floor_beta": [3.02]}
N24 = os.path.join(QUI, "leggi", "N24_nbis.yaml").replace("\\", "/")

for braccio in ("princ", "off", "corta", "markov", "potenza", "gamma"):
    s = yaml.safe_load(open(os.path.join(QUI, f"Y-{braccio}.yaml"), encoding="utf-8"))
    assert s["grid"]["params.capacity_condition_sensitivity"] == [0.6] and s["grid"]["params.ethical_floor_beta"] == [3.66]
    s["grid"].update(NUOVA)
    if braccio in ("princ", "off"):
        s["grid"]["params.regulatory_regime"] = s["grid"]["params.regulatory_regime"] + [N24]
    s["name"] = f"Z-{braccio}"
    s["outputs"]["dir"] = os.path.join(QUI, "runs", f"Z-{braccio}").replace("\\", "/")
    with open(os.path.join(QUI, f"Z-{braccio}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(s, f, sort_keys=False, allow_unicode=True)
    print(f"Z-{braccio}.yaml", s["grid"]["params.regulatory_regime"])
