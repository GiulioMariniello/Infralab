"""Campagna ZQ (control/NOTA-campagna-ZQ.md): norme con intervallo d'ispezione fisso e specifica ZQ-int.yaml.

Ogni norma e' N24 con lo stesso intervallo per tutte le classi di condizione. Specifica = Z-princ, solo gestore
tipico, le nove norme. Uso: python crea_specifiche_ZQ.py
"""
import copy
import os

import yaml

QUI = os.path.dirname(os.path.abspath(__file__))
TRIMESTRI = [2, 4, 8, 12, 16, 20, 40, 80, 120]
N24 = yaml.safe_load(open(os.path.join(QUI, "leggi", "N24_nbis.yaml"), encoding="utf-8"))
Z = yaml.safe_load(open(os.path.join(QUI, "Z-princ.yaml"), encoding="utf-8"))

file_norme = []
for q in TRIMESTRI:
    n = copy.deepcopy(N24)
    n["id"] = f"regbench_Q{q}_fissa"
    n["display_name"] = f"Ispezione di routine ogni {3 * q} mesi per ogni ponte, qualunque sia la condizione"
    n["source"] = "infralab-regbench, campagna ZQ (control/NOTA-campagna-ZQ.md): N24 con intervallo unico"
    for k in list(n["cda_config"]):
        if k.endswith("_due_quarters"):
            n["cda_config"][k] = q
    p = os.path.join(QUI, "leggi", f"Q{q}_fissa.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(n, f, sort_keys=False, allow_unicode=True)
    file_norme.append(p.replace("\\", "/"))

s = copy.deepcopy(Z)
s["name"] = "ZQ-int"
s["agents"] = [a for a in s["agents"] if a["id"] == "typical_owner"]
assert len(s["agents"]) == 1
s["grid"]["params.regulatory_regime"] = file_norme
s["outputs"]["dir"] = os.path.join(QUI, "runs", "ZQ-int").replace("\\", "/")
with open(os.path.join(QUI, "ZQ-int.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(s, f, sort_keys=False, allow_unicode=True)
print("ZQ-int:", len(file_norme), "norme x", s["replicates"], "semi")
