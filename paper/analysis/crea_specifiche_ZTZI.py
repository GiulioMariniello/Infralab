"""Specifiche delle campagne di sensibilita' ZT e ZI (control/NOTA-campagna-ZT.md, NOTA-campagna-ZI.md).

ZT: Z-princ con capacity_condition_sensitivity 0,6 e reliability_resistance_scale 1,30, norme D0 e D1, tre gestori.
ZI: Z-princ solo gestore tipico, norme D0/D1/N24, progressive_hydraulic_detect_factor_l1 0,2 e 0,8.
Tutto il resto identico a Z-princ (stessi semi, stesso motore). Uso: python crea_specifiche_ZTZI.py
"""
import copy
import os

import yaml

QUI = os.path.dirname(os.path.abspath(__file__))
Z = yaml.safe_load(open(os.path.join(QUI, "Z-princ.yaml"), encoding="utf-8"))
assert Z["grid"]["params.capacity_condition_sensitivity"] == [0.4]
assert Z["grid"]["params.reliability_resistance_scale"] == [1.25]
regimi = Z["grid"]["params.regulatory_regime"]
D0 = [r for r in regimi if r.endswith("D0_nessun_obbligo.yaml")]
D1 = [r for r in regimi if r.endswith("D1_periodica.yaml")]
assert len(D0) == 1 and len(D1) == 1 and len(regimi) == 3


def scrivi(s, nome):
    s["name"] = nome
    s["outputs"]["dir"] = os.path.join(QUI, "runs", nome).replace("\\", "/")
    with open(os.path.join(QUI, nome + ".yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(s, f, sort_keys=False, allow_unicode=True)
    print(nome, len(s["agents"]), "gestori,", len(s["grid"]["params.regulatory_regime"]), "norme")


s = copy.deepcopy(Z)
s["grid"]["params.capacity_condition_sensitivity"] = [0.6]
s["grid"]["params.reliability_resistance_scale"] = [1.30]
s["grid"]["params.regulatory_regime"] = D0 + D1
scrivi(s, "ZT-princ")

for p, tag in ((0.2, "02"), (0.8, "08")):
    s = copy.deepcopy(Z)
    s["agents"] = [a for a in s["agents"] if a["id"] == "typical_owner"]
    assert len(s["agents"]) == 1
    s["grid"]["params.progressive_hydraulic_detect_factor_l1"] = [p]
    scrivi(s, f"ZI-det{tag}")
