"""Campagna ZQ (control/NOTA-campagna-ZQ.md): lettura fissata nella nota prima delle partite, regole Q1-Q4.

Gestore tipico, intervallo d'ispezione fisso da 6 a 360 mesi; riferimenti D0, N24, D1 da Z-princ (stessi semi).
Differenze appaiate per seme rispetto a D0, bootstrap 95% (4000).
Uso: python analisi_ZQ.py  ->  analisi_ZQ.json, ZQ_per_partita.csv, ../../figures/fig_interval_study.(pdf|png)
"""
import json
import os
import re

import numpy as np
import pandas as pd

import analisi_Z as AZ


def norma(path):
    m = re.search(r"/(D\d|N24|Q\d+)_", str(path).replace("\\", "/"))
    return m.group(1) if m else "?"


AZ.norma = norma
from analisi_ZT import carica  # noqa: E402

AZ.rng = np.random.default_rng(20261108)
ESITI = ["S1", "crolli_attesi", "idr_attesi", "anni_sotto_302", "sotto_ponti_302", "spesa", "isp_imposte",
         "floor_beta", "floor_prog"]
TRIM = [2, 4, 8, 12, 16, 20, 40, 80, 120]
# per Q2: verso in cui l'esito migliora (S1 su, gli altri giu')
MEGLIO = {"S1": 1, "crolli_attesi": -1, "idr_attesi": -1, "anni_sotto_302": -1}


def main():
    q, _ = carica("ZQ", "int")
    z, _ = carica("Z", "princ")
    z = z[z.gestore == "typical_owner"]
    t = pd.concat([q, z], ignore_index=True)
    t.to_csv("ZQ_per_partita.csv", index=False)
    w = {m: t.pivot_table(index="seme", columns="legge", values=m) for m in ESITI}
    ris = {"partite_ZQ": int(len(q)), "livelli": {}, "diff_D0": {}}
    for leg in [f"Q{k}" for k in TRIM] + ["N24", "D1", "D0"]:
        ris["livelli"][leg] = {m: float(w[m][leg].mean()) for m in ESITI}
        if leg == "D0":
            continue
        ris["diff_D0"][leg] = {}
        for m in ESITI:
            dd = (w[m][leg] - w[m]["D0"]).to_numpy()
            lo, hi = AZ.boot_semi(dd)
            ris["diff_D0"][leg][m] = {"diff": float(dd.mean()), "lo": lo, "hi": hi,
                                      "positivi": int((dd > 0).sum()), "negativi": int((dd < 0).sum())}
    d = ris["diff_D0"]
    # Q2: passando a un intervallo piu' lungo il guadagno non migliora oltre la sovrapposizione degli intervalli
    q2 = {}
    for m, s in MEGLIO.items():
        viol = []
        for a, b in zip(TRIM[:-1], TRIM[1:]):
            ra, rb = d[f"Q{a}"][m], d[f"Q{b}"][m]
            # rb migliore di ra senza sovrapposizione
            if s > 0 and rb["lo"] > ra["hi"] or s < 0 and rb["hi"] < ra["lo"]:
                viol.append(f"{3 * a}->{3 * b}")
        mono_medie = all((d[f"Q{b}"][m]["diff"] - d[f"Q{a}"][m]["diff"]) * s <= 1e-12 for a, b in zip(TRIM[:-1], TRIM[1:]))
        q2[m] = {"regge": not viol, "risalite_significative": viol, "medie_monotone": mono_medie}
    # Q3: il piu' lungo intervallo che esclude lo zero
    q3 = {}
    for m in ("S1", "crolli_attesi", "idr_attesi"):
        ok = [3 * k for k in TRIM if d[f"Q{k}"][m]["lo"] > 0 or d[f"Q{k}"][m]["hi"] < 0]
        q3[m] = max(ok) if ok else None
    dd = (w["S1"]["Q8"] - w["S1"]["N24"]).to_numpy()
    lo, hi = AZ.boot_semi(dd)
    ris["regole"] = {"Q2": q2, "Q3_mesi": q3,
                     "Q4": {"diff": float(dd.mean()), "lo": lo, "hi": hi, "regge": abs(float(dd.mean())) <= 0.01}}
    json.dump(ris, open("analisi_ZQ.json", "w"), indent=1)

    print(f"{'norma':6s} " + " ".join(f"{m:>22s}" for m in ESITI))
    for leg, v in ris["livelli"].items():
        cell = []
        for m in ESITI:
            if leg == "D0":
                cell.append(f"{v[m]:10.3f}{'':12s}")
            else:
                r = d[leg][m]
                cell.append(f"{v[m]:9.3f} {r['diff']:+8.3f}{'*' if (r['lo'] > 0 or r['hi'] < 0) else ' ':3s}")
        print(f"{leg:6s} " + " ".join(f"{c:>22s}" for c in cell))
    print(json.dumps(ris["regole"], indent=1))
    for leg in [f"Q{k}" for k in TRIM]:
        print(leg, {m: (round(d[leg][m]["lo"], 3), round(d[leg][m]["hi"], 3)) for m in ("S1", "crolli_attesi", "idr_attesi", "anni_sotto_302")})


if __name__ == "__main__":
    main()
