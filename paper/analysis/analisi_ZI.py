"""Campagna ZI (control/NOTA-campagna-ZI.md): lettura fissata nella nota prima delle partite, regole I1-I2.

Solo gestore tipico; probabilita' di scoperta L1 dei difetti progressivi 0,2 e 0,8 (ZI) e 0,5 (Z-princ).
Differenze appaiate per seme D1-D0, N24-D0, D1-N24, bootstrap 95% (4000).
Uso: python analisi_ZI.py   ->  analisi_ZI.json, ZI_per_partita.csv
"""
import json

import numpy as np
import pandas as pd

import analisi_Z as AZ
from analisi_ZT import carica

AZ.rng = np.random.default_rng(20261107)
ESITI = ["idr_attesi", "S1", "crolli_attesi", "spesa", "floor_beta", "floor_prog", "prog_arrestati_soglia"]
CONFRONTI = [("D1", "D0"), ("N24", "D0"), ("D1", "N24")]


def main():
    bracci = {}
    for p, (camp, br) in {"0.2": ("ZI", "det02"), "0.5": ("Z", "princ"), "0.8": ("ZI", "det08")}.items():
        t, _ = carica(camp, br)
        t = t[t.gestore == "typical_owner"].copy()
        t["p_scoperta"] = p
        bracci[p] = t
    tutte = pd.concat(bracci.values(), ignore_index=True)
    tutte.to_csv("ZI_per_partita.csv", index=False)
    ris = {"medie": tutte.groupby(["p_scoperta", "legge"])[ESITI].mean().round(4).reset_index().to_dict("records"),
           "contrasti": {}}
    for p, t in bracci.items():
        for a, b in CONFRONTI:
            w = None
            cc = {}
            for m in ESITI:
                w = t.pivot_table(index="seme", columns="legge", values=m)
                dd = (w[a] - w[b]).to_numpy()
                lo, hi = AZ.boot_semi(dd)
                cc[m] = {"base": float(w[b].mean()), "diff": float(dd.mean()), "lo": lo, "hi": hi,
                         "positivi": int((dd > 0).sum()), "negativi": int((dd < 0).sum()), "semi": len(dd)}
            ris["contrasti"][f"{p}/{a}-{b}"] = cc
    c = ris["contrasti"]
    ris["regole"] = {
        "I1": {p: {k: c[f"{p}/{k}-D0"]["idr_attesi"]["negativi"] for k in ("D1", "N24")} for p in bracci},
        "I2": {p: {k: c[f"{p}/D1-N24"]["idr_attesi"][k] for k in ("diff", "lo", "hi", "negativi", "positivi")} for p in bracci},
    }
    ris["regole"]["I1_regge"] = all(v >= 9 for d in ris["regole"]["I1"].values() for v in d.values())
    ris["regole"]["I2_esclude_zero_ovunque"] = all(d["hi"] < 0 for d in ris["regole"]["I2"].values())
    json.dump(ris, open("analisi_ZI.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    print(tutte.groupby(["p_scoperta", "legge"])[ESITI].mean().round(3).to_string())
    for k, v in c.items():
        print(k)
        for m in ESITI:
            r = v[m]
            print(f"   {m:22s} base {r['base']:9.3f}  diff {r['diff']:8.3f} [{r['lo']:.3f}, {r['hi']:.3f}]  +{r['positivi']}/-{r['negativi']}/{r['semi']}")
    print(json.dumps(ris["regole"], indent=1))


if __name__ == "__main__":
    main()
