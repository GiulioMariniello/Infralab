"""Campagna X2 (control/NOTA-campagna-X2.md): lettura fissata nella nota prima delle partite.
Uso: python analisi_X2.py  ->  analisi_X2.json"""
import glob
import json

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261104)
MIS = ["scarto_medio", "stima_mediana_vero_sotto", "vero_mediano_vero_sotto", "controlli_vero_sotto", "rilevazione",
       "falsi_allarmi", "auc", "soglia_scattata", "S1", "sotto_ponti", "crolli_attesi", "spesa", "q_chiusi"]


def boot(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None, None
    m = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


t = pd.DataFrame([json.load(open(f)) for f in glob.glob("blocks/X2/*/*.json") if not f.endswith("motore.json")])
ris = {"partite": int(len(t)), "per_braccio": t.groupby("braccio").size().to_dict(), "medie": {}, "dal_base": {}}
for (b, r), x in t.groupby(["braccio", "regime"]):
    ris["medie"][f"{b}/{r}"] = {m: (None if x[m].isna().all() else float(x[m].astype(float).mean())) for m in MIS}

# controllo: il braccio base riproduce il passivo di Z-princ
z = pd.read_csv("Z_per_partita.csv")
z = z[(z.braccio == "princ") & (z.gestore == "do_nothing") & z.legge.isin(["D0", "D1"])].set_index(["legge", "seme"]).S1
bb = t[t.braccio == "base"].set_index(["regime", "replica"]).S1
comune = bb.index.intersection(z.index)
ris["base_uguale_a_Z"] = {"confrontate": int(len(comune)), "scarto_max": float((bb[comune] - z[comune]).abs().max())}

# differenze appaiate per seme dal braccio base, e quota dello scarto tolta
base = t[t.braccio == "base"].set_index(["regime", "replica"])
for b in sorted(set(t.braccio) - {"base"}):
    x = t[t.braccio == b].set_index(["regime", "replica"])
    for r in ("D0", "D1"):
        idx = [i for i in x.index if i[0] == r and i in base.index]
        out = {}
        for m in MIS:
            d = (x.loc[idx, m].astype(float) - base.loc[idx, m].astype(float)).to_numpy()
            lo, hi = boot(d)
            out[m] = {"diff": float(np.nanmean(d)) if not np.isnan(d).all() else None, "lo": lo, "hi": hi}
        sb = base.loc[idx, "scarto_medio"].mean()
        out["quota_scarto_tolta"] = float((sb - x.loc[idx, "scarto_medio"].mean()) / sb)
        ris["dal_base"][f"{b}/{r}"] = out

# domanda "con l'obbligo la stima e' piu' ottimista dove conta" (regola 9 su 10)
def d1_meno_d0(b):
    x = t[t.braccio == b].pivot_table(index="replica", columns="regime", values="stima_mediana_vero_sotto")
    return (x["D1"] - x["D0"]).dropna()

db, dr = d1_meno_d0("base"), d1_meno_d0("rif")
comuni = db.index.intersection(dr.index)
ris["regola_ottimismo"] = {"semi_base_D1_sopra_D0": int((db > 0).sum()), "semi_base": int(len(db)),
                           "semi_rif_riduce": int((dr[comuni] < db[comuni]).sum()), "semi_confronto": int(len(comuni)),
                           "diff_base_media": float(db.mean()), "diff_rif_media": float(dr.mean())}
ris["regola_ottimismo"]["regge"] = (ris["regola_ottimismo"]["semi_base_D1_sopra_D0"] >= 9
                                    and ris["regola_ottimismo"]["semi_rif_riduce"] >= 9)
json.dump(ris, open("analisi_X2.json", "w"), indent=1)

pd.set_option("display.width", 250)
print("partite", ris["partite"], ris["per_braccio"])
print("base = Z-princ:", ris["base_uguale_a_Z"])
print(pd.DataFrame(ris["medie"]).T[["scarto_medio", "stima_mediana_vero_sotto", "vero_mediano_vero_sotto", "controlli_vero_sotto",
                                    "rilevazione", "falsi_allarmi", "auc", "soglia_scattata", "S1", "sotto_ponti"]].round(3).to_string())
print("quota dello scarto tolta:", {k: round(v["quota_scarto_tolta"], 3) for k, v in ris["dal_base"].items()})
print("regola ottimismo:", ris["regola_ottimismo"])
