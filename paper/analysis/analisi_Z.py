"""Campagna Z (control/NOTA-campagna-Z.md): lettura fissata nella nota prima delle partite.

Per ogni braccio (princ = soglia accesa, off = soglia spenta, markov/potenza/gamma = altre leggi di degrado) e gestore:
differenze appaiate D1 - D0, N24 - D0, D1 - N24 su S1, crolli attesi, ponti sotto il limite almeno una volta,
anni-ponte sotto il limite, spesa, costo utenti, quota chiusa, interventi della soglia, ispezioni imposte.
Limite 3,02 (beta annuo vero), sensibilita' a 3,30 e 3,66. Intervalli al 95%: bootstrap sui semi e bootstrap a due
vie su ponti e semi (stessi ponti e semi per le due norme confrontate).
Uso: python analisi_Z.py   ->  analisi_Z.json, Z_per_partita.csv
"""
import glob
import json
import os
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

LIMITI = (3.02, 3.30, 3.66)
NB = 4000
M = 1e6
rng = np.random.default_rng(20261103)
PHI = np.vectorize(NormalDist().cdf)
BRACCI = ["princ", "off", "markov", "potenza", "gamma", "corta"]
CONFRONTI = [("D1", "D0"), ("N24", "D0"), ("D1", "N24")]
# esiti per partita; quelli per ponte hanno anche la matrice ponte x seme per il bootstrap a due vie
ESITI = ["S1", "crolli_attesi", "crolli", "sotto_ponti_302", "anni_sotto_302", "sotto_ponti_330", "anni_sotto_330",
         "sotto_ponti_366", "anni_sotto_366", "spesa", "utenti", "q_chiusi", "floor_interv", "isp_imposte"]
PER_PONTE = {"S1": "min_beta", "crolli_attesi": "pf", "sotto_ponti_302": "sotto_302", "anni_sotto_302": "anni_302"}


def norma(path):
    m = re.search(r"/(D\d|N24)_", str(path).replace("\\", "/"))
    return m.group(1) if m else "?"


def carica(braccio):
    righe, ponti = [], {}
    for bd in sorted(glob.glob(f"blocks/Z/{braccio}/b*")):
        if not os.path.exists(os.path.join(bd, "DONE.json")):
            continue
        r = pd.read_csv(os.path.join(bd, "runs.csv"))
        for x in r.itertuples():
            rid = int(x.run_id)
            leg = norma(r.loc[x.Index, "params.regulatory_regime"])
            d = pd.read_parquet(glob.glob(os.path.join(bd, "perbridge", f"*_id{rid:03d}_beta.parquet"))[0],
                                columns=["quarter", "bridge", "beta_annual"])
            d = d[d.quarter >= 1].copy()
            a = pd.read_parquet(glob.glob(os.path.join(bd, "perbridge", f"*_id{rid:03d}_annual.parquet"))[0], columns=["year", "factor"])
            a = a[a.year >= 1]
            d["pf"] = 1.0 - (1.0 - PHI(-d.beta_annual.to_numpy())) ** 0.25
            g = d.groupby("bridge")
            pb = pd.DataFrame({"min_beta": g.beta_annual.min(), "pf": g.pf.sum()})
            riga = {"braccio": braccio, "gestore": x.agent_id, "legge": leg, "seme": int(x.replicate),
                    "S1": float(pb.min_beta.mean()), "crolli_attesi": float(pb.pf.sum()), "crolli": float(x.final_failures_total),
                    "spesa": x.final_spent_total / M, "utenti": x.final_detour_user_cost_total / M,
                    "q_chiusi": float((a.factor <= 0).mean()),
                    "floor_interv": float(getattr(x, "final_floor_interventions", 0) or 0),
                    "isp_imposte": float(getattr(x, "final_regime_forced_actions", 0) or 0)}
            for L in LIMITI:
                k = f"{int(round(L * 100))}"
                sotto = d.beta_annual < L
                riga[f"sotto_ponti_{k}"] = int((pb.min_beta < L).sum())
                riga[f"anni_sotto_{k}"] = float(sotto.sum() / 4.0)
                if k == "302":
                    pb["sotto_302"] = (pb.min_beta < L).astype(float)
                    pb["anni_302"] = sotto.groupby(d.bridge).sum() / 4.0
            righe.append(riga)
            ponti[(x.agent_id, leg, int(x.replicate))] = pb.sort_index()
    return pd.DataFrame(righe), ponti


def boot_semi(v):
    idx = rng.integers(0, len(v), size=(NB, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_due_vie(D, somma):
    """D: matrice semi x ponti delle differenze per ponte. Ricampiona semi e ponti; somma=True per esiti che sono
    somme sui ponti (crolli attesi, ponti sotto, anni-ponte), altrimenti media (S1)."""
    ns, nb = D.shape
    out = np.empty(NB)
    for i in range(NB):
        s = rng.integers(0, ns, ns)
        b = rng.integers(0, nb, nb)
        out[i] = D[np.ix_(s, b)].mean()
    if somma:
        out *= nb
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def contrasto(t, ponti, g, a, b, m):
    w = t[t.gestore == g].pivot_table(index="seme", columns="legge", values=m)
    if a not in w or b not in w:
        return None
    w = w[[a, b]].dropna()
    dd = (w[a] - w[b]).to_numpy()
    lo, hi = boot_semi(dd)
    r = {"base": float(w[b].mean()), "diff": float(dd.mean()), "lo": lo, "hi": hi, "semi": len(dd),
         "positivi": int((dd > 0).sum()), "negativi": int((dd < 0).sum())}
    if m in PER_PONTE:
        col = PER_PONTE[m]
        D = np.vstack([ponti[(g, a, s)][col].to_numpy() - ponti[(g, b, s)][col].to_numpy() for s in w.index])
        r["lo_ponti_semi"], r["hi_ponti_semi"] = boot_due_vie(D, somma=(m != "S1"))
    return r


def main():
    dati = {br: carica(br) for br in BRACCI}
    tutte = pd.concat([v[0] for v in dati.values() if not v[0].empty], ignore_index=True)
    tutte.to_csv("Z_per_partita.csv", index=False)
    ris = {"limiti": LIMITI, "partite": {br: int(len(v[0])) for br, v in dati.items()}, "contrasti": {}, "medie": {}}
    for br in ("princ", "off", "markov", "potenza", "gamma"):
        t, ponti = dati[br]
        if t.empty:
            continue
        ris["medie"][br] = t.groupby(["gestore", "legge"])[ESITI].mean().round(4).reset_index().to_dict("records")
        for g in sorted(t.gestore.unique()):
            for a, b in CONFRONTI:
                cc = {m: contrasto(t, ponti, g, a, b, m) for m in ESITI}
                if all(v is None for v in cc.values()):
                    continue
                ris["contrasti"][f"{br}/{g}/{a}-{b}"] = cc
    # partita corta: differenze appaiate dal passivo
    c, _ = dati["corta"]
    if not c.empty:
        p = c[c.gestore == "do_nothing"].set_index("seme")
        corta = {}
        for g in sorted(c.gestore.unique()):
            x = c[c.gestore == g].set_index("seme")
            r = {"medie": {m: float(x[m].mean()) for m in ESITI}}
            if g != "do_nothing":
                for m in ESITI:
                    dd = (x[m] - p[m]).dropna().to_numpy()
                    lo, hi = boot_semi(dd)
                    r[f"{m}_meno_passivo"] = {"diff": float(dd.mean()), "lo": lo, "hi": hi,
                                              "positivi": int((dd > 0).sum()), "semi": len(dd)}
            corta[g] = r
        ris["corta"] = corta
    # regola del passivo (stessa di W e Y): D1-D0 su S1 entro 0,10 da zero con soglia spenta, >= 0,10 con soglia accesa
    on = ris["contrasti"].get("princ/do_nothing/D1-D0", {}).get("S1")
    off = ris["contrasti"].get("off/do_nothing/D1-D0", {}).get("S1")
    if on and off:
        ris["regola_passivo"] = {"off_diff": off["diff"], "on_diff": on["diff"],
                                 "off_entro_010": abs(off["diff"]) <= 0.10, "on_almeno_010": on["diff"] >= 0.10,
                                 "regge": abs(off["diff"]) <= 0.10 and on["diff"] >= 0.10}
    # Esempio 2: differenze assolute del passivo per legge, verso in almeno 9 semi su 10
    es2 = {}
    for br in ("princ", "markov", "potenza", "gamma"):
        cc = ris["contrasti"].get(f"{br}/do_nothing/D1-D0")
        if cc:
            es2[br] = {m: cc[m] for m in ("S1", "crolli_attesi", "anni_sotto_302")}
            es2[br]["verso_9_su_10"] = cc["S1"]["positivi"] >= 9
    ris["esempio2"] = es2
    json.dump(ris, open("analisi_Z.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    for k, v in ris["contrasti"].items():
        print(k)
        for m in ("S1", "crolli_attesi", "sotto_ponti_302", "anni_sotto_302", "sotto_ponti_366", "spesa", "utenti", "q_chiusi", "floor_interv", "isp_imposte"):
            r = v[m]
            if r is None:
                continue
            ps = f" ps[{r['lo_ponti_semi']:.3f}, {r['hi_ponti_semi']:.3f}]" if "lo_ponti_semi" in r else ""
            print(f"   {m:16s} base {r['base']:10.3f}  diff {r['diff']:9.3f} [{r['lo']:.3f}, {r['hi']:.3f}]{ps}  +{r['positivi']}/-{r['negativi']}/{r['semi']}")
    print("regola passivo", ris.get("regola_passivo"))
    if "corta" in ris:
        for g, r in ris["corta"].items():
            print("corta", g, {m: round(r["medie"][m], 3) for m in ("S1", "crolli_attesi", "sotto_ponti_302", "spesa", "utenti", "q_chiusi")})


if __name__ == "__main__":
    main()
