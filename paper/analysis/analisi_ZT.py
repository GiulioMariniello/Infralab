"""Campagna ZT (control/NOTA-campagna-ZT.md): lettura fissata nella nota prima delle partite, regole R1-R4.

Stessi esiti e stessi intervalli di analisi_Z.py (bootstrap sui semi, 4000) piu' crolli idraulici attesi da difetti
progressivi (0,05 x difetti maturati non arrestati) e interventi della soglia distinti fra beta stimato e difetto
progressivo. Per confronto riporta anche i livelli di Z-princ (taratura di base).
Uso: python analisi_ZT.py   ->  analisi_ZT.json, ZT_per_partita.csv
"""
import json

import numpy as np
import pandas as pd

import analisi_Z as AZ

AZ.rng = np.random.default_rng(20261106)
EXTRA = ["idr_attesi", "floor_beta", "floor_prog"]
ESITI = ["S1", "crolli_attesi", "idr_attesi", "sotto_ponti_302", "anni_sotto_302", "spesa", "floor_interv",
         "floor_beta", "floor_prog", "isp_imposte"]


def carica(campagna, braccio):
    AZ_glob = AZ.glob.glob
    t, ponti = None, None
    orig = AZ.glob.glob
    AZ.glob.glob = lambda p, **k: orig(p.replace("blocks/Z/", f"blocks/{campagna}/"), **k)
    try:
        t, ponti = AZ.carica(braccio)
    finally:
        AZ.glob.glob = AZ_glob
    # colonne idrauliche e soglia dai runs.csv
    extra = []
    for bd in sorted(orig(f"blocks/{campagna}/{braccio}/b*")):
        r = pd.read_csv(f"{bd}/runs.csv")
        for x in r.itertuples():
            fp = float(getattr(x, "final_progressive_floor_arrests", 0) or 0)
            fi = float(getattr(x, "final_floor_interventions", 0) or 0)
            extra.append({"gestore": x.agent_id, "legge": AZ.norma(r.loc[x.Index, "params.regulatory_regime"]),
                          "seme": int(x.replicate),
                          "idr_attesi": 0.05 * float(x.final_progressive_matured_hydraulic),
                          "floor_prog": fp, "floor_beta": fi - fp, "prog_arrestati_soglia": float(x.final_progressive_arrested_by_floor)})
    e = pd.DataFrame(extra)
    t = t.merge(e, on=["gestore", "legge", "seme"], how="left")
    return t, ponti


def main():
    t, ponti = carica("ZT", "princ")
    t.to_csv("ZT_per_partita.csv", index=False)
    z, _ = carica("Z", "princ")
    ris = {"partite": int(len(t)), "medie": t.groupby(["gestore", "legge"])[ESITI + ["prog_arrestati_soglia"]].mean().round(4).reset_index().to_dict("records"),
           "medie_Z_base": z.groupby(["gestore", "legge"])[ESITI + ["prog_arrestati_soglia"]].mean().round(4).reset_index().to_dict("records"),
           "contrasti": {}}
    for g in sorted(t.gestore.unique()):
        ris["contrasti"][g] = {m: AZ.contrasto(t, ponti, g, "D1", "D0", m) for m in ESITI}
    # contrasti Z con la stessa lettura (taratura di base), per confronto
    ris["contrasti_Z"] = {}
    for g in sorted(z.gestore.unique()):
        ris["contrasti_Z"][g] = {m: AZ.contrasto(z, {}, g, "D1", "D0", m) if m not in AZ.PER_PONTE else None for m in ESITI}
        for m in ESITI:
            if m in AZ.PER_PONTE:
                w = z[z.gestore == g].pivot_table(index="seme", columns="legge", values=m)
                dd = (w["D1"] - w["D0"]).to_numpy()
                lo, hi = AZ.boot_semi(dd)
                ris["contrasti_Z"][g][m] = {"base": float(w["D0"].mean()), "diff": float(dd.mean()), "lo": lo, "hi": hi,
                                            "positivi": int((dd > 0).sum()), "negativi": int((dd < 0).sum()), "semi": len(dd)}
    c = ris["contrasti"]
    tip = c["typical_owner"]
    # R2: rapporto per seme e rapporto delle medie
    w = t[t.gestore == "typical_owner"].pivot_table(index="seme", columns="legge", values="crolli_attesi")
    rapp = float(w["D1"].mean() / w["D0"].mean())
    idx = AZ.rng.integers(0, len(w), size=(AZ.NB, len(w)))
    rb = w["D1"].to_numpy()[idx].mean(1) / w["D0"].to_numpy()[idx].mean(1)
    pas = t[(t.gestore == "do_nothing") & (t.legge == "D1")]
    ris["regole"] = {
        "R1": {"positivi": tip["S1"]["positivi"], "semi": tip["S1"]["semi"], "regge": tip["S1"]["positivi"] >= 9},
        "R2": {"rapporto": rapp, "lo": float(np.percentile(rb, 2.5)), "hi": float(np.percentile(rb, 97.5)),
               "regge": 0.40 <= rapp <= 0.65},
        "R3": {"S1_diff": c["do_nothing"]["S1"], "floor_beta_medio_D1": float(pas.floor_beta.mean()),
               "floor_prog_medio_D1": float(pas.floor_prog.mean()), "regge": float(pas.floor_beta.mean()) < 1},
        "R4": {g: {"diff": c[g]["idr_attesi"]["diff"], "lo": c[g]["idr_attesi"]["lo"], "hi": c[g]["idr_attesi"]["hi"],
                   "negativi": c[g]["idr_attesi"]["negativi"]} for g in c},
    }
    json.dump(ris, open("analisi_ZT.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    print("partite", len(t))
    print(t.groupby(["gestore", "legge"])[ESITI].mean().round(3).to_string())
    print("--- Z base")
    print(z.groupby(["gestore", "legge"])[ESITI].mean().round(3).to_string())
    for lab, cc in (("ZT", ris["contrasti"]), ("Z", ris["contrasti_Z"])):
        for g, v in cc.items():
            print(lab, g, "D1-D0")
            for m in ESITI:
                r = v[m]
                if r is None:
                    continue
                ps = f" ps[{r['lo_ponti_semi']:.3f}, {r['hi_ponti_semi']:.3f}]" if "lo_ponti_semi" in r else ""
                print(f"   {m:16s} base {r['base']:10.3f}  diff {r['diff']:9.3f} [{r['lo']:.3f}, {r['hi']:.3f}]{ps}  +{r['positivi']}/-{r['negativi']}/{r['semi']}")
    print(json.dumps(ris["regole"], indent=1, default=float))


if __name__ == "__main__":
    main()
