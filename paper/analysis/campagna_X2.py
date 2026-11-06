"""Campagna X2 (control/NOTA-campagna-X2.md): scomposizione dello scarto fra la stima del gestore e il vero.
Gestore passivo, partite di Z-princ (D0 e D1, 10 semi); ogni braccio spegne un meccanismo dello scarto; ogni
controllo della soglia registrato (beta stimato e vero). Una partita = un file JSON in blocks/X2/<braccio>/; si
riprende da dove si e' fermata. Uso: python campagna_X2.py [--workers 3]"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "blocks", "X2")
LIMITE = 3.02
BRACCI = {
    "base": {},
    "rif": {"params.assessment_epistemic_mode": "reference"},
    "param": {"params.assessment_parameter_mode": "sampled"},
    "semi": {"params.assessment_seed_mode": "shared"},
    "proiez": {"params.assessment_state_mode": "projected"},
    "tutti": {"params.assessment_epistemic_mode": "reference", "params.assessment_parameter_mode": "sampled",
              "params.assessment_seed_mode": "shared"},
    "rumore3": {"params.inspection_sigma_scale": 3.0},
}
PHI = NormalDist().cdf


def auc(stima, sotto):
    """Probabilita' che un controllo con il vero sotto il limite abbia stima piu' bassa di uno sopra (Mann-Whitney)."""
    if not sotto.any() or sotto.all():
        return None
    from scipy.stats import rankdata
    r = rankdata(-stima)
    n1, n0 = int(sotto.sum()), int((~sotto).sum())
    return float((r[sotto].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def payloads():
    import run_blocks as rb
    rb.set_campaign("Z")
    out = []
    for p in rb.build_payloads("princ"):
        if p["agent_id"] != "do_nothing" or "N24_" in p["grid_point"]["params.regulatory_regime"]:
            continue
        regime = "D1" if "D1_" in p["grid_point"]["params.regulatory_regime"] else "D0"
        for braccio, over in BRACCI.items():
            q = json.loads(json.dumps(p))
            q["grid_point"].update(over)
            out.append((braccio, regime, q))
    return out


def gioca(arg):
    braccio, regime, p = arg
    import run_blocks as rb
    from infralab.core.environment import BridgeEnv
    rb.set_campaign("Z")
    controlli = []
    orig = BridgeEnv._floor_condition_triggered

    def spia(self, bridge):
        esito = orig(self, bridge)
        src = self._assessed_bridge(bridge) if self.params.observation_mode == "inspected" else bridge
        s = src.posterior_annual_reliability_index if src.posterior_annual_reliability_index is not None else src.annual_reliability_index
        v = bridge.posterior_annual_reliability_index if bridge.posterior_annual_reliability_index is not None else bridge.annual_reliability_index
        if s is not None and v is not None:
            controlli.append((float(s), float(v), bool(esito)))
        return esito

    BridgeEnv._floor_condition_triggered = spia
    run_row, _steps, beta, annual, _coll = rb.replay_one(p)
    BridgeEnv._floor_condition_triggered = orig
    c = np.array(controlli, dtype=float)
    stima, vero = c[:, 0], c[:, 1]
    sotto = vero < LIMITE
    b = np.array([(r["quarter"], r["bridge"], r["beta_annual"]) for r in beta if r["quarter"] >= 1], dtype=float)
    mins = np.array([np.nanmin(b[b[:, 1] == i, 2]) for i in np.unique(b[:, 1])])
    fac = np.array([r["factor"] for r in annual if r["year"] >= 1], dtype=float)
    ris = {"braccio": braccio, "regime": regime, "replica": p["replicate"], "seme": p["seed"],
           "controlli": int(len(c)), "scarto_medio": float((stima - vero).mean()),
           "controlli_vero_sotto": int(sotto.sum()),
           "stima_mediana_vero_sotto": float(np.median(stima[sotto])) if sotto.any() else None,
           "vero_mediano_vero_sotto": float(np.median(vero[sotto])) if sotto.any() else None,
           "rilevazione": float((stima[sotto] < LIMITE).mean()) if sotto.any() else None,
           "falsi_allarmi": float((stima[~sotto] < LIMITE).mean()) if (~sotto).any() else None,
           "auc": auc(stima, sotto),
           "soglia_scattata": int(c[:, 2].sum()),
           "S1": float(mins.mean()), "sotto_ponti": int((mins < LIMITE).sum()), "sotto_quota": float((b[:, 2] < LIMITE).mean()),
           "crolli_attesi": float((1.0 - (1.0 - np.array([PHI(-x) for x in b[:, 2]])) ** 0.25).sum()),
           "crolli": float(run_row.get("final_failures_total", 0.0)), "spesa": float(run_row["final_spent_total"]) / 1e6,
           "spesa_esatta": float(run_row["final_spent_total"]), "q_chiusi": float((fac <= 0).mean()),
           "floor_interv": float(run_row.get("final_floor_interventions", 0) or 0)}
    d = os.path.join(OUT, braccio)
    os.makedirs(d, exist_ok=True)
    json.dump(ris, open(os.path.join(d, f"{regime}_r{p['replicate']}.json"), "w"), indent=1)
    return braccio, regime, p["replicate"], ris["S1"]


if __name__ == "__main__":
    import run_blocks as rb
    rb.set_campaign("Z")
    rb.check_engine()
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    repo = os.path.join(HERE, "..", "..", "..", "..", "_hf_repo")
    stato = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    sporco = subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    os.makedirs(OUT, exist_ok=True)
    json.dump({"motore": stato, "modifiche_non_committate": sporco}, open(os.path.join(OUT, "motore.json"), "w"), indent=1)
    todo = [a for a in payloads() if not os.path.exists(os.path.join(OUT, a[0], f"{a[1]}_r{a[2]['replicate']}.json"))]
    print(f"motore {stato}; non committato: {bool(sporco)}; partite da giocare {len(todo)}", flush=True)
    with ProcessPoolExecutor(max_workers=w) as ex:
        for f in as_completed([ex.submit(gioca, a) for a in todo]):
            print(*f.result(), flush=True)
    print("FINE", flush=True)
