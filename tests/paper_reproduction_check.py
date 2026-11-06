"""Check that the library, called as in the paper, reproduces stored games of campaign Z (engine 6098d88).

Usage: python tests/paper_reproduction_check.py <campaign_blocks_dir> short|long
Compares S1, expected collapses, cash spend, collapses and the expected hydraulic collapses of one stored game
with the same game played through infralab.run.
"""
import glob
import os
import sys

import pandas as pd

import infralab

blocks, which = sys.argv[1], sys.argv[2]
arm, cfg, scen, owner, agent = {
    "short": ("corta", "paper-short", "regbench/aged-30", "passive", "do_nothing"),
    "long": ("princ", "paper", "regbench/aged-147", "typical", "typical_owner"),
}[which]
bd = sorted(glob.glob(os.path.join(blocks, arm, "b00")))[0]
runs = pd.read_csv(os.path.join(bd, "runs.csv"))
row = runs[(runs.agent_id == agent) & (runs.replicate == 0) & runs["params.regulatory_regime"].str.contains("D1_")].iloc[0]
seed = infralab.child_seeds(infralab.configuration(cfg)["root_seed"], 10)[0]
r = infralab.run(owner, scen, law="regbench/D1", config=cfg, seed=seed)
beta = pd.read_parquet(glob.glob(os.path.join(bd, "perbridge", f"*_id{int(row.run_id):03d}_beta.parquet"))[0])
beta = beta[beta.quarter >= 1]
s1 = beta.groupby("bridge").beta_annual.min().mean()
pairs = {"S1": (r["S1"], s1), "cash_spend": (r["cash_spend"], row.final_spent_total),
         "collapses": (r["collapses"], row.final_failures_total),
         "expected_hydraulic_collapses": (r["expected_hydraulic_collapses"], 0.05 * row.final_progressive_matured_hydraulic),
         "inspections_forced_by_law": (r["inspections_forced_by_law"], row.final_regime_forced_actions)}
ok = True
for k, (a, b) in pairs.items():
    same = abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
    ok &= same
    print(f"{k:32s} library {a!r:>24}  stored {b!r:>24}  {'same' if same else 'DIFFERENT'}")
print("REPRODUCED" if ok else "NOT REPRODUCED")
