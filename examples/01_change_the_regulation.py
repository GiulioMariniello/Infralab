"""Example 1. Change the regulation: what does a mandated periodic inspection add, and on what does it depend?

Regulation D0 (no obligation), N24 (routine inspection every 24 months, every 12 for rating 3 or lower) and D1
(periodic inspection, intervals graded by attention class), for three owners, with the ethical floor on and off.
Same stock, same owner, same random draws: only the regulation changes. Configuration of the paper.
"""
import json
import infralab

PAIRS = [("regbench/D0", "regbench/D1"), ("regbench/D0", "regbench/N24"), ("regbench/N24", "regbench/D1")]

if __name__ == "__main__":
    out = {}
    for owner in ("passive", "typical", "worst_first"):
        for floor in (True, False):
            for a, b in PAIRS:
                r = infralab.compare_laws(a, b, policy=owner, scenario="regbench/aged-147", config="paper",
                                          params={"ethical_floor_enabled": floor}, n_seeds=10, workers=5)
                key = f"{owner}/floor_{'on' if floor else 'off'}/{b.split('/')[1]}-{a.split('/')[1]}"
                out[key] = r
                print(key, "S1", r["difference_b_minus_a"]["S1"],
                      "expected hydraulic collapses", r["difference_b_minus_a"]["expected_hydraulic_collapses"], flush=True)
                json.dump(out, open("example1.json", "w"), indent=1)
