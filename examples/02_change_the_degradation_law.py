"""Example 2. Change the degradation law: does the effect of the inspection obligation hold?

The contrast of Example 1 is replayed under four degradation laws: the two built into the engine and two
written as files, a power law and a Markov chain estimated on National Bridge Inventory ratings.
"""
import json
import infralab

if __name__ == "__main__":
    out = {}
    for law in ("weibull", "gamma", "regbench/power", "regbench/markov-nbi"):
        for owner in ("passive", "worst_first"):
            r = infralab.compare_laws("regbench/D0", "regbench/D1", policy=owner, scenario="regbench/aged-147",
                                      degradation=law, n_seeds=10, workers=5)
            out[f"{law}/{owner}"] = r
            print(law, owner, "S1 difference", r["difference_b_minus_a"]["S1"], flush=True)
            json.dump(out, open("example2.json", "w"), indent=1)
