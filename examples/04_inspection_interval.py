"""Example 1, continued. Length of the imposed inspection interval, from 6 to 360 months.

A regulation is a file: this script writes one file per interval from N24, with the same interval for every
attention class, and compares each with D0 for the typical owner on the same seeds.
"""
import copy
import json
import os
import tempfile

import yaml
import infralab

MONTHS = [6, 12, 24, 36, 48, 60, 120, 240, 360]

if __name__ == "__main__":
    base = infralab.describe_law("regbench/N24")
    folder = tempfile.mkdtemp(prefix="infralab_intervals_")
    out = {}
    for months in MONTHS:
        law = copy.deepcopy(base)
        law["id"] = f"fixed_{months}_months"
        law["display_name"] = f"Routine inspection every {months} months, whatever the condition"
        for key in law["cda_config"]:
            if key.endswith("_due_quarters"):
                law["cda_config"][key] = months // 3
        path = os.path.join(folder, f"fixed_{months}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(law, f, sort_keys=False)
        r = infralab.compare_laws("regbench/D0", path, policy="typical", scenario="regbench/aged-147", config="paper",
                                  n_seeds=10, workers=5)
        out[months] = r["difference_b_minus_a"]
        print(months, "months: S1", r["difference_b_minus_a"]["S1"],
              "expected hydraulic collapses", r["difference_b_minus_a"]["expected_hydraulic_collapses"], flush=True)
        json.dump(out, open("example_intervals.json", "w"), indent=1)
