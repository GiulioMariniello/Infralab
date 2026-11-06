"""Tool functions over the environment, shared by the MCP server and by the tool-using language-model agent.

A `Toolkit` holds the runs of one client. Every function takes and returns plain JSON data, and reads
only what the owner may know: during a run it never returns outcomes computed on the true state
(the safety outcome S1 and the expected collapses are released when the run is over).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from infralab import api
from infralab.views import action_catalogue, feasible_for_asset

# outcomes that need the true state: released only at the end of a run
_TRUTH_KEYS = ("S1", "min_beta_stock", "expected_collapses", "imputed_expected_loss")
MAX_STOCK_FOR_TOOLS = 200


class ToolError(ValueError):
    """An error to be shown to the client as it is (wrong identifier, run over, ...)."""


class Toolkit:
    def __init__(self, log_dir: Optional[str] = None):
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.log_dir = log_dir or os.environ.get("INFRALAB_RUN_DIR")

    # ---- catalogues
    def list_scenarios(self) -> List[Dict[str, Any]]:
        out = []
        for sid in api.scenarios():
            s = api.load_scenario(sid)
            out.append({"id": sid, "name": s.get("name"), "assets": len(s.get("bridges", [])),
                        "asset_type": s.get("asset_type") or "bridge"})
        return out

    def list_laws(self) -> List[str]:
        return api.laws()

    def describe_law(self, law: str) -> Dict[str, Any]:
        return api.describe_law(law)

    def list_policies(self) -> Dict[str, Any]:
        return {"policies": api.policies(), "fixed_rules": ["basic:" + p for p in api.basic_policies()]}

    # ---- a run played step by step
    def new_run(self, scenario: str = "regbench/aged-30", law: Optional[str] = "regbench/D1", config: str = "paper",
                seed: int = 0, horizon_years: float = 10.0, label: str = "") -> Dict[str, Any]:
        s = api.load_scenario(scenario)
        if len(s.get("bridges", [])) > MAX_STOCK_FOR_TOOLS:
            raise ToolError(f"Scenario too large for step-by-step tools ({len(s['bridges'])} assets, limit {MAX_STOCK_FOR_TOOLS}).")
        env = api.make(scenario, law=law, config=config, seed=int(seed), horizon_years=horizon_years)
        env.reset()
        rid = uuid.uuid4().hex[:10]
        self.runs[rid] = {"env": env, "log": [], "started": time.time(), "label": label,
                          "setup": {"scenario": scenario, "law": law, "config": config, "seed": int(seed),
                                    "horizon_years": horizon_years}}
        return {"run_id": rid, "rules": env.rules(), "assets": len(env.observation().get("my_bridges", [])),
                "next": "Call observe(run_id), then step(run_id, plan) once per quarter until done."}

    def _run(self, run_id: str):
        if run_id not in self.runs:
            raise ToolError(f"Unknown run_id {run_id!r}. Create a run with new_run().")
        return self.runs[run_id]

    def rules(self, run_id: str) -> Dict[str, Any]:
        return self._run(run_id)["env"].rules()

    def observe(self, run_id: str, asset_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        env = self._run(run_id)["env"]
        obs = env.compact_observation(asset_ids)
        obs["done"] = env.done
        return obs

    def asset_actions(self, run_id: str, asset_id: str) -> Dict[str, Any]:
        env = self._run(run_id)["env"]
        try:
            fa = env.feasible_actions(asset_id)
        except KeyError:
            raise ToolError(f"Unknown asset {asset_id!r}.")
        return {"asset_id": str(asset_id), "actions": feasible_for_asset(fa)}

    def action_catalogue(self, run_id: str) -> List[Dict[str, Any]]:
        return action_catalogue(self._run(run_id)["env"].observation())

    def step(self, run_id: str, plan: Optional[Dict[str, str]] = None, quarters: int = 1, reason: str = "") -> Dict[str, Any]:
        """Apply a plan for this quarter, then advance. With quarters > 1 the following quarters pass with no
        action of the owner's and stop early at a collapse, an intervention of the ethical floor or a dropped request."""
        r = self._run(run_id)
        env = r["env"]
        if env.done:
            raise ToolError("The run is over. Call score(run_id).")
        plan = {str(k): str(v) for k, v in (plan or {}).items()}
        reports = []
        for i in range(max(1, int(quarters))):
            p = plan if i == 0 else {}
            q = env.quarter
            env.step(p)
            rep = dict(env.last_report)
            rep["quarter"] = q
            reports.append(rep)
            r["log"].append({"quarter": q, "plan": p, "reason": reason if i == 0 else "", "advanced_without_action": i > 0,
                             "report": rep, "t": round(time.time() - r["started"], 2)})
            if env.done or rep["collapses"] or rep["ethical_floor_interventions"] or rep["dropped"]:
                break
        out = {"reports": reports, "quarter": env.quarter, "done": env.done, "budget": reports[-1]["budget"]}
        if env.done:
            out["score"] = self.score(run_id)
        return out

    def score(self, run_id: str) -> Dict[str, Any]:
        r = self._run(run_id)
        env = r["env"]
        s = env.summary()
        if not env.done:
            s = {k: v for k, v in s.items() if k not in _TRUTH_KEYS}
            s["note"] = "Run in progress: outcomes computed on the true state are released at the end."
        s["done"] = env.done
        s["decisions"] = sum(1 for x in r["log"] if not x["advanced_without_action"])
        s["actions_requested"] = sum(len(x["plan"]) for x in r["log"])
        if env.done:
            self._save(run_id, s)
        return s

    def _save(self, run_id: str, score: Dict[str, Any]) -> None:
        if not self.log_dir:
            return
        r = self.runs[run_id]
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"run_{run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, "label": r["label"], "setup": r["setup"], "score": score, "log": r["log"]}, f,
                      ensure_ascii=False, indent=1, default=str)

    # ---- whole experiments
    def run_policy(self, policy: str = "passive", scenario: str = "regbench/aged-30", law: Optional[str] = "regbench/D1",
                   config: str = "paper", seed: int = 0, horizon_years: float = 10.0) -> Dict[str, Any]:
        """Run a supplied policy from start to end and return its outcomes (a baseline for a played run)."""
        if policy.startswith("llm:"):
            raise ToolError("Language-model policies are not run through this tool.")
        return api.run(policy, scenario, law=law, config=config, seed=int(seed), horizon_years=horizon_years)

    def compare_laws(self, law_a: Optional[str], law_b: Optional[str], policy: str = "passive",
                     scenario: str = "regbench/aged-30", config: str = "paper", n_seeds: int = 5,
                     horizon_years: float = 10.0) -> Dict[str, Any]:
        """Paired-seed contrast of two laws (law_b minus law_a) for one policy."""
        if int(n_seeds) > 10:
            raise ToolError("At most 10 seeds through this tool; use the Python API for larger campaigns.")
        res = api.compare_laws(law_a, law_b, policy=policy, scenario=scenario, config=config, n_seeds=int(n_seeds),
                               horizon_years=horizon_years)
        return {k: res[k] for k in ("law_a", "law_b", "policy", "seeds", "difference_b_minus_a", "mean_a", "mean_b")}
