"""A language model manages a stock through tools, one decision per quarter.

Each quarter the model receives the short observation (a few thousand tokens for 30 assets), may ask for
the actions available on single assets, and commits a plan with `submit_plan`. Nothing of a quarter is
carried to the next except a memo that the model writes for itself, so the cost of a run grows with the
number of quarters and not with its square.

    from infralab.llm import play
    play("anthropic:claude-opus-5-5", scenario="regbench/aged-30", law="regbench/D1", seed=0)
    play("openai:gpt-4.1-mini", ...)
    play("openai:llama3.1", base_url="http://localhost:11434/v1", ...)   # any endpoint with the same protocol

The model sees only what the owner may know. Outcomes computed on the true state are read after the run.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from infralab.toolkit import Toolkit, ToolError

TOOLS: List[Dict[str, Any]] = [
    {"name": "asset_actions",
     "description": "Actions available now on one asset, each with its cost. Blocked actions carry the reason. "
                    "Call it for the assets you consider acting on before you submit a plan.",
     "parameters": {"type": "object", "properties": {"asset_id": {"type": "string", "description": "Identifier of the asset."}},
                    "required": ["asset_id"], "additionalProperties": False}},
    {"name": "action_catalogue",
     "description": "All actions of the scenario: type, duration, unit cost, emissions and the condition gain on each "
                    "component. The catalogue does not change during a run.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "submit_plan",
     "description": "Commit the plan of this quarter and end the decision. `plan` maps asset identifiers to action names, "
                    "at most one action per asset; an empty object does nothing. `memo` is a note to yourself for the next "
                    "quarters (it is the only thing you will remember). `advance_quarters` above 1 lets the following "
                    "quarters pass with no action of yours, stopping early at a collapse, an intervention of the ethical "
                    "floor or a dropped request.",
     "parameters": {"type": "object",
                    "properties": {"plan": {"type": "object", "additionalProperties": {"type": "string"}},
                                   "reason": {"type": "string"}, "memo": {"type": "string"},
                                   "advance_quarters": {"type": "integer", "minimum": 1, "maximum": 8}},
                    "required": ["plan", "reason"], "additionalProperties": False}},
]

SYSTEM = """You manage a stock of bridges as their owner, one quarter at a time, in a simulation.

Rules of this run:
{rules}

How to decide:
- You know each bridge only through inspections: `known_condition_range` is the range last observed for each \
component (0 collapse, 1 intact) and it widens with time. Estimates of the reliability index rest on that knowledge.
- Each quarter you may request at most one action per bridge. Requests that are unfeasible or that the budget \
cannot pay are dropped and reported back.
- The law in force may impose inspections, and the ethical floor may order safety measures at your expense when \
a serious problem is known and you have not acted.
- Closing or restricting a bridge raises its reliability index because it carries less load, and it charges the \
road users: prefer repairs when the budget allows.
- Use `asset_actions` for the bridges you consider, then commit with `submit_plan`. Always end a decision with \
`submit_plan`, also when the plan is empty. Asset identifiers are the `id` values of the observation, as strings.
{profile}"""

PROFILES = {
    "balanced": "- Balance safety, spending and service to the road users.",
    "prudent": "- Put safety first: act early on the bridges in worst known condition, even at higher spending.",
    "frugal": "- Keep spending low: act only where a serious problem is known or a legal duty is due.",
}


def make_backend(spec: str, **kw: Any):
    """`anthropic:<model>` or `openai:<model>` (the second also for compatible endpoints, with base_url=...)."""
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        from infralab.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model or "claude-opus-5-5", **kw)
    if provider == "openai":
        from infralab.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(model, **kw)
    raise KeyError(f"Unknown provider {provider!r}: use 'anthropic:<model>' or 'openai:<model>'.")


def play(model: str, scenario: str = "regbench/aged-30", *, law: Optional[str] = "regbench/D1", config: str = "calibrated",
         seed: int = 0, horizon_years: float = 10.0, profile: str = "balanced", max_rounds: int = 6,
         log_dir: Optional[str] = None, backend: Any = None, verbose: bool = False, **backend_kw: Any) -> Dict[str, Any]:
    """Let a language model manage one run. Returns the outcomes, the token usage and the failure counts."""
    backend = backend or make_backend(model, **backend_kw)
    tk = Toolkit(log_dir=log_dir)
    start = tk.new_run(scenario, law, config, seed, horizon_years, label=model)
    rid = start["run_id"]
    system = SYSTEM.format(rules=json.dumps(start["rules"], ensure_ascii=False, indent=1), profile=PROFILES.get(profile, profile))
    memo, last = "", None
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    stats = {"decisions": 0, "failed_decisions": 0, "malformed_entries": 0, "tool_errors": 0, "errors": []}
    t0 = time.time()
    env = tk.runs[rid]["env"]
    while not env.done:
        obs = tk.observe(rid)
        user = {"quarter": obs["quarter"], "horizon_quarters": obs["horizon_quarters"], "budget": obs["budget"],
                "your_memo": memo, "report_of_previous_quarters": last, "assets": obs["assets"]}
        turn = {"done": False}

        def execute(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal memo, last
            try:
                if name == "asset_actions":
                    return {"result": tk.asset_actions(rid, str(args.get("asset_id")))}
                if name == "action_catalogue":
                    return {"result": tk.action_catalogue(rid)}
                if name == "submit_plan":
                    if turn["done"]:
                        return {"result": {"error": "plan already submitted for this quarter"}, "is_error": True}
                    plan = args.get("plan") or {}
                    good = {str(k): str(v) for k, v in plan.items() if str(k).strip().lstrip("-").isdigit()} if isinstance(plan, dict) else {}
                    stats["malformed_entries"] += (len(plan) - len(good)) if isinstance(plan, dict) else 1
                    out = tk.step(rid, good, int(args.get("advance_quarters") or 1), str(args.get("reason") or ""))
                    memo = str(args.get("memo") or memo)[:2000]
                    last = out["reports"]
                    turn["done"] = True
                    return {"result": {"accepted": True}, "finished": True}
                return {"result": {"error": f"unknown tool {name}"}, "is_error": True}
            except ToolError as e:
                stats["tool_errors"] += 1
                return {"result": {"error": str(e)}, "is_error": True}

        stats["decisions"] += 1
        try:
            res = backend.run_turn(system, json.dumps(user, ensure_ascii=False, default=str), TOOLS, execute, max_rounds)
        except Exception as e:  # a failed call must not stop the run: the quarter passes with no action and is counted
            res = {"finished": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "usage": {}}
        for k in usage:
            usage[k] += int((res.get("usage") or {}).get(k, 0))
        if not turn["done"]:
            stats["failed_decisions"] += 1
            stats["errors"].append({"quarter": obs["quarter"], "error": res.get("error")})
            last = tk.step(rid, {}, 1, "no plan submitted")["reports"]
        if verbose:
            print(f"q{obs['quarter']:>3} budget {obs['budget']:.0f} in {usage['input_tokens']} out {usage['output_tokens']}"
                  f" failed {stats['failed_decisions']}", flush=True)
    score = tk.score(rid)
    stats["errors"] = stats["errors"][:5]
    return {**score, "model": model, "profile": profile, "usage": usage, **stats, "seconds": round(time.time() - t0, 1),
            "log": tk.runs[rid]["log"]}
