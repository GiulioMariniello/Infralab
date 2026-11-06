"""Short views of what the owner knows, built only from the observation handed to a policy.

The full observation of the engine is about 65 kB per bridge. A language model or a tool client
needs a few hundred characters per asset: identity, traffic, what was last observed with its range,
the estimated reliability index against the target, legal deadlines and flags, works in progress.
Nothing here reads the engine: the input is the observation, so the information rule is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union


def _r(x: Any, n: int = 3) -> Any:
    return None if x is None else round(float(x), n)


def compact_asset(b: Dict[str, Any]) -> Dict[str, Any]:
    """One asset of the observation in short form."""
    comps = {}
    for name, c in (b.get("components") or {}).items():
        lo, hi = c.get("observed_condition_min"), c.get("observed_condition_max")
        comps[name] = [_r(lo, 2), _r(hi, 2)] if lo is not None and hi is not None else None
    det = b.get("reliability_details") or {}
    beta = b.get("posterior_annual_reliability_index")
    if beta is None:
        beta = b.get("annual_reliability_index")
    work = b.get("active_maintenance") or None
    out = {
        "id": b.get("id"), "name": b.get("name"),
        "traffic_adt": b.get("adt"), "deck_area_m2": _r(b.get("area"), 0), "detour_km": _r(b.get("detour_km"), 1),
        "status": b.get("status"), "traffic_factor": _r(b.get("traffic_restriction_factor", 1.0), 2),
        "attention_class": b.get("cda"),
        "last_inspection_quarter": b.get("last_inspection_quarter"),
        "next_inspection_due_quarter": b.get("next_inspection_quarter"),
        "inspection_overdue": bool(b.get("inspection_overdue")),
        "known_condition_range": comps,           # per component, 0 collapse .. 1 intact, as last observed
        "worst_known_condition": _r(b.get("min_observed_condition"), 2),
        "estimated_annual_beta": _r(beta, 2), "target_annual_beta": _r(det.get("target_beta_annual"), 2),
        "estimated_annual_failure_probability": b.get("posterior_annual_failure_probability", b.get("annual_failure_probability")),
        "asset_value": _r(b.get("current_asset_value"), 0),
    }
    if work:
        out["works_in_progress"] = {"action": work.get("action_name"), "ends_at_quarter": work.get("ends_at_quarter")}
    if b.get("regime_mandatory_action"):
        out["mandatory_action_by_law"] = b.get("regime_mandatory_action")
    if b.get("regime_sanction"):
        out["sanction"] = b.get("regime_sanction")
    if b.get("known_progressive_defects"):
        out["known_defects"] = [d.get("type") for d in b["known_progressive_defects"]]
    if b.get("regional_event_uninspected"):
        out["regional_event_not_yet_inspected"] = True
    if b.get("last_action_name"):
        out["last_action"] = [b.get("last_action_name"), b.get("last_action_quarter")]
    return out


def compact_observation(obs: Dict[str, Any], *, quarter: Optional[int] = None, horizon_quarters: Optional[int] = None,
                        asset_ids: Optional[Iterable[Union[int, str]]] = None) -> Dict[str, Any]:
    """The observation in short form: budget, quarter, and one short record per asset."""
    wanted = None if asset_ids is None else {str(a) for a in asset_ids}
    assets = [compact_asset(b) for b in obs.get("my_bridges", []) if wanted is None or str(b.get("id")) in wanted]
    rules = obs.get("game_rules") or {}
    out = {"quarter": quarter if quarter is not None else rules.get("quarter"),
           "horizon_quarters": horizon_quarters, "budget": rules.get("my_budget"),
           "emissions_tco2": rules.get("my_emissions_tco2"), "assets": assets}
    diag = obs.get("last_plan_diagnostics")
    if diag:
        out["previous_plan"] = diag
    return out


def action_catalogue(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The actions of the scenario with type, duration, unit cost, emissions and effect on each component."""
    out = []
    for name, op in (obs.get("maintenance_options") or {}).items():
        out.append({"action": name, "type": op.get("type"), "duration_quarters": op.get("duration_quarters", 0),
                    "cost_fixed": op.get("cost", 0), "cost_per_m2": op.get("cost_m2", 0),
                    "kgco2e_fixed": op.get("co2", 0), "kgco2e_per_m2": op.get("co2_m2", 0),
                    "condition_gain_by_component": op.get("gains") or {},
                    "observation_error_sigma": op.get("error_sigma")})
    return out


def feasible_for_asset(feasible: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Feasible and blocked actions of one asset, with cost and the reason of a block."""
    rows = []
    for name, f in feasible.items():
        row = {"action": name, "type": f.get("type"), "cost": round(float(f.get("cost") or 0.0), 0), "feasible": bool(f.get("feasible"))}
        if not f.get("feasible"):
            row["reason"] = f.get("reason")
        rows.append(row)
    return rows


def rules_view(env) -> Dict[str, Any]:
    """The rules of a run as plain data. Reads public settings only (never the state of the assets)."""
    p = env.engine.params
    obs = env.observation()
    reg = obs.get("regime") or {}
    g = lambda k, d=None: getattr(p, k, d)
    return {
        "goal": "Reach the horizon with a safe stock at low cost: few collapses and closures, limited spending.",
        "time_step": "one quarter; at most one action per asset per quarter",
        "horizon_quarters": env._horizon(), "quarter": env.quarter,
        "initial_budget": (obs.get("game_settings") or {}).get("initial_budget"),
        "information": "The true condition is hidden. A component shows the range last observed by an inspection; "
                       "ranges widen with time since the inspection. Reliability figures are estimates on what is known.",
        "law": {"id": reg.get("id"), "name": reg.get("display_name"), "enforcement": reg.get("enforcement"),
                "obligations": reg.get("obligations") or [],
                "inspection_interval_quarters_by_attention_class": {
                    k[:-len("_due_quarters")]: v for k, v in (obs.get("cda_config") or {}).items() if k.endswith("_due_quarters")}},
        "ethical_floor": {"active": bool(g("ethical_floor_enabled", False)),
                          "acts_when_known_condition_below": g("ethical_floor_condition"),
                          "response": g("ethical_floor_response", "repair_or_close"),
                          "note": "When a serious problem is known and the owner does not act, the floor orders the "
                                  "measure at the owner's expense."},
        "constraints": obs.get("management_constraints") or {},
        "outcomes_reported": ["S1 (mean over assets of the worst annual reliability index reached)", "expected collapses",
                              "cash spend", "user cost of closures and restrictions", "emissions (tCO2e) of the works"],
    }
