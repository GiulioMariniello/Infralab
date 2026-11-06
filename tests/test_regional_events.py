"""Eventi estremi regionali (hazards/regional_events.py; tutto opt-in)."""
import json

import numpy as np
import pytest

from infralab.agents.basic import BasicAgent
from infralab.agents.sandbox import belief_snapshot, sanitized_observation
from infralab.core.environment import BridgeEnv
from infralab.core.hazards import regional_events as rg
from infralab.core.hazards.seismic_hazard import haversine_km
from infralab.lab.build import build_env_from_scenario
from infralab.lab.leak_check import run_leak_check
from infralab.lab.runner import expand_runs, run_one
from infralab.lab.spec import spec_from_dict
from infralab.regimes import regime_from_dict, register_regime
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import N_QUARTERS, SNAPSHOT_PATH, canonical

# Eventi frequenti e larghi, tutti i ponti esposti, danno certo al centro, nascosto.
STRESS = {"regional_events_enabled": True, "regional_event_rate_y": 4.0, "regional_event_radius_km": 60.0,
          "regional_event_decay_km": 40.0, "regional_event_requires": "", "regional_event_multipliers": {},
          "regional_event_hit_prob": 1.0, "regional_event_announce_prob": 0.0,
          "regional_event_drop_dist": "fixed", "regional_event_drop_mean": 0.2}


def _env(overrides=None, seed=7, scenario="TRAIN-01"):
    return build_env_from_scenario(load_scenario(scenario), param_overrides=overrides or {}, seed=seed)


def _step(env, n=1, policy="do_nothing", plan=None):
    agent = env.params.agent_names[0]
    for _ in range(n):
        p = plan if plan is not None else BasicAgent(agent, policy).decide(env)[0]
        env.step({agent: p})


def _regional_pending(env):
    return sorted((bid, e["quarter"], e["component"]) for bid, evs in env.sudden_pending.items()
                  for e in evs if e["type"] == "regional")


def _first_event_env(overrides, max_q=40):
    env = _env(overrides)
    for _ in range(max_q):
        _step(env)
        if env.regional_events:
            return env
    raise AssertionError("nessun evento regionale")


# -- default invariati ---------------------------------------------------------

def test_defaults_are_neutral():
    p = SimParameters()
    assert p.regional_events_enabled is False and not rg.enabled(p)
    env = _env()
    _step(env, 3)
    assert env.regional_events == [] and env.regional_stats == {}
    data = env.to_dict()
    assert "regional_events" not in data and "regional_stats" not in data
    assert rg.metrics(env, env.params.agent_names[0]) == {}
    obs = env.get_agent_observations(env.params.agent_names[0])
    assert "regional_events" not in obs
    assert all("regional_event_uninspected" not in b for b in obs["my_bridges"])


def test_events_without_damage_keep_train01_golden_snapshot():
    """Canale acceso con eventi ma senza danno: traiettoria identica allo snapshot
    dorato (i flussi dedicati non spostano l'RNG condiviso ne' gli altri flussi)."""
    env = _env({**STRESS, "regional_event_hit_prob": 0.0}, seed=None)
    agent = env.params.agent_names[0]
    traj = {"quarters": []}
    for _ in range(N_QUARTERS):
        env.step({agent: {"maintenance": {}}})
        traj["quarters"].append({
            "quarter": env.quarter, "budget": env.budgets[agent],
            "bridges": {str(bid): {"min_condition": b.min_condition, "reliability_index": b.reliability_index,
                                   "annual_failure_probability": b.annual_failure_probability}
                        for bid, b in sorted(env.bridges.items())}})
    assert env.regional_events, "servono eventi per rendere significativo il controllo"
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(traj) == canonical(golden)


@pytest.mark.parametrize("bad", [
    {"regional_event_decay_model": "linear"}, {"regional_event_center_mode": "somewhere"},
    {"regional_event_hit_prob": 1.5}, {"regional_event_rate_y": -1.0}, {"regional_event_targets": ""},
    {"regional_event_multipliers": {"scour_critical": -2}}, {"regional_event_multipliers": [1, 2]},
    {"regional_events_enabled": "si"}, {"regional_event_drop_dist": "normal"},
    {"sudden_custom_events": [{"name": "regional", "rate_q": 0.1}]},   # nome riservato
])
def test_validation(bad):
    with pytest.raises(ValueError):
        SimParameters(**bad)


# -- numeri casuali comuni e correlazione spaziale -----------------------------

def test_crn_same_events_and_hits_across_policies():
    a = _env(STRESS, seed=11)
    b = _env(STRESS, seed=11)
    _step(a, 8, policy="do_nothing")
    _step(b, 8, policy="worst_first")
    assert a.regional_events and a.regional_events == b.regional_events
    ag = a.params.agent_names[0]
    for key in ("events_total", "bridges_in_area", "bridges_hit", "components_hit"):
        assert a.regional_stats[ag][key] == b.regional_stats[ag][key]
    # seme diverso -> eventi diversi
    c = _env(STRESS, seed=12)
    _step(c, 8)
    assert c.regional_events != a.regional_events


def test_footprint_correlation_fixed_center():
    env0 = _env(scenario="PAPERA-SHORT-A")     # quattro ponti
    b0 = env0.bridges[sorted(env0.bridges)[0]]
    dists = sorted(haversine_km(b.lat, b.lon, b0.lat, b0.lon) for b in env0.bridges.values())
    radius = 0.5 * (dists[2] + dists[3])     # tre ponti dentro, gli altri fuori
    ov = {**STRESS, "regional_event_center_mode": "fixed", "regional_event_center_lat": float(b0.lat),
          "regional_event_center_lon": float(b0.lon), "regional_event_radius_km": radius,
          "regional_event_decay_model": "uniform", "regional_event_rate_y": 40.0}
    env = _env(ov, scenario="PAPERA-SHORT-A")
    _step(env)
    assert env.regional_events
    q = env.regional_events[0]["quarter"]
    inside = {bid for bid, b in env.bridges.items()
              if haversine_km(b.lat, b.lon, b0.lat, b0.lon) <= radius}
    hit = {bid for bid, _, _ in _regional_pending(env)}
    assert hit == inside and len(inside) == 3 and len(env.bridges) == 4
    assert {int(x) for x, _ in env.regional_events[0]["bridges"]} == inside
    assert all(e["quarter"] == q for e in env.regional_events)


def test_exposure_filter_and_multiplier():
    env = _env()
    bridges = list(env.bridges.values())
    bridges[0].crosses_water = True
    bridges[0].scour_critical = True
    bridges[1].crosses_water = False
    p = SimParameters(regional_event_requires="crosses_water", regional_event_multipliers={"scour_critical": 3.0})
    assert rg.exposure_factor(p, bridges[0]) == pytest.approx(3.0)
    assert rg.exposure_factor(p, bridges[1]) == 0.0
    assert rg.intensity(p, 0.0) == pytest.approx(1.0)
    assert rg.intensity(p, p.regional_event_radius_km + 1) == 0.0


# -- informazione ---------------------------------------------------------------

def test_damage_hidden_until_inspection_and_event_public():
    env = _first_event_env({**STRESS, "observation_mode": "inspected"})
    agent = env.params.agent_names[0]
    pend = _regional_pending(env)
    assert pend
    bid, _, comp = pend[0]
    c = env.bridges[bid].components[comp]
    # stesso seme senza danno: stessa osservazione, condizione vera piu' alta
    ref = _env({**STRESS, "observation_mode": "inspected", "regional_event_hit_prob": 0.0})
    _step(ref, env.quarter - 1)
    r = ref.bridges[bid].components[comp]
    assert (r.observed_condition_min, r.observed_condition_max) == (c.observed_condition_min, c.observed_condition_max)
    assert r.current_condition > c.current_condition + 1e-6
    assert r.epistemic_uncertainty == c.epistemic_uncertainty
    obs = sanitized_observation(env, agent)
    assert obs["regional_events"] and obs["regional_events"][0]["quarter"] == env.regional_events[0]["quarter"]
    mine = {b["id"]: b for b in obs["my_bridges"]}
    assert mine[bid]["regional_event_uninspected"] is True
    text = json.dumps(obs)
    assert "sudden_pending" not in text and "bridges_hit" not in text
    snap = belief_snapshot(env, agent, seed=5)
    assert "regional_stats" not in snap and "sudden_pending" not in snap
    assert snap["regional_events"] == env.to_dict()["regional_events"]
    clone = env.belief_clone(rng=np.random.default_rng(3))
    assert clone.sudden_pending == {} and clone.regional_events == env.regional_events
    assert all(v["bridges_hit"] == 0 for v in clone.regional_stats.values())
    # ispezione: danno rilevato, area "ispezionata"
    env.step({agent: {"maintenance": {str(bid): "L1_inspect"}}})
    st = env.regional_stats[agent]
    assert st["detected_by_inspection"] + st["detected_by_other_announcement"] >= 1
    assert all(x[0] != bid or x[1] == env.quarter - 1 for x in _regional_pending(env))
    obs2 = env.get_agent_observations(agent)
    b2 = next(b for b in obs2["my_bridges"] if b["id"] == bid)
    last = env.regional_events[-1]
    if last["quarter"] < env.quarter - 1 or bid not in {int(x) for x, _ in last["bridges"]}:
        assert b2["regional_event_uninspected"] is False


def test_announced_fraction_reveals():
    env = _first_event_env({**STRESS, "regional_event_announce_prob": 1.0})
    agent = env.params.agent_names[0]
    st = env.regional_stats[agent]
    assert st["components_hit"] > 0
    assert st["announced"] == st["components_hit"]
    assert _regional_pending(env) == []


def test_state_roundtrip_and_metrics():
    env = _first_event_env(STRESS)
    back = BridgeEnv.from_dict(json.loads(json.dumps(env.to_dict())))
    assert back.regional_events == env.regional_events
    assert back.regional_stats == env.regional_stats
    assert _regional_pending(back) == _regional_pending(env)
    m = rg.metrics(env, env.params.agent_names[0])
    assert m["regional_events_total"] >= 1 and m["regional_pending_undetected"] == len(_regional_pending(env))


def test_lab_outputs_carry_counts():
    raw = {"name": "regional", "scenario": "TRAIN-01", "horizon_quarters": 6,
           "agents": [{"id": "dn", "type": "Basic", "params": {"policy": "do_nothing"}}],
           "grid": {f"params.{k}": [v] for k, v in STRESS.items()},
           "replicates": 1, "seed": 1, "outputs": {"dir": "results/regional", "formats": ["csv"]}}
    payload = expand_runs(spec_from_dict(raw), load_scenario("TRAIN-01"))[0]
    run_row, steps = run_one(payload)
    assert run_row["final_regional_events_total"] > 0
    assert "final_regional_bridges_hit" in run_row and "final_regional_pending_undetected" in run_row
    assert sum(s["metric_regional_events_quarter"] for s in steps) == run_row["final_regional_events_total"]


def test_backend_config_exposes_regional_fields():
    from backend.app import CONFIG_FIELDS
    keys = {f["key"] for f in CONFIG_FIELDS}
    assert {f["key"] for f in rg.UI_FIELDS} <= keys
    defaults = SimParameters()
    for f in rg.UI_FIELDS:
        assert hasattr(defaults, f["key"])


# -- leak check -------------------------------------------------------------------

@pytest.mark.parametrize("trajectory", [False, True])
def test_leak_check_clean_with_regional_events(trajectory):
    rep = run_leak_check("TRAIN-01", params={"observation_mode": "inspected", "policy_sandbox": True,
                                             "strict_information": True},
                         quarters=8, regional_events=True, trajectory=trajectory,
                         policy="inspect-only" if trajectory else "inspect")
    assert rep["regional_events_enabled"] and rep["regional_events"] > 0
    assert rep["clean"], rep["divergences"][:5]


# -- regime -----------------------------------------------------------------------

def test_regime_trigger_forces_special_inspection_in_area():
    reg = regime_from_dict({"id": "regional_special_inspection",
                            "obligations": [{"trigger": "regional_event_in_area", "action": "L1_inspect",
                                             "grace_quarters": 1}],
                            "enforcement": {"mode": "forced"}})
    register_regime(reg, overwrite=True)
    ov = {**STRESS, "regulatory_regime": "regional_special_inspection", "regional_event_rate_y": 1.0}
    env = _first_event_env(ov)
    agent = env.params.agent_names[0]
    ev = env.regional_events[0]
    area = {int(b) for b, _ in ev["bridges"]}
    assert area
    ob = env.regime.obligations[0]
    active = {bid for bid, b in env.bridges.items() if env._regime_trigger_active(b, ob)}
    assert area <= active
    outside = set(env.bridges) - {int(b) for e in env.regional_events for b, _ in e["bridges"]}
    assert not (active & outside)
    _step(env, 3)   # do_nothing: l'autorita' ispeziona d'ufficio dopo la tolleranza
    assert env.regime_stats[agent]["forced_actions"] > 0
    for bid in area:
        assert env.bridges[bid].last_inspection_quarter > ev["quarter"]
    # con un'intensita' minima sopra 1 il trigger non scatta mai
    reg2 = regime_from_dict({"id": "regional_none", "obligations": [
        {"trigger": "regional_event_in_area", "action": "L1_inspect", "threshold": 1.01}]})
    assert not any(env._regime_trigger_active(b, reg2.obligations[0]) for b in env.bridges.values()
                   if rg.area_status(env, b) is not None)


def test_regime_trigger_inactive_without_channel():
    env = _env()
    reg = regime_from_dict({"id": "x_regional", "obligations": [
        {"trigger": "regional_event_in_area", "action": "L1_inspect"}]})
    _step(env, 2)
    assert not any(env._regime_trigger_active(b, reg.obligations[0]) for b in env.bridges.values())
