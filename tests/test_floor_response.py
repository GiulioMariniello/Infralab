"""Risposta della soglia etica a un problema noto (ethical_floor_response, opt-in)."""
import json

import pytest

from infralab.settings.config import SimParameters
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory
from tests.test_floor_oracle import _do_nothing_step, _env, _make_known_critical

FLOOR = {"ethical_floor_enabled": True, "ethical_floor_condition": 0.2}


def test_default_is_repair_or_close_and_golden_unchanged():
    assert SimParameters().ethical_floor_response == "repair_or_close"
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


def test_validation():
    with pytest.raises(ValueError):
        SimParameters(ethical_floor_response="ripara")


def test_default_response_repairs_when_cash_allows():
    env = _env(FLOOR)
    b = _make_known_critical(env)
    _do_nothing_step(env)
    assert b.active_maintenance is not None
    assert not getattr(b, "floor_closure", None)


def test_close_only_never_repairs_and_closes():
    env = _env({**FLOOR, "ethical_floor_response": "close_only"})
    b = _make_known_critical(env)
    agent = env.params.agent_names[0]
    _do_nothing_step(env)
    assert b.active_maintenance is None
    assert b.floor_closure and b.traffic_restriction_factor == 0.0
    st = env.floor_stats[agent]
    assert st["interventions"] == 0 and st["spend"] == 0 and st["closures"] == 1
    assert st["budget_blocked"] == 0 and st["capacity_blocked"] == 0
    _do_nothing_step(env)                      # resta chiuso finche' lo stato noto e' sotto soglia
    assert b.floor_closure and b.traffic_restriction_factor == 0.0
    assert env.floor_stats[agent]["closures"] == 1


def test_restrict_only_limits_load_and_keeps_bridge_open():
    env = _env({**FLOOR, "ethical_floor_response": "restrict_only"})
    b = _make_known_critical(env)
    _do_nothing_step(env)
    assert b.active_maintenance is None
    assert b.floor_closure
    assert b.traffic_restriction_factor == pytest.approx(float(env.params.posting_restriction_factor))
    assert env.floor_stats[env.params.agent_names[0]]["interventions"] == 0


def test_restriction_is_lifted_when_known_state_recovers():
    env = _env({**FLOOR, "ethical_floor_response": "restrict_only"})
    b = _make_known_critical(env)
    _do_nothing_step(env)
    assert b.floor_closure
    c = b.components["pile"]
    c.observed_condition_min, c.observed_condition_max = 0.6, 0.7
    _do_nothing_step(env)
    assert not getattr(b, "floor_closure", None)
    assert b.traffic_restriction_factor == pytest.approx(1.0)
