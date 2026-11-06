"""Inventario iniziale noto, soglia etica e osservazione oracolo (tutti opt-in)."""
import copy
import json
import os
from types import SimpleNamespace

import pytest

from infralab.agents.basic import BasicAgent
from infralab.lab.build import build_env_from_scenario
from infralab.lab.runner import expand_runs, run_one
from infralab.lab.spec import spec_from_dict
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory


def _env(overrides=None, scenario=None, seed=7):
    return build_env_from_scenario(scenario or load_scenario("TRAIN-01"),
                                   param_overrides=overrides or {}, seed=seed)


def _make_known_critical(env, bid=1, comp="pile", true_cond=0.35, seen=0.15):
    """Ponte con componente noto sotto soglia (stato vero ancora sopra il collasso)."""
    b = env.bridges[bid]
    c = b.components[comp]
    c.apply_damage(c.current_condition - true_cond, env.params.deterioration_exponent)
    c.observed_condition_min, c.observed_condition_max = seen, seen + 0.1
    env.refresh_bridge_reliability(b)
    return b


def _do_nothing_step(env):
    agent = env.params.agent_names[0]
    plan, _ = BasicAgent(agent, "do_nothing").decide(env)
    return env.step({agent: plan})


# -- default invariati ---------------------------------------------------------

def test_defaults_are_neutral():
    p = SimParameters()
    assert p.initial_inventory_known is False
    assert p.ethical_floor_enabled is False
    assert p.oracle_observation is False
    assert p.ethical_floor_beta is None


def test_defaults_keep_train01_golden_snapshot():
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


@pytest.mark.parametrize("bad", [{"ethical_floor_condition": 1.5}, {"ethical_floor_condition": -0.1},
                                 {"ethical_floor_beta": float("nan")}, {"oracle_observation": "si"}])
def test_validation(bad):
    with pytest.raises(ValueError):
        SimParameters(**bad)


# -- inventario iniziale -------------------------------------------------------

def test_initial_inventory_gives_known_condition_at_zero_cost():
    base = _env()
    env = _env({"initial_inventory_known": True})
    agent = env.params.agent_names[0]
    for b in base.bridges.values():
        assert b.min_observed_condition() is None
    obs = env.get_agent_observations(agent)
    for b in obs["my_bridges"]:
        assert b["min_observed_condition"] is not None
        assert b["last_inspection_quarter"] == 0
        assert b["cda"] != "Sconosciuta"
    assert env.budgets == base.budgets
    assert env.emissions_tco2 == base.emissions_tco2
    # non e' un'azione del gestore
    for b in env.bridges.values():
        assert b.last_action_name is None
        assert b.next_inspection_quarter >= 1


@pytest.mark.parametrize("streams", ["shared", "keyed"])
def test_initial_inventory_does_not_shift_other_streams(streams):
    """Il flusso dedicato non sposta degrado e altri processi: stesse condizioni vere."""
    a = _env({"rng_streams": streams})
    b = _env({"rng_streams": streams, "initial_inventory_known": True})
    agent = a.params.agent_names[0]
    for _ in range(6):
        a.step({agent: {"maintenance": {}}})
        b.step({agent: {"maintenance": {}}})
        assert [x.min_condition for x in a.bridges.values()] == [x.min_condition for x in b.bridges.values()]


# -- soglia etica --------------------------------------------------------------

def test_floor_with_budget_turns_do_nothing_into_intervention():
    env = _env({"ethical_floor_enabled": True, "initial_budget": 50_000_000})
    agent = env.params.agent_names[0]
    b = _make_known_critical(env)
    costs, _ = _do_nothing_step(env)
    assert b.active_maintenance is not None
    st = env.floor_stats[agent]
    assert st["interventions"] == 1 and st["spend"] > 0
    assert st["closure_bridge_quarters"] == 0
    floor_costs = [c for c in costs[agent] if c.get("ethical_floor")]
    assert len(floor_costs) == 1 and floor_costs[0]["cost"] == pytest.approx(st["spend"])
    # l'intervento scelto riporta il componente noto sopra soglia (guadagno atteso)
    gains = env.maint_ops[floor_costs[0]["action"]]["gains"]
    assert 0.15 + gains.get("pile", 0.0) >= env.params.ethical_floor_condition


def test_floor_without_budget_closes_and_counts():
    env = _env({"ethical_floor_enabled": True, "initial_budget": 0})
    agent = env.params.agent_names[0]
    b = _make_known_critical(env)
    _do_nothing_step(env)
    assert b.active_maintenance is None
    assert b.floor_closure is not None and b.traffic_restriction_factor == 0.0
    st = env.floor_stats[agent]
    assert st["interventions"] == 0 and st["closures"] == 1 and st["budget_blocked"] == 1
    assert st["closure_bridge_quarters"] == 1
    _do_nothing_step(env)
    assert env.floor_stats[agent]["closure_bridge_quarters"] == 2
    assert env.floor_stats[agent]["closures"] == 1
    # il gestore non puo' riaprire per scelta propria
    assert env._action_block_reason(agent, b, "chiusura_ponte") in ("ethical_floor_closure", "unknown_action")
    # regime separato: nessuna sanzione contata
    assert env.regime_stats[agent]["closure_bridge_quarters"] == 0

    # arriva la cassa: parte l'intervento, il ponte resta chiuso fino al completamento,
    # poi la condizione nota torna sopra soglia e il ponte riapre
    env.budgets[agent] = 50_000_000
    _do_nothing_step(env)
    assert b.active_maintenance is not None and env.floor_stats[agent]["interventions"] == 1
    assert b.traffic_restriction_factor == 0.0
    for _ in range(8):
        _do_nothing_step(env)
        if not getattr(b, "floor_closure", None):
            break
    assert not b.floor_closure
    assert b.traffic_restriction_factor == 1.0
    assert b.min_observed_condition() >= env.params.ethical_floor_condition


def test_floor_capacity_blocked_closes():
    env = _env({"ethical_floor_enabled": True, "initial_budget": 50_000_000, "works_capacity": 1})
    agent = env.params.agent_names[0]
    b = _make_known_critical(env)
    other = env.bridges[2]
    other.active_maintenance = {"action_name": "ripristino_completo", "ends_at_quarter": 99}
    _do_nothing_step(env)
    assert b.active_maintenance is None and b.floor_closure is not None
    assert env.floor_stats[agent]["capacity_blocked"] == 1


def test_floor_uses_known_state_not_truth():
    """Un ponte vero in pessimo stato ma mai ispezionato non attiva la soglia."""
    env = _env({"ethical_floor_enabled": True, "initial_budget": 50_000_000})
    agent = env.params.agent_names[0]
    b = env.bridges[1]
    c = b.components["pile"]
    c.apply_damage(c.current_condition - 0.15, env.params.deterioration_exponent)
    assert b.min_observed_condition() is None
    _do_nothing_step(env)
    assert env.floor_stats[agent]["interventions"] == 0 and not getattr(b, "floor_closure", None)


def test_floor_state_survives_planning_clone():
    env = _env({"ethical_floor_enabled": True, "initial_budget": 0})
    agent = env.params.agent_names[0]
    _make_known_critical(env)
    _do_nothing_step(env)
    clone = env.planning_clone(salt=1)
    assert clone.params.ethical_floor_enabled is True
    assert clone.floor_stats[agent] == env.floor_stats[agent]
    assert clone.bridges[1].floor_closure == env.bridges[1].floor_closure


def test_floor_metrics_exported_by_lab_runner():
    scenario = copy.deepcopy(load_scenario("TRAIN-01"))
    scenario["bridges"][0]["eta_iniziale_anni"] = 75     # condizione nota ~0.3
    spec = spec_from_dict({
        "name": "floor", "scenario": "TRAIN-01", "horizon_quarters": 3,
        "agents": [{"id": "dn", "type": "Basic", "policy": "do_nothing"}],
        "grid": {"params.initial_inventory_known": [True], "params.ethical_floor_enabled": [True],
                 "params.ethical_floor_condition": [0.4], "params.initial_budget": [50_000_000]},
        "replicates": 1, "seed": 1, "outputs": {"dir": "results/floor", "formats": ["csv"]},
    })
    payload = expand_runs(spec, scenario)[0]
    run_row, steps = run_one(payload)
    assert run_row["final_floor_interventions"] >= 1
    assert run_row["final_floor_spend"] > 0
    assert "final_floor_closure_bridge_quarters" in run_row
    assert steps[0]["metric_floor_spend_quarter"] > 0
    assert steps[0]["metric_spent_quarter"] >= steps[0]["metric_floor_spend_quarter"]


def test_floor_metrics_zero_when_disabled():
    spec = spec_from_dict({
        "name": "nofloor", "scenario": "TRAIN-01", "horizon_quarters": 2,
        "agents": [{"id": "dn", "type": "Basic", "policy": "do_nothing"}], "grid": {},
        "replicates": 1, "seed": 1, "outputs": {"dir": "results/nofloor", "formats": ["csv"]},
    })
    run_row, _ = run_one(expand_runs(spec, load_scenario("TRAIN-01"))[0])
    assert run_row["final_floor_interventions"] == 0
    assert run_row["final_floor_closure_bridge_quarters"] == 0


# -- oracolo -------------------------------------------------------------------

def test_oracle_gives_true_state_to_agents_only():
    env = _env({"oracle_observation": True, "observation_mode": "inspected"})
    agent = env.params.agent_names[0]
    for _ in range(2):
        env.step({agent: {"maintenance": {}}})
    obs = env.get_agent_observations(agent)
    for bd in obs["my_bridges"]:
        b = env.bridges[bd["id"]]
        assert bd["observation_mode"] == "oracle"
        assert bd["min_observed_condition"] == pytest.approx(b.min_condition)
        for name, comp in bd["components"].items():
            assert comp["observed_condition_min"] == pytest.approx(b.components[name].current_condition)
        assert bd["annual_reliability_index"] == pytest.approx(b.annual_reliability_index)
        # lo stato del motore non e' stato toccato: nessuna ispezione e' avvenuta
        assert b.min_observed_condition() is None
        assert all(c.observed_condition_min is None for c in b.components.values())
    # i trigger di legge continuano a vedere lo stato accertato (nessuna evidenza)
    ob = SimpleNamespace(trigger="condition_below_threshold", threshold=1.01)
    assert not any(env._regime_trigger_active(b, ob) for b in env.bridges.values())


def test_oracle_off_keeps_assessed_view():
    env = _env({"observation_mode": "inspected"})
    agent = env.params.agent_names[0]
    obs = env.get_agent_observations(agent)
    assert all(bd["observation_mode"] == "inspected" for bd in obs["my_bridges"])
    assert all(bd["min_observed_condition"] is None for bd in obs["my_bridges"])


def test_oracle_lets_danger_only_act():
    """Con l'oracolo una regola che non ispeziona vede il pericolo vero."""
    env = _env({"oracle_observation": True, "initial_budget": 50_000_000})
    agent = env.params.agent_names[0]
    b = env.bridges[1]
    c = b.components["pile"]
    c.apply_damage(c.current_condition - 0.3, env.params.deterioration_exponent)
    plan, _ = BasicAgent(agent, "danger_only").decide(env)
    assert "1" in plan["maintenance"]
    blind = _env({"initial_budget": 50_000_000})
    bb = blind.bridges[1].components["pile"]
    bb.apply_damage(bb.current_condition - 0.3, blind.params.deterioration_exponent)
    plan_blind, _ = BasicAgent(agent, "danger_only").decide(blind)
    assert plan_blind["maintenance"] == {}


@pytest.mark.parametrize("mode", ["full", "inspected"])
def test_floor_beta_threshold_on_assessed_beta(mode):
    """Soglia sul beta annuo stimato: con una soglia irraggiungibile ogni ponte e' 'pericoloso'."""
    env = _env({"ethical_floor_enabled": True, "ethical_floor_condition": 0.0, "ethical_floor_beta": 50.0,
                "observation_mode": mode, "initial_budget": 50_000_000})
    agent = env.params.agent_names[0]
    _do_nothing_step(env)
    assert env.floor_stats[agent]["interventions"] == len(env.bridges)
