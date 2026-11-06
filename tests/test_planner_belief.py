"""Pianificatori sulla conoscenza del gestore e obiettivo allineato al benchmark (opt-in).

- planner_information="belief": la copia del pianificatore (planning_clone) non contiene
  lo stato vero nascosto; ogni copia e' una nuova estrazione dalla credenza del gestore;
  i flussi casuali del motore reale non vengono toccati; l'oracolo ripristina il vero.
- planner_objective="risk_adjusted": i rollout valgono -(spesa + perdita attesa +
  lambda x deficit di sicurezza).
- Default: comportamento storico (snapshot TRAIN-01 identico).
"""
import json
import random

import numpy as np
import pytest

from infralab.agents.basic import BasicAgent
from infralab.agents.legacy import (MCTSAgent, MemeticAgent, RollingHorizonGAAgent, agent_factory,
                                    planner_safety_deficit, planner_step_cost)
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.persistence import NumpyJSONEncoder
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory

HIDDEN = BridgeEnv.BELIEF_HIDDEN_COMPONENT_FIELDS


def _env(overrides=None, seed=3):
    base = {"observation_mode": "inspected", "planner_information": "belief", "rng_streams": "keyed"}
    base.update(overrides or {})
    return build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides=base, seed=seed)


def _canon(obj):
    return json.loads(json.dumps(obj, sort_keys=True, cls=NumpyJSONEncoder))


def _agent(env):
    return env.params.agent_names[0]


def _hidden_state(env):
    return {(bid, n): tuple(getattr(c, f) for f in HIDDEN)
            for bid, b in env.bridges.items() for n, c in b.components.items()}


# -- default e validazione -------------------------------------------------------

def test_defaults_are_historical():
    p = SimParameters()
    assert p.planner_information == "truth"
    assert p.planner_objective == "asset_value"
    assert p.planner_safety_weight == 0.0


def test_defaults_keep_train01_golden_snapshot():
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


@pytest.mark.parametrize("bad", [{"planner_information": "oracle"}, {"planner_objective": "npv"},
                                 {"planner_safety_weight": -1.0}, {"planner_safety_weight": float("nan")}])
def test_validation(bad):
    with pytest.raises(ValueError):
        SimParameters(**bad)


def test_truth_clone_is_the_historical_copy():
    env = _env({"planner_information": "truth"})
    for _ in range(2):
        env.step({_agent(env): {"maintenance": {}}})
    clone = env.planning_clone(salt=5)
    assert not hasattr(clone, "planner_view")
    assert clone.stream_salt == 0 and "stream_salt" not in env.to_dict()
    assert _hidden_state(clone) == _hidden_state(env)


# -- copia "belief" ----------------------------------------------------------------

def test_belief_clone_does_not_contain_truth_of_unobserved_component():
    env = _env()
    for _ in range(3):
        env.step({_agent(env): {"maintenance": {}}})
    comp = env.bridges[1].components["impalcato"]
    assert comp.observed_condition_min is None
    # stato vero volutamente lontano dalla credenza nominale
    comp.apply_damage(comp.current_condition - 0.35, env.params.deterioration_exponent)
    truth = comp.current_condition
    draws = [env.planning_clone(rng=np.random.default_rng(k)).bridges[1].components["impalcato"].current_condition
             for k in range(40)]
    assert all(abs(d - truth) > 0.2 for d in draws)          # credenza nominale, non il vero
    assert np.std(draws) > 1e-4                               # una nuova estrazione per copia
    clone = env.planning_clone(rng=np.random.default_rng(0))
    assert clone.planner_view == "belief"
    assert clone.bridges[1].components["impalcato"].degradation_scale != comp.degradation_scale


def test_belief_clone_never_reads_hidden_truth():
    """Due mondi veri diversi con la stessa conoscenza del gestore danno la stessa copia."""
    env_a, env_b = _env(), _env()
    for env in (env_a, env_b):
        for _ in range(3):
            env.step({_agent(env): {"maintenance": {}}})
    obs = env_a.bridges[2].components["pile"]
    obs.observed_condition_min, obs.observed_condition_max = 0.55, 0.75
    env_b.bridges[2].components["pile"].observed_condition_min = 0.55
    env_b.bridges[2].components["pile"].observed_condition_max = 0.75
    rng = np.random.default_rng(99)
    for b in env_b.bridges.values():
        b.hydraulic_damage_quarter = 3
        b.action_history.append({"quarter": 3, "action": "danno_idraulico", "status": "hazard"})
        for c in b.components.values():
            c.apply_damage(rng.uniform(0.05, 0.4), env_b.params.deterioration_exponent)
            for f in ("degradation_scale", "traffic_sensitivity", "resistance_bias", "demand_bias"):
                setattr(c, f, getattr(c, f) * rng.uniform(0.7, 1.3))
            c.last_process_multiplier = 3.0
        env_b.refresh_bridge_reliability(b)
        # stessa conoscenza: epistemica uguale a quella di env_a
        for n, c in b.components.items():
            c.epistemic_uncertainty = env_a.bridges[b.id].components[n].epistemic_uncertainty
    assert _hidden_state(env_a) != _hidden_state(env_b)
    ca = env_a.planning_clone(salt=1, rng=np.random.default_rng(5))
    cb = env_b.planning_clone(salt=1, rng=np.random.default_rng(5))
    assert _hidden_state(ca) == _hidden_state(cb)
    for bid in ca.bridges:
        assert ca.bridges[bid].annual_failure_probability == cb.bridges[bid].annual_failure_probability
        assert cb.bridges[bid].hydraulic_damage_quarter == 0
        assert not any(e.get("action") == "danno_idraulico" for e in cb.bridges[bid].action_history)


def test_belief_draws_for_observed_component_follow_the_interval():
    env = _env()
    c = env.bridges[1].components["spalle"]
    c.observed_condition_min, c.observed_condition_max = 0.50, 0.70
    draws = np.array([env.planning_clone(rng=np.random.default_rng(k)).bridges[1].components["spalle"].current_condition
                      for k in range(300)])
    assert abs(draws.mean() - 0.60) < 0.01
    assert abs(draws.std() - 0.20 / 3.92) < 0.01
    clone = env.planning_clone(rng=np.random.default_rng(1))
    cc = clone.bridges[1].components["spalle"]
    assert (cc.observed_condition_min, cc.observed_condition_max) == (0.50, 0.70)
    assert cc.epistemic_uncertainty == c.epistemic_uncertainty


def test_oracle_restores_truth_in_belief_mode():
    env = _env({"oracle_observation": True})
    for _ in range(2):
        env.step({_agent(env): {"maintenance": {}}})
    clone = env.planning_clone(rng=np.random.default_rng(3))
    assert clone.planner_view == "truth"
    assert _hidden_state(clone) == _hidden_state(env)
    for bid, b in env.bridges.items():
        assert clone.bridges[bid].annual_failure_probability == b.annual_failure_probability


def test_full_observation_keeps_truth():
    env = _env({"observation_mode": "full"})
    clone = env.planning_clone(rng=np.random.default_rng(3))
    assert clone.planner_view == "truth"
    assert _hidden_state(clone) == _hidden_state(env)


def test_belief_clone_future_streams_are_not_the_real_ones():
    env = _env()
    clone = env.planning_clone(rng=np.random.default_rng(4))
    assert clone.stream_salt != 0
    assert env._keyed_stream(2, 1, 77, 5).random() != clone._keyed_stream(2, 1, 77, 5).random()
    # ambiente reale: chiave storica
    assert env._keyed_stream(2, 1, 77, 5).random() == np.random.default_rng(
        [3, 1002, 1, 77, 5]).random()


@pytest.mark.parametrize("streams", ["keyed", "shared"])
def test_real_run_common_random_numbers_unchanged(streams):
    """Il motore reale fa lo stesso percorso con o senza copie belief dei pianificatori."""
    def run(with_clones, info):
        random.seed(0)
        env = _env({"rng_streams": streams, "planner_information": info,
                    "hydraulic_hazard_enabled": True, "posting_entry_rate_q": 0.05,
                    "failure_model": "reliability"}, seed=11)
        agent = _agent(env)
        for _ in range(6):
            plan, _ = BasicAgent(agent, "worst_first").decide(env)
            if with_clones:
                for _k in range(3):
                    c = env.planning_clone()
                    c.step({agent: {"maintenance": {}}})
                    c.step({agent: plan})
            env.step({agent: plan})
        d = _canon(env.to_dict())
        d.pop("params")
        return d
    reference = run(False, "truth")
    assert run(True, "belief") == reference
    assert run(False, "belief") == reference


# -- obiettivo dei pianificatori ---------------------------------------------------

def _below_target_env(**kw):
    return _env({"reliability_target_beta_annual": 6.5, **kw})


def test_safety_deficit_uses_exported_beta_and_target():
    env = _below_target_env()
    agent = _agent(env)
    expected = 0.0
    for b in env.bridges.values():
        beta = b.posterior_annual_reliability_index if b.posterior_annual_reliability_index is not None \
            else b.annual_reliability_index
        expected += max(0.0, b.reliability_details["target_beta_annual"] - beta)
    assert expected > 0
    assert planner_safety_deficit(env, agent) == pytest.approx(expected)


@pytest.mark.parametrize("lam", [0.0, 1e6])
def test_mcts_risk_adjusted_rollout_score(lam):
    env = _below_target_env()
    agent = _agent(env)
    clone = env.planning_clone(salt=3, rng=np.random.default_rng(8))
    ref = BridgeEnv.from_dict(clone.to_dict())
    ref.rng = np.random.default_rng(0)
    env.planning_clone = lambda *a, **k: clone
    plan = {"1": "L1_inspect"}
    mcts = MCTSAgent(agent, rollout_depth=1, objective="risk_adjusted", safety_weight=lam)
    value = mcts._rollout(plan, env)
    costs, _ = ref.step({agent: {"maintenance": plan}})
    spent = sum(c["cost"] for c in costs[agent])
    assert spent > 0
    expected = spent + ref.last_expected_loss[agent]["total"] + lam * planner_safety_deficit(ref, agent)
    assert value == pytest.approx(-expected)
    assert planner_step_cost(ref, agent, costs, lam) == pytest.approx(expected)


def test_objective_switch_changes_rollout_score():
    env = _below_target_env()
    agent = _agent(env)
    values = {}
    for key, kwargs in {"asset": {}, "risk0": {"objective": "risk_adjusted"},
                        "risk_lam": {"objective": "risk_adjusted", "safety_weight": 1e7}}.items():
        clone = env.planning_clone(salt=3, rng=np.random.default_rng(8))
        env_k = env
        env_k.planning_clone = lambda *a, **k: clone
        values[key] = MCTSAgent(agent, rollout_depth=1, **kwargs)._rollout({}, env_k)
        del env_k.planning_clone
    assert values["asset"] > 0 > values["risk0"] > values["risk_lam"]


def test_objective_from_sim_parameters_and_factory():
    env = _below_target_env(planner_objective="risk_adjusted", planner_safety_weight=2.0)
    ag = agent_factory("A", {"type": "MCTS", "mcts_params": {"n_rollouts": 2}})
    from infralab.agents.legacy import _planner_objective
    assert _planner_objective(ag, env) == ("risk_adjusted", 2.0)
    ag2 = agent_factory("A", {"type": "MCTS", "mcts_params": {"objective": "asset_value", "safety_weight": 0.0}})
    assert _planner_objective(ag2, env) == ("asset_value", 0.0)


@pytest.mark.parametrize("cls", [RollingHorizonGAAgent, MemeticAgent])
def test_ga_and_memetic_risk_adjusted(cls):
    env = _below_target_env(planner_objective="risk_adjusted", planner_safety_weight=1e6)
    agent = _agent(env)
    random.seed(1)
    ga = cls(agent, horizon=2, pop_size=4, generations=1)
    v_nothing = ga._evaluate([{}, {}], env)
    assert v_nothing < 0
    plan, _ = ga.decide(env)
    assert isinstance(plan["maintenance"], dict)


def test_mcts_decide_in_belief_mode_is_reproducible():
    def run():
        random.seed(5)
        env = _env({"planner_objective": "risk_adjusted"}, seed=5)
        ag = MCTSAgent(_agent(env), n_rollouts=6, rollout_depth=2, n_candidates=3)
        return [ag.decide(env)[0] for _ in range(1)]
    assert run() == run()


def test_termine_terminale_aggiunge_rischio_residuo():
    from infralab.agents.legacy import planner_terminal_cost
    from infralab.settings.config import SimParameters
    import pytest
    with pytest.raises(Exception):
        SimParameters(planner_terminal_quarters=-1)
    env = _env()
    env.params.planner_terminal_quarters = 0
    assert planner_terminal_cost(env, env.params.agent_names[0], 0.0) == 0.0
    env.params.planner_terminal_quarters = 40
    q = env.expected_quarterly_failure_loss(env.params.agent_names[0])["total"]
    assert planner_terminal_cost(env, env.params.agent_names[0], 0.0) == pytest.approx(40 * q)
