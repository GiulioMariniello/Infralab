"""Sandbox delle policy (policy_sandbox=True): integrita' del benchmark.

Ogni via di fuga individuata dall'audit e' chiusa sotto sandbox:
1. nessuna policy riceve il motore (decide riceve un PlanningHandle; info del gym senza "env");
2. tool agentici: state() dall'osservazione sanitizzata, simulate() sul mondo del gestore
   con futuro casuale proprio;
3. osservazione compatta dell'agente Custom dalle stime del gestore;
4. "full" resta dichiaratamente vero; 5. in "inspected" danno idraulico non rilevato,
   referto della prova di carico, curva di monitoraggio, parametri latenti e
   moltiplicatori non arrivano alla policy; 6. seme rimosso; 7. catalogo copiato.
Default: comportamento storico (snapshot TRAIN-01 identico, motore passato alle policy).
"""
import json
import random

import pytest

from infralab.agents.adapters import LegacyAgentAdapter, make_agent
from infralab.agents.agentic import AgenticLLMAgent, make_simulation_tools
from infralab.agents.legacy import AgentInterface, LLMAgent
from infralab.agents.protocol import AssetEnvGym
from infralab.agents.sandbox import (INSPECTED_DROPPED_COMPONENT_FIELDS, PlanningHandle, SandboxViolation,
                                     owner_view, policy_env, sanitized_observation)
from infralab.agents.scripted import ScriptedPluginAgent
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory


def _env(overrides=None, seed=3):
    base = {"observation_mode": "inspected", "policy_sandbox": True, "rng_streams": "keyed"}
    base.update(overrides or {})
    return build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides=base, seed=seed)


def _agent(env):
    return env.params.agent_names[0]


class _Spy(AgentInterface):
    """Policy che registra cio' che riceve."""

    def __init__(self, name):
        super().__init__(name)
        self.received = []

    def decide(self, env, **kwargs):
        self.received.append(env)
        return {"maintenance": {}}, "spy"


def _walk_objects(root, depth=3):
    """Oggetti raggiungibili da `root` via __dict__/contenitori/chiusure (profondita' limitata)."""
    seen, out, frontier = set(), [], [root]
    for _ in range(depth):
        nxt = []
        for obj in frontier:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            out.append(obj)
            if isinstance(obj, dict):
                nxt.extend(obj.values())
            elif isinstance(obj, (list, tuple, set)):
                nxt.extend(obj)
            elif callable(obj) and getattr(obj, "__closure__", None):
                nxt.extend(c.cell_contents for c in obj.__closure__)
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                nxt.extend(vars(obj).values())
        frontier = nxt
    return out


# -- default -------------------------------------------------------------------

def test_default_is_off_and_historical():
    assert SimParameters().policy_sandbox is False
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


def test_default_policies_still_receive_the_engine():
    env = _env({"policy_sandbox": False})
    assert policy_env(env, _agent(env)) is env
    spy = _Spy(_agent(env))
    adapter = LegacyAgentAdapter(spy)
    adapter.reset({}, {"env": env})
    adapter.act({})
    assert spy.received[0] is env
    gym = AssetEnvGym(lambda s: _env({"policy_sandbox": False}, seed=s), _agent(env))
    obs, info = gym.reset(seed=3)
    assert info["env"] is gym.env
    assert "seed" in obs["game_settings"]


def test_policy_sandbox_is_validated():
    with pytest.raises(ValueError):
        SimParameters(policy_sandbox="yes")


# -- 1. nessun motore alle policy ------------------------------------------------

def test_policy_receives_handle_not_engine():
    env = _env()
    spy = _Spy(_agent(env))
    adapter = LegacyAgentAdapter(spy)
    adapter.reset({}, {"env": env})
    adapter.act({})
    handle = spy.received[0]
    assert isinstance(handle, PlanningHandle)
    assert not any(isinstance(o, BridgeEnv) for o in _walk_objects(handle, depth=4))


def test_gym_info_and_obs_are_sanitized_under_sandbox():
    gym = AssetEnvGym(lambda s: _env(seed=s), _agent(_env()))
    obs, info = gym.reset(seed=3)
    assert "env" not in info and info["policy_sandbox"] is True
    assert "seed" not in obs["game_settings"]
    json.dumps(obs)   # JSON puro
    obs2, *_ , info2 = gym.step({"maintenance": {}})
    assert "env" not in info2 and "seed" not in obs2["game_settings"]


class _Malicious(AgentInterface):
    """Tenta ogni lettura vietata; registra quali sono riuscite."""

    ATTEMPTS = {
        "bridges": lambda e: e.bridges,
        "budgets": lambda e: e.budgets,
        "to_dict": lambda e: e.to_dict(),
        "rng": lambda e: e.rng,
        "assessed": lambda e: e._assessed_bridge,
        "step": lambda e: e.step,
        "seed": lambda e: e.params.seed,
        "params_write": lambda e: setattr(e.params, "observation_mode", "full"),
        "other_agent": lambda e: e.get_agent_observations("someone_else"),
        "truth_future": lambda e: e.predict_future_degradation,
    }

    def __init__(self, name):
        super().__init__(name)
        self.succeeded = []

    def decide(self, env, **kwargs):
        for key, fn in self.ATTEMPTS.items():
            try:
                fn(env)
                self.succeeded.append(key)
            except SandboxViolation:
                pass
        # ultima: senza try, la violazione arriva all'adapter
        env.bridges
        return {"maintenance": {"1": "L1_inspect"}}, "never"


def test_malicious_agent_gets_nothing_and_is_recorded():
    env = _env()
    bad = _Malicious(_agent(env))
    adapter = LegacyAgentAdapter(bad)
    adapter.reset({}, {"env": env})
    plan = adapter.act({})
    assert bad.succeeded == []
    assert plan == {}                                   # la violazione annulla la decisione
    assert adapter.last_rationale[0] == "sandbox_violation"
    accesses = [v["access"] for v in adapter.sandbox_violations]
    assert "bridges" in accesses and "params.seed" in accesses and "to_dict" in accesses
    assert len(accesses) == len(_Malicious.ATTEMPTS) + 1


# -- 2. tool agentici --------------------------------------------------------------

def test_agentic_tools_use_estimates_and_own_randomness():
    env = _env()
    name = _agent(env)
    tools = make_simulation_tools(policy_env(env, name), name)
    # nessun motore raggiungibile dalle chiusure dei tool
    assert not any(isinstance(o, BridgeEnv) and o.params.seed == env.params.seed
                   for fn in tools.values() for o in _walk_objects(fn, depth=5))
    st = tools["state"]()
    obs = sanitized_observation(env, name)
    est = {b["id"]: b for b in obs["my_bridges"]}
    diffs = 0
    for a in st["assets"]:
        assert a["min_condition"] == round(est[a["id"]]["min_condition"], 3)
        diffs += abs(a["min_condition"] - round(env.bridges[a["id"]].min_condition, 3)) > 1e-9
    assert diffs > 0                                     # non e' lo stato vero
    random.seed(1)
    r1 = tools["simulate"]({}, quarters=2)
    assert set(r1) >= {"final_budget", "min_beta", "mean_condition", "still_active"}


def test_handle_clones_never_carry_engine_seed_or_streams():
    env = _env()
    handle = PlanningHandle.build(env, _agent(env))
    clone = handle.planning_clone()
    assert clone.params.seed != env.params.seed
    assert clone.stream_salt != 0 and clone.failure_stream_salt != 0
    assert handle._snapshot["params"]["seed"] != env.params.seed
    # stato nascosto riestratto: non coincide col vero
    for bid, b in env.bridges.items():
        for n, c in b.components.items():
            assert clone.bridges[bid].components[n].degradation_scale != c.degradation_scale


def test_agentic_agent_runs_sandboxed():
    env = _env()
    calls = {"n": 0}

    def llm(system, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"action": "run_code",
                               "code": "s = state()\nprint(simulate({}, quarters=2)['still_active'])\n"
                                       "print(len(feasible_actions()))"})
        return json.dumps({"action": "final_plan", "plan": {"maintenance": {}}, "explanation": "ok"})

    agent = AgenticLLMAgent(_agent(env), "Bilanciato", "Locale", llm_callable=llm)
    adapter = LegacyAgentAdapter(agent)
    adapter.reset({}, {"env": env})
    adapter.act({})
    transcript = adapter.last_rationale[1][1]
    sandbox_out = [t["content"] for t in transcript if t["role"] == "sandbox"][0]
    assert "True" in sandbox_out and "ERRORE" not in sandbox_out
    assert adapter.sandbox_violations == []


# -- 3. agente Custom ----------------------------------------------------------------

def test_scripted_observation_is_the_owner_estimate():
    env = _env()
    name = _agent(env)
    seen = {}

    def decider(obs):
        seen.update(obs)
        return {}

    adapter = LegacyAgentAdapter(ScriptedPluginAgent(name, decider=decider))
    adapter.reset({}, {"env": env})
    adapter.act({})
    est = {b["id"]: b for b in sanitized_observation(env, name)["my_bridges"]}
    for a in seen["assets"]:
        assert a["condition_min"] == round(est[a["id"]]["min_condition"], 4)
        assert a["beta"] == round(est[a["id"]]["reliability_index"], 3)


# -- 5./6./7. osservazione "inspected" sanitizzata ------------------------------------

def test_hydraulic_damage_hidden_until_inspected():
    env = _env()
    name = _agent(env)
    bridge = next(b for b in env.bridges.values() if b.manager == name)
    env._record_action_event(bridge, "danno_idraulico", "hazard", {"condition_drop": 0.2})
    bridge.hydraulic_damage_quarter = env.quarter

    def view():
        return next(b for b in sanitized_observation(env, name)["my_bridges"] if b["id"] == bridge.id)

    raw = next(b for b in env.get_agent_observations(name)["my_bridges"] if b["id"] == bridge.id)
    assert raw["hydraulic_damage_quarter"] == env.quarter          # la vista del motore lo mostra
    v = view()
    assert v["hydraulic_damage_quarter"] == 0
    assert not any(e["action"] == "danno_idraulico" for e in v["action_history"])
    env.step({name: {"maintenance": {}}})                        # nessuna ispezione: resta nascosto
    assert view()["hydraulic_damage_quarter"] == 0
    env.step({name: {"maintenance": {str(bridge.id): "L1_inspect"}}})
    v = view()
    assert v["hydraulic_damage_quarter"] == bridge.hydraulic_damage_quarter != 0
    assert any(e["action"] == "danno_idraulico" for e in v["action_history"])


def test_proof_load_report_from_assessed_state():
    env = _env()
    name = _agent(env)
    bridge = next(b for b in env.bridges.values() if b.manager == name)
    comp = next(iter(bridge.components))
    bridge.last_proof_load_report = {
        "quarter": env.quarter, "alpha": 1.2, "step_count": 3, "passed": True, "screened": False,
        "components": [{"component": comp, "proof_load_alpha": 1.2, "passed": True, "screened": False,
                        "prior_beta": 9.99, "proof_load_failure_probability": 0.01,
                        "posterior_beta_if_passed": 9.99, "step_reports": []}],
        "posterior_beta": 9.99, "selected_failure_probability_during_test": 0.01,
    }
    rep = next(b for b in sanitized_observation(env, name)["my_bridges"] if b["id"] == bridge.id)["last_proof_load_report"]
    est = env._assessed_bridge(bridge)
    assert rep["basis"] == "assessed" and rep["passed"] is True and rep["alpha"] == 1.2
    assert rep["posterior_beta"] == pytest.approx(est.posterior_reliability_index)
    assert rep["components"][0]["prior_beta"] == pytest.approx(est.components[comp].reliability_index)
    assert "selected_failure_probability_during_test" not in rep
    assert "proof_load_failure_probability" not in rep["components"][0]
    assert "9.99" not in json.dumps(rep)


def test_monitoring_curve_from_belief_and_hidden_component_fields():
    env = _env()
    name = _agent(env)
    bridge = next(b for b in env.bridges.values() if b.manager == name)
    bridge.future_curve_range = env.predict_future_degradation(bridge, 0.005)   # centrata sul vero
    v = next(b for b in sanitized_observation(env, name)["my_bridges"] if b["id"] == bridge.id)
    expected = json.loads(json.dumps(env.predict_future_degradation(env._assessed_bridge(bridge), 0.005)))
    truth = json.loads(json.dumps(bridge.future_curve_range))
    assert v["future_curve_range"] == expected != truth
    for c in v["components"].values():
        for f in INSPECTED_DROPPED_COMPONENT_FIELDS:
            assert f not in c


def test_seed_removed_and_catalogue_is_a_copy():
    for mode in ({"observation_mode": "full"}, {"observation_mode": "inspected"}, {"oracle_observation": True}):
        env = _env(mode)
        name = _agent(env)
        obs = sanitized_observation(env, name)
        assert "seed" not in obs["game_settings"]
        handle = policy_env(env, name)
        with pytest.raises(SandboxViolation):
            handle.params.seed
        obs["maintenance_options"]["L1_inspect"]["cost_fixed"] = -1
        handle.maint_ops["L1_inspect"]["cost_fixed"] = -1
        handle.get_agent_observations()["maintenance_options"].clear()
        assert env.maint_ops["L1_inspect"].get("cost_fixed") != -1
        assert handle.get_agent_observations()["maintenance_options"]


def test_full_mode_is_declared_truth():
    env = _env({"observation_mode": "full"})
    name = _agent(env)
    obs = sanitized_observation(env, name)
    assert obs["information"]["mode"] == "full"
    for b in obs["my_bridges"]:
        assert b["min_condition"] == pytest.approx(env.bridges[b["id"]].min_condition)


# -- oracolo -------------------------------------------------------------------------

def test_oracle_grant_gives_truth():
    env = _env({"oracle_observation": True})
    name = _agent(env)
    obs = sanitized_observation(env, name)
    assert obs["information"]["mode"] == "oracle"
    handle = PlanningHandle.build(env, name)
    clone = handle.planning_clone()
    for b in obs["my_bridges"]:
        real = env.bridges[b["id"]]
        for n, c in b["components"].items():
            assert c["observed_condition_min"] == pytest.approx(real.components[n].current_condition)
            assert clone.bridges[b["id"]].components[n].current_condition == pytest.approx(
                real.components[n].current_condition)
    assert clone.params.seed != env.params.seed        # stato vero, futuro comunque proprio


# -- agenti di serie sotto sandbox -----------------------------------------------------

BUILTIN = [
    {"type": "Euristico"},
    {"type": "Basic", "policy": "risk_based"},
    {"type": "Basic", "policy": "inspect_and_repair"},
    {"type": "Basic", "policy": "target_keeper"},
    {"type": "GA Orizzonte Variabile", "ga_params": {"pop_size": 4, "generations": 1, "horizon": 2}},
    {"type": "Algoritmo Memetico", "ga_params": {"pop_size": 3, "generations": 1, "horizon": 2,
                                                 "local_search_steps": 1}},
    {"type": "MCTS", "mcts_params": {"n_rollouts": 3, "n_candidates": 3, "rollout_depth": 2}},
    {"type": "RL Q-Learning"},
    {"type": "Umano"},
]


@pytest.mark.parametrize("config", BUILTIN, ids=lambda c: c["type"] + c.get("policy", ""))
@pytest.mark.parametrize("mode", ["inspected", "full"])
def test_builtin_agents_run_sandboxed(config, mode):
    random.seed(0)
    env = _env({"observation_mode": mode})
    name = _agent(env)
    adapter = make_agent(name, config)
    adapter.reset({}, {"env": env})
    for _ in range(2):
        plan = adapter.act({})
        env.step({name: plan})
    assert adapter.sandbox_violations == []


def test_llm_agent_prompt_built_from_sanitized_observation(monkeypatch):
    env = _env()
    name = _agent(env)
    captured = {}

    def fake(self, obs, history_dfs):
        captured["obs"] = obs
        captured["prompt"] = self._get_prompt(obs, None, None)
        return {}, ("", "")

    monkeypatch.setattr(LLMAgent, "_decide_with_local", fake)
    adapter = make_agent(name, {"type": "LLM", "provider": "Locale"})
    adapter.reset({}, {"env": env})
    adapter.act({})
    assert "seed" not in captured["obs"]["game_settings"]
    assert adapter.sandbox_violations == []


def test_sandbox_does_not_perturb_engine_trajectory():
    """Stesso seme: la sandbox non consuma i flussi del motore e, in "full", non cambia
    le decisioni di un agente che legge solo l'osservazione."""
    def run(sandbox):
        env = _env({"observation_mode": "full", "policy_sandbox": sandbox})
        name = _agent(env)
        adapter = make_agent(name, {"type": "Euristico"})
        adapter.reset({}, {"env": env})
        for _ in range(4):
            env.step({name: adapter.act({})})
        return canonical({bid: [c.current_condition for c in b.components.values()] + [env.budgets[name]]
                          for bid, b in env.bridges.items()})
    assert run(True) == run(False)


def test_owner_view_for_ui():
    env = _env()
    b = next(iter(env.bridges.values()))
    assert owner_view(env, b) is env._assessed_bridge(b)
    env_off = _env({"policy_sandbox": False})
    b2 = next(iter(env_off.bridges.values()))
    assert owner_view(env_off, b2) is b2


def test_backend_exposes_switch_and_hides_truth():
    from fastapi.testclient import TestClient

    import backend.app as backend_app
    backend_app.SESSIONS.clear()
    with TestClient(backend_app.app) as client:
        keys = {f["key"] for f in client.get("/api/config").json()["fields"]}
        assert {"policy_sandbox", "observation_mode", "oracle_observation", "planner_information"} <= keys
        res = client.post("/api/sessions", json={"scenario_id": "TRAIN-01", "agent_type": "Euristico", "seed": 42,
                                                 "params": {"policy_sandbox": True, "observation_mode": "inspected"}})
        assert res.status_code == 200, res.text
        sid = res.json()["session_id"]
        state = client.post(f"/api/sessions/{sid}/step", json={"steps": 1}).json()
        env = backend_app.SESSIONS[sid].env
        assert env.params.policy_sandbox is True
        for a in state["assets"]:
            real = env.bridges[a["id"]]
            assert a["min_condition"] == round(env._assessed_bridge(real).min_condition, 4)
        detail = client.get(f"/api/sessions/{sid}/assets/{state['assets'][0]['id']}").json()
        assert detail["id"] == state["assets"][0]["id"]


def test_lab_spec_grid_switch_runs_sandboxed_and_reproducible():
    """Lab YAML/JSON: `params.policy_sandbox` e' un normale override di griglia; la run
    sandboxata di un pianificatore resta riproducibile per seme."""
    from infralab.lab.runner import expand_runs, run_one
    from infralab.lab.spec import spec_from_dict
    spec = spec_from_dict({
        "name": "sandbox", "scenario": "TRAIN-01", "horizon_quarters": 2, "replicates": 1, "seed": 5,
        "agents": [{"id": "mcts", "type": "MCTS",
                    "params": {"mcts_params": {"n_rollouts": 3, "rollout_depth": 1, "n_candidates": 3}}}],
        "grid": {"params.policy_sandbox": [True], "params.observation_mode": ["inspected"]},
    })
    payload = expand_runs(spec, load_scenario("TRAIN-01"))[0]
    row_a, steps_a = run_one(dict(payload))
    row_b, steps_b = run_one(dict(payload))
    row_a.pop("duration_s"); row_b.pop("duration_s")
    assert row_a["quarters_survived"] == 2
    assert json.dumps(steps_a, sort_keys=True, default=str) == json.dumps(steps_b, sort_keys=True, default=str)
