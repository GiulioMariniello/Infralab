"""Test dei plugin utente: loader sandbox, registry, degrado custom nel motore,
agente scriptato, e API. Il codice-plugin gira in sandbox (builtins ristretti,
import in whitelist); i modelli di default (weibull/gamma) restano invariati.
"""
import math

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
from infralab.agents.scripted import ScriptedPluginAgent
from infralab.lab.build import build_env_from_scenario
from infralab.plugins import registry
from infralab.plugins.loader import (
    PluginError,
    load_entrypoint,
    validate_agent,
    validate_degradation,
)
from tests.conftest import load_scenario

GOOD_DEGRADATION = '''
import math
def degrade(ctx):
    L = ctx["vita_utile_anni"]
    if L <= 0 or ctx["effective_age_years"] <= 0:
        return 1.0
    r = ctx["effective_age_years"] / L
    return math.exp(-ctx["scale"] * r ** ctx["shape"])
'''

# Degrado "aggressivo": condizione dimezzata rispetto al Weibull.
FAST_DEGRADATION = '''
import math
def degrade(ctx):
    L = ctx["vita_utile_anni"]
    if L <= 0 or ctx["effective_age_years"] <= 0:
        return 1.0
    r = ctx["effective_age_years"] / L
    base = math.exp(-ctx["scale"] * r ** ctx["shape"])
    return base * 0.5
'''

GOOD_AGENT = '''
def decide(obs):
    plan = {}
    for a in obs["assets"]:
        if a["condition_min"] < 0.9 and a["feasible_actions"]:
            act = min(a["feasible_actions"], key=a["feasible_actions"].get)
            plan[str(a["id"])] = act
    return plan
'''


class TestLoaderSandbox:
    def test_load_and_call(self):
        fn = load_entrypoint(GOOD_DEGRADATION, "degrade")
        v = fn({"vita_utile_anni": 100, "effective_age_years": 50, "scale": 1.0, "shape": 2.0})
        assert 0 < v < 1

    def test_missing_entrypoint(self):
        with pytest.raises(PluginError, match="deve definire"):
            load_entrypoint("x = 1", "degrade")

    def test_syntax_error(self):
        with pytest.raises(PluginError, match="sintassi"):
            load_entrypoint("def degrade(ctx) return 1", "degrade")

    def test_forbidden_import_blocked(self):
        with pytest.raises(PluginError, match="caricamento|Import"):
            load_entrypoint("import os\ndef degrade(ctx):\n    return 1", "degrade")

    def test_open_not_available(self):
        fn = load_entrypoint("def degrade(ctx):\n    return open('x')", "degrade")
        with pytest.raises(PluginError, match="Errore di esecuzione"):
            from infralab.plugins.loader import call_with_timeout
            call_with_timeout(fn, {}, 3.0)

    def test_validate_degradation_ok(self):
        ok, _ = validate_degradation(GOOD_DEGRADATION)
        assert ok

    def test_validate_degradation_bad_return(self):
        ok, msg = validate_degradation("def degrade(ctx):\n    return 'ciao'")
        assert not ok and "numero" in msg

    def test_validate_agent_ok(self):
        ok, _ = validate_agent(GOOD_AGENT)
        assert ok

    def test_validate_agent_bad_return(self):
        ok, msg = validate_agent("def decide(obs):\n    return [1,2]")
        assert not ok and "dict" in msg

    def test_templates_are_valid(self):
        from infralab.plugins.templates import TEMPLATES
        assert validate_agent(TEMPLATES["agent"])[0]
        assert validate_degradation(TEMPLATES["degradation"])[0]


class TestRegistry:
    def test_save_load_delete(self, isolated_persistence):
        registry.save_plugin("degradation", "mio-modello", GOOD_DEGRADATION)
        assert [p["name"] for p in registry.list_plugins("degradation")] == ["mio-modello"]
        fn = registry.load_degradation_model("mio-modello")
        assert callable(fn)
        assert registry.delete_plugin("degradation", "mio-modello") is True
        assert registry.list_plugins("degradation") == []

    def test_save_invalid_rejected(self, isolated_persistence):
        with pytest.raises(PluginError):
            registry.save_plugin("agent", "rotto", "def decide(obs):\n    return 5")

    def test_bad_name(self, isolated_persistence):
        with pytest.raises(PluginError, match="Nome plugin"):
            registry.save_plugin("agent", "../evil", GOOD_AGENT)


class TestCustomDegradationInEngine:
    def _env(self, model):
        scen = {
            "scenario_id": "T", "asset_type": "bridge",
            "params": {"seed": 7, "degradation_model": model},
            "bridges": [{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 5000, "eta_iniziale_anni": 20}],
        }
        return build_env_from_scenario(scen, agent_name="A", seed=7)

    def test_custom_model_changes_condition(self, isolated_persistence):
        registry.save_plugin("degradation", "fast", FAST_DEGRADATION)
        base = self._env("weibull")
        custom = self._env("custom:fast")
        for _ in range(4):
            base.step({"A": {"maintenance": {}}})
            custom.step({"A": {"maintenance": {}}})
        cond_base = base.bridges[1].min_condition
        cond_custom = custom.bridges[1].min_condition
        assert cond_custom < cond_base   # il modello aggressivo degrada di più

    def test_missing_plugin_falls_back(self, isolated_persistence):
        # 'custom:inesistente' non deve rompere: fallback al Weibull di default.
        env = self._env("custom:inesistente")
        env.step({"A": {"maintenance": {}}})
        assert env.bridges[1].min_condition is not None

    def test_projection_works_with_custom(self, isolated_persistence):
        registry.save_plugin("degradation", "fast", FAST_DEGRADATION)
        env = self._env("custom:fast")
        detail = env._project_component_life_cycle(env.bridges[1], next(iter(env.bridges[1].components.values())))
        assert detail["profile"] and all(0 <= p["condition"] <= 1 for p in detail["profile"])


class TestScriptedAgent:
    def test_agent_plans_and_applies(self, isolated_persistence):
        env = build_env_from_scenario(
            load_scenario("EXP-A-BUDGET"), agent_name="A", seed=42)
        agent = ScriptedPluginAgent("A", code=GOOD_AGENT)
        plan, rationale = agent.decide(env)
        assert "maintenance" in plan
        # applica alla partita vera senza errori
        env.step({"A": plan})
        assert env.quarter == 2

    def test_agent_ignores_infeasible_actions(self, isolated_persistence):
        env = build_env_from_scenario(load_scenario("EXP-A-BUDGET"), agent_name="A", seed=1)
        agent = ScriptedPluginAgent("A", code='def decide(obs):\n    return {"1": "azione_inventata"}')
        plan, _ = agent.decide(env)
        assert plan["maintenance"] == {}   # azione non fattibile scartata

    def test_agent_error_yields_empty_plan(self, isolated_persistence):
        env = build_env_from_scenario(load_scenario("EXP-A-BUDGET"), agent_name="A", seed=1)
        agent = ScriptedPluginAgent("A", code='def decide(obs):\n    return 1/0')
        plan, rationale = agent.decide(env)
        assert plan == {}

    def test_factory_builds_custom(self, isolated_persistence):
        registry.save_plugin("agent", "mio-agente", GOOD_AGENT)
        from infralab.agents.legacy import agent_factory
        agent = agent_factory("A", {"type": "Custom", "plugin": "mio-agente"})
        assert isinstance(agent, ScriptedPluginAgent)


@pytest.fixture()
def client(isolated_persistence):
    backend_app.SESSIONS.clear()
    research_module.ADMIN_TOKENS.clear()
    with TestClient(backend_app.app) as c:
        yield c


class TestApi:
    def test_templates_endpoint(self, client):
        d = client.get("/api/plugins/templates").json()
        assert "def decide(obs)" in d["agent"] and "def degrade(ctx)" in d["degradation"]

    def test_crud_and_validate(self, client):
        # validate senza salvare
        v = client.post("/api/plugins/validate", json={"kind": "degradation", "source": GOOD_DEGRADATION}).json()
        assert v["ok"]
        bad = client.post("/api/plugins/validate", json={"kind": "agent", "source": "def decide(obs):\n    return 9"}).json()
        assert not bad["ok"]
        # save
        r = client.post("/api/plugins/degradation", json={"name": "mio", "source": GOOD_DEGRADATION})
        assert r.status_code == 200, r.text
        assert "mio" in [p["name"] for p in client.get("/api/plugins/degradation").json()]
        # get source
        assert "def degrade" in client.get("/api/plugins/degradation/mio").json()["source"]
        # invalid save -> 400
        assert client.post("/api/plugins/agent", json={"name": "x", "source": "def decide(o):\n return 1"}).status_code == 400
        # delete
        assert client.delete("/api/plugins/degradation/mio").status_code == 200
        assert client.delete("/api/plugins/degradation/mio").status_code == 404

    def test_custom_degradation_in_config_options(self, client):
        client.post("/api/plugins/degradation", json={"name": "mymodel", "source": GOOD_DEGRADATION})
        fields = client.get("/api/config").json()["fields"]
        deg = next(f for f in fields if f["key"] == "degradation_model")
        assert "custom:mymodel" in deg["options"]

    def test_custom_agent_session(self, client):
        client.post("/api/plugins/agent", json={"name": "greedy", "source": GOOD_AGENT})
        r = client.post("/api/sessions", json={
            "scenario_id": "EXP-A-BUDGET", "seed": 42,
            "agents": [{"name": "Bot", "type": "Custom", "plugin": "greedy"}],
        })
        assert r.status_code == 200, r.text
        sid = r.json()["session_id"]
        out = client.post(f"/api/sessions/{sid}/step", json={"steps": 2}).json()
        assert out["kpis"]["quarter"] == 3

    def test_custom_agent_missing_plugin_rejected(self, client):
        r = client.post("/api/sessions", json={
            "scenario_id": "EXP-A-BUDGET", "seed": 42,
            "agents": [{"name": "Bot", "type": "Custom", "plugin": "non-esiste"}],
        })
        assert r.status_code == 400

    def test_unknown_kind(self, client):
        assert client.get("/api/plugins/boh").status_code == 404
