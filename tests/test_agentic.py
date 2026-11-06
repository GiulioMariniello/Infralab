"""Test dell'agente LLM agentico: sandbox, tool di simulazione, loop completo.

L'harness viene verificato SENZA rete con un `llm_callable` scriptato che
recita il ruolo del modello: prima chiede di eseguire codice (esplora e fa
rollout di piani candidati sul clone), poi decide dal risultato.
"""
import json

import pytest

from infralab.agents.agentic import AgenticLLMAgent, CodeSandbox, make_simulation_tools
from infralab.lab.build import build_env_from_scenario
from tests.conftest import load_scenario


@pytest.fixture()
def env():
    return build_env_from_scenario(load_scenario("EXP-A-BUDGET"), agent_name="Agente", seed=42)


class TestSandbox:
    def test_print_captured(self):
        out = CodeSandbox({}).run("print('ciao', 1+1)")
        assert "ciao 2" in out

    def test_allowed_import(self):
        out = CodeSandbox({}).run("import math\nprint(math.sqrt(16))")
        assert "4.0" in out

    def test_forbidden_import_blocked(self):
        out = CodeSandbox({}).run("import os\nprint(os.getcwd())")
        assert "Import non consentito" in out

    def test_open_blocked(self):
        out = CodeSandbox({}).run("open('x.txt','w')")
        assert "ERRORE" in out and "open" in out

    def test_error_reported_to_model(self):
        out = CodeSandbox({}).run("1/0")
        assert "ZeroDivisionError" in out

    def test_output_truncated(self):
        out = CodeSandbox({}).run("print('x' * 100000)")
        assert "troncato" in out


class TestSimulationTools:
    def test_state_tool(self, env):
        tools = make_simulation_tools(env, "Agente")
        s = tools["state"]()
        assert s["quarter"] == 1 and s["budget"] > 0
        assert len(s["assets"]) == 5
        assert all("feasible_actions" in a for a in s["assets"])

    def test_simulate_does_not_mutate_real_env(self, env):
        tools = make_simulation_tools(env, "Agente")
        before_quarter = env.quarter
        before_budget = env.budgets["Agente"]
        result = tools["simulate"]({"1": "L1_inspect"}, quarters=4)
        assert result["quarters_simulated"] == 4
        assert result["min_beta"] is not None
        assert env.quarter == before_quarter          # la partita reale non si muove
        assert env.budgets["Agente"] == before_budget

    def test_simulate_plan_effect_visible(self, env):
        tools = make_simulation_tools(env, "Agente")
        idle = tools["simulate"]({}, quarters=8)
        # replace_appoggi: costo fisso 100k, sempre affrontabile nello scenario
        with_action = tools["simulate"]({"1": "replace_appoggi"}, quarters=8)
        assert with_action["final_budget"] < idle["final_budget"]     # ha speso
        assert with_action["emissions_tco2"] > idle["emissions_tco2"]  # ha emesso

    def test_tools_usable_from_sandbox(self, env):
        tools = make_simulation_tools(env, "Agente")
        sandbox = CodeSandbox(tools)
        out = sandbox.run(
            "s = state()\n"
            "print('assets', len(s['assets']))\n"
            "r = simulate({'1': 'L1_inspect'}, quarters=2)\n"
            "print('beta', r['min_beta'], 'attivo', r['still_active'])"
        )
        assert "assets 5" in out and "beta" in out and "ERRORE" not in out


class TestAgenticLoop:
    def make_scripted_agent(self, plan):
        """LLM finto: round 1 esplora+simula, round 2 decide dal risultato."""
        responses = [
            json.dumps({
                "action": "run_code",
                "code": (
                    "cands = [{}, {'1': 'L1_inspect'}]\n"
                    "best, best_beta = None, -1e9\n"
                    "for c in cands:\n"
                    "    r = simulate(c, quarters=4)\n"
                    "    print(c, '->', r['min_beta'])\n"
                    "    if r['min_beta'] is not None and r['min_beta'] > best_beta:\n"
                    "        best, best_beta = c, r['min_beta']\n"
                    "print('BEST', best)"
                ),
            }),
            json.dumps({"action": "final_plan", "plan": plan,
                        "explanation": "scelto dal rollout"}),
        ]
        calls = {"n": 0, "prompts": []}

        def fake_llm(system, prompt):
            calls["prompts"].append(prompt)
            resp = responses[min(calls["n"], len(responses) - 1)]
            calls["n"] += 1
            return resp

        agent = AgenticLLMAgent("Agente", "Bilanciato", "Locale", llm_callable=fake_llm)
        return agent, calls

    def test_full_loop_returns_plan(self, env):
        plan = {"maintenance": {"1": "L1_inspect"}}
        agent, calls = self.make_scripted_agent(plan)
        decided, rationale = agent.decide(env)
        assert decided == plan
        assert calls["n"] == 2
        # il secondo prompt contiene l'output della sandbox (il modello 'vede' i rollout)
        assert "BEST" in calls["prompts"][1]
        assert "->" in calls["prompts"][1]

    def test_invalid_json_gets_retry(self, env):
        responses = ["non sono json",
                     json.dumps({"action": "final_plan", "plan": {"maintenance": {}}, "explanation": "ok"})]
        calls = {"n": 0}

        def fake_llm(system, prompt):
            r = responses[min(calls["n"], 1)]
            calls["n"] += 1
            return r

        agent = AgenticLLMAgent("Agente", "Bilanciato", "Locale", llm_callable=fake_llm)
        decided, _ = agent.decide(env)
        assert decided == {"maintenance": {}}
        assert calls["n"] == 2

    def test_budget_exhaustion_forces_plan(self, env):
        """Modello che vuole solo eseguire codice: al limite viene forzato."""
        def fake_llm(system, prompt):
            if "esaurito" in prompt:
                return json.dumps({"action": "final_plan", "plan": {"maintenance": {}}, "explanation": "forzato"})
            return json.dumps({"action": "run_code", "code": "print('ancora')"})

        agent = AgenticLLMAgent("Agente", "Bilanciato", "Locale", max_rounds=2, llm_callable=fake_llm)
        decided, _ = agent.decide(env)
        assert decided == {"maintenance": {}}

    def test_factory_creates_agentic(self):
        from infralab.agents.legacy import agent_factory
        agent = agent_factory("X", {"type": "LLM Agentico", "provider": "Locale", "max_rounds": 2})
        assert isinstance(agent, AgenticLLMAgent)
        assert agent.max_rounds == 2

    def test_agentic_plan_executes_in_real_env(self, env):
        """Il piano deciso dal loop agentico viene applicato alla partita vera."""
        plan = {"maintenance": {"1": "L1_inspect"}}
        agent, _ = self.make_scripted_agent(plan)
        decided, _ = agent.decide(env)
        budget_before = env.budgets["Agente"]
        env.step({"Agente": decided})
        assert env.quarter == 2
        assert env.budgets["Agente"] != budget_before
