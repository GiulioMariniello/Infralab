"""Test del copilota Uomo+LLM: dialogo, profilazione, proposta di piano, API.

Come per l'agente agentico, l'LLM viene recitato da un `llm_callable`
scriptato — nessuna rete. Si verifica che il copilota veda il contesto
tecnico, mantenga la conversazione, e che la proposta arrivi alla UI senza
mai essere applicata da sola (la decisione resta all'umano).
"""
import json

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
from infralab.agents.copilot import Copilot
from infralab.lab.build import build_env_from_scenario
from tests.conftest import load_scenario


@pytest.fixture()
def env():
    return build_env_from_scenario(load_scenario("EXP-A-BUDGET"), agent_name="Ing", seed=42)


class TestCopilotCore:
    def test_first_exchange_profiles_the_engineer(self, env):
        prompts = []

        def fake_llm(system, prompt):
            prompts.append((system, prompt))
            return json.dumps({"reply": "Piacere! Quanto pesa per te il budget rispetto alla sicurezza?"})

        cp = Copilot(llm_callable=fake_llm)
        out = cp.respond(env, "Ing", [], "Ciao, iniziamo.")
        assert "budget" in out["reply"]
        assert out["proposed_plan"] is None
        system, prompt = prompts[0]
        assert "NON decidi da solo" in system
        assert "CONTESTO TECNICO" in prompt and "PROIEZIONE INERZIALE" in prompt
        assert "primo scambio" in prompt

    def test_history_flows_into_prompt(self, env):
        prompts = []

        def fake_llm(system, prompt):
            prompts.append(prompt)
            return json.dumps({"reply": "ok"})

        history = [
            {"role": "user", "content": "Sono avverso al rischio."},
            {"role": "copilot", "content": "Capito: priorita' sicurezza."},
        ]
        Copilot(llm_callable=fake_llm).respond(env, "Ing", history, "Che facciamo sul ponte 1?")
        p = prompts[0]
        assert "[INGEGNERE] Sono avverso al rischio." in p
        assert "[COPILOTA] Capito: priorita' sicurezza." in p
        assert p.rstrip().endswith("Rispondi SOLO JSON.")

    def test_proposed_plan_returned(self, env):
        def fake_llm(system, prompt):
            return json.dumps({
                "reply": "Propongo un'ispezione sul ponte 1: beta in calo, costo minimo.",
                "proposed_plan": {"1": "L1_inspect"},
            })

        out = Copilot(llm_callable=fake_llm).respond(env, "Ing", [], "Proposta?")
        assert out["proposed_plan"] == {"1": "L1_inspect"}

    def test_prose_fallback(self, env):
        def fake_llm(system, prompt):
            return "Il ponte 1 mi preoccupa, parliamone."

        out = Copilot(llm_callable=fake_llm).respond(env, "Ing", [], "?")
        assert "mi preoccupa" in out["reply"]
        assert out["proposed_plan"] is None

    def test_llm_error_degrades_gracefully(self, env):
        def fake_llm(system, prompt):
            raise RuntimeError("server spento")

        out = Copilot(llm_callable=fake_llm).respond(env, "Ing", [], "?")
        assert "non disponibile" in out["reply"]
        assert out["proposed_plan"] is None

    def test_respond_does_not_mutate_env(self, env):
        def fake_llm(system, prompt):
            return json.dumps({"reply": "ok", "proposed_plan": {"1": "replace_appoggi"}})

        q, b = env.quarter, env.budgets["Ing"]
        Copilot(llm_callable=fake_llm).respond(env, "Ing", [], "vai")
        assert env.quarter == q and env.budgets["Ing"] == b


@pytest.fixture()
def client(monkeypatch):
    backend_app.SESSIONS.clear()
    backend_app.CAMPAIGNS.clear()
    research_module.ADMIN_TOKENS.clear()

    def scripted(system, prompt):
        if "CONVERSAZIONE FINORA" not in prompt:
            return json.dumps({"reply": "Presentazione: che profilo hai?"})
        return json.dumps({"reply": "Propongo ispezione.", "proposed_plan": {"1": "L1_inspect"}})

    real_init = Copilot.__init__

    def patched_init(self, provider="Locale", model=None, llm_callable=None):
        real_init(self, provider=provider, model=model, llm_callable=scripted)

    monkeypatch.setattr(backend_app.Copilot, "__init__", patched_init)
    with TestClient(backend_app.app) as c:
        yield c


def make_session(client, agents):
    res = client.post("/api/sessions", json={"scenario_id": "TRAIN-01", "seed": 42, "agents": agents})
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


class TestCopilotApi:
    def test_dialog_targets_first_human(self, client):
        sid = make_session(client, [
            {"name": "AI", "type": "Euristico"},
            {"name": "Giulia", "type": "Umano"},
        ])
        r1 = client.post(f"/api/sessions/{sid}/copilot", json={"message": "Ciao"}).json()
        assert r1["agent_name"] == "Giulia"
        assert "profilo" in r1["reply"]
        assert r1["proposed_plan"] is None

        r2 = client.post(f"/api/sessions/{sid}/copilot", json={"message": "Avverso al rischio: proposta?"}).json()
        assert r2["proposed_plan"] == {"1": "L1_inspect"}
        assert len(r2["history"]) == 4  # 2 scambi utente+copilota

        got = client.get(f"/api/sessions/{sid}/copilot").json()
        assert got["agent_name"] == "Giulia" and len(got["history"]) == 4

    def test_plan_not_applied_until_human_acts(self, client):
        sid = make_session(client, [{"name": "Giulia", "type": "Umano"}])
        client.post(f"/api/sessions/{sid}/copilot", json={"message": "x"})
        client.post(f"/api/sessions/{sid}/copilot", json={"message": "proposta?"})
        state = client.get(f"/api/sessions/{sid}").json()
        assert state["kpis"]["quarter"] == 1  # il copilota non ha mosso nulla

        # L'umano decide: applica la proposta via /act.
        out = client.post(f"/api/sessions/{sid}/act", json={"plan": {"1": "L1_inspect"}}).json()
        assert out["kpis"]["quarter"] == 2

    def test_validation_errors(self, client):
        sid = make_session(client, [{"name": "Giulia", "type": "Umano"}])
        assert client.post(f"/api/sessions/{sid}/copilot", json={"message": "  "}).status_code == 400
        assert client.post(f"/api/sessions/{sid}/copilot", json={"message": "x", "provider": "Boh"}).status_code == 400
        assert client.post(f"/api/sessions/{sid}/copilot", json={"message": "x", "agent_name": "Nessuno"}).status_code == 404
        assert client.post("/api/sessions/void/copilot", json={"message": "x"}).status_code == 404
