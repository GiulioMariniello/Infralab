"""Test delle meccaniche di esperimento: condizione per partecipante, copilota
gated, consenso, orizzonte, estrazione esiti umani, export, campagne LLM/Custom.
"""
import json

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
import infralab.persistence as persistence
from infralab.study.outcomes import aggregate_by_condition, session_run_metrics


@pytest.fixture()
def client(isolated_persistence, monkeypatch):
    backend_app.SESSIONS.clear()
    backend_app.CAMPAIGNS.clear()
    research_module.ADMIN_TOKENS.clear()
    # token dei giocatori nella cartella isolata
    monkeypatch.setattr("infralab.ui.auth.DATA_DIR", persistence.DATA_DIR, raising=False)
    with TestClient(backend_app.app) as c:
        yield c


def _admin(client):
    return {"X-Admin-Token": client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]}


def _link(client, scenario="TRAIN-01", condition="Uomo", participant="p1"):
    h = _admin(client)
    client.put("/api/admin/participants", json={"participants": [participant]}, headers=h)
    r = client.post("/api/admin/player-links",
                    json={"scenario_id": scenario, "participants": [participant], "condition": condition}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["links"][0]["token"]


class TestConditionAndCopilotGate:
    def test_link_carries_condition(self, client):
        tok = _link(client, condition="Uomo+LLM")
        state = client.post("/api/player/join", json={"token": tok}).json()
        assert state["condition"] == "Uomo+LLM" and state["copilot_enabled"] is True

    def test_uomo_condition_no_copilot(self, client):
        tok = _link(client, condition="Uomo")
        state = client.post("/api/player/join", json={"token": tok}).json()
        assert state["copilot_enabled"] is False
        # il copilota è vietato in condizione Uomo
        r = client.post(f"/api/sessions/{state['session_id']}/copilot", json={"message": "aiuto"})
        assert r.status_code == 403

    def test_invalid_condition_rejected(self, client):
        h = _admin(client)
        client.put("/api/admin/participants", json={"participants": ["p1"]}, headers=h)
        r = client.post("/api/admin/player-links",
                        json={"scenario_id": "TRAIN-01", "participants": ["p1"], "condition": "Marziano"}, headers=h)
        assert r.status_code == 400

    def test_conditions_endpoint(self, client):
        h = _admin(client)
        d = client.get("/api/admin/conditions", headers=h).json()
        assert "Uomo+LLM" in d["human_conditions"]


class TestConsentAndHorizon:
    def test_consent_recorded(self, client):
        tok = _link(client)
        state = client.post("/api/player/join", json={"token": tok}).json()
        assert state["consent_given"] is False
        r = client.post(f"/api/sessions/{state['session_id']}/consent", json={"agreed": True})
        assert r.status_code == 200 and r.json()["consent_given"] is True
        # persistito nei metadata
        sess = backend_app.SESSIONS[state["session_id"]]
        assert persistence.load_experiment_metadata(sess.persist_id)["consent_given"] is True

    def test_state_exposes_horizon_and_finished(self, client):
        tok = _link(client)
        state = client.post("/api/player/join", json={"token": tok}).json()
        assert state["horizon_quarters"] > 0 and state["finished"] is False

    def test_act_blocked_past_horizon(self, client):
        tok = _link(client)
        state = client.post("/api/player/join", json={"token": tok}).json()
        sid = state["session_id"]
        sess = backend_app.SESSIONS[sid]
        sess.env.quarter = state["horizon_quarters"] + 1     # forza oltre orizzonte
        r = client.post(f"/api/sessions/{sid}/act", json={"plan": {}})
        assert r.status_code == 409


class TestHumanOutcomes:
    def test_session_run_metrics_fields(self, client):
        tok = _link(client, condition="Uomo+LLM")
        state = client.post("/api/player/join", json={"token": tok}).json()
        sess = backend_app.SESSIONS[state["session_id"]]
        m = session_run_metrics(sess.env, sess.player_name, sess.condition)
        assert m["agent_id"] == "Uomo+LLM" and m["condition"] == "Uomo+LLM"
        for k in ("quarters_survived", "final_min_beta_margin", "final_health_index",
                  "final_emissions_tco2", "terminated"):
            assert k in m

    def test_aggregate_by_condition(self):
        runs = [
            {"condition": "Uomo", "final_min_beta_margin": 0.2, "final_health_index": 0.8,
             "final_emissions_tco2": 3, "quarters_survived": 20},
            {"condition": "Uomo", "final_min_beta_margin": 0.1, "final_health_index": 0.75,
             "final_emissions_tco2": 4, "quarters_survived": 20},
            {"condition": "Uomo+LLM", "final_min_beta_margin": 0.6, "final_health_index": 0.9,
             "final_emissions_tco2": 5, "quarters_survived": 20},
            {"condition": "Uomo+LLM", "final_min_beta_margin": 0.5, "final_health_index": 0.88,
             "final_emissions_tco2": 6, "quarters_survived": 20},
        ]
        out = aggregate_by_condition(runs)
        assert set(out["conditions"]) == {"Uomo", "Uomo+LLM"}
        beta = next(m for m in out["metrics"] if m["key"] == "final_min_beta_margin")
        assert len(beta["comparisons"]) == 1


class TestExperimentAnalysisApi:
    def _play_and_finish(self, client, condition, participant):
        tok = _link(client, condition=condition, participant=participant)
        state = client.post("/api/player/join", json={"token": tok}).json()
        sid = state["session_id"]
        client.post(f"/api/sessions/{sid}/consent", json={"agreed": True})
        client.post(f"/api/sessions/{sid}/act", json={"plan": {}, "rationale": "avanzo"})
        # compila questionario (persistito)
        client.post(f"/api/sessions/{sid}/questionnaire", json={"responses": {
            "tlx_mental": 50, "tlx_physical": 20, "tlx_temporal": 40, "tlx_performance": 30,
            "tlx_effort": 40, "tlx_frustration": 20}})
        return sid

    def test_experiment_analysis_and_export(self, client):
        self._play_and_finish(client, "Uomo", "a1")
        self._play_and_finish(client, "Uomo+LLM", "a2")
        h = _admin(client)
        d = client.get("/api/admin/experiment/TRAIN-01", headers=h).json()
        assert d["n_sessions"] == 2
        conds = {r["condition"] for r in d["rows"]}
        assert conds == {"Uomo", "Uomo+LLM"}
        assert any(r["nasa_tlx_raw"] is not None for r in d["rows"])
        # export CSV
        csv_resp = client.get("/api/admin/experiment/TRAIN-01/export.csv", headers=h)
        assert csv_resp.status_code == 200
        assert "condition" in csv_resp.text and "nasa_tlx_raw" in csv_resp.text

    def test_experiment_requires_admin(self, client):
        assert client.get("/api/admin/experiment/TRAIN-01").status_code == 401


class TestCampaignAgentSpecs:
    def test_string_agents_backcompat(self, client):
        specs = backend_app._normalize_campaign_agents(["Euristico", "MCTS"])
        assert [s["type"] for s in specs] == ["Euristico", "MCTS"]

    def test_custom_agent_in_campaign(self, client):
        from infralab.plugins import registry
        registry.save_plugin("agent", "cheap", "def decide(obs):\n    return {}")
        specs = backend_app._normalize_campaign_agents([{"id": "Mio", "type": "Custom", "plugin": "cheap"}])
        assert specs[0]["params"]["plugin"] == "cheap"

    def test_llm_without_key_rejected(self, client):
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(Exception):
            backend_app._normalize_campaign_agents([{"id": "GPT", "type": "LLM", "provider": "OpenAI"}])

    def test_duplicate_id_rejected(self, client):
        with pytest.raises(Exception):
            backend_app._normalize_campaign_agents(["Euristico", "Euristico"])

    def test_campaign_runs_custom_agent(self, client):
        from infralab.plugins import registry
        registry.save_plugin("agent", "greedy", "def decide(obs):\n    return {}")
        r = client.post("/api/campaigns", json={
            "scenario_id": "TRAIN-01", "horizon_quarters": 4, "replicates": 2, "seed": 1,
            "agents": ["Euristico", {"id": "Greedy", "type": "Custom", "plugin": "greedy"}]})
        assert r.status_code == 200, r.text
        job = r.json()["job_id"]
        import time
        for _ in range(40):
            time.sleep(0.5)
            st = client.get(f"/api/campaigns/{job}").json()
            if st["status"] in ("done", "error"):
                break
        assert st["status"] == "done", st.get("error")
        agents = {a["agent"] for a in st["result"]["aggregates"]}
        assert {"Euristico", "Greedy"} <= agents
