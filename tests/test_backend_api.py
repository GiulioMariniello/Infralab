"""Test dell'API FastAPI (backend/app.py) con TestClient.

Coprono il contratto usato dal frontend React: metadati (health, asset-types,
scenarios, config, codes), ciclo sessione (create/step/act/detail/seismic/
delete), errori (400/404/422) e campagne asincrone.
"""
import time

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module


@pytest.fixture()
def client():
    # Store puliti per ogni test: le sessioni sono in-memory nel modulo.
    backend_app.SESSIONS.clear()
    backend_app.CAMPAIGNS.clear()
    research_module.ADMIN_TOKENS.clear()
    with TestClient(backend_app.app) as c:
        yield c


@pytest.fixture()
def admin(client):
    """Client autenticato come admin (password di sviluppo)."""
    token = client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]
    return token


def create_session(client, **kwargs):
    payload = {"scenario_id": "TRAIN-01", "agent_type": "Euristico", "seed": 42}
    payload.update(kwargs)
    res = client.post("/api/sessions", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


class TestMetadata:
    def test_health(self, client):
        d = client.get("/api/health").json()
        assert d["status"] == "ok"

    def test_asset_types(self, client):
        names = {a["name"] for a in client.get("/api/asset-types").json()}
        assert {"bridge", "retaining_wall"} <= names

    def test_scenarios(self, client):
        ids = {s["scenario_id"] for s in client.get("/api/scenarios").json()}
        assert {"TRAIN-01", "EXP-A-BUDGET", "WALL-DEMO"} <= ids

    def test_config_schema(self, client):
        d = client.get("/api/config").json()
        keys = {f["key"] for f in d["fields"]}
        assert {"degradation_model", "code_framework", "earthquake_enabled"} <= keys
        assert d["defaults"]["code_framework"] == "custom"

    def test_codes(self, client):
        names = {c["name"] for c in client.get("/api/codes").json()}
        assert {"custom", "EN1990", "fib_MC2010", "AASHTO_LRFD"} <= names


class TestSessionLifecycle:
    def test_create_and_state(self, client):
        d = create_session(client)
        assert d["asset_type"] == "bridge"
        assert d["kpis"]["quarter"] == 1 and d["kpis"]["active"]
        assert len(d["assets"]) == 2
        assert d["assets"][0]["actions"], "azioni di fattibilita' presenti"
        got = client.get(f"/api/sessions/{d['session_id']}")
        assert got.status_code == 200

    def test_step_advances(self, client):
        d = create_session(client)
        out = client.post(f"/api/sessions/{d['session_id']}/step", json={"steps": 3}).json()
        assert out["kpis"]["quarter"] == 4
        assert len(out["history"]) == 4  # snapshot iniziale + 3 step

    def test_act_applies_plan(self, client):
        d = create_session(client, scenario_id="EXP-A-BUDGET")
        budget_before = d["kpis"]["budget"]
        out = client.post(f"/api/sessions/{d['session_id']}/act",
                          json={"plan": {"1": "L1_inspect"}}).json()
        assert out["kpis"]["quarter"] == 2
        assert out["kpis"]["budget"] != budget_before

    def test_wall_scenario_session(self, client):
        d = create_session(client, scenario_id="WALL-DEMO", seed=11)
        assert d["asset_type"] == "retaining_wall"
        assert set(d["assets"][0]["components"]) == {"paramento", "fondazione", "drenaggio"}

    def test_asset_detail(self, client):
        d = create_session(client)
        det = client.get(f"/api/sessions/{d['session_id']}/assets/1").json()
        assert det["beta_annual"] is not None
        assert len(det["projection"]) >= 2
        assert det["components"][0]["limit_states"]

    def test_param_overrides_applied(self, client):
        d = create_session(client, params={"initial_budget": 250000, "code_framework": "AASHTO_LRFD"})
        assert d["kpis"]["budget"] == 250000.0

    def test_delete(self, client):
        d = create_session(client)
        client.delete(f"/api/sessions/{d['session_id']}")
        assert client.get(f"/api/sessions/{d['session_id']}").status_code == 404


class TestMultiAgent:
    AGENTS = [
        {"name": "Alice", "type": "Umano"},
        {"name": "Bot", "type": "Euristico"},
    ]

    def test_create_multi_agent(self, client):
        d = create_session(client, scenario_id="EXP-A-BUDGET", agents=self.AGENTS)
        names = {a["name"] for a in d["agents"]}
        assert names == {"Alice", "Bot"}
        human = next(a for a in d["agents"] if a["name"] == "Alice")
        assert human["is_human"] is True
        # asset distribuiti round-robin tra i manager
        managers = {a["manager"] for a in d["assets"]}
        assert managers == {"Alice", "Bot"}
        # KPI aggregati = somma dei budget individuali
        assert d["kpis"]["budget"] == sum(a["kpis"]["budget"] for a in d["agents"])

    def test_step_human_plan_ai_autonomous(self, client):
        d = create_session(client, scenario_id="EXP-A-BUDGET", agents=self.AGENTS)
        sid = d["session_id"]
        alice_assets = [a for a in d["assets"] if a["manager"] == "Alice"]
        target = str(alice_assets[0]["id"])
        alice_before = next(a for a in d["agents"] if a["name"] == "Alice")["kpis"]["budget"]
        out = client.post(f"/api/sessions/{sid}/step",
                          json={"steps": 1, "plans": {"Alice": {target: "L1_inspect"}}}).json()
        assert out["kpis"]["quarter"] == 2
        alice_after = next(a for a in out["agents"] if a["name"] == "Alice")["kpis"]["budget"]
        assert alice_after != alice_before  # il piano umano e' stato applicato
        # storia con KPI per agente
        assert "agents" in out["history"][-1]
        assert set(out["history"][-1]["agents"]) == {"Alice", "Bot"}

    def test_duplicate_names_400(self, client):
        res = client.post("/api/sessions", json={
            "scenario_id": "TRAIN-01",
            "agents": [{"name": "X", "type": "Euristico"}, {"name": "X", "type": "Umano"}],
        })
        assert res.status_code == 400

    def test_too_many_agents_400(self, client):
        agents = [{"name": f"A{i}", "type": "Euristico"} for i in range(7)]
        res = client.post("/api/sessions", json={"scenario_id": "TRAIN-01", "agents": agents})
        assert res.status_code == 400

    def test_human_only_session_steps_idle(self, client):
        d = create_session(client, agents=[{"name": "Solo", "type": "Umano"}])
        out = client.post(f"/api/sessions/{d['session_id']}/step", json={"steps": 2}).json()
        assert out["kpis"]["quarter"] == 3  # avanza anche senza decisioni


class TestSeismic:
    def test_hazard_grid_and_distance(self, client):
        d = create_session(client, scenario_id="EXP-A-BUDGET")
        sz = client.get(f"/api/sessions/{d['session_id']}/seismic").json()
        assert len(sz["grid"]) == 26
        pgas = [a["site_pga"] for a in sz["assets"]]
        dists = [a["distance_to_hazard_km"] for a in sz["assets"]]
        # piu' lontano dal centro => PGA minore (ordinamento coerente)
        paired = sorted(zip(dists, pgas))
        assert paired[0][1] >= paired[-1][1]

    def test_toggle_changes_reliability(self, client):
        d = create_session(client, scenario_id="EXP-A-BUDGET")
        sid = d["session_id"]
        off = client.post(f"/api/sessions/{sid}/seismic", json={"enabled": False}).json()
        on = client.post(f"/api/sessions/{sid}/seismic", json={"enabled": True}).json()
        assert on["kpis"]["min_reliability_index"] <= off["kpis"]["min_reliability_index"]


class TestErrors:
    def test_unknown_scenario_404(self, client):
        assert client.post("/api/sessions", json={"scenario_id": "NOPE"}).status_code == 404

    def test_unknown_agent_400(self, client):
        res = client.post("/api/sessions", json={"scenario_id": "TRAIN-01", "agent_type": "Alieno"})
        assert res.status_code == 400

    def test_invalid_params_400(self, client):
        res = client.post("/api/sessions", json={"scenario_id": "TRAIN-01",
                                                 "params": {"reliability_consequence_class": "CC9"}})
        assert res.status_code == 400

    def test_unknown_session_404(self, client):
        assert client.get("/api/sessions/xxx").status_code == 404
        assert client.post("/api/sessions/xxx/step", json={"steps": 1}).status_code == 404

    def test_campaign_bad_sweep_400(self, client):
        res = client.post("/api/campaigns", json={"scenario_id": "TRAIN-01",
                                                  "grid_param": "seed", "grid_values": [1, 2]})
        assert res.status_code == 400

    def test_campaign_too_many_runs_400(self, client):
        res = client.post("/api/campaigns", json={
            "scenario_id": "TRAIN-01",
            "agents": ["Euristico"],
            "replicates": 15,
            "grid_param": "env_degradation_std",
            "grid_values": [0.05, 0.1, 0.15, 0.2, 0.25],
        })
        assert res.status_code == 400


class TestResearchAuth:
    def test_wrong_password_401(self, client):
        assert client.post("/api/auth/admin", json={"password": "sbagliata"}).status_code == 401

    def test_admin_endpoints_require_token(self, client):
        assert client.get("/api/admin/participants").status_code == 401
        assert client.get("/api/admin/research-sessions").status_code == 401

    def test_participants_roundtrip(self, client, admin, isolated_persistence):
        h = {"X-Admin-Token": admin}
        client.put("/api/admin/participants", json={"participants": ["Mario", "Anna"]}, headers=h)
        got = client.get("/api/admin/participants", headers=h).json()
        assert got["participants"] == ["Anna", "Mario"]

    def test_player_links_and_join_persisted(self, client, admin, isolated_persistence):
        h = {"X-Admin-Token": admin}
        res = client.post("/api/admin/player-links",
                          json={"scenario_id": "TRAIN-01", "participants": ["Tester Uno"]}, headers=h).json()
        token = res["links"][0]["token"]
        assert res["links"][0]["path"].startswith("/?ptoken=")

        # join: crea sessione persistita con agente umano
        joined = client.post("/api/player/join", json={"token": token}).json()
        assert joined["player_name"] == "Tester Uno"
        assert joined["agents"][0]["is_human"] is True
        sid = joined["session_id"]

        # avanza con un piano manuale → lo stato viene salvato su disco
        target = str(joined["assets"][0]["id"])
        client.post(f"/api/sessions/{sid}/step",
                    json={"steps": 1, "plans": {"Tester Uno": {target: "L1_inspect"}}})
        import infralab.persistence as p
        persist_id = p.find_session("TRAIN-01", "Tester Uno")
        assert persist_id is not None
        env = p.load_game_state(persist_id)
        assert env is not None and env.quarter == 2

        # ri-join con lo stesso token → riprende la stessa partita (web session riusata)
        rejoined = client.post("/api/player/join", json={"token": token}).json()
        assert rejoined["kpis"]["quarter"] == 2

    def test_invalid_player_token_401(self, client):
        assert client.post("/api/player/join", json={"token": "falso"}).status_code == 401

    def test_research_sessions_monitor(self, client, admin, isolated_persistence):
        h = {"X-Admin-Token": admin}
        res = client.post("/api/admin/player-links",
                          json={"scenario_id": "TRAIN-01", "participants": ["Monitorato"]}, headers=h).json()
        client.post("/api/player/join", json={"token": res["links"][0]["token"]})
        sessions = client.get("/api/admin/research-sessions", headers=h).json()
        assert any(s.get("scenario_id") == "TRAIN-01" and s.get("quarter") == 1 for s in sessions)


class TestLLM:
    def test_status_endpoint(self, client):
        d = client.get("/api/llm/status").json()
        assert {"openai", "google", "locale", "profiles"} <= set(d)
        assert d["locale"]["configured"] is True

    def test_llm_agent_requires_key_for_openai(self, client, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        res = client.post("/api/sessions", json={
            "scenario_id": "TRAIN-01",
            "agents": [{"name": "GPT", "type": "LLM", "provider": "OpenAI"}],
        })
        assert res.status_code == 400

    def test_llm_local_agent_creates(self, client):
        # Il provider Locale non richiede chiavi: la sessione si crea
        # (l'eventuale errore di rete emerge solo allo step, come 422).
        res = client.post("/api/sessions", json={
            "scenario_id": "TRAIN-01",
            "agents": [{"name": "Llama", "type": "LLM", "provider": "Locale", "model": "llama3.1"}],
        })
        assert res.status_code == 200
        assert res.json()["agents"][0]["type"] == "LLM"


class TestCampaigns:
    def _wait(self, client, job_id, timeout_s=120):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            d = client.get(f"/api/campaigns/{job_id}").json()
            if d["status"] != "running":
                return d
            time.sleep(0.5)
        raise TimeoutError("campagna non conclusa in tempo")

    def test_sweep_campaign_completes(self, client):
        res = client.post("/api/campaigns", json={
            "scenario_id": "TRAIN-01", "agents": ["Euristico"],
            "horizon_quarters": 4, "replicates": 2,
            "grid_param": "env_degradation_std", "grid_values": [0.08, 0.16],
        })
        assert res.status_code == 200
        d = self._wait(client, res.json()["job_id"])
        assert d["status"] == "done", d.get("error")
        r = d["result"]
        assert r["completed"] == 4 and r["failed"] == 0
        assert {a["grid_value"] for a in r["aggregates"]} == {0.08, 0.16}

    def test_framework_comparison_campaign(self, client):
        res = client.post("/api/campaigns", json={
            "scenario_id": "TRAIN-01", "agents": ["Euristico"],
            "horizon_quarters": 4, "replicates": 1,
            "compare_frameworks": ["EN1990", "AASHTO_LRFD"],
        })
        assert res.status_code == 200
        d = self._wait(client, res.json()["job_id"])
        assert d["status"] == "done", d.get("error")
        values = {a["grid_value"] for a in d["result"]["aggregates"]}
        assert values == {"EN1990", "AASHTO_LRFD"}

    def test_unknown_framework_400(self, client):
        res = client.post("/api/campaigns", json={"scenario_id": "TRAIN-01",
                                                  "compare_frameworks": ["Codice_Hammurabi"]})
        assert res.status_code == 400
