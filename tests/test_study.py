"""Test dell'apparato di studio: questionario (schema+scoring), statistica,
persistenza dialoghi/razionali, e le API di ricerca."""
import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
import infralab.persistence as persistence
from infralab.study.questionnaire import QUESTIONNAIRE_SCHEMA, score_questionnaire
from infralab.study.stats import bootstrap_ci, cliffs_delta, cohens_d, compare_conditions, mann_whitney_p


class TestQuestionnaireScoring:
    def test_schema_has_instruments(self):
        ids = {s["id"] for s in QUESTIONNAIRE_SCHEMA}
        assert {"profilo", "nasa_tlx", "fiducia", "strategia"} <= ids

    def test_nasa_tlx_mean(self):
        resp = {"tlx_mental": 60, "tlx_physical": 20, "tlx_temporal": 40,
                "tlx_performance": 30, "tlx_effort": 50, "tlx_frustration": 10}
        s = score_questionnaire(resp)
        assert s["nasa_tlx_raw"] == pytest.approx(35.0) and s["nasa_tlx_n"] == 6

    def test_trust_reverse_item(self):
        # tutti 7 tranne l'item invertito 'ingannevole' = 1 -> ri-orientato a 7
        resp = {"trust_reliable": 7, "trust_information": 7, "trust_deceptive": 1,
                "trust_rely": 7, "trust_improved": 7}
        s = score_questionnaire(resp)
        assert s["trust_score"] == pytest.approx(7.0) and s["trust_n"] == 5

    def test_partial_scales_none(self):
        s = score_questionnaire({"anni_esperienza": 5})
        assert s["nasa_tlx_raw"] is None and s["trust_score"] is None


class TestStats:
    def test_bootstrap_ci_brackets_mean(self):
        ci = bootstrap_ci([1, 2, 3, 4, 5], seed=1)
        assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] == 5

    def test_bootstrap_deterministic(self):
        a = bootstrap_ci([3, 1, 4, 1, 5, 9, 2, 6], seed=42)
        b = bootstrap_ci([3, 1, 4, 1, 5, 9, 2, 6], seed=42)
        assert a == b

    def test_cohens_d_sign(self):
        d = cohens_d([10, 11, 12], [1, 2, 3])
        assert d is not None and d > 0

    def test_cliffs_delta_full_separation(self):
        assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
        assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0

    def test_mann_whitney_detects_difference(self):
        p = mann_whitney_p([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15])
        assert p is not None and p < 0.05

    def test_compare_conditions_shape(self):
        groups = {"LLM": [4.1, 4.2, 4.0, 4.3], "Uomo": [3.1, 3.0, 3.3, 2.9], "Euristico": [4.5, 4.4, 4.6, 4.5]}
        out = compare_conditions(groups)
        assert set(out["groups"]) == set(groups)
        assert len(out["comparisons"]) == 3   # 3 coppie
        c = next(c for c in out["comparisons"] if {c["a"], c["b"]} == {"LLM", "Uomo"})
        assert c["significant"] in (True, False) and c["effect_magnitude"] is not None


class TestPersistence:
    def test_copilot_log_roundtrip(self, isolated_persistence):
        persistence.save_copilot_log("S1", {"Ing": [{"role": "user", "content": "ciao"}]})
        loaded = persistence.load_copilot_log("S1")
        assert loaded["Ing"][0]["content"] == "ciao"

    def test_rationale_append(self, isolated_persistence):
        persistence.append_decision_rationale("S1", 1, "Ing", {"1": "ispezione"}, "priorità sicurezza")
        persistence.append_decision_rationale("S1", 2, "Ing", {}, None)
        rows = persistence.load_decision_rationales("S1")
        assert len(rows) == 2 and rows[0]["rationale"] == "priorità sicurezza"


@pytest.fixture()
def client(isolated_persistence):
    backend_app.SESSIONS.clear()
    backend_app.CAMPAIGNS.clear()
    research_module.ADMIN_TOKENS.clear()
    with TestClient(backend_app.app) as c:
        yield c


def _player_token(client, scenario="TRAIN-01", condition="Uomo"):
    tok = client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]
    h = {"X-Admin-Token": tok}
    client.put("/api/admin/participants", json={"participants": ["p1"]}, headers=h)
    links = client.post("/api/admin/player-links",
                        json={"scenario_id": scenario, "participants": ["p1"], "condition": condition}, headers=h).json()
    return links["links"][0]["token"]


class TestApi:
    def test_questionnaire_schema_endpoint(self, client):
        d = client.get("/api/questionnaire/schema").json()
        assert any(s["id"] == "nasa_tlx" for s in d["sections"])

    def test_questionnaire_requires_research_session(self, client):
        # sessione effimera (non di ricerca) -> 400
        sid = client.post("/api/sessions", json={"scenario_id": "TRAIN-01", "seed": 1}).json()["session_id"]
        r = client.post(f"/api/sessions/{sid}/questionnaire", json={"responses": {"tlx_mental": 50}})
        assert r.status_code == 400

    def test_questionnaire_submit_on_player_session(self, client):
        token = _player_token(client)
        state = client.post("/api/player/join", json={"token": token}).json()
        sid = state["session_id"]
        resp = {"tlx_mental": 60, "tlx_physical": 20, "tlx_temporal": 40, "tlx_performance": 30,
                "tlx_effort": 50, "tlx_frustration": 10, "trust_reliable": 6, "ruolo": "Ricercatore"}
        r = client.post(f"/api/sessions/{sid}/questionnaire", json={"responses": resp})
        assert r.status_code == 200, r.text
        assert r.json()["scores"]["nasa_tlx_raw"] == pytest.approx(35.0)
        assert client.get(f"/api/sessions/{sid}/questionnaire").json()["submitted"] is True

    def test_copilot_dialogue_persisted(self, client, monkeypatch):
        import json as _json
        from infralab.agents.copilot import Copilot

        real = Copilot.__init__

        def scripted(system, prompt):
            return _json.dumps({"reply": "ciao ingegnere"})

        def patched(self, provider="Locale", model=None, llm_callable=None):
            real(self, provider=provider, model=model, llm_callable=scripted)

        monkeypatch.setattr(backend_app.Copilot, "__init__", patched)
        token = _player_token(client, condition="Uomo+LLM")   # copilota disponibile solo qui
        sid = client.post("/api/player/join", json={"token": token}).json()["session_id"]
        client.post(f"/api/sessions/{sid}/copilot", json={"message": "che faccio?"})
        # persistito su disco nella cartella della sessione di ricerca
        session = backend_app.SESSIONS[sid]
        log = persistence.load_copilot_log(session.persist_id)
        assert log and any("ciao ingegnere" in m["content"] for m in next(iter(log.values())))

    def test_human_rationale_persisted(self, client):
        token = _player_token(client)
        state = client.post("/api/player/join", json={"token": token}).json()
        sid = state["session_id"]
        aid = state["assets"][0]["id"]
        client.post(f"/api/sessions/{sid}/act", json={"plan": {str(aid): "ispezione_visiva"}, "rationale": "beta basso"})
        session = backend_app.SESSIONS[sid]
        rows = persistence.load_decision_rationales(session.persist_id)
        assert any(r.get("rationale") == "beta basso" for r in rows)

    def test_analysis_compare_endpoint(self, client):
        r = client.post("/api/analysis/compare", json={"groups": {
            "LLM": [4.1, 4.2, 4.0, 4.3], "Uomo": [3.0, 3.1, 2.9, 3.2]}})
        assert r.status_code == 200
        body = r.json()
        assert len(body["comparisons"]) == 1
        assert body["comparisons"][0]["a"] in ("LLM", "Uomo")
