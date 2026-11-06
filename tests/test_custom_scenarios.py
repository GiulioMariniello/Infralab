"""Test degli scenari custom (costruttore asset): persistenza + API."""
import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
import infralab.persistence as persistence


@pytest.fixture()
def client(isolated_persistence):
    backend_app.SESSIONS.clear()
    backend_app.CAMPAIGNS.clear()
    research_module.ADMIN_TOKENS.clear()
    with TestClient(backend_app.app) as c:
        yield c


def turbine_scenario(sid="WIND-TEST", n=2):
    return {
        "scenario_id": sid,
        "name": "Parco eolico di prova",
        "asset_type": "wind_turbine",
        "params": {"seed": 42, "initial_budget": 1500000},
        "bridges": [
            {"id": i, "name": f"WTG-{i:02d}", "length": 100, "width": 120, "adt": 2600,
             "incasso_trimestrale": 90000, "valore_asset": 3500000, "eta_iniziale_anni": 4}
            for i in range(1, n + 1)
        ],
    }


class TestPersistence:
    def test_round_trip(self, isolated_persistence):
        warnings = persistence.save_custom_scenario(turbine_scenario())
        assert warnings == []
        loaded = [s for s in persistence.load_scenarios(validate=False) if s["scenario_id"] == "WIND-TEST"]
        assert len(loaded) == 1
        assert loaded[0]["source"] == "custom"
        assert loaded[0]["asset_type"] == "wind_turbine"

    def test_base_id_collision_rejected(self, isolated_persistence):
        scen = turbine_scenario(sid="TRAIN-01")
        with pytest.raises(persistence.ScenarioValidationError, match="gia' usato"):
            persistence.save_custom_scenario(scen)

    def test_bad_id_rejected(self, isolated_persistence):
        scen = turbine_scenario(sid="../evil")
        with pytest.raises(persistence.ScenarioValidationError, match="scenario_id non valido"):
            persistence.save_custom_scenario(scen)

    def test_schema_validation_select_warning(self, isolated_persistence):
        scen = {
            "scenario_id": "BLDG-BAD", "asset_type": "building",
            "bridges": [{"id": 1, "name": "B", "length": 40, "width": 20, "adt": 500,
                         "classe_uso": "VII"}],
        }
        warnings = persistence.save_custom_scenario(scen)
        assert any("classe_uso" in w for w in warnings)

    def test_delete(self, isolated_persistence):
        persistence.save_custom_scenario(turbine_scenario())
        assert persistence.delete_custom_scenario("WIND-TEST") is True
        assert persistence.delete_custom_scenario("WIND-TEST") is False
        assert all(s["scenario_id"] != "WIND-TEST" for s in persistence.load_scenarios(validate=False))


class TestApi:
    def test_schema_endpoint(self, client):
        d = client.get("/api/asset-types/wind_turbine/schema").json()
        assert d["property_schema"]["adt"]["label"] == "Ore equivalenti vento"
        assert d["maintenance_ops"]["sostituzione_gearbox"]["cost"] == 520000.0
        assert d["hazard_sensitivities"]["seismic"] == 0.5
        assert client.get("/api/asset-types/boh/schema").status_code == 404

    def test_crud_and_playable(self, client):
        res = client.post("/api/custom-scenarios", json=turbine_scenario())
        assert res.status_code == 200, res.text
        ids = {s["scenario_id"] for s in client.get("/api/scenarios").json()}
        assert "WIND-TEST" in ids
        # Lo scenario custom e' giocabile come uno base.
        sess = client.post("/api/sessions", json={"scenario_id": "WIND-TEST", "seed": 42})
        assert sess.status_code == 200, sess.text
        state = sess.json()
        assert state["asset_type"] == "wind_turbine"
        assert len(state["assets"]) == 2
        out = client.post(f"/api/sessions/{state['session_id']}/step", json={"steps": 2}).json()
        assert out["kpis"]["quarter"] == 3

    def test_validate_endpoint(self, client):
        bad = turbine_scenario()
        bad["asset_type"] = "astronave"
        assert client.post("/api/custom-scenarios/validate", json=bad).status_code == 400
        ok = client.post("/api/custom-scenarios/validate", json=turbine_scenario())
        assert ok.status_code == 200

    def test_collision_and_delete(self, client):
        assert client.post("/api/custom-scenarios", json=turbine_scenario(sid="TRAIN-01")).status_code == 400
        client.post("/api/custom-scenarios", json=turbine_scenario())
        assert client.delete("/api/custom-scenarios/WIND-TEST").status_code == 200
        assert client.delete("/api/custom-scenarios/WIND-TEST").status_code == 404

    def test_asset_limits(self, client):
        empty = turbine_scenario(n=1)
        empty["bridges"] = []
        assert client.post("/api/custom-scenarios", json=empty).status_code == 400
        dup = turbine_scenario(n=2)
        dup["bridges"][1]["id"] = 1
        assert client.post("/api/custom-scenarios", json=dup).status_code == 400
