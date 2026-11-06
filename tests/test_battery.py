"""Test del disegno within-subjects: batteria controbilanciata, cadenza,
statistica appaiata (Wilcoxon), flusso giocatore a batteria e analisi appaiata.
"""
import json

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
import infralab.persistence as persistence
import infralab.ui.auth as auth
from infralab.study.battery import build_battery
from infralab.study.stats import paired_delta, wilcoxon_signed_rank


class TestBatteryCounterbalancing:
    def test_2x2_structure(self):
        b = build_battery(0, include_practice=False)
        assert len(b) == 4
        assert sorted(g["length_class"] for g in b) == ["long", "long", "short", "short"]
        assert sorted(g["condition"] for g in b) == ["Uomo", "Uomo", "Uomo+LLM", "Uomo+LLM"]
        # ogni classe usa scenari distinti
        shorts = [g["scenario_id"] for g in b if g["length_class"] == "short"]
        assert len(set(shorts)) == 2

    def test_practice_prepended(self):
        b = build_battery(0)   # default include_practice=True
        assert len(b) == 5
        assert b[0]["length_class"] == "practice" and b[0].get("practice") is True
        assert b[0]["index"] == 0 and b[1]["index"] == 1
        # le 4 partite vere restano la 2×2
        real = b[1:]
        assert sorted(g["length_class"] for g in real) == ["long", "long", "short", "short"]

    def test_eight_distinct_sequences(self):
        seqs = set()
        for i in range(8):
            b = build_battery(i, include_practice=False)
            seqs.add(tuple((g["scenario_id"], g["condition"]) for g in b))
        assert len(seqs) == 8   # 8 partecipanti -> 8 sequenze diverse

    def test_scenario_condition_swapped_across_participants(self):
        # In alcuni partecipanti short-A ha il copilota, in altri no (controllo
        # dell'effetto scenario).
        maps = set()
        for i in range(8):
            b = build_battery(i, include_practice=False)
            for g in b:
                if g["scenario_id"] == "PAPERA-SHORT-A":
                    maps.add(g["condition"])
        assert maps == {"Uomo", "Uomo+LLM"}


class TestPairedStats:
    def test_wilcoxon_detects_consistent_improvement(self):
        # con-LLM sempre migliore di senza -> significativo
        pairs = [(5.0, 3.0), (4.5, 3.2), (6.0, 4.0), (5.5, 3.8), (4.8, 3.5), (5.2, 4.1)]
        r = wilcoxon_signed_rank(pairs)
        assert r["n"] == 6 and r["mean_diff"] > 0 and r["p_value"] < 0.05
        assert r["effect_r"] is not None

    def test_wilcoxon_too_few(self):
        assert wilcoxon_signed_rank([(1.0, 1.0), (2.0, 2.0)]) is None

    def test_paired_delta(self):
        subjects = {
            "p1": {"Uomo+LLM": [5.0], "Uomo": [3.0]},
            "p2": {"Uomo+LLM": [4.5], "Uomo": [3.2]},
            "p3": {"Uomo+LLM": [6.0], "Uomo": [4.0]},
            "p4": {"Uomo+LLM": [5.5], "Uomo": [3.8]},
        }
        d = paired_delta(subjects, "Uomo+LLM", "Uomo")
        assert d["n_pairs"] == 4 and d["mean_diff"] > 0


@pytest.fixture()
def client(isolated_persistence, monkeypatch):
    backend_app.SESSIONS.clear()
    research_module.ADMIN_TOKENS.clear()
    # copilota scriptato (niente rete)
    from infralab.agents.copilot import Copilot
    real = Copilot.__init__

    def patched(self, provider="Locale", model=None, llm_callable=None):
        real(self, provider=provider, model=model, llm_callable=lambda s, p: json.dumps({"reply": "ok"}))

    monkeypatch.setattr(backend_app.Copilot, "__init__", patched)
    with TestClient(backend_app.app) as c:
        yield c


class TestCadence:
    def test_human_cadence_advances_multiple_quarters(self, client):
        # PAPERA-SHORT-A ha human_cadence_quarters=4: un act avanza 4 trimestri.
        tok = auth.issue_player_token("PAPERA-SHORT-A", "p1", "Uomo")
        st = client.post("/api/player/join", json={"token": tok}).json()
        assert st["cadence_quarters"] == 4 and st["kpis"]["quarter"] == 1
        out = client.post(f"/api/sessions/{st['session_id']}/act", json={"plan": {}}).json()
        assert out["kpis"]["quarter"] == 5   # 1 + 4


class TestBatteryFlow:
    def _small_battery(self):
        # 4 partite su TRAIN-01 con orizzonti piccoli per velocità (il flusso è
        # indipendente dallo scenario; il controbilanciamento è testato a parte).
        return [
            {"index": 0, "scenario_id": "TRAIN-01", "condition": "Uomo", "horizon_quarters": 2, "length_class": "short"},
            {"index": 1, "scenario_id": "TRAIN-01", "condition": "Uomo+LLM", "horizon_quarters": 2, "length_class": "short"},
            {"index": 2, "scenario_id": "TRAIN-01", "condition": "Uomo", "horizon_quarters": 2, "length_class": "long"},
            {"index": 3, "scenario_id": "TRAIN-01", "condition": "Uomo+LLM", "horizon_quarters": 2, "length_class": "long"},
        ]

    def _play_to_end(self, client, sid):
        for _ in range(6):
            st = client.get(f"/api/sessions/{sid}").json()
            if st["finished"]:
                break
            client.post(f"/api/sessions/{sid}/act", json={"plan": {}})

    def test_full_battery_progression(self, client):
        tok = auth.issue_battery_token("part1", self._small_battery())
        st = client.post("/api/player/join", json={"token": tok}).json()
        assert st["battery"]["total"] == 4 and st["battery"]["current_index"] == 0
        assert st["condition"] == "Uomo"

        # gioca le 4 partite in sequenza
        seen_conditions = []
        for expected in range(4):
            st = client.post("/api/player/join", json={"token": tok}).json()
            assert st["battery"]["current_index"] == expected
            seen_conditions.append(st["condition"])
            self._play_to_end(client, st["session_id"])
            adv = client.post("/api/battery/advance", json={"token": tok}).json()
            # dopo l'ultima, batteria completa
        final = client.post("/api/player/join", json={"token": tok}).json()
        assert final["battery"]["complete"] is True
        assert seen_conditions == ["Uomo", "Uomo+LLM", "Uomo", "Uomo+LLM"]

    def test_copilot_gated_per_game(self, client):
        tok = auth.issue_battery_token("part2", self._small_battery())
        st = client.post("/api/player/join", json={"token": tok}).json()  # game 0 = Uomo
        r = client.post(f"/api/sessions/{st['session_id']}/copilot", json={"message": "ciao"})
        assert r.status_code == 403                      # copilota vietato in Uomo
        self._play_to_end(client, st["session_id"])
        client.post("/api/battery/advance", json={"token": tok})
        st2 = client.post("/api/player/join", json={"token": tok}).json()  # game 1 = Uomo+LLM
        r2 = client.post(f"/api/sessions/{st2['session_id']}/copilot", json={"message": "ciao"})
        assert r2.status_code == 200                     # copilota consentito

    def test_advance_requires_finished(self, client):
        tok = auth.issue_battery_token("part3", self._small_battery())
        client.post("/api/player/join", json={"token": tok})
        r = client.post("/api/battery/advance", json={"token": tok})
        assert r.status_code == 409   # partita non ancora conclusa

    def test_battery_analysis(self, client):
        # gioca due batterie complete e verifica l'analisi appaiata
        for part in ("A", "B"):
            tok = auth.issue_battery_token(f"soggetto{part}", self._small_battery())
            for _ in range(4):
                st = client.post("/api/player/join", json={"token": tok}).json()
                if st["battery"]["complete"]:
                    break
                self._play_to_end(client, st["session_id"])
                client.post("/api/battery/advance", json={"token": tok})
        h = {"X-Admin-Token": client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]}
        d = client.get("/api/admin/experiment-battery", headers=h).json()
        assert d["n_participants"] == 2 and d["n_games"] == 8
        assert any(m["key"] == "final_min_beta_margin" for m in d["metrics"])
        csv = client.get("/api/admin/experiment-battery/export.csv", headers=h)
        assert csv.status_code == 200 and "length_class" in csv.text


class TestBatteryLinksApi:
    def test_generate_battery_links(self, client):
        h = {"X-Admin-Token": client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]}
        client.put("/api/admin/participants", json={"participants": ["m1", "m2"]}, headers=h)
        r = client.post("/api/admin/battery-links", json={"participants": ["m1", "m2"]}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_participants"] == 2
        assert all(l["games"] == 5 for l in body["links"])   # 1 pratica + 4 reali
        # partecipanti diversi -> sequenze controbilanciate diverse
        assert body["links"][0]["sequence"] != body["links"][1]["sequence"]

    def test_practice_excluded_from_analysis(self, client):
        # una batteria con pratica: le partite di pratica non entrano nell'analisi
        b = [
            {"index": 0, "scenario_id": "TRAIN-01", "condition": "Uomo+LLM", "horizon_quarters": 2, "length_class": "practice", "practice": True},
            {"index": 1, "scenario_id": "TRAIN-01", "condition": "Uomo", "horizon_quarters": 2, "length_class": "short"},
            {"index": 2, "scenario_id": "TRAIN-01", "condition": "Uomo+LLM", "horizon_quarters": 2, "length_class": "short"},
        ]
        tok = auth.issue_battery_token("pex", b)
        for _ in range(4):
            st = client.post("/api/player/join", json={"token": tok}).json()
            if st["battery"]["complete"]:
                break
            for _ in range(6):
                s = client.get(f"/api/sessions/{st['session_id']}").json()
                if s["finished"]:
                    break
                client.post(f"/api/sessions/{st['session_id']}/act", json={"plan": {}})
            client.post("/api/battery/advance", json={"token": tok})
        h = {"X-Admin-Token": client.post("/api/auth/admin", json={"password": "admin"}).json()["token"]}
        d = client.get("/api/admin/experiment-battery", headers=h).json()
        # la pratica non compare fra le classi di lunghezza analizzate
        assert "practice" not in d["length_classes"]
