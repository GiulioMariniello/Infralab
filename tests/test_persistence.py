"""Test della persistenza file-based (sessioni, stato di gioco, ricerca sessioni)."""
import json
import os

import pytest

from tests.conftest import build_env_from_scenario, load_scenario


@pytest.fixture
def small_env():
    return build_env_from_scenario(load_scenario("TRAIN-01"))


class TestGameStateRoundtrip:
    def test_save_load_roundtrip(self, isolated_persistence, small_env):
        p = isolated_persistence
        session_id = "TEST_session_1"
        p.save_game_state(small_env, session_id)
        loaded = p.load_game_state(session_id)
        assert loaded is not None
        assert loaded.quarter == small_env.quarter
        assert loaded.budgets == small_env.budgets
        assert set(loaded.bridges.keys()) == set(small_env.bridges.keys())
        for bid, bridge in small_env.bridges.items():
            clone = loaded.bridges[bid]
            assert clone.name == bridge.name
            for comp_name, comp in bridge.components.items():
                assert clone.components[comp_name].current_condition == pytest.approx(
                    comp.current_condition
                )

    def test_save_is_atomic_no_tmp_left(self, isolated_persistence, small_env):
        p = isolated_persistence
        session_id = "TEST_session_atomic"
        p.save_game_state(small_env, session_id)
        session_path = p.get_session_path(session_id)
        assert not any(name.endswith(".tmp") for name in os.listdir(session_path))

    def test_load_missing_returns_none(self, isolated_persistence):
        assert isolated_persistence.load_game_state("does_not_exist") is None

    def test_load_corrupt_returns_none(self, isolated_persistence):
        p = isolated_persistence
        session_id = "TEST_corrupt"
        p.initialize_session_storage(session_id)
        path = os.path.join(p.get_session_path(session_id), p.GAME_STATE_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        assert p.load_game_state(session_id) is None


class TestFindSession:
    def test_finds_existing_session(self, isolated_persistence):
        p = isolated_persistence
        os.makedirs(os.path.join(p.SESSIONS_DIR, "EXP-A_Mario_Rossi_20260101-000000"))
        assert p.find_session("EXP-A", "Mario Rossi") == "EXP-A_Mario_Rossi_20260101-000000"

    def test_returns_none_when_absent(self, isolated_persistence):
        assert isolated_persistence.find_session("EXP-A", "Nessuno") is None

    def test_no_prefix_collision(self, isolated_persistence):
        """'Anna' non deve matchare la sessione di 'Anna Maria'."""
        p = isolated_persistence
        os.makedirs(os.path.join(p.SESSIONS_DIR, "EXP-A_Anna_Maria_20260101-000000"))
        assert p.find_session("EXP-A", "Anna") is None


class TestLogs:
    def test_clear_and_initialize_creates_empty_csvs(self, isolated_persistence):
        p = isolated_persistence
        session_id = "TEST_logs"
        p.clear_and_initialize_log_files(session_id)
        session_path = p.get_session_path(session_id)
        for filename in p.LOG_FILENAMES.values():
            path = os.path.join(session_path, filename)
            assert os.path.exists(path)
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            assert header  # intestazione presente

    def test_log_history_appends_rows(self, isolated_persistence, small_env):
        import pandas as pd

        p = isolated_persistence
        session_id = "TEST_history"
        p.clear_and_initialize_log_files(session_id)
        agent = small_env.params.agent_names[0]
        small_env.step({agent: {"maintenance": {}}})
        p.log_history(small_env, session_id, {agent: []}, {agent: 0.0})
        session_path = p.get_session_path(session_id)
        actions = pd.read_csv(os.path.join(session_path, p.ACTION_LOG_FILENAME))
        kpis = pd.read_csv(os.path.join(session_path, p.KPI_LOG_FILENAME))
        assert len(actions) == len(small_env.bridges)
        assert len(kpis) == 1
        assert list(actions.columns) == p.ACTION_LOG_COLS


class TestParticipants:
    def test_roundtrip(self, isolated_persistence):
        p = isolated_persistence
        p.save_participants(["Alice", "Bob"])
        assert p.load_participants() == ["Alice", "Bob"]

    def test_missing_file_returns_empty(self, isolated_persistence):
        assert isolated_persistence.load_participants() == []


class TestScenarioBootstrap:
    def test_ensure_scenarios_file_copies_seed(self, isolated_persistence):
        p = isolated_persistence
        assert not os.path.exists(p.SCENARIOS_FILE)
        p.ensure_scenarios_file()
        assert os.path.exists(p.SCENARIOS_FILE)
        with open(p.SCENARIOS_FILE, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
        assert any(s["scenario_id"] == "TRAIN-01" for s in scenarios)

    def test_does_not_overwrite_existing(self, isolated_persistence):
        p = isolated_persistence
        os.makedirs(p.DATA_DIR, exist_ok=True)
        with open(p.SCENARIOS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"scenario_id": "CUSTOM"}], f)
        p.ensure_scenarios_file()
        with open(p.SCENARIOS_FILE, "r", encoding="utf-8") as f:
            assert json.load(f)[0]["scenario_id"] == "CUSTOM"
