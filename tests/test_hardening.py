"""Test hardening esperimenti umani: token, password admin, lock, migrazioni."""
import json
import multiprocessing
import os

import pytest

from tests.conftest import build_env_from_scenario, load_scenario


@pytest.fixture
def auth_isolated(isolated_persistence, monkeypatch):
    import infralab.ui.auth as auth
    return auth


class TestPlayerTokens:
    def test_issue_and_resolve(self, auth_isolated):
        auth = auth_isolated
        token = auth.issue_player_token("EXP-A-BUDGET", "Mario Rossi")
        assert auth.resolve_player_token(token) == {
            "scenario_id": "EXP-A-BUDGET", "player_name": "Mario Rossi", "condition": "Uomo"
        }

    def test_token_reused_for_same_player(self, auth_isolated):
        auth = auth_isolated
        t1 = auth.issue_player_token("EXP-A", "Anna")
        t2 = auth.issue_player_token("EXP-A", "Anna")
        assert t1 == t2

    def test_distinct_players_distinct_tokens(self, auth_isolated):
        auth = auth_isolated
        assert auth.issue_player_token("EXP-A", "Anna") != auth.issue_player_token("EXP-A", "Bruno")

    def test_invalid_token_none(self, auth_isolated):
        assert auth_isolated.resolve_player_token("inesistente") is None
        assert auth_isolated.resolve_player_token("") is None

    def test_revoke(self, auth_isolated):
        auth = auth_isolated
        token = auth.issue_player_token("EXP-A", "Carla")
        assert auth.revoke_player_token(token) is True
        assert auth.resolve_player_token(token) is None
        assert auth.revoke_player_token(token) is False


class TestAdminPassword:
    def test_dev_fallback(self, auth_isolated):
        # senza secrets configurati vale la password di sviluppo
        assert auth_isolated.verify_admin_password("admin") is True
        assert auth_isolated.verify_admin_password("sbagliata") is False


class TestMigrations:
    def test_v0_state_migrated(self, isolated_persistence):
        p = isolated_persistence
        env = build_env_from_scenario(load_scenario("TRAIN-01"))
        data = env.to_dict()
        # simula un salvataggio pre-InfraLab
        data.pop("schema_version", None)
        data.pop("asset_type", None)
        migrated = p.migrate_game_state(json.loads(json.dumps(data, default=float)))
        assert migrated["schema_version"] == p.GAME_STATE_SCHEMA_VERSION
        assert migrated["asset_type"] == "bridge"

    def test_migration_idempotent(self, isolated_persistence):
        p = isolated_persistence
        data = {"schema_version": 1, "asset_type": "retaining_wall"}
        assert p.migrate_game_state(dict(data)) == data

    def test_legacy_session_roundtrip(self, isolated_persistence):
        """Salvataggio v0 su disco -> load_game_state lo migra e carica."""
        p = isolated_persistence
        env = build_env_from_scenario(load_scenario("TRAIN-01"))
        session_id = "LEGACY_test"
        p.save_game_state(env, session_id)
        path = os.path.join(p.get_session_path(session_id), p.GAME_STATE_FILENAME)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("schema_version", None)
        data.pop("asset_type", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        loaded = p.load_game_state(session_id)
        assert loaded is not None
        assert loaded.asset_type == "bridge"


class TestSessionLock:
    def test_lock_file_created_and_released(self, isolated_persistence):
        p = isolated_persistence
        with p.session_lock("LOCK_test"):
            lock_path = os.path.join(p.get_session_path("LOCK_test"), ".session.lock")
            assert os.path.exists(lock_path)
        # rilasciato: si puo' riacquisire subito
        with p.session_lock("LOCK_test"):
            pass

    def test_concurrent_player_actions_not_lost(self, isolated_persistence):
        """Due scritture concorrenti di player_actions non devono perdersi."""
        p = isolated_persistence
        session_id = "CONC_test"
        p.initialize_session_storage(session_id)
        p.save_player_action(session_id, "AgenteA", {"maintenance": {"1": "L1_inspect"}})
        p.save_player_action(session_id, "AgenteB", {"maintenance": {"2": "monitoring"}})
        assert p.load_player_action(session_id, "AgenteA") == {"maintenance": {"1": "L1_inspect"}}
        assert p.load_player_action(session_id, "AgenteB") == {"maintenance": {"2": "monitoring"}}
