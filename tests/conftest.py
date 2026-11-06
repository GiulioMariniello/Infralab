"""Fixture condivise per la suite di test.

I test di questa suite "pinnano" il comportamento corrente del motore prima
del refactoring InfraLab: ogni modifica numerica intenzionale deve aggiornare
gli snapshot e essere registrata in docs/CHANGELOG_MODEL.md.
"""
import json
import os
import sys
from dataclasses import asdict

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infralab.settings.config import CdaConfig, DEFAULT_COMPONENTS, MaintOpsConfig, SimParameters  # noqa: E402
from infralab.core.environment import BridgeEnv  # noqa: E402


SCENARIOS_PATH = os.path.join(REPO_ROOT, "scenarios.json")


def load_scenario(scenario_id: str) -> dict:
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    scenario = next(s for s in scenarios if s["scenario_id"] == scenario_id)
    return scenario


def build_env_from_scenario(scenario: dict, player_name: str = "TestPlayer") -> BridgeEnv:
    """Replica la costruzione dell'ambiente fatta in app.py (player_init)."""
    from infralab.calibration_profiles import apply_calibration_profile

    base_params = asdict(SimParameters())
    base_params.update(scenario.get("params", {}))
    if scenario.get("max_quarters"):
        base_params["max_quarters"] = scenario["max_quarters"]
    params = SimParameters(**base_params)
    params.agent_names = [player_name]
    params.num_agents = 1
    agent_config = {player_name: {"type": "Umano"}}

    maint_ops = asdict(MaintOpsConfig())
    params, maint_ops = apply_calibration_profile(params, maint_ops, params.calibration_profile)
    allowed = scenario.get("allowed_actions")
    if allowed:
        maint_ops = {k: v for k, v in maint_ops.items() if k in allowed}

    env = BridgeEnv(params, maint_ops, agent_config, CdaConfig(), DEFAULT_COMPONENTS)
    env.initialize_bridges(scenario["bridges"])
    return env


@pytest.fixture
def train01_env() -> BridgeEnv:
    return build_env_from_scenario(load_scenario("TRAIN-01"))


@pytest.fixture
def isolated_persistence(tmp_path, monkeypatch):
    """Reindirizza tutta la persistenza file-based in una cartella temporanea."""
    import infralab.persistence as persistence

    data_dir = str(tmp_path / "data")
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "SESSIONS_DIR", os.path.join(data_dir, "sessions"))
    monkeypatch.setattr(persistence, "FREE_PLAY_DIR", os.path.join(data_dir, "free_play"))
    monkeypatch.setattr(persistence, "SCENARIOS_FILE", os.path.join(data_dir, "scenarios.json"))
    monkeypatch.setattr(persistence, "PARTICIPANTS_FILE", os.path.join(data_dir, "participants.json"))
    monkeypatch.setattr(persistence, "CUSTOM_SCENARIOS_DIR", os.path.join(data_dir, "custom_scenarios"))
    os.makedirs(os.path.join(data_dir, "sessions"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "free_play"), exist_ok=True)
    # I plugin utente leggono persistence.DATA_DIR (custom_plugins/): la cache
    # del registry va svuotata tra i test per non trascinare callable vecchie.
    try:
        from infralab.plugins import registry as _plugreg
        _plugreg._CACHE.clear()
    except Exception:
        pass
    return persistence
