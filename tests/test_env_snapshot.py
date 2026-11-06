"""Test di regressione end-to-end: run deterministico dello scenario TRAIN-01.

Due livelli di protezione:
1. Determinismo run-to-run: due costruzioni identiche con lo stesso seed devono
   produrre traiettorie identiche.
2. Snapshot dorato su file: cambi numerici intenzionali (fix del modello) vanno
   rigenerati con UPDATE_SNAPSHOTS=1 e documentati in docs/CHANGELOG_MODEL.md.
"""
import json
import os

import pytest

from tests.conftest import build_env_from_scenario, load_scenario

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "train01_snapshot.json")
N_QUARTERS = 6


def run_train01_trajectory() -> dict:
    scenario = load_scenario("TRAIN-01")
    env = build_env_from_scenario(scenario)
    agent = env.params.agent_names[0]
    trajectory = {"quarters": []}
    for _ in range(N_QUARTERS):
        plans = {agent: {"maintenance": {}}}
        env.step(plans)
        record = {
            "quarter": env.quarter,
            "budget": env.budgets[agent],
            "bridges": {
                str(bid): {
                    "min_condition": bridge.min_condition,
                    "reliability_index": bridge.reliability_index,
                    "annual_failure_probability": bridge.annual_failure_probability,
                }
                for bid, bridge in sorted(env.bridges.items())
            },
        }
        trajectory["quarters"].append(record)
    return trajectory


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=float)


@pytest.fixture(scope="module")
def trajectory():
    return run_train01_trajectory()


def test_run_to_run_determinism(trajectory):
    second = run_train01_trajectory()
    assert canonical(trajectory) == canonical(second)


def test_conditions_never_increase_without_maintenance(trajectory):
    for bid in trajectory["quarters"][0]["bridges"]:
        conditions = [q["bridges"][bid]["min_condition"] for q in trajectory["quarters"]]
        assert all(a >= b - 1e-12 for a, b in zip(conditions, conditions[1:]))


def test_seed_semantics():
    """seed intero (0 compreso) => deterministico; seed None => entropia."""
    scenario = load_scenario("TRAIN-01")

    def build_with_seed(seed):
        s = json.loads(json.dumps(scenario))
        s.setdefault("params", {})["seed"] = seed
        return build_env_from_scenario(s)

    env_a = build_with_seed(0)
    env_b = build_with_seed(0)
    assert canonical(env_a.to_dict()) == canonical(env_b.to_dict())

    env_c = build_with_seed(None)
    env_d = build_with_seed(None)
    assert canonical(env_c.to_dict()) != canonical(env_d.to_dict())


def test_no_global_random_contamination():
    """Costruire un env non deve toccare lo stato RNG globale di random/np.random."""
    import random as _random

    import numpy as _np

    _random.seed(99)
    state_before = _random.getstate()
    _np.random.seed(99)
    np_before = _np.random.get_state()[1].sum()
    build_env_from_scenario(load_scenario("TRAIN-01"))
    assert _random.getstate() == state_before
    assert _np.random.get_state()[1].sum() == np_before


def test_matches_golden_snapshot(trajectory):
    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not os.path.exists(SNAPSHOT_PATH):
        os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
        with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
            f.write(canonical(trajectory))
        pytest.skip("Snapshot rigenerato; rieseguire i test per confermarlo.")
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = f.read()
    assert canonical(trajectory) == golden, (
        "La traiettoria TRAIN-01 e' cambiata rispetto allo snapshot dorato. "
        "Se il cambiamento e' intenzionale: UPDATE_SNAPSHOTS=1 pytest e aggiorna docs/CHANGELOG_MODEL.md."
    )
