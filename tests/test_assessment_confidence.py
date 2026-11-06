"""Valutatore prudente: fattori di confidenza nella stima del gestore (assessment_confidence_mode, opt-in)."""
import json

import pytest

from infralab.assets.bridges.limit_states import confidence_factor
from infralab.settings.config import SimParameters
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory
from tests.test_floor_oracle import _env

BASE = {"observation_mode": "inspected", "true_reliability_mode": "physical"}


def _beta(env, bid=1):
    b = env.bridges[bid]
    est = env._assessed_bridge(b)
    return float(est.annual_reliability_index), float(b.annual_reliability_index)


def test_default_is_none_and_golden_unchanged():
    assert SimParameters().assessment_confidence_mode == "none"
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


def test_validation():
    with pytest.raises(ValueError):
        SimParameters(assessment_confidence_mode="prudente")
    with pytest.raises(ValueError):
        SimParameters(assessment_confidence_levels=((0.1, 1.2), (0.2, 1.35)))   # soglie non decrescenti
    with pytest.raises(ValueError):
        SimParameters(assessment_confidence_levels=((0.2, 0.9),))               # FC < 1


def test_confidence_factor_steps():
    lv = ((0.20, 1.35), (0.12, 1.20), (0.0, 1.00))
    assert confidence_factor(lv, 0.22) == 1.35
    assert confidence_factor(lv, 0.15) == 1.20
    assert confidence_factor(lv, 0.05) == 1.00


def test_estimate_lower_truth_unchanged():
    stima0, vero0 = _beta(_env(BASE))
    stima1, vero1 = _beta(_env({**BASE, "assessment_confidence_mode": "knowledge_level"}))
    assert vero1 == pytest.approx(vero0, abs=1e-12)     # la Pf fisica non cambia
    assert stima1 < stima0 - 0.1                         # la stima scende


def test_better_knowledge_lowers_the_factor():
    env = _env({**BASE, "assessment_confidence_mode": "knowledge_level"})
    b = env.bridges[1]
    stima_ignota, _ = _beta(env)
    for c in b.components.values():
        c.epistemic_uncertainty = 0.05
    env._assessed_cache.clear()
    stima_nota, _ = _beta(env)
    assert stima_nota > stima_ignota
