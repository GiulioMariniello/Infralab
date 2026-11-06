"""Scomposizione dello scarto stima - vero (opzioni opt-in, default invariati)."""
import json

import pytest

from infralab.settings.config import SimParameters
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory
from tests.test_floor_oracle import _env

BASE = {"observation_mode": "inspected", "true_reliability_mode": "physical"}
TUTTO_ALLINEATO = {**BASE, "assessment_parameter_mode": "sampled", "assessment_seed_mode": "shared",
                   "assessment_epistemic_mode": "reference"}


def _osserva_esatto(env, bid=1):
    """Osservazione perfetta: intervallo degenere sulla condizione vera di ogni componente."""
    b = env.bridges[bid]
    for c in b.components.values():
        c.observed_condition_min = c.observed_condition_max = float(c.current_condition)
    env._assessed_cache.clear()
    env.refresh_bridge_reliability(b)
    return b


def test_default_e_golden_invariati():
    p = SimParameters()
    assert (p.assessment_parameter_mode, p.assessment_state_mode, p.assessment_seed_mode,
            p.inspection_sigma_scale) == ("mean", "last_observation", "distinct", 1.0)
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


def test_validazione():
    for k, v in (("assessment_parameter_mode", "x"), ("assessment_state_mode", "x"),
                 ("assessment_seed_mode", "x"), ("inspection_sigma_scale", 0.0)):
        with pytest.raises(ValueError):
            SimParameters(**{k: v})


def test_tutto_allineato_stima_uguale_al_vero():
    """Con parametri campionati, stessi campioni MC, incertezza di riferimento e osservazione
    perfetta, la stima coincide con il vero: lo scarto residuo e' tutto nei meccanismi spenti."""
    env = _env(TUTTO_ALLINEATO)
    b = _osserva_esatto(env)
    est = env._assessed_bridge(b)
    assert float(est.annual_reliability_index) == pytest.approx(float(b.annual_reliability_index), abs=1e-6)


def test_default_stima_diversa_dal_vero():
    env = _env(BASE)
    b = _osserva_esatto(env)
    est = env._assessed_bridge(b)
    assert abs(float(est.annual_reliability_index) - float(b.annual_reliability_index)) > 1e-3


def test_proiezione_abbassa_la_condizione():
    env = _env({**BASE, "assessment_state_mode": "projected"})
    b = _osserva_esatto(env)
    oggi = {n: c.current_condition for n, c in env._assessed_bridge(b).components.items()}
    env.quarter += 8
    dopo = env._assessed_bridge(b).components
    assert all(dopo[n].current_condition <= oggi[n] + 1e-12 for n in oggi)
    assert any(dopo[n].current_condition < oggi[n] - 1e-6 for n in oggi)


def test_scala_sigma_ispezioni():
    env1, env3 = _env(BASE), _env({**BASE, "inspection_sigma_scale": 3.0})
    c1 = next(iter(env1.bridges[1].components.values()))
    c3 = next(iter(env3.bridges[1].components.values()))
    assert env3._component_observation_sigma(c3, 0.1) == pytest.approx(3.0 * env1._component_observation_sigma(c1, 0.1))
