"""Test estensioni fisiche: processo gamma, modello ispezioni, non-stazionarieta'."""
import json

import numpy as np
import pytest

from infralab.core.degradation import (
    condition_from_damage,
    damage_from_condition,
    expected_damage,
    gamma_increment,
)
from infralab.core.environment import Component
from infralab.core.inspection import inspect_component
from tests.conftest import build_env_from_scenario, load_scenario


EXPONENT = 3.0


def make_component(seed=1234, age=20.0):
    return Component("impalcato", eta_iniziale=age, vita_utile=80, exponent=EXPONENT,
                     rng=np.random.default_rng(seed))


class TestGammaProcess:
    def test_increment_mean_and_variance_match_analytic(self):
        """Incrementi Gamma(dD/theta, theta): media dD, varianza dD*theta."""
        comp = make_component()
        theta = 0.05
        rng = np.random.default_rng(7)
        age_a, age_b = 20.0, 21.0
        delta = expected_damage(comp, EXPONENT, age_b) - expected_damage(comp, EXPONENT, age_a)
        draws = np.array([
            gamma_increment(comp, EXPONENT, age_a, age_b, theta, rng) for _ in range(20000)
        ])
        assert draws.mean() == pytest.approx(delta, rel=0.02)
        assert draws.var() == pytest.approx(delta * theta, rel=0.05)

    def test_increment_nonnegative_and_zero_for_no_aging(self):
        comp = make_component()
        rng = np.random.default_rng(3)
        assert gamma_increment(comp, EXPONENT, 20.0, 20.0, 0.05, rng) == 0.0
        assert gamma_increment(comp, EXPONENT, 21.0, 20.0, 0.05, rng) == 0.0

    def test_damage_condition_roundtrip(self):
        for condition in (0.99, 0.7, 0.3, 0.05):
            assert condition_from_damage(damage_from_condition(condition)) == pytest.approx(condition, rel=1e-9)

    def test_gamma_env_runs_and_mean_tracks_weibull(self):
        """Con theta piccolo il percorso gamma segue da vicino la media Weibull."""
        scenario = load_scenario("TRAIN-01")

        def run(model, seed):
            s = json.loads(json.dumps(scenario))
            s.setdefault("params", {}).update({
                "degradation_model": model,
                "gamma_process_theta": 0.01,
                "deterioration_process_cov": 1e-6,
                "env_degradation_std": 1e-6,
                "seed": seed,
            })
            env = build_env_from_scenario(s)
            agent = env.params.agent_names[0]
            for _ in range(8):
                env.step({agent: {"maintenance": {}}})
            return {bid: b.min_condition for bid, b in env.bridges.items()}

        weibull = run("weibull", 42)
        gamma = run("gamma", 42)
        for bid in weibull:
            assert gamma[bid] == pytest.approx(weibull[bid], abs=0.03)

    def test_gamma_paths_more_variable_with_larger_theta(self):
        scenario = load_scenario("TRAIN-01")

        def final_condition(theta, seed):
            s = json.loads(json.dumps(scenario))
            s.setdefault("params", {}).update({
                "degradation_model": "gamma",
                "gamma_process_theta": theta,
                "seed": seed,
            })
            env = build_env_from_scenario(s)
            agent = env.params.agent_names[0]
            for _ in range(8):
                env.step({agent: {"maintenance": {}}})
            return min(b.min_condition for b in env.bridges.values())

        small = [final_condition(0.005, s) for s in range(10)]
        large = [final_condition(0.20, s) for s in range(10)]
        assert np.std(large) > np.std(small)

    def test_maintenance_syncs_accumulated_damage(self):
        comp = make_component()
        comp.apply_maintenance(gain=0.3, exponent=EXPONENT)
        assert comp.accumulated_damage == pytest.approx(damage_from_condition(comp.current_condition))


class TestInspectionModel:
    def test_no_bias_no_pod_centered(self):
        comp = make_component()
        comp.current_condition = 0.70  # lontano dai bordi: nessun clipping
        rng = np.random.default_rng(0)
        outcomes = [inspect_component(comp, 0.02, rng) for _ in range(2000)]
        centers = np.array([(o.observed_min + o.observed_max) / 2 for o in outcomes])
        assert centers.mean() == pytest.approx(comp.current_condition, abs=0.005)

    def test_positive_bias_optimistic(self):
        comp = make_component()
        comp.current_condition = 0.70
        rng = np.random.default_rng(0)
        outcomes = [inspect_component(comp, 0.02, rng, bias=0.10) for _ in range(2000)]
        centers = np.array([(o.observed_min + o.observed_max) / 2 for o in outcomes])
        assert centers.mean() > comp.current_condition + 0.05

    def test_pod_misses_small_damage(self):
        comp = make_component()
        comp.current_condition = 0.95  # danno piccolo
        rng = np.random.default_rng(0)
        outcomes = [inspect_component(comp, 0.02, rng, pod_enabled=True, pod_exponent=8.0)
                    for _ in range(2000)]
        detection_rate = np.mean([o.detected for o in outcomes])
        expected = 1.0 - 0.95 ** 8.0
        assert detection_rate == pytest.approx(expected, abs=0.03)

    def test_pod_detects_large_damage(self):
        comp = make_component()
        comp.current_condition = 0.3
        rng = np.random.default_rng(0)
        outcomes = [inspect_component(comp, 0.02, rng, pod_enabled=True) for _ in range(500)]
        assert np.mean([o.detected for o in outcomes]) > 0.99

    def test_bounds_respected(self):
        comp = make_component()
        comp.current_condition = 0.02
        rng = np.random.default_rng(0)
        for _ in range(200):
            o = inspect_component(comp, 0.3, rng, bias=-0.5)
            assert 0.0 <= o.observed_min <= o.observed_max <= 1.0


class TestNonStationarity:
    def run_env(self, extra_params, quarters=12):
        scenario = load_scenario("TRAIN-01")
        s = json.loads(json.dumps(scenario))
        s.setdefault("params", {}).update({"seed": 42, **extra_params})
        env = build_env_from_scenario(s)
        agent = env.params.agent_names[0]
        for _ in range(quarters):
            env.step({agent: {"maintenance": {}}})
        return env

    def test_traffic_growth_accelerates_degradation(self):
        base = self.run_env({})
        grown = self.run_env({"traffic_growth_annual": 0.20})
        assert min(b.min_condition for b in grown.bridges.values()) < \
               min(b.min_condition for b in base.bridges.values())

    def test_climate_trend_accelerates_degradation(self):
        base = self.run_env({})
        hot = self.run_env({"climate_degradation_trend_annual": 0.20})
        assert min(b.min_condition for b in hot.bridges.values()) < \
               min(b.min_condition for b in base.bridges.values())

    def test_zero_trends_are_identity(self):
        env = self.run_env({}, quarters=1)
        assert env._traffic_growth_multiplier() == 1.0
        assert env._climate_trend_multiplier() == 1.0
