"""Test del modello di degrado Weibull (Component) e dell'inversione eta' efficace."""
import math

import numpy as np
import pytest

from infralab.core.environment import Component


EXPONENT = 3.0


@pytest.fixture
def component():
    return Component("impalcato", eta_iniziale=20.0, vita_utile=80, exponent=EXPONENT,
                     rng=np.random.default_rng(1234))


def expected_condition(comp: Component, exponent: float) -> float:
    age_ratio = comp.effective_age_years / comp.vita_utile_anni
    degradation = comp._scale() * (max(age_ratio, 0.0) ** comp._shape(exponent))
    return max(0.0, min(1.0, math.exp(-degradation)))


class TestUpdateCondition:
    def test_new_component_perfect(self):
        comp = Component("impalcato", eta_iniziale=0.0, vita_utile=80, exponent=EXPONENT,
                         rng=np.random.default_rng(0))
        assert comp.current_condition == 1.0

    def test_condition_matches_weibull_survival(self, component):
        assert component.current_condition == pytest.approx(
            expected_condition(component, EXPONENT), rel=1e-12
        )

    def test_condition_monotone_decreasing_with_age(self, component):
        conditions = []
        for _ in range(40):
            component.effective_age_years += 1.0
            component.update_condition(EXPONENT)
            conditions.append(component.current_condition)
        assert all(a >= b for a, b in zip(conditions, conditions[1:]))

    def test_condition_bounded(self, component):
        component.effective_age_years = 1e6
        component.update_condition(EXPONENT)
        assert 0.0 <= component.current_condition <= 1.0


class TestMaintenanceInversion:
    def test_partial_gain_roundtrip(self, component):
        """Dopo apply_maintenance, update_condition deve riprodurre la stessa condizione:
        l'eta' efficace ricavata per inversione e' consistente con la curva."""
        component.apply_maintenance(gain=0.2, exponent=EXPONENT)
        condition_after = component.current_condition
        component.update_condition(EXPONENT)
        assert component.current_condition == pytest.approx(condition_after, rel=1e-9)

    def test_full_gain_resets(self, component):
        component.apply_maintenance(gain=1.0, exponent=EXPONENT)
        assert component.current_condition == 1.0
        assert component.effective_age_years == 0.0

    def test_gain_reduces_effective_age(self, component):
        age_before = component.effective_age_years
        recovered = component.apply_maintenance(gain=0.15, exponent=EXPONENT)
        assert component.effective_age_years < age_before
        assert recovered == pytest.approx(age_before - component.effective_age_years)

    def test_maintenance_reduces_epistemic_uncertainty(self, component):
        eps_before = component.epistemic_uncertainty
        component.apply_maintenance(gain=0.1, exponent=EXPONENT)
        assert component.epistemic_uncertainty < eps_before


class TestDamageInversion:
    def test_damage_roundtrip(self, component):
        component.apply_damage(loss=0.15, exponent=EXPONENT)
        condition_after = component.current_condition
        component.update_condition(EXPONENT)
        assert component.current_condition == pytest.approx(condition_after, rel=1e-9)

    def test_damage_increases_effective_age(self, component):
        age_before = component.effective_age_years
        component.apply_damage(loss=0.15, exponent=EXPONENT)
        assert component.effective_age_years > age_before

    def test_total_damage_maxes_age(self, component):
        component.apply_damage(loss=1.0, exponent=EXPONENT)
        assert component.current_condition == 0.0
        assert component.effective_age_years == float(component.vita_utile_anni)

    def test_damage_increases_epistemic_uncertainty(self, component):
        eps_before = component.epistemic_uncertainty
        component.apply_damage(loss=0.1, exponent=EXPONENT)
        assert component.epistemic_uncertainty > eps_before


class TestSerialization:
    def test_roundtrip_preserves_state(self, component):
        component.apply_damage(loss=0.05, exponent=EXPONENT)
        data = dict(component.to_dict())
        clone = Component.from_dict(data, EXPONENT)
        assert clone.current_condition == pytest.approx(component.current_condition)
        assert clone.effective_age_years == pytest.approx(component.effective_age_years)
        assert clone.degradation_scale == pytest.approx(component.degradation_scale)
        assert clone.epistemic_uncertainty == pytest.approx(component.epistemic_uncertainty)
