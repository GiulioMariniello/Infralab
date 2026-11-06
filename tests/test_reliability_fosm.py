"""Test del motore FOSM e delle conversioni di affidabilita' in reliability_model."""
import math

import pytest

import infralab.core.reliability as rm


BASE_LIMIT_STATE = {
    "resistance_mean": 2.0,
    "dead_load_mean": 0.5,
    "live_load_mean": 0.4,
    "code_live_effect": 0.7,
    "category": "ULS",
    "failure_mode_type": "ductile",
}


def make_states(**overrides):
    state = dict(BASE_LIMIT_STATE)
    state.update(overrides)
    return {"flexure": state}


class TestConversions:
    @pytest.mark.parametrize("beta", [0.5, 1.5, 2.5, 3.8, 4.7, 6.0])
    def test_pf_beta_roundtrip(self, beta):
        assert rm.beta_from_pf(rm.pf_from_beta(beta)) == pytest.approx(beta, abs=1e-9)

    def test_pf_from_beta_known_value(self):
        # Phi(-3.8) circa 7.23e-5 (EN1990 CC2, 50 anni)
        assert rm.pf_from_beta(3.8) == pytest.approx(7.2348e-5, rel=1e-3)

    @pytest.mark.parametrize("annual_pf,years", [(1e-4, 50), (1e-6, 100), (0.01, 1)])
    def test_reference_period_roundtrip(self, annual_pf, years):
        ref = rm.reference_period_pf(annual_pf, years)
        assert rm.annual_pf_from_reference(ref, years) == pytest.approx(annual_pf, rel=1e-9)

    def test_reference_period_one_year_identity(self):
        assert rm.reference_period_pf(1e-3, 1) == pytest.approx(1e-3)

    def test_combine_independent_probabilities(self):
        combined = rm.combine_independent_probabilities(0.01, 0.02)
        assert combined == pytest.approx(1.0 - 0.99 * 0.98, rel=1e-9)


class TestTargets:
    def test_cc2_default(self):
        t = rm.get_target_reliability("CC2", 50)
        assert t.beta_reference == 3.8
        # beta annuale derivato dalla conversione del periodo di riferimento
        expected_annual = rm.beta_from_pf(rm.annual_pf_from_reference(rm.pf_from_beta(3.8), 50))
        assert t.beta_annual == pytest.approx(expected_annual)

    @pytest.mark.parametrize("cc,beta_ref", [("CC1", 3.3), ("CC2", 3.8), ("CC3", 4.3)])
    def test_cc_table(self, cc, beta_ref):
        assert rm.get_target_reliability(cc, 50).beta_reference == beta_ref

    def test_unknown_cc_falls_back(self):
        assert rm.get_target_reliability("CCX", 50).beta_reference == 3.8

    def test_sls_override(self):
        global_t = rm.get_target_reliability("CC2", 50)
        sls = rm.get_limit_state_target({"category": "SLS"}, global_t, sls_beta_reference=1.5)
        assert sls.beta_reference == 1.5

    def test_brittle_increment(self):
        global_t = rm.get_target_reliability("CC2", 50)
        brittle = rm.get_limit_state_target(
            {"category": "ULS", "failure_mode_type": "brittle"}, global_t, brittle_beta_increment=0.3
        )
        assert brittle.beta_reference == pytest.approx(4.1)


class TestEvaluateLimitStates:
    def test_deterministic_given_seed(self):
        t = rm.get_target_reliability()
        r1 = rm.evaluate_limit_states(make_states(), t, seed=42)
        r2 = rm.evaluate_limit_states(make_states(), t, seed=42)
        assert r1.to_dict() == r2.to_dict()

    def test_seed_changes_results(self):
        t = rm.get_target_reliability()
        r1 = rm.evaluate_limit_states(make_states(), t, seed=1)
        r2 = rm.evaluate_limit_states(make_states(), t, seed=2)
        assert r1.annual_beta != r2.annual_beta

    def test_beta_monotone_in_resistance(self):
        t = rm.get_target_reliability()
        betas = [
            rm.evaluate_limit_states(make_states(resistance_mean=r), t, seed=7).annual_beta
            for r in (1.2, 1.6, 2.0, 2.6)
        ]
        assert betas == sorted(betas)

    def test_fosm_matches_sample_moments(self):
        """beta annuale = ln(mu_R/mu_S)/sqrt(sigma_lnR^2+sigma_lnS^2) sui momenti campionari."""
        t = rm.get_target_reliability()
        result = rm.evaluate_limit_states(make_states(), t, seed=11)
        ls = result.limit_state_results["flexure"]
        sig_r = math.sqrt(math.log(1 + ls["resistance_cov"] ** 2))
        sig_s = math.sqrt(math.log(1 + ls["demand_cov"] ** 2))
        expected = math.log(ls["resistance_mean"] / ls["demand_mean"]) / math.sqrt(sig_r**2 + sig_s**2)
        assert ls["annual_beta"] == pytest.approx(expected, rel=1e-9)

    def test_uls_aggregation_is_series_of_annual_pf(self):
        t = rm.get_target_reliability()
        states = {
            "flexure": dict(BASE_LIMIT_STATE),
            "shear": dict(BASE_LIMIT_STATE, failure_mode_type="brittle", resistance_mean=1.8),
        }
        result = rm.evaluate_limit_states(states, t, seed=3)
        pfs = [ls["annual_failure_probability"] for ls in result.limit_state_results.values()]
        assert result.structural_annual_failure_probability == pytest.approx(
            rm.combine_independent_probabilities(*pfs), rel=1e-9
        )

    def test_sls_state_separated_from_structural(self):
        t = rm.get_target_reliability()
        states = {
            "flexure": dict(BASE_LIMIT_STATE),
            "cracking": dict(BASE_LIMIT_STATE, category="SLS", resistance_mean=1.2),
        }
        result = rm.evaluate_limit_states(states, t, seed=5)
        assert result.serviceability_annual_failure_probability == pytest.approx(
            result.limit_state_results["cracking"]["annual_failure_probability"], rel=1e-9
        )
        assert result.structural_annual_failure_probability == pytest.approx(
            result.limit_state_results["flexure"]["annual_failure_probability"], rel=1e-9
        )

    def test_governing_limit_state_has_min_margin(self):
        t = rm.get_target_reliability()
        states = {
            "strong": dict(BASE_LIMIT_STATE, resistance_mean=3.0),
            "weak": dict(BASE_LIMIT_STATE, resistance_mean=1.3),
        }
        result = rm.evaluate_limit_states(states, t, seed=9)
        assert result.governing_limit_state == "weak"


class TestValidation:
    def test_empty_limit_states_raises(self):
        t = rm.get_target_reliability()
        with pytest.raises(rm.LimitStateSpecError, match="vuoto"):
            rm.evaluate_limit_states({}, t)

    def test_missing_required_key_raises_with_context(self):
        t = rm.get_target_reliability()
        state = dict(BASE_LIMIT_STATE)
        del state["resistance_mean"]
        with pytest.raises(rm.LimitStateSpecError, match="flexure.*resistance_mean"):
            rm.evaluate_limit_states({"flexure": state}, t)

    def test_non_finite_value_raises(self):
        t = rm.get_target_reliability()
        with pytest.raises(rm.LimitStateSpecError, match="finito"):
            rm.evaluate_limit_states(make_states(resistance_mean=float("nan")), t)

    def test_negative_value_raises(self):
        t = rm.get_target_reliability()
        with pytest.raises(rm.LimitStateSpecError, match="negativo"):
            rm.evaluate_limit_states(make_states(dead_load_mean=-1.0), t)

    def test_proof_load_requires_code_live_effect(self):
        t = rm.get_target_reliability()
        state = dict(BASE_LIMIT_STATE)
        del state["code_live_effect"]
        with pytest.raises(rm.LimitStateSpecError, match="code_live_effect"):
            rm.assess_proof_load({"flexure": state}, t, alpha=0.9)


class TestSeriesSystem:
    def test_empty_returns_safe_sentinel(self):
        result = rm.series_system_reliability([])
        assert result.annual_beta == 8.0
        assert result.annual_failure_probability == pytest.approx(1e-12)

    def test_two_components_product_rule(self):
        t = rm.get_target_reliability()
        r1 = rm.evaluate_limit_states(make_states(resistance_mean=1.8), t, seed=1)
        r2 = rm.evaluate_limit_states(make_states(resistance_mean=2.2), t, seed=2)
        system = rm.series_system_reliability([r1, r2])
        expected = rm.combine_independent_probabilities(
            r1.annual_failure_probability, r2.annual_failure_probability
        )
        assert system.annual_failure_probability == pytest.approx(expected, rel=1e-9)
        # il sistema serie e' sempre meno affidabile del componente migliore
        assert system.annual_beta <= min(r1.annual_beta, r2.annual_beta) + 1e-9

    def test_extra_probability_included(self):
        t = rm.get_target_reliability()
        r1 = rm.evaluate_limit_states(make_states(), t, seed=1)
        base = rm.series_system_reliability([r1])
        with_extra = rm.series_system_reliability([r1], extra_annual_failure_probability=1e-3)
        assert with_extra.annual_failure_probability > base.annual_failure_probability
