"""Verifica dei momenti della lognormale troncata e dell'aggiornamento proof-load.

I momenti analitici (Aitchison & Brown) sono confrontati con integrazione
numerica scipy della distribuzione condizionata R | R > soglia.
"""
import math

import numpy as np
import pytest
from scipy import integrate, stats

import infralab.core.reliability as rm
from tests.test_reliability_fosm import BASE_LIMIT_STATE, make_states


def _lognorm(mean: float, cov: float) -> stats.rv_continuous:
    sigma = math.sqrt(math.log(1 + cov**2))
    mu = math.log(mean) - 0.5 * sigma**2
    return stats.lognorm(s=sigma, scale=math.exp(mu))


def numeric_truncated_moments(mean: float, cov: float, threshold: float):
    dist = _lognorm(mean, cov)
    pass_prob = dist.sf(threshold)
    upper = dist.ppf(1 - 1e-12)
    e1, _ = integrate.quad(lambda x: x * dist.pdf(x), threshold, upper, limit=200)
    e2, _ = integrate.quad(lambda x: x * x * dist.pdf(x), threshold, upper, limit=200)
    e1 /= pass_prob
    e2 /= pass_prob
    var = max(0.0, e2 - e1**2)
    return e1, math.sqrt(var) / e1, pass_prob


class TestTruncatedLognormal:
    @pytest.mark.parametrize(
        "mean,cov,threshold",
        [
            (2.0, 0.15, 1.5),   # soglia sotto la media
            (2.0, 0.15, 2.0),   # soglia alla media
            (2.0, 0.15, 2.5),   # soglia sopra la media
            (1.0, 0.30, 1.2),   # COV alto
            (5.0, 0.05, 4.9),   # COV basso
        ],
    )
    def test_moments_match_numeric_integration(self, mean, cov, threshold):
        e_r, post_cov, pass_prob = rm._truncated_lognormal_posterior(mean, cov, threshold)
        n_e_r, n_cov, n_pass = numeric_truncated_moments(mean, cov, threshold)
        assert pass_prob == pytest.approx(n_pass, rel=1e-6)
        assert e_r == pytest.approx(n_e_r, rel=1e-6)
        assert post_cov == pytest.approx(n_cov, rel=1e-4)

    def test_posterior_mean_never_below_prior(self):
        # il condizionamento su R > soglia puo' solo aumentare la media
        for threshold in (0.5, 1.0, 2.0, 3.0):
            e_r, _, _ = rm._truncated_lognormal_posterior(2.0, 0.2, threshold)
            assert e_r >= 2.0 - 1e-12

    def test_posterior_cov_never_above_prior(self):
        for threshold in (0.5, 1.0, 2.0, 3.0):
            _, post_cov, _ = rm._truncated_lognormal_posterior(2.0, 0.2, threshold)
            assert post_cov <= 0.2 + 1e-12

    def test_degenerate_inputs_return_prior(self):
        assert rm._truncated_lognormal_posterior(0.0, 0.1, 1.0)[0] == 0.0
        mean, cov, pass_prob = rm._truncated_lognormal_posterior(2.0, 0.1, 0.0)
        assert (mean, pass_prob) == (2.0, 1.0)

    def test_extreme_threshold_small_pass_probability(self):
        _, _, pass_prob = rm._truncated_lognormal_posterior(1.0, 0.1, 3.0)
        assert pass_prob < 1e-9


class TestSurvivalPosterior:
    """Posterior generalizzato R | R > D_test con domanda di prova aleatoria."""

    def test_reduces_to_truncation_for_deterministic_demand(self):
        gen = rm._lognormal_survival_posterior(2.0, 0.15, 1.8, 0.0)
        trunc = rm._truncated_lognormal_posterior(2.0, 0.15, 1.8)
        assert gen == pytest.approx(trunc, rel=1e-12)

    def test_matches_monte_carlo(self):
        rng = np.random.default_rng(123)
        n = 2_000_000
        r_mean, r_cov, d_mean, d_cov = 2.0, 0.15, 1.9, 0.08
        sig_r = math.sqrt(math.log(1 + r_cov**2))
        sig_d = math.sqrt(math.log(1 + d_cov**2))
        r = np.exp(math.log(r_mean) - 0.5 * sig_r**2 + sig_r * rng.standard_normal(n))
        d = np.exp(math.log(d_mean) - 0.5 * sig_d**2 + sig_d * rng.standard_normal(n))
        passed = r > d
        e_r, post_cov, pass_prob = rm._lognormal_survival_posterior(r_mean, r_cov, d_mean, d_cov)
        assert pass_prob == pytest.approx(float(np.mean(passed)), rel=2e-3)
        assert e_r == pytest.approx(float(np.mean(r[passed])), rel=2e-3)
        assert post_cov == pytest.approx(float(np.std(r[passed]) / np.mean(r[passed])), rel=5e-3)

    def test_uncertain_demand_less_informative(self):
        """A parita' di media, una domanda di prova incerta seleziona meno la coda:
        il guadagno sulla media posteriore e' minore che con soglia deterministica."""
        det_mean, _, _ = rm._lognormal_survival_posterior(2.0, 0.2, 1.9, 0.0)
        unc_mean, _, _ = rm._lognormal_survival_posterior(2.0, 0.2, 1.9, 0.15)
        assert det_mean >= unc_mean - 1e-12


class TestAssessProofLoad:
    def make_targets(self):
        return rm.get_target_reliability("CC2", 50)

    def test_deterministic_given_seed(self):
        t = self.make_targets()
        r1 = rm.assess_proof_load(make_states(), t, alpha=0.9, seed=42)
        r2 = rm.assess_proof_load(make_states(), t, alpha=0.9, seed=42)
        assert r1["pass_probability"] == r2["pass_probability"]
        assert r1["posterior_result"].to_dict() == r2["posterior_result"].to_dict()

    def test_posterior_beta_not_below_prior(self):
        """Evidenza positiva (test superato) non puo' ridurre l'affidabilita'."""
        t = self.make_targets()
        prior = rm.evaluate_limit_states(make_states(), t, seed=42)
        assessment = rm.assess_proof_load(make_states(), t, alpha=0.9, seed=42)
        posterior = assessment["posterior_result"]
        assert posterior.annual_beta >= prior.annual_beta - 0.05

    def test_higher_alpha_more_informative(self):
        t = self.make_targets()
        low = rm.assess_proof_load(make_states(), t, alpha=0.5, seed=42)
        high = rm.assess_proof_load(make_states(), t, alpha=1.2, seed=42)
        # test piu' severo: meno probabilita' di passare, posterior piu' forte
        assert high["pass_probability"] <= low["pass_probability"]
        assert (
            high["posterior_result"].annual_beta
            >= low["posterior_result"].annual_beta - 1e-9
        )

    def test_step_reports_structure(self):
        t = self.make_targets()
        result = rm.assess_proof_load(make_states(), t, alpha=0.9, seed=1, step_count=3)
        assert len(result["step_reports"]) == 3
        alphas = [s["alpha"] for s in result["step_reports"]]
        assert alphas == sorted(alphas)
        assert alphas[-1] == pytest.approx(0.9)
        total = (
            result["pass_probability"]
            + result["screening_probability"]
            + result["failure_probability_during_test"]
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_sls_state_screens_not_fails(self):
        t = self.make_targets()
        states = {
            "flexure": dict(BASE_LIMIT_STATE),
            "cracking": dict(BASE_LIMIT_STATE, category="SLS", resistance_mean=1.0),
        }
        result = rm.assess_proof_load(states, t, alpha=0.9, seed=7)
        # lo stato SLS debole deve produrre screening, non collasso
        assert result["screening_probability"] > result["failure_probability_during_test"]

    def test_material_cov_reduced_geometry_preserved(self):
        t = self.make_targets()
        result = rm.assess_proof_load(make_states(material_cov=0.15, geometry_cov=0.04), t, alpha=1.0, seed=3)
        post_state = result["posterior_limit_states"]["flexure"]
        assert post_state["material_cov"] <= 0.15
        assert post_state["geometry_cov"] == 0.04
