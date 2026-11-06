import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from statistics import NormalDist
from typing import Dict, Iterable

import numpy as np
from scipy.special import ndtr


_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329

DEFAULT_TARGET_BETA_REFERENCE = {
    "CC1": 3.3,
    "CC2": 3.8,
    "CC3": 4.3,
}

DEFAULT_TARGET_BETA_ANNUAL = {
    "CC1": 4.2,
    "CC2": 4.7,
    "CC3": 5.2,
}

# COV di default per le variabili aleatorie degli stati limite (JCSS PMC).
DEFAULT_MATERIAL_COV = 0.10
DEFAULT_GEOMETRY_COV = 0.04
DEFAULT_MODEL_RESISTANCE_COV = 0.08
DEFAULT_MODEL_DEMAND_COV = 0.06
DEFAULT_DEAD_LOAD_COV = 0.08
DEFAULT_LIVE_LOAD_COV = 0.20
DEFAULT_PROOF_LOAD_COV = 0.04

# beta convenzionale usato quando la varianza e' degenere ("praticamente sicuro").
BETA_SAFE_CEILING = 8.0
# Sentinella per la selezione dello stato limite governante quando il margine e' None.
GOVERNING_MARGIN_SENTINEL = 999.0
# Floor del COV materiale posteriore dopo proof load (residuo irriducibile).
POSTERIOR_MIN_MATERIAL_COV = 0.01
# Floor e fattore di riduzione del COV di modello della domanda dopo proof load:
# il test informa marginalmente anche la risposta strutturale (riduzione 5%).
POSTERIOR_MIN_MODEL_DEMAND_COV = 0.02
POSTERIOR_DEMAND_MODEL_FACTOR = 0.95

REQUIRED_LIMIT_STATE_KEYS = ("resistance_mean", "dead_load_mean", "live_load_mean")


class LimitStateSpecError(ValueError):
    """Specifica di stato limite non valida (chiavi mancanti o valori non ammissibili)."""


def _validate_limit_states(
    limit_states: Dict[str, Dict[str, float]],
    require_code_live_effect: bool = False,
) -> None:
    if not limit_states:
        raise LimitStateSpecError("limit_states è vuoto: serve almeno uno stato limite.")
    required = REQUIRED_LIMIT_STATE_KEYS + (("code_live_effect",) if require_code_live_effect else ())
    for name, state in limit_states.items():
        if not isinstance(state, dict):
            raise LimitStateSpecError(f"Stato limite '{name}': attesa un dict, trovato {type(state).__name__}.")
        for key in required:
            if key not in state:
                raise LimitStateSpecError(f"Stato limite '{name}': chiave obbligatoria '{key}' mancante.")
            value = state[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise LimitStateSpecError(
                    f"Stato limite '{name}': '{key}' deve essere un numero finito, trovato {value!r}."
                )
            if float(value) < 0.0:
                raise LimitStateSpecError(f"Stato limite '{name}': '{key}' non può essere negativo ({value!r}).")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sigma_ln(cov: float) -> float:
    cov = max(cov, 1e-9)
    return math.sqrt(math.log(1.0 + cov ** 2))


def _cov(samples: np.ndarray) -> float:
    mean = float(np.mean(samples))
    if mean <= 0.0:
        return 0.0
    std = float(np.std(samples, ddof=0))
    return max(0.0, std / mean)


def pf_from_beta(beta: float) -> float:
    return _clamp(_NORMAL.cdf(-beta), 1e-12, 1.0 - 1e-12)


def beta_from_pf(probability: float) -> float:
    probability = _clamp(probability, 1e-12, 1.0 - 1e-12)
    return -_NORMAL.inv_cdf(probability)


def reference_period_pf(annual_pf: float, years: int) -> float:
    annual_pf = _clamp(annual_pf, 1e-12, 1.0 - 1e-12)
    years = max(1, int(years))
    return _clamp(1.0 - (1.0 - annual_pf) ** years, 1e-12, 1.0 - 1e-12)


def annual_pf_from_reference(reference_pf: float, years: int) -> float:
    reference_pf = _clamp(reference_pf, 1e-12, 1.0 - 1.0e-12)
    years = max(1, int(years))
    return _clamp(1.0 - (1.0 - reference_pf) ** (1.0 / years), 1e-12, 1.0 - 1.0e-12)


@dataclass
class ReliabilityTarget:
    consequence_class: str
    reference_period_years: int
    beta_reference: float
    beta_annual: float
    pf_reference: float
    pf_annual: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def get_target_reliability(
    consequence_class: str = "CC2",
    reference_period_years: int = 50,
    beta_reference_override: float | None = None,
    beta_annual_override: float | None = None,
) -> ReliabilityTarget:
    cc = (consequence_class or "CC2").upper()
    beta_reference = float(beta_reference_override) if beta_reference_override is not None else DEFAULT_TARGET_BETA_REFERENCE.get(cc, 3.8)
    if beta_annual_override is not None:
        beta_annual = float(beta_annual_override)
    else:
        annual_pf = annual_pf_from_reference(pf_from_beta(beta_reference), reference_period_years)
        beta_annual = beta_from_pf(annual_pf)
    return ReliabilityTarget(
        consequence_class=cc,
        reference_period_years=max(1, int(reference_period_years)),
        beta_reference=beta_reference,
        beta_annual=beta_annual,
        pf_reference=pf_from_beta(beta_reference),
        pf_annual=pf_from_beta(beta_annual),
    )


def get_limit_state_target(
    state: Dict[str, float],
    global_target: ReliabilityTarget,
    sls_beta_reference: float = 1.5,
    brittle_beta_increment: float = 0.0,
) -> ReliabilityTarget:
    category = (state.get("category") or "ULS").upper()
    failure_mode_type = (state.get("failure_mode_type") or "ductile").lower()
    if category == "SLS":
        return get_target_reliability(
            consequence_class=global_target.consequence_class,
            reference_period_years=global_target.reference_period_years,
            beta_reference_override=sls_beta_reference,
            beta_annual_override=None,
        )
    beta_reference = global_target.beta_reference
    if failure_mode_type == "brittle":
        beta_reference += brittle_beta_increment
    return get_target_reliability(
        consequence_class=global_target.consequence_class,
        reference_period_years=global_target.reference_period_years,
        beta_reference_override=beta_reference,
        beta_annual_override=None,
    )


@dataclass
class LimitStateResult:
    name: str
    category: str
    failure_mode_type: str
    beta_reference: float
    failure_probability: float
    annual_beta: float
    annual_failure_probability: float
    reserve_ratio: float
    resistance_mean: float
    resistance_cov: float
    demand_mean: float
    demand_cov: float
    target_beta_reference: float | None = None
    target_beta_annual: float | None = None
    beta_margin_reference: float | None = None
    beta_margin_annual: float | None = None

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ReliabilityResult:
    beta: float
    failure_probability: float
    demand_mean: float
    demand_cov: float
    resistance_mean: float
    resistance_cov: float
    reserve_ratio: float
    annual_beta: float
    annual_failure_probability: float
    fragility_median: float | None = None
    fragility_beta: float | None = None
    annual_failure_rate: float | None = None
    target_beta_reference: float | None = None
    target_beta_annual: float | None = None
    beta_margin_reference: float | None = None
    beta_margin_annual: float | None = None
    consequence_class: str | None = None
    reference_period_years: int = 50
    governing_limit_state: str | None = None
    limit_state_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    proof_load_assessment: Dict[str, float] | None = None
    structural_beta: float | None = None
    structural_failure_probability: float | None = None
    structural_annual_beta: float | None = None
    structural_annual_failure_probability: float | None = None
    serviceability_beta: float | None = None
    serviceability_failure_probability: float | None = None
    serviceability_annual_beta: float | None = None
    serviceability_annual_failure_probability: float | None = None

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Un seme distinto per (asset, componente, anno di proiezione): con 24 posti la
# cache veniva svuotata a ogni trimestre. 4096 banchi <= ~130 MB nel caso peggiore
# (256 campioni); tipicamente 128 campioni -> ~65 MB. Numericamente neutro.
@lru_cache(maxsize=4096)
def _normal_bank(sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sample_count, 8))
    u = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
    return z, u


def _sample_lognormal(mean: float, cov: float, z: np.ndarray) -> np.ndarray:
    mean = max(mean, 1e-9)
    sigma = _sigma_ln(cov)
    mu = math.log(mean) - 0.5 * sigma ** 2
    return np.exp(mu + sigma * z)


def _sample_normal(mean: float, cov: float, z: np.ndarray) -> np.ndarray:
    mean = max(mean, 1e-9)
    std = max(mean * cov, 1e-9)
    return np.clip(mean + std * z, 1e-9, None)


def _sample_gumbel(mean: float, cov: float, u: np.ndarray) -> np.ndarray:
    mean = max(mean, 1e-9)
    std = max(mean * cov, 1e-9)
    scale = max(std * math.sqrt(6.0) / math.pi, 1e-9)
    loc = mean - _EULER_GAMMA * scale
    return np.clip(loc - scale * np.log(-np.log(u)), 1e-9, None)


def _state_samples(limit_state: Dict[str, float], sample_count: int, seed: int) -> Dict[str, np.ndarray]:
    z, u = _normal_bank(sample_count, seed)
    r_material = _sample_lognormal(1.0, limit_state.get("material_cov", DEFAULT_MATERIAL_COV), z[:, 0])
    r_geometry = _sample_lognormal(1.0, limit_state.get("geometry_cov", DEFAULT_GEOMETRY_COV), z[:, 1])
    theta_r = _sample_lognormal(1.0, limit_state.get("model_resistance_cov", DEFAULT_MODEL_RESISTANCE_COV), z[:, 2])
    theta_s = _sample_lognormal(1.0, limit_state.get("model_demand_cov", DEFAULT_MODEL_DEMAND_COV), z[:, 3])
    g_dead = _sample_normal(limit_state["dead_load_mean"], limit_state.get("dead_load_cov", DEFAULT_DEAD_LOAD_COV), z[:, 4])
    if limit_state.get("live_load_distribution", "gumbel") == "gumbel":
        q_live = _sample_gumbel(limit_state["live_load_mean"], limit_state.get("live_load_cov", DEFAULT_LIVE_LOAD_COV), u[:, 5])
    else:
        q_live = _sample_lognormal(limit_state["live_load_mean"], limit_state.get("live_load_cov", DEFAULT_LIVE_LOAD_COV), z[:, 5])
    resistance = np.clip(limit_state["resistance_mean"] * r_material * r_geometry * theta_r, 1e-9, None)
    demand = np.clip(theta_s * (g_dead + q_live), 1e-9, None)
    return {
        "resistance": resistance,
        "demand": demand,
        "dead": g_dead,
        "live": q_live,
        "theta_test": _sample_lognormal(1.0, limit_state.get("proof_load_cov", DEFAULT_PROOF_LOAD_COV), z[:, 6]),
    }


def _summarize_limit_state(
    name: str,
    state: Dict[str, float],
    resistance_samples: np.ndarray,
    demand_samples: np.ndarray,
    targets: ReliabilityTarget,
) -> LimitStateResult:
    resistance_mean = float(np.mean(resistance_samples))
    demand_mean = float(np.mean(demand_samples))
    reserve_ratio = resistance_mean / max(demand_mean, 1e-9)
    r_cov = _cov(resistance_samples)
    s_cov = _cov(demand_samples)

    # FOSM from sample moments: lognormal approximation on R and S.
    # Replaces direct tail-counting (np.mean(R<=S)) which gives 0 failures for
    # β > ~3 at N=256, clamping all safe bridges to the same ceiling value.
    # Sample means and COVs converge quickly, so this is stable at any N.
    sigma_ln_r = _sigma_ln(r_cov)
    sigma_ln_s = _sigma_ln(s_cov)
    denom = math.sqrt(sigma_ln_r ** 2 + sigma_ln_s ** 2)
    if denom > 1e-9 and resistance_mean > 1e-9 and demand_mean > 1e-9:
        annual_beta = math.log(resistance_mean / demand_mean) / denom
    else:
        annual_beta = BETA_SAFE_CEILING if resistance_mean >= demand_mean else 0.0

    annual_pf = pf_from_beta(annual_beta)
    ref_pf = reference_period_pf(annual_pf, targets.reference_period_years)
    beta_ref = beta_from_pf(ref_pf)
    return LimitStateResult(
        name=name,
        category=(state.get("category") or "ULS").upper(),
        failure_mode_type=(state.get("failure_mode_type") or "ductile").lower(),
        beta_reference=beta_ref,
        failure_probability=ref_pf,
        annual_beta=annual_beta,
        annual_failure_probability=annual_pf,
        reserve_ratio=reserve_ratio,
        resistance_mean=resistance_mean,
        resistance_cov=r_cov,
        demand_mean=demand_mean,
        demand_cov=s_cov,
        target_beta_reference=targets.beta_reference,
        target_beta_annual=targets.beta_annual,
        beta_margin_reference=beta_ref - targets.beta_reference,
        beta_margin_annual=annual_beta - targets.beta_annual,
    )


def evaluate_limit_states(
    limit_states: Dict[str, Dict[str, float]],
    targets: ReliabilityTarget,
    sample_count: int = 256,
    seed: int = 0,
    sls_beta_reference: float = 1.5,
    brittle_beta_increment: float = 0.0,
) -> ReliabilityResult:
    _validate_limit_states(limit_states)
    detailed_results: Dict[str, LimitStateResult] = {}
    for idx, (name, state) in enumerate(limit_states.items()):
        samples = _state_samples(state, sample_count, seed + idx * 101)
        ls_target = get_limit_state_target(
            state,
            global_target=targets,
            sls_beta_reference=sls_beta_reference,
            brittle_beta_increment=brittle_beta_increment,
        )
        detailed_results[name] = _summarize_limit_state(name, state, samples["resistance"], samples["demand"], ls_target)

    def _aggregated_category(category: str):
        selected = [result for result in detailed_results.values() if result.category == category]
        if not selected:
            return None
        annual_pf = combine_independent_probabilities(*[result.annual_failure_probability for result in selected])
        ref_pf = reference_period_pf(annual_pf, targets.reference_period_years)
        return {
            "annual_failure_probability": annual_pf,
            "failure_probability": ref_pf,
            "annual_beta": beta_from_pf(annual_pf),
            "beta": beta_from_pf(ref_pf),
        }

    structural_summary = _aggregated_category("ULS")
    serviceability_summary = _aggregated_category("SLS")

    governing_name = min(
        detailed_results,
        key=lambda key: min(
            detailed_results[key].beta_margin_annual if detailed_results[key].beta_margin_annual is not None else GOVERNING_MARGIN_SENTINEL,
            detailed_results[key].beta_margin_reference if detailed_results[key].beta_margin_reference is not None else GOVERNING_MARGIN_SENTINEL,
        ),
    )
    governing = detailed_results[governing_name]
    primary = structural_summary or {
        "beta": governing.beta_reference,
        "failure_probability": governing.failure_probability,
        "annual_beta": governing.annual_beta,
        "annual_failure_probability": governing.annual_failure_probability,
    }
    return ReliabilityResult(
        beta=primary["beta"],
        failure_probability=primary["failure_probability"],
        demand_mean=governing.demand_mean,
        demand_cov=governing.demand_cov,
        resistance_mean=governing.resistance_mean,
        resistance_cov=governing.resistance_cov,
        reserve_ratio=governing.reserve_ratio,
        annual_beta=primary["annual_beta"],
        annual_failure_probability=primary["annual_failure_probability"],
        annual_failure_rate=primary["annual_failure_probability"],
        target_beta_reference=targets.beta_reference,
        target_beta_annual=targets.beta_annual,
        beta_margin_reference=primary["beta"] - targets.beta_reference,
        beta_margin_annual=primary["annual_beta"] - targets.beta_annual,
        consequence_class=targets.consequence_class,
        reference_period_years=targets.reference_period_years,
        governing_limit_state=governing_name,
        limit_state_results={name: result.to_dict() for name, result in detailed_results.items()},
        structural_beta=None if structural_summary is None else structural_summary["beta"],
        structural_failure_probability=None if structural_summary is None else structural_summary["failure_probability"],
        structural_annual_beta=None if structural_summary is None else structural_summary["annual_beta"],
        structural_annual_failure_probability=None if structural_summary is None else structural_summary["annual_failure_probability"],
        serviceability_beta=None if serviceability_summary is None else serviceability_summary["beta"],
        serviceability_failure_probability=None if serviceability_summary is None else serviceability_summary["failure_probability"],
        serviceability_annual_beta=None if serviceability_summary is None else serviceability_summary["annual_beta"],
        serviceability_annual_failure_probability=None if serviceability_summary is None else serviceability_summary["annual_failure_probability"],
    )


def _lognormal_survival_posterior(
    resistance_mean: float,
    resistance_cov: float,
    demand_mean: float,
    demand_cov: float,
) -> tuple[float, float, float]:
    """
    Momenti a posteriori di R lognormale condizionata all'evento di sopravvivenza
    R > D_test, con D_test lognormale indipendente (domanda di prova incerta).

    Con X = ln R ~ N(μ_R, σ_R²) e Y = ln D ~ N(μ_D, σ_D²) indipendenti,
    s² = σ_R² + σ_D², l'evento di prova superata è {X − Y > 0} e:

        P(pass)        = Φ((μ_R − μ_D) / s)
        E[R  | pass]   = R_mean   · Φ((μ_R + σ_R²  − μ_D) / s) / P(pass)
        E[R² | pass]   = R_mean²·e^{σ_R²} · Φ((μ_R + 2σ_R² − μ_D) / s) / P(pass)

    Per σ_D → 0 si riducono esattamente alle formule della lognormale troncata
    (Aitchison & Brown 1957; JCSS PMC §3.7). Questa forma è consistente con la
    convenzione del passo Monte Carlo di assess_proof_load, dove la domanda di
    prova campionata include l'incertezza θ_test e quella del peso proprio.

    Returns
    -------
    posterior_mean : float
    posterior_cov  : float   (COV composito della distribuzione condizionata)
    pass_probability : float  P(R > D_test)
    """
    if resistance_mean <= 0 or resistance_cov <= 0 or demand_mean <= 0:
        return resistance_mean, max(resistance_cov, 1e-4), 1.0

    sigma_r = _sigma_ln(resistance_cov)
    mu_r = math.log(resistance_mean) - 0.5 * sigma_r ** 2
    sigma_d = _sigma_ln(demand_cov) if demand_cov > 0 else 0.0
    mu_d = math.log(max(demand_mean, 1e-12)) - 0.5 * sigma_d ** 2

    s = math.sqrt(sigma_r ** 2 + sigma_d ** 2)
    if s < 1e-9:
        return resistance_mean, resistance_cov, 1.0 if resistance_mean > demand_mean else 1e-12

    m = mu_r - mu_d
    pass_prob = _NORMAL.cdf(m / s)

    if pass_prob < 1e-9:
        return resistance_mean, resistance_cov, max(pass_prob, 1e-12)

    e_r = resistance_mean * _NORMAL.cdf((m + sigma_r ** 2) / s) / pass_prob
    e_r2 = resistance_mean ** 2 * math.exp(sigma_r ** 2) * _NORMAL.cdf((m + 2.0 * sigma_r ** 2) / s) / pass_prob

    var_r = max(0.0, e_r2 - e_r ** 2)
    post_cov = math.sqrt(var_r) / max(e_r, 1e-9)

    return e_r, post_cov, pass_prob


def _truncated_lognormal_posterior(
    resistance_mean: float,
    resistance_cov: float,
    threshold: float,
) -> tuple[float, float, float]:
    """Caso particolare a soglia deterministica: R | R > threshold (σ_D = 0)."""
    if resistance_mean <= 0 or resistance_cov <= 0 or threshold <= 0:
        return resistance_mean, max(resistance_cov, 1e-4), 1.0
    return _lognormal_survival_posterior(resistance_mean, resistance_cov, threshold, 0.0)


def assess_proof_load(
    limit_states: Dict[str, Dict[str, float]],
    targets: ReliabilityTarget,
    alpha: float,
    sample_count: int = 256,
    seed: int = 0,
    step_count: int = 3,
    sls_beta_reference: float = 1.5,
    brittle_beta_increment: float = 0.0,
) -> Dict[str, object]:
    _validate_limit_states(limit_states, require_code_live_effect=True)
    state_samples: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, (name, state) in enumerate(limit_states.items()):
        state_samples[name] = _state_samples(state, sample_count, seed + idx * 101)

    step_count = max(1, int(step_count))
    step_alphas = np.linspace(alpha / step_count, alpha, step_count)
    alive = np.ones(sample_count, dtype=bool)
    screened = np.zeros(sample_count, dtype=bool)
    failed = np.zeros(sample_count, dtype=bool)
    step_reports = []

    for step_alpha in step_alphas:
        crack_trigger = np.zeros(sample_count, dtype=bool)
        collapse_trigger = np.zeros(sample_count, dtype=bool)
        for name, state in limit_states.items():
            samples = state_samples[name]
            test_live = state["code_live_effect"] * step_alpha
            test_demand = samples["theta_test"] * (samples["dead"] + test_live)
            failure_mask = samples["resistance"] <= test_demand
            if state.get("category", "").lower() == "sls":
                crack_trigger |= failure_mask
            else:
                collapse_trigger |= failure_mask
        newly_screened = alive & crack_trigger
        screened |= newly_screened
        alive &= ~newly_screened
        newly_failed = alive & collapse_trigger
        failed |= newly_failed
        alive &= ~newly_failed
        step_reports.append(
            {
                "alpha": float(step_alpha),
                "screening_probability": float(np.mean(newly_screened)),
                "failure_probability": float(np.mean(newly_failed)),
                "survival_probability": float(np.mean(alive)),
            }
        )

    pass_mask = alive
    pass_probability = _clamp(float(np.mean(pass_mask)), 1e-12, 1.0 - 1e-12)
    screening_probability = _clamp(float(np.mean(screened)), 0.0, 1.0 - 1e-12)
    failure_probability_during_test = _clamp(float(np.mean(failed)), 1e-12, 1.0 - 1e-12)

    # Aggiornamento bayesiano analitico: per ogni stato limite la resistenza R
    # (lognormale: prodotto di fattori LN materiale/geometria/modello) viene
    # condizionata all'evento di sopravvivenza R > D_test, dove D_test è la
    # domanda di prova ALEATORIA con la stessa convenzione del passo Monte Carlo:
    #   D_test = θ_test · (G_dead + α·code_live_effect)
    # cioè media = dead_load_mean + α·code_live_effect e COV che combina
    # l'incertezza del carico di prova (θ_test) e quella del peso proprio.
    # Geometria e incertezza di modello restano irriducibili (limiti epistemici
    # del modello strutturale): solo il COV del materiale viene aggiornato.
    # Il loop MC sopra resta la fonte delle probabilità di pass/screen/failure.
    posterior_limit_states: Dict[str, Dict[str, float]] = {}
    for name, state in limit_states.items():
        v_m   = state.get("material_cov", DEFAULT_MATERIAL_COV)
        v_g   = state.get("geometry_cov", DEFAULT_GEOMETRY_COV)
        v_mod = state.get("model_resistance_cov", DEFAULT_MODEL_RESISTANCE_COV)
        r_cov_prior = math.sqrt(v_m ** 2 + v_g ** 2 + v_mod ** 2)
        dead_mean = state["dead_load_mean"]
        test_live = alpha * state["code_live_effect"]
        test_demand_mean = dead_mean + test_live
        v_theta = state.get("proof_load_cov", DEFAULT_PROOF_LOAD_COV)
        dead_sigma = state.get("dead_load_cov", DEFAULT_DEAD_LOAD_COV) * dead_mean
        test_demand_cov = math.sqrt(
            v_theta ** 2 + (dead_sigma / max(test_demand_mean, 1e-9)) ** 2
        )
        post_r_mean, post_r_cov, _ = _lognormal_survival_posterior(
            state["resistance_mean"], r_cov_prior, test_demand_mean, test_demand_cov
        )
        posterior_state = dict(state)
        posterior_state["resistance_mean"] = post_r_mean
        # Decomposizione del COV posteriore: geometria e modello irriducibili,
        # il residuo va all'incertezza del materiale.
        v_fixed_sq = v_g ** 2 + v_mod ** 2
        new_v_m = math.sqrt(max(0.0, post_r_cov ** 2 - v_fixed_sq))
        posterior_state["material_cov"] = max(POSTERIOR_MIN_MATERIAL_COV, new_v_m)
        posterior_state["model_demand_cov"] = max(
            POSTERIOR_MIN_MODEL_DEMAND_COV,
            state.get("model_demand_cov", DEFAULT_MODEL_DEMAND_COV) * POSTERIOR_DEMAND_MODEL_FACTOR,
        )
        posterior_limit_states[name] = posterior_state

    posterior_result = evaluate_limit_states(
        posterior_limit_states,
        targets=targets,
        sample_count=sample_count,
        seed=seed + 4099,
        sls_beta_reference=sls_beta_reference,
        brittle_beta_increment=brittle_beta_increment,
    )
    posterior_result.proof_load_assessment = {
        "alpha": alpha,
        "pass_probability": pass_probability,
        "screening_probability": screening_probability,
        "failure_probability_during_test": failure_probability_during_test,
        "step_count": step_count,
        "conditioning_mode": "survival_conditioning_random_test_demand",
        "accepted_sample_ratio": float(np.mean(pass_mask)),
    }

    return {
        "alpha": alpha,
        "step_count": step_count,
        "step_reports": step_reports,
        "pass_probability": pass_probability,
        "screening_probability": screening_probability,
        "failure_probability_during_test": failure_probability_during_test,
        "posterior_result": posterior_result,
        "posterior_limit_states": posterior_limit_states,
        "conditioning_mode": "survival_conditioning_random_test_demand",
        "accepted_sample_ratio": float(np.mean(pass_mask)),
        "targets_satisfied": all(
            (ls.get("beta_margin_reference") is None or ls.get("beta_margin_reference", -999.0) >= 0.0)
            and (ls.get("beta_margin_annual") is None or ls.get("beta_margin_annual", -999.0) >= 0.0)
            for ls in posterior_result.limit_state_results.values()
        ),
    }


def combine_independent_probabilities(*probabilities: float) -> float:
    survival = 1.0
    for probability in probabilities:
        survival *= 1.0 - _clamp(probability, 1e-12, 1.0 - 1e-12)
    return _clamp(1.0 - survival, 1e-12, 1.0 - 1e-12)


def series_system_reliability(
    results: Iterable[ReliabilityResult],
    extra_annual_failure_probability: float = 0.0,
) -> ReliabilityResult:
    results = list(results)
    if not results:
        empty_target = get_target_reliability()
        return ReliabilityResult(
            beta=8.0,
            failure_probability=1e-12,
            demand_mean=0.0,
            demand_cov=0.0,
            resistance_mean=1.0,
            resistance_cov=0.0,
            reserve_ratio=1.0,
            annual_beta=8.0,
            annual_failure_probability=1e-12,
            annual_failure_rate=1e-12,
            target_beta_reference=empty_target.beta_reference,
            target_beta_annual=empty_target.beta_annual,
            beta_margin_reference=8.0 - empty_target.beta_reference,
            beta_margin_annual=8.0 - empty_target.beta_annual,
            consequence_class=empty_target.consequence_class,
            reference_period_years=empty_target.reference_period_years,
            structural_beta=8.0,
            structural_failure_probability=1e-12,
            structural_annual_beta=8.0,
            structural_annual_failure_probability=1e-12,
        )

    annual_probs = [result.annual_failure_probability for result in results]
    system_annual_pf = combine_independent_probabilities(*annual_probs, extra_annual_failure_probability)
    reference_period_years = results[0].reference_period_years
    system_reference_pf = reference_period_pf(system_annual_pf, reference_period_years)
    annual_beta = beta_from_pf(system_annual_pf)
    beta_ref = beta_from_pf(system_reference_pf)
    min_reserve = min(r.reserve_ratio for r in results)
    avg_demand = sum(r.demand_mean for r in results) / len(results)
    avg_resistance = sum(r.resistance_mean for r in results) / len(results)
    avg_demand_cov = sum(r.demand_cov for r in results) / len(results)
    avg_resistance_cov = sum(r.resistance_cov for r in results) / len(results)
    target_beta_reference = results[0].target_beta_reference
    target_beta_annual = results[0].target_beta_annual

    return ReliabilityResult(
        beta=beta_ref,
        failure_probability=system_reference_pf,
        demand_mean=avg_demand,
        demand_cov=avg_demand_cov,
        resistance_mean=avg_resistance,
        resistance_cov=avg_resistance_cov,
        reserve_ratio=min_reserve,
        annual_beta=annual_beta,
        annual_failure_probability=system_annual_pf,
        annual_failure_rate=system_annual_pf,
        target_beta_reference=target_beta_reference,
        target_beta_annual=target_beta_annual,
        beta_margin_reference=None if target_beta_reference is None else beta_ref - target_beta_reference,
        beta_margin_annual=None if target_beta_annual is None else annual_beta - target_beta_annual,
        consequence_class=results[0].consequence_class,
        reference_period_years=reference_period_years,
        governing_limit_state="series_system",
        limit_state_results={f"component_{idx + 1}": result.to_dict() for idx, result in enumerate(results)},
        structural_beta=beta_ref,
        structural_failure_probability=system_reference_pf,
        structural_annual_beta=annual_beta,
        structural_annual_failure_probability=system_annual_pf,
    )
