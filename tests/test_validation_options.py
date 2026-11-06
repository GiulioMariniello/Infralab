"""Tre aggiunte opt-in nate dall'analisi parametrica di validazione (docs/CHANGELOG_MODEL.md):
1. assessment_epistemic_mode="reference": la stima del gestore usa l'incertezza di
   riferimento del vero (stima e vero sulla stessa scala);
2. tipi utente del danno improvviso con tasso dipendente dalla condizione vera
   (condition_max, condition_exponent) e bersaglio "worst";
3. chiusura temporanea indipendente dallo stato (temporary_closure_rate_q).
Con i default nessuna delle tre cambia nulla (lo snapshot TRAIN-01 lo verifica a parte)."""
import copy

import pytest

from infralab.core.hazards import sudden_damage as sd
from infralab.settings.config import SimParameters
from tests.conftest import build_env_from_scenario, load_scenario


def _env(overrides=None, seed=11):
    scenario = copy.deepcopy(load_scenario("TRAIN-01"))
    scenario.setdefault("params", {}).update({"seed": seed, "rng_streams": "keyed", **(overrides or {})})
    return build_env_from_scenario(scenario)


def _step(env, n=1):
    agent = env.params.agent_names[0]
    for _ in range(n):
        env.step({agent: {"maintenance": {}}})


# ---------------------------------------------------------------- 1. stima del gestore
def _inspect_all(env):
    op = env.maint_ops["L1_inspect"]
    for b in env.bridges.values():
        env._inspect_bridge(b, op)


def test_default_is_manager_mode():
    assert SimParameters().assessment_epistemic_mode == "manager"
    with pytest.raises(ValueError):
        SimParameters(assessment_epistemic_mode="altro")


def test_reference_mode_removes_the_knowledge_bonus():
    base = {"observation_mode": "inspected", "true_reliability_mode": "physical", "strict_information": True}
    betas = {}
    for mode in ("manager", "reference"):
        env = _env({**base, "assessment_epistemic_mode": mode})
        for _ in range(6):                       # sei ispezioni: l'incertezza del gestore scende
            _inspect_all(env)
        assert max(c.epistemic_uncertainty for b in env.bridges.values() for c in b.components.values()) < 0.15
        betas[mode] = [env._assessed_bridge(b).annual_reliability_index for b in env.bridges.values()]
        true = [b.annual_reliability_index for b in env.bridges.values()]
        betas[mode + "_true"] = true
    # il vero non dipende dalla scelta
    assert betas["manager_true"] == betas["reference_true"]
    # con "manager" la stima e' piu' alta che con "reference" su ogni ponte
    assert all(m > r for m, r in zip(betas["manager"], betas["reference"]))


def test_reference_mode_leaves_manager_uncertainty_untouched():
    env = _env({"observation_mode": "inspected", "true_reliability_mode": "physical",
                "assessment_epistemic_mode": "reference"})
    _inspect_all(env)
    b = next(iter(env.bridges.values()))
    est = env._assessed_bridge(b)
    for name, comp in b.components.items():
        assert est.components[name].epistemic_uncertainty == comp.epistemic_uncertainty


# ---------------------------------------------------------------- 2. cedimento locale
EVENT = {"name": "cedimento_locale", "rate_q": 1.0, "targets": "impalcato,pile,spalle,appoggi",
         "hits": "worst", "drop_dist": "fixed", "drop_mean": 0.10, "announce_prob": 0.0}


def test_new_keys_are_validated():
    SimParameters(sudden_custom_events=[{**EVENT, "condition_max": 0.4, "condition_exponent": 2.0}])
    with pytest.raises(ValueError):
        SimParameters(sudden_custom_events=[{**EVENT, "condition_max": 1.5}])
    with pytest.raises(ValueError):
        SimParameters(sudden_custom_events=[{**EVENT, "condition_exponent": -1.0}])


def test_condition_max_gates_the_event():
    env = _env({"sudden_custom_events": [{**EVENT, "condition_max": 0.0}]})
    _step(env, 3)
    assert sum(st["events_total"] for st in env.sudden_stats.values()) == 0
    env = _env({"sudden_custom_events": [{**EVENT, "condition_max": 1.0}]})
    _step(env, 1)
    assert sum(st["events_total"] for st in env.sudden_stats.values()) == len(env.bridges)


def test_condition_exponent_scales_the_rate():
    env = _env({"sudden_custom_events": [{**EVENT, "rate_q": 0.5, "condition_exponent": 2.0}]})
    spec = sd.active_specs(env.params)[0]
    for b in env.bridges.values():
        worst = min(c.current_condition for n, c in b.components.items() if n in spec["targets"])
        assert sd._custom_rate(env, b, spec) == pytest.approx(0.5 * (1.0 - worst) ** 2)


def test_worst_hits_the_lowest_component_without_drawing():
    env = _env()
    b = next(iter(env.bridges.values()))
    targets = [t for t in ("impalcato", "pile", "spalle", "appoggi") if t in b.components]
    worst = min(targets, key=lambda t: (b.components[t].current_condition, t))

    class NoDraw:
        def __getattr__(self, name):
            raise AssertionError("nessuna estrazione attesa")

    assert sd.pick_hits(b, targets, "worst", NoDraw()) == [worst]
    assert sd.pick_hits(b, targets, "all", NoDraw()) == targets


# ---------------------------------------------------------------- 3. chiusura temporanea
def test_closure_default_off_and_validated():
    assert SimParameters().temporary_closure_rate_q == 0.0
    with pytest.raises(ValueError):
        SimParameters(temporary_closure_rate_q=2.0)
    with pytest.raises(ValueError):
        SimParameters(temporary_closure_mean_duration_quarters=0)
    env = _env()
    _step(env, 4)
    assert not any(getattr(b, "temporary_closure", None) for b in env.bridges.values())


def test_closure_closes_then_restores():
    env = _env({"temporary_closure_rate_q": 1.0, "temporary_closure_mean_duration_quarters": 1})
    _step(env, 1)
    assert all(b.traffic_restriction_factor == 0.0 and b.temporary_closure for b in env.bridges.values())
    env.params.temporary_closure_rate_q = 1e-12          # canale acceso, nessun nuovo ingresso
    _step(env, 2)                                        # durata media 1: geometrica(1) = 1 trimestre
    assert all(b.traffic_restriction_factor == 1.0 and not b.temporary_closure for b in env.bridges.values())


def test_closure_does_not_lift_a_posting_or_a_sanction():
    env = _env({"temporary_closure_rate_q": 1.0, "temporary_closure_mean_duration_quarters": 1})
    bridges = list(env.bridges.values())
    posted, sanctioned = bridges[0], bridges[1]
    posted.posting_until_quarter = env.quarter + 40
    posted.traffic_restriction_factor = env.params.posting_restriction_factor
    sanctioned.regime_sanction = {"action": "limitazione_carico", "since_quarter": 0, "restored_factor": 1.0}
    sanctioned.traffic_restriction_factor = 0.6
    _step(env, 1)
    assert posted.traffic_restriction_factor == 0.0 and sanctioned.traffic_restriction_factor == 0.0
    env.params.temporary_closure_rate_q = 1e-12
    _step(env, 2)
    assert posted.traffic_restriction_factor == pytest.approx(env.params.posting_restriction_factor)
    limit = float((env.maint_ops.get("limitazione_carico") or env._authority_catalogue["limitazione_carico"])
                  ["traffic_restriction_factor"])
    assert sanctioned.traffic_restriction_factor == pytest.approx(limit)
    assert sanctioned.regime_sanction["restored_factor"] == 1.0


def test_closure_survives_clone_and_reload():
    env = _env({"temporary_closure_rate_q": 1.0, "temporary_closure_mean_duration_quarters": 8})
    _step(env, 1)
    b = next(iter(env.bridges.values()))
    from infralab.core.environment import Bridge
    copy_b = Bridge.from_dict(b.to_dict(), env.params.deterioration_exponent)
    assert copy_b.temporary_closure == b.temporary_closure
    clone = env.planning_clone()
    assert clone.bridges[b.id].temporary_closure == b.temporary_closure
