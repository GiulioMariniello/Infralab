"""Tre aggiunte opt-in per la taratura sull'inventario reale (docs/CHANGELOG_MODEL.md):
1. policy elementare "typical_owner": probabilita' annua di intervento per voto osservato;
2. chiusura temporanea legata allo stato (temporary_closure_condition_rate_q);
3. declassamento di portata: pavimento della severita' configurabile
   (posting_severity_floor) e revoca a opera dei lavori (posting_lift_on_intervention).
Con i default nessuna delle tre cambia nulla (lo snapshot TRAIN-01 lo verifica a parte)."""
import copy

import pytest

from infralab.agents.basic import BASIC_POLICIES, TYPICAL_OWNER_RATES, BasicAgent
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


# ---------------------------------------------------------------- 1. gestore tipico
def _plan(env, params):
    name = env.params.agent_names[0]
    return BasicAgent(name, "typical_owner", params).decide(env)[0]["maintenance"]


def test_typical_owner_is_registered_with_the_inventory_table():
    d = BASIC_POLICIES["typical_owner"]["defaults"]
    assert d["annual_rate_by_rating"] == TYPICAL_OWNER_RATES and d["rate_scale"] == 1.0
    assert TYPICAL_OWNER_RATES[2] > TYPICAL_OWNER_RATES[5] > TYPICAL_OWNER_RATES[8] > TYPICAL_OWNER_RATES[9] == 0.0


def test_typical_owner_zero_rate_does_nothing_and_full_rate_acts_on_everything():
    env = _env({"initial_inventory_known": True})
    name = env.params.agent_names[0]
    env.budgets[name] = 1e12
    always = {str(r): 1.0 for r in range(10)}
    assert _plan(env, {"rate_scale": 0.0}) == {}
    plan = _plan(env, {"annual_rate_by_rating": always})
    assert len(plan) == len(env.bridges)
    ops = env.maint_ops
    assert all(ops[a]["type"] == "intervention" for a in plan.values())


def test_typical_owner_is_repeatable_and_seed_dependent():
    env = _env({"initial_inventory_known": True})
    env.budgets[env.params.agent_names[0]] = 1e12
    half = {"annual_rate_by_rating": {str(r): 0.9 for r in range(10)}}
    a, b = _plan(env, {**half, "seed": 1}), _plan(env, {**half, "seed": 1})
    assert a == b and 0 < len(a) < len(env.bridges)
    plans = {tuple(sorted(_plan(env, {**half, "seed": s}))) for s in range(8)}
    assert len(plans) > 1


def test_typical_owner_minor_work_replaces_the_worst_observed_component():
    env = _env({"initial_inventory_known": True})
    env.budgets[env.params.agent_names[0]] = 1e12
    for b in env.bridges.values():                      # appoggi: il componente osservato peggiore
        b.components["appoggi"].observed_condition_min = 0.20
        b.components["appoggi"].observed_condition_max = 0.30
    always = {"annual_rate_by_rating": {str(r): 1.0 for r in range(10)}}
    plan = _plan(env, {**always, "major_share": 0.0})
    assert len(plan) == len(env.bridges) and set(plan.values()) == {"replace_appoggi"}
    # quota di interventi maggiori = 1: sempre quello col maggior guadagno totale
    plan = _plan(env, {**always, "major_share": 1.0})
    assert set(plan.values()) == {"ripristino_completo"}


# ---------------------------------------------------------------- 2. chiusura legata allo stato
def test_condition_closure_default_off_and_validated():
    assert SimParameters().temporary_closure_condition_rate_q == 0.0
    with pytest.raises(ValueError):
        SimParameters(temporary_closure_condition_rate_q=0.9)


def test_condition_closure_hits_only_degraded_bridges():
    env = _env({"temporary_closure_condition_rate_q": 0.5, "temporary_closure_mean_duration_quarters": 8})
    bridges = list(env.bridges.values())
    good, bad = bridges[0], bridges[1]
    for c in good.components.values():
        c.current_condition = 1.0
    for c in bad.components.values():
        c.current_condition = 1e-6
    # probabilita' = 0.5 x 2 x (1 - c): ~1 sul ponte a zero, ~0 su quello nuovo
    env._sample_temporary_closures()
    assert bad.temporary_closure and bad.traffic_restriction_factor == 0.0
    assert not getattr(good, "temporary_closure", None)


def test_flat_closure_draws_are_unchanged_by_the_new_option():
    """Con la parte legata allo stato spenta gli ingressi sono quelli del solo tasso piatto."""
    out = []
    for extra in ({}, {"temporary_closure_condition_rate_q": 0.0}):
        env = _env({"temporary_closure_rate_q": 0.3, **extra})
        _step(env, 3)
        out.append({b.id: copy.deepcopy(b.temporary_closure) for b in env.bridges.values()})
    assert out[0] == out[1] and any(out[0].values())


# ---------------------------------------------------------------- 3. declassamento
def test_posting_options_default_to_historic_behaviour():
    p = SimParameters()
    assert p.posting_severity_floor == 0.25 and p.posting_lift_on_intervention is False
    with pytest.raises(ValueError):
        SimParameters(posting_severity_floor=-0.1)


def test_severity_floor_zero_spares_a_bridge_in_perfect_condition():
    for floor, expected in ((0.25, True), (0.0, False)):
        env = _env({"posting_entry_rate_q": 1.0, "posting_severity_floor": floor})
        b = next(iter(env.bridges.values()))
        for c in b.components.values():
            c.current_condition = 1.0
        # tasso 1 x severita': col pavimento storico 0.25 l'ingresso ha probabilita' 0.25
        hits = 0
        for q in range(1, 41):
            env.quarter = q
            b.posting_until_quarter = 0
            b.traffic_restriction_factor = 1.0
            for c in b.components.values():
                c.current_condition = 1.0
            others = [x for x in env.bridges.values() if x is not b]
            env._sample_posting_events()
            hits += int(b.posting_until_quarter > 0)
            for x in others:
                x.posting_until_quarter, x.traffic_restriction_factor = 0, 1.0
        assert (hits > 0) is expected


def test_structural_works_lift_the_posting_only_when_enabled():
    for lift in (False, True):
        env = _env({"posting_entry_rate_q": 1e-12, "posting_lift_on_intervention": lift})
        b = next(iter(env.bridges.values()))
        b.posting_until_quarter = env.quarter + 40
        b.traffic_restriction_factor = env.params.posting_restriction_factor
        b.active_maintenance = {"action_name": "ripristino_completo", "ends_at_quarter": env.quarter + 3}
        env._sample_posting_events()
        if lift:
            assert b.posting_until_quarter == 0 and b.traffic_restriction_factor == 1.0
        else:
            assert b.posting_until_quarter > 0
            assert b.traffic_restriction_factor == pytest.approx(env.params.posting_restriction_factor)


def test_works_do_not_reopen_a_sanctioned_bridge_and_block_new_postings():
    env = _env({"posting_entry_rate_q": 1.0, "posting_lift_on_intervention": True})
    b = next(iter(env.bridges.values()))
    for c in b.components.values():
        c.current_condition = 1e-6                       # severita' massima: ingresso certo senza lavori
    b.regime_sanction = {"action": "limitazione_carico", "since_quarter": 0, "restored_factor": 1.0}
    b.posting_until_quarter = env.quarter + 40
    b.traffic_restriction_factor = 0.6
    b.active_maintenance = {"action_name": "ripristino_completo", "ends_at_quarter": env.quarter + 3}
    env._sample_posting_events()
    assert b.posting_until_quarter == 0 and b.traffic_restriction_factor == 0.6
    env._sample_posting_events()
    assert b.posting_until_quarter == 0                  # durante i lavori non nasce un nuovo declassamento
    # la manutenzione preventiva non revoca nulla
    b.posting_until_quarter = env.quarter + 40
    b.active_maintenance = {"action_name": "manutenzione_ordinaria", "ends_at_quarter": env.quarter + 1}
    env._sample_posting_events()
    assert b.posting_until_quarter > 0
