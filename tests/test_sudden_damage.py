"""Canale di danno locale improvviso (hazards/sudden_damage.py; tutto opt-in)."""
import copy
import json

import numpy as np
import pytest

from infralab.agents.basic import BasicAgent
from infralab.core.environment import BridgeEnv
from infralab.core.hazards import sudden_damage as sd
from infralab.lab.build import build_env_from_scenario
from infralab.lab.runner import expand_runs, run_one
from infralab.lab.spec import spec_from_dict
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory

HIDDEN_IMPACT = {"impact_hazard_enabled": True, "impact_damage_rate_q": 0.5, "impact_adt_exponent": 0.0,
                 "impact_truck_weight": 0.0, "impact_damage_announce_prob": 0.0,
                 "impact_damage_drop_dist": "fixed", "impact_damage_drop_mean": 0.2}


def _env(overrides=None, seed=7, scenario=None):
    return build_env_from_scenario(scenario or load_scenario("TRAIN-01"),
                                   param_overrides=overrides or {}, seed=seed)


def _step(env, n=1, plan=None):
    agent = env.params.agent_names[0]
    for _ in range(n):
        if plan is None:
            p, _ = BasicAgent(agent, "do_nothing").decide(env)
        else:
            p = plan
        env.step({agent: p})


def _events(env, kind):
    return sorted((bid, e["quarter"], e["component"], e["loss"])
                  for bid, evs in env.sudden_pending.items() for e in evs if e["type"] == kind)


# -- default invariati ---------------------------------------------------------

def test_defaults_are_neutral():
    p = SimParameters()
    assert not sd.channel_active(p)
    assert p.hydraulic_legacy_mode is True
    assert not (p.impact_hazard_enabled or p.vessel_impact_enabled or p.fatigue_hazard_enabled)
    assert p.sudden_custom_events == []
    env = _env()
    _step(env, 3)
    assert env.sudden_stats == {} and env.sudden_pending == {}
    data = env.to_dict()
    assert "sudden_stats" not in data and "sudden_pending" not in data
    assert sd.metrics(env, env.params.agent_names[0]) == {}


def test_defaults_keep_train01_golden_snapshot():
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


@pytest.mark.parametrize("bad", [
    {"impact_damage_drop_dist": "normal"}, {"fatigue_damage_hits": "some"},
    {"impact_damage_announce_prob": 1.5}, {"impact_damage_rate_q": -1.0},
    {"vessel_damage_targets": ""}, {"impact_hazard_enabled": "si"},
    {"sudden_custom_events": [{"name": "frana"}]},                       # manca rate_q
    {"sudden_custom_events": [{"name": "impact", "rate_q": 0.1}]},       # nome riservato
    {"sudden_custom_events": [{"name": "x", "rate_q": 0.1, "colore": 1}]},
    {"sudden_custom_events": {"name": "x"}},
])
def test_validation(bad):
    with pytest.raises(ValueError):
        SimParameters(**bad)


def test_hydraulic_legacy_mode_is_the_historical_channel():
    ov = {"hydraulic_hazard_enabled": True, "hydraulic_collapse_rate_water_q": 0.05,
          "hydraulic_damage_rate_multiplier": 10.0}
    scen = copy.deepcopy(load_scenario("TRAIN-01"))
    for b in scen["bridges"]:
        b["crosses_water"] = True
    env = _env(ov, scenario=scen)
    _step(env, 6)
    assert env.sudden_stats == {} and env.sudden_pending == {}
    seen = [e for b in env.bridges.values() for e in b.action_history if e["action"] == "danno_idraulico"]
    assert seen, "il canale storico scrive l'evento nello storico azioni"


# -- numeri casuali comuni -----------------------------------------------------

@pytest.mark.parametrize("streams", ["shared", "keyed"])
def test_channel_does_not_shift_other_streams(streams):
    off = _env({"rng_streams": streams})
    on = _env({"rng_streams": streams, **HIDDEN_IMPACT, "fatigue_hazard_enabled": True,
               "fatigue_damage_rate_q": 0.5})
    _step(off, 6)
    _step(on, 6)
    assert on.sudden_stats[on.params.agent_names[0]]["events_total"] > 0
    # l'RNG condiviso del motore ha consumato esattamente le stesse estrazioni
    assert off.rng.bit_generator.state == on.rng.bit_generator.state


def test_types_are_independent_streams():
    only_impact = _env(HIDDEN_IMPACT)
    both = _env({**HIDDEN_IMPACT, "fatigue_hazard_enabled": True, "fatigue_damage_rate_q": 0.5,
                 "fatigue_damage_targets": "appoggi"})
    _step(only_impact, 8)
    _step(both, 8)
    assert _events(only_impact, "impact") and _events(both, "fatigue")
    # stessi urti (quando, dove) con o senza il secondo tipo: le perdite possono
    # differire solo perche' la condizione di partenza e' diversa
    assert [e[:3] for e in _events(only_impact, "impact")] == [e[:3] for e in _events(both, "impact")]


def test_same_events_across_policies():
    """Due strategie sullo stesso seme affrontano gli stessi eventi."""
    a = _env(HIDDEN_IMPACT)
    b = _env(HIDDEN_IMPACT)
    agent = a.params.agent_names[0]
    _step(a, 6)
    _step(b, 6, plan={"maintenance": {"1": "L1_inspect", "2": "L1_inspect"}})
    ev_a = sorted((bid, e["quarter"]) for bid, evs in a.sudden_pending.items() for e in evs)
    st_b = b.sudden_stats[agent]
    assert st_b["events_total"] == a.sudden_stats[agent]["events_total"]
    assert len(ev_a) == a.sudden_stats[agent]["events_total"]
    assert st_b["detected_by_inspection"] > 0


# -- visibilita' ---------------------------------------------------------------

def test_hidden_until_inspection():
    env = _env({**HIDDEN_IMPACT, "observation_mode": "inspected", "initial_inventory_known": True})
    agent = env.params.agent_names[0]
    before = {bid: {n: (c.observed_condition_min, c.observed_condition_max) for n, c in b.components.items()}
              for bid, b in env.bridges.items()}
    obs_before = {b["id"]: b["min_condition"] for b in env.get_agent_observations(agent)["my_bridges"]}
    _step(env, 1)
    assert env.sudden_pending, "con tasso 0.5 su 2 ponti serve almeno un evento (seme fisso)"
    for bid, evs in env.sudden_pending.items():
        b = env.bridges[bid]
        for e in evs:
            assert (b.components[e["component"]].observed_condition_min,
                    b.components[e["component"]].observed_condition_max) == before[bid][e["component"]]
        assert not any(h["action"] == "danno_improvviso" for h in b.action_history)
    # la vista del gestore non si muove per il danno (la condizione nominale e' quella
    # dell'ultima ispezione): stessa stima prima e dopo
    obs_after = {b["id"]: b["min_condition"] for b in env.get_agent_observations(agent)["my_bridges"]}
    assert obs_after == obs_before
    # ispezione: il danno viene rilevato e diventa noto
    damaged = sorted(env.sudden_pending)
    plan = {"maintenance": {str(bid): "L1_inspect" for bid in damaged}}
    pending_before = sum(len(v) for v in env.sudden_pending.values())
    env.params.impact_damage_rate_q = 0.0                  # nessun nuovo evento
    _step(env, 1, plan=plan)
    st = env.sudden_stats[agent]
    assert st["detected_by_inspection"] == pending_before
    assert env.sudden_pending == {}
    for bid in damaged:
        b = env.bridges[bid]
        assert any(h["action"] == "danno_improvviso" and h["status"] == "detected" for h in b.action_history)
        assert b.min_observed_condition() < min(v[0] for v in before[bid].values())


def test_self_announcing_fraction_is_known_immediately():
    env = _env({**HIDDEN_IMPACT, "impact_damage_announce_prob": 1.0, "observation_mode": "inspected",
                "initial_inventory_known": True})
    agent = env.params.agent_names[0]
    _step(env, 1)
    st = env.sudden_stats[agent]
    assert st["events_total"] > 0
    assert env.sudden_pending == {}
    assert st["announced"] == st["events_total"]
    for b in env.bridges.values():
        hits = [h for h in b.action_history if h["action"] == "danno_improvviso"]
        for h in hits:
            assert h["status"] == "announced"
            comp = b.components[h["components"][0]]
            # osservazione fresca attorno alla condizione vera (rumore L1)
            assert comp.observed_condition_min <= comp.current_condition + 1e-9 or \
                abs(comp.observed_condition_min - comp.current_condition) < 0.3


def test_announce_fraction_is_statistical():
    env = _env({**HIDDEN_IMPACT, "impact_damage_rate_q": 1.0, "impact_damage_announce_prob": 0.3,
                "impact_damage_drop_mean": 0.002})
    agent = env.params.agent_names[0]
    _step(env, 30)
    st = env.sudden_stats[agent]
    assert st["events_total"] == 60
    assert 0.1 < st["announced"] / st["events_total"] < 0.55


def test_repair_and_failure_bookkeeping():
    env = _env({**HIDDEN_IMPACT, "impact_damage_targets": "pile"})
    agent = env.params.agent_names[0]
    _step(env, 2)
    assert env.sudden_pending
    bid = sorted(env.sudden_pending)[0]
    b = env.bridges[bid]
    n = len(env.sudden_pending[bid])
    sd.on_repair(env, b, ["pile"])
    assert bid not in env.sudden_pending
    assert env.sudden_stats[agent]["repaired_before_detection"] == n
    env.sudden_pending[bid] = [{"type": "impact", "quarter": 1, "component": "pile", "loss": 0.1}]
    sd.on_failure(env, b)
    assert env.sudden_stats[agent]["undetected_at_failure"] == 1
    assert env.sudden_stats[agent]["failures_with_undetected_damage"] == 1


def test_monitoring_detection():
    env = _env({**HIDDEN_IMPACT, "sudden_damage_monitoring_detect_prob": 1.0})
    agent = env.params.agent_names[0]
    _step(env, 1)
    assert env.sudden_pending
    for b in env.bridges.values():
        b.monitoring_ends_quarter = env.quarter + 8
    env.params.impact_damage_rate_q = 0.0
    _step(env, 1)
    assert env.sudden_pending == {}
    assert env.sudden_stats[agent]["detected_by_monitoring"] > 0


# -- esposizione ---------------------------------------------------------------

def test_exposure_rules():
    env = _env({"impact_hazard_enabled": True, "vessel_impact_enabled": True, "fatigue_hazard_enabled": True,
                "impact_road_under_multiplier": 3.0, "impact_truck_weight": 2.0, "fatigue_truck_weight": 2.0,
                "sudden_custom_events": [{"name": "frana", "rate_q": 0.01, "requires": "crosses_water",
                                          "multipliers": {"scour_critical": 4.0}, "adt_exponent": 1.0}]})
    p = env.params
    specs = {s["name"]: s for s in sd.active_specs(p)}
    b = env.bridges[1]
    ref = float(p.traffic_reference_adt)
    b.truck_fraction = 0.25
    base = p.impact_damage_rate_q * (b.adt / ref) * (1 + 2.0 * 0.25)
    assert sd._impact_rate(env, b, specs["impact"]) == pytest.approx(base)
    b.service_under = "1"                       # strada sotto (item 042B)
    assert sd._impact_rate(env, b, specs["impact"]) == pytest.approx(3.0 * base)
    b.service_under = "5"                       # corso d'acqua sotto
    assert sd._impact_rate(env, b, specs["impact"]) == pytest.approx(base)
    # natante: solo su via navigabile
    assert sd._vessel_rate(env, b, specs["vessel"]) == 0.0
    b.navigable_waterway = True
    assert sd._vessel_rate(env, b, specs["vessel"]) == p.vessel_impact_rate_q
    assert sd._fatigue_rate(env, b, specs["fatigue"]) == pytest.approx(p.fatigue_damage_rate_q * 1.5)
    # tipo utente: requisito e moltiplicatore
    assert sd._custom_rate(env, b, specs["frana"]) == 0.0
    b.crosses_water = True
    r = sd._custom_rate(env, b, specs["frana"])
    assert r == pytest.approx(0.01 * b.adt / ref)
    b.scour_critical = True
    assert sd._custom_rate(env, b, specs["frana"]) == pytest.approx(4.0 * r)


def test_hydraulic_new_mode_only_water_bridges_hidden_and_keyed():
    ov = {"hydraulic_hazard_enabled": True, "hydraulic_legacy_mode": False,
          "hydraulic_collapse_rate_water_q": 0.0, "hydraulic_collapse_rate_scour_q": 0.05,
          "hydraulic_damage_rate_multiplier": 10.0}
    scen = copy.deepcopy(load_scenario("TRAIN-01"))
    scen["bridges"][0]["crosses_water"] = True
    scen["bridges"][0]["scour_critical"] = True
    off = _env({"rng_streams": "shared"}, scenario=scen)
    env = _env({"rng_streams": "shared", **ov}, scenario=scen)
    _step(off, 8)
    _step(env, 8)
    agent = env.params.agent_names[0]
    st = env.sudden_stats[agent]
    assert st["events_hydraulic"] > 0
    assert set(env.sudden_pending) == {scen["bridges"][0]["id"]}
    assert {e["component"] for evs in env.sudden_pending.values() for e in evs} <= {"pile", "spalle"}
    assert not any(h["action"] == "danno_idraulico" for b in env.bridges.values() for h in b.action_history)
    assert off.rng.bit_generator.state == env.rng.bit_generator.state


def test_drop_distributions():
    rng = np.random.default_rng(0)
    for dist in ("uniform", "lognormal"):
        spec = {"drop_dist": dist, "drop_mean": 0.2, "drop_cov": 0.3}
        xs = np.array([sd.draw_drop(spec, rng) for _ in range(4000)])
        assert xs.mean() == pytest.approx(0.2, rel=0.05)
        assert xs.std() / xs.mean() == pytest.approx(0.3, rel=0.12)
    assert sd.draw_drop({"drop_dist": "fixed", "drop_mean": 0.2, "drop_cov": 0.3}, rng) == 0.2


def test_damage_does_not_touch_manager_uncertainty():
    env = _env(HIDDEN_IMPACT)
    eps = {(bid, n): c.epistemic_uncertainty for bid, b in env.bridges.items() for n, c in b.components.items()}
    spec = sd.active_specs(env.params)[0]
    b = env.bridges[1]
    sd.apply_event(env, b, spec, np.random.default_rng(1))
    for n, c in b.components.items():
        assert c.epistemic_uncertainty == eps[(1, n)]


# -- stato, copie, uscite ------------------------------------------------------

def test_state_roundtrip_and_belief_clone_hides_pending():
    env = _env({**HIDDEN_IMPACT, "observation_mode": "inspected"})
    _step(env, 2)
    assert env.sudden_pending
    back = BridgeEnv.from_dict(json.loads(json.dumps(env.to_dict())))
    assert back.sudden_pending == env.sudden_pending
    assert back.sudden_stats == env.sudden_stats
    clone = env.belief_clone(rng=np.random.default_rng(3))
    assert clone.sudden_pending == {}


def test_lab_outputs_carry_counts():
    raw = {"name": "sudden", "scenario": "TRAIN-01", "horizon_quarters": 4,
           "agents": [{"id": "dn", "type": "Basic", "policy": "do_nothing"}],
           "grid": {"params.impact_hazard_enabled": [True], "params.impact_damage_rate_q": [0.5],
                    "params.impact_adt_exponent": [0.0],
                    "params.sudden_custom_events": [[{"name": "frana", "rate_q": 0.2, "targets": "spalle"}]]},
           "replicates": 1, "seed": 1, "outputs": {"dir": "results/sudden", "formats": ["csv"]}}
    payload = expand_runs(spec_from_dict(raw), load_scenario("TRAIN-01"))[0]
    run_row, steps = run_one(payload)
    assert run_row["final_sudden_events_total"] > 0
    assert "final_sudden_events_frana" in run_row and "final_sudden_undetected_at_failure" in run_row
    assert sum(s["metric_sudden_events_quarter"] for s in steps) == run_row["final_sudden_events_total"]


def test_backend_config_exposes_all_sudden_fields():
    from dataclasses import fields
    from backend.app import CONFIG_FIELDS
    keys = {f["key"] for f in CONFIG_FIELDS}
    names = {f.name for f in fields(SimParameters)}
    sudden = {n for n in names if n.startswith(("impact_", "vessel_", "fatigue_", "hydraulic_", "sudden_"))}
    assert sudden <= keys


# -- la cadenza ispettiva ora conta ---------------------------------------------

def _interval_run(quarters_between, channel_on, seed, horizon=40):
    from infralab.regimes.base import Enforcement, Obligation, RegulatoryRegime
    from infralab.regimes.registry import register_regime
    rid = f"test_inspection_only_{quarters_between}q"
    classes = ("bassa", "medio_bassa", "media", "medio_alta", "alta")
    register_regime(RegulatoryRegime(
        id=rid, obligations=[Obligation(trigger="inspection_overdue", action="L1_inspect")],
        enforcement=Enforcement(mode="forced"),
        cda_config={f"{c}_due_quarters": quarters_between for c in classes}), overwrite=True)
    ov = {"observation_mode": "inspected", "failure_model": "reliability", "true_reliability_mode": "physical",
          "ethical_floor_enabled": True, "ethical_floor_condition": 0.4, "initial_inventory_known": True,
          "rng_streams": "keyed", "regulatory_regime": rid, "inspection_overdue_grace_quarters": 0,
          "initial_budget": 1e9}
    if channel_on:
        ov.update({"impact_hazard_enabled": True, "impact_damage_rate_q": 0.05, "impact_damage_announce_prob": 0.0,
                   "impact_damage_drop_mean": 0.3, "impact_adt_exponent": 0.0})
    env = _env(ov, seed=seed)
    agent = env.params.agent_names[0]
    loss = 0.0
    for _ in range(horizon):
        _step(env, 1)
        loss += env.last_expected_loss[agent]["total"]
    return loss, env


def test_inspection_interval_matters_only_with_sudden_damage():
    """do_nothing + soglia etica, obbligo di sola ispezione ogni 1 o 10 anni.
    Senza danno improvviso la cadenza e' irrilevante; con il canale acceso l'ispezione
    annuale scopre i danni e la soglia li ripara, quella decennale no."""
    off_1y, _ = _interval_run(4, False, seed=1)
    off_10y, _ = _interval_run(40, False, seed=1)
    on_1y, env_1y = _interval_run(4, True, seed=1)
    on_10y, env_10y = _interval_run(40, True, seed=1)
    agent = env_1y.params.agent_names[0]
    assert abs(off_10y - off_1y) < 1e4                       # euro: trascurabile
    assert on_10y - on_1y > 1e5
    assert (env_1y.sudden_stats[agent]["detected_by_inspection"]
            > env_10y.sudden_stats[agent]["detected_by_inspection"])
    assert env_1y.sudden_stats[agent]["events_total"] == env_10y.sudden_stats[agent]["events_total"]
