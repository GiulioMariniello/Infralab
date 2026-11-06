"""Eventi progressivi (hazards/progressive_events.py), registro dei collassi e regola
di fine partita (tutto opt-in)."""
import copy
import json
from dataclasses import fields

import pytest

from infralab.agents.sandbox import belief_snapshot, sanitized_observation
from infralab.core.environment import BridgeEnv, _calculate_cost
from infralab.core.hazards import progressive_events as pe
from infralab.core.hazards import sudden_damage as sd
from infralab.lab.build import build_env_from_scenario
from infralab.lab.io import write_outputs
from infralab.lab.leak_check import perturb_hidden_truth, run_leak_check
from infralab.lab.runner import COLLAPSE_COLUMNS, expand_runs, run_experiment, run_one, run_one_full
from infralab.lab.spec import spec_from_dict
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import N_QUARTERS, SNAPSHOT_PATH, canonical

ON = {"progressive_events_enabled": True, "failure_model": "reliability"}
# Tipo utente senza requisiti di esposizione: un innesco per ponte per trimestre.
EVERY_QUARTER = {"name": "prova", "rate_y": 4.0, "duration_dist": "fixed", "duration_mean_q": 3,
                 "p_collapse_if_untreated": 0.0, "drop_mean": 0.2}


def _scenario(n=2, water=True, scour=(), slope=None):
    sc = copy.deepcopy(load_scenario("TRAIN-01"))
    base = sc["bridges"][0]
    sc.pop("allowed_actions", None)
    sc["bridges"] = [{**base, "id": i + 1, "name": f"P{i + 1}", "crosses_water": water,
                      "scour_critical": (i + 1) in scour,
                      **({"on_unstable_slope": slope} if slope is not None else {})} for i in range(n)]
    return sc


def _env(overrides=None, seed=7, n=2, **kw):
    return build_env_from_scenario(_scenario(n, **kw), param_overrides={**ON, **(overrides or {})}, seed=seed)


def _agent(env):
    return env.params.agent_names[0]


def _step(env, plan=None, n=1):
    for _ in range(n):
        env.step({_agent(env): {"maintenance": dict(plan or {})}})


def _inject(env, bid=1, typ="hydraulic", ends_in=3, known=False):
    """Difetto in sviluppo sul ponte: matura nel passo del trimestre corrente + ends_in."""
    q = env.quarter
    d = {"type": typ, "start_quarter": q - 1, "end_quarter": q + ends_in,
         "known_since": q - 1 if known else None, "source": "initiation"}
    env.progressive_defects.setdefault(bid, []).append(d)
    return d


def _stats(env):
    return env.progressive_stats[_agent(env)]


# -- default invariati ---------------------------------------------------------

def test_defaults_are_neutral():
    p = SimParameters()
    assert p.progressive_events_enabled is False and p.hydraulic_collapse_progressive_share == 0.0
    assert p.collapse_game_over == "none" and not pe.enabled(p) and pe.effective_share(p) == 0.0
    env = build_env_from_scenario(load_scenario("TRAIN-01"), seed=7)
    _step(env, n=3)
    assert env.progressive_defects == {} and env.progressive_stats == {} and env.collapse_log == []
    assert pe.ARREST_ACTION not in env.maint_ops
    data = env.to_dict()
    assert not {"progressive_defects", "progressive_stats", "collapse_log"} & set(data)
    agent = _agent(env)
    assert pe.metrics(env, agent) == {} and pe.collapse_metrics(env, agent) == {}
    obs = env.get_agent_observations(agent)
    assert all("known_progressive_defects" not in b for b in obs["my_bridges"])


def _train01_trajectory(overrides):
    env = build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides=overrides, seed=None)
    agent = _agent(env)
    traj = {"quarters": []}
    for _ in range(N_QUARTERS):
        env.step({agent: {"maintenance": {}}})
        traj["quarters"].append({
            "quarter": env.quarter, "budget": env.budgets[agent],
            "bridges": {str(bid): {"min_condition": b.min_condition, "reliability_index": b.reliability_index,
                                   "annual_failure_probability": b.annual_failure_probability}
                        for bid, b in sorted(env.bridges.items())}})
    return env, traj


def test_channel_on_without_consequences_keeps_train01_golden_snapshot():
    """Canale acceso, difetti che nascono ogni trimestre ma senza conseguenza (finestra
    oltre l'orizzonte): traiettoria identica allo snapshot dorato. I flussi dedicati non
    spostano l'RNG condiviso ne' gli altri flussi."""
    env, traj = _train01_trajectory({"progressive_events_enabled": True,
                                     "progressive_custom_events": [{**EVERY_QUARTER, "duration_mean_q": 400}]})
    assert _stats(env)["initiated"] == 2 * N_QUARTERS
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        assert canonical(traj) == canonical(json.load(f))


def test_default_run_adds_no_output_columns():
    raw = {"name": "d", "scenario": "TRAIN-01", "horizon_quarters": 3,
           "agents": [{"id": "dn", "type": "Basic", "params": {"policy": "do_nothing"}}],
           "replicates": 1, "seed": 1, "outputs": {"dir": "results/d", "formats": ["csv"]}}
    run_row, steps, collapses = run_one_full(expand_runs(spec_from_dict(raw), load_scenario("TRAIN-01"))[0])
    new = [k for k in list(run_row) + list(steps[0]) if "progressive" in k or "failures_cause" in k
           or "known_critical" in k or "failures_hydraulic" in k]
    assert new == [] and isinstance(collapses, list)


@pytest.mark.parametrize("bad", [
    {"hydraulic_collapse_progressive_share": 1.5},
    {"progressive_hydraulic_rate_mode": "boh"},
    {"progressive_hydraulic_duration_dist": "weibull"},
    {"progressive_hydraulic_duration_mean_q": 0.5},
    {"progressive_geotechnical_p_collapse_if_untreated": 2.0},
    {"progressive_geotechnical_targets": ""},
    {"progressive_arrest_duration_quarters": 1.5},
    {"collapse_game_over": "sempre"},
    {"progressive_detection_pod_mode": "x"},
    {"progressive_custom_events": [{"name": "hydraulic", "rate_y": 0.1}]},
    {"progressive_custom_events": [{"name": "a", "rate_y": 0.1, "boh": 1}]},
    {"progressive_custom_events": [{"name": "a"}]},
    {"progressive_events_enabled": True, "hydraulic_collapse_progressive_share": 0.5,
     "progressive_hydraulic_p_collapse_if_untreated": 0.0},
])
def test_validation(bad):
    with pytest.raises(ValueError):
        SimParameters(**bad)


# -- numeri casuali comuni --------------------------------------------------------

def test_crn_same_initiations_durations_and_outcomes_across_policies():
    """Stesso seme, strategie diverse (nessuna / ispezione continua): stessi inneschi,
    stesse durate, stessi esiti. Le ispezioni cambiano solo cio' che e' noto."""
    params = {"progressive_custom_events": [{**EVERY_QUARTER, "rate_y": 1.2, "duration_dist": "geometric",
                                             "p_collapse_if_untreated": 0.3, "detect_factor_l1": 0.5}]}
    logs = []
    for plan in ({}, {"1": "L1_inspect", "2": "L1_inspect", "3": "L1_inspect"}):
        env = _env(params, n=3)
        seen = set()
        for _ in range(14):
            _step(env, plan)
            seen |= {(bid, d["type"], d["start_quarter"], d["end_quarter"])
                     for bid, defs in env.progressive_defects.items() for d in defs}
        st = _stats(env)
        logs.append((sorted(seen), st["initiated"], st["matured"], st["collapses"],
                     [(r["quarter"], r["bridge_id"]) for r in env.collapse_log if r["cause"] == "progressive_prova"]))
    assert logs[0] == logs[1] and logs[0][1] > 3 and logs[0][2] > 0


# -- stato latente e rilevazione ----------------------------------------------------

def test_defect_hidden_until_detected_and_factor_applied():
    params = {"observation_mode": "inspected", "policy_sandbox": True, "strict_information": True,
              "progressive_hydraulic_detect_factor_l1": 0.0, "progressive_hydraulic_detect_factor_detailed": 1.0}
    env = _env(params)
    _inject(env, 1, ends_in=40)
    _inject(env, 2, ends_in=40)
    agent = _agent(env)
    for obs in (env.get_agent_observations(agent), sanitized_observation(env, agent)):
        assert all(b["known_progressive_defects"] == [] for b in obs["my_bridges"])
        assert "end_quarter" not in json.dumps(obs)
    assert not env.get_feasible_actions_for_bridge(agent, env.bridges[1])[pe.ARREST_ACTION]["feasible"]
    assert env.get_feasible_actions_for_bridge(agent, env.bridges[1])[pe.ARREST_ACTION]["reason"] == "no_known_defect"
    # ispezione visiva ordinaria con fattore 0: non vede; ispezione di dettaglio (L2): vede
    q = env.quarter
    _step(env, {"1": "L1_inspect", "2": "L2_inspect"})
    assert pe.known_defects(env, env.bridges[1]) == []
    assert pe.public_view(env, env.bridges[2]) == [{"type": "hydraulic", "known_since": q}]
    assert _stats(env)["inspections_missed"] == 1 and _stats(env)["detected_by_inspection"] == 1
    b2 = next(b for b in sanitized_observation(env, agent)["my_bridges"] if b["id"] == 2)
    assert b2["known_progressive_defects"] == [{"type": "hydraulic", "known_since": q}]
    assert b2["action_feasibility"][pe.ARREST_ACTION]["reason"] in (None, "insufficient_budget")


def test_detection_probability_scales_with_type_factor():
    n = 40
    found = {}
    for factor in (0.5, 1.0):
        env = _env({"progressive_hydraulic_detect_factor_l1": factor, "reliability_refresh_every_quarters": 40}, n=n)
        for bid in env.bridges:
            _inject(env, bid, ends_in=40)
        _step(env, {str(bid): "L1_inspect" for bid in env.bridges})
        found[factor] = sum(1 for b in env.bridges.values() if pe.known_defects(env, b))
    assert found[1.0] == n and 10 <= found[0.5] <= 30


def test_inspection_level_mapping_and_pod_mode():
    p = SimParameters()
    assert pe.is_detailed(p, {"inspection_level": 2}) and pe.is_detailed(p, {"inspection_level": 4})
    assert pe.is_detailed(p, {"type": "monitoring"})
    assert not pe.is_detailed(p, {"inspection_level": 1}) and not pe.is_detailed(p, {"type": "analysis"})
    assert not pe.is_detailed(p, {})                    # rilievo post-intervento
    # modello POD acceso su componenti sani: l'ispezione "non rileva" i bersagli, quindi
    # non vede il difetto ("inspection"); "independent" ignora la POD del motore
    seen = {}
    for mode in pe.POD_MODES:
        env = _env({"inspection_pod_enabled": True, "inspection_pod_exponent": 1e-9,
                    "progressive_hydraulic_detect_factor_l1": 1.0, "progressive_detection_pod_mode": mode})
        _inject(env, 1, ends_in=40)
        _step(env, {"1": "L1_inspect"})
        seen[mode] = bool(pe.known_defects(env, env.bridges[1]))
    assert seen == {"inspection": False, "independent": True}


def test_self_evident_fraction_known_at_once():
    env = _env({"progressive_custom_events": [{**EVERY_QUARTER, "self_evident_prob": 1.0, "duration_mean_q": 40}]})
    q = env.quarter
    _step(env)
    for b in env.bridges.values():
        assert pe.public_view(env, b) == [{"type": "prova", "known_since": q}]
    assert _stats(env)["self_evident"] == 2 == _stats(env)["initiated"]
    env = _env({"progressive_custom_events": [{**EVERY_QUARTER, "self_evident_prob": 0.0, "duration_mean_q": 40}]})
    _step(env)
    assert all(pe.known_defects(env, b) == [] for b in env.bridges.values()) and _stats(env)["initiated"] == 2


def test_monitoring_detection_is_optional():
    for prob, expected in ((0.0, False), (1.0, True)):
        env = _env({"progressive_monitoring_detect_prob": prob})
        env.bridges[1].monitoring_ends_quarter = env.quarter + 8
        _inject(env, 1, ends_in=40)
        _step(env)
        assert bool(pe.known_defects(env, env.bridges[1])) is expected
    assert _stats(env)["detected_by_monitoring"] == 1


# -- arresto -----------------------------------------------------------------------

def test_arrest_prevents_consequence_and_costs_reference_ratio():
    params = {"progressive_hydraulic_p_collapse_if_untreated": 1.0, "initial_budget": 1e9}
    env = _env(params)
    agent = _agent(env)
    _inject(env, 1, ends_in=0, known=True)        # maturerebbe in questo stesso trimestre
    _inject(env, 2, ends_in=0, known=True)
    ref = env.maint_ops["ripristino_strutturale"]
    cost = 0.3 * _calculate_cost(ref, env.bridges[1].area)
    assert env.get_feasible_actions_for_bridge(agent, env.bridges[1])[pe.ARREST_ACTION]["cost"] == pytest.approx(cost)
    assert env.maint_ops[pe.ARREST_ACTION]["reference_intervention"] == "ripristino_strutturale"
    costs, _ = env.step({agent: {"maintenance": {"1": pe.ARREST_ACTION}}})
    assert [(c["action"], c["cost"]) for c in costs[agent]] == [(pe.ARREST_ACTION, pytest.approx(cost))]
    # ponte 1 arrestato: nessuna conseguenza; ponte 2 non arrestato: crollo
    assert [(r["bridge_id"], r["cause"]) for r in env.collapse_log] == [(2, "progressive_hydraulic")]
    st = _stats(env)
    assert st["arrested_by_action"] == 1 and st["arrested_hydraulic"] == 1 and st["collapses"] == 1
    assert env.progressive_defects == {}
    # i lavori di arresto durano un trimestre e non applicano guadagni ne' rilievi
    assert env.bridges[1].active_maintenance["action_name"] == pe.ARREST_ACTION
    _step(env)
    assert env.bridges[1].active_maintenance is None


def test_any_restoring_intervention_also_arrests():
    env = _env({"initial_budget": 1e9}, n=3)
    _inject(env, 1, ends_in=40, known=True)
    _inject(env, 2, ends_in=40, known=False)
    _inject(env, 3, ends_in=40, known=True)
    # ripristino_strutturale (pile 0.5, spalle 0.5) ripristina i bersagli; replace_impalcato
    # (bonus 0.15) no
    _step(env, {"1": "ripristino_strutturale", "2": "ripristino_strutturale", "3": "replace_impalcato"})
    assert 1 not in env.progressive_defects and 2 not in env.progressive_defects and 3 in env.progressive_defects
    st = _stats(env)
    assert st["arrested_by_intervention"] == 1 and st["arrested_before_detection"] == 1 and st["arrested"] == 2
    hist2 = [e["action"] for e in env.bridges[2].action_history]
    assert "difetto_progressivo" not in hist2            # il difetto mai noto non lascia tracce


def test_ethical_floor_arrests_known_defect_or_closes_when_unfunded():
    params = {"ethical_floor_enabled": True, "ethical_floor_condition": 0.0, "initial_budget": 1e9,
              "progressive_hydraulic_p_collapse_if_untreated": 1.0}
    env = _env(params)
    agent = _agent(env)
    _inject(env, 1, ends_in=0, known=True)
    _inject(env, 2, ends_in=40, known=False)             # non noto: la soglia non lo vede
    assert env._floor_triggered(env.bridges[1]) and not env._floor_triggered(env.bridges[2])
    costs, _ = env.step({agent: {"maintenance": {}}})
    assert [(c["action"], c.get("ethical_floor")) for c in costs[agent]] == [(pe.ARREST_ACTION, True)]
    st = _stats(env)
    assert st["floor_arrests"] == 1 and st["arrested_by_floor"] == 1 and st["floor_arrest_spend"] > 0
    assert env.collapse_log == [] and 1 not in env.progressive_defects and 2 in env.progressive_defects
    assert env.floor_stats[agent]["interventions"] == 1

    # senza cassa: chiusura; la soglia riprova e arresta quando la cassa torna
    env = _env(params)
    agent = _agent(env)
    _inject(env, 1, ends_in=40, known=True)
    env.budgets[agent] = 0.0
    _step(env)
    st = _stats(env)
    assert st["floor_arrests"] == 0 and st["floor_budget_blocked"] == 1 and st["floor_closures"] == 1
    assert env.bridges[1].traffic_restriction_factor == 0.0 and env.bridges[1].floor_closure
    assert pe.known_defects(env, env.bridges[1])
    env.budgets[agent] = 1e9
    _step(env)
    assert _stats(env)["floor_arrests"] == 1 and not pe.known_defects(env, env.bridges[1])
    _step(env, n=2)
    assert not getattr(env.bridges[1], "floor_closure", None) and env.bridges[1].traffic_restriction_factor == 1.0

    # trigger disattivabile
    env = _env({**params, "progressive_floor_trigger": False})
    _inject(env, 1, ends_in=40, known=True)
    assert not env._floor_triggered(env.bridges[1])


# -- conseguenza a fine finestra ------------------------------------------------------

def test_maturation_applies_hidden_loss():
    env = _env({"progressive_hydraulic_p_collapse_if_untreated": 0.0, "observation_mode": "inspected",
                "initial_inventory_known": True})
    b = env.bridges[1]
    before = {n: c.current_condition for n, c in b.components.items()}
    seen = {n: (c.observed_condition_min, c.observed_condition_max) for n, c in b.components.items()}
    _inject(env, 1, ends_in=0)
    _step(env)
    st = _stats(env)
    assert st["matured"] == 1 and st["matured_hydraulic"] == 1 and st["collapses"] == 0
    # perdita = parametri del danno locale idraulico (pile e spalle, 0.20)
    for name in ("pile", "spalle"):
        assert before[name] - b.components[name].current_condition > 0.15
    assert before["impalcato"] - b.components["impalcato"].current_condition < 0.05
    assert st["condition_loss_total"] == pytest.approx(0.4, abs=0.02)
    # nascosta: osservazioni invariate, danno fra i pendenti del canale improvviso
    assert {n: (c.observed_condition_min, c.observed_condition_max) for n, c in b.components.items()} == seen
    assert sorted((e["type"], e["component"]) for e in env.sudden_pending[1]) == [
        ("progressive:hydraulic", "pile"), ("progressive:hydraulic", "spalle")]
    assert pe.metrics(env, _agent(env))["progressive_damage_pending_undetected"] == 2
    assert sd.metrics(env, _agent(env)).get("sudden_pending_undetected", 0) == 0
    assert env.collapse_log == [] and env.progressive_defects == {}
    _step(env, {"1": "L1_inspect"})
    assert 1 not in env.sudden_pending and _stats(env)["damage_detected_by_inspection"] == 2


def test_known_defect_maturing_reveals_the_damage():
    env = _env({"progressive_hydraulic_p_collapse_if_untreated": 0.0, "observation_mode": "inspected",
                "initial_inventory_known": True})
    _inject(env, 1, ends_in=0, known=True)
    seen = {n: c.observed_condition_min for n, c in env.bridges[1].components.items()}
    _step(env)
    assert 1 not in env.sudden_pending and _stats(env)["matured_known"] == 1
    assert _stats(env)["damage_revealed"] == 2
    now = {n: c.observed_condition_min for n, c in env.bridges[1].components.items()}
    assert now["pile"] != seen["pile"] and now["spalle"] != seen["spalle"] and now["impalcato"] == seen["impalcato"]
    assert [e["status"] for e in env.bridges[1].action_history if e["action"] == "difetto_progressivo"] == ["matured"]


def test_maturation_collapse_is_handled_like_a_direct_collapse():
    env = _env({"progressive_hydraulic_p_collapse_if_untreated": 1.0})
    agent = _agent(env)
    _inject(env, 1, ends_in=0)
    _step(env)
    assert env.failure_stats[agent]["failures"] == 1
    assert env.failure_stats[agent]["realised_loss"] > env.failure_stats[agent]["realised_loss_c0"] > 0
    assert all(c.current_condition == pytest.approx(1.0) for c in env.bridges[1].components.values())
    assert env.bridges[1].action_history[-1]["action"] == "crollo_progressivo"
    assert env.last_failure_info["reason"] == "PROGRESSIVE_COLLAPSE" and agent in env.active_agents
    assert _stats(env)["collapses_hydraulic"] == 1


def test_geotechnical_defaults_and_exposure_assignment():
    p = SimParameters()
    assert p.progressive_geotechnical_drop_mean == 0.35 and p.progressive_geotechnical_p_collapse_if_untreated == 0.10
    env = _env({}, n=30)                                   # quota 0: nessuno esposto
    assert not any(getattr(b, "on_unstable_slope", False) for b in env.bridges.values())
    spec = pe._spec_by_name(env.params)["geotechnical"]
    assert all(pe.initiation_rate_q(env.params, b, spec) == 0.0 for b in env.bridges.values())
    a = _env({"progressive_geotechnical_exposed_fraction": 0.5}, n=30)
    b = _env({"progressive_geotechnical_exposed_fraction": 0.5}, n=30)
    exposed = [bid for bid, br in a.bridges.items() if br.on_unstable_slope]
    assert 5 <= len(exposed) <= 25
    assert exposed == [bid for bid, br in b.bridges.items() if br.on_unstable_slope]
    assert BridgeEnv.from_dict(a.to_dict()).bridges[exposed[0]].on_unstable_slope is True
    # attributo dello scenario: prevale sulla quota
    c = _env({"progressive_geotechnical_exposed_fraction": 0.0}, n=3, slope=True)
    assert all(pe.initiation_rate_q(c.params, br, spec) == pytest.approx(1e-3 / 4) for br in c.bridges.values())
    # conseguenza geotecnica: lognormale, media 0.35
    c = _env({"progressive_geotechnical_p_collapse_if_untreated": 0.0}, n=3, slope=True)
    for bid in c.bridges:
        _inject(c, bid, typ="geotechnical", ends_in=0)
    _step(c)
    assert 0.1 < _stats(c)["condition_loss_total"] / 6 < 0.7


def test_regional_hit_can_initiate_a_latent_hydraulic_defect():
    regional = {"regional_events_enabled": True, "regional_event_rate_y": 4.0, "regional_event_radius_km": 600.0,
                "regional_event_decay_model": "uniform", "regional_event_hit_prob": 1.0,
                "regional_event_multipliers": {}}
    off = _env(regional, n=4)
    on = _env({**regional, "progressive_hydraulic_regional_trigger_prob": 1.0}, n=4)
    for env in (off, on):
        _step(env, n=6)
    assert off.regional_events and _stats(off)["regional_triggered"] == 0 == _stats(off)["initiated"]
    assert _stats(on)["regional_triggered"] > 0
    assert _stats(on)["regional_triggered"] == _stats(on)["initiated_hydraulic"]


# -- anti-sovradimensionamento ----------------------------------------------------------

HYD = {"hydraulic_hazard_enabled": True, "hydraulic_legacy_mode": False, "hydraulic_damage_rate_multiplier": 1.0,
       "hydraulic_collapse_rate_water_q": 0.06, "hydraulic_collapse_rate_scour_q": 0.18,
       "reliability_refresh_every_quarters": 40}


def _direct(env):
    return [(r["quarter"], r["bridge_id"]) for r in env.collapse_log if r["cause"] == "hydraulic_direct"]


def test_share_scales_the_direct_collapse_rate():
    n, quarters, share = 12, 12, 0.5
    full = _env({**HYD, "progressive_events_enabled": False}, n=n, scour=(1, 2))
    half = _env({**HYD, "hydraulic_collapse_progressive_share": share,
                 "progressive_hydraulic_p_collapse_if_untreated": 1.0,
                 "progressive_hydraulic_duration_mean_q": 400}, n=n, scour=(1, 2))
    off = _env({**HYD, "progressive_events_enabled": False, "hydraulic_collapse_progressive_share": share}, n=n,
               scour=(1, 2))
    for env in (full, half, off):
        _step(env, n=quarters)
    # stessi numeri casuali: i crolli diretti rimasti sono quelli con u < tasso x (1 - quota)
    expected_full, expected_half = [], []
    for q in range(1, quarters + 1):
        for bid, b in sorted(full.bridges.items()):
            u = float(full._keyed_stream(sd.PURPOSE["hydraulic"], bid, q).random())
            rate = 0.18 if b.scour_critical else 0.06
            if u < rate:
                expected_full.append((q, bid))
            if u < rate * (1 - share):
                expected_half.append((q, bid))
    assert sorted(_direct(full)) == sorted(expected_full) and len(expected_full) > 5
    assert sorted(_direct(half)) == sorted(expected_half) and len(expected_half) < len(expected_full)
    # senza il tipo idraulico progressivo la quota non toglie nulla
    assert sorted(_direct(off)) == sorted(expected_full)
    assert pe.effective_share(off.params) == 0.0 and pe.effective_share(half.params) == share


def test_calibrated_initiation_preserves_expected_rate_analytic():
    params = {**HYD, "hydraulic_collapse_rate_water_q": 2.7e-5, "hydraulic_collapse_rate_scour_q": 2.7e-4,
              "hydraulic_collapse_progressive_share": 0.5}
    env = _env(params, n=10, scour=(1, 2, 3))
    dry = _env(params, n=4, water=False)
    spec = pe._spec_by_name(env.params)["hydraulic"]
    for b in env.bridges.values():
        hist = pe.hydraulic_direct_rate_q(env.params, b)
        lam = pe.initiation_rate_q(env.params, b, spec)
        assert lam == pytest.approx(0.5 * hist / 0.05)
        # diretto rimasto + innesco x P(crollo se non trattato) = tasso storico
        assert (1 - 0.5) * hist + lam * spec["p_collapse"] == pytest.approx(hist)
    assert all(pe.initiation_rate_q(dry.params, b, spec) == 0.0 for b in dry.bridges.values())
    rep = pe.expected_hydraulic_collapse_rates(env, horizon_quarters=200)
    for key in ("per_bridge_year", "per_water_bridge_year"):
        assert rep[key]["total_stationary"] == pytest.approx(rep[key]["historical_direct"])
        assert rep[key]["direct"] == pytest.approx(0.5 * rep[key]["historical_direct"])
        assert rep[key]["total_horizon"] < rep[key]["total_stationary"]
    assert rep["per_bridge_year"]["historical_direct"] == pytest.approx(4 * (3 * 2.7e-4 + 7 * 2.7e-5) / 10)
    assert rep["matured_fraction"] == pytest.approx(pe.matured_fraction(spec, 200)) and 0.85 < rep["matured_fraction"] < 0.95
    # modo "fixed": tasso dato, con il fattore per fondazione a rischio erosione
    fixed = _env({**params, "progressive_hydraulic_rate_mode": "fixed", "progressive_hydraulic_rate_y": 4e-3}, n=2,
                 scour=(1,))
    f_spec = pe._spec_by_name(fixed.params)["hydraulic"]
    assert pe.initiation_rate_q(fixed.params, fixed.bridges[1], f_spec) == pytest.approx(3e-3)
    assert pe.initiation_rate_q(fixed.params, fixed.bridges[2], f_spec) == pytest.approx(1e-3)


def test_duration_distributions_match_their_cdf():
    import numpy as np
    rng = np.random.default_rng(3)
    for dist, mean, cov in (("geometric", 20.0, 0.0), ("lognormal", 4.0, 1.0), ("fixed", 6.0, 0.0)):
        spec = {"duration_dist": dist, "duration_mean_q": mean, "duration_cov": cov}
        draws = np.array([pe.draw_duration(spec, rng) for _ in range(20000)])
        assert draws.min() >= 1 and draws.mean() == pytest.approx(mean, rel=0.06)
        for k in (1, 3, 10, 40):
            assert float((draws <= k).mean()) == pytest.approx(pe.duration_cdf(spec, k), abs=0.02)


def test_calibrated_initiation_preserves_expected_rate_monte_carlo():
    """Gestore che non ispeziona e non arresta: i crolli idraulici totali (diretti +
    da difetto latente) restano quelli storici, entro la tolleranza del campione."""
    n, quarters, seeds = 16, 40, (11, 12, 13)
    params = {**HYD, "hydraulic_collapse_progressive_share": 0.5,
              "progressive_hydraulic_p_collapse_if_untreated": 0.5, "progressive_hydraulic_duration_mean_q": 2.0,
              "progressive_hydraulic_self_evident_prob": 0.0, "progressive_geotechnical_enabled": False}
    hist = direct = latent = initiated = 0
    for seed in seeds:
        base = _env({**HYD, "progressive_events_enabled": False}, n=n, seed=seed)
        env = _env(params, n=n, seed=seed)
        for e in (base, env):
            _step(e, n=quarters)
        hist += len(_direct(base))
        direct += len(_direct(env))
        latent += sum(1 for r in env.collapse_log if r["cause"] == "progressive_hydraulic")
        initiated += _stats(env)["initiated"]
    rep = pe.expected_hydraulic_collapse_rates(env, horizon_quarters=quarters)
    expected = rep["per_bridge_year"]["total_horizon"] / 4 * n * quarters * len(seeds)
    expected_hist = rep["per_bridge_year"]["historical_direct"] / 4 * n * quarters * len(seeds)
    assert expected_hist == pytest.approx(0.06 * n * quarters * len(seeds))
    assert 0.9 * expected_hist < expected < expected_hist
    assert initiated == pytest.approx(0.06 * n * quarters * len(seeds), rel=0.3)   # innesco = 0.5 x 0.06 / 0.5
    assert latent > 0 and direct < hist
    assert direct + latent == pytest.approx(expected, rel=0.25)
    assert direct + latent == pytest.approx(hist, rel=0.3)


# -- stato, copie dei pianificatori, sandbox ---------------------------------------------

def _mixed_env(extra=None):
    env = _env({"observation_mode": "inspected", "policy_sandbox": True, "strict_information": True,
                "initial_inventory_known": True, "progressive_hydraulic_p_collapse_if_untreated": 1.0,
                **(extra or {})}, n=3)
    _inject(env, 3, ends_in=0)                       # crolla subito: una riga nel registro
    _step(env)
    _inject(env, 1, ends_in=9, known=True)
    _inject(env, 2, ends_in=7, known=False)
    return env


def test_state_roundtrip_and_truth_clone():
    env = _mixed_env()
    back = BridgeEnv.from_dict(json.loads(json.dumps(env.to_dict())))
    assert back.progressive_defects == env.progressive_defects
    assert back.progressive_stats == env.progressive_stats and back.collapse_log == env.collapse_log
    assert pe.ARREST_ACTION in back.maint_ops and len(env.collapse_log) == 1
    for e in (env, back):
        _step(e, n=3)
    assert back.progressive_defects == env.progressive_defects and back.collapse_log == env.collapse_log
    truth = _mixed_env({"observation_mode": "full", "policy_sandbox": False, "strict_information": False})
    clone = truth.planning_clone(salt=5)
    assert clone.progressive_defects == truth.progressive_defects and clone.collapse_log == truth.collapse_log


def test_belief_clone_and_sandbox_snapshot_hide_latent_truth():
    env = _mixed_env()
    agent = _agent(env)
    import numpy as np
    clone = env.belief_clone(salt=1, rng=np.random.default_rng(0))
    assert list(clone.progressive_defects) == [1]
    d = clone.progressive_defects[1][0]
    known_since = env.progressive_defects[1][0]["known_since"]
    assert d["known_since"] == known_since == d["start_quarter"] and d["end_quarter"] >= clone.quarter
    assert clone.collapse_log == [] and clone.progressive_stats[agent]["initiated"] == 0
    assert clone.progressive_stats[agent]["collapses"] == 0
    # la data di fine del difetto noto e' riestratta a ogni copia: non e' quella vera
    ends = {env.belief_clone(salt=1, rng=np.random.default_rng(s)).progressive_defects[1][0]["end_quarter"]
            for s in range(12)}
    assert len(ends) > 1
    snap = belief_snapshot(env, agent, seed=3)
    assert snap["progressive_defects"] == [[1, [{"type": "hydraulic", "known_since": known_since}]]]
    assert "progressive_stats" not in snap and "collapse_log" not in snap
    text = json.dumps(snap)
    assert "end_quarter" not in text and "start_quarter" not in text
    from infralab.agents.sandbox import PlanningHandle
    world = PlanningHandle.build(env, agent, seed=3).belief_clone()
    assert list(world.progressive_defects) == [1] and world.collapse_log == []
    world.step({agent: {"maintenance": {"1": pe.ARREST_ACTION}}})     # il pianificatore puo' simulare l'arresto
    assert world.progressive_defects.get(1) in (None, [])
    obs = sanitized_observation(env, agent)
    assert '"collapse_log"' not in json.dumps(obs) and "end_quarter" not in json.dumps(obs)


def test_perturbation_changes_latent_truth_with_same_evidence():
    env = _mixed_env()
    world = perturb_hidden_truth(env, seed=4)
    agent = _agent(env)
    assert [pe.public_view(world, b) for b in world.bridges.values()] == [pe.public_view(env, b)
                                                                          for b in env.bridges.values()]
    assert world.progressive_defects != env.progressive_defects
    assert world.progressive_stats[agent] != env.progressive_stats[agent]
    assert json.dumps(sanitized_observation(world, agent), sort_keys=True) == json.dumps(
        sanitized_observation(env, agent), sort_keys=True)


@pytest.mark.parametrize("trajectory", [False, True])
def test_leak_check_clean_with_progressive_events(trajectory):
    rep = run_leak_check("TRAIN-01", params={"observation_mode": "inspected", "policy_sandbox": True,
                                             "strict_information": True, "failure_model": "reliability",
                                             "revenue_health_source": "assessed", "rng_streams": "keyed",
                                             "ethical_floor_enabled": True, "initial_budget": 1e9},
                         quarters=10, progressive_events=True, trajectory=trajectory, seed=2,
                         policy="inspect-only" if trajectory else "inspect")
    assert rep["progressive_types"] == ["hydraulic", "geotechnical", "prova"]
    assert rep["progressive_initiated"] > 0 and rep["progressive_known"] > 0
    assert rep["quarters_checked"] >= 6
    assert rep["clean"], rep["divergences"][:5]
    if trajectory:
        assert any(g["progressive_hidden_a"] != g["progressive_hidden_b"] for g in rep["truth_gap"])


def test_leak_check_flags_a_leaked_latent_defect(monkeypatch):
    """Controllo non vuoto: se l'osservazione mostrasse i difetti non noti, il cancello
    lo segnalerebbe."""
    monkeypatch.setattr(pe, "public_view", lambda env, bridge: sorted(
        ({"type": d["type"], "known_since": d.get("known_since") or -1}
         for d in (env.progressive_defects.get(bridge.id) or [])), key=lambda x: (x["known_since"], x["type"])))
    rep = run_leak_check("TRAIN-01", params={"observation_mode": "inspected", "policy_sandbox": True,
                                             "strict_information": True}, quarters=4, progressive_events=True,
                         include=("observation",))
    assert not rep["clean"] and rep["divergences_by_output"].get("observation", 0) > 0


def test_strict_history_cap_ignores_hidden_hydraulic_damage():
    """strict_information: un danno idraulico nascosto (canale storico) non toglie dallo
    storico un evento visibile (prima la lunghezza dello storico ripulito lo rivelava)."""
    visible = {}
    for strict in (False, True):
        env = build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides={"strict_information": strict}, seed=7)
        b = env.bridges[1]
        for _ in range(12):
            env._record_action_event(b, "L1_inspect", "executed")
        env._record_action_event(b, "danno_idraulico", "hazard", {"condition_drop": 0.1})
        visible[strict] = sum(1 for e in b.action_history if e["action"] != "danno_idraulico")
    assert visible == {False: 11, True: 12}


# -- registro dei collassi ------------------------------------------------------------

def test_collapse_log_content_and_causes():
    params = {**HYD, "hydraulic_collapse_rate_water_q": 1.0, "ethical_floor_enabled": False,
              "ethical_floor_condition": 0.4, "observation_mode": "inspected", "initial_inventory_known": True,
              "progressive_custom_events": [{**EVERY_QUARTER, "rate_y": 0.0, "p_collapse_if_untreated": 1.0}],
              "progressive_hydraulic_p_collapse_if_untreated": 1.0,
              "progressive_geotechnical_p_collapse_if_untreated": 1.0}
    env = _env(params, n=5)
    agent = _agent(env)
    for bid in (2, 3, 4, 5):
        env.bridges[bid].crosses_water = False       # solo il ponte 1 ha il crollo idraulico diretto
    _inject(env, 2, typ="hydraulic", ends_in=0, known=True)
    _inject(env, 3, typ="geotechnical", ends_in=0)
    _inject(env, 4, typ="prova", ends_in=0)
    env.sudden_pending[4] = [{"type": "impact", "quarter": 1, "component": "impalcato", "loss": 0.1}]
    b5 = env.bridges[5]
    for c in b5.components.values():                 # ponte 5: Pf vera ~ 1, stato noto ancora buono
        lo, hi = c.observed_condition_min, c.observed_condition_max
        c.current_condition = 0.02
        c.apply_maintenance(0.0, env.params.deterioration_exponent)
        c.observed_condition_min, c.observed_condition_max = lo, hi
    env.refresh_bridge_reliability(b5)
    env.get_agent_observations(agent)                # stima del gestore aggiornata (come nel harness)
    q = env.quarter
    _step(env)
    log = {r["bridge_id"]: r for r in env.collapse_log}
    assert {bid: r["cause"] for bid, r in log.items()} == {
        1: "hydraulic_direct", 2: "progressive_hydraulic", 3: "progressive_geotechnical",
        4: "progressive_prova", 5: "pf"}
    for r in log.values():
        assert r["quarter"] == q and r["agent"] == agent and r["last_inspection_quarter"] == 0
        assert r["observed_min_condition"] > 0.4 and r["assessed_annual_beta"] is not None
        assert r["floor_trigger_active"] is False and r["closed"] is False and r["works_active"] is None
    assert log[5]["annual_pf"] > 0.5 and 0 < log[5]["quarterly_p"] <= 1 and log[1]["annual_pf"] is None
    assert log[5]["true_min_condition"] < 0.05 < log[5]["observed_min_condition"]
    assert log[1]["quarterly_p"] == pytest.approx(1.0)
    assert log[2]["known_defects"] == [{"type": "hydraulic", "known_since": q - 1}] and log[2]["known_critical"]
    assert log[2]["defect_known_since"] == q - 1 and log[3]["defect_known_since"] is None
    assert log[3]["known_defects"] == [] and not log[3]["known_critical"]
    assert log[4]["hidden_damage_pending"] == 1
    m = pe.collapse_metrics(env, agent)
    assert m["failures_cause_pf"] == 1 and m["failures_cause_hydraulic_direct"] == 1
    assert m["failures_cause_progressive_hydraulic"] == 1 and m["failures_cause_progressive_prova"] == 1
    assert m["failures_hydraulic_total"] == 2 and m["failures_known_critical"] == 1
    assert m["failures_with_known_defect"] == 1
    assert sum(v for k, v in m.items() if k.startswith("failures_cause_")) == env.failure_stats[agent]["failures"] == 5


def test_collapse_log_floor_trigger_and_metrics_switch():
    env = _env({"ethical_floor_enabled": True, "ethical_floor_condition": 0.0, "initial_budget": 0.0,
                "progressive_hydraulic_p_collapse_if_untreated": 1.0})
    _inject(env, 1, ends_in=0, known=True)
    _step(env)
    (rec,) = env.collapse_log
    assert rec["floor_trigger_active"] and rec["known_critical"] and rec["closed"] and rec["floor_closure"]
    # contatori per causa anche a canale spento, se richiesti
    off = build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides={"collapse_log_metrics": True}, seed=7)
    assert pe.collapse_metrics(off, _agent(off)) == {
        "failures_cause_pf": 0.0, "failures_cause_hydraulic_direct": 0.0, "failures_hydraulic_total": 0.0,
        "failures_known_critical": 0.0, "failures_with_known_defect": 0.0}


def test_legacy_condition_failure_is_logged():
    env = build_env_from_scenario(_scenario(1), seed=7)          # failure_model="condition" (storico)
    agent = _agent(env)
    for c in env.bridges[1].components.values():
        c.current_condition = 0.01
        c.apply_maintenance(0.0, env.params.deterioration_exponent)
    _step(env)
    assert agent not in env.active_agents
    assert [r["cause"] for r in env.collapse_log] == ["condition_threshold"]


# -- regola di fine partita -------------------------------------------------------------

@pytest.mark.parametrize("rule,known,out", [
    ("none", True, False), ("any", False, True), ("any", True, True),
    ("known_critical", False, False), ("known_critical", True, True),
])
def test_collapse_game_over_modes(rule, known, out):
    env = _env({"collapse_game_over": rule, "progressive_hydraulic_p_collapse_if_untreated": 1.0})
    agent = _agent(env)
    _inject(env, 1, ends_in=0, known=known)
    _step(env)
    assert len(env.collapse_log) == 1 and env.collapse_log[0]["known_critical"] is known
    assert (agent not in env.active_agents) is out
    if out:
        info = env.last_failure_info
        assert info["reason"] == "BRIDGE_COLLAPSE" and info["bridge_id"] == 1 and info["rule"] == rule
        assert info["cause"] == "progressive_hydraulic"
    else:
        assert env.last_failure_info["reason"] == "PROGRESSIVE_COLLAPSE"


def test_known_critical_from_observed_condition_and_closed_exemption():
    # ponte noto sotto la soglia (anche a soglia etica spenta) che crolla per la Pf
    env = _env({"collapse_game_over": "known_critical", "ethical_floor_condition": 0.4,
                "progressive_events_enabled": False}, n=1)
    agent = _agent(env)
    b = env.bridges[1]
    for c in b.components.values():
        c.current_condition = 0.02
        c.apply_maintenance(0.0, env.params.deterioration_exponent)
        c.observed_condition_min, c.observed_condition_max = 0.0, 0.1
    env.refresh_bridge_reliability(b)
    _step(env)
    assert env.collapse_log[0]["cause"] == "pf" and env.collapse_log[0]["known_critical"]
    assert agent not in env.active_agents
    # ponte chiuso dalla soglia etica (senza cassa per l'arresto): esente, salvo scelta contraria
    for exempt, out in ((True, False), (False, True)):
        env = _env({"collapse_game_over": "known_critical", "collapse_game_over_exempt_closed": exempt,
                    "ethical_floor_enabled": True, "ethical_floor_condition": 0.0, "initial_budget": 0.0,
                    "progressive_hydraulic_p_collapse_if_untreated": 1.0})
        _inject(env, 1, ends_in=0, known=True)
        _step(env)
        assert env.collapse_log[0]["closed"] and (_agent(env) not in env.active_agents) is out


def test_game_over_terminates_the_lab_run():
    raw = {"name": "go", "scenario": "TRAIN-01", "horizon_quarters": 30,
           "agents": [{"id": "dn", "type": "Basic", "params": {"policy": "do_nothing"}}],
           "grid": {"params.progressive_events_enabled": [True], "params.failure_model": ["reliability"],
                    "params.collapse_game_over": ["any"],
                    "params.progressive_custom_events": [[{**EVERY_QUARTER, "p_collapse_if_untreated": 1.0}]]},
           "replicates": 1, "seed": 1, "outputs": {"dir": "results/go", "formats": ["csv"]}}
    run_row, steps, collapses = run_one_full(expand_runs(spec_from_dict(raw), load_scenario("TRAIN-01"))[0])
    assert run_row["terminated"] and run_row["failure_reason"] == "BRIDGE_COLLAPSE"
    assert run_row["quarters_survived"] == 4 and len(collapses) >= 1


# -- API, UI e spec del laboratorio -----------------------------------------------------

def test_backend_config_exposes_all_fields():
    from backend.app import CONFIG_FIELDS
    keys = {f["key"] for f in CONFIG_FIELDS}
    names = {f.name for f in fields(SimParameters)}
    own = {n for n in names if n.startswith(("progressive_", "collapse_"))} | {"hydraulic_collapse_progressive_share"}
    assert own <= keys and own == {f["key"] for f in pe.UI_FIELDS}
    defaults = SimParameters()
    for f in pe.UI_FIELDS:
        if f["type"] == "select":
            assert getattr(defaults, f["key"]) in f["options"]


def test_api_session_with_progressive_events():
    from fastapi.testclient import TestClient
    from backend.app import app
    client = TestClient(app)
    cfg = client.get("/api/config").json()
    assert cfg["defaults"]["progressive_events_enabled"] is False
    assert cfg["defaults"]["progressive_hydraulic_duration_mean_q"] == 20.0
    body = {"scenario_id": "TRAIN-01", "seed": 3,
            "params": {"progressive_events_enabled": True, "failure_model": "reliability",
                       "collapse_game_over": "known_critical", "progressive_arrest_duration_quarters": 1.0,
                       "progressive_custom_events": [{**EVERY_QUARTER, "self_evident_prob": 1.0,
                                                      "duration_mean_q": 40}]}}
    res = client.post("/api/sessions", json=body)
    assert res.status_code == 200, res.text
    state = client.post(f"/api/sessions/{res.json()['session_id']}/step", json={"steps": 1}).json()
    assert "failure_info" in state
    asset = state["assets"][0]
    assert asset["known_progressive_defects"] and asset["known_progressive_defects"][0]["type"] == "prova"
    assert any(a["action"] == pe.ARREST_ACTION for a in asset["actions"])
    assert client.post("/api/sessions", json={**body, "params": {"collapse_game_over": "boh"}}).status_code == 400


def test_lab_spec_parameters_and_collapses_file(tmp_path):
    raw = {"name": "prog", "scenario": "TRAIN-01", "horizon_quarters": 8,
           "agents": [{"id": "dn", "type": "Basic", "params": {"policy": "do_nothing"}}],
           "grid": {"params.progressive_events_enabled": [True], "params.failure_model": ["reliability"],
                    "params.progressive_custom_events": [[{**EVERY_QUARTER, "p_collapse_if_untreated": 0.5}]]},
           "replicates": 2, "seed": 1, "outputs": {"dir": str(tmp_path / "out"), "formats": ["csv"]}}
    spec = spec_from_dict(raw)
    results = run_experiment(spec, scenario=load_scenario("TRAIN-01"))
    run_row = results["runs"][0]
    assert run_row["final_progressive_initiated"] == 16 and run_row["final_progressive_matured"] > 0
    assert run_row["final_failures_cause_progressive_prova"] == run_row["final_progressive_collapses"] > 0
    assert "final_failures_known_critical" in run_row
    steps = [s for s in results["steps"] if s["run_id"] == run_row["run_id"]]
    assert sum(s["metric_progressive_initiated_quarter"] for s in steps) == run_row["final_progressive_initiated"]
    assert len(results["collapses"]) == sum(r["final_failures_total"] for r in results["runs"])
    written = write_outputs(str(tmp_path / "out"), results, {"spec_name": "prog"}, ["csv"])
    import pandas as pd
    df = pd.read_csv(written["collapses.csv"])
    assert list(df.columns)[:4] == ["run_id", "agent_id", "replicate", "seed"]
    assert set(COLLAPSE_COLUMNS) <= set(df.columns) and len(df) == len(results["collapses"])
    assert set(df["cause"]) <= {"progressive_prova", "pf"}
    # senza collassi: solo l'intestazione; runs e steps invariati
    empty = write_outputs(str(tmp_path / "empty"), {"runs": [], "steps": [], "collapses": [], "errors": []},
                          {"spec_name": "x"}, ["csv"])
    assert list(pd.read_csv(empty["collapses.csv"]).columns)[4:] == list(COLLAPSE_COLUMNS)
    legacy = write_outputs(str(tmp_path / "legacy"), {"runs": [], "steps": [], "errors": []}, {"spec_name": "x"}, ["csv"])
    assert "collapses.csv" not in legacy
    assert len(run_one(expand_runs(spec, load_scenario("TRAIN-01"))[0])) == 2      # firma storica


def test_hydraulic_rate_cli(capsys, tmp_path):
    from infralab.lab.hydraulic_rate import hydraulic_rate_report, main
    path = tmp_path / "sc.json"
    path.write_text(json.dumps(_scenario(6, scour=(1,))), encoding="utf-8")
    args = ["--scenario", str(path), "--horizon", "200", "--param", "hydraulic_hazard_enabled=true",
            "--param", "progressive_events_enabled=true", "--param", "hydraulic_collapse_progressive_share=0.5"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "ancora empirica" in out and "latente" in out and "su_acqua=6" in out
    rep = hydraulic_rate_report(str(path), params={"hydraulic_hazard_enabled": True})
    assert rep["share"] == 0.0 and rep["per_bridge_year"]["latent_stationary"] == 0.0
    assert rep["per_bridge_year"]["total_stationary"] == pytest.approx(4 * (2.7e-4 + 5 * 2.7e-5) / 6)
