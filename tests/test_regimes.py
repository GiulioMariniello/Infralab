"""Regime normativo come variabile dell'ambiente: registro, hook del motore, strategie base."""
import copy
import json

import pytest

from infralab.agents.adapters import make_agent
from infralab.agents.basic import BASIC_POLICIES, BasicAgent
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.regimes import (
    NULL_REGIME,
    RegulatoryRegime,
    available_regimes,
    get_regime,
    regime_from_dict,
    register_regime,
)
from tests.conftest import load_scenario

SCN = "PAPERA-SHORT-A"


def _env(regime="none", policy="do_nothing", seed=1, **overrides):
    sc = load_scenario(SCN)
    cfg = {"type": "Basic", "policy": policy}
    env = build_env_from_scenario(sc, agent_name="A", agent_config=cfg,
                                  param_overrides={"regulatory_regime": regime, **overrides}, seed=seed)
    return env


def _run(env, policy, quarters):
    ag = make_agent("A", {"type": "Basic", "policy": policy})
    ag.reset(env.get_agent_observations("A"), {"env": env})
    spent = 0.0
    for _ in range(quarters):
        costs, _ = env.step({"A": ag.act(env.get_agent_observations("A"))})
        spent += sum(c["cost"] for c in costs["A"])
    return spent


# ---------------------------------------------------------------- registro
def test_builtin_regimes_registered():
    names = available_regimes()
    for n in ("none", "it_lg2020_signal", "it_lg2020_forced", "it_lg2020_sanctioned", "hyp_risk_based_existing"):
        assert n in names
    assert get_regime(None) is NULL_REGIME and get_regime("none").is_null
    assert get_regime("it_lg2020_sanctioned").sanctions and not get_regime("it_lg2020_sanctioned").forces
    assert get_regime("it_lg2020_forced").forces
    assert get_regime("hyp_risk_based_existing").target_tier == "NEN8700_disapproval"


def test_regime_from_dict_validation():
    with pytest.raises(ValueError):
        regime_from_dict({"id": "x", "obligations": [{"trigger": "nope", "action": "L1_inspect"}]})
    with pytest.raises(ValueError):
        regime_from_dict({"id": "x", "enforcement": {"mode": "sanction"}})   # scala mancante
    r = regime_from_dict({"id": "x", "enforcement": {"mode": "forced"},
                          "obligations": [{"trigger": "inspection_overdue", "action": "L1_inspect"}]})
    assert r.forces and r.to_dict()["enforcement"]["mode"] == "forced"


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        _env(regime="does_not_exist")


# ---------------------------------------------------------------- neutralita' del regime nullo
def test_null_regime_is_transparent():
    a = _env("none"); b = _env("none")
    _run(a, "do_nothing", 8); _run(b, "do_nothing", 8)
    assert json.dumps(a.to_dict(), sort_keys=True, default=float) == json.dumps(b.to_dict(), sort_keys=True, default=float)
    assert all(v == 0 for v in a.regime_stats["A"].values())
    obs = a.get_agent_observations("A")
    assert obs["regime"]["id"] == "none"
    assert all(bd["regime_mandatory_action"] is None for bd in obs["my_bridges"])


# ---------------------------------------------------------------- tier di target
def test_regime_target_tier_overrides_code_framework():
    env = _env("hyp_risk_based_existing")
    assert env.params.code_framework == "NEN8700_disapproval"
    assert env.params.reliability_reference_period_years == 15
    assert env.params.reliability_target_beta == 2.5
    env_new = _env("none", code_framework="EN1990")
    t_exist = list(env.bridges.values())[0].reliability_details["target_beta_annual"]
    t_new = list(env_new.bridges.values())[0].reliability_details["target_beta_annual"]
    assert t_exist < t_new


def test_regime_param_overrides_applied():
    env = _env("it_lg2020_signal")
    assert env.params.l4_escalation_quarters == 4


# ---------------------------------------------------------------- esecuzione d'ufficio
def test_forced_execution_injects_and_charges():
    env = _env("it_lg2020_forced")
    spent = _run(env, "do_nothing", 12)
    st = env.regime_stats["A"]
    assert st["forced_actions"] > 0
    assert spent == pytest.approx(st["forced_cost"])
    diags = env.last_plan_diagnostics["A"] if env.last_plan_diagnostics["A"] else []
    # dopo 12 trimestri ogni asset e' stato ispezionato almeno una volta d'ufficio
    assert all(b.last_inspection_quarter > 0 for b in env.bridges.values())
    # il regime "solo segnale" non forza nulla ma registra le inadempienze
    env_s = _env("it_lg2020_signal"); _run(env_s, "do_nothing", 12)
    assert env_s.regime_stats["A"]["forced_actions"] == 0
    assert env_s.regime_stats["A"]["breach_bridge_quarters"] > 0


def test_forced_execution_not_duplicated_when_agent_complies():
    env = _env("it_lg2020_forced")
    _run(env, "compliance_minimal", 12)
    assert env.regime_stats["A"]["forced_actions"] == 0
    assert env.regime_stats["A"]["breach_bridge_quarters"] == 0


# ---------------------------------------------------------------- sanzione progressiva
def test_sanction_ladder_escalates_and_blocks_reopening():
    env = _env("it_lg2020_sanctioned")
    _run(env, "do_nothing", 16)
    st = env.regime_stats["A"]
    assert st["sanction_bridge_quarters"] > 0
    factors = [b.traffic_restriction_factor for b in env.bridges.values()]
    assert min(factors) < 1.0
    b = list(env.bridges.values())[0]
    assert b.regime_sanction is not None and b.regime_sanction["level"] >= 1
    # il gestore non puo' riaprire finche' la sanzione e' attiva (catalogo completo)
    env.maint_ops["riapertura"] = env._full_catalogue()["riapertura"]
    assert env._action_block_reason("A", b, "riapertura") == "regime_sanction_active"
    # i ricavi sono ridotti rispetto al regime nullo
    env0 = _env("none"); _run(env0, "do_nothing", 16)
    assert env.budgets["A"] < env0.budgets["A"]


def test_sanction_lifted_on_compliance():
    env = _env("it_lg2020_sanctioned")
    _run(env, "do_nothing", 10)
    b = list(env.bridges.values())[0]
    assert b.regime_sanction is not None
    # il gestore adempie: ispeziona tutti gli asset
    env.step({"A": {"maintenance": {str(x.id): "L1_inspect" for x in env.bridges.values()}}})
    assert b.regime_sanction is None
    assert b.traffic_restriction_factor == 1.0


# ---------------------------------------------------------------- risorse vincolate
def test_earmarked_budget_blocks_non_analysis_actions():
    reg = regime_from_dict({"id": "earmark_test", "resources": {"earmarked_fraction": 0.99}})
    register_regime(reg, overwrite=True)
    env = _env("earmark_test")
    b = list(env.bridges.values())[0]
    # azione non di analisi piu' costosa fra quelle affrontabili col budget pieno:
    # con il 99% riservato all'analisi deve essere rifiutata.
    feas = env.get_feasible_actions_for_bridge("A", b)
    budget = env.budgets["A"]
    costly = max((a for a, i in feas.items() if i["type"] != "analysis" and i["cost"] <= budget),
                 key=lambda a: feas[a]["cost"])
    assert feas[costly]["cost"] > 0.01 * budget
    env.step({"A": {"maintenance": {str(b.id): costly}}})
    reasons = {d["reason"] for d in env.last_plan_diagnostics["A"]}
    assert "earmarked_budget" in reasons
    assert env.regime_stats["A"]["rejected_actions"] == 1
    env.step({"A": {"maintenance": {str(b.id): "L1_inspect"}}})
    assert any(d["status"] == "accepted" for d in env.last_plan_diagnostics["A"])


# ---------------------------------------------------------------- persistenza
def test_regime_state_roundtrip():
    env = _env("it_lg2020_sanctioned")
    _run(env, "do_nothing", 10)
    data = json.loads(json.dumps(env.to_dict(), default=float))
    env2 = BridgeEnv.from_dict(copy.deepcopy(data))
    assert env2.regime.id == "it_lg2020_sanctioned"
    assert env2.regime_stats["A"] == env.regime_stats["A"]
    b1, b2 = list(env.bridges.values())[0], list(env2.bridges.values())[0]
    assert b2.regime_sanction == b1.regime_sanction
    assert b2.regime_trigger_since == b1.regime_trigger_since


# ---------------------------------------------------------------- strategie base
@pytest.mark.parametrize("policy", sorted(BASIC_POLICIES))
def test_basic_policies_produce_valid_plans(policy):
    env = _env("it_lg2020_signal")
    ag = BasicAgent("A", policy=policy)
    for _ in range(8):
        plan, rationale = ag.decide(env)
        assert set(plan) == {"maintenance"}
        for k, v in plan["maintenance"].items():
            assert int(k) in env.bridges and v in env.maint_ops
        env.step({"A": plan})
        assert all(d["status"] != "blocked" or d["reason"] != "unknown_action" for d in env.last_plan_diagnostics["A"])


def test_basic_policies_differ_and_factory_works():
    spent = {}
    for policy in ("do_nothing", "compliance_minimal", "inspect_and_repair"):
        env = _env("it_lg2020_signal")
        spent[policy] = _run(env, policy, 12)
    assert spent["do_nothing"] == 0.0
    assert spent["inspect_and_repair"] > spent["compliance_minimal"] > 0.0
    with pytest.raises(ValueError):
        BasicAgent("A", policy="not_a_policy")


def test_gym_metrics_expose_regime_stats():
    from infralab.agents.protocol import AssetEnvGym
    sc = load_scenario(SCN)
    gym = AssetEnvGym(lambda seed: build_env_from_scenario(
        sc, agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"},
        param_overrides={"regulatory_regime": "it_lg2020_forced"}, seed=seed), "A")
    obs, info = gym.reset(seed=3)
    for _ in range(10):
        obs, r, term, trunc, info = gym.step({"maintenance": {}})
    assert info["metrics"]["regime_forced_actions"] > 0
    assert "regime_sanctioned_assets" in info["metrics"]


def test_heuristic_inspection_rule_independent_of_enforce_guidelines():
    """La regola 'ispeziona alla scadenza' dell'euristico non deve dipendere dal throttle ricavi."""
    from infralab.agents.legacy import HeuristicAgent
    for flag in (True, False):
        env = _env("none", enforce_guidelines=flag)
        ag = HeuristicAgent("A")
        inspected = 0
        for _ in range(10):
            plan, _ = ag.decide(env)
            inspected += sum(1 for a in plan["maintenance"].values() if a == "L1_inspect")
            env.step({"A": plan})
        assert inspected > 0, f"euristico senza ispezioni con enforce_guidelines={flag}"


def test_nen8700_tiers_match_heron_table3():
    from infralab.codes import get_code_framework
    d = get_code_framework("NEN8700_disapproval"); r = get_code_framework("NEN8700_reconstruction")
    assert d.beta_reference_by_cc == {"CC1": 1.8, "CC2": 2.5, "CC3": 3.3} and d.reference_period_years == 15
    assert r.beta_reference_by_cc == {"CC1": 2.8, "CC2": 3.3, "CC3": 3.8} and r.reference_period_years == 15


def test_arena_regime_flag(tmp_path):
    from infralab.lab.arena import main
    import json as _json
    out = tmp_path / "arena"
    assert main(["--no-llm", "--competitors", "Euristico_Baseline", "--scenarios", "PAPERA-SHORT-A",
                 "--seeds", "1", "--horizon", "8", "--regime", "it_lg2020_forced",
                 "--param", "enforce_guidelines=false", "--output-dir", str(out)]) == 0
    payload = _json.loads(next(out.glob("arena_runs_*.json")).read_text(encoding="utf-8"))
    assert payload["campaign"]["param_overrides"] == {"enforce_guidelines": False, "regulatory_regime": "it_lg2020_forced"}
    assert payload["runs"][0]["status"] == "SUCCESS"


def test_expected_failure_loss_metric():
    """Perdita attesa: non negativa, coerente con Pf e valore, nessun effetto sulla dinamica."""
    env = _env("none")
    loss = env.expected_quarterly_failure_loss("A")
    assert loss["total"] >= 0 and abs(loss["total"] - loss["replacement"] - loss["consequence"]) < 1e-6
    assert abs(loss["consequence"] - 3.0 * loss["replacement"]) < 1e-6 * max(1.0, loss["total"])
    env2 = _env("none", failure_consequence_ratio=0.0)
    assert env2.expected_quarterly_failure_loss("A")["consequence"] == 0.0
    # lettura pura: due env identici, uno interrogato a ogni trimestre, restano identici
    a, b = _env("none"), _env("none")
    for _ in range(6):
        a.expected_quarterly_failure_loss("A")
        a.step({"A": {"maintenance": {}}}); b.step({"A": {"maintenance": {}}})
    assert json.dumps(a.to_dict(), sort_keys=True, default=float) == json.dumps(b.to_dict(), sort_keys=True, default=float)


def test_gym_risk_adjusted_cost():
    from infralab.agents.protocol import AssetEnvGym
    sc = load_scenario(SCN)
    gym = AssetEnvGym(lambda seed: build_env_from_scenario(
        sc, agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"}, seed=seed), "A")
    gym.reset(seed=5)
    for _ in range(8):
        _, _, _, _, info = gym.step({"maintenance": {}})
    m = info["metrics"]
    assert m["risk_cost_total"] > 0
    assert abs(m["risk_adjusted_cost"] - (m["spent_total"] + m["risk_cost_total"])) < 1e-6
    # ricalcolo a posteriori per qualunque H/C0: totale = C0-part x (1 + ratio)
    assert abs(m["risk_cost_total"] - 4.0 * m["risk_cost_c0_total"]) < 1e-6 * max(1.0, m["risk_cost_total"])


# ---------------------------------------------------------------- campaign-4 engine interface
def test_inspected_mode_hides_truth_and_keeps_dynamics():
    a = _env("none", observation_mode="inspected"); b = _env("none")
    for _ in range(4):
        a.step({"A": {"maintenance": {}}}); b.step({"A": {"maintenance": {}}})
    oa = a.get_agent_observations("A")["my_bridges"][0]
    true_b = list(a.bridges.values())[0]
    assert oa["observation_mode"] == "inspected"
    assert oa["annual_reliability_index"] != true_b.annual_reliability_index      # assessed, not true
    assert all("accumulated_damage" not in c for c in oa["components"].values())
    # observation mode never changes the true dynamics
    assert json.dumps({k: v for k, v in a.to_dict().items() if k != "params"}, sort_keys=True, default=float) == \
           json.dumps({k: v for k, v in b.to_dict().items() if k != "params"}, sort_keys=True, default=float)


def test_inspection_moves_assessment_toward_observation():
    env = _env("none", observation_mode="inspected")
    b = list(env.bridges.values())[0]
    before = env._assessed_bridge(b).min_condition
    env.step({"A": {"maintenance": {str(b.id): "L1_inspect"}}})
    after = env._assessed_bridge(b)
    lo = b.min_observed_condition()
    assert lo is not None and abs(after.min_condition - before) >= 0.0
    assert abs(after.min_condition - lo) < 0.25


def test_reliability_failure_model_keyed_stream_and_continuation():
    from infralab.lab.spec import resolve_scenario
    def mk(policy):
        return build_env_from_scenario(resolve_scenario("NBI-DE"), agent_name="A", agent_config={"type": "Basic", "policy": policy},
                                       param_overrides={"failure_model": "reliability", "enforce_guidelines": False}, seed=3)
    e = mk("do_nothing")
    for _ in range(32):
        e.step({"A": {"maintenance": {}}})
    assert "A" in e.active_agents                          # a collapse does not end the run
    fs = e.failure_stats["A"]
    assert fs["realised_loss"] == pytest.approx(fs["realised_loss_c0"] * 4.0)
    assert e.last_expected_loss["A"]["total"] >= 0
    # the failure draw does not consume the simulation RNG: condition model and
    # reliability model share the stochastic degradation path until the first collapse
    e2 = build_env_from_scenario(resolve_scenario("NBI-DE"), agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"},
                                 param_overrides={"failure_model": "condition", "failure_threshold": 0.0, "enforce_guidelines": False}, seed=3)
    e3 = mk("do_nothing")
    e2.step({"A": {"maintenance": {}}}); e3.step({"A": {"maintenance": {}}})
    if e3.failure_stats["A"]["failures"] == 0:
        assert [c.current_condition for bb in e2.bridges.values() for c in bb.components.values()] == \
               [c.current_condition for bb in e3.bridges.values() for c in bb.components.values()]


def test_regime_with_unknown_action_rejected_at_load():
    reg = regime_from_dict({"id": "bad_law", "enforcement": {"mode": "forced"},
                            "obligations": [{"trigger": "inspection_overdue", "action": "no_such_action"}]})
    register_regime(reg, overwrite=True)
    with pytest.raises(ValueError):
        _env("bad_law")


def test_authority_executes_actions_outside_scenario_catalogue():
    from infralab.lab.spec import resolve_scenario
    reg = regime_from_dict({"id": "force_l2", "enforcement": {"mode": "forced"},
                            "obligations": [{"trigger": "inspection_overdue", "action": "L2_inspect"}]})
    register_regime(reg, overwrite=True)
    env = build_env_from_scenario(resolve_scenario("EXP-A-BUDGET"), agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"},
                                  param_overrides={"regulatory_regime": "force_l2"}, seed=1)
    assert "L2_inspect" not in env.maint_ops
    for _ in range(12):
        env.step({"A": {"maintenance": {}}})
    st = env.regime_stats["A"]
    assert st["forced_actions"] > 0 and st["forced_unexecutable"] == 0


def test_self_inspect_variant_and_target_keeper():
    env = _env("none")
    ag = BasicAgent("A", policy="danger_only", params={"self_inspect_interval_quarters": 4})
    inspected = 0
    for _ in range(8):
        plan, _ = ag.decide(env)
        inspected += sum(1 for v in plan["maintenance"].values() if v == "L1_inspect")
        env.step({"A": plan})
    assert inspected > 0
    env2 = _env("none", code_framework="EN1990")
    tk = BasicAgent("A", policy="target_keeper")
    acted = 0
    for _ in range(6):
        plan, _ = tk.decide(env2); acted += len(plan["maintenance"]); env2.step({"A": plan})
    assert acted > 0          # below the EN1990 target from the start (D1): the keeper acts



def test_planning_clone_does_not_see_future_failure_draws():
    """Le copie dei pianificatori hanno un flusso di collassi diverso da quello reale."""
    from infralab.lab.spec import resolve_scenario
    real = build_env_from_scenario(resolve_scenario("NBI-DE"), agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"},
                                   param_overrides={"failure_model": "reliability", "failure_consequence_ratio": 3.0}, seed=11)
    real_seq, clone_seqs = [], []
    clones = [real.planning_clone(salt=k) for k in (101, 202, 303)]
    for _ in range(32):
        real.step({"A": {"maintenance": {}}}); real_seq.append(real.failure_stats["A"]["failures"])
    for c in clones:
        seq = []
        for _ in range(32):
            c.step({"A": {"maintenance": {}}}); seq.append(c.failure_stats["A"]["failures"])
        clone_seqs.append(seq)
    assert real.failure_stream_salt == 0
    assert any(seq != real_seq for seq in clone_seqs)       # the clone does not replay the real draws
    # same salt, same draws: planner rollouts stay reproducible
    c1 = real.planning_clone(salt=7); c2 = real.planning_clone(salt=7)
    for _ in range(8):
        c1.step({"A": {"maintenance": {}}}); c2.step({"A": {"maintenance": {}}})
    assert c1.failure_stats == c2.failure_stats


# ---------------------------------------------------- obbligo non finanziabile
def _lock_env(legacy):
    """Legge 'forced' a carico del gestore (owner_budget) con un ripristino dovuto
    ovunque e non pagabile (15.7 M contro 0.2 M di cassa) e l'ispezione periodica
    dovuta: la situazione del ponte 89 di quicklook-crew."""
    reg = regime_from_dict({"id": "unfunded_lock", "enforcement": {"mode": "forced"},
                            "obligations": [{"trigger": "condition_below_threshold", "threshold": 0.99,
                                             "action": "ripristino_completo", "grace_quarters": 0},
                                            {"trigger": "inspection_overdue", "action": "L1_inspect"}],
                            "param_overrides": {"forced_funding": "owner_budget",
                                                "inspection_overdue_grace_quarters": 0}})
    register_regime(reg, overwrite=True)
    return _env("unfunded_lock", regime_override_unfunded=legacy, initial_budget=200000.0)


def test_unfunded_forced_action_does_not_lock_owner_inspection():
    env = _lock_env(legacy=False)
    b = next(iter(env.bridges.values()))
    seen = []
    for _ in range(8):
        env.step({"A": {"maintenance": {str(b.id): "L1_inspect"}}})
        seen.append(b.last_inspection_quarter)
    st = env.regime_stats["A"]
    assert seen == list(range(1, 9))                   # l'ispezione del gestore si fa ogni trimestre
    assert st["overridden_owner_actions"] == 0
    assert st["forced_unfunded"] > 0                   # l'obbligo pagabile resta pendente e contato
    diags = env.last_plan_diagnostics["A"]
    assert any(d.get("reason") == "regime_forced_unfunded" for d in diags)
    assert not any(d.get("status") == "forced_by_regime" and d.get("requested_action") == "ripristino_completo"
                   for d in diags)


def test_unfunded_forced_action_lets_next_obligation_be_forced():
    """Senza piano del gestore, l'ispezione dovuta viene imposta anche se il ripristino
    (primo obbligo) non e' finanziabile."""
    env = _lock_env(legacy=False)
    for _ in range(8):
        env.step({"A": {"maintenance": {}}})
    assert all(b.last_inspection_quarter > 0 for b in env.bridges.values())
    assert env.regime_stats["A"]["forced_actions"] > 0


def test_legacy_flag_reproduces_old_lock():
    env = _lock_env(legacy=True)
    b = next(iter(env.bridges.values()))
    seen = []
    for _ in range(8):
        env.step({"A": {"maintenance": {str(b.id): "L1_inspect"}}})
        seen.append(b.last_inspection_quarter)
    assert set(seen) == {seen[0]}                        # bloccato: nessuna nuova ispezione
    assert env.regime_stats["A"]["overridden_owner_actions"] >= 7
