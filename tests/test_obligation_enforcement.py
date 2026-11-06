"""Modalita' di applicazione per singolo obbligo (opt-in; docs/CHANGELOG_MODEL.md).
Senza il campo `enforcement` sull'obbligo vale la modalita' della legge, come sempre."""
import pytest

from infralab.agents.adapters import make_agent
from infralab.lab.build import build_env_from_scenario
from infralab.regimes import regime_from_dict, register_regime
from tests.conftest import load_scenario

SCN = "PAPERA-SHORT-A"
LADDER = [{"after_quarters": 1, "action": "limitazione_carico"}]
ISPEZIONE = {"trigger": "inspection_overdue", "action": "L1_inspect"}
# obbligo di intervento sempre dovuto: probabilita' di collasso sopra una soglia nulla
INTERVENTO = {"trigger": "pf_above_threshold", "threshold": 0.0, "action": "ripristino_strutturale"}


def _regime(rid, obligations, mode):
    return register_regime(regime_from_dict({
        "id": rid, "obligations": obligations,
        "enforcement": {"mode": mode, "sanction_ladder": LADDER, "lift_on_compliance": True}})) or rid


def _run(rid, quarters=8):
    env = build_env_from_scenario(load_scenario(SCN), agent_name="A", agent_config={"type": "Basic", "policy": "do_nothing"},
                                  param_overrides={"regulatory_regime": rid}, seed=1)
    ag = make_agent("A", {"type": "Basic", "policy": "do_nothing"})
    ag.reset(env.get_agent_observations("A"), {"env": env})
    for _ in range(quarters):
        env.step({"A": ag.act(env.get_agent_observations("A"))})
    return env


def test_field_is_validated_and_defaults_to_the_law_mode():
    r = regime_from_dict({"id": "x1", "obligations": [ISPEZIONE], "enforcement": {"mode": "forced"}})
    assert r.obligations[0].enforcement is None and r.obligation_forced(r.obligations[0])
    assert "enforcement" not in r.to_dict()["obligations"][0]          # serializzazione storica invariata
    with pytest.raises(ValueError):
        regime_from_dict({"id": "x2", "obligations": [{**ISPEZIONE, "enforcement": "altro"}]})


def test_law_properties_see_the_obligation_modes():
    r = regime_from_dict({"id": "x3", "obligations": [{**ISPEZIONE, "enforcement": "forced"}],
                          "enforcement": {"mode": "sanction", "sanction_ladder": LADDER}})
    assert r.forces and r.sanctions and r.enforces
    assert r.obligation_forced(r.obligations[0]) and not r.obligation_sanctioned(r.obligations[0])
    assert r.to_dict()["obligations"][0]["enforcement"] == "forced"


def test_mixed_law_forces_the_inspection_and_sanctions_the_intervention():
    _regime("mista", [{**ISPEZIONE, "enforcement": "forced"}, {**INTERVENTO, "enforcement": "sanction"}], "none")
    env = _run("mista")
    st = env.regime_stats["A"]
    # l'ispezione e' eseguita d'ufficio, l'intervento mai
    assert st["forced_actions"] > 0
    assert all(b.last_inspection_quarter > 0 for b in env.bridges.values())
    assert not any(ev.get("action") == "ripristino_strutturale" or ev.get("action_name") == "ripristino_strutturale"
                   for b in env.bridges.values() for ev in b.action_history)
    # l'obbligo di intervento inadempiuto porta la limitazione di carico
    assert st["sanction_bridge_quarters"] > 0
    assert all(b.regime_sanction and b.regime_sanction["trigger"] == "pf_above_threshold" for b in env.bridges.values())


def test_forced_only_obligation_never_moves_the_sanction_ladder():
    """Legge a sanzione con il solo obbligo d'ispezione dichiarato 'forced': nessuna sanzione."""
    _regime("solo_ufficio", [{**ISPEZIONE, "enforcement": "forced"}], "sanction")
    env = _run("solo_ufficio")
    assert env.regime_stats["A"]["forced_actions"] > 0
    assert env.regime_stats["A"]["sanction_bridge_quarters"] == 0
    assert not any(b.regime_sanction for b in env.bridges.values())


def test_without_the_field_behaviour_is_the_historic_one():
    """Stessa legge scritta con e senza il campo (uguale alla modalita' della legge): stesse traiettorie."""
    _regime("storica", [ISPEZIONE], "forced")
    _regime("esplicita", [{**ISPEZIONE, "enforcement": "forced"}], "forced")
    a, b = _run("storica"), _run("esplicita")
    assert a.regime_stats["A"] == b.regime_stats["A"]
    assert [x.min_condition for x in a.bridges.values()] == [x.min_condition for x in b.bridges.values()]
    assert a.budgets == b.budgets
