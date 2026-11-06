"""Copie dei pianificatori e serializzazione: stato completo e osservazioni staccate.

Regressioni di due difetti:
1. Bridge.from_dict non ripristinava il declassamento di portata in corso
   (posting_until_quarter) ne' l'ultimo danno idraulico; BridgeEnv.to_dict non
   portava i contatori (squadre, ispezioni). Nelle copie dei pianificatori e nelle
   sessioni ricaricate un ponte declassato restava limitato per sempre.
2. Bridge/Component.to_dict consegnavano i contenitori vivi del motore: un agente
   che modificava la propria osservazione modificava il motore, e le copie dei
   pianificatori condividevano con l'ambiente reale lo storico delle azioni.
"""
import json

from infralab.agents.basic import BasicAgent
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.persistence import NumpyJSONEncoder
from tests.conftest import load_scenario


def _env(overrides=None, seed=3):
    return build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides=overrides or {}, seed=seed)


def _canon(obj):
    return json.loads(json.dumps(obj, sort_keys=True, cls=NumpyJSONEncoder))


def _run(env, quarters, policy="worst_first"):
    agent = env.params.agent_names[0]
    for _ in range(quarters):
        plan, _ = BasicAgent(agent, policy).decide(env)
        env.step({agent: plan})


# -- 1. stato completo nelle copie ---------------------------------------------

def test_roundtrip_keeps_every_bridge_field():
    env = _env({"posting_entry_rate_q": 0.5, "hydraulic_hazard_enabled": True,
                "rng_streams": "keyed", "ethical_floor_enabled": True})
    _run(env, 6)
    assert any(b.posting_until_quarter for b in env.bridges.values())
    clone = BridgeEnv.from_dict(env.to_dict())
    for bid, b in env.bridges.items():
        assert _canon(b.to_dict()) == _canon(clone.bridges[bid].to_dict()), bid


def test_posting_and_hydraulic_damage_survive_clone_and_reload():
    env = _env()
    b = env.bridges[1]
    b.posting_until_quarter = env.quarter + 2
    b.traffic_restriction_factor = 0.6
    b.hydraulic_damage_quarter = 5
    for other in (env.planning_clone(salt=11), BridgeEnv.from_dict(_canon(env.to_dict()))):
        ob = other.bridges[1]
        assert ob.posting_until_quarter == env.quarter + 2
        assert ob.hydraulic_damage_quarter == 5


def test_posting_expires_in_planning_clone():
    """Il declassamento scade anche nella copia: il fattore di traffico torna pieno."""
    env = _env({"posting_entry_rate_q": 1e-9})    # canale attivo, nuovi declassamenti ~impossibili
    b = env.bridges[1]
    b.posting_until_quarter = env.quarter + 2
    b.traffic_restriction_factor = 0.6
    clone = env.planning_clone(salt=11)
    _run(clone, 4, policy="do_nothing")
    cb = clone.bridges[1]
    assert cb.posting_until_quarter == 0
    assert cb.traffic_restriction_factor == 1.0


def test_counters_survive_serialisation():
    env = _env({"rng_streams": "keyed"})
    agent = env.params.agent_names[0]
    env.crew_stats = {agent: 4}
    env._insp_counter = {(1, 77, env.quarter): 2, (1, 77, env.quarter - 1): 1}
    env.last_expected_loss = {agent: {"expected_loss": 12.5}}
    data = _canon(env.to_dict())                    # passa per JSON come una sessione salvata
    back = BridgeEnv.from_dict(data)
    assert back.crew_stats == {agent: 4}
    assert back.last_expected_loss == {agent: {"expected_loss": 12.5}}
    # solo i contatori del trimestre corrente servono ancora
    assert back._insp_counter == {(1, 77, env.quarter): 2}


# -- 2. osservazioni e copie staccate dal motore ---------------------------------

def _mutate_observation(obs):
    for bd in obs["my_bridges"]:
        bd["action_history"].append({"quarter": -1, "action": "fake", "status": "fake"})
        bd["regime_trigger_since"]["fake"] = -1
        if isinstance(bd.get("reliability_details"), dict):
            det = bd["reliability_details"]
            det["fake"] = -1
            # secondo livello copiato anch'esso (le foglie piu' profonde sono
            # condivise in sola lettura, vedi _details_copy)
            if isinstance(det.get("life_cycle_profile"), dict):
                det["life_cycle_profile"]["fake"] = -1
            if isinstance(det.get("limit_state_results"), list):
                det["limit_state_results"].append({"fake": -1})
        for cd in bd["components"].values():
            cd["current_condition"] = -1.0
            cd["observed_condition_min"] = -1.0
            if isinstance(cd.get("profile"), dict):
                cd["profile"]["scale_mean"] = -1.0
            if isinstance(cd.get("reliability_details"), dict):
                cd["reliability_details"]["target_beta_annual"] = -1.0


def _engine_state(env):
    return _canon({bid: b.to_dict() for bid, b in env.bridges.items()})


def test_agent_cannot_mutate_engine_through_observation():
    for overrides in ({}, {"observation_mode": "inspected"}, {"oracle_observation": True}):
        env = _env(overrides)
        _run(env, 2)
        before = _engine_state(env)
        obs = env.get_agent_observations(env.params.agent_names[0])
        _mutate_observation(obs)
        assert _engine_state(env) == before, overrides


def test_component_and_bridge_dicts_are_copies():
    env = _env()
    b = env.bridges[1]
    comp = next(iter(b.components.values()))
    d = comp.to_dict()
    assert d is not comp.__dict__
    d["current_condition"] = -1.0
    assert comp.current_condition != -1.0
    bd = b.to_dict()
    assert bd["action_history"] is not b.action_history
    assert bd["components"][comp.name] is not comp.__dict__


def test_planning_clone_does_not_leak_into_real_env():
    env = _env()
    _run(env, 2)
    before = _engine_state(env)
    clone = env.planning_clone(salt=5)
    clone._record_action_event(clone.bridges[1], "evento_simulato", "hazard")
    _run(clone, 3)
    assert _engine_state(env) == before
