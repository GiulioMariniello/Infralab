"""Fughe d'informazione dal canale del danno locale improvviso e controllo a traiettoria.

RT009: la fotografia da cui il PlanningHandle genera le copie dei pianificatori
(`belief_snapshot`) e la copia "belief" del motore non devono contenere i danni
improvvisi non ancora rilevati (`sudden_pending`: componente, trimestre, perdita vera)
ne' i contatori veri del canale (`sudden_stats`: eventi avvenuti, perdita totale...).
RT010: `perturb_hidden_truth` inietta danni improvvisi nascosti diversi nel mondo B;
`--sudden-hazards` accende una configurazione di prova del canale.
RT011: modalita' traiettoria (`--trajectory`): verita' perturbata una volta, due mondi
che avanzano insieme con le stesse letture d'ispezione; rileva le fughe che si
accumulano nello stato del gestore, che il controllo istante per istante non vede.
"""
import json

import numpy as np
import pytest

import infralab.agents.sandbox as sandbox
from infralab.agents.sandbox import PlanningHandle, belief_snapshot
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.lab.leak_check import (TEST_HAZARD_PARAMS, compare_worlds, leak_check_env, main as leak_main,
                                     perturb_hidden_sudden, perturb_hidden_truth, run_leak_check,
                                     step_recording, step_replaying)
from tests.conftest import load_scenario

BASE = {"observation_mode": "inspected", "policy_sandbox": True, "rng_streams": "keyed",
        "strict_information": True}
HAZARDS = {"impact_hazard_enabled": True, "fatigue_hazard_enabled": True, "impact_damage_rate_q": 0.3,
           "impact_adt_exponent": 0.0, "fatigue_damage_rate_q": 0.3, "impact_damage_announce_prob": 0.0,
           "hydraulic_legacy_mode": False}


def _env(extra=None, seed=3):
    return build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides={**BASE, **(extra or {})}, seed=seed)


def _env_with_pending(extra=None):
    env = _env({**HAZARDS, **(extra or {})})
    for _ in range(6):
        if env.sudden_pending:
            break
        env.step({a: {"maintenance": {}} for a in env.params.agent_names})
    assert env.sudden_pending, "nessun danno improvviso nascosto: tassi di prova troppo bassi"
    return env


def _copy(env):
    w = BridgeEnv.from_dict(env.to_dict(), refresh=False)
    w.refresh_network_reliability()
    return w


# -- RT009: redazione --------------------------------------------------------------------

def test_belief_snapshot_redacts_sudden_pending_and_stats():
    env = _env_with_pending()
    agent = env.params.agent_names[0]
    assert env.sudden_stats[agent]["events_total"] > 0
    snap = belief_snapshot(env, agent, seed=5)
    assert "sudden_pending" not in snap and "sudden_stats" not in snap
    handle = PlanningHandle.build(env, agent, seed=5)
    assert "sudden_pending" not in handle._snapshot
    clone = handle.belief_clone(salt=1, rng=np.random.default_rng(1))
    assert clone.sudden_pending == {}
    assert all(v == 0 for v in clone.sudden_stats[agent].values())


def test_engine_belief_clone_resets_true_sudden_counters():
    env = _env_with_pending({"policy_sandbox": False, "planner_information": "belief"})
    agent = env.params.agent_names[0]
    clone = env.belief_clone(salt=1, rng=np.random.default_rng(1))
    assert clone.sudden_pending == {}
    assert all(v == 0 for v in clone.sudden_stats[agent].values())
    # condizione vera (con il danno nascosto) riestratta dall'evidenza: il componente
    # danneggiato non porta la perdita vera nella copia
    bid, evs = next(iter(env.sudden_pending.items()))
    comp = evs[0]["component"]
    true_c = env.bridges[bid].components[comp].current_condition
    lo = env.bridges[bid].components[comp].observed_condition_min
    hi = env.bridges[bid].components[comp].observed_condition_max
    c = clone.bridges[bid].components[comp].current_condition
    if lo is not None and hi is not None:
        assert lo - 0.2 <= c <= hi + 0.2
    assert c != pytest.approx(true_c)


def test_truth_views_keep_sudden_state():
    # oracolo / "full": lo stato vero e' concesso per dichiarazione, niente redazione
    env = _env_with_pending({"oracle_observation": True})
    snap = belief_snapshot(env, env.params.agent_names[0], seed=5)
    assert snap.get("sudden_pending")


def test_paired_worlds_same_evidence_different_sudden_truth():
    env = _env_with_pending()
    world_a, world_b = _copy(env), _copy(env)
    perturb_hidden_sudden(world_b, np.random.default_rng(7))
    world_b.refresh_network_reliability()
    assert world_a.to_dict().get("sudden_pending") != world_b.to_dict().get("sudden_pending")
    diffs = compare_worlds(world_a, world_b,
                           include=("observation", "belief_clone", "planner_snapshot", "classification",
                                    "floor_trigger", "regime_triggers"))
    assert not diffs, {k: v[:3] for k, v in diffs.items()}


def test_planner_snapshot_output_flags_unredacted_snapshot(monkeypatch):
    """Regressione di RT009: se la fotografia tornasse a portare sudden_pending, il
    controllo (entrambe le modalita') lo segnala."""
    original = sandbox.belief_snapshot

    def leaky(env, agent_name, seed):
        data = original(env, agent_name, seed)
        full = env.to_dict()
        for key in ("sudden_pending", "sudden_stats"):
            if key in full:
                data[key] = full[key]
        return data

    monkeypatch.setattr(sandbox, "belief_snapshot", leaky)
    for trajectory in (False, True):
        report = leak_check_env(_env(HAZARDS), quarters=2, policy="inspect-only", trajectory=trajectory)
        assert not report["clean"]
        assert report["divergences_by_output"].get("planner_snapshot")


# -- RT010: danni improvvisi nel mondo B ------------------------------------------------------

def test_perturb_hidden_truth_injects_sudden_damage_with_same_evidence():
    env = _env()
    world_b = perturb_hidden_truth(env, seed=1)
    assert sorted(world_b.sudden_pending) == sorted(env.bridges)
    for bid, bridge in env.bridges.items():
        for name, comp in bridge.components.items():
            cb = world_b.bridges[bid].components[name]
            assert (cb.observed_condition_min, cb.observed_condition_max) == \
                   (comp.observed_condition_min, comp.observed_condition_max)
            assert cb.epistemic_uncertainty == comp.epistemic_uncertainty
    assert not perturb_hidden_truth(env, seed=1, sudden=False).sudden_pending


def test_perturb_min_condition_never_pushes_truth_below_floor():
    env = _env()
    world_b = perturb_hidden_truth(env, seed=5, damage=(0.5, 0.6), min_condition=0.6)
    for bid, bridge in env.bridges.items():
        for name, comp in bridge.components.items():
            cb = world_b.bridges[bid].components[name].current_condition
            assert cb >= min(comp.current_condition, 0.6) - 1e-9


def test_perturb_uses_active_types_and_counts_events():
    env = _env(HAZARDS)
    world_b = perturb_hidden_truth(env, seed=2)
    types = {ev["type"] for evs in world_b.sudden_pending.values() for ev in evs}
    assert types <= {"impact", "fatigue"} and types
    agent = env.params.agent_names[0]
    assert world_b.sudden_stats[agent]["events_total"] >= len(env.bridges)


def test_sudden_hazards_configuration_and_clean_gate():
    report = run_leak_check("TRAIN-01", params=BASE, seed=3, quarters=3, sudden_hazards=True)
    for k, v in TEST_HAZARD_PARAMS.items():
        assert report["params"][k] == v
    assert set(report["sudden_channel_types"]) == {"hydraulic", "impact", "vessel", "fatigue"}
    assert report["sudden_injection"] is True
    assert report["clean"], report["divergences"][:5]


# -- RT011: traiettoria -----------------------------------------------------------------------

def test_replay_keeps_readings_identical_with_different_truth():
    env = _env(HAZARDS)
    world_b = perturb_hidden_truth(env, seed=4)
    plan = {a: {"maintenance": {str(b.id): "L1_inspect" for b in env.bridges.values()}}
            for a in env.params.agent_names}
    tape = step_recording(env, json.loads(json.dumps(plan)))
    unmatched = step_replaying(world_b, json.loads(json.dumps(plan)), tape)
    assert tape and not unmatched
    same, differ = 0, 0
    for bid, bridge in env.bridges.items():
        for name, comp in bridge.components.items():
            cb = world_b.bridges[bid].components[name]
            same += (cb.observed_condition_min, cb.observed_condition_max) == \
                    (comp.observed_condition_min, comp.observed_condition_max)
            differ += abs(cb.current_condition - comp.current_condition) > 1e-6
    assert same == sum(len(b.components) for b in env.bridges.values())
    assert differ > 0


def test_trajectory_clean_under_strict_with_accumulating_truth_gap():
    report = run_leak_check("TRAIN-01", params={**BASE, **HAZARDS}, seed=3, quarters=5,
                            policy="inspect-only", trajectory=True)
    assert report["mode"] == "trajectory"
    assert report["clean"], report["divergences"][:5]
    assert report["quarters_checked"] >= 3
    gaps = [g["condition_gap_mean"] for g in report["truth_gap"]]
    assert gaps[-1] > 0.0


def test_trajectory_catches_accumulating_leak_that_instant_mode_misses(monkeypatch):
    """Fuga sintetica che si accumula: a ogni trimestre il motore scrive nella
    conoscenza del gestore (incertezza epistemica) un valore derivato dalla verita'.
    Il controllo istante per istante perturba la verita' DOPO il passo (conoscenza
    uguale nei due mondi) e non la vede; la traiettoria si'."""
    original = BridgeEnv.step

    def leaky_step(self, plans):
        out = original(self, plans)
        for bridge in self.bridges.values():
            for comp in bridge.components.values():
                comp.epistemic_uncertainty = 0.05 + 0.1 * float(comp.current_condition)
        return out

    monkeypatch.setattr(BridgeEnv, "step", leaky_step)
    instant = leak_check_env(_env(), quarters=3, policy="inspect-only", include=("observation",))
    assert instant["clean"]
    traj = leak_check_env(_env(), quarters=3, policy="inspect-only", include=("observation",), trajectory=True)
    assert not traj["clean"]
    assert any("epistemic_uncertainty" in r["path"] for r in traj["divergences"])


def test_trajectory_flags_default_information_mode():
    report = run_leak_check("TRAIN-01", params={**BASE, "strict_information": False}, seed=3, quarters=3,
                            policy="inspect-only", trajectory=True)
    assert not report["clean"]


def test_cli_trajectory_and_sudden_flags(tmp_path, capsys):
    out = tmp_path / "t.json"
    args = ["--scenario", "TRAIN-01", "--quarters", "3", "--seed", "3", "--trajectory", "--sudden-hazards",
            "--policy", "inspect-only", "--param", "observation_mode=inspected", "--param", "policy_sandbox=true",
            "--param", "rng_streams=keyed", "--param", "strict_information=true", "--json", str(out)]
    assert leak_main(args) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["mode"] == "trajectory" and rep["clean"] is True and rep["truth_gap"]
    assert "modo=trajectory" in capsys.readouterr().out
    assert leak_main(args[:-2] + ["--no-sudden-injection"]) == 0
