"""Runner delle campagne: espande la spec in run deterministiche e le esegue.

Ogni run = (agente, punto griglia, replica) con un seed figlio derivato dal
seed radice via numpy SeedSequence: aggiungere run non cambia i seed delle
precedenti. Output tidy: una riga per run (runs) + una riga per trimestre
(steps).
"""
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from infralab.agents.adapters import make_agent
from infralab.agents.protocol import AssetEnvGym, RewardConfig
from infralab.lab.build import build_env_from_scenario
from infralab.lab.spec import ExperimentSpec, resolve_scenario

logger = logging.getLogger(__name__)

_GRID_PREFIX = "params."


def expand_runs(spec: ExperimentSpec, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lista deterministica dei payload di run (ordinamento stabile)."""
    root = np.random.SeedSequence(spec.seed)
    paired = getattr(spec, "seed_scope", "run") == "replicate"
    # seed_scope="run": un figlio per run (storico). "replicate": un figlio per
    # replica, condiviso da tutti gli agenti e punti griglia (disegno appaiato).
    children = root.spawn(spec.replicates if paired else spec.run_count)
    payloads = []
    index = 0
    for agent_spec in spec.agents:
        for grid_point in spec.grid_points():
            for replicate in range(spec.replicates):
                child = children[replicate] if paired else children[index]
                child_seed = int(child.generate_state(1)[0] % (2**31 - 1))
                payloads.append({
                    "run_id": index,
                    "agent_id": agent_spec.id,
                    "agent_config": agent_spec.to_agent_config(),
                    "grid_point": grid_point,
                    "replicate": replicate,
                    "seed": child_seed,
                    "scenario": scenario,
                    "horizon_quarters": spec.horizon_quarters,
                    "reward": spec.reward,
                })
                index += 1
    return payloads


# Colonne di collapses.* (una riga per collasso), dopo quelle che identificano la run.
COLLAPSE_COLUMNS = ("quarter", "bridge_id", "bridge_name", "agent", "cause", "annual_pf", "quarterly_p",
                    "observed_min_condition", "assessed_min_condition", "assessed_annual_beta",
                    "last_inspection_quarter", "known_defects", "floor_trigger_active", "known_critical",
                    "hidden_damage_pending", "hidden_defects_pending", "true_min_condition",
                    "traffic_restriction_factor", "closed", "floor_closure", "regime_sanction", "posting_active",
                    "works_active", "defect_start_quarter", "defect_known_since")


def _collapse_rows(env, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Registro dei collassi della run (env.collapse_log) in righe piatte."""
    rows = []
    for rec in (getattr(env, "collapse_log", None) or []):
        row = {**base, **{k: rec.get(k) for k in COLLAPSE_COLUMNS}}
        # difetti noti all'inizio del trimestre: "tipo@trimestre di scoperta" separati da ";"
        row["known_defects"] = ";".join(f"{d.get('type')}@{d.get('known_since')}"
                                        for d in rec.get("known_defects") or [])
        rows.append(row)
    return rows


def run_one(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Esegue una singola run e restituisce (riga run, righe step)."""
    run_row, steps, _collapses = run_one_full(payload)
    return run_row, steps


def run_one_full(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Come run_one, piu' le righe del registro dei collassi (riga run, righe step,
    righe collasso)."""
    param_overrides = {
        key[len(_GRID_PREFIX):]: value for key, value in payload["grid_point"].items()
    }
    reward_config = RewardConfig(**payload["reward"]) if payload["reward"] else RewardConfig()

    def env_factory(seed):
        return build_env_from_scenario(
            payload["scenario"],
            agent_name=payload["agent_id"],
            agent_config=payload["agent_config"],
            param_overrides=param_overrides,
            seed=seed,
        )

    gym = AssetEnvGym(
        env_factory,
        payload["agent_id"],
        reward_config=reward_config,
        max_quarters=payload["horizon_quarters"],
    )
    obs, info = gym.reset(seed=payload["seed"])
    agent = make_agent(payload["agent_id"], payload["agent_config"])
    # L'adapter e' harness: riceve il motore anche con policy_sandbox (la policy no).
    agent.reset(obs, {**info, "env": gym.harness_env()})

    base = {
        "run_id": payload["run_id"],
        "agent_id": payload["agent_id"],
        "replicate": payload["replicate"],
        "seed": payload["seed"],
        **payload["grid_point"],
    }
    steps: List[Dict[str, Any]] = []
    total_reward = 0.0
    terminated = truncated = False
    t_start = time.perf_counter()
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = gym.step(action)
        total_reward += reward
        steps.append({
            **base,
            "quarter": info["quarter"] - 1,
            "reward": reward,
            **{f"metric_{k}": v for k, v in info["metrics"].items()},
        })

    failure = info.get("failure_info") or {}
    run_row = {
        **base,
        "quarters_survived": len(steps),
        "duration_s": round(time.perf_counter() - t_start, 3),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "failure_reason": failure.get("reason") if terminated else None,
        **{f"final_{k}": v for k, v in info["metrics"].items()},
    }
    return run_row, steps, _collapse_rows(gym.harness_env(), base)


def _run_one_safe(payload: Dict[str, Any]):
    try:
        return run_one_full(payload), None
    except Exception:
        return None, {"run_id": payload["run_id"], "agent_id": payload["agent_id"],
                      "error": traceback.format_exc()}


def run_experiment(
    spec: ExperimentSpec,
    workers: int = 1,
    scenario: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Esegue l'intera campagna. Restituisce {runs, steps, collapses, errors}."""
    scenario = scenario or resolve_scenario(spec.scenario)
    payloads = expand_runs(spec, scenario)
    runs: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    collapses: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def consume(result, error):
        if error is not None:
            errors.append(error)
            logger.error("Run %s fallita:\n%s", error["run_id"], error["error"])
        else:
            run_row, step_rows, collapse_rows = result
            runs.append(run_row)
            steps.extend(step_rows)
            collapses.extend(collapse_rows)
        if progress_callback:
            progress_callback(len(runs) + len(errors), len(payloads))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result, error in pool.map(_run_one_safe, payloads):
                consume(result, error)
    else:
        for payload in payloads:
            result, error = _run_one_safe(payload)
            consume(result, error)

    runs.sort(key=lambda row: row["run_id"])
    steps.sort(key=lambda row: (row["run_id"], row["quarter"]))
    collapses.sort(key=lambda row: (row["run_id"], row["quarter"], row["bridge_id"]))
    return {"runs": runs, "steps": steps, "collapses": collapses, "errors": errors}
