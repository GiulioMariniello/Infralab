"""Costruzione dell'ambiente da uno scenario (stessa logica del flusso app.py)."""
import copy
from dataclasses import asdict
from typing import Any, Dict, Optional

from infralab.calibration_profiles import apply_calibration_profile
from infralab.core.environment import AssetEnv
from infralab.settings.config import CdaConfig, MaintOpsConfig, SimParameters


def build_env_from_scenario(
    scenario: Dict[str, Any],
    agent_name: str = "Agent",
    agent_config: Optional[Dict[str, Any]] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    agents: Optional[Dict[str, Dict[str, Any]]] = None,
) -> AssetEnv:
    """Costruisce un AssetEnv da uno scenario, replicando app.py player_init.

    param_overrides: chiavi di SimParameters da forzare (dopo scenario e
    profilo di calibrazione). seed: se non None, sovrascrive il seed.
    agents: mappa {nome: config} per sessioni multi-agente; se assente si usa
    il singolo (agent_name, agent_config). Gli asset senza 'manager' esplicito
    nello scenario vengono assegnati round-robin tra gli agenti.
    """
    scenario = copy.deepcopy(scenario)
    base_params = asdict(SimParameters())
    base_params.update(scenario.get("params", {}))
    if scenario.get("max_quarters"):
        base_params["max_quarters"] = scenario["max_quarters"]
    if param_overrides:
        base_params.update(param_overrides)
    if seed is not None:
        base_params["seed"] = seed
    params = SimParameters(**base_params)

    if agents:
        agent_config_map = {name: dict(cfg) for name, cfg in agents.items()}
    else:
        agent_config_map = {agent_name: agent_config or {"type": "Euristico"}}
    params.agent_names = list(agent_config_map)
    params.num_agents = len(agent_config_map)

    maint_ops = asdict(MaintOpsConfig())
    params, maint_ops = apply_calibration_profile(params, maint_ops, params.calibration_profile)
    # Operazioni specifiche dell'asset type (es. interventi sui muri di sostegno).
    from infralab.assets import get_asset_type

    asset_type = scenario.get("asset_type", "bridge")
    spec = get_asset_type(asset_type)
    maint_ops = {**maint_ops, **copy.deepcopy(spec.maintenance_ops)}
    allowed = scenario.get("allowed_actions")
    if allowed:
        maint_ops = {k: v for k, v in maint_ops.items() if k in allowed}

    env = AssetEnv(
        params,
        maint_ops,
        agent_config_map,
        CdaConfig(),
        {},
        asset_type=asset_type,
    )
    env.initialize_bridges(scenario["bridges"])
    return env
