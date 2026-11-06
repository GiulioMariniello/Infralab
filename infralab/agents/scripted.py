"""Agente definito da codice utente (plugin `decide(obs)`).

Costruisce l'osservazione compatta dall'ambiente (stessa forma dei tool
agentici: per asset condizioni, beta, azioni fattibili con costo), chiama la
funzione utente con un timeout e mappa il piano restituito nel formato del
motore. Errori/timeout ⇒ nessun intervento (l'agente non blocca la partita).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from infralab.agents.legacy import AgentInterface
from infralab.plugins.loader import PluginError, call_with_timeout, load_entrypoint

DECIDE_TIMEOUT_S = 8.0


def compact_observation(obs: Dict[str, Any], digits: Tuple[int, int, int] = (4, 3, 2)) -> Dict[str, Any]:
    """Osservazione compatta costruita dall'osservazione del gestore (policy_sandbox):
    stessa forma di quella storica, ma con i valori che il gestore conosce (in
    "inspected" condizioni, beta e Pf stimati dall'evidenza)."""
    dc, db, dm = digits
    assets = []
    for b in obs["my_bridges"]:
        feas = b.get("action_feasibility") or {}
        beta = b.get("reliability_index")
        assets.append({
            "id": b["id"],
            "name": b.get("name"),
            "condition_min": round(float(b["min_condition"]), dc),
            "beta": None if beta is None else round(float(beta), db),
            "annual_pf": b.get("annual_failure_probability"),
            "health_index": round(float(b.get("health_index", 0.0)), dc),
            "net_asset_value": round(float(b.get("net_asset_value", 0.0)), dm),
            "components": {n: round(float(c["current_condition"]), dc) for n, c in (b.get("components") or {}).items()},
            "feasible_actions": {a: round(float(i["cost"]), dm) for a, i in feas.items() if i.get("feasible")},
        })
    return {
        "quarter": obs["game_rules"]["quarter"],
        "budget": round(float(obs["game_rules"]["my_budget"]), 2),
        "emissions_tco2": round(float(obs["game_rules"].get("my_emissions_tco2", 0.0) or 0.0), 3),
        "assets": assets,
    }


class ScriptedPluginAgent(AgentInterface):
    """Agente il cui `decide` è codice utente (da plugin salvato o inline).

    `plugin`: nome del plugin agente salvato (registry). In alternativa `code`:
    sorgente inline. `decider`: callable già caricata (per i test).
    """

    def __init__(self, name: str, plugin: Optional[str] = None, code: Optional[str] = None,
                 decider: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        self.name = name
        self.plugin = plugin
        self._decider = decider
        if self._decider is None:
            if code is not None:
                self._decider = load_entrypoint(code, "decide")
            elif plugin is not None:
                from infralab.plugins.registry import load_agent_decider
                self._decider = load_agent_decider(plugin)
            else:
                raise PluginError("ScriptedPluginAgent richiede 'plugin', 'code' o 'decider'.")

    def _observation(self, env) -> Dict[str, Any]:
        from infralab.agents.sandbox import is_sandboxed
        if is_sandboxed(env):
            return compact_observation(env.get_agent_observations(self.name))
        assets = []
        for b in env.bridges.values():
            if b.manager != self.name:
                continue
            feas = env.get_feasible_actions_for_bridge(self.name, b)
            assets.append({
                "id": b.id,
                "name": b.name,
                "condition_min": round(b.min_condition, 4),
                "beta": None if b.reliability_index is None else round(b.reliability_index, 3),
                "annual_pf": b.annual_failure_probability,
                "health_index": round(env.get_health_index(b), 4),
                "net_asset_value": round(env.get_net_asset_value(b), 2),
                "components": {n: round(c.current_condition, 4) for n, c in b.components.items()},
                "feasible_actions": {a: round(float(i["cost"]), 2) for a, i in feas.items() if i["feasible"]},
            })
        return {
            "quarter": env.quarter,
            "budget": round(float(env.budgets.get(self.name, 0.0)), 2),
            "emissions_tco2": round(float(getattr(env, "emissions_tco2", {}).get(self.name, 0.0)), 3),
            "assets": assets,
        }

    def decide(self, env, **kwargs) -> Tuple[Dict[str, Any], Any]:
        obs = self._observation(env)
        try:
            raw = call_with_timeout(self._decider, obs, DECIDE_TIMEOUT_S)
        except PluginError as e:
            return {}, ("scripted", f"Agente in errore: {e}")
        if not isinstance(raw, dict):
            return {}, ("scripted", f"decide() deve restituire un dict, non {type(raw).__name__}")
        # Tiene solo le azioni fattibili per gli asset dell'agente.
        valid_actions = {a["id"]: set(a["feasible_actions"]) for a in obs["assets"]}
        plan: Dict[str, str] = {}
        for k, v in raw.items():
            try:
                aid = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, str) and v and v in valid_actions.get(aid, ()):  # ignora azioni non fattibili
                plan[str(aid)] = v
        return {"maintenance": plan}, ("scripted", {"obs_assets": len(obs["assets"]), "actions": len(plan)})
