"""Estrazione delle metriche-esito da una sessione umana persistita.

Calcola gli STESSI esiti che il runner delle campagne produce per gli agenti
computazionali (min_beta_margin, health_index, emissioni, trimestri, fallito),
così le condizioni umane (Uomo, Uomo+LLM) entrano nella stessa tabella di
confronto statistico delle condizioni computazionali (LLM, Euristico, ...).

Definizioni allineate a infralab.agents.protocol (stesso clamp del margine β).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from infralab.agents.protocol import RewardConfig, _health_index, _min_beta_margin


def session_run_metrics(env, agent_name: str, condition: str,
                        horizon_quarters: Optional[int] = None,
                        session_id: Optional[str] = None) -> Dict[str, Any]:
    """Riga run-level (come una run di campagna) dallo stato finale di una partita.

    `agent_id` è impostato alla CONDIZIONE sperimentale, così il confronto
    raggruppa per condizione a prescindere dal nome del partecipante.
    """
    clamp = RewardConfig().beta_margin_clamp
    horizon = horizon_quarters or int(env.params.max_quarters or env.params.years * 4)
    quarters = min(env.quarter - 1, horizon)          # trimestri effettivamente giocati
    terminated = agent_name not in env.active_agents   # fallito (bancarotta/collasso)
    assets = [a for a in env.bridges.values() if a.manager == agent_name]
    total_adt = sum(a.adt for a in assets) or 1
    nav = sum(env.get_net_asset_value(a) for a in assets)
    return {
        "agent_id": condition,
        "condition": condition,
        "player_name": agent_name,
        "session_id": session_id,
        "quarters_survived": int(quarters),
        "terminated": bool(terminated),
        "final_min_beta_margin": round(_min_beta_margin(env, agent_name, clamp), 4),
        "final_health_index": round(_health_index(env, agent_name), 4),
        "final_emissions_tco2": round(float(getattr(env, "emissions_tco2", {}).get(agent_name, 0.0)), 3),
        "final_net_asset_value": round(float(nav), 2),
        "final_budget": round(float(env.budgets.get(agent_name, 0.0)), 2),
    }


# Esiti confrontabili fra condizioni umane e computazionali (chiave, etichetta, campo run).
COMPARABLE_OUTCOMES = [
    ("final_min_beta_margin", "Margine β minimo", "final_min_beta_margin"),
    ("final_health_index", "Health index", "final_health_index"),
    ("final_emissions_tco2", "Emissioni (tCO₂)", "final_emissions_tco2"),
    ("quarters_survived", "Trimestri sopravvissuti", "quarters_survived"),
]


def aggregate_by_condition(runs: list) -> Dict[str, Any]:
    """Confronto statistico fra condizioni sugli esiti confrontabili.

    runs: righe run-level (umane e/o computazionali) con 'condition'/'agent_id'.
    """
    from infralab.study.stats import compare_conditions

    conditions = sorted({r.get("condition") or r.get("agent_id") for r in runs})
    metrics_out = []
    for key, label, field in COMPARABLE_OUTCOMES:
        groups = {c: [r.get(field) for r in runs
                      if (r.get("condition") or r.get("agent_id")) == c and r.get(field) is not None]
                  for c in conditions}
        groups = {c: v for c, v in groups.items() if v}
        if len(groups) >= 1:
            metrics_out.append({"key": key, "label": label, **compare_conditions(groups)})
    return {"conditions": conditions, "metrics": metrics_out}
