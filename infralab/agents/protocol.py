"""Protocollo agente gym-like e wrapper single-agent dell'ambiente.

Non introduce una dipendenza da gymnasium: ne replica la forma
(reset/step/reward/terminated) su spazi a dizionario, sufficiente per
benchmarking di policy euristiche, di ottimizzazione, RL e LLM.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

from infralab.core.hazards.sudden_damage import metrics as _sudden_metrics
from infralab.core.hazards.regional_events import metrics as _regional_metrics
from infralab.core.hazards import progressive_events as _progressive


@runtime_checkable
class Agent(Protocol):
    """Contratto minimo di una policy decisionale.

    `act` riceve l'osservazione (dict di get_agent_observations) e restituisce
    il piano azioni {"maintenance": {asset_id_str: action_name}}.
    """

    name: str

    def reset(self, observation: Dict[str, Any], info: Dict[str, Any]) -> None:
        ...

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class RewardConfig:
    """Obiettivo configurabile: combinazione pesata di KPI di gestione.

    reward = w_budget · Δbudget/budget_scale
           + w_health · Δhealth_index
           + w_beta   · Δ(margine beta annuale minimo, clampato)
           − penalita' se collasso / bancarotta
    """

    budget_weight: float = 0.2
    health_weight: float = 1.0
    beta_margin_weight: float = 1.0
    budget_scale: float = 1_000_000.0
    beta_margin_clamp: float = 3.0
    collapse_penalty: float = 50.0
    bankruptcy_penalty: float = 25.0


def seed_globals(seed: Optional[int]) -> None:
    """Seeda `random` e `numpy.random` (RNG globali) da un seed intero; None = entropia."""
    import random as _random

    import numpy as _np
    if seed is None:
        _random.seed()
        _np.random.seed()
    else:
        _random.seed(int(seed))
        _np.random.seed(int(seed) % (2 ** 32))


def _min_beta_margin(env, agent_name: str, clamp: float) -> float:
    margins = []
    for asset in env.bridges.values():
        if asset.manager != agent_name:
            continue
        beta = (asset.posterior_annual_reliability_index
                if asset.posterior_annual_reliability_index is not None
                else asset.annual_reliability_index)
        details = asset.reliability_details or {}
        target = details.get("target_beta_annual")
        if beta is None or target is None:
            continue
        margins.append(max(-clamp, min(clamp, beta - target)))
    return min(margins) if margins else 0.0


def _reliability_snapshot(env, agent_name: str) -> Dict[str, float]:
    """KPI di affidabilità di rete per l'agente (stessa convenzione di persistence/arena).

    min_beta      : β minimo (periodo di riferimento, posterior se disponibile) fra gli asset
    min_beta_annual: β annuale minimo (posterior se disponibile)
    pf_net        : massima probabilità di collasso fra gli asset
    assets_below_target: n. asset con β annuale < target annuale di normativa
    beta_target_annual : target annuale (dell'asset con β minimo; None se non definito)
    asset_value   : valore patrimoniale complessivo
    """
    assets = [a for a in env.bridges.values() if a.manager == agent_name]
    if not assets:
        return {"min_beta": 0.0, "min_beta_annual": 0.0, "pf_net": 0.0,
                "assets_below_target": 0, "beta_target_annual": None, "asset_value": 0.0}
    betas, betas_ann, pfs, below, target_of_min = [], [], [], 0, None
    worst = None
    for a in assets:
        b_ref = a.posterior_reliability_index if a.posterior_reliability_index is not None else a.reliability_index
        b_ann = (a.posterior_annual_reliability_index
                 if a.posterior_annual_reliability_index is not None else a.annual_reliability_index)
        pf = a.posterior_failure_probability if a.posterior_failure_probability is not None else a.failure_probability
        target = (a.reliability_details or {}).get("target_beta_annual")
        if b_ref is not None:
            betas.append(float(b_ref))
        if b_ann is not None:
            betas_ann.append(float(b_ann))
            if target is not None and b_ann < target:
                below += 1
            if worst is None or b_ann < worst:
                worst, target_of_min = b_ann, target
        if pf is not None:
            pfs.append(float(pf))
    return {
        "min_beta": min(betas) if betas else 0.0,
        "min_beta_annual": min(betas_ann) if betas_ann else 0.0,
        "pf_net": max(pfs) if pfs else 0.0,
        "assets_below_target": below,
        "beta_target_annual": float(target_of_min) if target_of_min is not None else None,
        "asset_value": float(sum(getattr(a, "current_asset_value", 0.0) or 0.0 for a in assets)),
    }


def _overdue_assets(env, agent_name: str) -> int:
    """Asset dell'agente con ispezione scaduta oltre la franchigia (Linee Guida)."""
    grace = getattr(env.params, "inspection_overdue_grace_quarters", 0) or 0
    return sum(1 for a in env.bridges.values()
               if a.manager == agent_name and env.quarter > a.next_inspection_quarter + grace)


def _health_index(env, agent_name: str) -> float:
    assets = [a for a in env.bridges.values() if a.manager == agent_name]
    total_adt = sum(a.adt for a in assets)
    if not assets or total_adt <= 0:
        return 0.0
    return sum(a.min_condition * a.adt for a in assets) / total_adt


class AssetEnvGym:
    """Vista single-agent gym-like su AssetEnv.

    L'ambiente sottostante viene ricreato ad ogni reset tramite `env_factory`
    (che riceve il seed), cosi' ogni episodio e' indipendente e riproducibile.
    """

    def __init__(
        self,
        env_factory: Callable[[Optional[int]], Any],
        agent_name: str,
        reward_config: RewardConfig = RewardConfig(),
        max_quarters: Optional[int] = None,
    ):
        self.env_factory = env_factory
        self.agent_name = agent_name
        self.reward_config = reward_config
        self.max_quarters = max_quarters
        self.env = None
        self._last_metrics: Dict[str, float] = {}
        # Contatori cumulativi di episodio (azzerati a reset).
        self._spent_total = 0.0
        self._invalid_total = 0
        self._last_detour_cost = 0.0
        self._detour_cost_total = 0.0
        self._overdue_asset_quarters = 0
        self._quarters_below_target = 0
        self._min_beta_traj = float("inf")
        self._last_spent = 0.0
        self._last_floor_spent = 0.0
        self._last_invalid = 0
        self._risk_cost_total = 0.0
        self._last_risk_cost = 0.0
        self._risk_c0_total = 0.0
        self._failures_before = 0.0

    # -- API gym-like -----------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Riproducibilita' degli agenti di ricerca/apprendimento (GA, memetico,
        # MCTS, Q-learning usano il modulo `random` globale): il seed della run
        # seeda anche gli RNG globali, cosi' lo stesso seed => stessa traiettoria.
        seed_globals(seed)
        self.env = self.env_factory(seed)
        self._spent_total = 0.0
        self._invalid_total = 0
        self._last_detour_cost = 0.0
        self._detour_cost_total = 0.0
        self._overdue_asset_quarters = 0
        self._quarters_below_target = 0
        self._min_beta_traj = float("inf")
        self._last_spent = 0.0
        self._last_floor_spent = 0.0
        self._last_invalid = 0
        self._risk_cost_total = 0.0
        self._last_risk_cost = 0.0
        self._risk_c0_total = 0.0
        self._failures_before = 0.0
        obs = self._observe()
        self._last_metrics = self._metrics()
        return obs, self._info()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("Chiamare reset() prima di step().")
        self._failures_before = float(self._fs().get("failures", 0))
        action_costs, _revenues = self.env.step({self.agent_name: action or {}})
        # Contabilità di episodio: spesa effettiva (dai costi restituiti dal
        # motore, non dal delta di budget che include i ricavi), azioni
        # richieste ma rifiutate dal sanitizer, scadenze ispettive mancate.
        self._last_spent = float(sum(c.get("cost", 0.0) for c in (action_costs or {}).get(self.agent_name, []) or []))
        self._spent_total += self._last_spent
        self._last_floor_spent = float(sum(c.get("cost", 0.0) for c in (action_costs or {}).get(self.agent_name, []) or []
                                           if c.get("ethical_floor")))
        diags = (getattr(self.env, "last_plan_diagnostics", {}) or {}).get(self.agent_name, []) or []
        self._last_invalid = sum(1 for d in diags if d.get("status") == "blocked")
        self._invalid_total += self._last_invalid
        # Perdita attesa da collasso nel trimestre appena simulato (stato post-step).
        _loss = (getattr(self.env, "last_expected_loss", {}) or {}).get(self.agent_name) \
            or self.env.expected_quarterly_failure_loss(self.agent_name)
        self._last_risk_cost = float(_loss["total"])
        self._risk_cost_total += self._last_risk_cost
        # Parte "sostituzione" (Pf x C0): permette di ricalcolare il costo del rischio
        # per qualunque H/C0 a posteriori, perche' nessun agente legge il rapporto.
        self._risk_c0_total += float(_loss["replacement"])
        # Costo per gli utenti del traffico deviato in questo trimestre (conto separato).
        self._last_detour_cost = float(self.env.detour_user_cost_quarter(self.agent_name))
        self._detour_cost_total += self._last_detour_cost
        obs = self._observe()
        metrics = self._metrics()
        self._overdue_asset_quarters += int(metrics["overdue_assets"])
        if metrics["assets_below_target"] > 0:
            self._quarters_below_target += 1
        self._min_beta_traj = min(self._min_beta_traj, metrics["min_beta"])
        metrics["min_beta_trajectory"] = self._min_beta_traj
        self._last_metrics = metrics
        reward = self._reward(self._last_metrics, metrics)
        self._last_metrics = metrics

        terminated = self.agent_name not in self.env.active_agents
        horizon = self.max_quarters or (self.env.params.max_quarters or self.env.params.years * 4)
        truncated = not terminated and self.env.quarter > horizon
        return obs, reward, terminated, truncated, self._info()

    def feasible_actions(self) -> Dict[str, Any]:
        """Descrizione dello spazio azioni: {asset_id: [azioni fattibili]}."""
        if self.env is None:
            raise RuntimeError("Chiamare reset() prima di feasible_actions().")
        result = {}
        for asset in self.env.bridges.values():
            if asset.manager == self.agent_name:
                result[str(asset.id)] = self.env.get_feasible_actions_for_bridge(self.agent_name, asset)
        return result

    # -- Interni -----------------------------------------------------------

    def _observe(self) -> Dict[str, Any]:
        """Osservazione consegnata alla policy: quella del motore (storico) o, con
        policy_sandbox, quella sanitizzata (JSON, senza seme, solo evidenza)."""
        if getattr(self.env.params, "policy_sandbox", False):
            from infralab.agents.sandbox import sanitized_observation
            return sanitized_observation(self.env, self.agent_name)
        return self.env.get_agent_observations(self.agent_name)

    def _info(self) -> Dict[str, Any]:
        info = {
            "env": self.env,
            "quarter": self.env.quarter,
            "metrics": dict(self._last_metrics),
            "failure_info": self.env.last_failure_info,
        }
        if getattr(self.env.params, "policy_sandbox", False):
            # Sandbox: info arriva alla policy, quindi niente motore. Il harness che
            # usa un adapter legacy gli passa il motore con harness_env().
            del info["env"]
            info["policy_sandbox"] = True
        return info

    def harness_env(self):
        """Motore per il solo harness (adapter legacy): mai da passare a una policy."""
        return self.env

    def _fs(self) -> Dict[str, Any]:
        return (getattr(self.env, "failure_stats", {}) or {}).get(self.agent_name, {}) or {}

    def _min_beta_annual_assessed(self) -> float:
        env = self.env
        if getattr(env.params, "observation_mode", "full") != "inspected":
            return _reliability_snapshot(env, self.agent_name)["min_beta_annual"]
        vals = []
        for a in env.bridges.values():
            if a.manager != self.agent_name:
                continue
            est = env._assessed_bridge(a)
            b = est.posterior_annual_reliability_index if est.posterior_annual_reliability_index is not None else est.annual_reliability_index
            if b is not None:
                vals.append(float(b))
        return min(vals) if vals else 0.0

    def _metrics(self) -> Dict[str, Any]:
        cfg = self.reward_config
        rel = _reliability_snapshot(self.env, self.agent_name)
        budget = float(self.env.budgets.get(self.agent_name, 0.0))
        return {
            # KPI usati dalla reward (invariati)
            "budget": budget,
            "health_index": _health_index(self.env, self.agent_name),
            "min_beta_margin": _min_beta_margin(self.env, self.agent_name, cfg.beta_margin_clamp),
            "emissions_tco2": float(getattr(self.env, "emissions_tco2", {}).get(self.agent_name, 0.0)),
            # KPI di affidabilità / economici / operativi per il benchmark
            "min_beta": rel["min_beta"],
            "min_beta_annual": rel["min_beta_annual"],
            "beta_target_annual": rel["beta_target_annual"],
            "pf_net": rel["pf_net"],
            "assets_below_target": rel["assets_below_target"],
            "asset_value": rel["asset_value"],
            "net_worth": budget + rel["asset_value"],
            "spent_quarter": self._last_spent,
            "spent_total": self._spent_total,
            # Costo corretto per il rischio = spesa effettiva + perdita attesa da collasso.
            # Le sanzioni (ricavi persi, multe) NON sono incluse: si leggono a parte.
            "risk_cost_quarter": self._last_risk_cost,
            "risk_cost_total": self._risk_cost_total,
            "risk_cost_c0_total": self._risk_c0_total,
            "risk_adjusted_cost": self._spent_total + self._risk_cost_total,
            # Collassi realizzati (solo failure_model="reliability"; altrimenti 0)
            "failures_total": float(self._fs().get("failures", 0)),
            "failures_quarter": float(self._fs().get("failures", 0)) - self._failures_before,
            "failure_loss_total": float(self._fs().get("realised_loss", 0.0)),
            "failure_loss_c0_total": float(self._fs().get("realised_loss_c0", 0.0)),
            # beta annuo minimo STIMATO dal gestore (= vero se observation_mode="full")
            "min_beta_annual_assessed": self._min_beta_annual_assessed(),
            # Costo per gli utenti del traffico deviato (0 se detour_accounting e' spento).
            # Conto separato: non entra nel costo corretto per il rischio.
            "detour_user_cost_quarter": self._last_detour_cost,
            "detour_user_cost_total": self._detour_cost_total,
            # Azioni rinviate per mancanza di squadre (0 se crew_capacity = 0).
            "crew_deferred_total": float((getattr(self.env, "crew_stats", {}) or {}).get(self.agent_name, 0)),
            "invalid_actions_quarter": self._last_invalid,
            "invalid_actions_total": self._invalid_total,
            "overdue_assets": _overdue_assets(self.env, self.agent_name),
            "overdue_asset_quarters": self._overdue_asset_quarters,
            "quarters_below_target": self._quarters_below_target,
            "min_beta_trajectory": self._min_beta_traj if self._min_beta_traj != float("inf") else rel["min_beta"],
            # Regime normativo (tutti 0 senza regime)
            **{f"regime_{k}": v for k, v in (getattr(self.env, "regime_stats", {}) or {}).get(self.agent_name, {}).items()},
            "regime_sanctioned_assets": sum(1 for a in self.env.bridges.values()
                                            if a.manager == self.agent_name and getattr(a, "regime_sanction", None)),
            # Soglia etica (tutti 0 se ethical_floor_enabled e' spenta): conto separato
            # dalle sanzioni. La spesa della soglia e' gia' dentro spent_quarter/spent_total.
            **{f"floor_{k}": v for k, v in (getattr(self.env, "floor_stats", {}) or {}).get(self.agent_name, {}).items()},
            "floor_spend_quarter": self._last_floor_spent,
            "floor_closed_assets": sum(1 for a in self.env.bridges.values()
                                       if a.manager == self.agent_name and getattr(a, "floor_closure", None)),
            # Danno locale improvviso (chiavi presenti solo con il canale acceso): eventi
            # per tipo, resi noti, rilevati, riparati prima della scoperta, nascosti al collasso.
            **_sudden_metrics(self.env, self.agent_name),
            # Eventi estremi regionali (chiavi presenti solo con il canale acceso).
            **_regional_metrics(self.env, self.agent_name),
            # Eventi progressivi (chiavi presenti solo con il canale acceso) e collassi
            # per causa / di ponti noti come critici (solo se richiesti).
            **_progressive.metrics(self.env, self.agent_name),
            **_progressive.collapse_metrics(self.env, self.agent_name),
        }

    def _reward(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        cfg = self.reward_config
        reward = (
            cfg.budget_weight * (after["budget"] - before["budget"]) / cfg.budget_scale
            + cfg.health_weight * (after["health_index"] - before["health_index"])
            + cfg.beta_margin_weight * (after["min_beta_margin"] - before["min_beta_margin"])
        )
        failure = self.env.last_failure_info
        if failure and failure.get("agent_name") == self.agent_name and self.agent_name not in self.env.active_agents:
            if failure.get("reason") == "BRIDGE_COLLAPSE":
                reward -= cfg.collapse_penalty
            elif failure.get("reason") == "BANKRUPTCY":
                reward -= cfg.bankruptcy_penalty
        return reward
