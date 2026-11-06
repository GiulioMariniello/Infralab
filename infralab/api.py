"""Public Python API of InfraLab.

A small, stable surface over the engine:

    import infralab

    env = infralab.make("regbench/aged-30", law="regbench/D1", seed=0, horizon_years=10)
    obs, info = env.reset()
    while not env.done:
        plan = {}                       # {asset_id: action_name}; an empty plan does nothing
        obs, reward, terminated, truncated, info = env.step(plan)
    print(env.summary())

    infralab.run("worst_first", "regbench/aged-30", law="regbench/D1", seed=0)
    infralab.compare_laws("regbench/D0", "regbench/D1", policy="passive", n_seeds=10)

Nothing here changes the dynamics of the engine: the module builds environments, runs them and
reads outcomes. The observation handed to a policy is the one of the engine under the information
rule of the chosen configuration (inspected observation and policy sandbox in the benchmark).
"""
from __future__ import annotations

import copy
import json
import os
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

_PKG = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_PKG, "benchmarks", "regbench")
_SEED_SCENARIOS = [os.path.join(_PKG, "benchmarks", "scenarios.json"),
                   os.path.join(os.path.dirname(_PKG), "scenarios.json")]
_PHI = NormalDist().cdf
# limit for existing bridges used in the outcomes (annual beta): operating level of load rating, 2.5 over 5 years
BETA_LIMIT = 3.02

# Short English names for the policies supplied with the library -> engine agent configuration.
POLICIES: Dict[str, Dict[str, Any]] = {
    "passive": {"type": "Basic", "policy": "do_nothing"},
    "worst_first": {"type": "Basic", "policy": "worst_first"},
    "typical": {"type": "Basic", "policy": "typical_owner"},
    "diligent": {"type": "Basic", "policy": "inspect_and_repair"},
    "heuristic": {"type": "Euristico"},
    "monte_carlo": {"type": "MCTS", "mcts_params": {"n_rollouts": 30, "rollout_depth": 4, "n_candidates": 8}},
    "q_learning": {"type": "RL Q-Learning"},
    "genetic": {"type": "GA Orizzonte Variabile"},
    "memetic": {"type": "Algoritmo Memetico"},
}


# English names for the actions of the bridge catalogue (the engine names are accepted as well).
ACTIONS: Dict[str, str] = {
    "census": "L0_census", "inspect": "L1_inspect", "interim_check": "sorveglianza_intermedia",
    "detailed_inspection": "L2_inspect", "structural_verification": "L4_verify", "monitor": "monitoring",
    "proof_load_test": "proof_load_test",
    "replace_deck": "replace_impalcato", "replace_piers": "replace_pile", "replace_abutments": "replace_spalle",
    "replace_bearings": "replace_appoggi", "restore": "ripristino_strutturale", "restore_fully": "ripristino_completo",
    "maintain": "manutenzione_ordinaria", "protect_from_corrosion": "protezione_anticorrosione",
    "limit_load": "limitazione_carico", "one_way_traffic": "senso_unico_alternato", "close": "chiusura_ponte",
    "reopen": "riapertura", "seismic_improvement": "miglioramento_sismico", "seismic_upgrade": "adeguamento_sismico",
    "arrest_defect": "arresto_difetto",
}


def actions() -> Dict[str, str]:
    """English action names accepted in a plan, with the engine name of each."""
    return dict(ACTIONS)


# ----------------------------------------------------------------------------- catalogues
def _load_yaml(path: str) -> Any:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _seed_scenarios() -> List[Dict[str, Any]]:
    for path in _SEED_SCENARIOS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
    return []


def scenarios() -> List[str]:
    """Identifiers of the scenarios shipped with the library."""
    out = ["regbench/" + f[:-5] for f in sorted(os.listdir(os.path.join(_BENCH, "scenarios"))) if f.endswith(".json")]
    return out + [s["scenario_id"] for s in _seed_scenarios() if s.get("scenario_id")]


def load_scenario(ref: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A scenario as a dictionary: a shipped identifier, a path to a JSON file, or a dictionary (copied)."""
    if isinstance(ref, dict):
        return copy.deepcopy(ref)
    if ref.startswith("regbench/"):
        path = os.path.join(_BENCH, "scenarios", ref.split("/", 1)[1] + ".json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    if os.path.exists(ref):
        with open(ref, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[0] if isinstance(data, list) else data
    for s in _seed_scenarios():
        if s.get("scenario_id") == ref:
            return copy.deepcopy(s)
    raise KeyError(f"Unknown scenario {ref!r}. Available: {scenarios()} or a path to a JSON file.")


def laws() -> List[str]:
    """Identifiers of the laws (regulatory regimes) shipped with the library."""
    from infralab.regimes.registry import available_regimes
    bench = ["regbench/" + f[:-5] for f in sorted(os.listdir(os.path.join(_BENCH, "laws"))) if f.endswith(".yaml")]
    return bench + [r for r in available_regimes()]


def resolve_law(ref: Optional[str]) -> Optional[str]:
    """The value of the engine parameter `regulatory_regime` for a law reference (identifier or file path)."""
    if ref is None or str(ref).lower() in ("", "none", "null"):
        return None
    ref = str(ref)
    if ref.startswith("regbench/"):
        path = os.path.join(_BENCH, "laws", ref.split("/", 1)[1] + ".yaml")
        if not os.path.exists(path):
            raise KeyError(f"Unknown law {ref!r}. Available: {laws()}")
        return path
    return ref


def describe_law(ref: Optional[str]) -> Dict[str, Any]:
    """The content of a law as plain data (obligations, enforcement, classification)."""
    path = resolve_law(ref)
    if path is None:
        return {"id": None, "display_name": "no law"}
    if os.path.isfile(path):
        return _load_yaml(path) if path.lower().endswith((".yaml", ".yml")) else json.load(open(path, encoding="utf-8"))
    from dataclasses import asdict, is_dataclass
    from infralab.regimes.registry import get_regime
    reg = get_regime(path)
    return asdict(reg) if is_dataclass(reg) else {"id": getattr(reg, "id", path)}


def degradation_laws() -> List[str]:
    """Degradation laws: the two built into the engine and those shipped as files with the benchmark."""
    d = os.path.join(_BENCH, "degradation")
    return ["weibull", "gamma"] + ["regbench/" + f[:-3].replace("_", "-") for f in sorted(os.listdir(d)) if f.endswith(".py")]


def resolve_degradation(ref: Optional[str]) -> Optional[str]:
    """The value of the engine parameter `degradation_model` for a degradation law.

    `ref` is "weibull", "gamma", a shipped identifier ("regbench/markov-nbi"), or the path of a Python file
    that defines `degrade(ctx) -> condition`. A file is validated and registered as a user law of the engine.
    """
    if ref is None:
        return None
    ref = str(ref)
    if ref in ("weibull", "gamma") or ref.startswith("custom:"):
        return ref
    path = os.path.join(_BENCH, "degradation", ref.split("/", 1)[1].replace("-", "_") + ".py") if ref.startswith("regbench/") else ref
    if not os.path.isfile(path):
        raise KeyError(f"Unknown degradation law {ref!r}. Available: {degradation_laws()} or the path of a Python file.")
    from infralab.plugins import registry
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in os.path.splitext(os.path.basename(path))[0])[:48]
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        same = registry.get_plugin("degradation", name)["source"] == source
    except registry.PluginError:
        same = False
    if not same:
        registry.save_plugin("degradation", name, source)
    return "custom:" + name


def configurations() -> List[str]:
    """Names of the benchmark configurations (complete sets of parameter values)."""
    return list(_load_yaml(os.path.join(_BENCH, "configurations.yaml")))


def configuration(name: str) -> Dict[str, Any]:
    """Parameter values of a benchmark configuration, with its root seed and horizon."""
    cfg = _load_yaml(os.path.join(_BENCH, "configurations.yaml"))
    if name not in cfg:
        raise KeyError(f"Unknown configuration {name!r}. Available: {list(cfg)}")
    return cfg[name]


def policies() -> List[str]:
    """Names of the policies supplied with the library (see also `basic_policies`)."""
    return list(POLICIES)


def basic_policies() -> List[str]:
    """Names of the fixed rules of the engine, usable as `policy("basic:<name>")`."""
    from infralab.agents.basic import BASIC_POLICIES
    return list(BASIC_POLICIES)


def policy_config(name_or_config: Union[str, Dict[str, Any]], **params: Any) -> Dict[str, Any]:
    """Engine agent configuration for a policy name; keyword arguments are policy parameters."""
    if isinstance(name_or_config, dict):
        cfg = copy.deepcopy(name_or_config)
    elif name_or_config in POLICIES:
        cfg = copy.deepcopy(POLICIES[name_or_config])
    elif name_or_config.startswith("basic:"):
        cfg = {"type": "Basic", "policy": name_or_config.split(":", 1)[1]}
    elif name_or_config.startswith("llm:"):
        # "llm:<provider>:<model>", e.g. "llm:OpenAI:gpt-4.1-mini"
        _, provider, model = (name_or_config.split(":", 2) + [None, None])[:3]
        cfg = {"type": "LLM", "provider": provider, "model": model, "profile": "Bilanciato"}
    else:
        raise KeyError(f"Unknown policy {name_or_config!r}. Available: {policies()}, 'basic:<rule>', 'llm:<provider>:<model>'.")
    if params:
        if cfg.get("type") == "Basic":
            cfg.setdefault("params", {}).update(params)
        else:
            cfg.update(params)
    return cfg


# ----------------------------------------------------------------------------- environment
class Env:
    """One owner managing a stock, quarter by quarter (reset / step, in the form of Gymnasium).

    A plan is `{asset_id: action_name}`; at most one action per asset per quarter. Actions that are
    unfeasible or that the budget cannot pay are dropped and reported in `last_report`.
    """

    def __init__(self, scenario: Union[str, Dict[str, Any]] = "regbench/aged-30", *, law: Optional[str] = None,
                 config: Union[str, Dict[str, Any], None] = "paper", seed: int = 0,
                 horizon_years: Optional[float] = None, params: Optional[Dict[str, Any]] = None,
                 degradation: Optional[str] = None,
                 owner: str = "owner", owner_config: Optional[Dict[str, Any]] = None):
        self.scenario = load_scenario(scenario)
        self.law = law
        self.owner = owner
        self.seed = int(seed)
        self._owner_config = owner_config or {"type": "Umano"}
        over: Dict[str, Any] = {}
        horizon_q = None
        if isinstance(config, str):
            c = configuration(config)
            over.update(c["params"])
            horizon_q = int(c["horizon_quarters"])
        elif isinstance(config, dict):
            over.update(config)
        if params:
            over.update(params)
        if degradation is not None:
            over["degradation_model"] = resolve_degradation(degradation)
        reg = resolve_law(law)
        if reg is not None:
            over["regulatory_regime"] = reg
        elif law is not None or "regulatory_regime" not in over:
            over.pop("regulatory_regime", None)
        if horizon_years is not None:
            horizon_q = int(round(4 * float(horizon_years)))
        self.params = over
        self.horizon_quarters = horizon_q
        self._gym = None
        self._beta: List[Tuple[int, int, float]] = []
        self._factor: List[Tuple[int, int, float]] = []
        self._steps = 0
        self._terminated = self._truncated = False
        self.last_report: Dict[str, Any] = {}

    # -- construction
    def _factory(self, seed):
        from infralab.lab.build import build_env_from_scenario
        return build_env_from_scenario(self.scenario, agent_name=self.owner, agent_config=self._owner_config,
                                       param_overrides=self.params or None, seed=seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from infralab.agents.protocol import AssetEnvGym
        if seed is not None:
            self.seed = int(seed)
        self._gym = AssetEnvGym(self._factory, self.owner, max_quarters=self.horizon_quarters)
        obs, info = self._gym.reset(seed=self.seed)
        self._beta, self._factor, self._steps = [], [], 0
        self._terminated = self._truncated = False
        self.last_report = {}
        self._snap(0)
        self._obs = obs
        return obs, self._public_info(info)

    # -- stepping
    def step(self, plan: Optional[Dict[str, Any]] = None):
        if self._gym is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("The run is over: call reset().")
        action = plan or {}
        if "maintenance" not in action:
            action = {"maintenance": {str(k): ACTIONS.get(v, v) if isinstance(v, str) else v for k, v in action.items()}}
        before = dict(self._gym._last_metrics)
        obs, reward, self._terminated, self._truncated, info = self._gym.step(action)
        self._steps += 1
        self._snap(self._steps)
        self._obs = obs
        self.last_report = self._report(action["maintenance"], before, info["metrics"])
        return obs, reward, self._terminated, self._truncated, self._public_info(info)

    # -- reading
    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def quarter(self) -> int:
        return int(self.engine.quarter)

    @property
    def engine(self):
        """The engine object. For harness code only: a policy must not read it (it holds the true state)."""
        if self._gym is None:
            raise RuntimeError("Call reset() first.")
        return self._gym.harness_env()

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self._gym._last_metrics)

    def observation(self) -> Dict[str, Any]:
        """The full observation of the engine (large). See `compact_observation` for a short form."""
        return self._obs

    def feasible_actions(self, asset_id: Optional[Union[int, str]] = None) -> Dict[str, Any]:
        fa = self._gym.feasible_actions()
        return fa if asset_id is None else fa[str(asset_id)]

    def compact_observation(self, asset_ids: Optional[Iterable[Union[int, str]]] = None) -> Dict[str, Any]:
        """What the owner knows, in a few hundred characters per asset (for language models and tools)."""
        from infralab.views import compact_observation
        return compact_observation(self._obs, quarter=self.quarter, horizon_quarters=self._horizon(),
                                   asset_ids=asset_ids)

    def rules(self) -> Dict[str, Any]:
        """The rules of this run: horizon, budget, law in force, ethical floor, information rule."""
        from infralab.views import rules_view
        return rules_view(self)

    def _horizon(self) -> int:
        p = self.engine.params
        return int(self.horizon_quarters or p.max_quarters or p.years * 4)

    def _public_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        return {"quarter": info["quarter"], "metrics": info["metrics"], "failure_info": info.get("failure_info"),
                "report": self.last_report}

    def _snap(self, n: int) -> None:
        env = self.engine
        for b in env.bridges.values():
            ba = b.posterior_annual_reliability_index
            if ba is None:
                ba = b.annual_reliability_index
            self._beta.append((n, int(b.id), float("nan") if ba is None else float(ba)))
            if n % 4 == 0:
                self._factor.append((n // 4, int(b.id), float(getattr(b, "traffic_restriction_factor", 1.0))))

    def _report(self, requested: Dict[str, Any], before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        diags = (getattr(self.engine, "last_plan_diagnostics", {}) or {}).get(self.owner, []) or []
        done, dropped = [], []
        for d in diags:
            row = {k: d.get(k) for k in ("bridge_id", "requested_action", "action", "status", "reason", "cost") if k in d}
            (dropped if d.get("status") == "blocked" else done).append(row)
        delta = lambda k: float(after.get(k, 0) or 0) - float(before.get(k, 0) or 0)
        return {"quarter": int(after.get("quarter", self.quarter - 1)) if "quarter" in after else self.quarter - 1,
                "requested": len(requested), "accepted": done, "dropped": dropped,
                "spent": float(after.get("spent_quarter", 0.0)),
                "spent_by_ethical_floor": float(after.get("floor_spend_quarter", 0.0)),
                "inspections_forced_by_law": delta("regime_forced_actions"),
                "ethical_floor_interventions": delta("floor_interventions"),
                "collapses": float(after.get("failures_quarter", 0.0)),
                "budget": float(after.get("budget", 0.0))}

    def _expected_progressive_hydraulic(self, m: Dict[str, Any]) -> float:
        """Expected hydraulic collapses from progressive defects: collapse probability of a defect that
        matures unarrested times the number of such defects (direct hydraulic collapses are left out)."""
        n = float(m.get("progressive_matured_hydraulic", 0.0) or 0.0)
        if n == 0.0:
            return 0.0
        from infralab.core.hazards.progressive_events import _spec_by_name
        spec = _spec_by_name(self.engine.params).get("hydraulic")
        return n * float(spec["p_collapse"]) if spec else float("nan")

    def summary(self) -> Dict[str, Any]:
        """Outcomes of the run so far, read on the true state (for scoring, never for a policy)."""
        beta = np.array([(q, b, v) for q, b, v in self._beta if q >= 1], dtype=float)
        out: Dict[str, Any] = {"quarters": self._steps, "seed": self.seed}
        if len(beta):
            ids = np.unique(beta[:, 1])
            mins = np.array([np.nanmin(beta[beta[:, 1] == i, 2]) for i in ids])
            pfq = 1.0 - (1.0 - np.array([_PHI(-x) for x in beta[:, 2]])) ** 0.25
            out.update(S1=float(np.nanmean(mins)), min_beta_stock=float(np.nanmin(mins)),
                       expected_collapses=float(np.nansum(pfq)),
                       bridges_below_limit=int((mins < BETA_LIMIT).sum()),
                       bridge_years_below_limit=float((beta[:, 2] < BETA_LIMIT).sum() / 4.0))
        fac = np.array([(y, b, v) for y, b, v in self._factor if y >= 1], dtype=float)
        if len(fac):
            out.update(closed_share=float((fac[:, 2] <= 0).mean()),
                       restricted_share=float(((fac[:, 2] > 0) & (fac[:, 2] < 1)).mean()))
        m = self.metrics
        out["expected_hydraulic_collapses"] = self._expected_progressive_hydraulic(m)
        out.update(collapses=float(m.get("failures_total", 0.0)), cash_spend=float(m.get("spent_total", 0.0)),
                   imputed_expected_loss=float(m.get("risk_cost_total", 0.0)),
                   user_cost=float(m.get("detour_user_cost_total", 0.0)),
                   emissions_tco2=float(m.get("emissions_tco2", 0.0)),
                   invalid_requests=float(m.get("invalid_actions_total", 0.0)),
                   inspections_forced_by_law=float(m.get("regime_forced_actions", 0.0) or 0.0),
                   ethical_floor_interventions=float(m.get("floor_interventions", 0.0) or 0.0),
                   budget=float(m.get("budget", 0.0)))
        return out


def make(scenario: Union[str, Dict[str, Any]] = "regbench/aged-30", **kwargs: Any) -> Env:
    """Create an environment. See `Env` for the arguments."""
    return Env(scenario, **kwargs)


# ----------------------------------------------------------------------------- running policies
PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def child_seeds(root_seed: int, n: int) -> List[int]:
    """The paired seeds of a campaign: one child of the root seed per replicate (as in the campaign runner)."""
    return [int(c.generate_state(1)[0] % (2 ** 31 - 1)) for c in np.random.SeedSequence(int(root_seed)).spawn(int(n))]


def run(policy: Union[str, Dict[str, Any], PolicyFn] = "passive", scenario: Union[str, Dict[str, Any]] = "regbench/aged-30", *,
        law: Optional[str] = None, config: Union[str, Dict[str, Any], None] = "paper", seed: int = 0,
        horizon_years: Optional[float] = None, params: Optional[Dict[str, Any]] = None,
        degradation: Optional[str] = None,
        policy_params: Optional[Dict[str, Any]] = None, on_step: Optional[Callable[[Env], None]] = None) -> Dict[str, Any]:
    """Run one policy from the first quarter to the horizon and return the outcomes.

    `policy` is a name from `policies()`, an engine agent configuration, or a function
    `observation -> {asset_id: action}`.
    """
    if callable(policy):
        env = Env(scenario, law=law, config=config, seed=seed, horizon_years=horizon_years, params=params,
                  degradation=degradation)
        obs, _ = env.reset()
        while not env.done:
            obs, *_ = env.step(policy(obs) or {})
            if on_step:
                on_step(env)
        return {**env.summary(), "policy": getattr(policy, "__name__", "function")}
    from infralab.agents.adapters import make_agent
    cfg = policy_config(policy, **(policy_params or {}))
    label = policy if isinstance(policy, str) else cfg.get("policy", cfg.get("type", "policy"))
    if cfg.get("policy") == "typical_owner":
        # the typical owner does not know the seed of the run: it is given the seed of the replicate
        cfg.setdefault("params", {}).setdefault("seed", int(seed))
    env = Env(scenario, law=law, config=config, seed=seed, horizon_years=horizon_years, params=params,
              degradation=degradation, owner=str(label).replace(":", "_"), owner_config=cfg)
    obs, info = env.reset()
    agent = make_agent(env.owner, cfg)
    agent.reset(obs, {"quarter": info["quarter"], "metrics": info["metrics"], "env": env.engine})
    failed = 0
    while not env.done:
        plan = agent.act(obs)
        rat = getattr(agent, "last_rationale", None)
        if isinstance(rat, tuple) and len(rat) == 2 and isinstance(rat[1], str) and rat[1].startswith("Errore"):
            failed += 1
        obs, *_ = env.step(plan)
        if on_step:
            on_step(env)
    return {**env.summary(), "policy": label, "failed_decisions": failed}


def _bootstrap(v: np.ndarray, n: int = 10000, seed: int = 0) -> List[float]:
    v = np.asarray(v, dtype=float)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def _run_kw(kw: Dict[str, Any]) -> Dict[str, Any]:
    return run(**kw)


def compare_laws(law_a: Optional[str], law_b: Optional[str], *, policy: Union[str, Dict[str, Any]] = "passive",
                 scenario: Union[str, Dict[str, Any]] = "regbench/aged-30", config: Union[str, Dict[str, Any], None] = "paper",
                 n_seeds: int = 10, root_seed: Optional[int] = None, horizon_years: Optional[float] = None,
                 params: Optional[Dict[str, Any]] = None, policy_params: Optional[Dict[str, Any]] = None,
                 degradation: Optional[str] = None, workers: int = 1) -> Dict[str, Any]:
    """Paired-seed contrast of two laws: the same policy, stock and random draws, with law B minus law A.

    Returns the per-seed outcomes and, for each outcome, the mean paired difference with a bootstrap
    95 percent interval over seeds. This is the protocol of the benchmark as one call.
    """
    if root_seed is None:
        root_seed = int(configuration(config)["root_seed"]) if isinstance(config, str) else 0
    seeds = child_seeds(root_seed, n_seeds)
    degradation = resolve_degradation(degradation)  # registered once, here, before any worker starts
    jobs = [dict(policy=policy, scenario=scenario, law=law, config=config, seed=s, horizon_years=horizon_years,
                 params=params, policy_params=policy_params, degradation=degradation) for law in (law_a, law_b) for s in seeds]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_run_kw, jobs))
    else:
        res = [_run_kw(j) for j in jobs]
    a, b = res[:len(seeds)], res[len(seeds):]
    keys = ["S1", "expected_collapses", "expected_hydraulic_collapses", "bridge_years_below_limit", "bridges_below_limit",
            "collapses", "cash_spend", "user_cost", "emissions_tco2", "closed_share"]
    diff = {k: _bootstrap(np.array([y.get(k, np.nan) - x.get(k, np.nan) for x, y in zip(a, b)])) for k in keys if k in a[0]}
    return {"law_a": law_a, "law_b": law_b, "policy": policy if isinstance(policy, str) else "custom", "seeds": seeds,
            "runs_a": a, "runs_b": b, "difference_b_minus_a": diff,
            "mean_a": {k: float(np.mean([x[k] for x in a])) for k in diff}, "mean_b": {k: float(np.mean([x[k] for x in b])) for k in diff}}
