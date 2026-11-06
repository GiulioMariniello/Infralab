"""InfraLab Open Benchmark & Arena Competitiva.

Permette l'esecuzione standardizzata e riproducibile di campagne di confronto
tra agenti decisori (LLM open-source da Hugging Face, LLM commerciali,
GA, MCTS, Euristiche) su scenari infrastrutturali fissi.

I risultati vengono aggregati per produrre tabelle e statistiche per il paper:
- Sicurezza strutturale (beta_net, beta_min, Pf)
- Efficienza economica (NAV, budget residuo, spesa)
- Compliance operativa (violazioni di vincoli e budget)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from infralab.agents.adapters import make_agent
from infralab.calibration_profiles import apply_calibration_profile
from infralab.core.environment import BridgeEnv
from infralab.lab.build import build_env_from_scenario
from infralab.lab.spec import resolve_scenario
from infralab.settings.config import (
    CdaConfig,
    DEFAULT_COMPONENTS,
    MaintOpsConfig,
    SimParameters,
)


@dataclass
class ArenaRunResult:
    competitor_name: str
    agent_type: str
    provider: str
    model: str
    scenario_id: str
    seed: int
    quarters_simulated: int
    final_health_index: float
    min_beta: float
    final_beta_net: float
    final_failure_prob: float
    final_budget: float
    final_net_worth: float
    total_spent: float
    invalid_actions_attempted: int
    duration_s: float
    status: str
    error_message: str = ""
    risk_cost_total: float = 0.0
    risk_adjusted_cost: float = 0.0
    trace: Optional[List[Dict[str, float]]] = None


# Configurazione default di competitor per l'Arena
DEFAULT_COMPETITORS: List[Dict[str, Any]] = [
    {
        "name": "Euristico_Baseline",
        "type": "Euristico",
    },
    {
        "name": "GA_Rolling_Horizon",
        "type": "GA Orizzonte Variabile",
        "ga_params": {"horizon": 4, "pop_size": 40, "generations": 15},
    },
    {
        "name": "MCTS_Planner",
        "type": "MCTS",
        "mcts_params": {"n_rollouts": 30, "rollout_depth": 4, "n_candidates": 8},
    },
    {
        "name": "HF_Llama_3.3_70B",
        "type": "LLM",
        "provider": "HuggingFace",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "profile": "Bilanciato",
    },
    {
        "name": "HF_Qwen_2.5_72B",
        "type": "LLM",
        "provider": "HuggingFace",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "profile": "Bilanciato",
    },
    {
        "name": "HF_DeepSeek_R1_32B",
        "type": "LLM",
        "provider": "HuggingFace",
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "profile": "Bilanciato",
    },
]


DEFAULT_SCENARIOS = ["PAPERA-SHORT-A", "PAPERA-LONG-A", "NBI-DE", "EXP-A-BUDGET", "WALL-DEMO"]


def load_default_scenario() -> Dict[str, Any]:
    """Carica uno scenario di default (TRAIN-01 o primo scenario in scenarios.json)."""
    try:
        return resolve_scenario("TRAIN-01")
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        sc_file = repo_root / "scenarios.json"
        if sc_file.exists():
            with open(sc_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) and data else {}
        return {}


def calculate_agent_kpis(env: BridgeEnv, agent: str) -> Dict[str, float]:
    """Calcola i KPI strutturali ed economici per un agente coerentemente con persistence.py."""
    my_bridges = {bid: b for bid, b in env.bridges.items() if b.manager == agent}
    if not my_bridges:
        return {
            "health_index": 0.0,
            "network_reliability_index": 0.0,
            "network_failure_probability": 0.0,
            "total_asset_value": 0.0,
        }
    total_adt = sum(b.adt for b in my_bridges.values())
    reliability_values = [
        b.posterior_reliability_index if b.posterior_reliability_index is not None else b.reliability_index
        for b in my_bridges.values()
    ]
    failure_probs = [
        b.posterior_failure_probability if b.posterior_failure_probability is not None else b.failure_probability
        for b in my_bridges.values()
    ]
    hi = sum(b.min_condition * b.adt for b in my_bridges.values()) / total_adt if total_adt > 0 else 0.0
    beta_net = min(reliability_values) if reliability_values else 0.0
    pf_net = max(failure_probs) if failure_probs else 0.0
    val = sum(b.current_asset_value for b in my_bridges.values())
    return {
        "health_index": hi,
        "network_reliability_index": beta_net,
        "network_failure_probability": pf_net,
        "total_asset_value": val,
    }


def run_single_simulation(
    competitor: Dict[str, Any],
    scenario_spec: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    scenario_id: str = "default_nbi",
    horizon_quarters: Optional[int] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> ArenaRunResult:
    """Esegue una singola partita dell'arena per un concorrente e restituisce i KPI."""
    name = competitor.get("name", "Competitor")
    agent_type = competitor.get("type", "Euristico")
    provider = competitor.get("provider", "N/A")
    model = competitor.get("model", "N/A")

    t_start = time.time()
    try:
        from infralab.agents.protocol import seed_globals
        seed_globals(seed)   # riproducibilita' degli agenti che usano il random globale
        agent_config = {name: competitor}
        agent_adapter = make_agent(name, competitor)

        if not scenario_spec:
            scenario_spec = load_default_scenario()

        env = build_env_from_scenario(
            scenario_spec,
            agent_name=name,
            agent_config=competitor,
            param_overrides=param_overrides,
            seed=seed,
        )

        # Reset dell'agente nel protocollo
        obs = env.get_agent_observations(name)
        agent_adapter.reset(obs, {"env": env})

        min_beta_observed = float("inf")
        invalid_actions_count = 0
        total_spent = 0.0
        risk_cost_total = 0.0
        trace: List[Dict[str, float]] = []

        quarters_to_run = horizon_quarters if horizon_quarters is not None else (env.params.years * 4)
        for q in range(quarters_to_run):
            current_obs = env.get_agent_observations(name)
            plan = agent_adapter.act(current_obs)

            # Il motore attende {agente: {"maintenance": {bridge_id: azione}}}.
            if not isinstance(plan, dict):
                invalid_actions_count += 1
                plan = {}
            maint_plan = plan.get("maintenance", {})
            if not isinstance(maint_plan, dict):
                invalid_actions_count += 1
                maint_plan = {}

            action_costs, _revenues = env.step({name: {"maintenance": maint_plan}})
            spent = float(sum(c.get("cost", 0.0) for c in (action_costs.get(name) or [])))
            total_spent += spent
            # Azioni richieste ma rifiutate dal sanitizer (vincoli, budget, ponte occupato...).
            diags = (getattr(env, "last_plan_diagnostics", {}) or {}).get(name, [])
            invalid_actions_count += sum(1 for d in diags if d.get("status") != "accepted")
            post_budget = env.budgets.get(name, 0.0)
            risk_cost_total += float(env.expected_quarterly_failure_loss(name)["total"])

            kpis = calculate_agent_kpis(env, name)
            beta_net = kpis["network_reliability_index"]
            if beta_net < min_beta_observed:
                min_beta_observed = beta_net
            trace.append({"quarter": q + 1, "beta_net": beta_net,
                          "health_index": kpis.get("health_index", 0.0),
                          "budget": post_budget, "spent": spent})

        final_kpis = calculate_agent_kpis(env, name)
        final_budget = env.budgets.get(name, 0.0)
        final_asset_val = final_kpis["total_asset_value"]
        final_net_worth = final_budget + final_asset_val

        duration = time.time() - t_start
        return ArenaRunResult(
            competitor_name=name,
            agent_type=agent_type,
            provider=provider,
            model=model,
            scenario_id=scenario_id,
            seed=seed,
            quarters_simulated=quarters_to_run,
            final_health_index=final_kpis.get("health_index", 0.0),
            min_beta=min_beta_observed if min_beta_observed != float("inf") else 0.0,
            final_beta_net=final_kpis.get("network_reliability_index", 0.0),
            final_failure_prob=final_kpis.get("network_failure_probability", 0.0),
            final_budget=final_budget,
            final_net_worth=final_net_worth,
            total_spent=total_spent,
            invalid_actions_attempted=invalid_actions_count,
            duration_s=round(duration, 2),
            status="SUCCESS",
            trace=trace,
            risk_cost_total=risk_cost_total,
            risk_adjusted_cost=total_spent + risk_cost_total,
        )

    except Exception as e:
        duration = time.time() - t_start
        return ArenaRunResult(
            competitor_name=name,
            agent_type=agent_type,
            provider=provider,
            model=model,
            scenario_id=scenario_id,
            seed=seed,
            quarters_simulated=0,
            final_health_index=0.0,
            min_beta=0.0,
            final_beta_net=0.0,
            final_failure_prob=1.0,
            final_budget=0.0,
            final_net_worth=0.0,
            total_spent=0.0,
            invalid_actions_attempted=0,
            duration_s=round(duration, 2),
            status="FAILED",
            error_message=str(e),
        )


def run_arena_campaign(
    competitors: List[Dict[str, Any]],
    scenarios: List[Tuple[str, Optional[Dict[str, Any]]]],
    seeds: List[int],
    output_dir: str = "arena_results",
    horizon_quarters: Optional[int] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Esegue l'intera campagna benchmark dell'Arena e salva i risultati.

    Output (in `output_dir`):
      arena_runs_<ts>.csv       una riga per run (KPI finali)
      arena_runs_<ts>.json      stesse righe + traccia per trimestre + provenance
      arena_paper_summary.csv   media ± std per (competitor, scenario)
    """
    from infralab.lab.provenance import collect_provenance

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    campaign_cfg = {
        "competitors": competitors,
        "scenarios": [sc_id for sc_id, _ in scenarios],
        "seeds": seeds,
        "horizon_quarters": horizon_quarters,
        "param_overrides": dict(param_overrides or {}),
    }
    provenance = collect_provenance(json.dumps(campaign_cfg, sort_keys=True, default=str))
    provenance["hf_model_revisions"] = _hf_model_revisions(competitors)
    provenance["campaign_start_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    results: List[ArenaRunResult] = []
    total_runs = len(competitors) * len(scenarios) * len(seeds)
    print(f"=== InfraLab Arena: avvio di {total_runs} simulazioni totali ===")

    current_run = 0
    for comp in competitors:
        for sc_id, sc_spec in scenarios:
            for seed in seeds:
                current_run += 1
                comp_name = comp.get("name", "Unknown")
                print(f"[{current_run}/{total_runs}] {comp_name} | {sc_id} | seed {seed} ...", end=" ", flush=True)
                res = run_single_simulation(comp, sc_spec, seed=seed, scenario_id=sc_id,
                                            horizon_quarters=horizon_quarters,
                                            param_overrides=param_overrides)
                print(f"-> {res.status} (HI={res.final_health_index:.3f}, beta_min={res.min_beta:.2f}, "
                      f"spesa={res.total_spent:,.0f}, t={res.duration_s}s)"
                      + (f" {res.error_message[:80]}" if res.error_message else ""))
                results.append(res)

    provenance["campaign_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = [asdict(r) for r in results]
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "trace"} for r in rows])
    csv_file = out_path / f"arena_runs_{ts}.csv"
    df_results.to_csv(csv_file, index=False)
    json_file = out_path / f"arena_runs_{ts}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"provenance": provenance, "campaign": campaign_cfg, "runs": rows},
                  f, indent=1, ensure_ascii=False, default=str)
    print(f"\n[OK] Run salvati in: {csv_file}  e  {json_file}")

    ok = df_results[df_results["status"] == "SUCCESS"]
    if not ok.empty:
        summary_df = ok.groupby(["competitor_name", "scenario_id"]).agg(
            n=("seed", "count"),
            health_index_mean=("final_health_index", "mean"), health_index_std=("final_health_index", "std"),
            min_beta_mean=("min_beta", "mean"), min_beta_std=("min_beta", "std"),
            beta_net_mean=("final_beta_net", "mean"), beta_net_std=("final_beta_net", "std"),
            pf_mean=("final_failure_prob", "mean"),
            net_worth_mean=("final_net_worth", "mean"), net_worth_std=("final_net_worth", "std"),
            spent_mean=("total_spent", "mean"), spent_std=("total_spent", "std"),
            invalid_mean=("invalid_actions_attempted", "mean"),
            duration_mean_s=("duration_s", "mean"),
        )
        summary_file = out_path / "arena_paper_summary.csv"
        summary_df.to_csv(summary_file)
        print(f"[OK] Tabella aggregata per paper salvata in: {summary_file}")
    n_fail = int((df_results["status"] != "SUCCESS").sum())
    if n_fail:
        print(f"[WARN] {n_fail} run FAILED - vedi colonna error_message")

    return df_results


def _split_list(values: Optional[List[str]]) -> Optional[List[str]]:
    """Accetta sia `--x a b c` sia `--x a,b,c` (o misti)."""
    if values is None:
        return None
    out: List[str] = []
    for v in values:
        out.extend(x.strip() for x in str(v).split(",") if x.strip())
    return out


def _resolve_scenarios(ids: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(sid, resolve_scenario(sid)) for sid in ids]


def _hf_model_revisions(competitors: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Revisione (sha) corrente su Hugging Face Hub dei modelli usati: pinning in provenance."""
    out: Dict[str, Optional[str]] = {}
    hf_models = {c["model"] for c in competitors if c.get("provider") == "HuggingFace" and c.get("model")}
    if not hf_models:
        return out
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
        for m in sorted(hf_models):
            try:
                out[m] = api.model_info(m).sha
            except Exception as e:  # rete / gated repo: registra il motivo, non bloccare
                out[m] = f"UNRESOLVED: {type(e).__name__}"
    except ImportError:
        for m in sorted(hf_models):
            out[m] = "UNRESOLVED: huggingface_hub non installato"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="InfraLab Arena - benchmark riproducibile multi-agente",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", default="data/arena_results", help="directory di salvataggio")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS,
                        help="scenario_id in scenarios.json (o path a JSON)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="seed stocastici")
    parser.add_argument("--horizon", type=int, default=None,
                        help="trimestri simulati (default: orizzonte dello scenario)")
    parser.add_argument("--competitors", nargs="+", default=None,
                        help="subset per nome fra DEFAULT_COMPETITORS (default: tutti)")
    parser.add_argument("--competitors-file", default=None,
                        help="JSON con lista di competitor custom (sostituisce i default)")
    parser.add_argument("--no-llm", action="store_true", help="escludi i competitor LLM (dry run gratuito)")
    parser.add_argument("--hf-token", default=None, help="token Hugging Face (o env HF_TOKEN)")
    parser.add_argument("--include-commercial", action="store_true",
                        help="aggiungi GPT-4o / Gemini se le chiavi sono in env")
    parser.add_argument("--regime", default=None,
                        help="regime normativo (id built-in o path YAML); default: nessuno")
    parser.add_argument("--param", action="append", default=[], metavar="CAMPO=VALORE",
                        help="override di SimParameters (ripetibile), es. --param code_framework=EN1990; "
                             "i valori sono interpretati come JSON se possibile (true/false/numeri), altrimenti stringa")
    parser.add_argument("--list", action="store_true", help="elenca competitor e scenari default ed esci")
    args = parser.parse_args(argv)
    args.scenarios = _split_list(args.scenarios)
    args.competitors = _split_list(args.competitors)

    if args.list:
        print("Competitor default:")
        for c in DEFAULT_COMPETITORS:
            extra = f" provider={c['provider']} model={c['model']}" if "provider" in c else ""
            print(f"  {c['name']:24s} type={c['type']}{extra}")
        print("Scenari default:", ", ".join(DEFAULT_SCENARIOS))
        return 0

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.competitors_file:
        with open(args.competitors_file, "r", encoding="utf-8") as f:
            competitors = json.load(f)
    else:
        competitors = list(DEFAULT_COMPETITORS)
    if args.competitors:
        wanted = set(args.competitors)
        available = {c["name"] for c in competitors}
        missing = wanted - available
        if missing:
            parser.error(f"competitor sconosciuti: {sorted(missing)}; disponibili: {sorted(available)}")
        competitors = [c for c in competitors if c["name"] in wanted]
    if args.no_llm:
        competitors = [c for c in competitors if c.get("type") not in ("LLM", "LLM Agentico")]
    if args.include_commercial:
        if os.environ.get("OPENAI_API_KEY"):
            competitors.append({"name": "Commercial_GPT_4o", "type": "LLM", "provider": "OpenAI",
                                "model": "gpt-4o", "profile": "Bilanciato"})
        if os.environ.get("GOOGLE_API_KEY"):
            competitors.append({"name": "Commercial_Gemini_Flash", "type": "LLM", "provider": "Google",
                                "model": "gemini-2.0-flash", "profile": "Bilanciato"})
    if not competitors:
        parser.error("nessun competitor selezionato")

    overrides: Dict[str, Any] = {}
    for kv in args.param:
        if "=" not in kv:
            parser.error(f"--param atteso CAMPO=VALORE, trovato {kv!r}")
        k, v = kv.split("=", 1)
        try:
            overrides[k.strip()] = json.loads(v)
        except ValueError:
            overrides[k.strip()] = v.strip()
    if args.regime:
        overrides["regulatory_regime"] = args.regime
    scenarios = _resolve_scenarios(args.scenarios)
    run_arena_campaign(competitors, scenarios, args.seeds, output_dir=args.output_dir,
                       horizon_quarters=args.horizon, param_overrides=overrides or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
