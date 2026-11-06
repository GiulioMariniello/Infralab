"""Command line of the library.

    infralab list                                   scenarios, laws, policies, configurations
    infralab run worst_first --law regbench/D1      one policy, one run
    infralab compare regbench/D0 regbench/D1        paired-seed contrast of two laws
    infralab play anthropic:claude-opus-5-5         a language model manages a run through tools
    infralab mcp                                    start the MCP server (stdio)
"""
import argparse
import json
import sys


def _common(p):
    p.add_argument("--scenario", default="regbench/aged-30")
    p.add_argument("--config", default="paper")
    p.add_argument("--years", type=float, default=10.0, help="horizon in years")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="infralab", description="InfraLab benchmark environment")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="catalogues of the library")
    p = sub.add_parser("run", help="run one policy")
    p.add_argument("policy")
    p.add_argument("--law", default=None)
    p.add_argument("--seed", type=int, default=0)
    _common(p)
    p = sub.add_parser("compare", help="paired-seed contrast of two laws (B minus A)")
    p.add_argument("law_a")
    p.add_argument("law_b")
    p.add_argument("--policy", default="passive")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=1)
    _common(p)
    p = sub.add_parser("play", help="a language model manages one run")
    p.add_argument("model", help="anthropic:<model> or openai:<model>")
    p.add_argument("--law", default="regbench/D1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", default="balanced")
    p.add_argument("--base-url", default=None)
    p.add_argument("--run-dir", default=None)
    _common(p)
    sub.add_parser("mcp", help="start the MCP server on stdio")
    a = ap.parse_args(argv)

    from infralab import api
    none = lambda x: None if x in (None, "none", "None") else x
    if a.cmd == "list":
        out = {"scenarios": api.scenarios(), "laws": api.laws(), "policies": api.policies(),
               "fixed_rules": api.basic_policies(), "configurations": api.configurations()}
    elif a.cmd == "run":
        out = api.run(a.policy, a.scenario, law=none(a.law), config=a.config, seed=a.seed, horizon_years=a.years)
    elif a.cmd == "compare":
        r = api.compare_laws(none(a.law_a), none(a.law_b), policy=a.policy, scenario=a.scenario, config=a.config,
                             n_seeds=a.seeds, horizon_years=a.years, workers=a.workers)
        out = {k: r[k] for k in ("law_a", "law_b", "policy", "seeds", "mean_a", "mean_b", "difference_b_minus_a")}
    elif a.cmd == "play":
        from infralab.llm import play
        kw = {"base_url": a.base_url} if a.base_url else {}
        out = play(a.model, a.scenario, law=none(a.law), config=a.config, seed=a.seed, horizon_years=a.years,
                   profile=a.profile, log_dir=a.run_dir, verbose=True, **kw)
        out.pop("log", None)
    else:
        from infralab.mcp_server import main as mcp_main
        return mcp_main([])
    json.dump(out, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
