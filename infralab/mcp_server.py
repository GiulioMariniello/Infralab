"""Model Context Protocol server of InfraLab.

Any MCP client (a desktop assistant, an IDE agent, a script) can manage a bridge stock through these
tools: list scenarios and laws, start a run, read what the owner knows, apply a plan each quarter,
read the outcome, and run paired-seed contrasts of two laws.

    pip install "infralab[mcp]"
    infralab-mcp                      # stdio transport (default)
    infralab-mcp --transport sse      # HTTP/SSE transport

The server holds no key and calls no model. Set INFRALAB_RUN_DIR to keep a JSON record of every finished run.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from infralab.toolkit import Toolkit, ToolError

INSTRUCTIONS = (
    "InfraLab simulates the management of a stock of bridges by one owner, quarter by quarter. "
    "The true condition of the bridges is hidden: you see what inspections last observed. "
    "Typical use: list_scenarios and list_laws; new_run; then repeat observe -> (asset_actions) -> step until done; "
    "score gives the outcome. A plan maps asset ids to action names, at most one action per asset per quarter; "
    "an empty plan does nothing. Laws may force inspections and an ethical floor may order safety measures at the "
    "owner's expense: each step returns a report of what happened. compare_laws runs the paired-seed protocol."
)


def build_server(log_dir: Optional[str] = None):
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP

    tk = Toolkit(log_dir=log_dir)
    mcp = FastMCP("infralab", instructions=INSTRUCTIONS)

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except ToolError as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_scenarios() -> List[Dict[str, Any]]:
        """Scenarios shipped with the library: identifier, name, number of assets, asset type."""
        return guard(tk.list_scenarios)

    @mcp.tool()
    def list_laws() -> List[str]:
        """Identifiers of the laws (regulatory regimes) that can be put in force in a run."""
        return guard(tk.list_laws)

    @mcp.tool()
    def describe_law(law: str) -> Dict[str, Any]:
        """The content of a law: obligations (trigger, action, deadline), enforcement, attention classes."""
        return guard(tk.describe_law, law)

    @mcp.tool()
    def list_policies() -> Dict[str, Any]:
        """Names of the supplied policies (fixed rules, heuristic, planners) usable with run_policy and compare_laws."""
        return guard(tk.list_policies)

    @mcp.tool()
    def new_run(scenario: str = "regbench/aged-30", law: Optional[str] = "regbench/D1", config: str = "paper",
                seed: int = 0, horizon_years: float = 10.0, label: str = "") -> Dict[str, Any]:
        """Start a run to be managed step by step. Returns run_id and the rules (horizon, budget, law, ethical floor)."""
        return guard(tk.new_run, scenario, law, config, seed, horizon_years, label)

    @mcp.tool()
    def rules(run_id: str) -> Dict[str, Any]:
        """The rules of a run: goal, horizon, law in force, ethical floor, information rule."""
        return guard(tk.rules, run_id)

    @mcp.tool()
    def observe(run_id: str, asset_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """What the owner knows now: quarter, budget and, per asset, the condition ranges last observed, the estimated
        reliability index against the target, inspection deadlines, legal flags and works in progress."""
        return guard(tk.observe, run_id, asset_ids)

    @mcp.tool()
    def asset_actions(run_id: str, asset_id: str) -> Dict[str, Any]:
        """Actions available now on one asset, with cost; blocked actions carry the reason."""
        return guard(tk.asset_actions, run_id, asset_id)

    @mcp.tool()
    def action_catalogue(run_id: str) -> List[Dict[str, Any]]:
        """All actions of the scenario: type, duration, unit cost, emissions and effect on each component."""
        return guard(tk.action_catalogue, run_id)

    @mcp.tool()
    def step(run_id: str, plan: Optional[Dict[str, str]] = None, quarters: int = 1, reason: str = "") -> Dict[str, Any]:
        """Apply a plan {asset_id: action} for this quarter and advance. With quarters > 1 the next quarters pass with no
        action and stop early at a collapse, an ethical-floor intervention or a dropped request. Returns a report per
        quarter: accepted and dropped requests, spending, inspections forced by the law, floor interventions, collapses."""
        return guard(tk.step, run_id, plan, quarters, reason)

    @mcp.tool()
    def score(run_id: str) -> Dict[str, Any]:
        """Outcome of a run. During the run: spending, user cost, emissions, collapses. At the end also the safety
        outcome S1 and the expected collapses, which are computed on the true state."""
        return guard(tk.score, run_id)

    @mcp.tool()
    def run_policy(policy: str = "passive", scenario: str = "regbench/aged-30", law: Optional[str] = "regbench/D1",
                   config: str = "paper", seed: int = 0, horizon_years: float = 10.0) -> Dict[str, Any]:
        """Run a supplied policy from start to end on the same setting and seed, as a baseline for a managed run."""
        return guard(tk.run_policy, policy, scenario, law, config, seed, horizon_years)

    @mcp.tool()
    def compare_laws(law_a: Optional[str], law_b: Optional[str], policy: str = "passive",
                     scenario: str = "regbench/aged-30", config: str = "paper", n_seeds: int = 5,
                     horizon_years: float = 10.0) -> Dict[str, Any]:
        """Paired-seed contrast of two laws for one policy: mean difference (law_b minus law_a) of safety, spending,
        user cost and emissions with a bootstrap 95 percent interval over seeds. Can take minutes."""
        return guard(tk.compare_laws, law_a, law_b, policy, scenario, config, n_seeds, horizon_years)

    return mcp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="infralab-mcp", description="InfraLab MCP server")
    ap.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    ap.add_argument("--run-dir", default=None, help="folder for the JSON record of finished runs")
    a = ap.parse_args(argv)
    build_server(a.run_dir).run(transport=a.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
