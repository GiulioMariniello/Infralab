# InfraLab

InfraLab is an open Python library in which the management of a bridge stock is a game played quarter by
quarter under partial information, a budget and a regulation. The four choices behind a bridge-management
study are exchangeable pieces:

| Piece | Form | Argument |
|---|---|---|
| Stock | a file of bridges with their inventory attributes (an importer reads National Bridge Inventory files) | `scenario` |
| Degradation law | a Python function `degrade(ctx)` that returns the new condition of a component | `degradation` |
| Regulation | a YAML file of duties: trigger, action, deadline, enforcement | `law` |
| Player | a function from the observation to a plan | `policy` |

Every random draw is keyed to seed, process, bridge and quarter, so two settings meet the same events and are
compared seed by seed in one call. A game is replayed exactly from its settings, its seed and the library version.

## Install

```bash
pip install infralab                 # core: numpy, scipy, pandas, PyYAML
pip install "infralab[mcp]"          # MCP server
pip install "infralab[anthropic]"    # Claude players
pip install "infralab[openai]"       # OpenAI and compatible endpoints
pip install "infralab[web]"          # web platform (board for human players)
```

From the source: `pip install .` in the root of the repository.

## Quick start

```python
import infralab

env = infralab.make("regbench/aged-30", law="regbench/D1", seed=0, horizon_years=10)
obs, info = env.reset()
while not env.done:
    plan = {}                                  # {asset_id: action_name}; empty = no action
    obs, reward, terminated, truncated, info = env.step(plan)
print(env.summary())
```

`env.compact_observation()` gives what the owner knows in a few hundred characters per asset;
`env.feasible_actions(asset_id)` the actions available with their cost; `env.rules()` the rules of the run;
`info["report"]` what happened in the quarter (accepted and dropped requests, inspections forced by the
regulation, interventions of the ethical floor, collapses).

## Outcomes

`run`, `Env.summary` and `compare_laws` report outcomes computed on the true state, never shown to a player:

| Key | Meaning |
|---|---|
| `S1` | mean over bridges of the lowest true annual reliability index reached during the game |
| `expected_collapses` | expected strength collapses: sum over bridges and quarters of the quarterly failure probability of the true state |
| `expected_hydraulic_collapses` | expected collapses from progressive hydraulic defects that matured unarrested |
| `bridges_below_limit`, `bridge_years_below_limit` | bridges, and bridge-years, with a true annual index below 3.02 |
| `collapses` | collapses drawn in the game |
| `cash_spend`, `user_cost`, `closed_share` | what the owner paid, the extra travel cost of road users, the closed share of bridge-years |

## Change one piece

### The player

```python
def inspect_then_fix(observation):
    bridges = observation["my_bridges"]
    plan = {b["id"]: "inspect" for b in bridges if b["inspection_overdue"]}
    free = [b for b in bridges if b["id"] not in plan and b["active_maintenance"] is None]
    if free:
        worst = min(free, key=lambda b: b["min_observed_condition"])
        if worst["min_observed_condition"] < 0.3:
            plan[worst["id"]] = "restore"
    return plan

infralab.run(inspect_then_fix, "regbench/aged-30", law="regbench/D1", config="paper-short")
```

`infralab.policies()` lists the supplied players (`passive`, `worst_first`, `typical`, `diligent`, `heuristic`,
`monte_carlo`, `q_learning`, `genetic`, `memetic`); `basic:<rule>` reaches the other fixed rules
(`infralab.basic_policies()`); `infralab.actions()` lists the action names accepted in a plan.

### The regulation

```python
r = infralab.compare_laws("regbench/D0", "regbench/D1", policy="passive",
                          scenario="regbench/aged-147", config="paper",
                          params={"ethical_floor_enabled": True}, n_seeds=10, workers=5)
r["difference_b_minus_a"]["S1"]                # mean paired difference, lower and upper bound (95 percent)
```

The two regulations are played on the same stock, player and random draws. A regulation is a YAML file:
pass a path instead of an identifier to test your own (`examples/04_inspection_interval.py` writes nine).

### The degradation law

```python
# markov.py: a Markov chain on inventory ratings
P = {9: 0.1483, 8: 0.0672, 7: 0.0334, 6: 0.0321, 5: 0.0355, 4: 0.0421, 3: 0.0428, 2: 0.0852, 1: 0.0193}
D = {9: 1.12, 8: 1.06, 7: 1.08, 6: 1.14, 5: 1.27, 4: 1.35, 3: 2.22, 2: 1.80, 1: 1.00}

def degrade(ctx):
    c = ctx["condition"]
    if ctx["age_increment_years"] <= 0:
        return c
    r = min(9, max(1, round(9 * c)))
    if ctx["rand"]() < 1 - (1 - P[r]) ** 0.25:
        c -= D[r] / 9
    return c
```

```python
infralab.compare_laws("regbench/D0", "regbench/D1", policy="passive", scenario="regbench/aged-147",
                      config="paper", degradation="markov.py", n_seeds=10, workers=5)
```

`infralab.degradation_laws()` lists the laws supplied: `weibull` (default), `gamma`, `regbench/power`,
`regbench/markov-nbi`.

### The stock

`infralab.data.nbi` reads a National Bridge Inventory file and builds a scenario
(`python -m infralab.data.nbi --help`); `infralab.load_scenario(path)` reads any scenario file.

## The benchmark of the paper

| Item | Identifiers |
|---|---|
| Stocks | `regbench/aged-147`, `regbench/aged-30`: aged samples of the 2023 Delaware inventory |
| Regulations | `regbench/D0` no obligation; `regbench/N24` routine inspection every 24 months, 12 for rating 3 or lower; `regbench/D1` intervals graded by attention class; `D2` post-event; `D3` both; `K0` flat biennial; `G0`–`G3`, `J0`–`J1` with a practice rule |
| Configurations | `paper` (147 bridges, 50 years; default), `paper-short` (30 bridges, 5 years), `paper-less-optimistic` (sensitivity of the reliability calibration); earlier configurations `first`, `second`, `third`, `calibrated`, `calibrated-short` |

`examples/` holds the examples of the paper: `01_change_the_regulation.py`, `02_change_the_degradation_law.py`,
`03_change_the_player.py` and `04_inspection_interval.py`. Called as in these scripts, the library reproduces
the stored games of the paper to the last digit (`tests/paper_reproduction_check.py`).

The parameter values are plausible and, apart from the matching of the reliability index to load ratings, not
calibrated. The reliability index serves paired comparisons between settings, not levels; results carry to a
real stock only through a stock and a degradation law fitted to it.

## Interfaces

**Command line**

```bash
infralab list
infralab run worst_first --law regbench/D1 --years 10
infralab compare regbench/D0 regbench/D1 --policy passive --seeds 10 --years 10
```

**MCP server.** Any Model Context Protocol client can manage a stock through tools.

```bash
infralab-mcp                         # stdio
infralab-mcp --transport sse         # HTTP
```

```json
{"mcpServers": {"infralab": {"command": "infralab-mcp", "env": {"INFRALAB_RUN_DIR": "runs"}}}}
```

Tools: `list_scenarios`, `list_laws`, `describe_law`, `list_policies`, `new_run`, `rules`, `observe`,
`asset_actions`, `action_catalogue`, `step`, `score`, `run_policy`, `compare_laws`. The server holds no key and
calls no model. During a run it returns only what the owner may know; the outcomes computed on the true state
are released when the run is over.

**Language-model players.**

```python
from infralab.llm import play

play("anthropic:claude-opus-5-5", "regbench/aged-30", law="regbench/D1", seed=0)
play("openai:gpt-4.1-mini", "regbench/aged-30", law="regbench/D1", seed=0)
play("openai:llama3.1", base_url="http://localhost:11434/v1")     # any endpoint with the same protocol
```

Keys are read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the variable named with `api_key_env`).

**Web platform.** A board for human players (FastAPI backend and React frontend, see `docs/WEBAPP.md`), and the
Streamlit interface `app.py`.

## Tests

```bash
pip install ".[dev]"
pytest
```

## Citation

See `CITATION.cff`. Model documentation: `docs/MODEL_REPORT.md`; numerical changes to the engine:
`docs/CHANGELOG_MODEL.md`.

## License

MIT, see `LICENSE`.
