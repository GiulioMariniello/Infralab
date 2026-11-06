"""Example 3. Change the player: fixed rules, planners and a user-written rule on the same short game.

30 bridges, 5 years, law D1. The same game is the one played by people on the board.
"""
import json
import infralab


def inspect_then_fix(observation):
    """A rule written by the user: do the inspections that are due, then restore the worst known bridge."""
    bridges = observation["my_bridges"]
    plan = {b["id"]: "inspect" for b in bridges if b["inspection_overdue"]}
    free = [b for b in bridges if b["id"] not in plan and b["active_maintenance"] is None]
    if free:
        worst = min(free, key=lambda b: b["min_observed_condition"])
        if worst["min_observed_condition"] < 0.3:
            plan[worst["id"]] = "restore"
    return plan


PLAYERS = ["passive", "worst_first", "heuristic", "q_learning", "monte_carlo", inspect_then_fix]

if __name__ == "__main__":
    seeds = infralab.child_seeds(infralab.configuration("paper-short")["root_seed"], 10)
    out = {}
    for player in PLAYERS:
        name = player if isinstance(player, str) else player.__name__
        out[name] = [infralab.run(player, "regbench/aged-30", law="regbench/D1", config="paper-short", seed=s) for s in seeds]
        print(name, sum(r["S1"] for r in out[name]) / len(seeds), flush=True)
        json.dump(out, open("example3.json", "w"), indent=1)
