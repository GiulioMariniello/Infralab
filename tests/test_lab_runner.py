"""Test del lab engine: spec, espansione run, esecuzione golden, protocol agenti."""
import json
import os

import pytest

from infralab.agents.adapters import make_agent
from infralab.agents.protocol import AssetEnvGym, RewardConfig
from infralab.lab.build import build_env_from_scenario
from infralab.lab.runner import expand_runs, run_experiment, run_one
from infralab.lab.spec import SpecError, load_spec, spec_from_dict
from tests.conftest import load_scenario

RAW_SPEC = {
    "name": "unit-test",
    "scenario": "TRAIN-01",
    "horizon_quarters": 4,
    "agents": [
        {"id": "heuristic-a", "type": "Euristico"},
        {"id": "heuristic-b", "type": "Euristico"},
    ],
    "grid": {"params.env_degradation_std": [0.08, 0.12]},
    "replicates": 2,
    "seed": 123,
    "outputs": {"dir": "results/unit-test", "formats": ["csv"]},
}


class TestSpec:
    def test_valid_spec_parses(self):
        spec = spec_from_dict(RAW_SPEC)
        assert spec.run_count == 2 * 2 * 2
        assert len(spec.grid_points()) == 2

    @pytest.mark.parametrize(
        "mutation,match",
        [
            ({"agents": []}, "agents"),
            ({"agents": [{"id": "a", "type": "X"}, {"id": "a", "type": "Y"}]}, "duplicato"),
            ({"grid": {"env_degradation_std": [0.1]}}, "params\\."),
            ({"replicates": 0}, "replicates"),
            ({"outputs": {"formats": ["xlsx"]}}, "xlsx"),
        ],
    )
    def test_invalid_specs_raise(self, mutation, match):
        raw = {**RAW_SPEC, **mutation}
        with pytest.raises(SpecError, match=match):
            spec_from_dict(raw)

    def test_yaml_roundtrip(self, tmp_path):
        import yaml

        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(RAW_SPEC), encoding="utf-8")
        spec = load_spec(str(path))
        assert spec.name == "unit-test"


class TestExpansion:
    def test_deterministic_seeds(self):
        spec = spec_from_dict(RAW_SPEC)
        scenario = load_scenario("TRAIN-01")
        runs_a = expand_runs(spec, scenario)
        runs_b = expand_runs(spec, scenario)
        assert [r["seed"] for r in runs_a] == [r["seed"] for r in runs_b]
        assert len({r["seed"] for r in runs_a}) == len(runs_a)  # tutti distinti

    def test_stable_ordering(self):
        spec = spec_from_dict(RAW_SPEC)
        runs = expand_runs(spec, load_scenario("TRAIN-01"))
        assert runs[0]["agent_id"] == "heuristic-a"
        assert [r["run_id"] for r in runs] == list(range(len(runs)))


class TestRunOne:
    def make_payload(self, seed=7):
        spec = spec_from_dict({**RAW_SPEC, "replicates": 1,
                               "agents": [{"id": "heuristic", "type": "Euristico"}],
                               "grid": {}})
        payload = expand_runs(spec, load_scenario("TRAIN-01"))[0]
        payload["seed"] = seed
        return payload

    def test_run_completes_horizon(self):
        run_row, steps = run_one(self.make_payload())
        assert run_row["quarters_survived"] == 4
        assert len(steps) == 4
        assert run_row["truncated"] and not run_row["terminated"]
        assert "final_health_index" in run_row

    def test_run_row_has_benchmark_metrics(self):
        run_row, steps = run_one(self.make_payload())
        for key in ("duration_s", "final_min_beta", "final_min_beta_trajectory", "final_pf_net",
                    "final_spent_total", "final_invalid_actions_total", "final_overdue_asset_quarters",
                    "final_quarters_below_target", "final_net_worth"):
            assert key in run_row, key
        assert run_row["final_min_beta_trajectory"] <= run_row["final_min_beta"] + 1e-12
        assert run_row["final_spent_total"] == sum(s["metric_spent_quarter"] for s in steps)

    def test_run_deterministic(self):
        row_a, steps_a = run_one(self.make_payload(seed=7))
        row_b, steps_b = run_one(self.make_payload(seed=7))
        # La durata wall-clock e' legittimamente non deterministica.
        row_a.pop("duration_s"); row_b.pop("duration_s")
        assert json.dumps(row_a, sort_keys=True, default=str) == json.dumps(row_b, sort_keys=True, default=str)
        assert json.dumps(steps_a, sort_keys=True, default=str) == json.dumps(steps_b, sort_keys=True, default=str)

    @pytest.mark.parametrize("agent", [
        {"id": "ga", "type": "GA Orizzonte Variabile", "params": {"ga_params": {"horizon": 1, "pop_size": 4, "generations": 2}}},
        {"id": "mcts", "type": "MCTS", "params": {"mcts_params": {"n_rollouts": 3, "rollout_depth": 1, "n_candidates": 3}}},
        {"id": "rl", "type": "RL Q-Learning", "params": {}},
    ])
    def test_search_agents_reproducible_by_seed(self, agent):
        """GA/MCTS/RL usano il random globale: il seed della run deve renderli riproducibili."""
        spec = spec_from_dict({**RAW_SPEC, "replicates": 1, "agents": [agent], "grid": {}})
        payload = expand_runs(spec, load_scenario("TRAIN-01"))[0]
        payload["seed"] = 11
        row_a, steps_a = run_one(dict(payload))
        row_b, steps_b = run_one(dict(payload))
        row_a.pop("duration_s"); row_b.pop("duration_s")
        assert json.dumps(row_a, sort_keys=True, default=str) == json.dumps(row_b, sort_keys=True, default=str)
        assert json.dumps(steps_a, sort_keys=True, default=str) == json.dumps(steps_b, sort_keys=True, default=str)

    def test_different_seed_different_trajectory(self):
        _, steps_a = run_one(self.make_payload(seed=7))
        _, steps_b = run_one(self.make_payload(seed=8))
        assert json.dumps(steps_a, default=str) != json.dumps(steps_b, default=str)


class TestRunExperiment:
    def test_full_campaign(self):
        spec = spec_from_dict({**RAW_SPEC, "replicates": 1, "horizon_quarters": 2})
        results = run_experiment(spec)
        assert not results["errors"]
        assert len(results["runs"]) == spec.run_count
        assert len(results["steps"]) == spec.run_count * 2
        # ogni run porta con se' il punto griglia
        assert {r["params.env_degradation_std"] for r in results["runs"]} == {0.08, 0.12}


class TestProtocol:
    def test_gym_wrapper_with_legacy_heuristic(self):
        scenario = load_scenario("TRAIN-01")

        def factory(seed):
            return build_env_from_scenario(scenario, agent_name="H",
                                           agent_config={"type": "Euristico"}, seed=seed)

        gym = AssetEnvGym(factory, "H", RewardConfig(), max_quarters=3)
        obs, info = gym.reset(seed=11)
        agent = make_agent("H", {"type": "Euristico"})
        agent.reset(obs, info)
        rewards = []
        for _ in range(3):
            obs, reward, terminated, truncated, info = gym.step(agent.act(obs))
            rewards.append(reward)
            if terminated or truncated:
                break
        assert len(rewards) >= 1
        assert all(isinstance(r, float) for r in rewards)

    def test_feasible_actions_shape(self):
        scenario = load_scenario("TRAIN-01")

        def factory(seed):
            return build_env_from_scenario(scenario, agent_name="H",
                                           agent_config={"type": "Euristico"}, seed=seed)

        gym = AssetEnvGym(factory, "H")
        gym.reset(seed=1)
        feasible = gym.feasible_actions()
        assert feasible  # almeno un asset
        for actions in feasible.values():
            assert isinstance(actions, (list, dict))


class TestSeedScope:
    def test_replicate_scope_pairs_seeds_across_agents_and_grid(self):
        spec = spec_from_dict({**RAW_SPEC, "replicates": 3, "seed_scope": "replicate",
                               "agents": [{"id": "a", "type": "Euristico"}, {"id": "b", "type": "Basic", "params": {"policy": "do_nothing"}}],
                               "grid": {"params.regulatory_regime": ["none", "it_lg2020_forced"]}})
        payloads = expand_runs(spec, load_scenario("TRAIN-01"))
        by_rep = {}
        for p in payloads:
            by_rep.setdefault(p["replicate"], set()).add(p["seed"])
        assert all(len(v) == 1 for v in by_rep.values())
        assert len({next(iter(v)) for v in by_rep.values()}) == 3   # repliche diverse -> seed diversi
        assert len({p["run_id"] for p in payloads}) == len(payloads)

    def test_run_scope_is_historical_default(self):
        spec = spec_from_dict({**RAW_SPEC, "replicates": 2, "grid": {},
                               "agents": [{"id": "a", "type": "Euristico"}, {"id": "b", "type": "Euristico"}]})
        assert spec.seed_scope == "run"
        seeds = [p["seed"] for p in expand_runs(spec, load_scenario("TRAIN-01"))]
        assert len(set(seeds)) == len(seeds)

    def test_unknown_key_and_bad_scope_rejected(self):
        with pytest.raises(SpecError):
            spec_from_dict({**RAW_SPEC, "seed_scopee": "replicate"})
        with pytest.raises(SpecError):
            spec_from_dict({**RAW_SPEC, "seed_scope": "agent"})
