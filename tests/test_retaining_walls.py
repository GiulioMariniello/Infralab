"""Test dell'asset type retaining_wall: registrazione, fisica, run end-to-end."""
import pytest

from infralab.assets import get_asset_type
from infralab.lab.build import build_env_from_scenario
from infralab.lab.runner import run_one
from infralab.lab.spec import spec_from_dict
from infralab.lab.runner import expand_runs
from tests.conftest import load_scenario


@pytest.fixture
def wall_scenario():
    return load_scenario("WALL-DEMO")


class TestSpecRegistration:
    def test_registered(self):
        spec = get_asset_type("retaining_wall")
        assert spec.component_names == ("paramento", "fondazione", "drenaggio")
        assert spec.structural_components == frozenset({"paramento", "fondazione"})
        assert "pulizia_drenaggio" in spec.maintenance_ops

    def test_scenario_validates(self, isolated_persistence, wall_scenario):
        problems = isolated_persistence.validate_scenario(wall_scenario)
        assert problems == []


class TestWallEnvironment:
    def make_env(self, wall_scenario, seed=11):
        return build_env_from_scenario(wall_scenario, agent_name="W",
                                       agent_config={"type": "Euristico"}, seed=seed)

    def test_components_and_limit_states(self, wall_scenario):
        env = self.make_env(wall_scenario)
        wall = env.bridges[1]
        assert set(wall.components) == {"paramento", "fondazione", "drenaggio"}
        states = env._component_limit_states(wall, wall.components["paramento"])
        assert set(states) == {"sliding", "overturning", "cracking"}
        states_f = env._component_limit_states(wall, wall.components["fondazione"])
        assert set(states_f) == {"bearing_capacity", "base_sliding"}

    def test_reliability_computed(self, wall_scenario):
        env = self.make_env(wall_scenario)
        for wall in env.bridges.values():
            assert wall.reliability_index is not None
            assert 0 < wall.annual_failure_probability < 1

    def test_clogged_drainage_increases_demand(self, wall_scenario):
        env = self.make_env(wall_scenario)
        wall = env.bridges[1]
        paramento = wall.components["paramento"]
        base_states = env._component_limit_states(wall, paramento)
        wall.components["drenaggio"].current_condition = 0.2
        clogged_states = env._component_limit_states(wall, paramento)
        assert clogged_states["sliding"]["dead_load_mean"] > base_states["sliding"]["dead_load_mean"]

    def test_wall_maintenance_op_applies(self, wall_scenario):
        env = self.make_env(wall_scenario)
        assert "pulizia_drenaggio" in env.maint_ops
        wall = env.bridges[3]  # muro piu' vecchio
        drainage_before = wall.components["drenaggio"].current_condition
        env.step({"W": {"maintenance": {"3": "pulizia_drenaggio"}}})
        # intervento con durata 1 trimestre: si completa al trimestre successivo
        env.step({"W": {"maintenance": {}}})
        assert wall.components["drenaggio"].current_condition > drainage_before

    def test_heuristic_campaign_end_to_end(self, wall_scenario):
        raw = {
            "name": "wall-unit",
            "scenario": "WALL-DEMO",
            "horizon_quarters": 6,
            "agents": [{"id": "heuristic", "type": "Euristico"}],
            "replicates": 1,
            "seed": 3,
        }
        spec = spec_from_dict(raw)
        payload = expand_runs(spec, wall_scenario)[0]
        run_row, steps = run_one(payload)
        assert run_row["quarters_survived"] == 6
        assert len(steps) == 6
