"""Test importer FHWA NBI e contabilità di sostenibilità (CO2e)."""
import math
import os

import pytest

from infralab.core.environment import _calculate_emissions_tco2
from infralab.data.nbi import (
    condition_to_effective_age,
    dms_packed_to_decimal,
    nbi_to_bridges,
    parse_nbi_file,
    rating_to_condition,
)
from infralab.lab.build import build_env_from_scenario
from tests.conftest import REPO_ROOT, load_scenario

NBI_FILE = os.path.join(REPO_ROOT, "data_sources", "DE23.txt")


class TestNbiParsing:
    def test_dms_conversion(self):
        # 39°46'09.47" N -> 39.7693; 075°34'36.12" W -> -75.5767
        assert dms_packed_to_decimal("39460947") == pytest.approx(39.7693, abs=1e-3)
        assert dms_packed_to_decimal("075343612", is_longitude=True) == pytest.approx(-75.5767, abs=1e-3)
        assert dms_packed_to_decimal("0") is None
        assert dms_packed_to_decimal("N") is None

    def test_rating_mapping(self):
        assert rating_to_condition("9") == 1.0
        assert rating_to_condition("5") == pytest.approx(5 / 9, abs=1e-3)
        assert rating_to_condition("N") is None
        assert rating_to_condition("") is None

    def test_condition_age_inversion_roundtrip(self):
        """L'età efficace invertita riproduce la condizione sulla curva media."""
        from infralab.assets.bridges.profiles import BRIDGE_COMPONENT_PROFILES
        for comp in BRIDGE_COMPONENT_PROFILES:
            for condition in (0.9, 0.7, 0.5):
                age = condition_to_effective_age(condition, comp, life_years=100)
                profile = BRIDGE_COMPONENT_PROFILES[comp]
                k = 3.0 * profile["shape_factor"]
                reproduced = math.exp(-profile["scale_mean"] * (age / 100) ** k)
                assert reproduced == pytest.approx(condition, abs=0.01)

    @pytest.mark.skipif(not os.path.exists(NBI_FILE), reason="file NBI non scaricato")
    def test_real_delaware_file(self):
        df = parse_nbi_file(NBI_FILE)
        bridges = nbi_to_bridges(df, max_bridges=5)
        assert len(bridges) == 5
        for b in bridges:
            assert 38.0 < b["lat"] < 40.5 and -76.5 < b["lon"] < -74.5  # Delaware
            assert b["adt"] > 0 and b["length"] > 2
            assert set(b["nbi_conditions"]) == {"impalcato", "pile", "spalle", "appoggi"}
            for comp in b["nbi_conditions"]:
                assert f"{comp}_eta_iniziale_anni" in b
                assert b[f"{comp}_vita_utile_anni"] == 100

    def test_nbi_scenario_matches_observed_conditions(self, isolated_persistence):
        """La simulazione parte (a meno del rumore di campionamento) dalle condizioni NBI."""
        sc = load_scenario("NBI-DE")
        env = build_env_from_scenario(sc, seed=2023)
        errors = []
        for bridge_data in sc["bridges"]:
            bridge = env.bridges[bridge_data["id"]]
            for comp, target in bridge_data["nbi_conditions"].items():
                errors.append(abs(bridge.components[comp].current_condition - target))
        assert sum(errors) / len(errors) < 0.05


class TestSustainability:
    def test_emission_calculation(self):
        assert _calculate_emissions_tco2({"co2_m2": 450.0}, 1000.0) == pytest.approx(450.0)
        assert _calculate_emissions_tco2({"co2": 8000.0}, 1000.0) == pytest.approx(8.0)
        assert _calculate_emissions_tco2({}, 1000.0) == 0.0

    def test_emissions_accumulate_on_action(self):
        sc = load_scenario("EXP-A-BUDGET")
        env = build_env_from_scenario(sc, agent_name="A", seed=42)
        assert env.emissions_tco2["A"] == 0.0
        env.step({"A": {"maintenance": {"1": "L1_inspect"}}})
        after_inspect = env.emissions_tco2["A"]
        assert after_inspect > 0
        env.step({"A": {"maintenance": {"2": "L1_inspect"}}})
        assert env.emissions_tco2["A"] > after_inspect

    def test_no_action_no_emissions(self):
        sc = load_scenario("TRAIN-01")
        env = build_env_from_scenario(sc, agent_name="A", seed=42)
        for _ in range(3):
            env.step({"A": {"maintenance": {}}})
        assert env.emissions_tco2["A"] == 0.0

    def test_emissions_serialization_roundtrip(self):
        from infralab.core.environment import BridgeEnv

        sc = load_scenario("EXP-A-BUDGET")
        env = build_env_from_scenario(sc, agent_name="A", seed=42)
        env.step({"A": {"maintenance": {"1": "L1_inspect"}}})
        clone = BridgeEnv.from_dict(env.to_dict())
        assert clone.emissions_tco2["A"] == pytest.approx(env.emissions_tco2["A"])

    def test_gym_metrics_include_emissions(self):
        from infralab.agents.protocol import AssetEnvGym

        sc = load_scenario("TRAIN-01")

        def factory(seed):
            return build_env_from_scenario(sc, agent_name="A", seed=seed)

        gym = AssetEnvGym(factory, "A", max_quarters=2)
        _, info = gym.reset(seed=1)
        assert "emissions_tco2" in info["metrics"]
