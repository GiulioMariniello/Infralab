"""Test della validazione di SimParameters, CdaConfig e degli scenari."""
import pytest

from infralab.settings.config import CdaConfig, SimParameters


class TestSimParameters:
    def test_defaults_valid(self):
        SimParameters()

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("years", 0, "years"),
            ("deterioration_exponent", -1.0, "deterioration_exponent"),
            ("reliability_consequence_class", "CC9", "CC1/CC2/CC3"),
            ("reliability_mc_samples", 4, "reliability_mc_samples"),
            ("failure_threshold", 1.5, "failure_threshold"),
            ("traffic_load_model", "SUPERLOAD", "traffic_load_model"),
            ("earthquake_annual_prob", 2.0, "earthquake_annual_prob"),
            ("seed", 1.5, "seed"),
        ],
    )
    def test_invalid_values_raise(self, field, value, match):
        with pytest.raises(ValueError, match=match):
            SimParameters(**{field: value})

    def test_seed_none_allowed(self):
        assert SimParameters(seed=None).seed is None


class TestCdaConfig:
    def test_defaults_valid(self):
        CdaConfig()

    def test_non_decreasing_thresholds_raise(self):
        with pytest.raises(ValueError, match="decrescenti"):
            CdaConfig(bassa_threshold=0.5, medio_bassa_threshold=0.8)

    def test_out_of_range_threshold_raises(self):
        with pytest.raises(ValueError):
            CdaConfig(bassa_threshold=1.5, medio_bassa_threshold=1.2,
                      media_threshold=1.1, medio_alta_threshold=1.05)


class TestScenarioValidation:
    def test_valid_scenarios_pass(self, isolated_persistence):
        p = isolated_persistence
        scenarios = p.load_scenarios()
        assert {s["scenario_id"] for s in scenarios} >= {"TRAIN-01", "EXP-A-BUDGET", "EXP-B-AGING"}

    def test_missing_scenario_id_raises(self, isolated_persistence):
        p = isolated_persistence
        with pytest.raises(p.ScenarioValidationError, match="scenario_id"):
            p.validate_scenario({"bridges": []})

    def test_bridge_missing_keys_raises(self, isolated_persistence):
        p = isolated_persistence
        scenario = {"scenario_id": "X", "bridges": [{"id": 1, "name": "P1"}]}
        with pytest.raises(p.ScenarioValidationError, match="chiavi obbligatorie"):
            p.validate_scenario(scenario)

    def test_unknown_allowed_action_is_warning(self, isolated_persistence):
        p = isolated_persistence
        scenario = {
            "scenario_id": "X",
            "bridges": [],
            "allowed_actions": ["L1_inspect", "azione_inventata"],
        }
        problems = p.validate_scenario(scenario)
        assert len(problems) == 1
        assert "azione_inventata" in problems[0]
