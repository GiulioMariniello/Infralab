"""Test delle nuove tipologie (wind_turbine, building, solar_tracker) e
delle estensioni di AssetTypeSpec (schema, hook rate/ricavi/vite, sismica).

Valori e accoppiamenti concordati nel dialogo architetto-strutturista:
- turbina: vite IEC 61400 (fondazione 30 / torre 25 / pale 22 / drivetrain 12),
  squilibrio aerodinamico cap 1.30, grouted connection ULS fragile;
- edificio: classe d'uso NTC (IV raddoppia le vite strutturali), punzonamento
  ULS fragile, accoppiamento sul RATE cap x1.8;
- tracker: mancato stow cap 3.5x, sfilamento micropalo, ricavi dai moduli.
"""
import pytest

from infralab.assets import get_asset_type
from infralab.assets.buildings.limit_states import building_degradation_rate, building_lifetime
from infralab.lab.build import build_env_from_scenario

BASE = {"length": 100, "width": 10, "adt": 3000, "incasso_trimestrale": 40000, "valore_asset": 3000000}


def make_env(asset_type, assets, seed=42, params=None):
    scenario = {
        "scenario_id": "T", "asset_type": asset_type,
        "params": {"seed": seed, **(params or {})},
        "bridges": assets,
    }
    return build_env_from_scenario(scenario, agent_name="A", seed=seed)


class TestRegistry:
    def test_types_registered_with_extensions(self):
        for name in ("wind_turbine", "building", "solar_tracker"):
            spec = get_asset_type(name)
            assert spec.asset_property_schema, name
            assert spec.ui_meta.get("demand_label"), name
            assert "seismic" in spec.hazard_sensitivities, name
            assert spec.component_default_lifetimes, name

    def test_legacy_types_untouched_defaults(self):
        bridge = get_asset_type("bridge")
        assert bridge.hazard_sensitivities.get("seismic") == 1.0   # nessun cambio numerico
        assert bridge.revenue_condition_adapter is None
        assert bridge.degradation_rate_adapter is None


class TestWindTurbine:
    def test_lifetimes_iec(self, ):
        env = make_env("wind_turbine", [{"id": 1, "name": "T1", **BASE}])
        comps = env.bridges[1].components
        # vita ~ default +-10%
        assert 27 <= comps["fondazione"].vita_utile_anni <= 33
        assert 22 <= comps["torre"].vita_utile_anni <= 28
        assert 10 <= comps["drivetrain"].vita_utile_anni <= 14

    def test_grouted_connection_brittle(self):
        env = make_env("wind_turbine", [{"id": 1, "name": "T1", **BASE}])
        b = env.bridges[1]
        ls = env.spec.limit_state_adapter(env, b, b.components["fondazione"])
        assert ls["grouted_connection"]["failure_mode_type"] == "brittle"
        ls_torre = env.spec.limit_state_adapter(env, b, b.components["torre"])
        assert "flange_weld_fatigue" in ls_torre

    def test_aero_imbalance_coupling_capped(self):
        env = make_env("wind_turbine", [{"id": 1, "name": "T1", **BASE}])
        b = env.bridges[1]
        torre = b.components["torre"]
        healthy = env.spec.limit_state_adapter(env, b, torre)["flange_weld_fatigue"]["live_load_mean"]
        b.components["pale"].current_condition = 0.0
        degraded = env.spec.limit_state_adapter(env, b, torre)["flange_weld_fatigue"]["live_load_mean"]
        assert degraded == pytest.approx(healthy * 1.30, rel=1e-6)   # cap +30%

    def test_revenue_follows_production_components(self):
        env = make_env("wind_turbine", [{"id": 1, "name": "T1", **BASE}])
        b = env.bridges[1]
        assert env._get_revenue_multiplier(b) > 0.95
        b.components["drivetrain"].current_condition = 0.5
        assert env._get_revenue_multiplier(b) == pytest.approx(0.5, abs=0.01)
        b.components["drivetrain"].current_condition = 0.0
        assert env._get_revenue_multiplier(b) == pytest.approx(0.2, abs=0.01)  # floor

    def test_gearbox_replacement_is_fixed_cost(self):
        env = make_env("wind_turbine", [{"id": 1, "name": "T1", **BASE}],
                       params={"initial_budget": 2000000})
        costs, _ = env.step({"A": {"maintenance": {"1": "sostituzione_gearbox"}}})
        spent = sum(item["cost"] for item in costs["A"])
        assert spent == pytest.approx(520000, rel=0.01)   # fisso, non x area


class TestBuilding:
    def test_classe_uso_doubles_structural_lifetimes(self):
        assert building_lifetime({"classe_uso": "II"}, "struttura_verticale") == 100.0
        assert building_lifetime({"classe_uso": "IV"}, "struttura_verticale") == 200.0
        assert building_lifetime({"classe_uso": "IV"}, "solai") == 160.0
        # non strutturali invariati
        assert building_lifetime({"classe_uso": "IV"}, "involucro") == 40.0
        assert building_lifetime({}, "struttura_verticale") == 100.0   # default II

    def test_punching_shear_brittle(self):
        env = make_env("building", [{"id": 1, "name": "B1", **BASE}])
        b = env.bridges[1]
        ls = env.spec.limit_state_adapter(env, b, b.components["solai"])
        assert ls["punching_shear"]["failure_mode_type"] == "brittle"
        assert ls["flexure"]["failure_mode_type"] == "ductile"

    def test_envelope_rate_coupling_capped(self):
        env = make_env("building", [{"id": 1, "name": "B1", **BASE}])
        b = env.bridges[1]
        struct = b.components["struttura_verticale"]
        assert building_degradation_rate(env, b, struct) == pytest.approx(1.0, abs=0.01)
        b.components["involucro"].current_condition = 0.0
        assert building_degradation_rate(env, b, struct) == pytest.approx(1.8)   # cap
        # componenti non strutturali non accelerati
        assert building_degradation_rate(env, b, b.components["impianti"]) == 1.0

    def test_rate_coupling_accelerates_aging_in_step(self):
        def run(envelope_condition):
            env = make_env("building", [{"id": 1, "name": "B1", **BASE}], seed=7)
            env.bridges[1].components["involucro"].current_condition = envelope_condition
            age0 = env.bridges[1].components["struttura_verticale"].effective_age_years
            env.step({"A": {"maintenance": {}}})
            return env.bridges[1].components["struttura_verticale"].effective_age_years - age0

        assert run(0.0) == pytest.approx(run(1.0) * 1.8, rel=1e-6)   # stesso seed, stesso rumore


class TestSolarTracker:
    def test_stow_coupling_capped_3_5x(self):
        env = make_env("solar_tracker", [{"id": 1, "name": "S1", **BASE}])
        b = env.bridges[1]
        struct = b.components["struttura_torsione"]
        healthy = env.spec.limit_state_adapter(env, b, struct)["extreme_wind_galloping"]["live_load_mean"]
        b.components["attuatori"].current_condition = 0.0
        stowless = env.spec.limit_state_adapter(env, b, struct)["extreme_wind_galloping"]["live_load_mean"]
        assert stowless == pytest.approx(healthy * 3.5, rel=1e-6)

    def test_stow_failure_hits_reliability(self):
        env = make_env("solar_tracker", [{"id": 1, "name": "S1", **BASE}])
        b = env.bridges[1]
        beta_healthy = b.annual_reliability_index
        b.components["attuatori"].current_condition = 0.1
        env.refresh_bridge_reliability(b)
        assert b.annual_reliability_index < beta_healthy - 1.0   # crollo del margine

    def test_micropile_pullout_present(self):
        env = make_env("solar_tracker", [{"id": 1, "name": "S1", **BASE}])
        b = env.bridges[1]
        ls = env.spec.limit_state_adapter(env, b, b.components["fondazioni_micropali"])
        assert ls["micropile_pullout"]["failure_mode_type"] == "brittle"

    def test_revenue_follows_modules(self):
        env = make_env("solar_tracker", [{"id": 1, "name": "S1", **BASE}])
        b = env.bridges[1]
        b.components["moduli"].current_condition = 0.6
        assert env._get_revenue_multiplier(b) == pytest.approx(0.6, abs=0.01)


class TestSeismicSensitivity:
    def test_type_default_scales_pga(self):
        assets = [{"id": 1, "name": "X", **BASE, "lat": 40.85, "lon": 14.27}]
        env_b = make_env("building", assets, params={"earthquake_enabled": True})
        env_t = make_env("solar_tracker", assets, params={"earthquake_enabled": True})
        pga_b = env_b.bridges[1].seismic_fragility["controlling_pga"]
        pga_t = env_t.bridges[1].seismic_fragility["controlling_pga"]
        # tracker 0.15 vs building 1.0, a parita' di sito e condizione
        assert pga_t == pytest.approx(pga_b * 0.15, rel=0.05)

    def test_per_asset_override(self):
        assets = [{"id": 1, "name": "X", **BASE, "lat": 40.85, "lon": 14.27, "seismic_sensitivity": 0.4}]
        env = make_env("building", assets, params={"earthquake_enabled": True})
        base = make_env("building", [{**assets[0], "seismic_sensitivity": None}],
                        params={"earthquake_enabled": True})
        assert env.bridges[1].seismic_fragility["controlling_pga"] == pytest.approx(
            base.bridges[1].seismic_fragility["controlling_pga"] * 0.4, rel=0.05)

    def test_override_survives_serialization(self):
        from infralab.core.environment import BridgeEnv
        assets = [{"id": 1, "name": "X", **BASE, "seismic_sensitivity": 0.4}]
        env = make_env("building", assets)
        clone = BridgeEnv.from_dict(env.to_dict())
        assert clone.bridges[1].seismic_sensitivity == 0.4
