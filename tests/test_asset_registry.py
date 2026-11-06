"""Test del registry degli asset type e della genericita' del motore.

Il test chiave registra un asset type giocattolo a 2 componenti e verifica che
l'ambiente simuli end-to-end senza alcuna conoscenza dei ponti: e' la garanzia
che aggiungere un nuovo tipo di opera non richiede di toccare il motore.
"""
import pytest

from infralab.assets import available_asset_types, get_asset_type, register_asset_type
from infralab.assets.base import AssetTypeSpec
from infralab.assets.registry import _REGISTRY
from infralab.core.environment import AssetEnv, BridgeEnv
from infralab.settings.config import CdaConfig, SimParameters


def toy_limit_states(env, asset, component):
    condition = max(component.current_condition, 0.02)
    return {
        "sliding": {
            "category": "ULS",
            "failure_mode_type": "ductile",
            "resistance_mean": 2.0 * condition,
            "dead_load_mean": 0.5,
            "live_load_mean": 0.3,
            "code_live_effect": 0.5,
        }
    }


TOY_SPEC = AssetTypeSpec(
    name="_toy_wall",
    display_name="Muro giocattolo",
    component_profiles={
        "paramento": {
            "shape_factor": 1.0, "scale_mean": 1.0, "traffic_sensitivity": 0.3,
            "environment_sensitivity": 1.0, "resistance_bias": 1.0,
            "demand_bias": 0.8, "crack_bias": 1.0,
        },
        "fondazione": {
            "shape_factor": 1.2, "scale_mean": 1.1, "traffic_sensitivity": 0.2,
            "environment_sensitivity": 1.1, "resistance_bias": 1.1,
            "demand_bias": 0.7, "crack_bias": 1.0,
        },
    },
    structural_components=frozenset({"paramento", "fondazione"}),
    default_component_weights={"paramento": 0.6, "fondazione": 0.4},
    limit_state_adapter=toy_limit_states,
    fallback_component="paramento",
)


@pytest.fixture
def toy_registered():
    register_asset_type(TOY_SPEC, overwrite=True)
    yield TOY_SPEC
    _REGISTRY.pop(TOY_SPEC.name, None)


class TestRegistry:
    def test_bridge_registered_by_default(self):
        assert "bridge" in available_asset_types()
        spec = get_asset_type("bridge")
        assert spec.component_names == ("impalcato", "pile", "spalle", "appoggi")
        assert spec.structural_components == frozenset({"impalcato", "pile", "spalle"})

    def test_unknown_asset_type_raises(self):
        with pytest.raises(KeyError, match="non registrato"):
            get_asset_type("astronave")

    def test_duplicate_registration_raises(self):
        spec = get_asset_type("bridge")
        with pytest.raises(ValueError, match="gia' registrato"):
            register_asset_type(spec)

    def test_spec_validation(self):
        with pytest.raises(ValueError, match="fallback_component"):
            AssetTypeSpec(
                name="x", display_name="X", component_profiles={"a": {}},
                structural_components=frozenset({"a"}),
                default_component_weights={"a": 1.0},
                limit_state_adapter=toy_limit_states,
                fallback_component="inesistente",
            )


class TestGenericEngine:
    def make_env(self, toy_spec):
        params = SimParameters(seed=5, agent_names=["A"], num_agents=1)
        env = AssetEnv(params, {}, {"A": {"type": "Umano"}}, CdaConfig(),
                       dict(toy_spec.default_component_weights), asset_type=toy_spec.name)
        env.initialize_bridges([
            {"id": 1, "name": "Muro 1", "length": 40, "width": 6, "adt": 100,
             "eta_iniziale_anni": 15, "vita_utile_anni": 60,
             "incasso_trimestrale": 0, "valore_asset": 500000},
        ])
        return env

    def test_toy_asset_simulates_end_to_end(self, toy_registered):
        env = self.make_env(toy_registered)
        asset = env.bridges[1]
        assert set(asset.components) == {"paramento", "fondazione"}
        assert asset.reliability_index is not None
        for _ in range(4):
            env.step({"A": {"maintenance": {}}})
        assert env.quarter == 5
        assert asset.min_condition < 1.0

    def test_toy_asset_serialization_roundtrip(self, toy_registered):
        env = self.make_env(toy_registered)
        data = env.to_dict()
        assert data["asset_type"] == "_toy_wall"
        clone = BridgeEnv.from_dict(data)
        assert clone.asset_type == "_toy_wall"
        assert set(clone.bridges[1].components) == {"paramento", "fondazione"}

    def test_structural_components_from_spec(self, toy_registered):
        env = self.make_env(toy_registered)
        asset = env.bridges[1]
        assert set(asset.structural_component_names) == {"paramento", "fondazione"}
