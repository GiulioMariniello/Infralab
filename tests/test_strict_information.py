"""Informazione stretta (strict_information=True): nessun segnale derivato dal vero
raggiunge la vista del gestore in observation_mode="inspected".

Per ogni canale chiuso, test a coppie "due mondi, stessa evidenza": i due mondi
differiscono solo per lo stato nascosto. Con strict_information=False la fuga storica
si riproduce; con True l'osservazione del gestore (e classificazione del regime,
trigger della soglia etica, copia belief, piano della prova di carico) e' identica.
1. incertezza epistemica: +10% a un danno vero -> solo da evidenza osservata;
2. prova di carico: alpha e componenti provati scelti sul vero -> sullo stato stimato;
3. componenti mai osservati: vita utile campionata -> vita nominale del profilo;
4. classificazione del regime: beta ed eta' del ponte vero -> stato stimato.
Default: storico (snapshot TRAIN-01 identico).
"""
import json

import pytest

from infralab.core.environment import BridgeEnv
from infralab.core.hazards.earthquake import EarthquakeEvent, apply_earthquake_damage
from infralab.lab.build import build_env_from_scenario
from infralab.lab.leak_check import compare_worlds, main as leak_main, run_leak_check
from infralab.regimes.base import regime_from_dict
from infralab.regimes.registry import register_regime
from infralab.settings.config import SimParameters
from tests.conftest import load_scenario
from tests.test_env_snapshot import SNAPSHOT_PATH, canonical, run_train01_trajectory


def _env(strict, extra=None, seed=3):
    p = {"observation_mode": "inspected", "policy_sandbox": True, "rng_streams": "keyed",
         "strict_information": strict}
    p.update(extra or {})
    return build_env_from_scenario(load_scenario("TRAIN-01"), param_overrides=p, seed=seed)


def _copy(env):
    w = BridgeEnv.from_dict(env.to_dict(), refresh=False)
    w.refresh_network_reliability()
    return w


def _paths(diffs):
    return [p for items in diffs.values() for p, _, _ in items]


# -- default -------------------------------------------------------------------

def test_default_is_off_and_train01_identical():
    assert SimParameters().strict_information is False
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    assert canonical(run_train01_trajectory()) == canonical(golden)


def test_switch_is_validated():
    with pytest.raises(ValueError):
        SimParameters(strict_information="yes")
    with pytest.raises(ValueError):
        SimParameters(strict_uncertainty_drop=-0.1)


def test_default_does_not_store_nominal_life():
    env = _env(False)
    comp = next(iter(next(iter(env.bridges.values())).components.values()))
    assert not hasattr(comp, "nominal_service_life")
    env_s = _env(True)
    comp_s = next(iter(next(iter(env_s.bridges.values())).components.values()))
    assert comp_s.nominal_service_life > 0


# -- canale 1: incertezza epistemica ------------------------------------------------

@pytest.mark.parametrize("strict", [False, True])
def test_hidden_damage_uncertainty_channel(strict):
    a = _copy(_env(strict))
    b = _copy(a)
    for bridge in b.bridges.values():
        comp = bridge.components[sorted(bridge.components)[0]]
        b._apply_hidden_damage(comp, 0.12)
    b.refresh_network_reliability()
    diffs = compare_worlds(a, b, include=("observation", "floor_trigger", "belief_clone", "classification"))
    eps_paths = [p for p in _paths(diffs) if "epistemic_uncertainty" in p]
    if strict:
        assert diffs == {}
    else:
        assert eps_paths, "la fuga storica (incertezza +10% al danno vero) deve riprodursi"


def test_strict_uncertainty_follows_observed_drop():
    env = _env(True, {"strict_uncertainty_drop": 0.05})
    bridge = next(iter(env.bridges.values()))
    comp = bridge.components[sorted(bridge.components)[0]]
    op = {"error_sigma": 0.001, "knowledge_gain": 0.0}
    env._inspect_bridge(bridge, op)
    eps0 = comp.epistemic_uncertainty
    env._apply_hidden_damage(comp, 0.0)
    assert comp.epistemic_uncertainty == eps0          # il danno nascosto non la cambia
    comp.current_condition = max(0.01, comp.current_condition - 0.3)
    env._inspect_bridge(bridge, op)
    assert comp.epistemic_uncertainty == pytest.approx(min(0.40, eps0 * 1.10))


def test_earthquake_damage_uncertainty_flag():
    env = _env(True)
    event = EarthquakeEvent(quarter=1, magnitude=7.5, epicenter_lat=env.params.bridge_region_lat,
                            epicenter_lon=env.params.bridge_region_lon)
    before = {(b.id, n): c.epistemic_uncertainty for b in env.bridges.values() for n, c in b.components.items()}
    apply_earthquake_damage(env.bridges, event, env.params.deterioration_exponent, raise_uncertainty=False)
    after = {(b.id, n): c.epistemic_uncertainty for b in env.bridges.values() for n, c in b.components.items()}
    assert before == after


# -- canale 2: prova di carico ---------------------------------------------------------

def _proof_op(env):
    """Prova di carico del catalogo completo con vincolo di rischio (alpha a griglia)."""
    from dataclasses import asdict
    from infralab.settings.config import MaintOpsConfig
    return "proof_load_test", dict(asdict(MaintOpsConfig())["proof_load_test"], max_failure_probability=1e-3,
                                   alpha_grid_size=4)


def _truth_shift_for_proof(b):
    """Verita' diversa a parita' di evidenza: sensibilita' al traffico (latente) e
    condizione vera, intervalli osservati invariati."""
    for bridge in b.bridges.values():
        names = sorted(bridge.components)
        for i, n in enumerate(names):
            c = bridge.components[n]
            c.traffic_sensitivity = 0.9 if i == 0 else 0.5
            c.current_condition = max(0.05, c.current_condition - 0.25)
    b.refresh_network_reliability()


@pytest.mark.parametrize("strict", [False, True])
def test_proof_load_plan_channel(strict):
    a = _copy(_env(strict))
    b = _copy(a)
    _truth_shift_for_proof(b)
    _, op = _proof_op(a)
    plans_a = {bid: a._proof_load_plan(br, op) for bid, br in a.bridges.items()}
    plans_b = {bid: b._proof_load_plan(br, op) for bid, br in b.bridges.items()}
    if strict:
        assert plans_a == plans_b
    else:
        assert plans_a != plans_b


def test_strict_proof_load_report_alpha_and_components_from_evidence():
    a = _copy(_env(True))
    b = _copy(a)
    _truth_shift_for_proof(b)
    name, op = _proof_op(a)
    bid = sorted(a.bridges)[0]
    a._apply_proof_load(a.bridges[bid], op)
    b._apply_proof_load(b.bridges[bid], op)
    ra, rb = a.bridges[bid].last_proof_load_report, b.bridges[bid].last_proof_load_report
    assert ra["alpha"] == rb["alpha"]
    assert [c["component"] for c in ra["components"]] == [c["component"] for c in rb["components"]]


# -- canale 3: vita utile dei componenti mai osservati ----------------------------------

@pytest.mark.parametrize("strict", [False, True])
def test_never_inspected_uses_nominal_life(strict):
    a = _copy(_env(strict))
    for _ in range(4):
        a.step({n: {"maintenance": {}} for n in a.params.agent_names})
    b = _copy(a)
    for bridge in b.bridges.values():
        for c in bridge.components.values():
            c.vita_utile_anni = int(round(c.vita_utile_anni * 1.3))
    b.refresh_network_reliability()
    diffs = compare_worlds(a, b, include=("observation", "belief_clone", "floor_trigger"))
    if strict:
        assert diffs == {}
    else:
        assert "observation" in diffs and "belief_clone" in diffs


# -- canale 4: classificazione del regime ------------------------------------------------

def _classifying_regime():
    reg = regime_from_dict({
        "id": "strict_cls_test",
        "classification": {"rules": [{"when": {"observed_beta_margin": {"lt": 0.0}}, "cda": "Alta"},
                                     {"when": {"age_years": {"gt": 40}}, "cda": "Medio-Alta"}],
                           "intervals": {"Alta": 2, "Medio-Alta": 4, "Bassa": 8}, "default_cda": "Bassa"},
    })
    register_regime(reg, overwrite=True)
    return reg.id


@pytest.mark.parametrize("strict", [False, True])
def test_regime_classification_channel(strict):
    rid = _classifying_regime()
    a = _copy(_env(strict, {"regulatory_regime": rid, "ethical_floor_enabled": True, "ethical_floor_beta": 3.0}))
    for bridge in a.bridges.values():
        a._inspect_bridge(bridge, {"error_sigma": 0.05, "knowledge_gain": 0.1})
    b = _copy(a)
    for bridge in b.bridges.values():
        for c in bridge.components.values():
            c.current_condition = max(0.05, c.current_condition - 0.3)
            c.effective_age_years += 25.0
    b.refresh_network_reliability()
    diffs = compare_worlds(a, b, include=("classification", "floor_trigger", "regime_triggers"))
    if strict:
        assert diffs == {}
    else:
        assert "classification" in diffs


# -- diagnostica riusabile -----------------------------------------------------------------

def test_leak_check_gate_clean_under_strict_and_flags_default():
    base = {"observation_mode": "inspected", "policy_sandbox": True, "rng_streams": "keyed"}
    strict = run_leak_check("TRAIN-01", params={**base, "strict_information": True}, seed=3, quarters=3)
    assert strict["clean"], strict["divergences"][:5]
    loose = run_leak_check("TRAIN-01", params={**base, "strict_information": False}, seed=3, quarters=3)
    assert not loose["clean"]
    assert loose["divergences_by_output"].get("observation")


def test_leak_check_cli_exit_codes(tmp_path, capsys):
    out = tmp_path / "r.json"
    args = ["--scenario", "TRAIN-01", "--quarters", "2", "--seed", "3",
            "--param", "observation_mode=inspected", "--param", "policy_sandbox=true",
            "--param", "rng_streams=keyed", "--only", "observation"]
    assert leak_main(args + ["--param", "strict_information=true", "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["clean"] is True
    assert leak_main(args + ["--param", "strict_information=false"]) == 1
    assert "FUGA" in capsys.readouterr().out


# -- esposizione -------------------------------------------------------------------------

def test_exposed_in_api_config_and_lab_spec():
    from fastapi.testclient import TestClient

    import backend.app as backend_app
    with TestClient(backend_app.app) as client:
        fields = {f["key"]: f for f in client.get("/api/config").json()["fields"]}
    assert fields["strict_information"]["group"] == "Benchmark"
    assert fields["strict_uncertainty_drop"]["group"] == "Benchmark"
    from infralab.lab.runner import expand_runs
    from infralab.lab.spec import spec_from_dict
    spec = spec_from_dict({"name": "strict", "scenario": "TRAIN-01", "horizon_quarters": 1, "replicates": 1,
                           "seed": 5, "agents": [{"id": "e", "type": "Euristico"}],
                           "grid": {"params.strict_information": [True]}})
    payload = expand_runs(spec, load_scenario("TRAIN-01"))[0]
    assert json.dumps(payload, default=str).count("strict_information") >= 1
