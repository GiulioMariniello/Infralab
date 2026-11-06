"""Modello del costo utenti delle deviazioni: opzioni detour_* (default = storico)."""
import pytest

from infralab.settings.config import SimParameters
from infralab.lab.build import build_env_from_scenario
from tests.conftest import load_scenario


def _env(**kw):
    env = build_env_from_scenario(load_scenario("NBI-DE"), param_overrides={"detour_accounting": True, **kw}, seed=1)
    agent = env.params.agent_names[0]
    b = next(x for x in env.bridges.values() if x.manager == agent)
    b.detour_km, b.adt, b.truck_fraction = 199.0, 1000.0, 0.1
    for o in env.bridges.values():
        if o is not b:
            o.detour_km = 0.0
    return env, agent, b


def test_validazione():
    with pytest.raises(Exception):
        SimParameters(detour_partial_diversion="cars")
    with pytest.raises(Exception):
        SimParameters(detour_km_cap=-1.0)


def test_tetto_km_sulla_chiusura():
    env, a, b = _env()
    b.traffic_restriction_factor = 0.0
    storico = env.detour_user_cost_quarter(a)
    env.params.detour_km_cap = 50.0
    assert env.detour_user_cost_quarter(a) == pytest.approx(storico * 50.0 / 199.0)


def test_limitazione_parziale_solo_mezzi_pesanti():
    env, a, b = _env(detour_partial_diversion="trucks")
    b.traffic_restriction_factor = 0.6
    atteso = 0.1 * 1000.0 * 199.0 * env.params.detour_cost_truck_eur_km * 91.25
    assert env.detour_user_cost_quarter(a) == pytest.approx(atteso)
    b.traffic_restriction_factor = 0.0          # la chiusura vera devia tutti
    assert env.detour_user_cost_quarter(a) > atteso
