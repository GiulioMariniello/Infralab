"""Test delle azioni e del workflow Linee Guida Ponti 2020 aggiunti al motore:
livelli di ispezione (L0/L1/L2/L4), misure operative (limitazione/chiusura),
manutenzione preventiva, interventi sismici, CdA multi-rischio ed escalation L4.
"""
import pytest

from infralab.lab.build import build_env_from_scenario
from infralab.settings.config import MaintOpsConfig
from tests.conftest import load_scenario


def _env(params=None, bridges=None):
    scen = {
        "scenario_id": "LG", "asset_type": "bridge",
        "params": {"seed": 3, **(params or {})},
        "bridges": bridges or [{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 8000,
                                "eta_iniziale_anni": 45}],
    }
    return build_env_from_scenario(scen, agent_name="A", seed=3)


class TestNewActionsInCatalog:
    def test_all_levels_and_actions_present(self):
        ops = MaintOpsConfig().__dict__
        for a in ["L0_census", "L1_inspect", "L2_inspect", "L4_verify",
                  "limitazione_carico", "senso_unico_alternato", "chiusura_ponte",
                  "manutenzione_ordinaria", "protezione_anticorrosione",
                  "miglioramento_sismico", "adeguamento_sismico"]:
            assert a in ops, a

    def test_inspection_levels_have_decreasing_sigma(self):
        ops = MaintOpsConfig()
        assert ops.L2_inspect["error_sigma"] < ops.L1_inspect["error_sigma"]
        assert ops.L4_verify["error_sigma"] < ops.L2_inspect["error_sigma"]


class TestOperationalMeasures:
    def test_chiusura_reduces_traffic_and_revenue(self):
        env = _env(bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 20000,
                             "eta_iniziale_anni": 40, "incasso_trimestrale": 50000}])
        b = env.bridges[1]
        rev0 = env._get_revenue_multiplier(b)
        traf0 = env._traffic_multiplier(b)
        env.step({"A": {"maintenance": {"1": "chiusura_ponte"}}})
        assert b.traffic_restriction_factor == 0.0
        assert env._get_revenue_multiplier(b) == 0.0          # chiuso non incassa
        assert env._traffic_multiplier(b) < traf0             # nessun traffico
        assert env._get_revenue_multiplier(b) < rev0

    def test_limitazione_then_riapertura(self):
        env = _env()
        env.step({"A": {"maintenance": {"1": "limitazione_carico"}}})
        assert env.bridges[1].traffic_restriction_factor == pytest.approx(0.6)
        env.step({"A": {"maintenance": {"1": "riapertura"}}})
        assert env.bridges[1].traffic_restriction_factor == 1.0


class TestPreventiveMaintenance:
    def test_slows_degradation(self):
        def run(action):
            env = _env(bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 15000,
                                 "eta_iniziale_anni": 50}])
            if action:
                env.step({"A": {"maintenance": {"1": action}}})
            else:
                env.step({"A": {"maintenance": {}}})
            for _ in range(6):
                env.step({"A": {"maintenance": {}}})
            return env.bridges[1].min_condition
        # con protezione anticorrosione la condizione finale è migliore
        assert run("protezione_anticorrosione") > run(None)


class TestSeismicIntervention:
    def test_adeguamento_lowers_seismic_sensitivity(self):
        env = _env(params={"earthquake_enabled": True, "initial_budget": 20_000_000},
                   bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 8000,
                             "eta_iniziale_anni": 40, "lat": 40.85, "lon": 14.27}])
        b = env.bridges[1]
        env.step({"A": {"maintenance": {"1": "adeguamento_sismico"}}})
        # avanza fino a completamento (6 trimestri)
        for _ in range(7):
            env.step({"A": {"maintenance": {}}})
        assert b.seismic_sensitivity is not None and b.seismic_sensitivity <= 0.25 + 1e-9


class TestCdaAndEscalation:
    def test_multi_hazard_never_lowers_condition_cda(self):
        # La regola max() non deve MAI abbassare la CdA da condizione: un asset
        # molto degradato resta Alta anche con traffico/sismica bassi.
        order = {"Sconosciuta": 0, "Bassa": 1, "Medio-Bassa": 2, "Media": 3, "Medio-Alta": 4, "Alta": 5}
        env = _env(params={"earthquake_enabled": True, "cda_multi_hazard": True},
                   bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 500,
                             "eta_iniziale_anni": 80, "lat": 40.85, "lon": 14.27}])
        env.step({"A": {"maintenance": {"1": "L1_inspect"}}})   # imposta la CdA base
        cda_after = env.bridges[1].cda
        # struttura vecchia e degradata -> CdA da condizione alta, non abbassata
        assert order.get(cda_after, 0) >= 3

    def test_multi_hazard_seismic_can_raise(self):
        # Sismica severa può alzare la CdA di un asset in buona condizione.
        order = {"Sconosciuta": 0, "Bassa": 1, "Medio-Bassa": 2, "Media": 3, "Medio-Alta": 4, "Alta": 5}
        def cda_for(multi):
            env = _env(params={"earthquake_enabled": True, "cda_multi_hazard": multi,
                               "seismic_hazard_peak_pga": 1.2, "seismic_hazard_far_pga": 0.6},
                       bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 40000,
                                 "eta_iniziale_anni": 6, "lat": 40.85, "lon": 14.27, "seismic_sensitivity": 2.0}])
            env.step({"A": {"maintenance": {"1": "L1_inspect"}}})
            return order.get(env.bridges[1].cda, 0)
        # con multi-hazard la CdA non è inferiore che senza (il rischio sismico non sottrae)
        assert cda_for(True) >= cda_for(False)

    def test_l4_escalation_flags_overdue(self):
        env = _env(params={"l4_escalation_quarters": 2},
                   bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 8000,
                             "eta_iniziale_anni": 78}])   # molto vecchio -> CdA Alta
        b = env.bridges[1]
        for _ in range(6):
            env.step({"A": {"maintenance": {}}})
        obs = env.get_agent_observations("A")
        bo = obs["my_bridges"][0]
        if b.cda == "Alta":
            assert b.quarters_in_cda_alta_without_L4 > 0
            assert bo["overdue_L4_assessment"] is True

    def test_l4_verify_resets_escalation(self):
        env = _env(params={"l4_escalation_quarters": 2},
                   bridges=[{"id": 1, "name": "P", "length": 100, "width": 10, "adt": 8000,
                             "eta_iniziale_anni": 78}])
        b = env.bridges[1]
        for _ in range(6):
            env.step({"A": {"maintenance": {}}})
        env.step({"A": {"maintenance": {"1": "L4_verify"}}})
        assert b.last_L4_quarter == env.quarter - 1
        assert b.quarters_in_cda_alta_without_L4 == 0


class TestCensus:
    def test_non_censused_blocks_interventions(self):
        env = _env()
        b = env.bridges[1]
        b.censused = False
        assert env._action_block_reason("A", b, "ripristino_completo") == "not_censused"
        assert env._action_block_reason("A", b, "L0_census") is None      # censimento sempre ok
        assert env._action_block_reason("A", b, "L1_inspect") is None
        # L0 ripristina la gestibilità
        env.step({"A": {"maintenance": {"1": "L0_census"}}})
        assert b.censused is True
