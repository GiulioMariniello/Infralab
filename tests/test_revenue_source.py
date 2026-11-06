"""Ricavi e stato nascosto: revenue_health_source (RT012 del red team di infralab-regbench)."""
import pytest

from infralab.lab.leak_check import run_leak_check
from infralab.settings.config import SimParameters

BASE = {"observation_mode": "inspected", "policy_sandbox": True, "strict_information": True,
        "enforce_guidelines": True}


def test_valori_ammessi():
    assert SimParameters().revenue_health_source == "truth"
    with pytest.raises(Exception):
        SimParameters(revenue_health_source="true")


def test_ricavo_dal_vero_e_una_fuga():
    rep = run_leak_check("NBI-DE", params={**BASE, "revenue_health_source": "truth"}, seed=1,
                         quarters=6, policy="inspect-only", trajectory=True)
    assert rep["revenue_truth_dependence"]
    assert not rep["clean"]


@pytest.mark.parametrize("source", ["assessed", "none"])
def test_ricavo_dalla_stima_o_indipendente_e_pulito(source):
    rep = run_leak_check("NBI-DE", params={**BASE, "revenue_health_source": source}, seed=1,
                         quarters=6, policy="inspect-only", trajectory=True)
    assert not rep["revenue_truth_dependence"]
    assert rep["clean"]
