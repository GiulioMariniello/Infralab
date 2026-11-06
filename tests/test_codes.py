"""Test delle normative (CodeFramework): risoluzione target + default invariato."""
import pytest

from infralab.codes import available_code_frameworks, get_code_framework
from infralab.core.environment import _apply_code_framework
from infralab.settings.config import SimParameters


class TestRegistry:
    def test_builtins_registered(self):
        names = available_code_frameworks()
        assert {"custom", "EN1990", "fib_MC2010", "AASHTO_LRFD"} <= set(names)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="non registrata"):
            get_code_framework("Napoleonico")


class TestResolve:
    def test_custom_is_passthrough(self):
        assert get_code_framework("custom").resolve("CC2") == {}

    @pytest.mark.parametrize("cc,beta", [("CC1", 3.3), ("CC2", 3.8), ("CC3", 4.3)])
    def test_en1990_cc_table(self, cc, beta):
        r = get_code_framework("EN1990").resolve(cc)
        assert r["reliability_target_beta"] == beta
        assert r["reliability_reference_period_years"] == 50

    def test_aashto_different_target_and_period(self):
        r = get_code_framework("AASHTO_LRFD").resolve("CC2")
        assert r["reliability_target_beta"] == 3.5
        assert r["reliability_reference_period_years"] == 75

    def test_resolve_clears_manual_annual_override(self):
        assert get_code_framework("EN1990").resolve("CC2")["reliability_target_beta_annual"] is None


class TestApplyToParams:
    def test_custom_leaves_params_unchanged(self):
        p = SimParameters(reliability_target_beta=4.0)  # profilo manuale
        out = _apply_code_framework(p)
        assert out.reliability_target_beta == 4.0  # custom non tocca

    def test_framework_overrides_target(self):
        p = SimParameters(code_framework="AASHTO_LRFD", reliability_consequence_class="CC2", reliability_target_beta=4.0)
        out = _apply_code_framework(p)
        assert out.reliability_target_beta == 3.5
        assert out.reliability_reference_period_years == 75

    def test_unknown_framework_raises_valueerror(self):
        p = SimParameters(code_framework="Inesistente")
        with pytest.raises(ValueError):
            _apply_code_framework(p)
