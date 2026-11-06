from infralab.calibration_profiles import apply_calibration_profile, default_maint_ops_dict
from infralab.settings.config import SimParameters


def test_profilo_b_prezza_appoggi_a_metro_quadro():
    _, ops = apply_calibration_profile(SimParameters(), default_maint_ops_dict(), "fhwa_unit_costs_2025_b")
    op = ops["replace_appoggi"]
    assert op["cost"] == 0 and op["cost_m2"] == 190.0
    assert op["gains"]["appoggi"] == 1.0 and op["gains"]["impalcato"] == 0.05
    assert ops["ripristino_completo"]["cost_m2"] == 2040.0


def test_profilo_originale_invariato():
    _, ops = apply_calibration_profile(SimParameters(), default_maint_ops_dict(), "fhwa_unit_costs_2025")
    assert ops["replace_appoggi"]["cost"] == 100000
