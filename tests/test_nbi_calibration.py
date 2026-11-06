"""Test della calibrazione del degrado su serie storiche NBI.

Il fitter Weibull-plot deve recuperare parametri noti da dati sintetici; il
loader deve ricostruire le traiettorie unendo gli anni sullo stesso ponte.
"""
import math

import pandas as pd
import pytest

from infralab.data import nbi_calibration as cal


class TestFitWeibull:
    def test_recovers_known_parameters(self):
        # Genera (età, condizione) dalla curva C = exp(-scale*(t/L)^k) e verifica
        # che il fit recuperi scale e k (nessun rumore).
        scale, k, L = 1.15, 2.4, 100.0
        ages = [5, 10, 20, 30, 40, 50, 60, 70, 80]
        conds = [math.exp(-scale * (t / L) ** k) for t in ages]
        fit = cal.fit_weibull(ages, conds, life_years=L)
        assert fit["scale"] == pytest.approx(scale, abs=0.02)
        assert fit["k"] == pytest.approx(k, abs=0.02)
        assert fit["r2"] > 0.999 and fit["n"] == len(ages)

    def test_recovers_with_noise(self):
        scale, k, L = 0.9, 1.8, 100.0
        # rumore deterministico piccolo alternato
        ages = list(range(5, 85, 5))
        conds = []
        for i, t in enumerate(ages):
            c = math.exp(-scale * (t / L) ** k)
            conds.append(min(0.999, max(0.001, c * (1.0 + (0.03 if i % 2 else -0.03)))))
        fit = cal.fit_weibull(ages, conds, life_years=L)
        assert fit["scale"] == pytest.approx(scale, abs=0.15)
        assert fit["k"] == pytest.approx(k, abs=0.2)
        # La linearizzazione ln(-ln C) amplifica il rumore vicino a C≈1: R²
        # moderato è atteso anche recuperando bene i parametri.
        assert fit["r2"] > 0.6

    def test_filters_invalid_points(self):
        # C=1 (rating 9) ed età 0 vanno esclusi; restano < 3 punti -> None
        assert cal.fit_weibull([0, 10], [1.0, 1.0]) is None

    def test_year_from_filename(self):
        assert cal._year_from_filename("DE23.txt") == 2023
        assert cal._year_from_filename("path/to/DE2016.txt") == 2016
        assert cal._year_from_filename("DE95.txt") == 1995


class TestLoadObservations:
    def _fake_year_file(self, tmp_path, year, rows):
        # Costruisce un file NBI-like minimale con le colonne richieste.
        cols = ["STRUCTURE_NUMBER_008", "YEAR_BUILT_027", "ADT_029",
                "STRUCTURE_LEN_MT_049", "DECK_WIDTH_MT_052",
                "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060",
                "LAT_016", "LONG_017"]
        df = pd.DataFrame(rows, columns=cols)
        p = tmp_path / f"DE{str(year)[2:]}.txt"
        df.to_csv(p, index=False)
        return str(p)

    def test_builds_trajectories(self, tmp_path):
        # Stesso ponte (str 1) in due anni: costruito nel 2000, rating cala.
        f20 = self._fake_year_file(tmp_path, 2020, [
            ["0001", "2000", "5000", "50", "12", "7", "7", "8", "0", "0"]])
        f23 = self._fake_year_file(tmp_path, 2023, [
            ["0001", "2000", "5000", "50", "12", "6", "6", "7", "0", "0"]])
        obs = cal.load_observations({2020: f20, 2023: f23})
        deck = obs[obs["component"] == "deck"].sort_values("year")
        assert list(deck["age"]) == [20.0, 23.0]           # 2020-2000, 2023-2000
        assert list(deck["rating"]) == [7, 6]
        assert obs["structure_number"].nunique() == 1

    def test_skips_rating_N_and_bad_year(self, tmp_path):
        f = self._fake_year_file(tmp_path, 2023, [
            ["0002", "N", "1", "50", "12", "7", "7", "7", "0", "0"],   # anno non valido
            ["0003", "2010", "1", "50", "12", "N", "7", "7", "0", "0"]])  # deck rating N
        obs = cal.load_observations({2023: f})
        # ponte 0002 escluso del tutto; 0003 senza deck ma con super/sub
        assert set(obs["structure_number"]) == {"0003"}
        assert "deck" not in set(obs["component"])

    def test_calibrate_and_report(self, tmp_path):
        rows_20, rows_23 = [], []
        # 30 ponti sintetici lungo la curva, due anni.
        scale, k, L = 1.0, 2.1, 100.0
        for i in range(30):
            age0 = 10 + i * 2
            yb = 2020 - age0
            c20 = math.exp(-scale * (age0 / L) ** k)
            c23 = math.exp(-scale * ((age0 + 3) / L) ** k)
            r20 = max(1, min(9, round(c20 * 9)))
            r23 = max(1, min(9, round(c23 * 9)))
            rows_20.append([f"{i:04d}", str(yb), "5000", "50", "12", str(r20), str(r20), str(min(9, r20 + 1)), "0", "0"])
            rows_23.append([f"{i:04d}", str(yb), "5000", "50", "12", str(r23), str(r23), str(min(9, r23 + 1)), "0", "0"])
        f20 = self._fake_year_file(tmp_path, 2020, rows_20)
        f23 = self._fake_year_file(tmp_path, 2023, rows_23)
        files = {2020: f20, 2023: f23}
        calib = cal.calibrate(files)
        assert calib["n_bridges"] == 30 and calib["n_observations"] > 100
        deck = calib["components"]["deck"]
        # Rating interi (quantizzazione) -> R² moderato ma parametri sensati.
        assert deck["fit"] is not None and deck["fit"]["r2"] > 0.7
        assert 1.0 < deck["fit"]["k"] < 3.5     # forma plausibile recuperata
        assert "calibrated_shape_factor" in deck
        # report markdown si genera senza errori e cita la fonte
        report = cal.build_report(calib, cal.observed_transition_years(files), "DE-test")
        assert "National Bridge Inventory" in report and "scale" in report

    def test_emitted_plugin_is_valid(self, tmp_path):
        # Un plugin generato dai parametri calibrati deve superare il validatore
        # del sistema plugin (esecuzione di prova) ed essere caricabile.
        from infralab.plugins.loader import load_entrypoint, validate_degradation

        rows_a, rows_b = [], []
        scale, k, L = 1.0, 2.0, 100.0
        for i in range(20):
            age = 15 + i * 3
            yb = 2020 - age
            for yr, rows in ((2020, rows_a), (2023, rows_b)):
                c = math.exp(-scale * ((age + (yr - 2020)) / L) ** k)
                r = max(1, min(9, round(c * 9)))
                rows.append([f"{i:04d}", str(yb), "5000", "50", "12", str(r), str(r), str(min(9, r + 1)), "0", "0"])
        cols = ["STRUCTURE_NUMBER_008", "YEAR_BUILT_027", "ADT_029", "STRUCTURE_LEN_MT_049",
                "DECK_WIDTH_MT_052", "DECK_COND_058", "SUPERSTRUCTURE_COND_059",
                "SUBSTRUCTURE_COND_060", "LAT_016", "LONG_017"]
        fa = tmp_path / "DE20.txt"; pd.DataFrame(rows_a, columns=cols).to_csv(fa, index=False)
        fb = tmp_path / "DE23.txt"; pd.DataFrame(rows_b, columns=cols).to_csv(fb, index=False)
        calib = cal.calibrate({2020: str(fa), 2023: str(fb)})
        source = cal.emit_degradation_plugin(calib, state="DE-test")
        ok, msg = validate_degradation(source)
        assert ok, msg
        fn = load_entrypoint(source, "degrade")
        v = fn({"vita_utile_anni": 100, "effective_age_years": 40, "scale": 1.0,
                "shape": 3.0, "component_name": "impalcato"})
        assert 0.0 <= v <= 1.0
