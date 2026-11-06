"""Test import asset da Excel/CSV: modello, parsing, mappatura colonne, API."""
import io

import pytest
from fastapi.testclient import TestClient

import backend.app as backend_app
import backend.research as research_module
from infralab.assets import get_asset_type
from infralab.data.excel_import import (
    build_template_rows,
    build_template_xlsx,
    parse_asset_workbook,
)


def make_xlsx(headers, rows):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestTemplate:
    def test_headers_follow_schema(self):
        spec = get_asset_type("wind_turbine")
        headers, example, notes = build_template_rows(spec)
        assert headers[0] == "id" and headers[1] == "name"
        assert any("Ore equivalenti vento" in h for h in headers)
        assert headers[-2:] == ["lat", "lon"]
        assert "OBBLIGATORIO" in " ".join(notes)

    def test_xlsx_bytes_roundtrip(self):
        spec = get_asset_type("building")
        data = build_template_xlsx(spec)
        assert data[:2] == b"PK"   # zip/xlsx magic
        # Il modello si re-importa a vuoto (solo la riga d'esempio).
        res = parse_asset_workbook(data, "t.xlsx", spec)
        assert res["errors"] == []
        assert len(res["assets"]) == 1


class TestParsing:
    def test_maps_labels_and_keys(self):
        spec = get_asset_type("wind_turbine")
        data = make_xlsx(
            ["id", "name", "Ore equivalenti vento [h/anno]", "Altezza mozzo [m]", "Diametro rotore [m]"],
            [[1, "WTG-A", 2600, 95, 120], [2, "WTG-B", 3000, 110, 130]],
        )
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert res["errors"] == []
        assert [a["name"] for a in res["assets"]] == ["WTG-A", "WTG-B"]
        assert res["assets"][0]["adt"] == 2600
        assert res["assets"][0]["length"] == 95      # 'Altezza mozzo' -> length

    def test_csv_supported(self):
        spec = get_asset_type("bridge")
        csv = "id,name,Lunghezza,Larghezza,Traffico giornaliero medio\n1,Ponte X,120,12,8000\n"
        res = parse_asset_workbook(csv.encode("utf-8"), "p.csv", spec)
        assert res["assets"][0]["length"] == 120 and res["assets"][0]["adt"] == 8000

    def test_out_of_range_is_warning_not_fatal(self):
        spec = get_asset_type("bridge")
        data = make_xlsx(["id", "name", "length", "width", "adt"], [[1, "P", 120, 12, 9_999_999]])
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert res["assets"] and any("fuori range" in w for w in res["warnings"])

    def test_missing_required_filled_with_default(self):
        spec = get_asset_type("bridge")
        # width/adt mancanti ma hanno default nello schema: riempiti + warning,
        # cosi' l'asset resta valido per il motore (che richiede width/adt).
        data = make_xlsx(["id", "name", "length"], [[1, "P", 120]])
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert len(res["assets"]) == 1
        a = res["assets"][0]
        assert a["width"] == 12 and a["adt"] == 8000     # default dello schema ponte
        assert any("mancante, uso il default" in w for w in res["warnings"])

    def test_duplicate_id_skipped(self):
        spec = get_asset_type("bridge")
        data = make_xlsx(["id", "name", "length", "width", "adt"],
                         [[1, "A", 100, 10, 5000], [1, "B", 100, 10, 5000]])
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert len(res["assets"]) == 1 and any("duplicato" in e for e in res["errors"])

    def test_auto_id_and_name(self):
        spec = get_asset_type("bridge")
        data = make_xlsx(["length", "width", "adt"], [[100, 10, 5000], [120, 12, 6000]])
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert [a["id"] for a in res["assets"]] == [1, 2]
        assert res["assets"][0]["name"].startswith("Ponte")

    def test_lat_lon_and_classe_uso(self):
        spec = get_asset_type("building")
        data = make_xlsx(
            ["id", "name", "Lunghezza in pianta", "Larghezza in pianta", "Occupanti equivalenti/giorno", "Classe d'uso NTC", "lat", "lon"],
            [[1, "Ospedale", 90, 30, 2500, "IV", 40.85, 14.27]],
        )
        res = parse_asset_workbook(data, "p.xlsx", spec)
        a = res["assets"][0]
        assert a["classe_uso"] == "IV" and a["lat"] == 40.85 and a["lon"] == 14.27

    def test_ignored_columns_reported(self):
        spec = get_asset_type("bridge")
        data = make_xlsx(["id", "name", "length", "width", "adt", "note_a_caso"],
                         [[1, "P", 100, 10, 5000, "boh"]])
        res = parse_asset_workbook(data, "p.xlsx", spec)
        assert "note_a_caso" in res["ignored_columns"]


@pytest.fixture()
def client():
    backend_app.SESSIONS.clear()
    research_module.ADMIN_TOKENS.clear()
    with TestClient(backend_app.app) as c:
        yield c


class TestApi:
    def test_template_download(self, client):
        r = client.get("/api/asset-types/solar_tracker/template.xlsx")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/vnd.openxml")
        assert r.content[:2] == b"PK"
        assert client.get("/api/asset-types/nope/template.xlsx").status_code == 404

    def test_import_endpoint(self, client):
        data = make_xlsx(["id", "name", "length", "width", "adt"],
                         [[1, "Ponte Uno", 120, 12, 8000], [2, "Ponte Due", 90, 10, 4000]])
        r = client.post(
            "/api/asset-import",
            data={"asset_type": "bridge"},
            files={"file": ("parco.xlsx", data,
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["assets"]) == 2 and body["errors"] == []

    def test_import_then_save_scenario(self, client, isolated_persistence):
        data = make_xlsx(["id", "name", "Altezza mozzo", "Diametro rotore", "Ore equivalenti vento"],
                         [[1, "T1", 100, 120, 2600]])
        imported = client.post(
            "/api/asset-import",
            data={"asset_type": "wind_turbine"},
            files={"file": ("p.xlsx", data, "application/octet-stream")},
        ).json()
        res = client.post("/api/custom-scenarios", json={
            "scenario_id": "IMPORTATO", "asset_type": "wind_turbine",
            "params": {"seed": 1}, "bridges": imported["assets"],
        })
        assert res.status_code == 200, res.text

    def test_import_unknown_type(self, client):
        data = make_xlsx(["id", "name"], [[1, "x"]])
        r = client.post("/api/asset-import", data={"asset_type": "ufo"},
                        files={"file": ("p.xlsx", data, "application/octet-stream")})
        assert r.status_code == 404
