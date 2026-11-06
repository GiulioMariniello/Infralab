"""Import del parco asset da foglio Excel/CSV.

Genera un modello scaricabile per ogni tipo di asset (colonne guidate dallo
`asset_property_schema` della spec) e reimporta un file compilato mappando le
intestazioni (accent/case-insensitive) sulle chiavi dello schema. Restituisce
la lista di asset pronta per uno scenario custom, piu' warning/errori per
riga cosi' l'utente corregge prima di salvare.
"""
from __future__ import annotations

import io
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Colonne comuni a ogni tipo (oltre allo schema per-asset).
ID_KEYS = {"id", "identificativo"}
NAME_KEYS = {"name", "nome", "denominazione"}
LAT_KEYS = {"lat", "latitudine", "latitude"}
LON_KEYS = {"lon", "lng", "longitudine", "longitude"}


def _norm(text: Any) -> str:
    """Normalizza un'intestazione: minuscolo, senza accenti/spazi/punteggiatura."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _schema_column_index(schema: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Mappa header-normalizzato -> chiave di schema (accetta chiave o label)."""
    index: Dict[str, str] = {}
    for key, meta in schema.items():
        index[_norm(key)] = key
        label = meta.get("label")
        if label:
            index[_norm(label)] = key
    return index


def build_template_rows(spec) -> Tuple[List[str], List[Any], List[str]]:
    """Righe del modello: intestazioni, riga d'esempio, note per colonna."""
    schema = spec.asset_property_schema or {}
    headers = ["id", "name"]
    example: List[Any] = [1, f"{spec.display_name} 1"]
    notes = ["intero univoco", "etichetta libera"]
    for key, meta in schema.items():
        unit = meta.get("unit")
        label = meta.get("label", key)
        headers.append(f"{label} [{unit}]" if unit and unit != "-" else label)
        example.append(meta.get("default", ""))
        bits = []
        if meta.get("required"):
            bits.append("OBBLIGATORIO")
        if meta.get("options"):
            bits.append("valori: " + "/".join(str(o) for o in meta["options"]))
        if meta.get("min") is not None or meta.get("max") is not None:
            bits.append(f"range {meta.get('min', '')}–{meta.get('max', '')}")
        notes.append("; ".join(bits))
    headers += ["lat", "lon"]
    example += ["", ""]
    notes += ["opzionale (GIS)", "opzionale (GIS)"]
    return headers, example, notes


def build_template_xlsx(spec) -> bytes:
    """Modello .xlsx: riga intestazioni + riga note (commento) + esempio."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill

    headers, example, notes = build_template_rows(spec)
    wb = Workbook()
    ws = wb.active
    ws.title = spec.name[:31]
    header_fill = PatternFill("solid", fgColor="1F6FB2")
    header_font = Font(color="FFFFFF", bold=True)
    for col, (h, note) in enumerate(zip(headers, notes), start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font
        if note:
            cell.comment = Comment(note, "InfraLab")
        ws.column_dimensions[cell.column_letter].width = max(14, min(30, len(str(h)) + 4))
    for col, val in enumerate(example, start=1):
        ws.cell(row=2, column=col, value=val)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _coerce(value: Any, meta: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Converte un valore secondo il tipo di schema; ritorna (valore, warning)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    if meta.get("type") == "number":
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None, f"'{value}' non numerico"
        lo, hi = meta.get("min"), meta.get("max")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            return v, f"{v} fuori range [{lo}, {hi}]"
        return (int(v) if v.is_integer() else v), None
    if meta.get("options"):
        opts = [str(o) for o in meta["options"]]
        if str(value) not in opts:
            return str(value), f"'{value}' non tra {opts}"
    return value, None


def _read_table(file_bytes: bytes, filename: str) -> Tuple[List[str], List[List[Any]]]:
    """Legge xlsx/xls/csv -> (intestazioni, righe). Prima riga = intestazioni."""
    name = (filename or "").lower()
    if name.endswith(".csv"):
        import csv as _csv

        text = file_bytes.decode("utf-8-sig", errors="replace")
        rows = list(_csv.reader(io.StringIO(text)))
        if not rows:
            return [], []
        return [str(h) for h in rows[0]], rows[1:]
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    try:
        header = list(next(it))
    except StopIteration:
        return [], []
    headers = [str(h) if h is not None else "" for h in header]
    rows = [list(r) for r in it]
    return headers, rows


def parse_asset_workbook(file_bytes: bytes, filename: str, spec) -> Dict[str, Any]:
    """Parsa un file compilato in asset per lo scenario del tipo `spec`.

    Ritorna {assets, warnings, errors, matched_columns, ignored_columns}.
    Gli errori (riga saltata) sono fatali per quella riga; i warning no.
    """
    schema = spec.asset_property_schema or {}
    col_index = _schema_column_index(schema)
    headers, rows = _read_table(file_bytes, filename)
    if not headers:
        return {"assets": [], "warnings": [], "errors": ["File vuoto o illeggibile."],
                "matched_columns": [], "ignored_columns": []}

    # Mappa indice-colonna -> ('id'|'name'|'lat'|'lon'|<schema key>).
    col_target: Dict[int, str] = {}
    ignored: List[str] = []
    for i, h in enumerate(headers):
        n = _norm(h)
        # 'altezza mozzo [m]' normalizza includendo l'unita': prova anche senza.
        n_base = _norm(str(h).split("[")[0])
        if n in ID_KEYS or n_base in ID_KEYS:
            col_target[i] = "id"
        elif n in NAME_KEYS or n_base in NAME_KEYS:
            col_target[i] = "name"
        elif n in LAT_KEYS or n_base in LAT_KEYS:
            col_target[i] = "lat"
        elif n in LON_KEYS or n_base in LON_KEYS:
            col_target[i] = "lon"
        elif n in col_index:
            col_target[i] = col_index[n]
        elif n_base in col_index:
            col_target[i] = col_index[n_base]
        elif str(h).strip():
            ignored.append(str(h))

    assets: List[Dict[str, Any]] = []
    warnings: List[str] = []
    errors: List[str] = []
    seen_ids = set()
    auto_id = 0
    for r_i, row in enumerate(rows, start=2):
        if all(c is None or (isinstance(c, str) and not c.strip()) for c in row):
            continue
        asset: Dict[str, Any] = {}
        row_errs: List[str] = []
        for c_i, target in col_target.items():
            raw = row[c_i] if c_i < len(row) else None
            if target in ("id", "name"):
                if raw is not None and str(raw).strip():
                    asset[target] = int(float(raw)) if target == "id" else str(raw).strip()
            elif target in ("lat", "lon"):
                if raw is not None and str(raw).strip():
                    try:
                        asset[target] = float(raw)
                    except (TypeError, ValueError):
                        warnings.append(f"Riga {r_i}: {target} '{raw}' ignorata (non numerica).")
            else:
                v, warn = _coerce(raw, schema.get(target, {}))
                if warn:
                    warnings.append(f"Riga {r_i}, {target}: {warn}.")
                if v is not None:
                    asset[target] = v
        # id/name obbligatori: se assenti, autogenera.
        if "id" not in asset:
            auto_id = max(auto_id, max(seen_ids) if seen_ids else 0) + 1
            asset["id"] = auto_id
        if asset["id"] in seen_ids:
            errors.append(f"Riga {r_i}: id {asset['id']} duplicato (riga saltata).")
            continue
        if "name" not in asset:
            asset["name"] = f"{spec.display_name} {asset['id']}"
        # Campi obbligatori mancanti: usa il default (con warning) o, se assente,
        # salta la riga. Il valore DEVE finire nell'asset o il motore va in errore.
        for key, meta in schema.items():
            if meta.get("required") and asset.get(key) is None:
                default = meta.get("default")
                if default is not None:
                    asset[key] = default
                    warnings.append(
                        f"Riga {r_i}, {meta.get('label', key)}: mancante, uso il default {default}."
                    )
                else:
                    row_errs.append(f"'{meta.get('label', key)}' mancante")
        if row_errs:
            errors.append(f"Riga {r_i} ({asset.get('name')}): {', '.join(row_errs)} (riga saltata).")
            continue
        seen_ids.add(asset["id"])
        auto_id = max(auto_id, asset["id"])
        assets.append(asset)

    matched = sorted({t for t in col_target.values()})
    return {
        "assets": assets,
        "warnings": warnings,
        "errors": errors,
        "matched_columns": matched,
        "ignored_columns": ignored,
    }
