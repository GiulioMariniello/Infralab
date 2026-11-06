"""Importer FHWA NBI (National Bridge Inventory) -> scenari InfraLab.

Il NBI è il censimento pubblico dei ponti USA (https://www.fhwa.dot.gov/bridge/nbi.cfm):
per ogni struttura riporta condition rating 0–9 di impalcato/sovrastruttura/
sottostruttura, ADT, anno di costruzione, geometria e coordinate. Questo modulo
trasforma un file "delimited" NBI in uno scenario InfraLab con ponti REALI.

Mapping (documentato, approssimato — v1):
- Condition rating r (0–9) -> condizione C = r/9 (9=eccellente->1.0, 0=collassato->0).
- Componenti: impalcato ← media(DECK_058, SUPERSTRUCTURE_059);
  pile e spalle ← SUBSTRUCTURE_060; appoggi ← SUPERSTRUCTURE_059.
- Età efficace per componente: inversione della curva media Weibull del profilo
  (C = exp(-scale·(t/L)^k)) così che all'avvio la condizione simulata coincida
  con il rating osservato.
- Coordinate NBI in DMS packed (DDMMSSss / DDDMMSSss, longitudine Ovest -> negativa).
- Economia sintetica dichiarata: valore_asset = area·3000 €/m²,
  incasso_trimestrale ≈ 2 €·ADT (proxy di canone/pedaggio, configurabile).

Uso CLI:
    python -m infralab.data.nbi data_sources/DE23.txt --scenario-id NBI-DE \
        --name "Delaware (FHWA NBI 2023)" --max-bridges 10 --write
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from infralab.assets.bridges.profiles import BRIDGE_COMPONENT_PROFILES

# Rating NBI minimo per considerare il ponte (sotto: fuori servizio/dati sporchi).
MIN_RATING = 2
DEFAULT_LIFE_YEARS = 100
DETERIORATION_EXPONENT = 3.0
ASSET_VALUE_EUR_M2 = 3000.0
REVENUE_EUR_PER_ADT = 2.0


def dms_packed_to_decimal(value: str, is_longitude: bool = False) -> Optional[float]:
    """Converte il formato NBI DDMMSSss (lat) / DDDMMSSss (lon) in gradi decimali."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or not raw.isdigit() or int(raw) == 0:
        return None
    if is_longitude:
        deg, minutes, seconds = int(raw[:3]), int(raw[3:5]), int(raw[5:]) / 100.0
    else:
        deg, minutes, seconds = int(raw[:2]), int(raw[2:4]), int(raw[4:]) / 100.0
    if minutes >= 60 or seconds >= 60:
        return None
    decimal = deg + minutes / 60.0 + seconds / 3600.0
    if is_longitude:
        decimal = -decimal  # NBI: longitudine Ovest
    return round(decimal, 6)


def rating_to_condition(rating: Any) -> Optional[float]:
    """Condition rating NBI 0–9 -> condizione [0,1]."""
    try:
        r = int(str(rating).strip())
    except (ValueError, TypeError):
        return None
    if not 0 <= r <= 9:
        return None
    return round(r / 9.0, 4)


def condition_to_effective_age(condition: float, component: str,
                               life_years: float = DEFAULT_LIFE_YEARS,
                               exponent: float = DETERIORATION_EXPONENT) -> float:
    """Inverte la curva media Weibull del profilo: età efficace che produce C.

    C = exp(-scale·(t/L)^k) con k = exponent·shape_factor  ->
    t = L · (−ln C / scale)^(1/k)
    """
    profile = BRIDGE_COMPONENT_PROFILES[component]
    scale = profile["scale_mean"]
    k = exponent * profile["shape_factor"]
    c = min(max(condition, 1e-4), 0.9999)
    damage = -math.log(c)
    return round(life_years * (damage / scale) ** (1.0 / k), 2)


def _clean_text(value: Any) -> str:
    return str(value or "").strip().strip("'").strip()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_nbi_file(path: str) -> pd.DataFrame:
    """Legge un file NBI delimited e restituisce i record utilizzabili."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    required = ["STRUCTURE_NUMBER_008", "YEAR_BUILT_027", "ADT_029",
                "STRUCTURE_LEN_MT_049", "DECK_WIDTH_MT_052",
                "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060",
                "LAT_016", "LONG_017"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File NBI senza colonne obbligatorie: {missing}")
    return df


def nbi_to_bridges(df: pd.DataFrame, max_bridges: int = 10,
                   data_year: int = 2023) -> List[Dict[str, Any]]:
    """Seleziona e converte i record NBI in ponti InfraLab (ordinati per ADT)."""
    bridges: List[Dict[str, Any]] = []
    records = []
    for _, row in df.iterrows():
        conditions = {
            "deck": rating_to_condition(row["DECK_COND_058"]),
            "superstructure": rating_to_condition(row["SUPERSTRUCTURE_COND_059"]),
            "substructure": rating_to_condition(row["SUBSTRUCTURE_COND_060"]),
        }
        if any(v is None for v in conditions.values()):
            continue  # culvert o rating mancante ("N")
        if min(int(str(row[c]).strip()) for c in
               ("DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060")) < MIN_RATING:
            continue
        try:
            adt = int(float(row["ADT_029"]))
            length = float(row["STRUCTURE_LEN_MT_049"])
            width = float(row["DECK_WIDTH_MT_052"])
            year_built = int(row["YEAR_BUILT_027"])
        except (ValueError, TypeError):
            continue
        if adt <= 0 or length <= 2 or width <= 2:
            continue
        lat = dms_packed_to_decimal(row["LAT_016"], is_longitude=False)
        lon = dms_packed_to_decimal(row["LONG_017"], is_longitude=True)
        if lat is None or lon is None:
            continue
        records.append((adt, row, conditions, length, width, year_built, lat, lon))

    records.sort(key=lambda t: -t[0])  # priorità agli assi più trafficati
    for i, (adt, row, cond, length, width, year_built, lat, lon) in enumerate(records[:max_bridges], start=1):
        # Mapping componenti (v. docstring modulo).
        component_conditions = {
            "impalcato": round((cond["deck"] + cond["superstructure"]) / 2.0, 4),
            "pile": cond["substructure"],
            "spalle": cond["substructure"],
            "appoggi": cond["superstructure"],
        }
        facility = _clean_text(row.get("FACILITY_CARRIED_007"))
        location = _clean_text(row.get("LOCATION_009"))
        name = " · ".join(x for x in (facility, location) if x) or f"NBI {_clean_text(row['STRUCTURE_NUMBER_008'])}"
        area = length * width
        bridge: Dict[str, Any] = {
            "id": i,
            "name": name[:60],
            "nbi_structure_number": _clean_text(row["STRUCTURE_NUMBER_008"]),
            "length": round(length, 1),
            "width": round(width, 1),
            "adt": adt,
            "lat": lat,
            "lon": lon,
            "eta_iniziale_anni": max(0, data_year - year_built),
            "vita_utile_anni": DEFAULT_LIFE_YEARS,
            "incasso_trimestrale": round(min(adt * REVENUE_EUR_PER_ADT, 250_000)),
            "valore_asset": round(area * ASSET_VALUE_EUR_M2),
            "nbi_conditions": component_conditions,
        }
        # Attributi di rete e di esposizione, presenti nello stesso file NBI e finora
        # ignorati. Servono a misurare la rilevanza di rete come fa la scheda FHWA
        # (termine S3 "essentiality for public use", guidato dagli item 19 e 29) e
        # come fa l'esposizione della classe di attenzione delle Linee Guida
        # (traffico, presenza di alternativa stradale, natura dell'ostacolo scavalcato).
        detour = _to_float(row.get("DETOUR_KILOS_019"))
        truck = _to_float(row.get("PERCENT_ADT_TRUCK_109"))
        waterway = _clean_text(row.get("WATERWAY_EVAL_071"))
        scour = _clean_text(row.get("SCOUR_CRITICAL_113"))
        bridge["detour_km"] = round(detour, 1) if detour is not None else 0.0
        bridge["truck_fraction"] = round(min(max((truck or 0.0) / 100.0, 0.0), 1.0), 3)
        bridge["crosses_water"] = bool(waterway and waterway.upper() != "N")
        # item 113 codici 0-3 = fondazione a rischio erosione; "N" = non su acqua
        bridge["scour_critical"] = bool(scour.isdigit() and int(scour) <= 3)
        bridge["service_under"] = _clean_text(row.get("SERVICE_UND_042B"))
        # item 038 (controllo della navigazione): "1" = via navigabile con controllo;
        # "0" o "N" = no. Serve all'urto di natante (vessel_impact_enabled).
        navigation = _clean_text(row.get("NAVIGATION_038"))
        if navigation:
            bridge["navigable_waterway"] = navigation == "1"
        # Età efficaci per componente: la simulazione parte dalla condizione
        # osservata. Vita utile ESATTA (no perturbazione) per mantenere
        # consistente l'inversione condizione -> età.
        for comp, condition in component_conditions.items():
            bridge[f"{comp}_eta_iniziale_anni"] = condition_to_effective_age(condition, comp)
            bridge[f"{comp}_vita_utile_anni"] = DEFAULT_LIFE_YEARS
        bridges.append(bridge)
    return bridges


def build_scenario(bridges: List[Dict[str, Any]], scenario_id: str, name: str,
                   source_note: str) -> Dict[str, Any]:
    # Auto-calibrazione del riferimento di traffico: il modello di domanda
    # normalizza l'ADT su traffic_reference_adt; con gli ADT interstate USA
    # (~100k) il riferimento italiano di default (10k) farebbe esplodere la
    # domanda. Riferimento = mediana degli ADT importati.
    adts = sorted(b["adt"] for b in bridges)
    reference_adt = adts[len(adts) // 2] if adts else 10000
    return {
        "scenario_id": scenario_id,
        "name": name,
        "description": f"Ponti reali dal FHWA National Bridge Inventory ({source_note}). "
                       "Condizioni iniziali dai condition rating NBI; economia sintetica documentata.",
        "asset_type": "bridge",
        "use_in_analysis": False,
        "data_source": source_note,
        "allowed_actions": ["L1_inspect", "L4_verify", "monitoring", "proof_load_test",
                            "replace_impalcato", "replace_pile", "replace_spalle", "replace_appoggi",
                            "ripristino_strutturale", "ripristino_completo"],
        "params": {"seed": 2023, "initial_budget": 3_000_000, "years": 30,
                   "traffic_load_model": "GL_heavy",
                   "traffic_reference_adt": int(reference_adt)},
        "bridges": bridges,
    }


def import_into_scenarios(scenario: Dict[str, Any], scenarios_path: str = "scenarios.json") -> None:
    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    scenarios = [s for s in scenarios if s.get("scenario_id") != scenario["scenario_id"]]
    scenarios.append(scenario)
    with open(scenarios_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Importa un file FHWA NBI in uno scenario InfraLab")
    parser.add_argument("nbi_file", help="file NBI delimited (es. DE23.txt)")
    parser.add_argument("--scenario-id", default="NBI-IMPORT")
    parser.add_argument("--name", default="FHWA NBI import")
    parser.add_argument("--max-bridges", type=int, default=10)
    parser.add_argument("--data-year", type=int, default=2023)
    parser.add_argument("--write", action="store_true", help="scrive in scenarios.json (e copia runtime)")
    args = parser.parse_args(argv)

    df = parse_nbi_file(args.nbi_file)
    bridges = nbi_to_bridges(df, max_bridges=args.max_bridges, data_year=args.data_year)
    print(f"Record NBI: {len(df)} -> ponti importabili: {len(bridges)}")
    for b in bridges:
        print(f"  [{b['id']}] {b['name'][:44].ljust(44)} ADT {str(b['adt']).rjust(6)} "
              f"cond {b['nbi_conditions']}")
    scenario = build_scenario(bridges, args.scenario_id, args.name,
                              source_note=f"file {args.nbi_file}, anno dati {args.data_year}")
    if args.write:
        import_into_scenarios(scenario)
        import shutil
        shutil.copyfile("scenarios.json", "data/scenarios.json")
        print(f"Scenario '{args.scenario_id}' scritto in scenarios.json (+ copia runtime).")
    else:
        print(json.dumps(scenario, indent=2, ensure_ascii=False)[:800] + " …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
