"""Diagnostica della taratura del crollo idraulico (anti-sovradimensionamento).

Per uno scenario e una configurazione stampa il tasso ATTESO di crollo idraulico per
ponte-anno di un gestore che non ispeziona e non arresta mai: crollo diretto piu'
percorso latente degli eventi progressivi, a regime e sull'orizzonte dato, da
confrontare con l'ancora empirica (circa 1e-4 per ponte-anno; 0.7e-4 - 1.2e-4 sui ponti
in acqua, vedi docs/CHANGELOG_MODEL.md). Calcolo analitico: nessuna simulazione.

Uso:
    python -m infralab.lab.hydraulic_rate --scenario scenario.json --horizon 200 \\
        --param hydraulic_hazard_enabled=true --param progressive_events_enabled=true \\
        --param hydraulic_collapse_progressive_share=0.5 [--json out.json]
API: `hydraulic_rate_report(scenario, params=..., horizon_quarters=...)` -> dict.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

ANCHOR_PER_BRIDGE_YEAR = 1e-4
ANCHOR_WATER_RANGE = (0.7e-4, 1.2e-4)


def hydraulic_rate_report(scenario, params: Optional[Dict[str, Any]] = None,
                          horizon_quarters: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    from infralab.core.hazards.progressive_events import expected_hydraulic_collapse_rates
    from infralab.lab.build import build_env_from_scenario
    from infralab.lab.spec import resolve_scenario
    sc = resolve_scenario(scenario) if isinstance(scenario, str) else scenario
    env = build_env_from_scenario(sc, param_overrides=dict(params or {}), seed=seed)
    report = expected_hydraulic_collapse_rates(env, horizon_quarters=horizon_quarters)
    report["scenario"] = sc.get("scenario_id") if isinstance(sc, dict) else str(scenario)
    report["anchor_per_bridge_year"] = ANCHOR_PER_BRIDGE_YEAR
    report["anchor_water_range"] = list(ANCHOR_WATER_RANGE)
    return report


def format_report(r: Dict[str, Any]) -> str:
    lines = [f"scenario={r.get('scenario')} ponti={r['bridges']} su_acqua={r['bridges_over_water']} "
             f"a_rischio_erosione={r['bridges_scour_critical']} quota_riassegnata={r['share']:.3f} "
             f"innesco={r.get('rate_mode') or 'canale progressivo idraulico spento'} "
             f"p_crollo_se_non_trattato={r.get('p_collapse_if_untreated')}",
             f"inneschi attesi per ponte-anno: {r['initiations_per_bridge_year']:.3e}; quota che matura "
             f"entro l'orizzonte ({r.get('horizon_quarters') or 'a regime'} trimestri): {r['matured_fraction']:.3f}",
             "tasso atteso di crollo idraulico, gestore che non ispeziona e non arresta mai:"]
    for key, label in (("per_bridge_year", "per ponte-anno (tutti i ponti)"),
                       ("per_water_bridge_year", "per ponte-anno (ponti su acqua)")):
        v = r[key]
        lines.append(f"  {label}: storico diretto {v['historical_direct']:.3e} | diretto {v['direct']:.3e} + "
                     f"latente {v['latent_stationary']:.3e} = {v['total_stationary']:.3e} a regime | "
                     f"{v['total_horizon']:.3e} sull'orizzonte")
    lo, hi = r["anchor_water_range"]
    lines.append(f"ancora empirica: circa {r['anchor_per_bridge_year']:.1e} per ponte-anno "
                 f"({lo:.1e} - {hi:.1e} sui ponti in acqua)")
    return "\n".join(lines)


def main(argv=None) -> int:
    from infralab.lab.leak_check import _parse_value
    parser = argparse.ArgumentParser(prog="infralab.lab.hydraulic_rate",
                                     description="Tasso atteso di crollo idraulico (diretto + latente)")
    parser.add_argument("--scenario", default="TRAIN-01", help="id di scenarios.json o path JSON")
    parser.add_argument("--param", action="append", default=[], metavar="NOME=VALORE",
                        help="override di SimParameters (ripetibile; valore JSON o testo)")
    parser.add_argument("--horizon", type=int, default=None, help="orizzonte in trimestri (default: a regime)")
    parser.add_argument("--json", default=None, help="scrive il rapporto in questo file")
    args = parser.parse_args(argv)
    params = {}
    for item in args.param:
        if "=" not in item:
            print(f"--param senza '=': {item!r}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        params[k.strip()] = _parse_value(v.strip())
    try:
        report = hydraulic_rate_report(args.scenario, params=params, horizon_quarters=args.horizon)
    except Exception as e:     # noqa: BLE001 - CLI: messaggio leggibile, codice 2
        print(f"hydraulic_rate: errore: {e}", file=sys.stderr)
        return 2
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
