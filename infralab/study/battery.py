"""Batteria di partite per il disegno within-subjects del Paper A.

Ogni partecipante gioca più partite variando due fattori — copilota (OFF="Uomo"
/ ON="Uomo+LLM") × lunghezza (breve/lunga) — su scenari DISTINTI ma appaiati
per difficoltà, con ordine e assegnazione scenario↔condizione CONTROBILANCIATI
tra partecipanti (schema tipo quadrato latino, 8 sequenze).

Il controbilanciamento neutralizza apprendimento e fatica; gli scenari distinti
evitano che si rigiochi la stessa mappa con e senza copilota.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Scenari core di default (v. docs/EXPERIMENT_PLAN_PaperA.md).
DEFAULT_SHORT: Tuple[str, str] = ("PAPERA-SHORT-A", "PAPERA-SHORT-B")
DEFAULT_LONG: Tuple[str, str] = ("PAPERA-LONG-A", "PAPERA-LONG-B")
SHORT_HORIZON = 24   # 6 anni
LONG_HORIZON = 48    # 12 anni
PRACTICE_SCENARIO = "PAPERA-PRACTICE"
PRACTICE_HORIZON = 12   # 3 anni

OFF = "Uomo"
ON = "Uomo+LLM"


def build_battery(index: int,
                  short: Tuple[str, str] = DEFAULT_SHORT,
                  long: Tuple[str, str] = DEFAULT_LONG,
                  short_horizon: int = SHORT_HORIZON,
                  long_horizon: int = LONG_HORIZON,
                  include_practice: bool = True) -> List[Dict[str, Any]]:
    """Batteria 2×2 controbilanciata per il partecipante `index` (0-based).

    Tre bit dell'indice scelgono: assegnazione scenario↔condizione, ordine delle
    lunghezze, ordine del copilota entro una classe → 8 sequenze bilanciate. Se
    `include_practice`, antepone una partita di pratica (classe "practice", con
    copilota per far provare l'interfaccia completa), esclusa dall'analisi.
    """
    b_map = index & 1          # quale scenario della coppia riceve il copilota
    b_len = (index >> 1) & 1   # breve-prima vs lunga-prima
    b_cop = (index >> 2) & 1   # OFF-prima vs ON-prima entro la classe

    def pair(scn_a: str, scn_b: str, horizon: int, length_class: str) -> List[Dict[str, Any]]:
        off_scn, on_scn = (scn_a, scn_b) if b_map == 0 else (scn_b, scn_a)
        games = [
            {"scenario_id": off_scn, "condition": OFF, "horizon_quarters": horizon, "length_class": length_class},
            {"scenario_id": on_scn, "condition": ON, "horizon_quarters": horizon, "length_class": length_class},
        ]
        return list(reversed(games)) if b_cop else games

    shorts = pair(short[0], short[1], short_horizon, "short")
    longs = pair(long[0], long[1], long_horizon, "long")
    battery = (shorts + longs) if b_len == 0 else (longs + shorts)
    if include_practice:
        battery = [{"scenario_id": PRACTICE_SCENARIO, "condition": ON,
                    "horizon_quarters": PRACTICE_HORIZON, "length_class": "practice",
                    "practice": True}] + battery
    for i, g in enumerate(battery):
        g["index"] = i
    return battery


def battery_signature(battery: List[Dict[str, Any]]) -> str:
    """Firma compatta della batteria (per confronti/debug)."""
    return " | ".join(f"{g['index']}:{g['length_class'][0]}/{'ON' if g['condition'] == ON else 'OFF'}"
                      f"@{g['scenario_id']}" for g in battery)
