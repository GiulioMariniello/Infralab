"""Calibrazione del modello di degrado sulle serie storiche NBI (longitudinale).

Il NBI pubblica un censimento annuale: unendo più anni sullo stesso ponte
(`STRUCTURE_NUMBER_008`) si ottengono traiettorie reali di condizione nel
tempo. Da queste si calibrano i parametri della curva di degrado Weibull del
modello InfraLab, ancorandola a dati osservati anziché a valori assunti.

Modello:  C(t) = exp(-D),  D = scale · (t/L)^k
Linearizzazione (Weibull-plot):  ln(-ln C) = ln(scale) + k · ln(t/L)
=> regressione lineare y = a + k·x con x = ln(t/L), y = ln(-ln C) dà
   k = pendenza, scale = exp(a). Robusta, senza ottimizzatore.

I rating NBI sono interi 0–9 -> C = r/9; si usano solo i punti con 0<C<1 ed
età>0 (r=9 non porta informazione di degrado, r=0 satura il logaritmo). La
componente longitudinale (stesso ponte a età diverse) infittisce il
campionamento della curva e riduce l'eterogeneità non osservata rispetto al
solo cross-section.

Uso CLI:
    python -m infralab.data.nbi_calibration data_sources/nbi_series/DE*.txt \
        --state DE --report docs/NBI_CALIBRATION_DE.md
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from infralab.assets.bridges.profiles import BRIDGE_COMPONENT_PROFILES
from infralab.data.nbi import DETERIORATION_EXPONENT, parse_nbi_file, rating_to_condition

DEFAULT_LIFE_YEARS = 100.0

# Componente InfraLab -> colonna di rating NBI usata per calibrarlo.
# (impalcato è media deck+super nell'import; per la calibrazione teniamo i
#  segnali distinti così ogni curva è ancorata alla propria misura fisica.)
COMPONENT_RATING_COLUMN = {
    "deck": "DECK_COND_058",
    "superstructure": "SUPERSTRUCTURE_COND_059",
    "substructure": "SUBSTRUCTURE_COND_060",
}
# Mappatura sui componenti del profilo bridge InfraLab per il confronto.
COMPONENT_PROFILE_MAP = {
    "deck": "impalcato",
    "superstructure": "appoggi",
    "substructure": "pile",
}


def _year_from_filename(path: str) -> Optional[int]:
    """Estrae l'anno da un nome tipo DE23.txt / DE2023.txt."""
    base = os.path.basename(path)
    m = re.search(r"(\d{2,4})\D*\.txt$", base, re.IGNORECASE)
    if not m:
        return None
    n = int(m.group(1))
    if n >= 1900:
        return n
    return 2000 + n if n <= 79 else 1900 + n


def load_observations(files: Dict[int, str], min_age: float = 1.0) -> pd.DataFrame:
    """Costruisce la tabella lunga delle osservazioni (età, condizione) reali.

    files: {anno: percorso}. Ritorna un DataFrame con colonne
    structure_number, year, age, component (deck/super/sub), rating, condition.
    Filtra rating non validi e ponti senza anno di costruzione plausibile.
    """
    rows: List[Dict[str, Any]] = []
    for year, path in sorted(files.items()):
        df = parse_nbi_file(path)
        for _, r in df.iterrows():
            struct = str(r["STRUCTURE_NUMBER_008"]).strip()
            try:
                year_built = int(str(r["YEAR_BUILT_027"]).strip())
            except (ValueError, TypeError):
                continue
            if not 1900 <= year_built <= year:
                continue
            age = year - year_built
            if age < min_age:
                continue
            for comp, col in COMPONENT_RATING_COLUMN.items():
                cond = rating_to_condition(r.get(col))
                if cond is None:
                    continue
                try:
                    rating = int(str(r[col]).strip())
                except (ValueError, TypeError):
                    continue
                rows.append({"structure_number": struct, "year": year, "age": float(age),
                             "component": comp, "rating": rating, "condition": cond})
    return pd.DataFrame(rows)


def fit_weibull(ages: List[float], conditions: List[float],
                life_years: float = DEFAULT_LIFE_YEARS) -> Optional[Dict[str, float]]:
    """Fit Weibull-plot: ritorna scale, k, R², n dai punti (età, condizione).

    Usa solo 0<C<1 ed età>0. Regressione OLS su y=ln(-ln C) vs x=ln(t/L).
    """
    xs: List[float] = []
    ys: List[float] = []
    for t, c in zip(ages, conditions):
        if t <= 0 or not (0.0 < c < 1.0):
            continue
        xs.append(math.log(t / life_years))
        ys.append(math.log(-math.log(c)))
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    k = sxy / sxx
    a = my - k * mx
    scale = math.exp(a)
    # R² della retta.
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + k * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"scale": round(scale, 4), "k": round(k, 4), "r2": round(r2, 4), "n": n,
            "life_years": life_years}


def calibrate(files: Dict[int, str], life_years: float = DEFAULT_LIFE_YEARS,
              exponent: float = DETERIORATION_EXPONENT) -> Dict[str, Any]:
    """Calibra ogni componente e confronta con i default del profilo InfraLab.

    Per ogni componente riporta i parametri fittati (scale, k) e i valori
    corrispondenti nel profilo: scale_mean e k_default = exponent·shape_factor.
    """
    obs = load_observations(files)
    result: Dict[str, Any] = {
        "years": sorted(files),
        "n_observations": int(len(obs)),
        "n_bridges": int(obs["structure_number"].nunique()) if not obs.empty else 0,
        "life_years": life_years,
        "exponent": exponent,
        "components": {},
    }
    for comp in COMPONENT_RATING_COLUMN:
        sub = obs[obs["component"] == comp]
        fit = fit_weibull(list(sub["age"]), list(sub["condition"]), life_years) if not sub.empty else None
        profile_name = COMPONENT_PROFILE_MAP[comp]
        profile = BRIDGE_COMPONENT_PROFILES[profile_name]
        default_k = round(exponent * profile["shape_factor"], 4)
        entry = {
            "profile_component": profile_name,
            "default_scale": profile["scale_mean"],
            "default_k": default_k,
            "fit": fit,
        }
        if fit:
            # Fattore di forma calibrato: k_fit / exponent (così il motore, che
            # usa k = exponent·shape_factor, riproduce la curva calibrata).
            entry["calibrated_shape_factor"] = round(fit["k"] / exponent, 4)
            entry["calibrated_scale_mean"] = fit["scale"]
        result["components"][comp] = entry
    return result


def observed_transition_years(files: Dict[int, str]) -> Dict[str, Optional[float]]:
    """Anni medi osservati per perdere un livello di rating, per componente.

    Metrica longitudinale onesta: per ogni ponte con declino netto sulla propria
    finestra di osservazione, (anni fra prima e ultima misura) / (livelli persi);
    media pesata sui ponti in declino. Cattura il ritmo reale di deterioramento
    (non condizionato all'evento di calo, come sarebbe il confronto anno-su-anno).
    """
    obs = load_observations(files, min_age=0.0)
    out: Dict[str, Optional[float]] = {}
    for comp in COMPONENT_RATING_COLUMN:
        sub = obs[obs["component"] == comp]
        total_years = 0.0
        total_levels = 0
        for _, g in sub.groupby("structure_number"):
            g = g.sort_values("year")
            span = float(g["year"].iloc[-1] - g["year"].iloc[0])
            decline = int(g["rating"].iloc[0] - g["rating"].iloc[-1])
            if span > 0 and decline > 0:   # solo declino netto (esclude manutenzioni/risalite)
                total_years += span
                total_levels += decline
        out[comp] = round(total_years / total_levels, 2) if total_levels else None
    return out


def build_report(calib: Dict[str, Any], transitions: Dict[str, Optional[float]],
                 state: str) -> str:
    """Report markdown della calibrazione, pronto per docs/ o supplemento paper."""
    lines = [
        f"# Calibrazione del degrado su dati NBI reali — {state}",
        "",
        f"Fonte: FHWA National Bridge Inventory, anni {min(calib['years'])}–{max(calib['years'])} "
        f"({len(calib['years'])} rilevazioni annuali dello stesso parco).",
        "",
        f"- Ponti distinti: **{calib['n_bridges']}**",
        f"- Osservazioni (ponte × anno × componente) usate: **{calib['n_observations']}**",
        f"- Vita utile assunta L = {calib['life_years']:.0f} anni · esponente globale = {calib['exponent']}",
        "",
        "## Parametri calibrati vs default del modello",
        "",
        "Curva: C(t) = exp(−scale·(t/L)^k). Fit per linearizzazione Weibull-plot "
        "(ln(−ln C) vs ln(t/L)); R² = bontà della retta.",
        "",
        "| Componente | scale (fit → default) | k (fit → default) | shape_factor calibrato | R² | n |",
        "|---|---|---|---|---|---|",
    ]
    for comp, e in calib["components"].items():
        fit = e["fit"]
        if fit:
            lines.append(
                f"| {comp} ({e['profile_component']}) | {fit['scale']} → {e['default_scale']} "
                f"| {fit['k']} → {e['default_k']} | {e['calibrated_shape_factor']} "
                f"(def {BRIDGE_COMPONENT_PROFILES[e['profile_component']]['shape_factor']}) "
                f"| {fit['r2']} | {fit['n']} |"
            )
        else:
            lines.append(f"| {comp} ({e['profile_component']}) | dati insufficienti | — | — | — | — |")
    lines += [
        "",
        "## Tasso di deterioramento osservato (longitudinale)",
        "",
        "Anni medi per perdere **un livello** di condition rating, sui ponti in "
        "declino netto: (anni fra prima e ultima misura) / (livelli persi), aggregato:",
        "",
    ]
    for comp, v in transitions.items():
        lines.append(f"- {comp}: **{v if v is not None else '—'}** anni/livello")
    # Interpretazione automatica del confronto k fittato vs default.
    fits = [(c, e) for c, e in calib["components"].items() if e["fit"]]
    if fits:
        avg_k = sum(e["fit"]["k"] for _, e in fits) / len(fits)
        avg_def = sum(e["default_k"] for _, e in fits) / len(fits)
        direction = "più concava (degrado front-loaded, decelera in età normalizzata)" if avg_k < avg_def \
            else "più convessa (degrado accelerante a fine vita)"
        lines += [
            "",
            "## Interpretazione",
            "",
            f"Il fattore di forma osservato (k medio ≈ {avg_k:.2f}) è nettamente "
            f"{'inferiore' if avg_k < avg_def else 'superiore'} al default del modello "
            f"(k medio ≈ {avg_def:.2f}): la traiettoria reale dei rating NBI è {direction}. "
            "Il modello di default, con k alto, concentra il degrado a fine vita (curva a "
            "\"scogliera\") più di quanto i dati mostrino. L'R² moderato riflette la "
            "granularità dei rating (interi 0–9) e l'eterogeneità del parco: la calibrazione "
            "va letta come ancoraggio dell'ordine di grandezza, non come fit puntuale.",
        ]
    lines += [
        "",
        "## Note metodologiche",
        "",
        "- Rating NBI interi 0–9 mappati a C = r/9; usati solo 0<C<1 ed età>0.",
        "- Longitudinale: lo stesso ponte compare a età diverse negli anni, "
        "infittendo il campionamento della curva e attenuando l'eterogeneità non osservata.",
        "- Limite: L=100 assunto (non stimato congiuntamente); i rating aggregano "
        "modalità di degrado eterogenee; nessuna correzione per interventi di manutenzione "
        "intercorsi (un rating che risale indica manutenzione ed è escluso dai decrementi).",
    ]
    return "\n".join(lines) + "\n"


# Componenti InfraLab (bridge) -> componente NBI da cui prendere i parametri.
_INFRALAB_TO_NBI = {"impalcato": "deck", "appoggi": "superstructure",
                    "pile": "substructure", "spalle": "substructure"}


def emit_degradation_plugin(calib: Dict[str, Any], state: str = "NBI") -> str:
    """Genera il sorgente di un plugin di degrado `degrade(ctx)` calibrato.

    Il plugin applica la curva Weibull con i parametri (scale, k) fittati dai
    dati NBI, scelti per nome di componente; per componenti non calibrati usa
    scale/shape del profilo (ctx). Caricabile dal laboratorio codice o dal
    registry dei plugin.
    """
    params: Dict[str, Dict[str, float]] = {}
    for infl, nbi in _INFRALAB_TO_NBI.items():
        fit = calib["components"].get(nbi, {}).get("fit")
        if fit:
            params[infl] = {"scale": fit["scale"], "k": fit["k"]}
    life = calib.get("life_years", DEFAULT_LIFE_YEARS)
    header = (
        f"# Modello di degrado calibrato su NBI {state} "
        f"(anni {min(calib['years'])}-{max(calib['years'])}, "
        f"{calib['n_bridges']} ponti, {calib['n_observations']} osservazioni).\n"
        "# Generato da infralab.data.nbi_calibration. Curva C=exp(-scale*(t/L)^k)\n"
        "# con parametri fittati per componente; fallback ai default del profilo.\n"
    )
    return (
        header
        + "import math\n\n"
        + f"CALIBRATED = {params!r}\n"
        + f"LIFE = {life}\n\n"
        + "def degrade(ctx):\n"
          "    L = ctx['vita_utile_anni'] or LIFE\n"
          "    t = ctx['effective_age_years']\n"
          "    if L <= 0 or t <= 0:\n"
          "        return 1.0\n"
          "    p = CALIBRATED.get(ctx['component_name'])\n"
          "    scale = p['scale'] if p else ctx['scale']\n"
          "    k = p['k'] if p else ctx['shape']\n"
          "    return math.exp(-scale * (t / L) ** k)\n"
    )


def _expand_files(patterns: List[str]) -> Dict[int, str]:
    files: Dict[int, str] = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            year = _year_from_filename(path)
            if year is not None:
                files[year] = path
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibra il degrado su serie storiche NBI")
    parser.add_argument("files", nargs="+", help="file NBI delimited di anni diversi (glob ammesso)")
    parser.add_argument("--state", default="NBI")
    parser.add_argument("--life-years", type=float, default=DEFAULT_LIFE_YEARS)
    parser.add_argument("--report", default=None, help="scrive il report markdown nel percorso dato")
    parser.add_argument("--emit-plugin", default=None, help="scrive un plugin di degrado calibrato (.py)")
    args = parser.parse_args(argv)

    files = _expand_files(args.files)
    if len(files) < 2:
        print("Servono almeno 2 anni per la calibrazione longitudinale.")
        return 1
    print(f"Anni: {sorted(files)}")
    calib = calibrate(files, life_years=args.life_years)
    transitions = observed_transition_years(files)
    print(f"Ponti {calib['n_bridges']} · osservazioni {calib['n_observations']}")
    for comp, e in calib["components"].items():
        fit = e["fit"]
        if fit:
            print(f"  {comp:14s} scale {fit['scale']:.3f} (def {e['default_scale']}) "
                  f"k {fit['k']:.3f} (def {e['default_k']}) R² {fit['r2']:.3f} n {fit['n']}")
    report = build_report(calib, transitions, args.state)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report scritto in {args.report}")
    if args.emit_plugin:
        os.makedirs(os.path.dirname(args.emit_plugin) or ".", exist_ok=True)
        with open(args.emit_plugin, "w", encoding="utf-8") as f:
            f.write(emit_degradation_plugin(calib, state=args.state))
        print(f"Plugin di degrado calibrato scritto in {args.emit_plugin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
