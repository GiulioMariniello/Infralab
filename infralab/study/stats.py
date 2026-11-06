"""Confronto statistico fra condizioni sperimentali.

Per ogni gruppo (condizione: LLM / Uomo / Uomo+LLM / Euristico) calcola media
e intervallo di confidenza bootstrap; per ogni coppia calcola effect size
(Cohen's d parametrico, Cliff's delta non parametrico) e un test non
parametrico (Mann-Whitney U) — adatto a campioni piccoli e non gaussiani come
quelli di uno studio umano.

Determinismo: il bootstrap usa un `np.random.default_rng(seed)` con seed fisso,
così lo stesso input dà lo stesso intervallo (riproducibilità).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(samples: List[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 12345) -> Optional[Dict[str, float]]:
    """Media + intervallo di confidenza bootstrap percentile."""
    xs = [float(x) for x in samples if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    arr = np.asarray(xs, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return {"mean": round(mean, 4), "lo": round(mean, 4), "hi": round(mean, 4), "n": len(arr)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "sd": round(float(arr.std(ddof=1)), 4), "n": len(arr)}


def cohens_d(a: List[float], b: List[float]) -> Optional[float]:
    """Effect size parametrico (differenza standardizzata, pooled SD)."""
    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return round((np.mean(a) - np.mean(b)) / pooled, 3)


def cliffs_delta(a: List[float], b: List[float]) -> Optional[float]:
    """Effect size non parametrico in [-1, 1]: P(a>b) - P(a<b)."""
    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if not a or not b:
        return None
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return round((gt - lt) / (len(a) * len(b)), 3)


def _cliff_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "trascurabile"
    if d < 0.33:
        return "piccolo"
    if d < 0.474:
        return "medio"
    return "grande"


def mann_whitney_p(a: List[float], b: List[float]) -> Optional[float]:
    """p-value bilaterale del test di Mann-Whitney U (scipy)."""
    a = [float(x) for x in a if x is not None]
    b = [float(x) for x in b if x is not None]
    if len(a) < 1 or len(b) < 1 or (len(set(a + b)) == 1):
        return None
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        return round(float(p), 5)
    except Exception:
        return None


def wilcoxon_signed_rank(pairs: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    """Test appaiato di Wilcoxon (within-subjects): confronta coppie (con, senza).

    pairs: lista di (valore_con_LLM, valore_senza_LLM) per lo stesso soggetto.
    Ritorna {n, median_diff, p_value, effect_r, mean_diff, ci} o None se < 3 coppie
    con differenza non nulla. Effect size r = Z/√N (approssimazione normale scipy).
    """
    diffs = [float(a) - float(b) for a, b in pairs
             if a is not None and b is not None]
    nonzero = [d for d in diffs if d != 0.0]
    if len(nonzero) < 3:
        return None
    import numpy as _np
    arr = _np.asarray(diffs, dtype=float)
    out: Dict[str, Any] = {
        "n": len(diffs),
        "mean_diff": round(float(arr.mean()), 4),
        "median_diff": round(float(_np.median(arr)), 4),
        "ci": bootstrap_ci(list(arr)),
    }
    try:
        from scipy.stats import norm, wilcoxon
        res = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        p = float(res.pvalue)
        out["p_value"] = round(p, 5)
        out["significant"] = p < 0.05
        # r = |Z|/sqrt(N): ricava Z dal p bilaterale.
        z = abs(float(norm.ppf(p / 2.0)))
        out["effect_r"] = round(z / (len(nonzero) ** 0.5), 3)
    except Exception:
        out["p_value"] = None
        out["significant"] = False
        out["effect_r"] = None
    return out


def paired_delta(subjects: Dict[str, Dict[str, List[float]]],
                 treat: str, control: str) -> Optional[Dict[str, Any]]:
    """Δ intra-soggetto (treat − control) su più soggetti, con Wilcoxon.

    subjects: {soggetto: {condizione: [valori]}}. Per ogni soggetto media i
    valori di treat e control e forma la coppia; poi Wilcoxon sulle coppie.
    """
    import numpy as _np
    pairs: List[Tuple[float, float]] = []
    for cond_map in subjects.values():
        t = [v for v in cond_map.get(treat, []) if v is not None]
        c = [v for v in cond_map.get(control, []) if v is not None]
        if t and c:
            pairs.append((float(_np.mean(t)), float(_np.mean(c))))
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "note": "meno di 3 soggetti con entrambe le condizioni"}
    result = wilcoxon_signed_rank(pairs)
    return {"n_pairs": len(pairs), "treat": treat, "control": control,
            **(result or {"note": "differenze tutte nulle"})}


def compare_conditions(groups: Dict[str, List[float]], alpha: float = 0.05,
                       seed: int = 12345) -> Dict[str, object]:
    """Confronto completo fra condizioni per un singolo esito.

    groups: {nome_condizione: [valori]}. Ritorna medie+CI per gruppo e i
    confronti a coppie (Mann-Whitney p, Cohen's d, Cliff's delta con magnitudo).
    """
    per_group = {name: bootstrap_ci(vals, alpha=alpha, seed=seed) for name, vals in groups.items()}
    names = [n for n in groups if per_group.get(n) is not None]
    comparisons = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = groups[a_name], groups[b_name]
            p = mann_whitney_p(a, b)
            delta = cliffs_delta(a, b)
            comparisons.append({
                "a": a_name,
                "b": b_name,
                "mean_diff": round(float(np.mean([x for x in a if x is not None]) -
                                         np.mean([x for x in b if x is not None])), 4),
                "p_value": p,
                "significant": (p is not None and p < alpha),
                "cohens_d": cohens_d(a, b),
                "cliffs_delta": delta,
                "effect_magnitude": _cliff_magnitude(delta) if delta is not None else None,
            })
    return {"groups": per_group, "comparisons": comparisons, "alpha": alpha}
