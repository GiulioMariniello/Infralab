"""Modelli di degrado dei componenti.

Due modelli condividono la stessa traiettoria MEDIA di danno
D(t) = scale · (t_eff / L)^shape, con condizione C = exp(−D):

- "weibull" (default storico): degrado deterministico dato il clock di età
  efficace; la stocasticità entra solo tramite i moltiplicatori del clock.
- "gamma": processo gamma alla van Noortwijk (incrementi indipendenti
  Gamma(ΔD/θ, θ)); stessa media, varianza dei percorsi ∝ θ. Consente studi su
  incertezza temporale del degrado e value-of-information.

Il clock (età efficace) resta l'unico stato condiviso: manutenzione e danno
lo invertono con la stessa formula in entrambi i modelli.
"""
import math
from typing import Any

import numpy as np


def expected_damage(component: Any, exponent: float, effective_age: float | None = None) -> float:
    """Danno medio D(t) = scale · (t_eff/L)^shape (0 = nuovo)."""
    age = component.effective_age_years if effective_age is None else effective_age
    if age <= 0 or component.vita_utile_anni <= 0:
        return 0.0
    age_ratio = age / component.vita_utile_anni
    return component._scale() * (max(age_ratio, 0.0) ** component._shape(exponent))


def gamma_increment(
    component: Any,
    exponent: float,
    age_before: float,
    age_after: float,
    theta: float,
    rng: np.random.Generator,
) -> float:
    """Incremento stocastico di danno tra due età efficaci (processo gamma).

    Shape = ΔD/θ, scale = θ → media dell'incremento = ΔD (consistente con la
    traiettoria media Weibull), varianza = ΔD·θ.
    """
    delta = max(
        0.0,
        expected_damage(component, exponent, age_after) - expected_damage(component, exponent, age_before),
    )
    if delta <= 0.0:
        return 0.0
    return float(rng.gamma(shape=delta / theta, scale=theta))


def condition_from_damage(damage: float) -> float:
    return max(0.0, min(1.0, math.exp(-max(0.0, damage))))


def damage_from_condition(condition: float) -> float:
    return -math.log(max(condition, 1e-6))
