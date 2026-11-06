"""Modello di misura delle ispezioni: rumore, bias sistematico, POD opzionale.

Il valore osservato è un intervallo [min, max] centrato su una misura rumorosa
della condizione vera. Con la POD abilitata, il danno può NON essere rilevato:
la probabilità di detection cresce con il danno (P = 1 − (C)^k), e in caso di
mancata detection l'osservazione resta ottimistica (componente "sembra sana").
Consente esperimenti di value-of-information con qualità d'ispezione variabile.
"""
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass(frozen=True)
class InspectionOutcome:
    observed_min: float
    observed_max: float
    detected: bool


def inspect_component(
    component: Any,
    sigma: float,
    rng: np.random.Generator,
    bias: float = 0.0,
    pod_enabled: bool = False,
    pod_exponent: float = 8.0,
) -> InspectionOutcome:
    """Osserva la condizione di un componente con rumore, bias e POD.

    sigma: deviazione standard dell'osservazione (gia' scalata per l'incertezza
    epistemica dal chiamante). bias > 0 sposta l'osservazione verso l'alto
    (ottimismo). pod_exponent k: P(detect) = 1 - C^k (danno piccolo, C~1 →
    detection improbabile).
    """
    true_condition = component.current_condition
    detected = True
    if pod_enabled:
        detection_probability = 1.0 - max(0.0, min(1.0, true_condition)) ** pod_exponent
        detected = bool(rng.random() < detection_probability)

    if pod_enabled and not detected:
        # Danno non rilevato: l'ispezione riporta una condizione apparente sana.
        center = float(np.clip(max(true_condition, 0.9) + bias + rng.normal(0, sigma * 0.5), 0, 1))
    else:
        center = float(np.clip(true_condition + bias + rng.normal(0, sigma * 0.5), 0, 1))

    half_width = 1.96 * sigma
    return InspectionOutcome(
        observed_min=float(np.clip(center - half_width, 0, 1)),
        observed_max=float(np.clip(center + half_width, 0, 1)),
        detected=detected,
    )
