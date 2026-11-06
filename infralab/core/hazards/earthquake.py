# earthquake_module.py
"""
Modulo per la simulazione di eventi sismici e il calcolo del danno strutturale sui ponti.
Utilizza una relazione di attenuazione semplificata (Boore-Atkinson) per stimare
la Peak Ground Acceleration (PGA) e calcolare i danni per componente.
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EarthquakeEvent:
    """Rappresenta un evento sismico singolo."""
    quarter: int
    magnitude: float          # Scala Richter
    epicenter_lat: float
    epicenter_lon: float
    name: str = "Terremoto"
    damage_reports: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quarter": self.quarter,
            "magnitude": self.magnitude,
            "epicenter_lat": self.epicenter_lat,
            "epicenter_lon": self.epicenter_lon,
            "name": self.name,
            "damage_reports": self.damage_reports,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EarthquakeEvent":
        ev = cls(
            quarter=data.get("quarter", 0),
            magnitude=data.get("magnitude", 5.0),
            epicenter_lat=data.get("epicenter_lat", 41.0),
            epicenter_lon=data.get("epicenter_lon", 14.5),
            name=data.get("name", "Terremoto"),
        )
        ev.damage_reports = data.get("damage_reports", [])
        return ev


# ---------------------------------------------------------------------------
# Funzioni di calcolo sismico
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcola la distanza in km tra due punti geografici (formula di Haversine)."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def calculate_pga(magnitude: float, distance_km: float) -> float:
    """
    Peak Ground Acceleration in g.
    Relazione di attenuazione semplificata ispirata a Boore & Atkinson (2008).
    """
    d = max(distance_km, 1.0)
    log_pga = -1.5 + 0.8 * magnitude - 1.8 * math.log10(d + 1.0)
    return 10.0 ** log_pga


def calculate_damage_factor(pga: float) -> float:
    """
    Fattore di danno strutturale normalizzato [0, 1] in funzione della PGA in g.
    Basato su curve di fragilita' semplificate per ponti stradali.
    """
    if pga < 0.05:
        return 0.0
    elif pga < 0.10:
        return 0.05 + (pga - 0.05) / 0.05 * 0.10
    elif pga < 0.20:
        return 0.15 + (pga - 0.10) / 0.10 * 0.20
    elif pga < 0.40:
        return 0.35 + (pga - 0.20) / 0.20 * 0.30
    else:
        return min(0.90, 0.65 + (pga - 0.40) / 0.40 * 0.25)


def get_damage_level(damage_factor: float) -> str:
    """Restituisce la classificazione testuale del livello di danno."""
    if damage_factor < 0.05:
        return "Nessuno"
    elif damage_factor < 0.15:
        return "Lieve"
    elif damage_factor < 0.35:
        return "Moderato"
    elif damage_factor < 0.65:
        return "Severo"
    else:
        return "Molto Severo"


# ---------------------------------------------------------------------------
# Applicazione del danno sismico
# ---------------------------------------------------------------------------

# Vulnerabilita' relativa per componente (pile piu' vulnerabili dei giunti)
COMPONENT_VULNERABILITY = {
    "pile": 1.35,
    "appoggi": 1.25,
    "spalle": 1.10,
    "impalcato": 0.85,
}


def apply_earthquake_damage(
    bridges: Dict,
    event: "EarthquakeEvent",
    deterioration_exponent: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    raise_uncertainty: bool = True,
) -> List[Dict]:
    """
    Applica i danni sismici ai ponti con coordinate geografiche.

    Args:
        bridges: dizionario {id: Bridge} dell'ambiente
        event: l'evento sismico
        deterioration_exponent: esponente del modello di degrado (per aggiornare l'eta' effettiva)
        rng: generatore numpy per la variabilita' del danno (default: entropia)
        raise_uncertainty: False con strict_information (il danno sismico non
            osservato non aumenta l'incertezza epistemica del gestore)

    Returns:
        Lista di report danni per i ponti colpiti.
    """
    sampler = rng if rng is not None else np.random.default_rng()
    damage_reports = []

    for bridge in bridges.values():
        lat = getattr(bridge, "lat", None)
        lon = getattr(bridge, "lon", None)
        if lat is None or lon is None:
            continue

        distance = haversine_distance(event.epicenter_lat, event.epicenter_lon, lat, lon)
        pga = calculate_pga(event.magnitude, distance)
        base_damage = calculate_damage_factor(pga)

        if base_damage < 0.02:
            continue

        bridge_report: Dict[str, Any] = {
            "bridge_id": bridge.id,
            "bridge_name": bridge.name,
            "distance_km": round(distance, 1),
            "pga_g": round(pga, 4),
            "base_damage_factor": round(base_damage, 3),
            "damage_level": get_damage_level(base_damage),
            "components": {},
        }

        for comp_name, comp in bridge.components.items():
            vuln = COMPONENT_VULNERABILITY.get(comp_name, 1.0)
            # Variabilita' casuale nel danno (CoV ~30%)
            actual_damage = min(base_damage * vuln * float(sampler.uniform(0.70, 1.30)), 0.90)
            if actual_damage < 0.02:
                continue

            old_cond = comp.current_condition
            # Perdita di condizione applicata tramite Component.apply_damage:
            # l'inversione dell'eta' efficace resta consistente con la curva di
            # sopravvivenza Weibull del componente. Floor a 0.05 come in origine.
            loss = min(old_cond * actual_damage, max(0.0, old_cond - 0.05))
            comp.apply_damage(loss, deterioration_exponent, raise_uncertainty=raise_uncertainty)

            bridge_report["components"][comp_name] = {
                "damage_factor": round(actual_damage, 3),
                "condition_before": round(old_cond, 3),
                "condition_after": round(comp.current_condition, 3),
                "damage_level": get_damage_level(actual_damage),
            }

        if bridge_report["components"]:
            damage_reports.append(bridge_report)

    return damage_reports


# ---------------------------------------------------------------------------
# Generazione evento casuale
# ---------------------------------------------------------------------------

def generate_random_earthquake(
    quarter: int,
    region_lat: float = 41.0,
    region_lon: float = 14.5,
    region_radius_km: float = 80.0,
    min_magnitude: float = 4.5,
    max_magnitude: float = 6.5,
    rng: Optional[np.random.Generator] = None,
) -> EarthquakeEvent:
    """Genera un terremoto casuale all'interno di una regione geografica."""
    sampler = rng if rng is not None else np.random.default_rng()
    angle = float(sampler.uniform(0.0, 2.0 * math.pi))
    dist = float(sampler.uniform(0.0, region_radius_km))
    dlat = (dist * math.cos(angle)) / 111.0
    dlon = (dist * math.sin(angle)) / (111.0 * math.cos(math.radians(region_lat)))
    mag = float(sampler.uniform(min_magnitude, max_magnitude))
    return EarthquakeEvent(
        quarter=quarter,
        magnitude=round(mag, 1),
        epicenter_lat=round(region_lat + dlat, 4),
        epicenter_lon=round(region_lon + dlon, 4),
        name=f"M{mag:.1f} - Trimestre {quarter}",
    )


# ---------------------------------------------------------------------------
# Calcolo probabilita' terremoto per trimestre
# ---------------------------------------------------------------------------

def should_earthquake_occur(annual_probability: float, rng: Optional[np.random.Generator] = None) -> bool:
    """Determina se un terremoto avviene in questo trimestre (probabilita' trimestrale)."""
    sampler = rng if rng is not None else np.random.default_rng()
    quarterly_prob = 1.0 - (1.0 - annual_probability) ** (1.0 / 4.0)
    return float(sampler.random()) < quarterly_prob


# ---------------------------------------------------------------------------
# Curve di fragilita' strutturale (modello lognormale — HAZUS-like)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """CDF della distribuzione normale standard (implementazione senza scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fragility_lognormal(pga: float, median_pga: float, beta: float = 0.5) -> float:
    """
    Probabilita' di superare uno stato di danno dato il PGA.
    Modello lognormale standard (HAZUS / FEMA P-58).

    Args:
        pga: Peak Ground Acceleration in g
        median_pga: PGA mediano che provoca il danno (g)
        beta: dispersione logaritmica (tipicamente 0.5–0.6)

    Returns:
        Probabilita' [0, 1] di superare lo stato di danno.
    """
    if pga <= 0.0 or median_pga <= 0.0:
        return 0.0
    return _norm_cdf(math.log(pga / median_pga) / beta)


# Parametri curve di fragilita' per ponti stradali (HAZUS MH MR5 Table 7.15)
# (mediana PGA in g, dispersione beta) per 4 stati di danno
FRAGILITY_PARAMS = {
    "standard": {
        "Lieve":       {"median_pga": 0.10, "beta": 0.60},
        "Moderato":    {"median_pga": 0.20, "beta": 0.55},
        "Esteso":      {"median_pga": 0.35, "beta": 0.55},
        "Completo":    {"median_pga": 0.55, "beta": 0.60},
    },
    "vecchio": {
        "Lieve":       {"median_pga": 0.08, "beta": 0.65},
        "Moderato":    {"median_pga": 0.15, "beta": 0.60},
        "Esteso":      {"median_pga": 0.25, "beta": 0.60},
        "Completo":    {"median_pga": 0.40, "beta": 0.65},
    },
    "moderno": {
        "Lieve":       {"median_pga": 0.15, "beta": 0.55},
        "Moderato":    {"median_pga": 0.30, "beta": 0.50},
        "Esteso":      {"median_pga": 0.50, "beta": 0.50},
        "Completo":    {"median_pga": 0.75, "beta": 0.55},
    },
}

FRAGILITY_COLORS = {
    "Lieve":    "#f1c40f",
    "Moderato": "#e67e22",
    "Esteso":   "#e74c3c",
    "Completo": "#8e44ad",
}


SEISMIC_ARCHETYPE_FRAGILITY = {
    "TV": {
        "UPD": {"median_pga": 0.22, "beta": 0.38},
        "SD": {"median_pga": 0.52, "beta": 0.42},
    },
    "SV": {
        "UPD": {"median_pga": 0.18, "beta": 0.40},
        "SD": {"median_pga": 0.44, "beta": 0.44},
    },
    "TO": {
        "UPD": {"median_pga": 0.15, "beta": 0.43},
        "SD": {"median_pga": 0.35, "beta": 0.48},
    },
    "IAB": {
        "UPD": {"median_pga": 0.20, "beta": 0.39},
        "SD": {"median_pga": 0.40, "beta": 0.46},
    },
    "standard": {
        "UPD": {"median_pga": 0.16, "beta": 0.45},
        "SD": {"median_pga": 0.36, "beta": 0.50},
    },
    "moderno": {
        "UPD": {"median_pga": 0.24, "beta": 0.36},
        "SD": {"median_pga": 0.58, "beta": 0.40},
    },
}


def get_fragility_profile(pga: float, bridge_type: str = "standard") -> dict:
    """
    Calcola le probabilita' di danno per tutti gli stati dato un PGA.

    Returns:
        Dict {stato: probabilita'} per 4 stati di danno.
    """
    params = FRAGILITY_PARAMS.get(bridge_type, FRAGILITY_PARAMS["standard"])
    return {
        state: fragility_lognormal(pga, p["median_pga"], p["beta"])
        for state, p in params.items()
    }


def get_code_conforming_seismic_fragility(
    pga: float,
    bridge_class: str = "standard",
    limit_state: str = "SD",
) -> dict:
    params = SEISMIC_ARCHETYPE_FRAGILITY.get(bridge_class, SEISMIC_ARCHETYPE_FRAGILITY["standard"])
    state = params.get(limit_state, params["SD"])
    pf = fragility_lognormal(pga, state["median_pga"], state["beta"])
    return {
        "bridge_class": bridge_class,
        "limit_state": limit_state,
        "median_pga": state["median_pga"],
        "beta": state["beta"],
        "failure_probability": pf,
    }


# ---------------------------------------------------------------------------
# Intensita' sismica MCS (Mercalli-Cancani-Sieberg) da PGA
# ---------------------------------------------------------------------------

def pga_to_mcs(pga: float) -> str:
    """Converte PGA in g alla scala MCS (approssimativa)."""
    if pga < 0.003:
        return "I-II (Impercettibile)"
    elif pga < 0.01:
        return "III (Debole)"
    elif pga < 0.025:
        return "IV (Leggero)"
    elif pga < 0.05:
        return "V (Moderato)"
    elif pga < 0.10:
        return "VI (Forte)"
    elif pga < 0.20:
        return "VII (Molto Forte)"
    elif pga < 0.40:
        return "VIII (Rovinoso)"
    else:
        return "IX+ (Distruttivo)"
