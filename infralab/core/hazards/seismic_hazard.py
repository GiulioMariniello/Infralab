"""Modello di pericolosità sismica dipendente dalla posizione (mappa di hazard).

Sostituisce il proxy uniforme con una PGA di sito che dipende dalla distanza
dell'asset da una (o più) zone ad alta pericolosità sismica. La PGA di sito
decade con la distanza dal centro-sorgente:

    PGA(sito) = pga_far + (pga_peak - pga_far) * exp(-(d / decay_km)^2)

dove d è la distanza haversine (km) dal centro di pericolosità. Modello
illustrativo (gaussiano su un centro seismogenetico); si può sostituire con una
griglia di zonazione reale (es. MPS04 / NTC) senza toccare il resto del motore.
"""
from dataclasses import dataclass
from math import asin, cos, exp, radians, sin, sqrt
from typing import List, Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = la2 - la1, lo2 - lo1
    h = sin(dlat / 2) ** 2 + cos(la1) * cos(la2) * sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * asin(min(1.0, sqrt(h)))


@dataclass(frozen=True)
class SeismicHazardModel:
    """Campo di pericolosità: PGA di riferimento (es. 475 anni) su lat/lon."""
    center_lat: float
    center_lon: float
    peak_pga: float          # PGA al centro-sorgente [g]
    far_pga: float           # PGA di fondo lontano dal centro [g]
    decay_km: float          # lunghezza di decadimento [km]

    def site_pga(self, lat: float, lon: float) -> float:
        if lat is None or lon is None:
            return self.far_pga
        d = haversine_km(lat, lon, self.center_lat, self.center_lon)
        return self.far_pga + (self.peak_pga - self.far_pga) * exp(-((d / max(self.decay_km, 1e-6)) ** 2))

    def distance_km(self, lat: float, lon: float) -> float:
        if lat is None or lon is None:
            return float("inf")
        return haversine_km(lat, lon, self.center_lat, self.center_lon)

    def field_grid(self, lat0: float, lon0: float, lat1: float, lon1: float, n: int = 24) -> List[List[float]]:
        """Griglia n×n di PGA sul rettangolo [lat0,lat1]×[lon0,lon1] (per heatmap)."""
        rows: List[List[float]] = []
        for i in range(n):
            lat = lat0 + (lat1 - lat0) * (i / (n - 1))
            row = [self.site_pga(lat, lon0 + (lon1 - lon0) * (jx / (n - 1))) for jx in range(n)]
            rows.append(row)
        return rows

    def bounds_for(self, points: List[Tuple[float, float]], margin_km: float = 25.0) -> Tuple[float, float, float, float]:
        """Rettangolo lat/lon che racchiude i punti e il centro, con margine."""
        lats = [p[0] for p in points if p[0] is not None] + [self.center_lat]
        lons = [p[1] for p in points if p[1] is not None] + [self.center_lon]
        dlat = margin_km / 111.0
        mlat = sum(lats) / len(lats)
        dlon = margin_km / (111.0 * max(0.2, cos(radians(mlat))))
        return (min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon)


def hazard_from_params(params) -> SeismicHazardModel:
    """Costruisce il modello dai SimParameters (centro = zona di pericolosità)."""
    return SeismicHazardModel(
        center_lat=params.seismic_hazard_center_lat,
        center_lon=params.seismic_hazard_center_lon,
        peak_pga=params.seismic_hazard_peak_pga,
        far_pga=params.seismic_hazard_far_pga,
        decay_km=params.seismic_hazard_decay_km,
    )
