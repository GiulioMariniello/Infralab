# Modello di degrado calibrato su NBI Delaware (anni 2016-2023, 674 ponti, 15467 osservazioni).
# Generato da infralab.data.nbi_calibration. Curva C=exp(-scale*(t/L)^k)
# con parametri fittati per componente; fallback ai default del profilo.
import math

CALIBRATED = {'impalcato': {'scale': 0.3573, 'k': 0.2138}, 'appoggi': {'scale': 0.4198, 'k': 0.3024}, 'pile': {'scale': 0.4663, 'k': 0.2875}, 'spalle': {'scale': 0.4663, 'k': 0.2875}}
LIFE = 100.0

def degrade(ctx):
    L = ctx['vita_utile_anni'] or LIFE
    t = ctx['effective_age_years']
    if L <= 0 or t <= 0:
        return 1.0
    p = CALIBRATED.get(ctx['component_name'])
    scale = p['scale'] if p else ctx['scale']
    k = p['k'] if p else ctx['shape']
    return math.exp(-scale * (t / L) ** k)
