"""Eventi estremi regionali (opt-in; con i default non fa nulla).

Perche' esiste. Il canale del danno locale improvviso (sudden_damage.py) sorteggia gli
eventi ponte per ponte, in modo indipendente: non produce mai lo shock simultaneo di
una piena di bacino o di una tempesta, che colpisce molti ponti nello stesso trimestre
e mette sotto pressione insieme budget, squadre e soglia etica. Questo canale aggiunge
eventi rari, spazialmente correlati, sull'intero parco.

Che cosa fa. Ogni trimestre un flusso DEDICATO a chiave (seme, trimestre) estrae il
numero di eventi (Poisson con media regional_event_rate_y / 4). Per ogni evento:
- centro: un ponte scelto a caso (uniforme) fra quelli con coordinate, oppure il punto
  fisso regional_event_center_lat/lon;
- impronta: i ponti entro regional_event_radius_km dal centro (distanza haversine) che
  soddisfano l'esposizione richiesta (regional_event_requires, es. "crosses_water");
- intensita' locale I(d) in [0,1], decrescente con la distanza: "exponential"
  exp(-d/decay), "gaussian" exp(-(d/decay)^2), "uniform" 1 dentro il raggio;
- per ogni ponte dell'impronta, da un flusso a chiave (seme, evento, ponte, trimestre):
  colpito con probabilita' min(1, regional_event_hit_prob x I x moltiplicatori di
  esposizione); se colpito, perdita di condizione VERA dei componenti bersaglio estratta
  dalla distribuzione configurata (come in sudden_damage) e scalata per I (se
  regional_event_scale_damage); autodenuncia con probabilita' regional_event_announce_prob.

Informazione. L'EVENTO e' pubblico: gestore e legge sanno che e' avvenuto, quando, dove
(centro, raggio) e quali ponti esposti ricadono nell'area, con l'intensita' locale (dato
geografico, non strutturale). Il danno per ponte e' invece nascosto finche' un'ispezione
(o il monitoraggio) non lo rileva, salvo la quota autodenunciante: usa la stessa lista
dei danni pendenti del canale improvviso (env.sudden_pending, tipo "regional"), quindi
le stesse chiusure di sandbox, belief_clone e leak_check. I contatori veri del canale
(env.regional_stats: ponti colpiti, perdita...) sono verita' e sono tolti dalle copie
dei pianificatori.

Legge. Il trigger di regime "regional_event_in_area" (environment._regime_trigger_active)
e' vero per un ponte esposto nell'area di un evento non ancora seguito da un'ispezione:
una legge puo' imporre l'ispezione straordinaria entro grace_quarters trimestri
(Linee Guida 2020: ispezioni speciali dopo eventi eccezionali). obligation.threshold,
se dato, e' l'intensita' locale minima.

Numeri casuali comuni. Tutti i flussi sono a chiave e non dipendono dallo stato: stesso
seme -> stessi eventi, stessi centri, stessi sorteggi per ponte, qualunque sia la
strategia; accendere il canale non sposta l'RNG condiviso ne' gli altri flussi.
Il collasso resta campionato dalla Pf (nessun aumento diretto).
"""
import math
from typing import Any, Dict, List, Optional

from infralab.core.hazards import sudden_damage as _sd
from infralab.core.hazards.seismic_hazard import haversine_km

# Scopi dei flussi a chiave: oltre l'intervallo dei tipi utente del canale improvviso
# (PURPOSE_CUSTOM_BASE + 0..99999), quindi senza collisioni.
PURPOSE_OCCURRENCE = 200101
PURPOSE_HIT = 200102

PENDING_TYPE = "regional"
DECAY_MODELS = ("exponential", "gaussian", "uniform")
CENTER_MODES = ("random_bridge", "fixed")
TRIGGER = "regional_event_in_area"

STAT_KEYS = ("events_total", "events_quarter", "bridges_in_area", "bridges_hit", "bridges_hit_quarter",
             "components_hit", "announced", "detected_by_inspection", "detected_by_monitoring",
             "detected_by_other_announcement", "repaired_before_detection", "undetected_at_failure",
             "condition_loss_total")


def enabled(params) -> bool:
    return bool(getattr(params, "regional_events_enabled", False))


def new_stats() -> Dict[str, float]:
    st: Dict[str, float] = {k: 0 for k in STAT_KEYS}
    st["condition_loss_total"] = 0.0
    return st


def _stats(env, agent: str) -> Dict[str, float]:
    if getattr(env, "regional_stats", None) is None:
        env.regional_stats = {}
    st = env.regional_stats.get(agent)
    if st is None:
        st = env.regional_stats[agent] = new_stats()
    return st


def spec(params) -> Dict[str, Any]:
    """Specifica del danno nel formato di sudden_damage.draw_drop."""
    return {"name": PENDING_TYPE,
            "targets": _sd.parse_targets(params.regional_event_targets),
            "hits": params.regional_event_hits,
            "drop_dist": params.regional_event_drop_dist,
            "drop_mean": float(params.regional_event_drop_mean),
            "drop_cov": float(params.regional_event_drop_cov),
            "announce_prob": float(params.regional_event_announce_prob)}


# ---------------------------------------------------------------------------
# Geometria e esposizione (dati pubblici: coordinate e inventario)
# ---------------------------------------------------------------------------
def intensity(params, dist_km: float) -> float:
    radius = float(params.regional_event_radius_km)
    if dist_km > radius:
        return 0.0
    model = params.regional_event_decay_model
    if model == "uniform":
        return 1.0
    decay = max(1e-6, float(params.regional_event_decay_km))
    if model == "gaussian":
        return math.exp(-((dist_km / decay) ** 2))
    return math.exp(-dist_km / decay)


def exposure_factor(params, bridge) -> float:
    """0 se il ponte non e' esposto (regional_event_requires), altrimenti il prodotto
    dei moltiplicatori di esposizione presenti (regional_event_multipliers)."""
    for attr in _sd.parse_targets(params.regional_event_requires):
        if not _sd.exposure_attr(bridge, attr):
            return 0.0
    f = 1.0
    for attr, mult in (params.regional_event_multipliers or {}).items():
        if _sd.exposure_attr(bridge, attr):
            f *= float(mult)
    return f


def footprint(env, lat: float, lon: float) -> Dict[int, float]:
    """{ponte: intensita' locale} per i ponti ESPOSTI dentro il raggio (pubblico)."""
    out: Dict[int, float] = {}
    for bid in sorted(env.bridges):
        b = env.bridges[bid]
        if b.lat is None or b.lon is None or exposure_factor(env.params, b) <= 0.0:
            continue
        i = intensity(env.params, haversine_km(float(b.lat), float(b.lon), lat, lon))
        if i > 0.0:
            out[int(bid)] = round(float(i), 6)
    return out


def _center(env, rng):
    p = env.params
    if p.regional_event_center_mode == "fixed":
        return float(p.regional_event_center_lat), float(p.regional_event_center_lon), None
    located = [bid for bid in sorted(env.bridges)
               if env.bridges[bid].lat is not None and env.bridges[bid].lon is not None]
    if not located:
        return None, None, None
    bid = located[int(rng.integers(len(located)))]
    b = env.bridges[bid]
    return float(b.lat), float(b.lon), int(bid)


# ---------------------------------------------------------------------------
# Dinamica
# ---------------------------------------------------------------------------
def sample_events(env) -> None:
    if not enabled(env.params):
        return
    if getattr(env, "regional_events", None) is None:
        env.regional_events = []
    if getattr(env, "sudden_pending", None) is None:
        env.sudden_pending = {}
    for agent in env.active_agents:
        st = _stats(env, agent)
        st["events_quarter"] = 0
        st["bridges_hit_quarter"] = 0
    # Il monitoraggio rileva anche i danni regionali; se il canale improvviso e' acceso
    # lo ha gia' fatto lui in questo trimestre.
    if not _sd.channel_active(env.params):
        _sd._monitoring_detection(env)
    p = env.params
    lam = max(0.0, float(p.regional_event_rate_y)) / 4.0
    if lam <= 0.0:
        return
    occ = env._keyed_stream(PURPOSE_OCCURRENCE, env.quarter)
    n = int(occ.poisson(lam))
    for k in range(n):
        lat, lon, center_bid = _center(env, occ)
        if lat is None:
            continue
        area = footprint(env, lat, lon)
        event = {"id": len(env.regional_events), "quarter": int(env.quarter), "lat": round(lat, 6),
                 "lon": round(lon, 6), "radius_km": float(p.regional_event_radius_km),
                 "decay_km": float(p.regional_event_decay_km), "decay_model": p.regional_event_decay_model,
                 "center_bridge": center_bid,
                 "bridges": [[bid, i] for bid, i in sorted(area.items())]}
        env.regional_events.append(event)
        for agent in env.active_agents:
            st = _stats(env, agent)
            st["events_total"] += 1
            st["events_quarter"] += 1
        _apply_event(env, event, k, area)


def _apply_event(env, event: Dict[str, Any], k: int, area: Dict[int, float]) -> None:
    p = env.params
    sp = spec(p)
    hit_prob = float(p.regional_event_hit_prob)
    scale = bool(p.regional_event_scale_damage)
    exponent = p.deterioration_exponent
    for bid, inten in sorted(area.items()):
        bridge = env.bridges.get(bid)
        if bridge is None or bridge.manager not in env.active_agents:
            continue
        st = _stats(env, bridge.manager)
        st["bridges_in_area"] += 1
        # Ordine delle estrazioni fisso: occorrenza, bersaglio, intensita', autodenuncia.
        rng = env._keyed_stream(PURPOSE_HIT, k, bid, env.quarter)
        prob = min(1.0, hit_prob * inten * exposure_factor(p, bridge))
        if not float(rng.random()) < prob:
            continue
        targets = [t for t in sp["targets"] if t in bridge.components]
        if not targets:
            continue
        hit = _sd.pick_hits(bridge, targets, sp["hits"], rng)
        losses = {}
        for name in hit:
            drop = _sd.draw_drop(sp, rng) * (inten if scale else 1.0)
            losses[name] = _sd._damage_hidden(bridge.components[name], drop, exponent)
        announced = bool(rng.random() < sp["announce_prob"])
        st["bridges_hit"] += 1
        st["bridges_hit_quarter"] += 1
        st["components_hit"] += len(hit)
        st["condition_loss_total"] += float(sum(losses.values()))
        pending = env.sudden_pending.setdefault(bridge.id, [])
        for name in hit:
            pending.append({"type": PENDING_TYPE, "quarter": int(env.quarter), "component": name,
                            "loss": round(float(losses[name]), 6), "event": int(event["id"])})
        if announced:
            _sd.reveal(env, bridge, hit, source="regional", event_type=PENDING_TYPE,
                       own_event=int(event["id"]))
        if getattr(p, "progressive_events_enabled", False):
            # il ponte colpito puo' innescare un difetto idraulico latente (default: mai)
            from infralab.core.hazards import progressive_events as _progressive
            _progressive.on_regional_hit(env, bridge, int(event["id"]))
        env.refresh_bridge_reliability(bridge)


def on_resolved(env, bridge, entries: List[Dict[str, Any]], source: Optional[str],
                own_event: Optional[int] = None) -> None:
    """Contabilita' dei danni regionali usciti dalla lista dei pendenti (chiamata da
    sudden_damage._resolve)."""
    if not entries:
        return
    st = _stats(env, bridge.manager)
    if source == "inspection":
        st["detected_by_inspection"] += len(entries)
    elif source == "monitoring":
        st["detected_by_monitoring"] += len(entries)
    elif source == "repair":
        st["repaired_before_detection"] += len(entries)
    elif source == "failure":
        st["undetected_at_failure"] += len(entries)
    elif source == "regional":
        own = sum(1 for e in entries if own_event is not None and e.get("event") == own_event
                  and int(e.get("quarter", -1)) == int(env.quarter))
        st["announced"] += own
        st["detected_by_other_announcement"] += len(entries) - own
    else:   # autodenuncia di un evento del canale improvviso
        st["detected_by_other_announcement"] += len(entries)


def metrics(env, agent: str) -> Dict[str, float]:
    """Metriche per l'output del laboratorio (vuoto se il canale e' spento)."""
    if not enabled(env.params):
        return {}
    st = (getattr(env, "regional_stats", None) or {}).get(agent) or new_stats()
    pending = sum(1 for bid, evs in (getattr(env, "sudden_pending", None) or {}).items()
                  if bid in env.bridges and env.bridges[bid].manager == agent
                  for e in evs if e.get("type") == PENDING_TYPE)
    out = {f"regional_{k}": float(v) for k, v in st.items()}
    out["regional_pending_undetected"] = float(pending)
    return out


# ---------------------------------------------------------------------------
# Informazione pubblica
# ---------------------------------------------------------------------------
def area_status(env, bridge, min_intensity: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Ultimo evento la cui area contiene il ponte (esposto) e che non e' ancora stato
    seguito da un'ispezione: {quarter, event, intensity}; None altrimenti. Solo dati
    pubblici (evento, geografia, data dell'ultima ispezione)."""
    events = getattr(env, "regional_events", None) or []
    if not events:
        return None
    last_insp = int(getattr(bridge, "last_inspection_quarter", 0) or 0)
    th = None if min_intensity is None else float(min_intensity)
    for ev in reversed(events):
        # l'ispezione del trimestre dell'evento precede l'evento (azioni prima dei pericoli)
        if int(ev["quarter"]) < last_insp:
            break
        for bid, inten in ev.get("bridges") or []:
            if int(bid) == int(bridge.id) and (th is None or float(inten) >= th):
                return {"quarter": int(ev["quarter"]), "event": int(ev["id"]), "intensity": float(inten)}
    return None


def public_events(env) -> List[Dict[str, Any]]:
    return [{"id": e["id"], "quarter": e["quarter"], "lat": e["lat"], "lon": e["lon"],
             "radius_km": e["radius_km"], "decay_km": e["decay_km"], "decay_model": e["decay_model"],
             "bridges_in_area": [int(b) for b, _ in e.get("bridges") or []]}
            for e in (getattr(env, "regional_events", None) or [])]


def observation_fields(env, bridge) -> Dict[str, Any]:
    status = area_status(env, bridge)
    return {"regional_event_uninspected": status is not None,
            "regional_event_quarter": status["quarter"] if status else None,
            "regional_event_intensity": status["intensity"] if status else 0.0}


# ---------------------------------------------------------------------------
# Validazione (chiamata da SimParameters.__post_init__)
# ---------------------------------------------------------------------------
def validate_params(params) -> List[str]:
    problems: List[str] = []

    def _num(name, lo=0.0, hi=None):
        value = getattr(params, name)
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v >= lo and (hi is None or v <= hi)):
            rng = f"[{lo},{hi}]" if hi is not None else f">= {lo}"
            problems.append(f"{name} deve essere un numero finito {rng} (trovato {value!r})")

    if type(params.regional_events_enabled).__name__ not in ("bool", "bool_"):
        problems.append(f"regional_events_enabled deve essere un booleano (trovato {params.regional_events_enabled!r})")
    for name in ("regional_event_rate_y", "regional_event_drop_cov"):
        _num(name)
    _num("regional_event_radius_km", 0.0)
    _num("regional_event_decay_km", 1e-6)
    for name in ("regional_event_hit_prob", "regional_event_drop_mean", "regional_event_announce_prob"):
        _num(name, 0.0, 1.0)
    _num("regional_event_center_lat", -90.0, 90.0)
    _num("regional_event_center_lon", -180.0, 180.0)
    if params.regional_event_decay_model not in DECAY_MODELS:
        problems.append(f"regional_event_decay_model {params.regional_event_decay_model!r} non valido ({DECAY_MODELS})")
    if params.regional_event_center_mode not in CENTER_MODES:
        problems.append(f"regional_event_center_mode {params.regional_event_center_mode!r} non valido ({CENTER_MODES})")
    if params.regional_event_drop_dist not in _sd.DROP_DISTS:
        problems.append(f"regional_event_drop_dist {params.regional_event_drop_dist!r} non valido ({_sd.DROP_DISTS})")
    if params.regional_event_hits not in _sd.HITS:
        problems.append(f"regional_event_hits {params.regional_event_hits!r} non valido ({_sd.HITS})")
    if not _sd.parse_targets(params.regional_event_targets):
        problems.append("regional_event_targets: serve almeno un componente bersaglio")
    if type(params.regional_event_scale_damage).__name__ not in ("bool", "bool_"):
        problems.append("regional_event_scale_damage deve essere un booleano")
    mults = params.regional_event_multipliers
    if not isinstance(mults, dict):
        problems.append("regional_event_multipliers deve essere un dizionario {attributo: fattore}")
    else:
        for k, v in mults.items():
            try:
                ok = math.isfinite(float(v)) and float(v) >= 0.0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                problems.append(f"regional_event_multipliers.{k} deve essere un numero finito >= 0 (trovato {v!r})")
    return problems


# ---------------------------------------------------------------------------
# Metadati per i form (backend /api/config e UI Streamlit): un'unica fonte
# ---------------------------------------------------------------------------
_G = "Eventi estremi regionali"
UI_FIELDS: List[Dict[str, Any]] = [
    {"key": "regional_events_enabled", "label": "Eventi regionali abilitati", "group": _G, "type": "bool"},
    {"key": "regional_event_rate_y", "label": "Eventi/anno sull'intero parco", "group": _G, "type": "number",
     "min": 0, "max": 10, "step": 0.01},
    {"key": "regional_event_center_mode", "label": "Centro dell'evento", "group": _G, "type": "select",
     "options": list(CENTER_MODES)},
    {"key": "regional_event_center_lat", "label": "Centro fisso: lat", "group": _G, "type": "number",
     "min": -90, "max": 90, "step": 0.01},
    {"key": "regional_event_center_lon", "label": "Centro fisso: lon", "group": _G, "type": "number",
     "min": -180, "max": 180, "step": 0.01},
    {"key": "regional_event_radius_km", "label": "Raggio dell'area colpita (km)", "group": _G, "type": "number",
     "min": 0, "max": 1000, "step": 1},
    {"key": "regional_event_decay_model", "label": "Decadimento dell'intensita'", "group": _G, "type": "select",
     "options": list(DECAY_MODELS)},
    {"key": "regional_event_decay_km", "label": "Lunghezza di decadimento (km)", "group": _G, "type": "number",
     "min": 0.001, "max": 1000, "step": 1},
    {"key": "regional_event_requires", "label": "Esposizione richiesta (attributi, virgole; vuoto = tutti)",
     "group": _G, "type": "text"},
    {"key": "regional_event_multipliers", "label": "Moltiplicatori di esposizione (JSON {attributo: fattore})",
     "group": _G, "type": "json"},
    {"key": "regional_event_hit_prob", "label": "Probabilita' di danno al centro", "group": _G, "type": "number",
     "min": 0, "max": 1, "step": 0.05},
    {"key": "regional_event_targets", "label": "Componenti bersaglio", "group": _G, "type": "text"},
    {"key": "regional_event_hits", "label": "Bersagli colpiti", "group": _G, "type": "select", "options": list(_sd.HITS)},
    {"key": "regional_event_drop_dist", "label": "Distribuzione del danno", "group": _G, "type": "select",
     "options": list(_sd.DROP_DISTS)},
    {"key": "regional_event_drop_mean", "label": "Perdita di condizione media (al centro)", "group": _G,
     "type": "number", "min": 0, "max": 1, "step": 0.01},
    {"key": "regional_event_drop_cov", "label": "CoV della perdita", "group": _G, "type": "number",
     "min": 0, "max": 3, "step": 0.05},
    {"key": "regional_event_scale_damage", "label": "Perdita scalata per l'intensita' locale", "group": _G,
     "type": "bool"},
    {"key": "regional_event_announce_prob", "label": "Quota autodenunciante per ponte", "group": _G,
     "type": "number", "min": 0, "max": 1, "step": 0.05},
]
