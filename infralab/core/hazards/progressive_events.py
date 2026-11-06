"""Eventi progressivi (latenti): innesco, sviluppo in una finestra, arrestabili se visti
(opt-in; con i default non fa nulla).

Perche' esiste. Il crollo idraulico diretto (environment._sample_hydraulic_events) e'
istantaneo e identico per ogni strategia: nessun gestore puo' evitarlo. Nella realta'
lo scalzamento e i movimenti di versante si innescano, si sviluppano per mesi o anni e
un'ispezione nella finestra permette di fermarli. Struttura: modello delay-time
(arrivo del difetto, ritardo casuale, conseguenza se non trattato, ispezione imperfetta
nel ritardo). Fonti e provenienza dei default: docs/CHANGELOG_MODEL.md e la nota
papers/infralab-regbench/experiments/v3-design/letteratura-eventi-progressivi.md.

Che cosa fa (tipi: "hydraulic", "geotechnical", tipi utente `progressive_custom_events`).
1. Innesco. Ogni trimestre, per tipo, un flusso DEDICATO a chiave (seme, tipo,
   trimestre) estrae un numero per ponte (ordine fisso degli id): innesco se minore del
   tasso del ponte. Il tasso dipende solo dall'inventario, quindi stessi inneschi per
   ogni strategia. Idraulico: ponti su acqua, con la stessa esposizione del crollo
   diretto. Geotecnico: attributo `on_unstable_slope` (dallo scenario o assegnato a una
   quota configurabile dei ponti con un flusso a chiave). Facoltativo: un ponte colpito
   da un evento regionale innesca un difetto idraulico con probabilita' configurabile.
2. Anti-sovradimensionamento. `hydraulic_collapse_progressive_share` = quota dei crolli
   idraulici diretti RIASSEGNATA al canale: il tasso diretto e' moltiplicato per
   (1 - quota) e, nel modo "calibrated", il tasso di innesco del ponte e'
   quota x tasso diretto storico / p_collapse_if_untreated: per un gestore che non
   ispeziona e non arresta il tasso atteso di crollo idraulico resta quello storico
   (a regime; su orizzonte finito vedi `expected_hydraulic_collapse_rates`).
3. Stato latente. Alla nascita il difetto riceve una durata (trimestri) dalla
   distribuzione del tipo; fa parte dello stato VERO ed e' nascosto a gestore e legge.
4. Rilevazione. A ogni ispezione del ponte: probabilita' = POD dell'ispezione x fattore
   del tipo (ispezione visiva ordinaria / ispezione di dettaglio). Una quota e'
   autoevidente alla nascita. Il monitoraggio attivo puo' rilevarlo (facoltativo). Un
   difetto NOTO compare nell'osservazione (tipo, trimestre in cui e' diventato noto),
   mai con la data di fine.
5. Arresto. Azione dedicata `arresto_difetto` (costo = rapporto del tipo x costo
   dell'intervento di riferimento del catalogo) oppure qualunque intervento che
   ripristina i componenti bersaglio. Con la soglia etica un difetto noto e' un trigger:
   arresto d'ufficio se cassa e squadre lo consentono, altrimenti chiusura.
6. Conseguenza a fine finestra se non arrestato: con probabilita'
   p_collapse_if_untreated crollo (come un crollo diretto: perdita, ricostruzione,
   evento pubblico), altrimenti perdita di condizione dei componenti bersaglio, nascosta
   fino all'ispezione con la lista dei danni pendenti del canale improvviso
   (env.sudden_pending, tipo "progressive:<tipo>"); se il difetto era noto il danno e'
   reso noto subito.

Informazione. Difetti non noti, date di inizio e fine e contatori veri non escono mai
verso policy e pianificatori in modalita' "inspected": vedi agents/sandbox.py,
BridgeEnv.belief_clone e lab/leak_check.py. I difetti noti sono evidenza legittima.

Numeri casuali comuni. Tutti i flussi sono a chiave (indipendenti da rng_streams):
stesso seme -> stessi inneschi, stesse durate, stessi sorteggi di esito per ogni
strategia; accendere il canale non sposta nessun altro flusso.
"""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from infralab.core.hazards import sudden_damage as _sd

# Scopi dei flussi a chiave (oltre quelli del canale improvviso e degli eventi regionali).
PURPOSE_INIT = 210101        # (tipo, trimestre): un numero per ponte
PURPOSE_DETAIL = 210102      # (tipo, ponte, trimestre, origine): durata, autoevidenza
PURPOSE_OUTCOME = 210103     # (ponte, tipo, trimestre di innesco, origine): crollo, perdita
PURPOSE_DETECT = 210104      # (ponte, trimestre, n-esima ispezione, difetto)
PURPOSE_MONITOR = 210105     # (ponte, trimestre, difetto)
PURPOSE_EXPOSURE = 210106    # (ponte): esposizione geotecnica assegnata
PURPOSE_REGIONAL = 210107    # (evento, ponte, trimestre): innesco da evento regionale
PURPOSE_REDRAW = 210108      # (ponte, trimestre, n): durata mancante (stato importato)

BUILTIN_TYPES = ("hydraulic", "geotechnical")
TYPE_CODE = {"hydraulic": 1, "geotechnical": 2}
SOURCE_CODE = {"initiation": 0, "regional": 1, "leak_check": 2}
DURATION_DISTS = ("geometric", "lognormal", "fixed")
RATE_MODES = ("calibrated", "fixed")
POD_MODES = ("inspection", "independent")
GAME_OVER_MODES = ("none", "any", "known_critical")
PENDING_PREFIX = "progressive:"
ARREST_ACTION = "arresto_difetto"
ARREST_TYPE = "arrest"
GEOTECHNICAL_ATTR = "on_unstable_slope"
CUSTOM_EVENT_KEYS = ("name", "rate_y", "requires", "multipliers", "targets", "hits", "drop_dist", "drop_mean",
                     "drop_cov", "duration_dist", "duration_mean_q", "duration_cov", "self_evident_prob",
                     "detect_factor_l1", "detect_factor_detailed", "p_collapse_if_untreated",
                     "arrest_cost_ratio")

# Gancio per lab/leak_check (modalita' traiettoria): funzione
# (env, ponte, "inspection"|"monitoring", n, difetti trovati) -> difetti trovati.
DETECTION_HOOK: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Specifiche dei tipi
# ---------------------------------------------------------------------------
def enabled(params) -> bool:
    return bool(getattr(params, "progressive_events_enabled", False))


def custom_code(name: str) -> int:
    return 1000 + sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 100000


def _builtin_spec(params, name: str) -> Dict[str, Any]:
    def g(key):
        return getattr(params, f"progressive_{name}_{key}")
    spec = {"name": name, "code": TYPE_CODE[name],
            "duration_dist": g("duration_dist"), "duration_mean_q": float(g("duration_mean_q")),
            "duration_cov": float(g("duration_cov")),
            "detect_factor_l1": float(g("detect_factor_l1")),
            "detect_factor_detailed": float(g("detect_factor_detailed")),
            "self_evident_prob": float(g("self_evident_prob")),
            "p_collapse": float(g("p_collapse_if_untreated")),
            "arrest_cost_ratio": float(g("arrest_cost_ratio"))}
    if name == "hydraulic":
        # perdita a fine finestra = quella del danno locale idraulico (nessun secondo
        # livello di gravita' non giustificato)
        h = _sd.hydraulic_spec(params)
        spec.update({k: h[k] for k in ("targets", "hits", "drop_dist", "drop_mean", "drop_cov")})
    else:
        spec.update({"targets": _sd.parse_targets(g("targets")), "hits": g("hits"), "drop_dist": g("drop_dist"),
                     "drop_mean": float(g("drop_mean")), "drop_cov": float(g("drop_cov"))})
    return spec


def _custom_spec(ev: Dict[str, Any]) -> Dict[str, Any]:
    name = str(ev["name"])
    return {"name": name, "code": custom_code(name), "custom": ev,
            "duration_dist": ev.get("duration_dist", "geometric"),
            "duration_mean_q": float(ev.get("duration_mean_q", 4.0)),
            "duration_cov": float(ev.get("duration_cov", 1.0)),
            "detect_factor_l1": float(ev.get("detect_factor_l1", 1.0)),
            "detect_factor_detailed": float(ev.get("detect_factor_detailed", 1.0)),
            "self_evident_prob": float(ev.get("self_evident_prob", 0.0)),
            "p_collapse": float(ev.get("p_collapse_if_untreated", 0.0)),
            "arrest_cost_ratio": float(ev.get("arrest_cost_ratio", 0.3)),
            "targets": _sd.parse_targets(ev.get("targets", "pile,spalle")), "hits": ev.get("hits", "all"),
            "drop_dist": ev.get("drop_dist", "fixed"), "drop_mean": float(ev.get("drop_mean", 0.2)),
            "drop_cov": float(ev.get("drop_cov", 0.0))}


def active_specs(params) -> List[Dict[str, Any]]:
    if not enabled(params):
        return []
    specs = [_builtin_spec(params, name) for name in BUILTIN_TYPES
             if bool(getattr(params, f"progressive_{name}_enabled", False))]
    specs.extend(_custom_spec(ev) for ev in (getattr(params, "progressive_custom_events", None) or []))
    return specs


def type_names(params) -> List[str]:
    return [s["name"] for s in active_specs(params)]


def _spec_by_name(params) -> Dict[str, Dict[str, Any]]:
    return {s["name"]: s for s in active_specs(params)}


def _type_code(name: str) -> int:
    return TYPE_CODE.get(name) or custom_code(name)


def _dkey(defect: Dict[str, Any]) -> Tuple[int, int, int]:
    """Identita' del difetto nei flussi a chiave: non dipende dalla posizione nella
    lista (che cambia con gli arresti), quindi e' la stessa per ogni strategia."""
    return (_type_code(defect["type"]), int(defect.get("start_quarter") or 0),
            SOURCE_CODE.get(defect.get("source", "initiation"), 9))


# ---------------------------------------------------------------------------
# Esposizione e tassi (solo inventario: nessuna dipendenza dallo stato)
# ---------------------------------------------------------------------------
def hydraulic_progressive_active(params) -> bool:
    return enabled(params) and bool(getattr(params, "progressive_hydraulic_enabled", False))


def effective_share(params) -> float:
    """Quota dei crolli idraulici diretti riassegnata al canale. Vale solo con il tipo
    idraulico acceso: senza, togliere crolli diretti sarebbe una sottrazione e basta."""
    if not hydraulic_progressive_active(params):
        return 0.0
    return min(1.0, max(0.0, float(getattr(params, "hydraulic_collapse_progressive_share", 0.0) or 0.0)))


def hydraulic_direct_rate_q(params, bridge) -> float:
    """Tasso trimestrale STORICO del crollo idraulico diretto del ponte (prima della
    quota riassegnata): stessa esposizione di environment._sample_hydraulic_events."""
    if not getattr(params, "hydraulic_hazard_enabled", False) or not getattr(bridge, "crosses_water", False):
        return 0.0
    return (float(params.hydraulic_collapse_rate_scour_q) if getattr(bridge, "scour_critical", False)
            else float(params.hydraulic_collapse_rate_water_q))


def initiation_rate_q(params, bridge, spec: Dict[str, Any]) -> float:
    """Probabilita' trimestrale di innesco del tipo sul ponte."""
    name = spec["name"]
    if name == "hydraulic":
        if not getattr(bridge, "crosses_water", False):
            return 0.0
        if params.progressive_hydraulic_rate_mode == "calibrated":
            p = float(spec["p_collapse"])
            if p <= 0.0:
                return 0.0
            rate = effective_share(params) * hydraulic_direct_rate_q(params, bridge) / p
        else:
            rate = float(params.progressive_hydraulic_rate_y) / 4.0
            if getattr(bridge, "scour_critical", False):
                rate *= float(params.progressive_hydraulic_scour_multiplier)
    elif name == "geotechnical":
        rate = (float(params.progressive_geotechnical_rate_y) if bool(getattr(bridge, GEOTECHNICAL_ATTR, False))
                else float(params.progressive_geotechnical_rate_other_y)) / 4.0
    else:
        ev = spec["custom"]
        for attr in _sd.parse_targets(ev.get("requires")):
            if not _sd.exposure_attr(bridge, attr):
                return 0.0
        rate = float(ev.get("rate_y", 0.0)) / 4.0
        for attr, mult in (ev.get("multipliers") or {}).items():
            if _sd.exposure_attr(bridge, attr):
                rate *= float(mult)
    return min(1.0, max(0.0, rate))


def assign_exposure(env) -> None:
    """Esposizione geotecnica dei ponti che non la portano dallo scenario: una quota
    `progressive_geotechnical_exposed_fraction` e' assegnata con un flusso a chiave
    (seme, ponte). E' un dato d'inventario (pubblico), fissato alla costruzione."""
    p = env.params
    if not enabled(p) or not bool(getattr(p, "progressive_geotechnical_enabled", False)):
        return
    fraction = float(getattr(p, "progressive_geotechnical_exposed_fraction", 0.0) or 0.0)
    if fraction <= 0.0:
        return
    for bid in sorted(env.bridges):
        bridge = env.bridges[bid]
        if getattr(bridge, GEOTECHNICAL_ATTR, None) is None:
            u = float(env._keyed_stream(PURPOSE_EXPOSURE, bridge.id).random())
            setattr(bridge, GEOTECHNICAL_ATTR, bool(u < fraction))


# ---------------------------------------------------------------------------
# Durata della finestra
# ---------------------------------------------------------------------------
def draw_duration(spec: Dict[str, Any], rng) -> int:
    """Trimestri fra l'innesco e la fine della finestra (>= 1)."""
    mean = max(1.0, float(spec["duration_mean_q"]))
    dist = spec["duration_dist"]
    cov = max(0.0, float(spec.get("duration_cov", 0.0)))
    if dist == "geometric":      # senza memoria: "fino alla prossima piena sopra soglia"
        return int(rng.geometric(min(1.0, 1.0 / mean)))
    if dist == "fixed" or cov <= 0.0:
        return max(1, int(round(mean)))
    s2 = math.log(1.0 + cov * cov)   # lognormale con media e CoV assegnati, arrotondata
    return max(1, int(round(float(rng.lognormal(math.log(mean) - 0.5 * s2, math.sqrt(s2))))))


def duration_cdf(spec: Dict[str, Any], k: int) -> float:
    """P(durata <= k trimestri) per la distribuzione del tipo (come in draw_duration)."""
    if k < 1:
        return 0.0
    mean = max(1.0, float(spec["duration_mean_q"]))
    dist = spec["duration_dist"]
    cov = max(0.0, float(spec.get("duration_cov", 0.0)))
    if dist == "geometric":
        return 1.0 - (1.0 - min(1.0, 1.0 / mean)) ** k
    if dist == "fixed" or cov <= 0.0:
        return 1.0 if k >= max(1, int(round(mean))) else 0.0
    s2 = math.log(1.0 + cov * cov)
    z = (math.log(k + 0.5) - (math.log(mean) - 0.5 * s2)) / math.sqrt(s2)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def matured_fraction(spec: Dict[str, Any], horizon_quarters: int) -> float:
    """Quota degli inneschi dei trimestri 1..H che matura entro H (orizzonte finito)."""
    h = int(horizon_quarters)
    if h <= 0:
        return 1.0
    return sum(duration_cdf(spec, h - t) for t in range(1, h + 1)) / h


# ---------------------------------------------------------------------------
# Contabilita'
# ---------------------------------------------------------------------------
# Contatori VERI per gestore. initiated: difetti nati (per tipo: initiated_<tipo>);
# self_evident: noti dalla nascita; regional_triggered: nati da un evento regionale;
# detected_by_inspection / detected_by_monitoring: resi noti; inspections_missed:
# ispezioni che non hanno visto un difetto presente; arrested: fermati (dall'azione
# dedicata, da un intervento, dalla soglia etica); arrested_before_detection: fermati da
# un intervento senza essere mai stati noti; matured: arrivati a fine finestra;
# matured_known: di cui noti; collapses: crolli; removed_by_failure: cancellati dal
# crollo del ponte per altra causa. damage_*: sorte del danno nascosto lasciato da una
# maturazione senza crollo (per componente). floor_*: azioni della soglia etica sui
# difetti noti.
STAT_KEYS = ("initiated", "initiated_quarter", "self_evident", "regional_triggered",
             "detected_by_inspection", "detected_by_monitoring", "inspections_missed",
             "arrested", "arrested_by_action", "arrested_by_intervention", "arrested_by_floor",
             "arrested_before_detection", "matured", "matured_known", "collapses", "removed_by_failure",
             "condition_loss_total", "damage_revealed", "damage_detected_by_inspection",
             "damage_detected_by_monitoring", "damage_repaired_before_detection",
             "damage_undetected_at_failure",
             "floor_arrests", "floor_arrest_spend", "floor_budget_blocked", "floor_capacity_blocked",
             "floor_closures")
PER_TYPE_KEYS = ("initiated", "known", "arrested", "matured", "collapses")
_FLOAT_KEYS = ("condition_loss_total", "floor_arrest_spend")


def new_stats(params) -> Dict[str, float]:
    st: Dict[str, float] = {k: 0 for k in STAT_KEYS}
    for k in _FLOAT_KEYS:
        st[k] = 0.0
    for name in type_names(params):
        for k in PER_TYPE_KEYS:
            st[f"{k}_{name}"] = 0
    return st


def _ensure(env) -> None:
    if getattr(env, "progressive_defects", None) is None:
        env.progressive_defects = {}
    if getattr(env, "progressive_stats", None) is None:
        env.progressive_stats = {}
    if getattr(env, "sudden_pending", None) is None:
        env.sudden_pending = {}


def _stats(env, agent: str) -> Dict[str, float]:
    if getattr(env, "progressive_stats", None) is None:
        env.progressive_stats = {}
    st = env.progressive_stats.get(agent)
    if st is None:
        st = env.progressive_stats[agent] = new_stats(env.params)
    return st


def _bump(st: Dict[str, float], key: str, n: float = 1) -> None:
    st[key] = st.get(key, 0) + n


def metrics(env, agent: str) -> Dict[str, float]:
    """Metriche per l'output del laboratorio (vuoto se il canale e' spento)."""
    if not enabled(env.params):
        return {}
    st = (getattr(env, "progressive_stats", None) or {}).get(agent) or new_stats(env.params)
    hidden = known = 0
    for bid, defs in (getattr(env, "progressive_defects", None) or {}).items():
        if bid in env.bridges and env.bridges[bid].manager == agent:
            for d in defs:
                if d.get("known_since") is None:
                    hidden += 1
                else:
                    known += 1
    damage = sum(1 for bid, evs in (getattr(env, "sudden_pending", None) or {}).items()
                 if bid in env.bridges and env.bridges[bid].manager == agent
                 for e in evs if is_pending_type(e.get("type")))
    out = {f"progressive_{k}": float(v) for k, v in st.items()}
    out["progressive_pending_hidden"] = float(hidden)
    out["progressive_pending_known"] = float(known)
    out["progressive_damage_pending_undetected"] = float(damage)
    return out


def is_pending_type(type_name: Any) -> bool:
    return isinstance(type_name, str) and type_name.startswith(PENDING_PREFIX)


# ---------------------------------------------------------------------------
# Conoscenza del gestore (evidenza legittima)
# ---------------------------------------------------------------------------
def known_defects(env, bridge) -> List[Dict[str, Any]]:
    defs = (getattr(env, "progressive_defects", None) or {}).get(bridge.id)
    if not defs:
        return []
    return [d for d in defs if d.get("known_since") is not None]


def public_view(env, bridge) -> List[Dict[str, Any]]:
    """Difetti noti del ponte come li vede il gestore: tipo e trimestre in cui sono
    diventati noti. Mai inizio vero, fine, origine."""
    return sorted(({"type": d["type"], "known_since": int(d["known_since"])} for d in known_defects(env, bridge)),
                  key=lambda x: (x["known_since"], x["type"]))


def floor_trigger(env, bridge) -> bool:
    """Un difetto noto e' un trigger della soglia etica (progressive_floor_trigger)."""
    p = env.params
    return (enabled(p) and bool(getattr(p, "progressive_floor_trigger", True))
            and bool(known_defects(env, bridge)))


def _make_known(env, bridge, defect: Dict[str, Any], how: str) -> None:
    defect["known_since"] = int(env.quarter)
    st = _stats(env, bridge.manager)
    _bump(st, f"known_{defect['type']}")
    env._record_action_event(bridge, "difetto_progressivo", how, {"event_type": defect["type"]})


# ---------------------------------------------------------------------------
# Innesco
# ---------------------------------------------------------------------------
def _start_defect(env, bridge, spec: Dict[str, Any], source: str, key: Tuple[int, ...] = ()) -> Dict[str, Any]:
    """Nasce un difetto. Ordine delle estrazioni fisso: durata, autoevidenza."""
    rng = env._keyed_stream(PURPOSE_DETAIL, spec["code"], bridge.id, env.quarter, SOURCE_CODE[source], *key)
    duration = draw_duration(spec, rng)
    evident = bool(float(rng.random()) < float(spec["self_evident_prob"]))
    defect = {"type": spec["name"], "start_quarter": int(env.quarter),
              "end_quarter": int(env.quarter) + int(duration), "known_since": None, "source": source}
    env.progressive_defects.setdefault(bridge.id, []).append(defect)
    st = _stats(env, bridge.manager)
    _bump(st, "initiated")
    _bump(st, "initiated_quarter")
    _bump(st, f"initiated_{spec['name']}")
    if source == "regional":
        _bump(st, "regional_triggered")
    if evident:
        _bump(st, "self_evident")
        _make_known(env, bridge, defect, "self_evident")
    return defect


def _initiate(env) -> None:
    p = env.params
    bids = sorted(env.bridges)
    for spec in active_specs(p):
        rates = [initiation_rate_q(p, env.bridges[bid], spec) for bid in bids]
        if not any(r > 0.0 for r in rates):
            continue
        # un solo flusso per (tipo, trimestre): un numero per ponte in ordine di id,
        # estratto sempre (anche per i ponti non esposti), cosi' non dipende dallo stato
        u = env._keyed_stream(PURPOSE_INIT, spec["code"], env.quarter).random(len(bids))
        for i, bid in enumerate(bids):
            if rates[i] <= 0.0 or float(u[i]) >= rates[i]:
                continue
            bridge = env.bridges[bid]
            if bridge.manager not in env.active_agents:
                continue
            _start_defect(env, bridge, spec, "initiation")


def on_regional_hit(env, bridge, event_index: int) -> None:
    """Un ponte colpito da un evento regionale puo' innescare un difetto idraulico
    (progressive_hydraulic_regional_trigger_prob; inneschi IN PIU' rispetto alla taratura)."""
    p = env.params
    if not hydraulic_progressive_active(p) or not getattr(bridge, "crosses_water", False):
        return
    prob = float(getattr(p, "progressive_hydraulic_regional_trigger_prob", 0.0) or 0.0)
    if prob <= 0.0:
        return
    u = float(env._keyed_stream(PURPOSE_REGIONAL, int(event_index), bridge.id, env.quarter).random())
    if u < prob:
        _ensure(env)
        _start_defect(env, bridge, _builtin_spec(p, "hydraulic"), "regional", key=(int(event_index),))


# ---------------------------------------------------------------------------
# Rilevazione
# ---------------------------------------------------------------------------
def is_detailed(params, op: Dict[str, Any]) -> bool:
    """Ispezione di dettaglio (fattore "detailed") o visiva ordinaria (fattore "l1")?

    Di dettaglio: `inspection_level` >= progressive_detailed_inspection_min_level (L2,
    L4) e il monitoraggio strumentale. Ordinaria: L1, sorveglianza intermedia,
    censimento L0 e rilievo post-intervento (nessun livello dichiarato)."""
    level = op.get("inspection_level")
    if level is not None:
        return int(level) >= int(getattr(params, "progressive_detailed_inspection_min_level", 2))
    return op.get("type") == "monitoring"


def _inspection_index(env, bridge_id: int) -> int:
    tag = getattr(env, "_prog_insp", None)
    if tag is None or tag[0] != env.quarter:
        tag = env._prog_insp = (env.quarter, {})
    tag[1][bridge_id] = tag[1].get(bridge_id, 0) + 1
    return tag[1][bridge_id]


def on_inspection(env, bridge, op: Dict[str, Any], detected_components) -> None:
    """Un'ispezione del ponte puo' scoprire i difetti latenti: per ciascuno,
    probabilita' = POD dell'ispezione x fattore del tipo per il livello dell'ispezione.

    POD dell'ispezione (progressive_detection_pod_mode="inspection"): 1 se il modello
    POD del motore e' spento; se e' acceso, il difetto puo' essere visto solo se
    l'ispezione ha "rilevato" almeno un componente bersaglio (stesso esito POD usato
    per il danno improvviso). "independent": la POD del motore non conta."""
    n = _inspection_index(env, bridge.id)
    p = env.params
    specs = _spec_by_name(p)
    detailed = is_detailed(p, op)
    use_pod = (bool(getattr(p, "inspection_pod_enabled", False))
               and getattr(p, "progressive_detection_pod_mode", "inspection") == "inspection")
    found, missed = [], 0
    for d in (env.progressive_defects.get(bridge.id) or []):
        if d.get("known_since") is not None:
            continue
        spec = specs.get(d["type"])
        factor = 1.0 if spec is None else float(spec["detect_factor_detailed" if detailed else "detect_factor_l1"])
        base = 1.0
        if use_pod and spec is not None:
            base = 1.0 if any(t in detected_components for t in spec["targets"]) else 0.0
        u = float(env._keyed_stream(PURPOSE_DETECT, bridge.id, env.quarter, n, *_dkey(d)).random())
        if u < base * factor:
            found.append(d)
        else:
            missed += 1
    if DETECTION_HOOK is not None:
        found = DETECTION_HOOK(env, bridge, "inspection", n, found)
        missed = 0
    if missed:
        _bump(_stats(env, bridge.manager), "inspections_missed", missed)
    for d in found:
        _bump(_stats(env, bridge.manager), "detected_by_inspection")
        _make_known(env, bridge, d, "detected")


def _monitoring_detection(env) -> None:
    prob = float(getattr(env.params, "progressive_monitoring_detect_prob", 0.0) or 0.0)
    if prob <= 0.0:
        return
    for bid in sorted(env.bridges):
        bridge = env.bridges[bid]
        if bridge.monitoring_ends_quarter <= env.quarter or bridge.manager not in env.active_agents:
            continue
        found = [d for d in (env.progressive_defects.get(bid) or [])
                 if d.get("known_since") is None
                 and float(env._keyed_stream(PURPOSE_MONITOR, bid, env.quarter, *_dkey(d)).random()) < prob]
        if DETECTION_HOOK is not None:
            found = DETECTION_HOOK(env, bridge, "monitoring", 0, found)
        for d in found:
            _bump(_stats(env, bridge.manager), "detected_by_monitoring")
            _make_known(env, bridge, d, "detected")


# ---------------------------------------------------------------------------
# Arresto
# ---------------------------------------------------------------------------
def arrest_reference(env) -> Tuple[Optional[str], Dict[str, Any]]:
    """Intervento di riferimento per il costo dell'arresto
    (progressive_arrest_reference_intervention): dal catalogo del gestore o, se lo
    scenario lo ha tolto, dal catalogo completo."""
    name = str(getattr(env.params, "progressive_arrest_reference_intervention", "") or "")
    op = (env.maint_ops or {}).get(name) or (getattr(env, "_authority_catalogue", None) or {}).get(name)
    return (name, op) if op else (None, {})


def arrest_op(env) -> Dict[str, Any]:
    """Voce di catalogo dell'azione di arresto. Il costo esposto e' indicativo (rapporto
    del tipo idraulico): quello pagato dipende dai difetti noti del ponte (arrest_cost)."""
    p = env.params
    name, ref = arrest_reference(env)
    ratio = float(p.progressive_hydraulic_arrest_cost_ratio)
    op: Dict[str, Any] = {"type": ARREST_TYPE,
                          "duration_quarters": int(p.progressive_arrest_duration_quarters),
                          "revenue_impact": 0.0, "reference_intervention": name, "cost_ratio": ratio}
    for key in ("cost", "cost_m2", "co2", "co2_m2"):
        value = ref.get(key)
        if isinstance(value, (int, float)) and value > 0:
            op[key] = ratio * float(value)
    return op


def arrest_cost(env, bridge) -> float:
    """Costo dell'arresto dei difetti NOTI del ponte: somma, per difetto, di
    rapporto del tipo x costo dell'intervento di riferimento su quel ponte."""
    from infralab.core import environment as _envmod
    _name, ref = arrest_reference(env)
    base = _envmod._calculate_cost(ref, bridge.area) if ref else 0.0
    specs = _spec_by_name(env.params)
    ratios = [float((specs.get(d["type"]) or {}).get("arrest_cost_ratio", 0.0)) for d in known_defects(env, bridge)]
    return float(base * sum(ratios))


def _remove(env, bridge, defect: Dict[str, Any]) -> None:
    defs = env.progressive_defects.get(bridge.id) or []
    keep = [d for d in defs if d is not defect]
    if keep:
        env.progressive_defects[bridge.id] = keep
    else:
        env.progressive_defects.pop(bridge.id, None)


def arrest_known(env, bridge, how: str) -> List[str]:
    """Arresta i difetti noti del ponte (how: "action" | "floor"). Restituisce i tipi."""
    st = _stats(env, bridge.manager)
    types = []
    for d in known_defects(env, bridge):
        _remove(env, bridge, d)
        _bump(st, "arrested")
        _bump(st, "arrested_by_floor" if how == "floor" else "arrested_by_action")
        _bump(st, f"arrested_{d['type']}")
        types.append(d["type"])
    return types


def intervention_arrests(env, bridge, op: Dict[str, Any], defect: Dict[str, Any]) -> bool:
    """L'intervento ripristina i componenti bersaglio del difetto? Guadagno previsto
    >= progressive_arrest_min_gain su ogni componente bersaglio presente nel ponte."""
    spec = _spec_by_name(env.params).get(defect["type"])
    if spec is None:
        return False
    targets = [t for t in spec["targets"] if t in bridge.components]
    if not targets:
        return False
    gains = op.get("gains") or {}
    need = float(getattr(env.params, "progressive_arrest_min_gain", 0.3))
    return all(float(gains.get(t, 0.0) or 0.0) >= need for t in targets)


def on_intervention_started(env, bridge, op: Dict[str, Any]) -> None:
    """Un intervento che ripristina i componenti bersaglio ferma anche i difetti in
    sviluppo, noti o no, dall'avvio dei lavori. Quelli non noti restano ignoti al
    gestore (nessuna traccia nello storico)."""
    defs = (getattr(env, "progressive_defects", None) or {}).get(bridge.id)
    if not defs:
        return
    st = _stats(env, bridge.manager)
    stopped_known = []
    for d in list(defs):
        if not intervention_arrests(env, bridge, op, d):
            continue
        _remove(env, bridge, d)
        _bump(st, "arrested")
        _bump(st, f"arrested_{d['type']}")
        if d.get("known_since") is None:
            _bump(st, "arrested_before_detection")
        else:
            _bump(st, "arrested_by_intervention")
            stopped_known.append(d["type"])
    if stopped_known:
        env._record_action_event(bridge, "difetto_progressivo", "arrested",
                                 {"event_types": stopped_known, "source": "intervention"})


def on_failure(env, bridge) -> None:
    """Collasso del ponte (qualunque causa): l'opera e' ricostruita, i difetti in
    sviluppo sono cancellati."""
    defs = (getattr(env, "progressive_defects", None) or {}).pop(bridge.id, None)
    if defs:
        _bump(_stats(env, bridge.manager), "removed_by_failure", len(defs))


def on_resolved(env, bridge, entries: List[Dict[str, Any]], source: Optional[str]) -> None:
    """Contabilita' dei danni lasciati da una maturazione usciti dalla lista dei
    pendenti (chiamata da sudden_damage._resolve / on_failure)."""
    if not entries:
        return
    st = _stats(env, bridge.manager)
    key = {"inspection": "damage_detected_by_inspection", "monitoring": "damage_detected_by_monitoring",
           "repair": "damage_repaired_before_detection", "failure": "damage_undetected_at_failure"
           }.get(source, "damage_revealed")
    _bump(st, key, len(entries))


# ---------------------------------------------------------------------------
# Maturazione
# ---------------------------------------------------------------------------
def _end_quarter(env, bridge, defect: Dict[str, Any], index: int) -> int:
    end = defect.get("end_quarter")
    if end is None:
        # stato importato senza data di fine (es. fotografia ripulita): estratta ora
        spec = _spec_by_name(env.params).get(defect["type"])
        rng = env._keyed_stream(PURPOSE_REDRAW, bridge.id, env.quarter, index)
        end = defect["end_quarter"] = int(env.quarter) + (draw_duration(spec, rng) if spec else 1)
    return int(end)


def _mature(env) -> None:
    q = int(env.quarter)
    for bid in sorted(env.progressive_defects):
        bridge = env.bridges.get(bid)
        if bridge is None or bridge.manager not in env.active_agents:
            continue
        due = [d for i, d in enumerate(list(env.progressive_defects.get(bid) or []))
               if _end_quarter(env, bridge, d, i) <= q]
        for d in due:
            # un crollo in questo stesso trimestre cancella gli altri difetti del ponte
            if any(x is d for x in env.progressive_defects.get(bid) or []):
                _apply_maturation(env, bridge, d)


def _apply_maturation(env, bridge, defect: Dict[str, Any]) -> None:
    """Fine della finestra senza arresto. Ordine delle estrazioni fisso: crollo,
    bersaglio, intensita' per bersaglio."""
    _remove(env, bridge, defect)
    spec = _spec_by_name(env.params).get(defect["type"])
    if spec is None:
        return
    known = defect.get("known_since") is not None
    st = _stats(env, bridge.manager)
    _bump(st, "matured")
    _bump(st, f"matured_{spec['name']}")
    if known:
        _bump(st, "matured_known")
    rng = env._keyed_stream(PURPOSE_OUTCOME, bridge.id, *_dkey(defect))
    if float(rng.random()) < float(spec["p_collapse"]):
        _bump(st, "collapses")
        _bump(st, f"collapses_{spec['name']}")
        env._progressive_collapse(bridge, defect, spec)
        return
    targets = [t for t in spec["targets"] if t in bridge.components]
    if not targets:
        return
    hit = _sd.pick_hits(bridge, targets, spec["hits"], rng)
    exponent = env.params.deterioration_exponent
    losses = {name: _sd._damage_hidden(bridge.components[name], _sd.draw_drop(spec, rng), exponent) for name in hit}
    _bump(st, "condition_loss_total", float(sum(losses.values())))
    pending = env.sudden_pending.setdefault(bridge.id, [])
    for name in hit:
        pending.append({"type": PENDING_PREFIX + spec["name"], "quarter": int(env.quarter), "component": name,
                        "loss": round(float(losses[name]), 6)})
    if known:
        # il gestore seguiva il difetto: l'esito e' visibile (osservazione come un'L1)
        env._record_action_event(bridge, "difetto_progressivo", "matured", {"event_type": spec["name"]})
        _sd.reveal(env, bridge, hit, source="progressive", event_type=PENDING_PREFIX + spec["name"])
    env.refresh_bridge_reliability(bridge)


def step(env) -> None:
    """Passo trimestrale del canale (dopo le azioni e gli altri pericoli): rilevazione
    da monitoraggio, maturazione dei difetti a fine finestra, nuovi inneschi."""
    if not enabled(env.params):
        return
    _ensure(env)
    for agent in env.active_agents:
        _stats(env, agent)["initiated_quarter"] = 0
    from infralab.core.hazards import regional_events as _regional
    if not _sd.channel_active(env.params) and not _regional.enabled(env.params):
        _sd._monitoring_detection(env)      # danni nascosti lasciati dalle maturazioni
    _monitoring_detection(env)
    _mature(env)
    _initiate(env)


# ---------------------------------------------------------------------------
# Copie dei pianificatori e fotografie ripulite
# ---------------------------------------------------------------------------
def redact_for_owner(defects: Dict[Any, List[Dict[str, Any]]], keep_bridge=None) -> List[List[Any]]:
    """Solo i difetti noti e solo tipo e trimestre di scoperta, nel formato di to_dict
    ([[ponte, [difetti]]]). `keep_bridge(id)`: filtro sui ponti (evidenza del gestore)."""
    out = []
    for bid in sorted(defects or {}, key=int):
        if keep_bridge is not None and not keep_bridge(int(bid)):
            continue
        known = [{"type": d["type"], "known_since": int(d["known_since"])}
                 for d in defects[bid] if d.get("known_since") is not None]
        if known:
            out.append([int(bid), known])
    return out


def reset_belief(clone, rng) -> None:
    """Copia "belief": via i difetti non noti e i contatori veri; per quelli noti la
    data di fine e' riestratta dalla distribuzione del tipo con il generatore del
    pianificatore (il gestore non la conosce), l'inizio e' il trimestre di scoperta."""
    specs = _spec_by_name(clone.params)
    kept: Dict[int, List[Dict[str, Any]]] = {}
    for bid in sorted(getattr(clone, "progressive_defects", None) or {}):
        for d in clone.progressive_defects[bid]:
            if d.get("known_since") is None:
                continue
            spec = specs.get(d["type"])
            duration = draw_duration(spec, rng) if spec else 1
            kept.setdefault(int(bid), []).append(
                {"type": d["type"], "start_quarter": int(d["known_since"]),
                 "end_quarter": int(clone.quarter) + int(duration) - 1,
                 "known_since": int(d["known_since"]), "source": "initiation"})
    clone.progressive_defects = kept
    clone.progressive_stats = ({a: new_stats(clone.params) for a in clone.params.agent_names}
                               if enabled(clone.params) else {})
    clone.collapse_log = []


def load_defects(data) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for bid, defs in (data or []):
        rows = []
        for d in defs:
            known = d.get("known_since")
            rows.append({"type": d["type"],
                         "start_quarter": int(d.get("start_quarter", known if known is not None else 0) or 0),
                         "end_quarter": None if d.get("end_quarter") is None else int(d["end_quarter"]),
                         "known_since": None if known is None else int(known),
                         "source": d.get("source", "initiation")})
        if rows:
            out[int(bid)] = rows
    return out


# ---------------------------------------------------------------------------
# Registro dei collassi: contatori
# ---------------------------------------------------------------------------
def collapse_metrics_enabled(params) -> bool:
    return (enabled(params) or bool(getattr(params, "collapse_log_metrics", False))
            or getattr(params, "collapse_game_over", "none") != "none")


def collapse_causes(params) -> List[str]:
    return ["pf", "hydraulic_direct"] + [f"progressive_{n}" for n in type_names(params)]


def collapse_metrics(env, agent: str) -> Dict[str, float]:
    """Collassi per causa e collassi di ponti noti come critici (vuoto se non richiesto:
    canale spento, collapse_log_metrics=False e collapse_game_over="none")."""
    if not collapse_metrics_enabled(env.params):
        return {}
    rows = [r for r in (getattr(env, "collapse_log", None) or []) if r.get("agent") == agent]
    out = {f"failures_cause_{c}": float(sum(1 for r in rows if r.get("cause") == c))
           for c in collapse_causes(env.params)}
    out["failures_hydraulic_total"] = float(sum(1 for r in rows if r.get("cause") in
                                                ("hydraulic_direct", "progressive_hydraulic")))
    out["failures_known_critical"] = float(sum(1 for r in rows if r.get("known_critical")))
    out["failures_with_known_defect"] = float(sum(1 for r in rows if r.get("known_defects")))
    return out


# ---------------------------------------------------------------------------
# Diagnostica della taratura (anti-sovradimensionamento)
# ---------------------------------------------------------------------------
def expected_hydraulic_collapse_rates(env, horizon_quarters: Optional[int] = None) -> Dict[str, Any]:
    """Tasso ATTESO di crollo idraulico per un gestore che non ispeziona e non arresta
    mai, per ponte-anno: diretto + via difetto latente, da confrontare con l'ancora
    empirica (~1e-4 per ponte-anno; 0.7e-4 - 1.2e-4 sui ponti in acqua).

    Analitico, dai soli parametri e dall'inventario. "stationary": a regime (ogni
    innesco matura). "horizon": su `horizon_quarters` trimestri, dove gli inneschi
    vicini alla fine non maturano (matured_fraction). Non include: gli inneschi da
    evento regionale (in piu' rispetto alla taratura), la quota autoevidente (conta solo
    se qualcuno arresta) e la cancellazione dei difetti al crollo per altra causa."""
    p = env.params
    bridges = [env.bridges[b] for b in sorted(env.bridges)]
    n = len(bridges)
    water = [b for b in bridges if getattr(b, "crosses_water", False)]
    scour = [b for b in water if getattr(b, "scour_critical", False)]
    share = effective_share(p)
    historical = sum(hydraulic_direct_rate_q(p, b) for b in bridges)
    direct = (1.0 - share) * historical
    spec = _builtin_spec(p, "hydraulic") if hydraulic_progressive_active(p) else None
    initiation = sum(initiation_rate_q(p, b, spec) for b in bridges) if spec else 0.0
    latent = initiation * float(spec["p_collapse"]) if spec else 0.0
    frac = matured_fraction(spec, horizon_quarters) if (spec and horizon_quarters) else 1.0

    def per_year(total_q: float, count: int) -> float:
        return 4.0 * total_q / count if count else 0.0

    out: Dict[str, Any] = {
        "bridges": n, "bridges_over_water": len(water), "bridges_scour_critical": len(scour),
        "share": share, "rate_mode": getattr(p, "progressive_hydraulic_rate_mode", None) if spec else None,
        "p_collapse_if_untreated": float(spec["p_collapse"]) if spec else None,
        "horizon_quarters": horizon_quarters, "matured_fraction": frac,
        "initiations_per_bridge_year": per_year(initiation, n),
    }
    for label, count in (("per_bridge_year", n), ("per_water_bridge_year", len(water))):
        out[label] = {
            "historical_direct": per_year(historical, count),
            "direct": per_year(direct, count),
            "latent_stationary": per_year(latent, count),
            "total_stationary": per_year(direct + latent, count),
            "latent_horizon": per_year(latent * frac, count),
            "total_horizon": per_year(direct + latent * frac, count),
        }
    return out


# ---------------------------------------------------------------------------
# Validazione (chiamata da SimParameters.__post_init__)
# ---------------------------------------------------------------------------
def validate_params(params) -> List[str]:
    problems: List[str] = []

    def _num(name, value, lo=0.0, hi=None):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v >= lo and (hi is None or v <= hi)):
            rng = f"[{lo},{hi}]" if hi is not None else f">= {lo}"
            problems.append(f"{name} deve essere un numero finito {rng} (trovato {value!r})")

    def _bool(name):
        if type(getattr(params, name)).__name__ not in ("bool", "bool_"):
            problems.append(f"{name} deve essere un booleano (trovato {getattr(params, name)!r})")

    def _shape(label, dist, hits, targets, check_targets=True):
        if dist not in _sd.DROP_DISTS:
            problems.append(f"{label}: distribuzione del danno {dist!r} non valida ({_sd.DROP_DISTS})")
        if hits not in _sd.HITS:
            problems.append(f"{label}: 'hits' {hits!r} non valido ({_sd.HITS})")
        if check_targets and not _sd.parse_targets(targets):
            problems.append(f"{label}: serve almeno un componente bersaglio")

    for flag in ("progressive_events_enabled", "progressive_hydraulic_enabled", "progressive_geotechnical_enabled",
                 "progressive_floor_trigger", "collapse_game_over_exempt_closed", "collapse_log_metrics"):
        _bool(flag)
    _num("hydraulic_collapse_progressive_share", params.hydraulic_collapse_progressive_share, 0.0, 1.0)
    if params.progressive_hydraulic_rate_mode not in RATE_MODES:
        problems.append(f"progressive_hydraulic_rate_mode {params.progressive_hydraulic_rate_mode!r} "
                        f"non valido ({RATE_MODES})")
    for name in ("progressive_hydraulic_rate_y", "progressive_hydraulic_scour_multiplier",
                 "progressive_geotechnical_rate_y", "progressive_geotechnical_rate_other_y",
                 "progressive_geotechnical_drop_cov", "progressive_arrest_min_gain"):
        _num(name, getattr(params, name))
    for name in ("progressive_hydraulic_regional_trigger_prob", "progressive_geotechnical_exposed_fraction",
                 "progressive_geotechnical_drop_mean", "progressive_monitoring_detect_prob"):
        _num(name, getattr(params, name), 0.0, 1.0)
    for prefix in BUILTIN_TYPES:
        pre = f"progressive_{prefix}"
        dist = getattr(params, f"{pre}_duration_dist")
        if dist not in DURATION_DISTS:
            problems.append(f"{pre}_duration_dist {dist!r} non valida ({DURATION_DISTS})")
        _num(f"{pre}_duration_mean_q", getattr(params, f"{pre}_duration_mean_q"), 1.0)
        _num(f"{pre}_duration_cov", getattr(params, f"{pre}_duration_cov"))
        for key in ("detect_factor_l1", "detect_factor_detailed", "self_evident_prob", "p_collapse_if_untreated"):
            _num(f"{pre}_{key}", getattr(params, f"{pre}_{key}"), 0.0, 1.0)
        _num(f"{pre}_arrest_cost_ratio", getattr(params, f"{pre}_arrest_cost_ratio"))
    _shape("progressive_geotechnical", params.progressive_geotechnical_drop_dist,
           params.progressive_geotechnical_hits, params.progressive_geotechnical_targets)
    for name in ("progressive_arrest_duration_quarters", "progressive_detailed_inspection_min_level"):
        v = getattr(params, name)     # intero, anche scritto come 1.0 (form numerici di UI e API)
        try:
            ok = not isinstance(v, bool) and float(v).is_integer() and float(v) >= 0.0
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            problems.append(f"{name} deve essere un intero >= 0 (trovato {v!r})")
    if not isinstance(params.progressive_arrest_reference_intervention, str) \
            or not params.progressive_arrest_reference_intervention:
        problems.append("progressive_arrest_reference_intervention deve essere il nome di un intervento del catalogo")
    if params.progressive_detection_pod_mode not in POD_MODES:
        problems.append(f"progressive_detection_pod_mode {params.progressive_detection_pod_mode!r} "
                        f"non valido ({POD_MODES})")
    if params.collapse_game_over not in GAME_OVER_MODES:
        problems.append(f"collapse_game_over {params.collapse_game_over!r} non valido ({GAME_OVER_MODES})")
    try:
        calibrated_without_p = (bool(params.progressive_events_enabled) and bool(params.progressive_hydraulic_enabled)
                                and params.progressive_hydraulic_rate_mode == "calibrated"
                                and float(params.hydraulic_collapse_progressive_share) > 0.0
                                and float(params.progressive_hydraulic_p_collapse_if_untreated) <= 0.0)
    except (TypeError, ValueError):
        calibrated_without_p = False
    if calibrated_without_p:
        problems.append("progressive_hydraulic_rate_mode='calibrated' con quota > 0 richiede "
                        "progressive_hydraulic_p_collapse_if_untreated > 0 (altrimenti i crolli tolti al canale "
                        "diretto non sarebbero sostituiti)")

    events = params.progressive_custom_events
    if not isinstance(events, list):
        problems.append("progressive_custom_events deve essere una lista di dizionari")
        return problems
    seen = set()
    for i, ev in enumerate(events):
        label = f"progressive_custom_events[{i}]"
        if not isinstance(ev, dict):
            problems.append(f"{label} deve essere un dizionario")
            continue
        unknown = sorted(k for k in ev if k not in CUSTOM_EVENT_KEYS)
        if unknown:
            problems.append(f"{label}: chiavi non riconosciute {unknown} (ammesse {list(CUSTOM_EVENT_KEYS)})")
        name = ev.get("name")
        if not name or not isinstance(name, str):
            problems.append(f"{label}: serve 'name' (stringa)")
        elif name in seen or name in BUILTIN_TYPES:
            problems.append(f"{label}: nome {name!r} duplicato o riservato")
        seen.add(name)
        if "rate_y" not in ev:
            problems.append(f"{label}: serve 'rate_y' (inneschi per ponte-anno)")
        else:
            _num(f"{label}.rate_y", ev["rate_y"], 0.0, 4.0)
        _shape(label, ev.get("drop_dist", "fixed"), ev.get("hits", "all"), ev.get("targets", "pile,spalle"))
        if ev.get("duration_dist", "geometric") not in DURATION_DISTS:
            problems.append(f"{label}.duration_dist {ev.get('duration_dist')!r} non valida ({DURATION_DISTS})")
        _num(f"{label}.duration_mean_q", ev.get("duration_mean_q", 4.0), 1.0)
        for key, default in (("duration_cov", 1.0), ("drop_cov", 0.0), ("arrest_cost_ratio", 0.3)):
            _num(f"{label}.{key}", ev.get(key, default))
        for key, default in (("drop_mean", 0.2), ("self_evident_prob", 0.0), ("detect_factor_l1", 1.0),
                             ("detect_factor_detailed", 1.0), ("p_collapse_if_untreated", 0.0)):
            _num(f"{label}.{key}", ev.get(key, default), 0.0, 1.0)
        mults = ev.get("multipliers") or {}
        if not isinstance(mults, dict):
            problems.append(f"{label}.multipliers deve essere un dizionario {{attributo: fattore}}")
        else:
            for k, v in mults.items():
                _num(f"{label}.multipliers.{k}", v)
    return problems


# ---------------------------------------------------------------------------
# Metadati per i form (backend /api/config e UI Streamlit): un'unica fonte
# ---------------------------------------------------------------------------
def _type_fields(prefix: str, label: str, group: str) -> List[Dict[str, Any]]:
    pre = f"progressive_{prefix}"
    return [
        {"key": f"{pre}_duration_dist", "label": f"{label}: distribuzione della finestra", "group": group,
         "type": "select", "options": list(DURATION_DISTS)},
        {"key": f"{pre}_duration_mean_q", "label": f"{label}: durata media della finestra (trimestri)",
         "group": group, "type": "number", "min": 1, "max": 400, "step": 1},
        {"key": f"{pre}_duration_cov", "label": f"{label}: CoV della durata (lognormale)", "group": group,
         "type": "number", "min": 0, "max": 5, "step": 0.05},
        {"key": f"{pre}_detect_factor_l1", "label": f"{label}: fattore di rilevazione, ispezione visiva ordinaria",
         "group": group, "type": "number", "min": 0, "max": 1, "step": 0.05},
        {"key": f"{pre}_detect_factor_detailed", "label": f"{label}: fattore di rilevazione, ispezione di dettaglio",
         "group": group, "type": "number", "min": 0, "max": 1, "step": 0.05},
        {"key": f"{pre}_self_evident_prob", "label": f"{label}: quota autoevidente alla nascita", "group": group,
         "type": "number", "min": 0, "max": 1, "step": 0.05},
        {"key": f"{pre}_p_collapse_if_untreated", "label": f"{label}: P(crollo) a fine finestra se non arrestato",
         "group": group, "type": "number", "min": 0, "max": 1, "step": 0.01},
        {"key": f"{pre}_arrest_cost_ratio", "label": f"{label}: costo dell'arresto / costo dell'intervento di riferimento",
         "group": group, "type": "number", "min": 0, "max": 5, "step": 0.05},
    ]


_G = "Eventi progressivi (latenti)"
_G_H = "Eventi progressivi: idraulico (scalzamento)"
_G_G = "Eventi progressivi: geotecnico (versante)"
_G_C = "Collassi: registro e fine partita"
UI_FIELDS: List[Dict[str, Any]] = [
    {"key": "progressive_events_enabled", "label": "Eventi progressivi abilitati", "group": _G, "type": "bool"},
    {"key": "progressive_arrest_reference_intervention",
     "label": "Arresto: intervento di riferimento per il costo", "group": _G, "type": "text"},
    {"key": "progressive_arrest_duration_quarters", "label": "Arresto: durata dei lavori (trimestri)", "group": _G,
     "type": "number", "min": 0, "max": 20, "step": 1},
    {"key": "progressive_arrest_min_gain",
     "label": "Guadagno minimo sui bersagli perche' un intervento arresti il difetto", "group": _G,
     "type": "number", "min": 0, "max": 1, "step": 0.05},
    {"key": "progressive_detailed_inspection_min_level",
     "label": "Livello d'ispezione minimo considerato di dettaglio", "group": _G, "type": "number",
     "min": 0, "max": 4, "step": 1},
    {"key": "progressive_detection_pod_mode", "label": "Rilevazione: uso della POD dell'ispezione", "group": _G,
     "type": "select", "options": list(POD_MODES)},
    {"key": "progressive_monitoring_detect_prob", "label": "Rilevazione/trim. da monitoraggio attivo", "group": _G,
     "type": "number", "min": 0, "max": 1, "step": 0.05},
    {"key": "progressive_floor_trigger", "label": "Difetto noto = trigger della soglia etica", "group": _G,
     "type": "bool"},
    {"key": "progressive_custom_events", "label": "Tipi utente (JSON: lista di eventi progressivi)", "group": _G,
     "type": "json"},
    {"key": "progressive_hydraulic_enabled", "label": "Idraulico progressivo abilitato", "group": _G_H, "type": "bool"},
    {"key": "hydraulic_collapse_progressive_share",
     "label": "Quota dei crolli idraulici diretti riassegnata al canale progressivo", "group": _G_H,
     "type": "number", "min": 0, "max": 1, "step": 0.05},
    {"key": "progressive_hydraulic_rate_mode", "label": "Idraulico: tasso di innesco", "group": _G_H,
     "type": "select", "options": list(RATE_MODES)},
    {"key": "progressive_hydraulic_rate_y", "label": "Idraulico: inneschi per ponte-anno (modo fixed)",
     "group": _G_H, "type": "number", "min": 0, "max": 4, "step": 1e-4},
    {"key": "progressive_hydraulic_scour_multiplier",
     "label": "Idraulico: fattore per fondazione a rischio erosione (modo fixed)", "group": _G_H,
     "type": "number", "min": 0, "max": 100, "step": 0.5},
    *_type_fields("hydraulic", "Idraulico", _G_H),
    {"key": "progressive_hydraulic_regional_trigger_prob",
     "label": "Idraulico: P(innesco) per ponte colpito da un evento regionale", "group": _G_H,
     "type": "number", "min": 0, "max": 1, "step": 0.05},
    {"key": "progressive_geotechnical_enabled", "label": "Geotecnico progressivo abilitato", "group": _G_G,
     "type": "bool"},
    {"key": "progressive_geotechnical_exposed_fraction",
     "label": "Geotecnico: quota di ponti su versante instabile (se lo scenario non lo dice)", "group": _G_G,
     "type": "number", "min": 0, "max": 1, "step": 0.01},
    {"key": "progressive_geotechnical_rate_y", "label": "Geotecnico: inneschi per ponte-anno, ponti esposti",
     "group": _G_G, "type": "number", "min": 0, "max": 4, "step": 1e-4},
    {"key": "progressive_geotechnical_rate_other_y", "label": "Geotecnico: inneschi per ponte-anno, altri ponti",
     "group": _G_G, "type": "number", "min": 0, "max": 4, "step": 1e-4},
    *_type_fields("geotechnical", "Geotecnico", _G_G),
    {"key": "progressive_geotechnical_targets", "label": "Geotecnico: componenti bersaglio", "group": _G_G,
     "type": "text"},
    {"key": "progressive_geotechnical_hits", "label": "Geotecnico: bersagli colpiti", "group": _G_G,
     "type": "select", "options": list(_sd.HITS)},
    {"key": "progressive_geotechnical_drop_dist", "label": "Geotecnico: distribuzione della perdita", "group": _G_G,
     "type": "select", "options": list(_sd.DROP_DISTS)},
    {"key": "progressive_geotechnical_drop_mean", "label": "Geotecnico: perdita di condizione media", "group": _G_G,
     "type": "number", "min": 0, "max": 1, "step": 0.01},
    {"key": "progressive_geotechnical_drop_cov", "label": "Geotecnico: CoV della perdita", "group": _G_G,
     "type": "number", "min": 0, "max": 3, "step": 0.05},
    {"key": "collapse_game_over", "label": "Fine partita al collasso", "group": _G_C, "type": "select",
     "options": list(GAME_OVER_MODES)},
    {"key": "collapse_game_over_exempt_closed",
     "label": "Nessuna fine partita se il ponte crollato era chiuso al traffico (known_critical)", "group": _G_C,
     "type": "bool"},
    {"key": "collapse_log_metrics", "label": "Contatori dei collassi per causa nelle uscite", "group": _G_C,
     "type": "bool"},
]
