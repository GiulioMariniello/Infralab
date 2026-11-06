"""Canale di danno locale improvviso (opt-in; con i default non fa nulla).

Perche' esiste. Con il solo degrado lento e monotono un'ispezione piu' frequente non
puo' cambiare gli esiti: fra un'ispezione e l'altra non compare nulla di nuovo. I crolli
reali sono invece dominati da cause improvvise: idraulica (piena ed erosione al fondo)
54.9% del campione nazionale americano e 80.5% di quello italiano, urto 12.9%,
sovraccarico 11.3%, degrado 8.8% (vedi docs/CHANGELOG_MODEL.md e le fonti citate li').

Che cosa fa. Ogni trimestre, per ogni ponte e per ogni tipo di evento attivo, un
sorteggio da un flusso casuale DEDICATO a chiave (seme, tipo, ponte, trimestre) decide
se l'evento avviene, con un tasso che dipende dall'esposizione del ponte. L'evento
riduce la condizione VERA dei componenti bersaglio di una quantita' estratta da una
distribuzione configurabile. Il danno non e' visibile al gestore ne' alla legge finche'
un'ispezione (o il monitoraggio, se attivo) non lo rileva: l'osservazione del
componente resta quella dell'ultima ispezione. Una frazione configurabile di eventi e'
"autodenunciante" (per esempio un urto visibile): il danno diventa noto subito, con
un'osservazione come quella di un'ispezione L1. Il danno accelera la perdita di
affidabilita' attraverso il legame condizione->capacita' gia' presente; il collasso
resta campionato dalla Pf come prima.

Numeri casuali comuni. I flussi sono sempre a chiave, indipendenti da rng_streams e da
cio' che e' stato estratto prima: accendere un tipo di evento non sposta ne' l'RNG
condiviso ne' i flussi degli altri processi, e due strategie sullo stesso seme
affrontano gli stessi eventi.

Tipi disponibili:
- "hydraulic": il canale idraulico esistente (environment._sample_hydraulic_events)
  quando hydraulic_legacy_mode=False; esposizione e tassi restano quelli del canale;
- "impact": urto di veicolo su impalcato o pile; tasso crescente con ADT e quota di
  mezzi pesanti, moltiplicatore per i ponti che scavalcano una strada (item NBI 042B);
- "vessel": urto di natante sulle pile, solo per ponti su via navigabile (item NBI 038);
- "fatigue": danno locale di tipo fatica/frattura, raro, crescente con i mezzi pesanti;
- tipi utente: lista `sudden_custom_events` di dizionari (vedi CUSTOM_EVENT_KEYS).
"""
import math
from typing import Any, Dict, List, Optional

# Scopi dei flussi a chiave (distinti da tutti quelli gia' usati dal motore).
PURPOSE = {"hydraulic": 4221, "impact": 4222, "vessel": 4223, "fatigue": 4224}
PURPOSE_CUSTOM_BASE = 4300
PURPOSE_REVEAL = 4240       # osservazione del danno autodenunciato / rilevato dal monitoraggio
PURPOSE_MONITOR = 4241      # rilevazione da monitoraggio

BUILTIN_TYPES = ("hydraulic", "impact", "vessel", "fatigue")
DROP_DISTS = ("fixed", "uniform", "lognormal")
HITS = ("one", "all", "worst")
# Item NBI 042B (servizio sotto il ponte): codici con strada sotto.
HIGHWAY_UNDER_CODES = {"1", "4", "6", "8"}
CUSTOM_EVENT_KEYS = ("name", "rate_q", "targets", "hits", "drop_dist", "drop_mean", "drop_cov",
                     "announce_prob", "requires", "multipliers", "adt_exponent", "truck_weight",
                     "condition_max", "condition_exponent")


# ---------------------------------------------------------------------------
# Specifiche dei tipi di evento
# ---------------------------------------------------------------------------
def parse_targets(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    return [str(t).strip() for t in value if str(t).strip()]


def _spec(params, prefix: str, name: str, mean_attr: Optional[str] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "purpose": PURPOSE[name],
        "targets": parse_targets(getattr(params, f"{prefix}_damage_targets")),
        "hits": getattr(params, f"{prefix}_damage_hits"),
        "drop_dist": getattr(params, f"{prefix}_damage_drop_dist"),
        "drop_mean": float(getattr(params, mean_attr or f"{prefix}_damage_drop_mean")),
        "drop_cov": float(getattr(params, f"{prefix}_damage_drop_cov")),
        "announce_prob": float(getattr(params, f"{prefix}_damage_announce_prob")),
    }


def hydraulic_spec(params) -> Dict[str, Any]:
    """Danno idraulico come tipo del canale (hydraulic_legacy_mode=False). L'intensita'
    media resta hydraulic_damage_condition_drop del canale esistente."""
    return _spec(params, "hydraulic", "hydraulic", mean_attr="hydraulic_damage_condition_drop")


def custom_purpose(name: str) -> int:
    return PURPOSE_CUSTOM_BASE + sum((i + 1) * ord(ch) for i, ch in enumerate(str(name))) % 100000


def active_specs(params) -> List[Dict[str, Any]]:
    """Tipi campionati da sample_events (l'idraulico e' campionato dal suo canale)."""
    specs: List[Dict[str, Any]] = []
    if getattr(params, "impact_hazard_enabled", False):
        s = _spec(params, "impact", "impact")
        s["rate"] = _impact_rate
        specs.append(s)
    if getattr(params, "vessel_impact_enabled", False):
        s = _spec(params, "vessel", "vessel")
        s["rate"] = _vessel_rate
        specs.append(s)
    if getattr(params, "fatigue_hazard_enabled", False):
        s = _spec(params, "fatigue", "fatigue")
        s["rate"] = _fatigue_rate
        specs.append(s)
    for ev in (getattr(params, "sudden_custom_events", None) or []):
        name = str(ev["name"])
        specs.append({
            "name": name,
            "purpose": custom_purpose(name),
            "targets": parse_targets(ev.get("targets", "impalcato")),
            "hits": ev.get("hits", "one"),
            "drop_dist": ev.get("drop_dist", "fixed"),
            "drop_mean": float(ev.get("drop_mean", 0.1)),
            "drop_cov": float(ev.get("drop_cov", 0.0)),
            "announce_prob": float(ev.get("announce_prob", 0.0)),
            "rate": _custom_rate,
            "custom": ev,
        })
    return specs


def channel_active(params) -> bool:
    """True se almeno un tipo del canale e' attivo (idraulico solo fuori dal modo storico)."""
    hydraulic_new = (bool(getattr(params, "hydraulic_hazard_enabled", False))
                     and not bool(getattr(params, "hydraulic_legacy_mode", True)))
    return bool(hydraulic_new or getattr(params, "impact_hazard_enabled", False)
                or getattr(params, "vessel_impact_enabled", False)
                or getattr(params, "fatigue_hazard_enabled", False)
                or (getattr(params, "sudden_custom_events", None) or []))


def type_names(params) -> List[str]:
    names = [s["name"] for s in active_specs(params)]
    if bool(getattr(params, "hydraulic_hazard_enabled", False)) and not bool(getattr(params, "hydraulic_legacy_mode", True)):
        names.insert(0, "hydraulic")
    return names


# ---------------------------------------------------------------------------
# Regole di esposizione
# ---------------------------------------------------------------------------
def exposure_attr(bridge, name: str) -> Any:
    """Attributo di esposizione del ponte. "road_under" e' derivato dall'item NBI 042B."""
    if name == "road_under":
        code = str(getattr(bridge, "service_under", "") or "").strip()
        return code[:1] in HIGHWAY_UNDER_CODES if code else False
    return getattr(bridge, name, None)


def _adt_factor(env, bridge, exponent: float) -> float:
    if exponent == 0.0:
        return 1.0
    ref = float(getattr(env.params, "traffic_reference_adt", 10000) or 10000)
    adt = max(0.0, float(getattr(bridge, "adt", 0) or 0))
    if adt <= 0.0 or ref <= 0.0:
        return 0.0
    return (adt / ref) ** exponent


def _truck(bridge) -> float:
    return max(0.0, min(1.0, float(getattr(bridge, "truck_fraction", 0.0) or 0.0)))


def _impact_rate(env, bridge, spec) -> float:
    p = env.params
    rate = float(p.impact_damage_rate_q) * _adt_factor(env, bridge, float(p.impact_adt_exponent))
    rate *= 1.0 + float(p.impact_truck_weight) * _truck(bridge)
    if exposure_attr(bridge, "road_under"):
        rate *= float(p.impact_road_under_multiplier)
    return rate


def _vessel_rate(env, bridge, spec) -> float:
    if not bool(getattr(bridge, "navigable_waterway", False)):
        return 0.0
    return float(env.params.vessel_impact_rate_q)


def _fatigue_rate(env, bridge, spec) -> float:
    p = env.params
    rate = float(p.fatigue_damage_rate_q) * _adt_factor(env, bridge, float(p.fatigue_adt_exponent))
    return rate * (1.0 + float(p.fatigue_truck_weight) * _truck(bridge))


def _custom_rate(env, bridge, spec) -> float:
    ev = spec["custom"]
    for attr in parse_targets(ev.get("requires")):
        if not exposure_attr(bridge, attr):
            return 0.0
    rate = float(ev.get("rate_q", 0.0))
    rate *= _adt_factor(env, bridge, float(ev.get("adt_exponent", 0.0)))
    rate *= 1.0 + float(ev.get("truck_weight", 0.0)) * _truck(bridge)
    for attr, mult in (ev.get("multipliers") or {}).items():
        if exposure_attr(bridge, attr):
            rate *= float(mult)
    # Dipendenza dalla condizione VERA dei bersagli (cedimento locale di un elemento
    # gia' degradato). Senza le due chiavi il tasso resta quello storico.
    cmax, cexp = ev.get("condition_max"), float(ev.get("condition_exponent", 0.0) or 0.0)
    if cmax is not None or cexp:
        conds = [float(bridge.components[t].current_condition)
                 for t in spec["targets"] if t in bridge.components]
        if not conds:
            return 0.0
        worst = min(conds)
        if cmax is not None and worst > float(cmax):
            return 0.0
        if cexp:
            rate *= max(0.0, 1.0 - worst) ** cexp
    return rate


# ---------------------------------------------------------------------------
# Validazione (chiamata da SimParameters.__post_init__)
# ---------------------------------------------------------------------------
def validate_params(params) -> List[str]:
    problems: List[str] = []

    def _prob(name, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not 0.0 <= v <= 1.0:
            problems.append(f"{name} deve stare in [0,1] (trovato {value!r})")

    def _nonneg(name, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and v >= 0.0):
            problems.append(f"{name} deve essere un numero finito >= 0 (trovato {value!r})")

    def _shape(label, dist, hits, targets):
        if dist not in DROP_DISTS:
            problems.append(f"{label}: distribuzione del danno {dist!r} non valida ({DROP_DISTS})")
        if hits not in HITS:
            problems.append(f"{label}: 'hits' {hits!r} non valido ({HITS})")
        if not parse_targets(targets):
            problems.append(f"{label}: serve almeno un componente bersaglio")

    for prefix in BUILTIN_TYPES:
        _shape(prefix, getattr(params, f"{prefix}_damage_drop_dist"), getattr(params, f"{prefix}_damage_hits"),
               getattr(params, f"{prefix}_damage_targets"))
        _prob(f"{prefix}_damage_announce_prob", getattr(params, f"{prefix}_damage_announce_prob"))
        _nonneg(f"{prefix}_damage_drop_cov", getattr(params, f"{prefix}_damage_drop_cov"))
    for name in ("impact_damage_drop_mean", "vessel_damage_drop_mean", "fatigue_damage_drop_mean",
                 "hydraulic_damage_condition_drop"):
        _prob(name, getattr(params, name))
    for name in ("impact_damage_rate_q", "vessel_impact_rate_q", "fatigue_damage_rate_q",
                 "impact_truck_weight", "impact_road_under_multiplier", "fatigue_truck_weight",
                 "impact_adt_exponent", "fatigue_adt_exponent"):
        _nonneg(name, getattr(params, name))
    _prob("sudden_damage_monitoring_detect_prob", params.sudden_damage_monitoring_detect_prob)
    _nonneg("sudden_damage_reveal_sigma", params.sudden_damage_reveal_sigma)
    for flag in ("hydraulic_legacy_mode", "impact_hazard_enabled", "vessel_impact_enabled", "fatigue_hazard_enabled"):
        if type(getattr(params, flag)).__name__ not in ("bool", "bool_"):
            problems.append(f"{flag} deve essere un booleano (trovato {getattr(params, flag)!r})")

    events = params.sudden_custom_events
    if not isinstance(events, list):
        problems.append("sudden_custom_events deve essere una lista di dizionari")
        return problems
    seen = set()
    for i, ev in enumerate(events):
        label = f"sudden_custom_events[{i}]"
        if not isinstance(ev, dict):
            problems.append(f"{label} deve essere un dizionario")
            continue
        unknown = sorted(k for k in ev if k not in CUSTOM_EVENT_KEYS)
        if unknown:
            problems.append(f"{label}: chiavi non riconosciute {unknown} (ammesse {list(CUSTOM_EVENT_KEYS)})")
        name = ev.get("name")
        if not name or not isinstance(name, str):
            problems.append(f"{label}: serve 'name' (stringa)")
        elif name in seen or name in BUILTIN_TYPES or name == "regional":
            problems.append(f"{label}: nome {name!r} duplicato o riservato")
        seen.add(name)
        if "rate_q" not in ev:
            problems.append(f"{label}: serve 'rate_q'")
        else:
            _prob(f"{label}.rate_q", ev["rate_q"])
        _shape(label, ev.get("drop_dist", "fixed"), ev.get("hits", "one"), ev.get("targets", "impalcato"))
        _prob(f"{label}.drop_mean", ev.get("drop_mean", 0.1))
        _nonneg(f"{label}.drop_cov", ev.get("drop_cov", 0.0))
        _prob(f"{label}.announce_prob", ev.get("announce_prob", 0.0))
        _nonneg(f"{label}.adt_exponent", ev.get("adt_exponent", 0.0))
        _nonneg(f"{label}.truck_weight", ev.get("truck_weight", 0.0))
        if ev.get("condition_max") is not None:
            _prob(f"{label}.condition_max", ev["condition_max"])
        _nonneg(f"{label}.condition_exponent", ev.get("condition_exponent", 0.0))
        mults = ev.get("multipliers") or {}
        if not isinstance(mults, dict):
            problems.append(f"{label}.multipliers deve essere un dizionario {{attributo: fattore}}")
        else:
            for k, v in mults.items():
                _nonneg(f"{label}.multipliers.{k}", v)
    return problems


# ---------------------------------------------------------------------------
# Contabilita'
# ---------------------------------------------------------------------------
# events_total, events_quarter, events_<tipo>: eventi avvenuti (uno per sorteggio
# riuscito). Gli altri contatori sono per COMPONENTE danneggiato (un evento con
# hits="all" ne colpisce piu' d'uno): announced = danni resi noti dall'evento stesso;
# detected_by_other_announcement = danni nascosti piu' vecchi scoperti con l'autodenuncia
# di un evento successivo sullo stesso componente; detected_by_inspection /
# detected_by_monitoring; repaired_before_detection = cancellati da un intervento sul
# componente prima di essere scoperti; undetected_at_failure = ancora nascosti al
# collasso del ponte (failures_with_undetected_damage conta i collassi).
STAT_KEYS = ("events_total", "events_quarter", "announced", "detected_by_other_announcement",
             "detected_by_inspection",
             "detected_by_monitoring", "repaired_before_detection", "undetected_at_failure",
             "failures_with_undetected_damage", "condition_loss_total")


def new_stats(params) -> Dict[str, float]:
    st: Dict[str, float] = {k: 0 for k in STAT_KEYS}
    st["condition_loss_total"] = 0.0
    for name in type_names(params):
        st[f"events_{name}"] = 0
    return st


def _stats(env, agent: str) -> Dict[str, float]:
    st = env.sudden_stats.get(agent)
    if st is None:
        st = env.sudden_stats[agent] = new_stats(env.params)
    return st


def metrics(env, agent: str) -> Dict[str, float]:
    """Metriche per l'output del laboratorio (vuoto se il canale e' spento)."""
    st = (getattr(env, "sudden_stats", None) or {}).get(agent)
    if st is None:
        return {}
    pending = sum(1 for bid, evs in (getattr(env, "sudden_pending", None) or {}).items()
                  if bid in env.bridges and env.bridges[bid].manager == agent
                  for e in evs if not _other_channel(e))
    out = {f"sudden_{k}": float(v) for k, v in st.items()}
    out["sudden_pending_undetected"] = float(pending)
    return out


# ---------------------------------------------------------------------------
# Dinamica
# ---------------------------------------------------------------------------
def _other_channel(entry: Dict[str, Any]) -> bool:
    """Danno pendente di un altro canale che usa la stessa lista: eventi regionali
    (tipo "regional") e maturazioni degli eventi progressivi (tipo "progressive:<tipo>")."""
    t = entry.get("type")
    return t == "regional" or (isinstance(t, str) and t.startswith("progressive:"))


def draw_drop(spec: Dict[str, Any], rng) -> float:
    mean = max(0.0, float(spec["drop_mean"]))
    cov = max(0.0, float(spec["drop_cov"]))
    dist = spec["drop_dist"]
    if dist == "fixed" or mean <= 0.0 or cov <= 0.0:
        drop = mean
    elif dist == "uniform":
        half = math.sqrt(3.0) * cov * mean
        drop = float(rng.uniform(max(0.0, mean - half), mean + half))
    else:   # lognormal con media e CoV assegnati
        s2 = math.log(1.0 + cov * cov)
        drop = float(rng.lognormal(math.log(mean) - 0.5 * s2, math.sqrt(s2)))
    return max(0.0, min(1.0, drop))


def _damage_hidden(comp, drop: float, exponent: float) -> float:
    """Riduce la condizione vera. L'incertezza epistemica e' conoscenza del gestore: un
    danno che il gestore non ha visto non la cambia (apply_damage la aumenterebbe)."""
    before = float(comp.current_condition)
    eps = comp.epistemic_uncertainty
    comp.apply_damage(drop, exponent)
    comp.epistemic_uncertainty = eps
    return before - float(comp.current_condition)


def pick_hits(bridge, targets: List[str], hits: str, rng) -> List[str]:
    """Componenti colpiti fra i bersagli presenti: "one" = uno a caso (un'estrazione),
    "worst" = quello in condizione vera peggiore (nessuna estrazione), "all" = tutti."""
    if hits == "one":
        return [targets[int(rng.integers(len(targets)))]]
    if hits == "worst":
        return [min(targets, key=lambda t: (float(bridge.components[t].current_condition), t))]
    return list(targets)


def apply_event(env, bridge, spec: Dict[str, Any], rng) -> bool:
    """Applica un evento avvenuto. `rng` e' il flusso dedicato dell'evento (dopo il
    sorteggio dell'occorrenza). Ordine delle estrazioni fisso: bersaglio, intensita'
    per bersaglio, autodenuncia."""
    targets = [t for t in spec["targets"] if t in bridge.components]
    if not targets:
        return False
    hit = pick_hits(bridge, targets, spec["hits"], rng)
    exponent = env.params.deterioration_exponent
    losses = {}
    for name in hit:
        losses[name] = _damage_hidden(bridge.components[name], draw_drop(spec, rng), exponent)
    announced = bool(rng.random() < float(spec["announce_prob"]))
    st = _stats(env, bridge.manager)
    st["events_total"] += 1
    st["events_quarter"] += 1
    key = f"events_{spec['name']}"
    st[key] = st.get(key, 0) + 1
    st["condition_loss_total"] += float(sum(losses.values()))
    pending = env.sudden_pending.setdefault(bridge.id, [])
    for name in hit:
        pending.append({"type": spec["name"], "quarter": int(env.quarter), "component": name,
                        "loss": round(float(losses[name]), 6)})
    if announced:
        reveal(env, bridge, hit, source="announced", event_type=spec["name"])
    env.refresh_bridge_reliability(bridge)
    return True


def reveal(env, bridge, comp_names: List[str], source: str, event_type: Optional[str] = None,
           sigma: Optional[float] = None, own_event: Optional[int] = None) -> None:
    """Il danno diventa noto: osservazione dei componenti colpiti (rumore come un'L1,
    senza POD), pendenze risolte, classe aggiornata; la prossima ispezione dovuta non
    viene mai posticipata."""
    from infralab.core.inspection import inspect_component
    base_sigma = float(env.params.sudden_damage_reveal_sigma if sigma is None else sigma)
    for name in comp_names:
        comp = bridge.components.get(name)
        if comp is None:
            continue
        rng = env._keyed_stream(PURPOSE_REVEAL, bridge.id, env._comp_key(name), env.quarter)
        out = inspect_component(comp, env._component_observation_sigma(comp, base_sigma), rng,
                                bias=env.params.inspection_bias, pod_enabled=False)
        comp.observed_condition_min = out.observed_min
        comp.observed_condition_max = out.observed_max
    resolved = _resolve(env, bridge, comp_names, source=source, own_event=own_event)
    if resolved <= 0:
        pass    # nessun danno del canale improvviso (i regionali li conta il loro canale)
    elif source == "monitoring":
        _stats(env, bridge.manager)["detected_by_monitoring"] += resolved
    elif source in ("regional", "progressive"):
        # danni improvvisi nascosti scoperti con l'autodenuncia di un evento regionale
        # o con l'esito visibile di un difetto progressivo noto
        _stats(env, bridge.manager)["detected_by_other_announcement"] += resolved
    elif source == "announced":
        # i danni dell'evento stesso (announced) e quelli nascosti piu' vecchi sugli
        # stessi componenti, scoperti con l'occasione
        st = _stats(env, bridge.manager)
        own = len(comp_names)
        st["announced"] += min(own, resolved)
        st["detected_by_other_announcement"] += max(0, resolved - own)
    prev_due = bridge.next_inspection_quarter
    env.classify_bridge(bridge)
    if prev_due is not None and bridge.next_inspection_quarter is not None:
        bridge.next_inspection_quarter = min(int(prev_due), int(bridge.next_inspection_quarter))
    detail = {"components": list(comp_names), "source": source}
    if event_type:
        detail["event_type"] = event_type
    env._record_action_event(bridge, "danno_improvviso", source, detail)


def _resolve(env, bridge, comp_names, source: Optional[str] = None, own_event: Optional[int] = None) -> int:
    """Toglie dai pendenti i danni sui componenti indicati. Restituisce quanti erano del
    canale improvviso; quelli degli eventi regionali (tipo "regional", stessa lista)
    sono contati dal loro canale (regional_events.on_resolved)."""
    evs = env.sudden_pending.get(bridge.id)
    if not evs:
        return 0
    names = set(comp_names)
    keep = [e for e in evs if e["component"] not in names]
    removed = [e for e in evs if e["component"] in names]
    if keep:
        env.sudden_pending[bridge.id] = keep
    else:
        env.sudden_pending.pop(bridge.id, None)
    regional = [e for e in removed if e.get("type") == "regional"]
    if regional:
        from infralab.core.hazards import regional_events as _regional
        _regional.on_resolved(env, bridge, regional, source, own_event=own_event)
        if source == "inspection":
            env._record_action_event(bridge, "danno_regionale", "detected",
                                     {"components": sorted({e["component"] for e in regional}),
                                      "source": "inspection"})
    progressive = [e for e in removed if _other_channel(e) and e.get("type") != "regional"]
    if progressive:
        from infralab.core.hazards import progressive_events as _progressive
        _progressive.on_resolved(env, bridge, progressive, source)
        if source == "inspection":
            env._record_action_event(bridge, "danno_progressivo", "detected",
                                     {"components": sorted({e["component"] for e in progressive}),
                                      "source": "inspection"})
    return len(removed) - len(regional) - len(progressive)


def on_inspection(env, bridge, comp_name: str, detected: bool) -> None:
    """Un'ispezione che rileva il componente (POD) scopre i danni improvvisi su di esso."""
    if not detected:
        return
    n = _resolve(env, bridge, [comp_name], source="inspection")
    if n:
        _stats(env, bridge.manager)["detected_by_inspection"] += n
        env._record_action_event(bridge, "danno_improvviso", "detected",
                                 {"components": [comp_name], "source": "inspection"})


def on_repair(env, bridge, comp_names) -> None:
    """Un intervento sul componente ripara anche il danno non ancora scoperto."""
    n = _resolve(env, bridge, list(comp_names), source="repair")
    if n:
        _stats(env, bridge.manager)["repaired_before_detection"] += n


def on_failure(env, bridge) -> None:
    """Collasso: i danni ancora nascosti sono contati e cancellati (opera ricostruita)."""
    evs = env.sudden_pending.pop(bridge.id, None)
    regional = [e for e in (evs or []) if e.get("type") == "regional"]
    if regional:
        from infralab.core.hazards import regional_events as _regional
        _regional.on_resolved(env, bridge, regional, "failure")
    progressive = [e for e in (evs or []) if _other_channel(e) and e.get("type") != "regional"]
    if progressive:
        from infralab.core.hazards import progressive_events as _progressive
        _progressive.on_resolved(env, bridge, progressive, "failure")
    evs = [e for e in (evs or []) if not _other_channel(e)]
    if evs:
        st = _stats(env, bridge.manager)
        st["undetected_at_failure"] += len(evs)
        st["failures_with_undetected_damage"] += 1


def _monitoring_detection(env) -> None:
    p = float(env.params.sudden_damage_monitoring_detect_prob)
    if p <= 0.0 or not env.sudden_pending:
        return
    sigma = float((env.maint_ops.get("monitoring") or {}).get("error_sigma", env.params.sudden_damage_reveal_sigma))
    for bid in sorted(env.sudden_pending):
        bridge = env.bridges.get(bid)
        if bridge is None or bridge.monitoring_ends_quarter <= env.quarter:
            continue
        comps = sorted({e["component"] for e in env.sudden_pending.get(bid, [])})
        found = [c for c in comps
                 if float(env._keyed_stream(PURPOSE_MONITOR, bid, env._comp_key(c), env.quarter).random()) < p]
        if found:
            reveal(env, bridge, found, source="monitoring", sigma=sigma)


def sample_events(env) -> None:
    """Campionamento trimestrale dei tipi attivi (idraulico escluso: vedi il suo canale)."""
    if not channel_active(env.params):
        return
    if getattr(env, "sudden_pending", None) is None:     # stato ricaricato da versioni precedenti
        env.sudden_pending = {}
    if getattr(env, "sudden_stats", None) is None:
        env.sudden_stats = {}
    for agent in env.active_agents:
        _stats(env, agent)["events_quarter"] = 0
    _monitoring_detection(env)
    specs = active_specs(env.params)
    if not specs:
        return
    for spec in specs:
        for bid in sorted(env.bridges):
            bridge = env.bridges[bid]
            if bridge.manager not in env.active_agents:
                continue
            rate = spec["rate"](env, bridge, spec)
            if rate <= 0.0:
                continue
            rng = env._keyed_stream(spec["purpose"], bridge.id, env.quarter)
            if float(rng.random()) < rate:
                apply_event(env, bridge, spec, rng)


# ---------------------------------------------------------------------------
# Metadati per i form (backend /api/config e UI Streamlit): un'unica fonte
# ---------------------------------------------------------------------------
def _type_fields(prefix: str, label: str, group: str, mean_key: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        {"key": f"{prefix}_damage_targets", "label": f"{label}: componenti bersaglio", "group": group, "type": "text"},
        {"key": f"{prefix}_damage_hits", "label": f"{label}: bersagli colpiti", "group": group, "type": "select",
         "options": list(HITS)},
        {"key": f"{prefix}_damage_drop_dist", "label": f"{label}: distribuzione del danno", "group": group,
         "type": "select", "options": list(DROP_DISTS)},
        {"key": mean_key or f"{prefix}_damage_drop_mean", "label": f"{label}: perdita di condizione media",
         "group": group, "type": "number", "min": 0, "max": 1, "step": 0.01},
        {"key": f"{prefix}_damage_drop_cov", "label": f"{label}: CoV della perdita", "group": group,
         "type": "number", "min": 0, "max": 3, "step": 0.05},
        {"key": f"{prefix}_damage_announce_prob", "label": f"{label}: quota autodenunciante", "group": group,
         "type": "number", "min": 0, "max": 1, "step": 0.05},
    ]


_G_H = "Pericolo idraulico"
_G_I = "Urto (veicolo, natante)"
_G_F = "Fatica e tipi utente"
UI_FIELDS: List[Dict[str, Any]] = [
    {"key": "hydraulic_hazard_enabled", "label": "Idraulico abilitato", "group": _G_H, "type": "bool"},
    {"key": "hydraulic_legacy_mode", "label": "Idraulico: modo storico (danno visibile, RNG condiviso)",
     "group": _G_H, "type": "bool"},
    {"key": "hydraulic_collapse_rate_scour_q", "label": "Crollo/trim. fondazione a rischio erosione",
     "group": _G_H, "type": "number", "min": 0, "max": 0.1, "step": 1e-5},
    {"key": "hydraulic_collapse_rate_water_q", "label": "Crollo/trim. altri attraversamenti d'acqua",
     "group": _G_H, "type": "number", "min": 0, "max": 0.1, "step": 1e-6},
    {"key": "hydraulic_damage_rate_multiplier", "label": "Rapporto danno/crollo", "group": _G_H,
     "type": "number", "min": 0, "max": 1000, "step": 1},
    *_type_fields("hydraulic", "Idraulico", _G_H, mean_key="hydraulic_damage_condition_drop"),
    {"key": "impact_hazard_enabled", "label": "Urto di veicolo abilitato", "group": _G_I, "type": "bool"},
    {"key": "impact_damage_rate_q", "label": "Urto: eventi/trim. all'ADT di riferimento", "group": _G_I,
     "type": "number", "min": 0, "max": 1, "step": 1e-5},
    {"key": "impact_adt_exponent", "label": "Urto: esponente dell'ADT", "group": _G_I, "type": "number",
     "min": 0, "max": 3, "step": 0.1},
    {"key": "impact_truck_weight", "label": "Urto: peso della quota mezzi pesanti", "group": _G_I,
     "type": "number", "min": 0, "max": 20, "step": 0.5},
    {"key": "impact_road_under_multiplier", "label": "Urto: fattore se sotto passa una strada", "group": _G_I,
     "type": "number", "min": 0, "max": 20, "step": 0.5},
    *_type_fields("impact", "Urto", _G_I),
    {"key": "vessel_impact_enabled", "label": "Urto di natante abilitato (vie navigabili)", "group": _G_I,
     "type": "bool"},
    {"key": "vessel_impact_rate_q", "label": "Natante: eventi/trim.", "group": _G_I, "type": "number",
     "min": 0, "max": 1, "step": 1e-5},
    *_type_fields("vessel", "Natante", _G_I),
    {"key": "fatigue_hazard_enabled", "label": "Fatica/frattura abilitata", "group": _G_F, "type": "bool"},
    {"key": "fatigue_damage_rate_q", "label": "Fatica: eventi/trim.", "group": _G_F, "type": "number",
     "min": 0, "max": 1, "step": 1e-5},
    {"key": "fatigue_adt_exponent", "label": "Fatica: esponente dell'ADT", "group": _G_F, "type": "number",
     "min": 0, "max": 3, "step": 0.1},
    {"key": "fatigue_truck_weight", "label": "Fatica: peso della quota mezzi pesanti", "group": _G_F,
     "type": "number", "min": 0, "max": 20, "step": 0.5},
    *_type_fields("fatigue", "Fatica", _G_F),
    {"key": "sudden_custom_events", "label": "Tipi utente (JSON: lista di eventi)", "group": _G_F, "type": "json"},
    {"key": "sudden_damage_monitoring_detect_prob", "label": "Rilevazione/trim. da monitoraggio attivo",
     "group": _G_F, "type": "number", "min": 0, "max": 1, "step": 0.05},
    {"key": "sudden_damage_reveal_sigma", "label": "Rumore dell'osservazione del danno noto", "group": _G_F,
     "type": "number", "min": 0, "max": 0.5, "step": 0.01},
]
