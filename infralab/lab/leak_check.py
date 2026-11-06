"""Controllo delle fughe d'informazione: "due mondi, stessa evidenza".

Diagnostica da usare come cancello prima di una campagna di benchmark: per uno
scenario e una configurazione dati, verifica che tutto cio' che dipende dalla sola
conoscenza del gestore sia identico in due mondi che differiscono SOLO per lo stato
nascosto (verita') e hanno la stessa evidenza.

Procedura, per ciascuno degli N trimestri di una traiettoria di riferimento (mondo A,
condotto da una policy semplice che produce evidenza: ispezioni a rotazione e prove di
carico):
1. fotografia del mondo A (A');
2. mondo B = stessa fotografia con lo stato nascosto perturbato (`perturb_hidden_truth`):
   parametri latenti riestratti dalla priori del profilo, vita utile campionata
   variata, condizione vera spostata (intervalli osservati invariati), moltiplicatori
   di processo, rischio della prova e numeri "veri" del referto di prova variati, un
   danno idraulico nascosto per asset (primitiva del motore `_apply_hidden_damage`,
   trimestre ed evento nello storico come nel motore) e, per default, danni locali
   improvvisi nascosti diversi (`perturb_hidden_sudden`: un evento in piu' per asset,
   pendenti con intensita' diversa o tolti), danni regionali nascosti diversi nell'area
   degli eventi pubblici (`perturb_hidden_regional`) ed eventi progressivi diversi
   (`perturb_hidden_progressive`: difetti latenti non rilevati tolti, spostati o
   aggiunti; inizio e fine veri dei difetti noti; contatori veri e registro dei
   collassi), anche a canali spenti. L'evidenza
   (intervalli osservati, storico osservabile, esiti dei referti, eventi pubblici) e'
   la stessa;
3. per ogni gestore si confrontano le uscite che devono dipendere solo dall'evidenza
   (`owner_outputs`): l'osservazione consegnata alla policy (sanitizzata se
   policy_sandbox), la classificazione del regime (grandezze e classe), i trigger degli
   obblighi, il trigger della soglia etica, il piano della prova di carico (alpha,
   passi, componenti provati) per ogni prova del catalogo (e una variante a griglia con
   vincolo di rischio), la copia "belief" del pianificatore a seme fisso e, con il
   sandbox, la fotografia da cui il handle genera le copie (`planner_snapshot`).

Modalita' traiettoria (`--trajectory`, `leak_check_trajectory`): la verita' e'
perturbata UNA volta e i due mondi avanzano insieme con lo stesso piano; le letture
d'ispezione (e il ricavo, legato alla salute vera) di B sono quelle di A, il resto dello
stato del gestore evolve in B: copre le fughe che si accumulano nel tempo. Il confronto
si ferma alla prima evidenza legittimamente diversa che non si puo' forzare (collassi,
declassamenti, esito delle prove di carico): `--policy inspect-only` evita le prove.

Ogni differenza e' una fuga: l'uscita dipende da qualcosa che il gestore non conosce.
Il controllo ha senso con observation_mode="inspected" e senza oracolo (in "full" e
con l'oracolo lo stato vero e' concesso per dichiarazione); in quei casi il rapporto
lo segnala.

Uso:
    python -m infralab.lab.leak_check --scenario TRAIN-01 --quarters 8 \\
        --param observation_mode=inspected --param policy_sandbox=true \\
        --param strict_information=true [--trajectory] [--sudden-hazards] \\
        [--regional-events] [--progressive-events] [--no-sudden-injection] \\
        [--policy inspect|inspect-only|none] [--json out.json]
Codice di uscita: 0 nessuna divergenza, 1 divergenze trovate, 2 errore.
API: `run_leak_check(scenario, params=..., quarters=...)` -> rapporto (dict).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Variante a griglia della prova di carico usata come sonda quando il catalogo non
# fissa un vincolo di rischio (alpha scelto dal motore, non fisso).
PROBE_MAX_FAILURE_PROBABILITY = 1e-3
PROBE_ALPHA_GRID_SIZE = 4
# Modalita' traiettoria con il collasso storico: margine sopra failure_threshold sotto
# cui la perturbazione non porta la condizione vera.
TRAJECTORY_MARGIN = 0.20

# Configurazione di prova del canale di danno locale improvviso (--sudden-hazards):
# tutti i tipi accesi (idraulico fuori dal modo storico) con tassi alti, perche' nel
# breve orizzonte del controllo avvengano eventi veri, autodenunciati e nascosti, e il
# monitoraggio li possa rilevare. Gli override espliciti (--param) prevalgono.
TEST_HAZARD_PARAMS: Dict[str, Any] = {
    "hydraulic_hazard_enabled": True, "hydraulic_legacy_mode": False, "hydraulic_damage_rate_multiplier": 200.0,
    "impact_hazard_enabled": True, "impact_damage_rate_q": 0.05, "impact_adt_exponent": 0.0,
    "vessel_impact_enabled": True, "vessel_impact_rate_q": 0.05,
    "fatigue_hazard_enabled": True, "fatigue_damage_rate_q": 0.05,
    "sudden_damage_monitoring_detect_prob": 0.5,
}
# Configurazione di prova degli eventi estremi regionali (--regional-events): eventi
# frequenti, area larga, tutti i ponti esposti, danno certo al centro, una parte
# autodenunciata, perche' nel breve orizzonte ci siano eventi pubblici e danni nascosti.
TEST_REGIONAL_PARAMS: Dict[str, Any] = {
    "regional_events_enabled": True, "regional_event_rate_y": 2.0, "regional_event_radius_km": 60.0,
    "regional_event_decay_km": 40.0, "regional_event_requires": "", "regional_event_hit_prob": 1.0,
    "regional_event_announce_prob": 0.3,
}

# Configurazione di prova degli eventi progressivi (--progressive-events): inneschi
# frequenti su tutti i ponti (un tipo utente senza requisiti e il geotecnico con tutti i
# ponti esposti; l'idraulico a tasso fisso sui ponti su acqua), finestre brevi, una
# parte autoevidente, rilevazione imperfetta e da monitoraggio, crollo raro: nel breve
# orizzonte ci sono difetti latenti, noti, arrestati e maturati.
TEST_PROGRESSIVE_PARAMS: Dict[str, Any] = {
    "progressive_events_enabled": True,
    "hydraulic_hazard_enabled": True, "hydraulic_legacy_mode": False, "hydraulic_collapse_progressive_share": 0.5,
    "progressive_hydraulic_rate_mode": "fixed", "progressive_hydraulic_rate_y": 0.6,
    "progressive_hydraulic_duration_mean_q": 4.0, "progressive_hydraulic_p_collapse_if_untreated": 0.02,
    "progressive_geotechnical_exposed_fraction": 1.0, "progressive_geotechnical_rate_y": 0.6,
    "progressive_geotechnical_p_collapse_if_untreated": 0.02,
    "progressive_custom_events": [{"name": "prova", "rate_y": 0.8, "duration_dist": "geometric",
                                   "duration_mean_q": 3.0, "self_evident_prob": 0.3, "detect_factor_l1": 0.5,
                                   "p_collapse_if_untreated": 0.02, "drop_mean": 0.1}],
    "progressive_monitoring_detect_prob": 0.5,
}


# ---------------------------------------------------------------------------
# Mondo B: stessa evidenza, verita' diversa
# ---------------------------------------------------------------------------

def perturb_hidden_truth(env, seed: int = 0, damage: Tuple[float, float] = (0.05, 0.20), sudden: bool = True,
                         min_condition: Optional[float] = None):
    """Copia del motore con lo stato NASCOSTO perturbato e la stessa evidenza.

    Non tocca: intervalli osservati, incertezza epistemica, azioni e loro esiti,
    inventario (eta' iniziale, profilo), budget. Aggiunge un danno idraulico nascosto
    per asset con la stessa via del motore (`_apply_hidden_damage`, trimestre, evento
    "danno_idraulico" nello storico: verita' che solo un'osservazione successiva rende
    nota) e varia i numeri del referto di prova calcolati sul vero.

    `sudden=True` (default): anche il canale del danno locale improvviso
    (hazards/sudden_damage.py) e' perturbato, acceso o no: per asset un evento nascosto
    in piu' (tipo attivo o "leak_check", componente bersaglio, perdita estratta) e i
    danni gia' pendenti con intensita' diversa o tolti (`perturb_hidden_sudden`).
    `min_condition`: la perturbazione non porta la condizione vera sotto questo valore
    (se gia' sotto, non la abbassa): serve alla modalita' traiettoria con il modello di
    collasso storico, dove un solo ponte sotto la soglia elimina il gestore."""
    from infralab.core.environment import BridgeEnv, Component
    rng = np.random.default_rng(int(seed))
    world = BridgeEnv.from_dict(env.to_dict(), refresh=False)
    exponent = world.params.deterioration_exponent
    lo_d, hi_d = float(damage[0]), float(damage[1])
    for bid in sorted(world.bridges):
        bridge = world.bridges[bid]
        names = sorted(bridge.components)
        for name in names:
            comp = bridge.components[name]
            for key, value in Component.draw_latent_parameters(comp.profile or {}, rng).items():
                setattr(comp, key, value)
            comp.vita_utile_anni = max(1, int(round(float(comp.vita_utile_anni) * float(rng.uniform(0.8, 1.25)))))
            cond = float(np.clip(float(comp.current_condition) + float(rng.normal(0.0, 0.05)), 1e-3, 1.0))
            if min_condition is not None:
                cond = max(cond, min(float(comp.current_condition), float(min_condition)))
            lo, hi, eps = comp.observed_condition_min, comp.observed_condition_max, comp.epistemic_uncertainty
            comp.current_condition = cond
            comp.apply_maintenance(0.0, exponent)          # eta' efficace coerente con la nuova verita'
            comp.observed_condition_min, comp.observed_condition_max = lo, hi
            comp.epistemic_uncertainty = eps
            comp.last_process_multiplier = float(comp.last_process_multiplier or 1.0) * float(rng.uniform(0.7, 1.4))
            if comp.last_proof_load_risk is not None:
                comp.last_proof_load_risk = float(comp.last_proof_load_risk) * float(rng.uniform(0.5, 2.0))
        if names:
            # stessa via del danno idraulico del motore: danno, trimestre, evento nello storico
            target = names[int(rng.integers(0, len(names)))]
            drop = _capped(bridge.components[target], float(rng.uniform(lo_d, hi_d)), min_condition)
            world._apply_hidden_damage(bridge.components[target], drop)
            bridge.hydraulic_damage_quarter = world.quarter
            world._record_action_event(bridge, "danno_idraulico", "hazard", {"condition_drop": drop})
        report = getattr(bridge, "last_proof_load_report", None)
        if report:
            _perturb_truth_numbers(report, rng)
    if sudden:
        perturb_hidden_sudden(world, rng, damage, min_condition=min_condition)
        perturb_hidden_regional(world, rng, damage, min_condition=min_condition)
        perturb_hidden_progressive(world, rng)
    world.refresh_network_reliability()
    return world


def perturb_hidden_progressive(world, rng) -> None:
    """Eventi progressivi diversi nel mondo B, con la stessa evidenza.

    Evidenza (invariata): i difetti NOTI, con tipo e trimestre di scoperta. Verita'
    (variata, a canale acceso o spento): ogni difetto latente non rilevato e' tolto con
    probabilita' 1/2 o riceve un'altra data di fine; i difetti noti ricevono un altro
    inizio vero e un'altra data di fine; con probabilita' 1/2 per ponte nasce un difetto
    latente in piu' (tipo attivo, o "leak_check" se il canale e' spento); i contatori
    veri del canale e i numeri nascosti del registro dei collassi sono variati."""
    from infralab.core.hazards import progressive_events as pe
    if getattr(world, "progressive_defects", None) is None:
        world.progressive_defects = {}
    names = pe.type_names(world.params)
    q = int(world.quarter)
    for bid in sorted(world.bridges):
        kept = []
        for d in world.progressive_defects.get(bid, []):
            d = dict(d)
            if d.get("known_since") is None:
                if float(rng.random()) < 0.5:
                    continue
                d["end_quarter"] = q + int(rng.integers(1, 12))
            else:
                d["start_quarter"] = max(0, int(d["known_since"]) - int(rng.integers(0, 6)))
                d["end_quarter"] = max(q, int(d.get("end_quarter") or q)) + int(rng.integers(0, 8))
            kept.append(d)
        if float(rng.random()) < 0.5:
            kept.append({"type": names[int(rng.integers(0, len(names)))] if names else "leak_check",
                         "start_quarter": max(0, q - int(rng.integers(0, 4))),
                         "end_quarter": q + int(rng.integers(1, 8)), "known_since": None,
                         "source": "leak_check"})
        if kept:
            world.progressive_defects[bid] = kept
        else:
            world.progressive_defects.pop(bid, None)
    for st in (getattr(world, "progressive_stats", None) or {}).values():
        for key in ("initiated", "matured", "inspections_missed", "arrested_before_detection"):
            st[key] = int(st.get(key, 0)) + int(rng.integers(1, 4))
        st["condition_loss_total"] = float(st.get("condition_loss_total", 0.0)) + float(rng.uniform(0.05, 0.2))
    for rec in (getattr(world, "collapse_log", None) or []):
        rec["hidden_damage_pending"] = int(rec.get("hidden_damage_pending") or 0) + int(rng.integers(1, 3))
        rec["hidden_defects_pending"] = int(rec.get("hidden_defects_pending") or 0) + int(rng.integers(1, 3))
        rec["true_min_condition"] = float(rec.get("true_min_condition") or 0.0) * float(rng.uniform(0.5, 1.0))


def perturb_hidden_regional(world, rng, damage: Tuple[float, float] = (0.05, 0.20),
                            min_condition: Optional[float] = None) -> None:
    """Eventi estremi regionali diversi nel mondo B, con la stessa evidenza.

    L'elenco degli eventi (trimestre, centro, raggio, ponti nell'area) e' pubblico e resta
    identico. E' verita' invece il danno per ponte: per ogni ponte nell'area di un evento
    (o, se non ci sono eventi, per ogni ponte: il canale puo' essere spento) con
    probabilita' 1/2 un danno regionale nascosto in piu' su un componente bersaglio,
    con perdita estratta; i contatori veri del canale (ponti colpiti, componenti, perdita)
    sono variati. Come nel motore: condizione vera ridotta, osservazioni invariate."""
    from infralab.core.hazards import regional_events as rg
    from infralab.core.hazards import sudden_damage as sd
    exponent = world.params.deterioration_exponent
    lo_d, hi_d = float(damage[0]), float(damage[1])
    if getattr(world, "sudden_pending", None) is None:
        world.sudden_pending = {}
    events = getattr(world, "regional_events", None) or []
    in_area = {int(b): int(e["id"]) for e in events for b, _ in e.get("bridges") or []}
    targets_cfg = sd.parse_targets(getattr(world.params, "regional_event_targets", "")) or []
    for bid in sorted(world.bridges):
        if events and bid not in in_area:
            continue
        bridge = world.bridges[bid]
        names = sorted(bridge.components)
        if not names or float(rng.random()) < 0.5:
            continue
        targets = [t for t in targets_cfg if t in bridge.components] or names
        target = targets[int(rng.integers(0, len(targets)))]
        loss = sd._damage_hidden(bridge.components[target],
                                 _capped(bridge.components[target], float(rng.uniform(lo_d, hi_d)), min_condition),
                                 exponent)
        world.sudden_pending.setdefault(bid, []).append(
            {"type": rg.PENDING_TYPE, "quarter": int(world.quarter), "component": target,
             "loss": round(loss, 6), "event": in_area.get(bid, -1)})
    for st in (getattr(world, "regional_stats", None) or {}).values():
        for key in ("bridges_hit", "components_hit"):
            st[key] = int(st.get(key, 0)) + int(rng.integers(1, 4))
        st["condition_loss_total"] = float(st.get("condition_loss_total", 0.0)) + float(rng.uniform(lo_d, hi_d))


def _capped(comp, drop: float, min_condition: Optional[float]) -> float:
    if min_condition is None:
        return drop
    return max(0.0, min(drop, float(comp.current_condition) - float(min_condition)))


def perturb_hidden_sudden(world, rng, damage: Tuple[float, float] = (0.05, 0.20),
                          min_condition: Optional[float] = None) -> None:
    """Danno locale improvviso diverso nel mondo B, con la stessa evidenza.

    Per ogni asset (ordine fisso): i danni pendenti (non rilevati) hanno un'intensita'
    diversa (perdita aggiuntiva sul componente) o, con probabilita' 1/2, non ci sono
    (la lista e' verita': il danno resta nella condizione vera ma non come evento); poi
    un evento nascosto in piu' su un componente bersaglio di un tipo attivo (o su un
    componente qualsiasi, tipo "leak_check", se il canale e' spento). Come nel motore
    (`sudden_damage._damage_hidden`): condizione vera ridotta, intervalli osservati e
    incertezza epistemica invariati, contatori veri del canale aggiornati."""
    from infralab.core.hazards import sudden_damage as sd
    exponent = world.params.deterioration_exponent
    lo_d, hi_d = float(damage[0]), float(damage[1])
    if getattr(world, "sudden_pending", None) is None:
        world.sudden_pending = {}
    specs = sd.active_specs(world.params)
    if "hydraulic" in sd.type_names(world.params):
        specs = [sd.hydraulic_spec(world.params)] + specs
    for bid in sorted(world.bridges):
        bridge = world.bridges[bid]
        names = sorted(bridge.components)
        if not names:
            continue
        kept = []
        for ev in world.sudden_pending.get(bid, []):
            if float(rng.random()) < 0.5:
                continue
            comp = bridge.components.get(ev.get("component"))
            if comp is not None:
                extra = sd._damage_hidden(comp, _capped(comp, float(rng.uniform(lo_d, hi_d)), min_condition),
                                          exponent)
                ev = dict(ev, loss=round(float(ev.get("loss", 0.0)) + extra, 6))
            kept.append(ev)
        spec = specs[int(rng.integers(0, len(specs)))] if specs else None
        targets = [t for t in (spec["targets"] if spec else []) if t in bridge.components] or names
        target = targets[int(rng.integers(0, len(targets)))]
        loss = sd._damage_hidden(bridge.components[target],
                                 _capped(bridge.components[target], float(rng.uniform(lo_d, hi_d)), min_condition),
                                 exponent)
        etype = spec["name"] if spec else "leak_check"
        kept.append({"type": etype, "quarter": int(world.quarter), "component": target, "loss": round(loss, 6)})
        world.sudden_pending[bid] = kept
        st = (getattr(world, "sudden_stats", None) or {}).get(bridge.manager)
        if st is not None:
            st["events_total"] = st.get("events_total", 0) + 1
            st[f"events_{etype}"] = st.get(f"events_{etype}", 0) + 1
            st["condition_loss_total"] = float(st.get("condition_loss_total", 0.0)) + loss


# Campi del referto della prova di carico che sono evidenza (esito osservato): gli altri
# numeri (probabilita' di rottura/scarto, beta calcolati sul vero) sono verita'.
_REPORT_EVIDENCE = ("quarter", "alpha", "step_count", "proof_load_alpha", "passed", "screened")


def _perturb_truth_numbers(obj, rng) -> None:
    items = obj.items() if isinstance(obj, dict) else enumerate(obj) if isinstance(obj, list) else []
    for k, v in list(items):
        if isinstance(v, (dict, list)):
            _perturb_truth_numbers(v, rng)
        elif isinstance(v, float) and k not in _REPORT_EVIDENCE:
            obj[k] = v * float(rng.uniform(0.5, 1.5))


def _fresh_copy(env):
    from infralab.core.environment import BridgeEnv
    world = BridgeEnv.from_dict(env.to_dict(), refresh=False)
    world.refresh_network_reliability()
    return world


# ---------------------------------------------------------------------------
# Uscite che devono dipendere solo dall'evidenza
# ---------------------------------------------------------------------------

def _policy_observation(env, agent: str) -> Dict[str, Any]:
    from infralab.agents.sandbox import _jsonable, sanitized_observation
    if bool(getattr(env.params, "policy_sandbox", False)):
        return sanitized_observation(env, agent)
    return _jsonable(env.get_agent_observations(agent))


def _proof_load_ops(env) -> List[Tuple[str, Dict[str, Any]]]:
    ops = []
    for name in sorted(env.maint_ops):
        op = env.maint_ops[name] or {}
        if op.get("type") != "proof_load":
            continue
        ops.append((name, op))
        if not op.get("max_failure_probability"):
            probe = dict(op, max_failure_probability=PROBE_MAX_FAILURE_PROBABILITY,
                         alpha_grid_size=PROBE_ALPHA_GRID_SIZE)
            ops.append((f"{name}[grid]", probe))
    return ops


def _belief_clone(env, agent: str, seed: int):
    rng = np.random.default_rng(int(seed))
    if bool(getattr(env.params, "policy_sandbox", False)):
        from infralab.agents.sandbox import PlanningHandle
        return PlanningHandle.build(env, agent, seed=int(seed)).belief_clone(salt=1, rng=rng)
    return env.belief_clone(salt=1, rng=rng)


def owner_outputs(env, agent: str, belief_seed: int = 12345,
                  include: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """Uscite osservazione-dipendenti del gestore `agent` (JSON)."""
    from infralab.agents.sandbox import _jsonable
    want = set(include) if include else None

    def on(name):
        return want is None or name in want

    owned = [env.bridges[bid] for bid in sorted(env.bridges) if env.bridges[bid].manager == agent]
    out: Dict[str, Any] = {}
    if on("observation"):
        out["observation"] = _policy_observation(env, agent)
    cls = getattr(env.regime, "classification", None) if env.regime else None
    if on("classification"):
        rows = {}
        for b in owned:
            values = env._classifier_values(b)
            rows[str(b.id)] = {"values": values, "cda": cls.classify(values) if cls is not None else None}
        out["classification"] = rows
    if on("regime_triggers"):
        obligations = list(getattr(env.regime, "obligations", []) or [])
        out["regime_triggers"] = {str(b.id): {env._obligation_key(ob): bool(env._regime_trigger_active(b, ob))
                                              for ob in obligations} for b in owned}
    if on("floor_trigger"):
        out["floor_trigger"] = {str(b.id): bool(env._floor_triggered(b)) for b in owned}
    if on("proof_load_plan"):
        plans = {}
        for b in owned:
            plans[str(b.id)] = {name: env._proof_load_plan(b, op) for name, op in _proof_load_ops(env)}
        out["proof_load_plan"] = plans
    if on("belief_clone"):
        clone = _belief_clone(env, agent, belief_seed)
        data = clone.to_dict()
        data.pop("params", None)
        data["bridges"] = {k: v for k, v in (data.get("bridges") or {}).items()
                           if (v or {}).get("manager") == agent}
        out["belief_clone"] = data
    if on("planner_snapshot") and bool(getattr(env.params, "policy_sandbox", False)):
        # fotografia da cui il PlanningHandle genera le copie (resta nel handle della
        # policy): deve dipendere solo dall'evidenza (RT009: danni improvvisi pendenti)
        from infralab.agents.sandbox import belief_snapshot
        snap = belief_snapshot(env, agent, int(belief_seed))
        snap.pop("params", None)
        out["planner_snapshot"] = snap
    return _jsonable(out)


# ---------------------------------------------------------------------------
# Confronto
# ---------------------------------------------------------------------------

def _same_number(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def diff_json(a: Any, b: Any, path: str = "", rtol: float = 1e-9, atol: float = 1e-12,
              out: Optional[List[Tuple[str, Any, Any]]] = None, limit: int = 200) -> List[Tuple[str, Any, Any]]:
    """Differenze fra due strutture JSON: lista di (percorso, a, b)."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append((p, a.get(k, "<assente>"), b.get(k, "<assente>")))
            else:
                diff_json(a[k], b[k], p, rtol, atol, out, limit)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", rtol, atol, out, limit)
        return out
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        if not _same_number(float(a), float(b), rtol, atol):
            out.append((path, a, b))
        return out
    if a != b:
        out.append((path, a, b))
    return out


def compare_worlds(world_a, world_b, agents: Optional[List[str]] = None, belief_seed: int = 12345,
                   include: Optional[Tuple[str, ...]] = None, limit: int = 50) -> Dict[str, List[Tuple[str, Any, Any]]]:
    """{uscita: differenze} fra i due mondi, per tutti i gestori."""
    agents = agents or list(world_a.params.agent_names)
    result: Dict[str, List[Tuple[str, Any, Any]]] = {}
    for agent in agents:
        oa = owner_outputs(world_a, agent, belief_seed, include)
        ob = owner_outputs(world_b, agent, belief_seed, include)
        for key in oa:
            d = diff_json(oa[key], ob.get(key), limit=limit)
            if d:
                result.setdefault(key, []).extend((f"{agent}:{p}", x, y) for p, x, y in d)
    return result


# ---------------------------------------------------------------------------
# Traiettoria di riferimento
# ---------------------------------------------------------------------------

def _evidence_plan(env, policy: str) -> Dict[str, Dict[str, Any]]:
    """Piano semplice che produce evidenza: ispezione a rotazione (1 asset su 3 per
    trimestre) con l'analisi piu' economica e, ogni 4 trimestri, una prova di carico
    (non con policy="inspect-only")."""
    plans = {a: {"maintenance": {}} for a in env.params.agent_names}
    if policy == "none":
        return plans
    analysis = sorted((float(op.get("cost_m2", 0.0) or 0.0), name) for name, op in env.maint_ops.items()
                      if (op or {}).get("type") == "analysis")
    proof = sorted(name for name, op in env.maint_ops.items() if (op or {}).get("type") == "proof_load")
    if policy == "inspect-only":     # niente prove di carico: il loro esito (superata/scartata)
        proof = []                   # dipende dalla verita' e ferma la modalita' traiettoria
    for agent in env.params.agent_names:
        owned = sorted(b.id for b in env.bridges.values() if b.manager == agent)
        for i, bid in enumerate(owned):
            if proof and env.quarter % 4 == 0 and i == (env.quarter // 4) % max(1, len(owned)):
                plans[agent]["maintenance"][str(bid)] = proof[0]
            elif analysis and (i + env.quarter) % 3 == 0:
                plans[agent]["maintenance"][str(bid)] = analysis[0][1]
    return plans


def leak_check_env(env, quarters: int = 8, policy: str = "inspect", perturb_seed: int = 0,
                   belief_seed: int = 12345, include: Optional[Tuple[str, ...]] = None,
                   limit: int = 50, sudden: bool = True, trajectory: bool = False) -> Dict[str, Any]:
    """Esegue il controllo sul motore `env` (che viene fatto avanzare di `quarters`).

    trajectory=False: a ogni trimestre il mondo B e' una perturbazione nuova del mondo A
    (istante per istante). trajectory=True: vedi `leak_check_trajectory`."""
    if trajectory:
        return leak_check_trajectory(env, quarters=quarters, policy=policy, perturb_seed=perturb_seed,
                                     belief_seed=belief_seed, include=include, limit=limit, sudden=sudden)
    import time
    p = env.params
    warnings = []
    if p.observation_mode != "inspected":
        warnings.append("observation_mode='full': lo stato vero e' concesso per dichiarazione, "
                        "le divergenze dell'osservazione sono attese")
    if bool(getattr(p, "oracle_observation", False)):
        warnings.append("oracle_observation=True: lo stato vero e' concesso per dichiarazione")
    rows = []
    k = -1
    by_output: Dict[str, int] = {}
    t0 = time.perf_counter()
    for k in range(int(quarters)):
        world_a = _fresh_copy(env)
        world_b = perturb_hidden_truth(env, seed=perturb_seed + k, sudden=sudden)
        diffs = compare_worlds(world_a, world_b, belief_seed=belief_seed, include=include, limit=limit)
        for key, items in diffs.items():
            by_output[key] = by_output.get(key, 0) + len(items)
            for path, a, b in items[:limit]:
                rows.append({"quarter": env.quarter, "output": key, "path": path, "world_a": a, "world_b": b})
        if env.quarter > int(p.years) * 4:
            break
        env.step(_evidence_plan(env, policy))
    return {
        **_report_header(env, "instant", sudden),
        "quarters_checked": k + 1,
        "observation_mode": p.observation_mode,
        "oracle_observation": bool(getattr(p, "oracle_observation", False)),
        "policy_sandbox": bool(getattr(p, "policy_sandbox", False)),
        "strict_information": bool(getattr(p, "strict_information", False)),
        "regime": getattr(env.regime, "id", None),
        "divergences_by_output": by_output,
        "divergences": rows,
        "clean": not by_output,
        "warnings": warnings,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def _report_header(env, mode: str, sudden: bool) -> Dict[str, Any]:
    from infralab.core.hazards import progressive_events as pe
    from infralab.core.hazards import regional_events as rg
    from infralab.core.hazards import sudden_damage as sd
    stats = (getattr(env, "progressive_stats", None) or {}).values()
    return {"mode": mode, "sudden_injection": bool(sudden),
            "sudden_channel_types": sd.type_names(env.params) if sd.channel_active(env.params) else [],
            "regional_events_enabled": rg.enabled(env.params),
            "regional_events": len(getattr(env, "regional_events", None) or []),
            "progressive_types": pe.type_names(env.params),
            "progressive_initiated": int(sum(st.get("initiated", 0) for st in stats)),
            "progressive_known": int(sum(sum(st.get(f"known_{n}", 0) for n in pe.type_names(env.params))
                                         for st in stats))}


@contextlib.contextmanager
def _progressive_hook(hook):
    """Sostituisce l'esito della rilevazione dei difetti progressivi con
    hook(env, ponte, "inspection"|"monitoring", n, trovati) per la durata del blocco."""
    from infralab.core.hazards import progressive_events as pe
    previous = pe.DETECTION_HOOK
    pe.DETECTION_HOOK = hook
    try:
        yield
    finally:
        pe.DETECTION_HOOK = previous


# ---------------------------------------------------------------------------
# Modalita' traiettoria (RT011): verita' diversa che si accumula, evidenza fissa
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _inspection_hook(hook):
    """Sostituisce `inspect_component` (motore e canale improvviso) con
    hook(originale, componente, *args, **kwargs) per la durata del blocco."""
    import infralab.core.environment as envmod
    import infralab.core.inspection as insp
    orig = insp.inspect_component

    def wrapped(component, *args, **kwargs):
        return hook(orig, component, *args, **kwargs)

    envmod.inspect_component = insp.inspect_component = wrapped
    try:
        yield
    finally:
        envmod.inspect_component = insp.inspect_component = orig


def _component_index(env) -> Dict[int, Tuple[int, str]]:
    return {id(c): (int(bid), str(name)) for bid, b in env.bridges.items() for name, c in b.components.items()}


def step_recording(env, plans) -> Dict[Tuple[Any, int], Any]:
    """Un trimestre del mondo A registrando ogni lettura d'ispezione (ispezioni,
    rilievi post-intervento, prove di carico, danni autodenunciati o rilevati dal
    monitoraggio): {((asset, componente), n-esima lettura): esito}."""
    index, counts, tape = _component_index(env), {}, {}

    def hook(orig, component, *args, **kwargs):
        out = orig(component, *args, **kwargs)
        key = index.get(id(component))
        counts[key] = counts.get(key, 0) + 1
        tape[(key, counts[key])] = out
        return out

    revenue = env.get_agent_revenue_preview

    def revenue_hook(agent):
        value = revenue(agent)
        n = counts[("revenue", agent)] = counts.get(("revenue", agent), 0) + 1
        tape[(("revenue", agent), n)] = value
        return value

    def defect_hook(env_, bridge, kind, n, found):
        # difetti progressivi resi noti da questa ispezione (o dal monitoraggio)
        tape[(("progressive", int(bridge.id), kind), n)] = [
            (d["type"], int(d.get("start_quarter") or 0), d.get("source", "initiation")) for d in found]
        return found

    env.get_agent_revenue_preview = revenue_hook
    try:
        with _inspection_hook(hook), _progressive_hook(defect_hook):
            env.step(plans)
    finally:
        del env.get_agent_revenue_preview
    return tape


def step_replaying(env, plans, tape, revenue_log: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
    """Un trimestre del mondo B con le letture d'ispezione di A (stessa lettura, stesso
    esito POD): la lettura nativa e' comunque estratta (stesso consumo di numeri
    casuali) e scartata. Anche il ricavo del trimestre e' quello di A: il motore lo
    lega all'indice di salute VERO (`_get_revenue_multiplier`), quindi e' evidenza che
    dipende dalla verita' (conto del gestore); la differenza nativa B-A e' annotata in
    `revenue_log`. Ritorna le letture non appaiate (lette in un solo mondo)."""
    index, counts, used, unmatched = _component_index(env), {}, set(), []
    revenue = env.get_agent_revenue_preview

    def revenue_hook(agent):
        value = revenue(agent)
        n = counts[("revenue", agent)] = counts.get(("revenue", agent), 0) + 1
        rec = tape.get((("revenue", agent), n))
        if rec is None:
            unmatched.append(["solo_B", ["revenue", agent], n])
            return value
        used.add((("revenue", agent), n))
        if revenue_log is not None and not _same_number(float(value), float(rec), 1e-9, 1e-9):
            revenue_log.append({"quarter": env.quarter, "agent": agent, "world_a": rec, "world_b_native": value})
        return rec

    env.get_agent_revenue_preview = revenue_hook

    def hook(orig, component, *args, **kwargs):
        out = orig(component, *args, **kwargs)
        key = index.get(id(component))
        counts[key] = counts.get(key, 0) + 1
        rec = tape.get((key, counts[key]))
        if rec is None:
            unmatched.append(["solo_B", key, counts[key]])
            return out
        used.add((key, counts[key]))
        return rec

    def defect_hook(env_, bridge, kind, n, found):
        # Difetti progressivi: in B diventano noti quelli (e solo quelli) scoperti in A
        # dalla stessa ispezione. Se B non ha il difetto (tolto dalla perturbazione) lo
        # riceve ora: l'evidenza e' quella di A, la verita' resta diversa.
        key = (("progressive", int(bridge.id), kind), n)
        rec = tape.get(key)
        if rec is None:
            unmatched.append(["solo_B", list(key[0]), n])
            return []
        used.add(key)
        hidden = [d for d in env_.progressive_defects.get(bridge.id, []) if d.get("known_since") is None]
        out = []
        for typ, start, source in rec:
            match = next((d for d in hidden if d["type"] == typ and int(d.get("start_quarter") or 0) == start
                          and d.get("source", "initiation") == source and all(d is not o for o in out)), None)
            if match is None:
                match = {"type": typ, "start_quarter": start, "end_quarter": int(env_.quarter) + 8,
                         "known_since": None, "source": source}
                env_.progressive_defects.setdefault(bridge.id, []).append(match)
            out.append(match)
        return out

    try:
        with _inspection_hook(hook), _progressive_hook(defect_hook):
            env.step(plans)
    finally:
        del env.get_agent_revenue_preview
    unmatched.extend(["solo_A", k, n] for (k, n) in tape if (k, n) not in used)
    return unmatched


# Eventi dello storico che etichettano un danno vero reso noto da un'osservazione
# (danno idraulico storico rivelato dall'ispezione, danno improvviso rilevato): nel
# controfattuale "stessa evidenza" sono quelli di A.
_TRUTH_LABELLED_EVENTS = ("danno_idraulico", "danno_improvviso", "danno_regionale", "danno_progressivo")
# Eventi pubblici che dipendono legittimamente dalla verita' e non si possono forzare.
_PUBLIC_EVENTS = ("collasso_e_ricostruzione", "crollo_idraulico", "crollo_progressivo", "declassamento_portata")


def sync_labelled_evidence(world_a, world_b) -> None:
    """Allinea in B le etichette di danno di A (storico, hydraulic_damage_quarter) quando
    lo storico dei due mondi differisce SOLO per quelle: il resto dello storico (azioni
    eseguite, bloccate, imposte) resta quello nativo di B, e una differenza li' e' una
    fuga o una divergenza pubblica, non viene coperta.

    Lo storico del motore tiene gli ultimi 12 eventi: le etichette di danno presenti in
    un solo mondo ne spingono fuori di piu' vecchi, quindi il resto dello storico di un
    mondo puo' essere la coda di quello dell'altro. Anche quello e' effetto delle sole
    etichette e viene allineato (un evento nascosto che accorcia lo storico visibile
    resta coperto dal confronto istante per istante)."""
    for bid in sorted(world_a.bridges):
        ba, bb = world_a.bridges[bid], world_b.bridges.get(bid)
        if bb is None:
            continue
        strip_a = [e for e in ba.action_history if e.get("action") not in _TRUTH_LABELLED_EVENTS]
        strip_b = [e for e in bb.action_history if e.get("action") not in _TRUTH_LABELLED_EVENTS]
        ja, jb = _jsonable_any(strip_a), _jsonable_any(strip_b)
        short, long_ = (ja, jb) if len(ja) <= len(jb) else (jb, ja)
        if ja == jb or (short and long_[len(long_) - len(short):] == short):
            bb.action_history = copy.deepcopy(ba.action_history)
            bb.hydraulic_damage_quarter = ba.hydraulic_damage_quarter


def _jsonable_any(obj):
    from infralab.agents.sandbox import _jsonable
    return _jsonable(obj)


def evidence_fingerprint(env) -> Dict[str, Any]:
    """Evidenza che dipende legittimamente dalla verita' e che il controllo non puo'
    forzare uguale: gestori attivi, collassi, eventi pubblici per asset, esito delle
    prove di carico (superata/scartata per componente). Se differisce fra A e B, da
    quel trimestre in poi le storie si separano davvero e il confronto si ferma."""
    proof = {}
    for bid in sorted(env.bridges):
        rep = getattr(env.bridges[bid], "last_proof_load_report", None) or {}
        if rep:
            proof[str(bid)] = [rep.get("quarter"), rep.get("passed"), rep.get("screened"),
                               [[c.get("component"), c.get("passed"), c.get("screened")]
                                for c in rep.get("components") or []]]
    return _jsonable_any({
        "active_agents": sorted(env.active_agents),
        "failures": {a: int((v or {}).get("failures", 0)) for a, v in sorted((env.failure_stats or {}).items())},
        "events": {str(bid): [[e.get("quarter"), e.get("action"), e.get("status")]
                              for e in env.bridges[bid].action_history if e.get("action") in _PUBLIC_EVENTS]
                   for bid in sorted(env.bridges)},
        "proof_load": proof,
        # eventi regionali: pubblici e indipendenti dalla verita' (flussi a chiave)
        "regional_events": [[e.get("quarter"), e.get("lat"), e.get("lon")]
                            for e in getattr(env, "regional_events", None) or []],
        # difetti progressivi NOTI: un difetto noto che matura (o no) secondo la verita'
        # cambia legittimamente l'evidenza
        "progressive_known": _progressive_known(env),
    })


def _progressive_known(env) -> Dict[str, Any]:
    from infralab.core.hazards import progressive_events as pe
    out = {}
    for bid in sorted(getattr(env, "progressive_defects", None) or {}):
        if bid in env.bridges:
            view = pe.public_view(env, env.bridges[bid])
            if view:
                out[str(bid)] = [[d["type"], d["known_since"]] for d in view]
    return out


def _progressive_hidden(env) -> int:
    return sum(1 for defs in (getattr(env, "progressive_defects", None) or {}).values()
               for d in defs if d.get("known_since") is None)


def truth_gap(world_a, world_b) -> Dict[str, float]:
    """Quanto differisce la verita' dei due mondi (il controllo non e' vuoto): scarto
    medio e massimo della condizione vera, danni improvvisi pendenti."""
    gaps = []
    for bid in sorted(world_a.bridges):
        ba, bb = world_a.bridges[bid], world_b.bridges.get(bid)
        if bb is None:
            continue
        for name, ca in ba.components.items():
            cb = bb.components.get(name)
            if cb is not None:
                gaps.append(abs(float(ca.current_condition) - float(cb.current_condition)))
    pa = sum(len(v) for v in (getattr(world_a, "sudden_pending", None) or {}).values())
    pb = sum(len(v) for v in (getattr(world_b, "sudden_pending", None) or {}).values())
    return {"condition_gap_mean": round(float(np.mean(gaps)) if gaps else 0.0, 6),
            "condition_gap_max": round(float(max(gaps)) if gaps else 0.0, 6),
            "sudden_pending_a": pa, "sudden_pending_b": pb,
            "progressive_hidden_a": _progressive_hidden(world_a),
            "progressive_hidden_b": _progressive_hidden(world_b)}


def leak_check_trajectory(env, quarters: int = 8, policy: str = "inspect", perturb_seed: int = 0,
                          belief_seed: int = 12345, include: Optional[Tuple[str, ...]] = None,
                          limit: int = 50, sudden: bool = True) -> Dict[str, Any]:
    """Traiettoria: verita' perturbata UNA volta, poi i due mondi avanzano insieme.

    1. B = `perturb_hidden_truth(A)` al primo trimestre (danni improvvisi compresi);
    2. a ogni trimestre: confronto delle uscite del gestore (copie di A e di B); stesso
       piano (`_evidence_plan` su A) eseguito in A e in B, ciascuno con la PROPRIA
       dinamica (degrado con i propri parametri latenti, danni, riparazioni, collassi);
       l'evidenza generata e' tenuta identica cosi':
       - ogni lettura d'ispezione di B (ispezioni, rilievi post-intervento, prove di
         carico, autodenunce, monitoraggio) e' quella di A, esito POD compreso
         (`step_recording` / `step_replaying`);
       - le etichette di danno nello storico (danno idraulico rivelato, danno
         improvviso rilevato) sono quelle di A (`sync_labelled_evidence`);
       - con `sudden`, a ogni trimestre B riceve altri danni improvvisi nascosti
         (`perturb_hidden_sudden`: solo verita').
       Tutto il resto dello stato di B (conoscenza derivata: incertezza epistemica,
       classi, scadenze, stato del regime e della soglia etica, budget, diagnostica,
       cache) evolve in B senza essere toccato: una fuga che si accumula per trimestri
       in quello stato emerge nel confronto.
    3. l'evidenza che dipende legittimamente dalla verita' e non si puo' forzare
       (collassi, gestori attivi, declassamenti, esito delle prove di carico) e le letture
       non appaiate sono controllate dopo ogni trimestre (`evidence_fingerprint`): alla
       prima differenza il confronto si ferma (`evidence_divergence`) perche' da li' le
       due storie sono legittimamente diverse.

    Limiti: le letture di B sono quelle di A anche se la verita' di B avrebbe dato
    letture diverse (controfattuale "stessa evidenza", non una seconda storia
    plausibile); il confronto copre solo i trimestri prima della prima divergenza
    legittima; i processi casuali pubblici che usano l'RNG condiviso
    (rng_streams="shared") possono separare presto le storie: "keyed" le tiene allineate."""
    import time
    p = env.params
    warnings = []
    if p.observation_mode != "inspected":
        warnings.append("observation_mode='full': lo stato vero e' concesso per dichiarazione, "
                        "le divergenze dell'osservazione sono attese")
    if bool(getattr(p, "oracle_observation", False)):
        warnings.append("oracle_observation=True: lo stato vero e' concesso per dichiarazione")
    rng = np.random.default_rng([int(perturb_seed) & 0x7FFFFFFF, 0x7A1])
    # Modello di collasso storico: un ponte sotto la soglia elimina il gestore (evento
    # pubblico). La perturbazione non deve causarlo da sola: margine sopra la soglia.
    floor = None if p.failure_model == "reliability" else float(p.failure_threshold) + TRAJECTORY_MARGIN
    world_b = perturb_hidden_truth(env, seed=perturb_seed, sudden=sudden, min_condition=floor)
    rows: List[Dict[str, Any]] = []
    by_output: Dict[str, int] = {}
    gaps: List[Dict[str, Any]] = []
    stop: Optional[Dict[str, Any]] = None
    revenue_log: List[Dict[str, Any]] = []
    compared = 0
    t0 = time.perf_counter()
    for _ in range(int(quarters)):
        world_a = _fresh_copy(env)
        gaps.append({"quarter": env.quarter, **truth_gap(env, world_b)})
        diffs = compare_worlds(world_a, _fresh_copy(world_b), belief_seed=belief_seed, include=include, limit=limit)
        compared += 1
        for key, items in diffs.items():
            by_output[key] = by_output.get(key, 0) + len(items)
            for path, a, b in items[:limit]:
                rows.append({"quarter": env.quarter, "output": key, "path": path, "world_a": a, "world_b": b})
        if env.quarter > int(p.years) * 4 or compared >= int(quarters):
            break
        plan = _evidence_plan(env, policy)
        tape = step_recording(env, copy.deepcopy(plan))
        unmatched = step_replaying(world_b, copy.deepcopy(plan), tape, revenue_log)
        sync_labelled_evidence(env, world_b)
        fa, fb = evidence_fingerprint(env), evidence_fingerprint(world_b)
        if unmatched or fa != fb:
            stop = {"quarter": env.quarter - 1,
                    "unmatched_readings": unmatched[:10],
                    "differences": [d[0] for d in diff_json(fa, fb, limit=10)]}
            break
        if sudden:
            perturb_hidden_sudden(world_b, rng, (0.01, 0.05), min_condition=floor)
            perturb_hidden_regional(world_b, rng, (0.01, 0.05), min_condition=floor)
            perturb_hidden_progressive(world_b, rng)
            world_b.refresh_network_reliability()
    return {
        **_report_header(env, "trajectory", sudden),
        "quarters_checked": compared,
        "observation_mode": p.observation_mode,
        "oracle_observation": bool(getattr(p, "oracle_observation", False)),
        "policy_sandbox": bool(getattr(p, "policy_sandbox", False)),
        "strict_information": bool(getattr(p, "strict_information", False)),
        "regime": getattr(env.regime, "id", None),
        "divergences_by_output": by_output,
        "divergences": rows,
        # Un ricavo che dipende dalla salute vera e' un segnale dal vero nel budget del
        # gestore (game_rules.my_budget): fa fallire il controllo.
        "clean": not by_output and not revenue_log,
        "warnings": warnings,
        "truth_gap": gaps,
        "evidence_divergence": stop,
        "revenue_truth_dependence": revenue_log,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def run_leak_check(scenario, params: Optional[Dict[str, Any]] = None, seed: int = 0, quarters: int = 8,
                   policy: str = "inspect", perturb_seed: int = 0, agents: Optional[Dict[str, Dict[str, Any]]] = None,
                   include: Optional[Tuple[str, ...]] = None, sudden: bool = True, trajectory: bool = False,
                   sudden_hazards: bool = False, regional_events: bool = False,
                   progressive_events: bool = False) -> Dict[str, Any]:
    """Costruisce il motore dallo scenario (id di scenarios.json, path JSON o dict) con
    gli override `params` e lancia il controllo. `sudden_hazards=True`: configurazione
    di prova del canale improvviso (TEST_HAZARD_PARAMS, sotto gli override `params`);
    `sudden=False`: nessun danno improvviso iniettato nel mondo B; `trajectory=True`:
    modalita' traiettoria (`leak_check_trajectory`); `regional_events=True`: configurazione
    di prova degli eventi regionali (TEST_REGIONAL_PARAMS, sotto gli override `params`);
    `progressive_events=True`: configurazione di prova degli eventi progressivi
    (TEST_PROGRESSIVE_PARAMS, sotto gli override `params`)."""
    from infralab.lab.build import build_env_from_scenario
    from infralab.lab.spec import resolve_scenario
    sc = resolve_scenario(scenario) if isinstance(scenario, str) else scenario
    overrides = {**(TEST_HAZARD_PARAMS if sudden_hazards else {}),
                 **(TEST_REGIONAL_PARAMS if regional_events else {}),
                 **(TEST_PROGRESSIVE_PARAMS if progressive_events else {}), **dict(params or {})}
    env = build_env_from_scenario(sc, param_overrides=overrides, seed=seed, agents=agents)
    report = leak_check_env(env, quarters=quarters, policy=policy, perturb_seed=perturb_seed, include=include,
                            sudden=sudden, trajectory=trajectory)
    report["scenario"] = sc.get("scenario_id") if isinstance(sc, dict) else str(scenario)
    report["params"] = overrides
    report["seed"] = seed
    return report


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        return text


def format_report(report: Dict[str, Any], max_rows: int = 15) -> str:
    lines = [f"leak_check scenario={report.get('scenario')} seed={report.get('seed')} "
             f"trimestri={report['quarters_checked']} mode={report['observation_mode']} "
             f"sandbox={report['policy_sandbox']} strict_information={report['strict_information']} "
             f"modo={report.get('mode', 'instant')} danni_improvvisi_iniettati={report.get('sudden_injection')} "
             f"canale={','.join(report.get('sudden_channel_types') or []) or 'spento'} "
             f"eventi_regionali={report.get('regional_events', 0) if report.get('regional_events_enabled') else 'spento'} "
             f"eventi_progressivi="
             f"{(str(report.get('progressive_initiated', 0)) + ' inneschi/' + str(report.get('progressive_known', 0)) + ' noti') if report.get('progressive_types') else 'spento'} "
             f"({report['seconds']} s)"]
    gaps = report.get("truth_gap") or []
    if gaps:
        last = gaps[-1]
        lines.append(f"verita' diversa lungo la traiettoria: scarto di condizione medio {last['condition_gap_mean']:.4f} "
                     f"(max {last['condition_gap_max']:.4f}), danni pendenti A={last['sudden_pending_a']} "
                     f"B={last['sudden_pending_b']} (q{last['quarter']})")
    rev = report.get("revenue_truth_dependence") or []
    if rev:
        worst = max(abs(float(r["world_b_native"]) - float(r["world_a"])) for r in rev)
        lines.append(f"FUGA: il ricavo dipende dalla salute VERA (_get_revenue_multiplier): {len(rev)} trimestri/gestore "
                     f"con ricavo nativo di B diverso da A (max |diff| {worst:.2f}); usare revenue_health_source="
                     f"'assessed' o 'none'")
    stop = report.get("evidence_divergence")
    if stop:
        lines.append(f"nota: evidenza legittimamente diversa dopo il trimestre {stop['quarter']} "
                     f"({', '.join(stop['differences'] or [str(u) for u in stop['unmatched_readings']])[:160]}): "
                     f"confronto fermato")
    for w in report["warnings"]:
        lines.append(f"ATTENZIONE: {w}")
    if report["clean"]:
        lines.append("OK: nessuna divergenza fra i due mondi (stessa evidenza, verita' diversa).")
        return "\n".join(lines)
    lines.append("FUGA: uscite che dipendono dallo stato nascosto -> " +
                 ", ".join(f"{k}={v}" for k, v in sorted(report["divergences_by_output"].items())))
    for row in report["divergences"][:max_rows]:
        lines.append(f"  q{row['quarter']} {row['output']}: {row['path']}  A={row['world_a']!r}  B={row['world_b']!r}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="infralab.lab.leak_check",
                                     description="Controllo 'due mondi, stessa evidenza' (cancello pre-campagna)")
    parser.add_argument("--scenario", default="TRAIN-01", help="id di scenarios.json o path JSON")
    parser.add_argument("--param", action="append", default=[], metavar="NOME=VALORE",
                        help="override di SimParameters (ripetibile; valore JSON o testo)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--policy", choices=("inspect", "inspect-only", "none"), default="inspect",
                        help="piano che produce evidenza: ispezioni a rotazione + prove di carico (inspect), "
                             "solo ispezioni (inspect-only), nessuna azione (none)")
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--only", action="append", default=None,
                        choices=("observation", "classification", "regime_triggers", "floor_trigger",
                                 "proof_load_plan", "belief_clone", "planner_snapshot"),
                        help="limita il confronto a queste uscite (ripetibile)")
    parser.add_argument("--agents", type=int, default=1, help="numero di gestori (default 1)")
    parser.add_argument("--trajectory", action="store_true",
                        help="modalita' traiettoria: verita' perturbata una volta, i due mondi avanzano insieme "
                             "con l'evidenza tenuta identica")
    parser.add_argument("--sudden-hazards", action="store_true",
                        help="accende la configurazione di prova del danno improvviso (tutti i tipi, tassi alti; "
                             "i --param espliciti prevalgono)")
    parser.add_argument("--regional-events", action="store_true",
                        help="accende la configurazione di prova degli eventi estremi regionali (eventi "
                             "frequenti, area larga; i --param espliciti prevalgono)")
    parser.add_argument("--progressive-events", action="store_true",
                        help="accende la configurazione di prova degli eventi progressivi (inneschi frequenti "
                             "su tutti i ponti, finestre brevi; i --param espliciti prevalgono)")
    parser.add_argument("--no-sudden-injection", action="store_true",
                        help="non iniettare danni improvvisi nascosti nel mondo B (default: iniettati)")
    parser.add_argument("--json", default=None, help="scrive il rapporto completo in questo file")
    args = parser.parse_args(argv)
    params = {}
    for item in args.param:
        if "=" not in item:
            print(f"--param senza '=': {item!r}", file=sys.stderr)
            return 2
        k, v = item.split("=", 1)
        params[k.strip()] = _parse_value(v.strip())
    try:
        report = run_leak_check(args.scenario, params=params, seed=args.seed, quarters=args.quarters,
                                policy=args.policy, perturb_seed=args.perturb_seed,
                                agents=({f"G{i + 1}": {"type": "Euristico"} for i in range(args.agents)}
                                        if args.agents > 1 else None),
                                include=tuple(args.only) if args.only else None,
                                sudden=not args.no_sudden_injection, trajectory=args.trajectory,
                                sudden_hazards=args.sudden_hazards, regional_events=args.regional_events,
                                progressive_events=args.progressive_events)
    except Exception as e:     # noqa: BLE001 - CLI: messaggio leggibile, codice 2
        print(f"leak_check: errore: {e}", file=sys.stderr)
        return 2
    print(format_report(report))
    if args.json:
        from infralab.settings.config import NumpyJSONEncoder
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=NumpyJSONEncoder, default=str)
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
