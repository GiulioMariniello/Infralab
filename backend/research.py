"""Modalità ricerca dell'API: login admin, gestione esperimenti, gioco umano.

- Admin: login con password (hash in st.secrets o fallback sviluppo) → bearer
  token in-memory; gestione partecipanti; generazione link-partita con token
  per-giocatore; monitor delle sessioni di ricerca persistite su disco.
- Player: ingresso via token (?ptoken=...), creazione/ripresa di una sessione
  PERSISTITA (data/sessions/<id>/game_state.json + log CSV), gioco manuale.
- LLM: stato dei provider configurati (OpenAI/Google via chiavi env, Locale
  via endpoint OpenAI-compatibile).
"""
from __future__ import annotations

import json
import os
import secrets as _secrets
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import csv
import io

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import infralab.persistence as persistence
from infralab.ui import auth as ui_auth

router = APIRouter(prefix="/api")

# Token admin attivi (in-memory: si azzerano al riavvio del server).
ADMIN_TOKENS: set[str] = set()


# ---------------------------------------------------------------------------
# Auth admin
# ---------------------------------------------------------------------------

class AdminLoginRequest(BaseModel):
    password: str


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> str:
    if not x_admin_token or x_admin_token not in ADMIN_TOKENS:
        raise HTTPException(status_code=401, detail="Autenticazione admin richiesta")
    return x_admin_token


@router.post("/auth/admin")
def admin_login(req: AdminLoginRequest) -> Dict[str, Any]:
    if not ui_auth.verify_admin_password(req.password):
        raise HTTPException(status_code=401, detail="Password errata")
    token = _secrets.token_urlsafe(24)
    ADMIN_TOKENS.add(token)
    return {"token": token}


@router.post("/auth/admin/logout")
def admin_logout(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    ADMIN_TOKENS.discard(x_admin_token or "")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Partecipanti
# ---------------------------------------------------------------------------

class ParticipantsRequest(BaseModel):
    participants: List[str]


@router.get("/admin/participants")
def get_participants(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_admin(x_admin_token)
    return {"participants": persistence.load_participants()}


@router.put("/admin/participants")
def put_participants(req: ParticipantsRequest, x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_admin(x_admin_token)
    names = [n.strip() for n in req.participants if n.strip()]
    persistence.save_participants(sorted(set(names)))
    return {"participants": persistence.load_participants()}


# ---------------------------------------------------------------------------
# Link partita (token per-giocatore)
# ---------------------------------------------------------------------------

class PlayerLinksRequest(BaseModel):
    scenario_id: str
    participants: List[str] = Field(default_factory=list)
    condition: str = ui_auth.DEFAULT_CONDITION   # "Uomo" | "Uomo+LLM"


@router.get("/admin/conditions")
def experiment_conditions(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_admin(x_admin_token)
    return {"human_conditions": list(ui_auth.HUMAN_CONDITIONS)}


class BatteryLinksRequest(BaseModel):
    participants: List[str] = Field(default_factory=list)
    # Scenari core della batteria (default: ponti Paper A short/long A/B).
    short_scenarios: Optional[List[str]] = None
    long_scenarios: Optional[List[str]] = None


@router.post("/admin/battery-links")
def battery_links(req: BatteryLinksRequest, x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Genera link a BATTERIA (within-subjects) controbilanciati per partecipante.

    Ogni partecipante riceve una batteria 2×2 (copilota × lunghezza) con ordine
    e assegnazione scenario↔condizione bilanciati dalla sua posizione in lista."""
    require_admin(x_admin_token)
    from infralab.study.battery import (DEFAULT_LONG, DEFAULT_SHORT, PRACTICE_SCENARIO,
                                         battery_signature, build_battery)

    participants = req.participants or persistence.load_participants()
    if not participants:
        raise HTTPException(status_code=400, detail="Nessun partecipante (aggiungili prima)")
    short = tuple(req.short_scenarios) if req.short_scenarios else DEFAULT_SHORT
    long = tuple(req.long_scenarios) if req.long_scenarios else DEFAULT_LONG
    known = {s.get("scenario_id") for s in persistence.load_scenarios(validate=False)}
    missing = [s for s in list(short) + list(long) + [PRACTICE_SCENARIO] if s not in known]
    if missing:
        raise HTTPException(status_code=404, detail=f"Scenari batteria mancanti: {missing}")
    links = []
    for i, name in enumerate(participants):
        battery = build_battery(i, short=short, long=long)
        token = ui_auth.issue_battery_token(name, battery)
        links.append({"participant": name, "token": token, "path": f"/?ptoken={token}",
                      "games": len(battery), "sequence": battery_signature(battery)})
    return {"kind": "battery", "n_participants": len(links), "links": links}


@router.post("/admin/player-links")
def player_links(req: PlayerLinksRequest, x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_admin(x_admin_token)
    scenarios = {s.get("scenario_id") for s in persistence.load_scenarios(validate=False)}
    if req.scenario_id not in scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' non trovato")
    if req.condition not in ui_auth.HUMAN_CONDITIONS:
        raise HTTPException(status_code=400, detail=f"Condizione non valida: {req.condition!r}. Valide: {list(ui_auth.HUMAN_CONDITIONS)}")
    participants = req.participants or persistence.load_participants()
    if not participants:
        raise HTTPException(status_code=400, detail="Nessun partecipante (aggiungili prima)")
    links = []
    for name in participants:
        token = ui_auth.issue_player_token(req.scenario_id, name, req.condition)
        links.append({"participant": name, "token": token, "condition": req.condition,
                      "path": f"/?ptoken={token}"})
    return {"scenario_id": req.scenario_id, "condition": req.condition, "links": links}


# ---------------------------------------------------------------------------
# Monitor sessioni di ricerca persistite
# ---------------------------------------------------------------------------

@router.get("/admin/research-sessions")
def research_sessions(x_admin_token: Optional[str] = Header(default=None)) -> List[Dict[str, Any]]:
    require_admin(x_admin_token)
    out: List[Dict[str, Any]] = []
    sessions_dir = persistence.SESSIONS_DIR
    if not os.path.isdir(sessions_dir):
        return out
    for name in sorted(os.listdir(sessions_dir)):
        path = os.path.join(sessions_dir, name)
        if not os.path.isdir(path):
            continue
        entry: Dict[str, Any] = {"session_id": name, "quarter": None, "archived": name.endswith("_ARCHIVED")}
        meta = persistence.load_experiment_metadata(name)
        entry["scenario_id"] = meta.get("scenario_id")
        entry["phase"] = meta.get("experiment_phase")
        gs_path = os.path.join(path, persistence.GAME_STATE_FILENAME)
        if os.path.exists(gs_path):
            try:
                with open(gs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry["quarter"] = data.get("quarter")
                entry["agents"] = list((data.get("budgets") or {}).keys())
                entry["active_agents"] = data.get("active_agents", [])
            except (json.JSONDecodeError, OSError):
                entry["error"] = "game_state non leggibile"
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Analisi esperimento: esiti umani per condizione + export
# ---------------------------------------------------------------------------

def _collect_experiment_rows(scenario_id: str) -> List[Dict[str, Any]]:
    """Righe run-level (una per sessione umana conclusa) dello scenario."""
    from infralab.study.outcomes import session_run_metrics

    rows: List[Dict[str, Any]] = []
    sessions_dir = persistence.SESSIONS_DIR
    if not os.path.isdir(sessions_dir):
        return rows
    for name in sorted(os.listdir(sessions_dir)):
        path = os.path.join(sessions_dir, name)
        if not os.path.isdir(path):
            continue
        meta = persistence.load_experiment_metadata(name)
        if meta.get("scenario_id") != scenario_id:
            continue
        env = persistence.load_game_state(name)
        if env is None:
            continue
        player = meta.get("player_name") or (env.params.agent_names[0] if env.params.agent_names else "player")
        condition = meta.get("condition", "Uomo")
        row = session_run_metrics(env, player, condition, session_id=name)
        # Punteggi questionario (se compilato) affiancati all'esito.
        q = persistence.load_questionnaire(name) or {}
        scores = q.get("scores") or {}
        row["nasa_tlx_raw"] = scores.get("nasa_tlx_raw")
        row["trust_score"] = scores.get("trust_score")
        row["questionnaire_submitted"] = bool(q)
        row["consent_given"] = bool(meta.get("consent_given"))
        rows.append(row)
    return rows


@router.get("/admin/experiment/{scenario_id}")
def experiment_analysis(scenario_id: str, x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Esiti umani raggruppati per condizione + confronto statistico.

    Le condizioni computazionali (LLM, Euristico, ...) si aggiungono lanciando
    una campagna sullo stesso scenario; questo endpoint copre le arm umane
    (Uomo, Uomo+LLM) dalle sessioni persistite."""
    require_admin(x_admin_token)
    from infralab.study.outcomes import aggregate_by_condition

    rows = _collect_experiment_rows(scenario_id)
    return {
        "scenario_id": scenario_id,
        "n_sessions": len(rows),
        "rows": rows,
        **aggregate_by_condition(rows),
    }


def _collect_battery_rows() -> List[Dict[str, Any]]:
    """Righe run-level di tutte le partite di batteria (una per partita giocata)."""
    from infralab.study.outcomes import session_run_metrics

    rows: List[Dict[str, Any]] = []
    sessions_dir = persistence.SESSIONS_DIR
    if not os.path.isdir(sessions_dir):
        return rows
    for name in sorted(os.listdir(sessions_dir)):
        if not name.startswith("BATTERY_") or not os.path.isdir(os.path.join(sessions_dir, name)):
            continue
        meta = persistence.load_experiment_metadata(name)
        env = persistence.load_game_state(name)
        if env is None:
            continue
        player = meta.get("player_name") or "player"
        condition = meta.get("condition", "Uomo")
        row = session_run_metrics(env, player, condition, session_id=name)
        row["battery_id"] = meta.get("battery_id")
        row["length_class"] = meta.get("length_class")
        row["game_index"] = meta.get("game_index")
        row["game_done"] = bool(meta.get("game_done"))
        q = persistence.load_questionnaire(name) or {}
        row["nasa_tlx_raw"] = (q.get("scores") or {}).get("nasa_tlx_raw")
        rows.append(row)
    return rows


@router.get("/admin/experiment-battery")
def experiment_battery(x_admin_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Analisi APPAIATA (within-subjects) delle batterie: Δ con-senza copilota.

    Raggruppa le partite per partecipante e calcola, per ogni esito e per classe
    di lunghezza, la Δ intra-soggetto (Uomo+LLM − Uomo) con Wilcoxon signed-rank."""
    require_admin(x_admin_token)
    from infralab.study.outcomes import COMPARABLE_OUTCOMES
    from infralab.study.stats import paired_delta

    # Escluse le partite di pratica (length_class == "practice") dall'analisi.
    rows = [r for r in _collect_battery_rows()
            if r.get("game_done") and r.get("length_class") != "practice"]
    # {participant: {length_class: {condition: {metric: [valori]}}}}
    metrics_out = []
    length_classes = sorted({r.get("length_class") for r in rows if r.get("length_class")})
    for key, label, field in COMPARABLE_OUTCOMES + [("nasa_tlx_raw", "Carico (NASA-TLX)", "nasa_tlx_raw")]:
        per_length = {}
        for lc in length_classes or [None]:
            subjects: Dict[str, Dict[str, List[float]]] = {}
            for r in rows:
                if lc is not None and r.get("length_class") != lc:
                    continue
                if r.get(field) is None:
                    continue
                subjects.setdefault(r["player_name"], {}).setdefault(r["condition"], []).append(r[field])
            per_length[lc or "all"] = paired_delta(subjects, treat="Uomo+LLM", control="Uomo")
        metrics_out.append({"key": key, "label": label, "by_length": per_length})
    return {"n_games": len(rows),
            "n_participants": len({r["player_name"] for r in rows}),
            "length_classes": length_classes,
            "metrics": metrics_out,
            "rows": rows}


@router.get("/admin/experiment-battery/export.csv")
def experiment_battery_export(x_admin_token: Optional[str] = Header(default=None)) -> Response:
    """Esporta una riga per partita di batteria (tidy, per analisi appaiata esterna)."""
    require_admin(x_admin_token)
    rows = _collect_battery_rows()
    cols = ["battery_id", "player_name", "game_index", "length_class", "condition", "game_done",
            "quarters_survived", "terminated", "final_min_beta_margin", "final_health_index",
            "final_emissions_tco2", "final_net_asset_value", "final_budget", "nasa_tlx_raw"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="experiment_battery.csv"'})


@router.get("/admin/experiment/{scenario_id}/export.csv")
def experiment_export(scenario_id: str, x_admin_token: Optional[str] = Header(default=None)) -> Response:
    """Esporta le righe run-level umane come CSV tidy (una riga per partecipante)."""
    require_admin(x_admin_token)
    rows = _collect_experiment_rows(scenario_id)
    cols = ["session_id", "condition", "player_name", "quarters_survived", "terminated",
            "final_min_beta_margin", "final_health_index", "final_emissions_tco2",
            "final_net_asset_value", "final_budget", "nasa_tlx_raw", "trust_score",
            "questionnaire_submitted", "consent_given"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="experiment_{scenario_id}.csv"'})


# ---------------------------------------------------------------------------
# LLM: stato provider
# ---------------------------------------------------------------------------

@router.get("/llm/status")
def llm_status() -> Dict[str, Any]:
    return {
        "openai": {"configured": bool(os.environ.get("OPENAI_API_KEY")),
                   "model": os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")},
        "google": {"configured": bool(os.environ.get("GOOGLE_API_KEY")),
                   "model": os.environ.get("GOOGLE_LLM_MODEL", "gemini-2.0-flash")},
        "locale": {"configured": True,  # best-effort: endpoint locale sempre selezionabile
                   "base_url": os.environ.get("INFRALAB_LLM_BASE_URL", "http://localhost:11434/v1"),
                   "model": os.environ.get("INFRALAB_LLM_MODEL", "llama3.1")},
        "profiles": ["Molto Prudente", "Prudente", "Bilanciato", "Dinamico", "Aggressivo"],
    }


# ---------------------------------------------------------------------------
# Player join (usato da backend.app che possiede lo store sessioni)
# ---------------------------------------------------------------------------

class PlayerJoinRequest(BaseModel):
    token: str


def resolve_player(token: str) -> Dict[str, str]:
    payload = ui_auth.resolve_player_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Link di gioco non valido o revocato")
    return payload


def new_research_session_id(scenario_id: str, player_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{scenario_id}_{player_name.replace(' ', '_')}_{timestamp}"
