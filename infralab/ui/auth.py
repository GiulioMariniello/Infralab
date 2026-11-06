"""Autenticazione per la modalita' ricerca.

- Token per-giocatore: l'admin genera link con ?token=..., non piu'
  ?scenario_id=&player_name= (impersonabile conoscendo il nome). I token sono
  salvati in data/player_tokens.json con lock di file.
- Password admin: confronto con hash SHA-256 in st.secrets['ADMIN_PASSWORD_HASH']
  (o password in chiaro in st.secrets['ADMIN_PASSWORD']); fallback al valore
  di sviluppo in config solo se nessun secret e' configurato.
"""
import hashlib
import json
import logging
import os
import secrets
from typing import Any, Dict, Optional

from filelock import FileLock

import infralab.persistence as persistence
from infralab.settings.config import ADMIN_PASSWORD as _DEV_ADMIN_PASSWORD

logger = logging.getLogger(__name__)

TOKENS_FILENAME = "player_tokens.json"


def _tokens_path() -> str:
    return os.path.join(persistence.DATA_DIR, TOKENS_FILENAME)


def _tokens_lock() -> FileLock:
    os.makedirs(persistence.DATA_DIR, exist_ok=True)
    return FileLock(_tokens_path() + ".lock", timeout=10)


def _load_tokens() -> Dict[str, Dict[str, Any]]:
    path = _tokens_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("File token non leggibile (%s): registro vuoto.", e)
        return {}


# Condizioni sperimentali assegnabili a un giocatore umano (Paper A).
# "Uomo": gioca da solo. "Uomo+LLM": ha il copilota AI a disposizione.
HUMAN_CONDITIONS = ("Uomo", "Uomo+LLM")
DEFAULT_CONDITION = "Uomo"


def issue_player_token(scenario_id: str, player_name: str,
                       condition: str = DEFAULT_CONDITION) -> str:
    """Genera (o riusa) il token di accesso di un giocatore per uno scenario+condizione.

    La condizione fa parte dell'identità del link: lo stesso partecipante può
    avere link distinti per condizioni diverse (assegnazione sperimentale)."""
    if condition not in HUMAN_CONDITIONS:
        condition = DEFAULT_CONDITION
    with _tokens_lock():
        tokens = _load_tokens()
        for token, payload in tokens.items():
            if (payload.get("scenario_id") == scenario_id
                    and payload.get("player_name") == player_name
                    and payload.get("condition", DEFAULT_CONDITION) == condition):
                return token
        token = secrets.token_urlsafe(16)
        tokens[token] = {"scenario_id": scenario_id, "player_name": player_name, "condition": condition}
        with open(_tokens_path(), "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)
        return token


def issue_battery_token(player_name: str, battery: list) -> str:
    """Genera (o riusa) un token per una BATTERIA di partite (within-subjects).

    Il payload porta la lista di partite (`battery`); il flusso giocatore le
    percorre in sequenza. Riusa il token se esiste già identico."""
    with _tokens_lock():
        tokens = _load_tokens()
        for token, payload in tokens.items():
            if (payload.get("kind") == "battery"
                    and payload.get("player_name") == player_name
                    and payload.get("battery") == battery):
                return token
        token = secrets.token_urlsafe(16)
        tokens[token] = {"kind": "battery", "player_name": player_name, "battery": battery}
        with open(_tokens_path(), "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)
        return token


def resolve_player_token(token: str) -> Optional[Dict]:
    """Payload del token se valido, altrimenti None.

    Link singolo: {scenario_id, player_name, condition}.
    Link a batteria: {kind:"battery", player_name, battery:[...]}.
    """
    if not token:
        return None
    payload = _load_tokens().get(token)
    if payload is None:
        return None
    if payload.get("kind") == "battery":
        return {"kind": "battery", "player_name": payload["player_name"],
                "battery": payload["battery"]}
    return {"scenario_id": payload["scenario_id"], "player_name": payload["player_name"],
            "condition": payload.get("condition", DEFAULT_CONDITION)}


def revoke_player_token(token: str) -> bool:
    with _tokens_lock():
        tokens = _load_tokens()
        if token not in tokens:
            return False
        del tokens[token]
        with open(_tokens_path(), "w", encoding="utf-8") as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)
        return True


def verify_admin_password(password: str) -> bool:
    """Verifica la password admin: secrets (hash o chiaro) > fallback sviluppo."""
    try:
        import streamlit as st

        secrets_store = st.secrets
    except Exception:
        secrets_store = {}

    try:
        stored_hash = secrets_store["ADMIN_PASSWORD_HASH"]
    except (KeyError, FileNotFoundError):
        stored_hash = None
    if stored_hash:
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == str(stored_hash).lower()

    try:
        stored_plain = secrets_store["ADMIN_PASSWORD"]
    except (KeyError, FileNotFoundError):
        stored_plain = None
    if stored_plain:
        return secrets.compare_digest(password, str(stored_plain))

    logger.warning(
        "Nessun secret ADMIN_PASSWORD(_HASH) configurato: uso la password di "
        "sviluppo. Configurare .streamlit/secrets.toml in produzione."
    )
    return secrets.compare_digest(password, _DEV_ADMIN_PASSWORD)
