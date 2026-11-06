"""Questionario post-partita (schema data-driven) + scoring.

Lo schema è una lista di sezioni con item tipizzati; il frontend lo rende
genericamente e il backend calcola i punteggi derivati validati:
- NASA-TLX (carico di lavoro): media dei 6 subscale 0–100.
- Fiducia nell'automazione (subset Jian/Bisantz/Drury): media 1–7 con l'item
  invertito ("ingannevole") ri-orientato. Rilevante per le condizioni con AI.

Aggiungere/modificare item = editare lo schema; lo scoring usa gli item con
`scale` noto, quindi resta robusto ai cambi non strutturali.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Tipi di item: "number", "select", "radio", "likert" (min..max interi),
# "slider" (min..max con step), "text". `scale` raggruppa gli item per il
# calcolo dei punteggi; `reverse=True` inverte l'item nella scala.
QUESTIONNAIRE_SCHEMA: List[Dict[str, Any]] = [
    {
        "id": "profilo",
        "title": "Profilo",
        "items": [
            {"key": "anni_esperienza", "label": "Anni di esperienza in ingegneria civile/strutturale",
             "type": "number", "min": 0, "max": 50, "default": 0},
            {"key": "ruolo", "label": "Ruolo attuale", "type": "select",
             "options": ["Studente magistrale", "Dottorando / Assegnista", "Ricercatore",
                         "Ingegnere professionista", "Altro"]},
            {"key": "ha_gestito_strutture", "label": "Hai mai gestito o ispezionato strutture reali?",
             "type": "radio", "options": ["Sì", "No"]},
            {"key": "familiarita_affidabilita", "label": "Familiarità con l'analisi di affidabilità strutturale",
             "type": "likert", "min": 1, "max": 5, "low": "Nessuna", "high": "Esperto"},
        ],
    },
    {
        "id": "nasa_tlx",
        "title": "Carico di lavoro (NASA-TLX)",
        "hint": "Per ciascuna voce, indica il valore da 0 (basso) a 100 (alto).",
        "items": [
            {"key": "tlx_mental", "label": "Richiesta mentale", "type": "slider", "min": 0, "max": 100, "step": 5,
             "scale": "nasa_tlx", "low": "Bassa", "high": "Alta"},
            {"key": "tlx_physical", "label": "Richiesta fisica", "type": "slider", "min": 0, "max": 100, "step": 5,
             "scale": "nasa_tlx", "low": "Bassa", "high": "Alta"},
            {"key": "tlx_temporal", "label": "Pressione temporale", "type": "slider", "min": 0, "max": 100, "step": 5,
             "scale": "nasa_tlx", "low": "Bassa", "high": "Alta"},
            {"key": "tlx_performance", "label": "Insuccesso percepito (0 = perfetto, 100 = pessimo)",
             "type": "slider", "min": 0, "max": 100, "step": 5, "scale": "nasa_tlx", "low": "Successo", "high": "Fallimento"},
            {"key": "tlx_effort", "label": "Sforzo", "type": "slider", "min": 0, "max": 100, "step": 5,
             "scale": "nasa_tlx", "low": "Basso", "high": "Alto"},
            {"key": "tlx_frustration", "label": "Frustrazione", "type": "slider", "min": 0, "max": 100, "step": 5,
             "scale": "nasa_tlx", "low": "Bassa", "high": "Alta"},
        ],
    },
    {
        "id": "fiducia",
        "title": "Fiducia nel supporto AI",
        "hint": "Rispondi solo se hai usato un copilota o un agente AI (condizione Uomo+LLM). 1 = per niente, 7 = moltissimo.",
        "optional": True,
        "items": [
            {"key": "trust_reliable", "label": "Il sistema AI è affidabile", "type": "likert", "min": 1, "max": 7, "scale": "trust"},
            {"key": "trust_information", "label": "Mi fido delle informazioni fornite dal sistema", "type": "likert", "min": 1, "max": 7, "scale": "trust"},
            {"key": "trust_deceptive", "label": "Il sistema è ingannevole o fuorviante", "type": "likert", "min": 1, "max": 7, "scale": "trust", "reverse": True},
            {"key": "trust_rely", "label": "Posso contare sul sistema quando decido", "type": "likert", "min": 1, "max": 7, "scale": "trust"},
            {"key": "trust_improved", "label": "Il supporto AI ha migliorato le mie decisioni", "type": "likert", "min": 1, "max": 7, "scale": "trust"},
        ],
    },
    {
        "id": "strategia",
        "title": "Strategia decisionale",
        "items": [
            {"key": "strategia_principale", "label": "Criterio usato principalmente", "type": "select",
             "options": ["Priorità alla sicurezza (asset più rischiosi prima)",
                         "Priorità al budget (azioni più economiche)",
                         "Bilanciamento sicurezza / costo",
                         "Preventiva (ispezionare sistematicamente)",
                         "Reattiva (intervenire solo all'emergenza)", "Altro"]},
            {"key": "uso_beta", "label": "Hai usato l'indice di affidabilità β per prioritizzare?", "type": "radio",
             "options": ["Sì, sistematicamente", "Solo in alcuni casi", "No, mi sono basato sulla condizione"]},
            {"key": "consapevole_incertezza",
             "label": "Eri consapevole che la condizione reale può differire da quella osservata?",
             "type": "radio", "options": ["Sì, l'ho considerato nelle decisioni",
                                          "Sì, ma non ha cambiato il mio approccio", "No"]},
            {"key": "difficolta_percepita", "label": "Difficoltà percepita dello scenario", "type": "likert",
             "min": 1, "max": 5, "low": "Molto facile", "high": "Molto difficile"},
            {"key": "note_libere", "label": "Note libere sulla tua strategia (opzionale)", "type": "text", "max": 800},
        ],
    },
]


def _scale_items(scale: str) -> List[Dict[str, Any]]:
    out = []
    for section in QUESTIONNAIRE_SCHEMA:
        for item in section["items"]:
            if item.get("scale") == scale:
                out.append(item)
    return out


def score_questionnaire(responses: Dict[str, Any]) -> Dict[str, Any]:
    """Calcola i punteggi derivati dalle risposte grezze.

    Ritorna {nasa_tlx_raw, nasa_tlx_n, trust_score, trust_n}. I punteggi sono
    None se nessun item della scala è stato compilato (es. condizione senza AI
    per la fiducia).
    """
    scores: Dict[str, Any] = {}

    tlx_vals = [responses[i["key"]] for i in _scale_items("nasa_tlx")
                if isinstance(responses.get(i["key"]), (int, float))]
    scores["nasa_tlx_raw"] = round(sum(tlx_vals) / len(tlx_vals), 2) if tlx_vals else None
    scores["nasa_tlx_n"] = len(tlx_vals)

    trust_terms: List[float] = []
    for item in _scale_items("trust"):
        v = responses.get(item["key"])
        if isinstance(v, (int, float)):
            lo, hi = item.get("min", 1), item.get("max", 7)
            trust_terms.append((lo + hi - v) if item.get("reverse") else v)
    scores["trust_score"] = round(sum(trust_terms) / len(trust_terms), 2) if trust_terms else None
    scores["trust_n"] = len(trust_terms)
    return scores
