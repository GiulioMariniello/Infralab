"""Copilota ingegneristico (condizione sperimentale "Uomo + LLM").

Un LLM che NON decide da solo: dialoga con l'ingegnere umano, ne apprende il
profilo di rischio conversando, discute le ipotesi, propone piani motivati
(spiegabilità) — la decisione finale resta all'umano, che può applicare,
modificare o ignorare la proposta. È il quarto braccio dell'esperimento
LLM / Uomo / Uomo+LLM / Euristico.

Protocollo di risposta del modello (SOLO JSON):
  {"reply": "testo del copilota (discussione, domande, motivazioni)",
   "proposed_plan": {"ID_ASSET": "azione"}}          # opzionale
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from infralab.agents.agentic import make_simulation_tools
from infralab.agents.legacy import LLMAgent, NumpyJSONEncoder

COPILOT_SYSTEM = (
    "Sei un copilota di ingegneria strutturale per la gestione di asset. "
    "Collabori con un ingegnere umano: NON decidi da solo. Nel primo scambio "
    "presentati brevemente e fai 1-2 domande per capire il suo profilo "
    "(propensione al rischio, priorita': sicurezza/budget/emissioni). Poi, ad "
    "ogni scambio: discuti le ipotesi, motiva i trade-off (beta, costi, CO2) e "
    "quando ha senso proponi un piano concreto. Rispondi SOLO con JSON: "
    '{"reply": "...", "proposed_plan": {"ID_ASSET": "azione"}} — '
    "proposed_plan e' opzionale, includilo solo se proponi azioni."
)

MAX_HISTORY_MESSAGES = 14


def _chat_completion(provider: str, model_override: Optional[str], system: str, prompt: str,
                     llm_callable: Optional[Callable[[str, str], str]] = None) -> str:
    """Chiamata uniforme ai provider (riusa la config di LLMAgent)."""
    if llm_callable is not None:
        return llm_callable(system, prompt)
    ref = LLMAgent("copilot", "Bilanciato", provider, model=model_override)
    import openai as _openai
    if provider == "OpenAI":
        client = _openai.OpenAI()
        model = ref.openai_model
    elif provider == "Locale":
        client = _openai.OpenAI(base_url=ref.local_base_url,
                                api_key=os.environ.get("INFRALAB_LLM_API_KEY", "not-needed"))
        model = ref.local_model
        if model.startswith("qwen3"):
            prompt = prompt + "\n/no_think"
    elif provider == "Google":
        import google.generativeai as genai
        response = genai.GenerativeModel(ref.google_model).generate_content(system + "\n\n" + prompt)
        return response.text or ""
    else:
        raise ValueError(f"Provider non supportato: {provider}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content or ""


def build_context(env, agent_name: str) -> str:
    """Contesto tecnico compatto: stato asset + rollout inerziale a 4 trimestri."""
    # policy_sandbox: il copilota consiglia un umano in gara, quindi vede cio' che vede
    # il gestore (handle), non il motore.
    from infralab.agents.sandbox import policy_env
    tools = make_simulation_tools(policy_env(env, agent_name), agent_name)
    state = tools["state"]()
    idle = tools["simulate"]({}, quarters=4)
    return (
        "CONTESTO TECNICO (trimestre corrente):\n"
        + json.dumps(state, indent=1, cls=NumpyJSONEncoder)
        + "\n\nPROIEZIONE INERZIALE (nessun intervento, 4 trimestri):\n"
        + json.dumps(idle, cls=NumpyJSONEncoder)
        + "\n\nLe azioni proponibili per ogni asset sono quelle in 'feasible_actions' (col costo in EUR)."
    )


class Copilot:
    """Sessione di dialogo copilota↔ingegnere legata a una partita."""

    def __init__(self, provider: str = "Locale", model: Optional[str] = None,
                 llm_callable: Optional[Callable[[str, str], str]] = None):
        self.provider = provider
        self.model = model
        self.llm_callable = llm_callable

    def respond(self, env, agent_name: str, history: List[Dict[str, str]],
                message: str) -> Dict[str, Any]:
        """Un turno di dialogo. history = [{'role': 'user'|'copilot', 'content': ...}]."""
        recent = history[-MAX_HISTORY_MESSAGES:]
        convo = "\n".join(
            f"[{'INGEGNERE' if m['role'] == 'user' else 'COPILOTA'}] {m['content']}" for m in recent
        )
        prompt = (
            build_context(env, agent_name)
            + ("\n\nCONVERSAZIONE FINORA:\n" + convo if convo else "\n\n(primo scambio: presentati e profila l'ingegnere)")
            + f"\n\n[INGEGNERE] {message}\n\nRispondi SOLO JSON."
        )
        try:
            raw = _chat_completion(self.provider, self.model, COPILOT_SYSTEM, prompt, self.llm_callable)
        except Exception as e:
            return {"reply": f"(copilota non disponibile: {e})", "proposed_plan": None}
        try:
            data = LLMAgent._extract_json(raw)
        except Exception:
            # Il modello ha risposto in prosa: usala come reply, senza proposta.
            return {"reply": raw.strip()[:2000], "proposed_plan": None}
        plan = data.get("proposed_plan") or None
        if plan is not None and not isinstance(plan, dict):
            plan = None
        return {"reply": str(data.get("reply", "")).strip()[:4000], "proposed_plan": plan}
