"""Agente LLM *agentico*: decide scrivendo ed eseguendo codice sulla simulazione.

Differenza rispetto a LLMAgent (risposta diretta): questo agente entra in un
loop ReAct con un ambiente di esecuzione Python sandboxato in cui può
- ispezionare lo stato (`state()`, `feasible_actions()`),
- CLONARE la simulazione e fare rollout dei propri piani (`simulate(...)`),
- costruirsi ottimizzazioni proprie (ricerche greedy, enumerazioni, euristiche)
prima di impegnarsi in un piano. Il rollout avviene su una copia serializzata
dell'ambiente (`to_dict`/`from_dict`): l'ambiente reale non viene mai toccato.

Protocollo di risposta del modello (SOLO JSON):
  {"action": "run_code", "code": "<python>"}          -> esegue e restituisce l'output
  {"action": "final_plan", "plan": {...}, "explanation": "..."}
Budget: al massimo `max_rounds` esecuzioni di codice per decisione; timeout per
esecuzione; output troncato. Al termine del budget viene forzato il piano.

Sicurezza (sandbox best-effort per un contesto di ricerca):
- builtins whitelist (niente open/exec/eval), import consentiti solo da
  ALLOWED_IMPORTS (math, statistics, itertools, random, json, heapq),
- nessun accesso a file/rete, print catturato, esecuzione in thread con timeout.
"""
from __future__ import annotations

import builtins as _builtins
import io
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from infralab.agents.legacy import LLMAgent, NumpyJSONEncoder

ALLOWED_IMPORTS = {"math", "statistics", "itertools", "random", "json", "heapq", "functools", "collections"}

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "range",
    "repr", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "ValueError", "KeyError", "TypeError", "Exception", "StopIteration",
]

MAX_OUTPUT_CHARS = 4000
CODE_TIMEOUT_S = 15.0


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"Import non consentito nella sandbox: {name!r} (consentiti: {sorted(ALLOWED_IMPORTS)})")
    return __import__(name, globals, locals, fromlist, level)


class CodeSandbox:
    """Esegue codice Python con builtins ristretti e tool di simulazione."""

    def __init__(self, tools: Dict[str, Callable]):
        safe = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES}
        safe["__import__"] = _safe_import
        self._base_globals: Dict[str, Any] = {"__builtins__": safe, **tools}

    def run(self, code: str, timeout_s: float = CODE_TIMEOUT_S) -> str:
        buffer = io.StringIO()

        def _print(*args, **kwargs):
            kwargs.pop("file", None)
            print(*args, file=buffer, **kwargs)

        env = dict(self._base_globals)
        env["print"] = _print
        result: Dict[str, Any] = {}

        def _target():
            try:
                exec(compile(code, "<agent_code>", "exec"), env)  # noqa: S102 - sandboxata
                result["ok"] = True
            except Exception as e:  # noqa: BLE001 - l'errore torna al modello
                result["error"] = f"{type(e).__name__}: {e}"

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive():
            return f"[TIMEOUT dopo {timeout_s}s — semplifica il codice]"
        out = buffer.getvalue()
        if "error" in result:
            out += f"\n[ERRORE] {result['error']}"
        if not out.strip():
            out = "[nessun output — usa print(...) per vedere i risultati]"
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n[output troncato a {MAX_OUTPUT_CHARS} caratteri]"
        return out


# ---------------------------------------------------------------------------
# Tool di simulazione (operano su CLONI dell'ambiente)
# ---------------------------------------------------------------------------

def make_sandboxed_tools(handle, agent_name: str) -> Dict[str, Callable]:
    """Tool con policy_sandbox: `handle` e' un PlanningHandle (nessun motore).

    state() e feasible_actions() leggono l'osservazione sanitizzata; simulate() fa il
    rollout su una copia del mondo del gestore (belief_clone) con futuro casuale
    proprio: non e' chiaroveggente sui sorteggi reali. Le chiusure dei tool tengono
    solo il handle, che non contiene riferimenti al motore."""
    from infralab.agents.scripted import compact_observation

    def state() -> Dict[str, Any]:
        """Stato compatto: per asset -> condizioni, beta, budget, azioni fattibili (stime del gestore)."""
        c = compact_observation(handle.get_agent_observations(agent_name), digits=(3, 2, 0))
        return {"quarter": c["quarter"], "budget": round(c["budget"]), "emissions_tco2": round(c["emissions_tco2"], 2),
                "assets": [{"id": a["id"], "name": a["name"], "min_condition": a["condition_min"], "beta": a["beta"],
                            "annual_pf": a["annual_pf"], "components": a["components"],
                            "feasible_actions": {k: round(v) for k, v in a["feasible_actions"].items()}}
                           for a in c["assets"]]}

    def feasible_actions() -> Dict[str, List[str]]:
        """{asset_id: [azioni fattibili adesso]}"""
        return {k: [a for a, i in v.items() if i.get("feasible")] for k, v in handle.feasible_actions().items()}

    def simulate(plan: Optional[Dict[str, str]] = None, quarters: int = 4,
                 plans_by_quarter: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Rollout su una COPIA del mondo del gestore (credenza, futuro casuale proprio)."""
        return _rollout_kpis(handle.planning_clone(), agent_name, plan, quarters, plans_by_quarter)

    return {"state": state, "feasible_actions": feasible_actions, "simulate": simulate}


def _rollout_kpis(clone, agent_name: str, plan, quarters, plans_by_quarter) -> Dict[str, Any]:
    quarters = max(1, min(int(quarters), 24))
    schedule = plans_by_quarter if plans_by_quarter is not None else [plan or {}]
    for q in range(quarters):
        step_plan = schedule[q] if q < len(schedule) else {}
        clone.step({agent_name: {"maintenance": {str(k): v for k, v in (step_plan or {}).items()}}})
    my_assets = [b for b in clone.bridges.values() if b.manager == agent_name]
    betas = [b.reliability_index for b in my_assets if b.reliability_index is not None]
    return {
        "quarters_simulated": quarters,
        "final_budget": round(clone.budgets.get(agent_name, 0.0)),
        "min_beta": round(min(betas), 3) if betas else None,
        "mean_condition": round(sum(b.min_condition for b in my_assets) / max(1, len(my_assets)), 3),
        "emissions_tco2": round(getattr(clone, "emissions_tco2", {}).get(agent_name, 0.0), 2),
        "still_active": agent_name in clone.active_agents,
    }


def make_simulation_tools(env, agent_name: str) -> Dict[str, Callable]:
    """Costruisce i tool esposti nella sandbox per un dato ambiente/agente."""
    from infralab.agents.sandbox import is_sandboxed
    if is_sandboxed(env):
        return make_sandboxed_tools(env, agent_name)
    from infralab.core.environment import BridgeEnv

    snapshot = env.to_dict()  # stato congelato al momento della decisione

    def _clone():
        return BridgeEnv.from_dict(json.loads(json.dumps(snapshot, cls=NumpyJSONEncoder)))

    def state() -> Dict[str, Any]:
        """Stato compatto: per asset -> condizioni, beta, budget, azioni fattibili."""
        assets = []
        for b in env.bridges.values():
            if b.manager != agent_name:
                continue
            feas = env.get_feasible_actions_for_bridge(agent_name, b)
            assets.append({
                "id": b.id,
                "name": b.name,
                "min_condition": round(b.min_condition, 3),
                "beta": None if b.reliability_index is None else round(b.reliability_index, 2),
                "annual_pf": b.annual_failure_probability,
                "components": {n: round(c.current_condition, 3) for n, c in b.components.items()},
                "feasible_actions": {a: round(i["cost"]) for a, i in feas.items() if i["feasible"]},
            })
        return {"quarter": env.quarter, "budget": round(env.budgets.get(agent_name, 0.0)),
                "emissions_tco2": round(getattr(env, "emissions_tco2", {}).get(agent_name, 0.0), 2),
                "assets": assets}

    def feasible_actions() -> Dict[str, List[str]]:
        """{asset_id: [azioni fattibili adesso]}"""
        out = {}
        for b in env.bridges.values():
            if b.manager != agent_name:
                continue
            feas = env.get_feasible_actions_for_bridge(agent_name, b)
            out[str(b.id)] = [a for a, i in feas.items() if i["feasible"]]
        return out

    def simulate(plan: Optional[Dict[str, str]] = None, quarters: int = 4,
                 plans_by_quarter: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Rollout su un CLONE: applica `plan` al primo trimestre (o la lista
        `plans_by_quarter`) poi avanza senza azioni. Restituisce i KPI finali.
        Non modifica la partita reale."""
        clone = _clone()
        quarters = max(1, min(int(quarters), 24))
        schedule = plans_by_quarter if plans_by_quarter is not None else [plan or {}]
        for q in range(quarters):
            step_plan = schedule[q] if q < len(schedule) else {}
            clone.step({agent_name: {"maintenance": {str(k): v for k, v in (step_plan or {}).items()}}})
        my_assets = [b for b in clone.bridges.values() if b.manager == agent_name]
        betas = [b.reliability_index for b in my_assets if b.reliability_index is not None]
        return {
            "quarters_simulated": quarters,
            "final_budget": round(clone.budgets.get(agent_name, 0.0)),
            "min_beta": round(min(betas), 3) if betas else None,
            "mean_condition": round(sum(b.min_condition for b in my_assets) / max(1, len(my_assets)), 3),
            "emissions_tco2": round(getattr(clone, "emissions_tco2", {}).get(agent_name, 0.0), 2),
            "still_active": agent_name in clone.active_agents,
        }

    return {"state": state, "feasible_actions": feasible_actions, "simulate": simulate}


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM = "Rispondi SOLO con un oggetto JSON valido, senza testo fuori dal JSON."

AGENTIC_INSTRUCTIONS = """Sei un asset manager AGENTICO con profilo '{profile}'. Trimestre {quarter}.
Puoi SCRIVERE ED ESEGUIRE CODICE PYTHON per analizzare la rete e testare piani
su una copia della simulazione prima di decidere. Strumenti disponibili nel codice:

- state() -> dict: stato dei tuoi asset (condizioni, beta, budget, azioni fattibili con costo)
- feasible_actions() -> dict: {{asset_id: [azioni fattibili]}}
- simulate(plan=None, quarters=4, plans_by_quarter=None) -> dict:
    rollout su un CLONE (non tocca la partita reale). plan = {{asset_id: azione}}
    applicato al primo trimestre. Ritorna final_budget, min_beta, mean_condition,
    emissions_tco2, still_active.
- import consentiti: math, statistics, itertools, random, json, heapq, functools, collections
- usa print(...) per vedere i risultati.

Strategia consigliata: ispeziona lo stato, genera piani candidati, valutali con
simulate() (es. confronto min_beta/budget/emissioni), scegli il migliore.
Hai al massimo {rounds_left} esecuzioni di codice rimaste.

RISPONDI SOLO JSON, in uno dei due formati:
{{"action": "run_code", "code": "..."}}
{{"action": "final_plan", "plan": {{"maintenance": {{"ID_ASSET": "azione"}}}}, "explanation": "..."}}
"""


class AgenticLLMAgent(LLMAgent):
    """LLM con loop agentico: codice + rollout su cloni, poi piano finale.

    `llm_callable` (test/ricerca): funzione (system, prompt) -> str che
    sostituisce la chiamata al provider — permette policy scriptate e test
    deterministici dell'intero harness senza rete.
    """

    def __init__(self, name: str, profile: str, provider: str, model: str | None = None,
                 max_rounds: int = 3, llm_callable: Optional[Callable[[str, str], str]] = None,
                 hf_token: Optional[str] = None):
        super().__init__(name, profile, provider, model=model, hf_token=hf_token)
        self.max_rounds = max_rounds
        self.llm_callable = llm_callable

    # -- chiamata al modello -------------------------------------------------
    def _chat(self, prompt: str) -> str:
        if self.llm_callable is not None:
            return self.llm_callable(AGENTIC_SYSTEM, prompt)
        import os

        import openai as _openai
        if self.provider == "OpenAI":
            client = _openai.OpenAI()
            model = self.openai_model
        elif self.provider == "Locale":
            client = _openai.OpenAI(base_url=self.local_base_url,
                                    api_key=os.environ.get("INFRALAB_LLM_API_KEY", "not-needed"))
            model = self.local_model
            # Qwen3: disattiva il thinking-mode (soft switch) — altrimenti la
            # generazione può richiedere minuti per risposta su hardware locale.
            if model.startswith("qwen3"):
                prompt = prompt + "\n/no_think"
        elif self.provider == "HuggingFace":
            from huggingface_hub import InferenceClient
            token = self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            client = InferenceClient(token=token)
            response = client.chat.completions.create(
                model=self.hf_model,
                messages=[{"role": "system", "content": AGENTIC_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        elif self.provider == "Google":
            import google.generativeai as genai
            response = genai.GenerativeModel(self.google_model).generate_content(
                AGENTIC_SYSTEM + "\n\n" + prompt)
            return response.text or ""
        else:
            raise ValueError(f"Provider non supportato: {self.provider}")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": AGENTIC_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    # -- loop agentico -------------------------------------------------------
    def decide(self, env, **kwargs) -> Tuple[Dict[str, Any], Any]:
        tools = make_simulation_tools(env, self.name)
        sandbox = CodeSandbox(tools)
        obs_state = tools["state"]()
        transcript: List[Dict[str, str]] = []

        context = (AGENTIC_INSTRUCTIONS.format(profile=self.profile, quarter=env.quarter,
                                               rounds_left=self.max_rounds)
                   + "\nSTATO ATTUALE:\n" + json.dumps(obs_state, indent=1, cls=NumpyJSONEncoder))

        prompt = context
        code_runs = 0
        # Chiamate LLM totali: esecuzioni di codice + margine per retry JSON e
        # per la risposta finale forzata dopo l'esaurimento del budget.
        max_llm_calls = self.max_rounds + 3
        for _ in range(max_llm_calls):
            try:
                raw = self._chat(prompt)
            except Exception as e:  # errore provider: nessuna azione, log
                return {}, ("", f"Errore LLM agentico: {e}")
            try:
                data = self._extract_json(raw)
            except Exception:
                transcript.append({"role": "model", "content": raw[:500]})
                prompt = context + "\n\nLa tua ultima risposta non era JSON valido. Rispondi SOLO JSON."
                continue

            action = data.get("action")
            if action == "final_plan":
                transcript.append({"role": "model", "content": json.dumps(data)[:800]})
                return data.get("plan", {}), (context, (data, transcript))

            if action == "run_code" and code_runs < self.max_rounds:
                code = data.get("code", "")
                code_runs += 1
                output = sandbox.run(code)
                transcript.append({"role": "model", "content": code[:800]})
                transcript.append({"role": "sandbox", "content": output[:800]})
                rounds_left = self.max_rounds - code_runs
                prompt = (context
                          + f"\n\nESECUZIONE #{code_runs} — codice:\n{code}\n\nOUTPUT:\n{output}"
                          + f"\n\nEsecuzioni rimaste: {rounds_left}. "
                            "Se hai abbastanza informazioni, rispondi con final_plan.")
                continue

            # Budget esaurito o azione ignota: forza il piano alla prossima chiamata.
            prompt = context + "\n\nBudget di esecuzioni esaurito: rispondi ORA con final_plan (SOLO JSON)."

        return {}, (context, ("Budget di round esaurito senza piano", transcript))
