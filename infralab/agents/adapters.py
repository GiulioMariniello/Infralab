"""Adapter tra gli agenti legacy (AgentInterface.decide(env)) e il protocollo gym-like.

Gli agenti storici decidono leggendo l'intero BridgeEnv; il protocollo passa
l'osservazione e mette l'ambiente in info["env"] al reset. L'adapter conserva
quel riferimento e delega a decide().

Con policy_sandbox=True l'adapter (lato harness) resta l'unico a tenere il motore:
alla policy passa, a ogni decisione, un PlanningHandle nuovo (infralab.agents.sandbox).
Un accesso vietato (SandboxViolation) non ferma la partita: la decisione del
trimestre diventa "nessuna azione" e l'accesso e' registrato in `sandbox_violations`.
"""
from typing import Any, Dict, List

from infralab.agents.legacy import AgentInterface, agent_factory
from infralab.agents.sandbox import PlanningHandle, SandboxViolation, sandbox_enabled


class LegacyAgentAdapter:
    """Avvolge un AgentInterface legacy nel protocollo Agent (reset/act)."""

    def __init__(self, legacy_agent: AgentInterface, **decide_kwargs: Any):
        self.legacy_agent = legacy_agent
        self.decide_kwargs = decide_kwargs
        self._env = None
        self.last_rationale: Any = None
        # Diagnostica del sandbox: [{"quarter", "access"}] (vuota senza sandbox).
        self.sandbox_violations: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self.legacy_agent.name

    def reset(self, observation: Dict[str, Any], info: Dict[str, Any]) -> None:
        self._env = info.get("env")
        if self._env is None:
            raise ValueError("LegacyAgentAdapter richiede info['env'] al reset.")

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("Chiamare reset() prima di act().")
        if not sandbox_enabled(self._env):
            plan, rationale = self.legacy_agent.decide(self._env, **self.decide_kwargs)
            self.last_rationale = rationale
            return plan or {}
        handle = PlanningHandle.build(self._env, self.name)
        try:
            plan, rationale = self.legacy_agent.decide(handle, **self.decide_kwargs)
        except SandboxViolation as e:
            plan, rationale = {}, ("sandbox_violation", str(e))
        for access in handle.violations:
            self.sandbox_violations.append({"quarter": handle.quarter, "access": access})
        self.last_rationale = rationale
        return plan if isinstance(plan, dict) else {}


def make_agent(name: str, config: Dict[str, Any], **decide_kwargs: Any) -> LegacyAgentAdapter:
    """Costruisce un agente protocol-compatibile dalla config legacy.

    config: {"type": "Euristico" | "LLM" | "GA Orizzonte Variabile" | ...}
    """
    return LegacyAgentAdapter(agent_factory(name, config), **decide_kwargs)
