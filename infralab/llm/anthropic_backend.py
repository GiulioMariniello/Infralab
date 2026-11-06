"""Claude backend of the tool-using player (Anthropic SDK, Messages API, manual tool loop)."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: Optional[str] = None, max_tokens: int = 16000):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def run_turn(self, system: str, user_text: str, tools: List[Dict[str, Any]],
                 execute: Callable[[str, Dict[str, Any]], Dict[str, Any]], max_rounds: int = 6) -> Dict[str, Any]:
        """One decision: the model may call tools until `execute` reports the turn finished."""
        tool_defs = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        kwargs: Dict[str, Any] = {}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        for _ in range(max_rounds):
            # the stable prefix (system and tools) is cached across the quarters of a run
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=tool_defs, messages=messages, **kwargs)
            u = resp.usage
            usage["calls"] += 1
            usage["input_tokens"] += (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
                + (getattr(u, "cache_creation_input_tokens", 0) or 0)
            usage["output_tokens"] += u.output_tokens or 0
            if resp.stop_reason == "refusal":
                return {"finished": False, "error": "refusal", "usage": usage}
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                return {"finished": False, "error": "no tool call (stop_reason: %s)" % resp.stop_reason, "usage": usage}
            # the assistant turn goes back unchanged (thinking blocks included)
            messages.append({"role": "assistant", "content": resp.content})
            results, finished = [], False
            for c in calls:
                out = execute(c.name, c.input if isinstance(c.input, dict) else {})
                finished = finished or bool(out.get("finished"))
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(out.get("result"), default=str),
                                "is_error": bool(out.get("is_error"))})
            if finished:
                return {"finished": True, "usage": usage}
            messages.append({"role": "user", "content": results})
        return {"finished": False, "error": "tool rounds exhausted", "usage": usage}
