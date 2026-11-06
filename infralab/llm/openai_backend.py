"""Backend for the OpenAI chat-completions protocol with function calling.

Serves OpenAI and every endpoint that speaks the same protocol (set `base_url` and `api_key_env`):
OpenRouter, Groq, Mistral, Together, a local Ollama or vLLM server.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


class OpenAIBackend:
    provider = "openai"

    def __init__(self, model: str, base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        from openai import OpenAI
        key = os.environ.get(api_key_env) or ("not-needed" if base_url else None)
        self.client = OpenAI(api_key=key, base_url=base_url) if (key or base_url) else OpenAI()
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def run_turn(self, system: str, user_text: str, tools: List[Dict[str, Any]],
                 execute: Callable[[str, Dict[str, Any]], Dict[str, Any]], max_rounds: int = 6) -> Dict[str, Any]:
        tool_defs = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["parameters"]}} for t in tools]
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        kwargs: Dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_tokens
        for _ in range(max_rounds):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=tool_defs, **kwargs)
            usage["calls"] += 1
            if resp.usage:
                usage["input_tokens"] += resp.usage.prompt_tokens or 0
                usage["output_tokens"] += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                return {"finished": False, "error": "no tool call", "usage": usage}
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            finished = False
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                    out = execute(c.function.name, args if isinstance(args, dict) else {})
                except json.JSONDecodeError as e:
                    out = {"result": {"error": f"arguments are not valid JSON: {e}"}, "is_error": True}
                finished = finished or bool(out.get("finished"))
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out.get("result"), default=str)})
            if finished:
                return {"finished": True, "usage": usage}
        return {"finished": False, "error": "tool rounds exhausted", "usage": usage}
