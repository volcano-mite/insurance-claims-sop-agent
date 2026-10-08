"""Scripted stand-in for the LLM so the pipeline can be tested without an API key."""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.llm import prompts
from app.llm.client import LLMError


def tool_call(name: str, args: dict, id_: str = "call_1"):
    return SimpleNamespace(id=id_, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeLLM:
    def __init__(self, extractions=None, replies=None, fail_json=False):
        self.extractions = list(extractions or [])
        self.replies = list(replies or [])
        self.fail_json = fail_json
        self.chat_calls: list[dict] = []

    def complete_json(self, model, system, user):
        if self.fail_json:
            raise LLMError("boom")
        if system == prompts.EXTRACTOR_SYSTEM:
            return self.extractions.pop(0) if self.extractions else {}
        return {"subject": "Your claim CL-2048 summary", "body": "What we discussed: ..."}

    def chat(self, model, messages, tools=None, temperature=0.7):
        self.chat_calls.append({"messages": messages, "tools": tools})
        reply = self.replies.pop(0) if self.replies else "OK."
        if isinstance(reply, list):  # a list of tool calls
            return SimpleNamespace(content="", tool_calls=reply)
        return SimpleNamespace(content=reply, tool_calls=None)
