"""Thin wrapper over an OpenAI-compatible chat API (DeepSeek by default)."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Protocol

from openai import AuthenticationError, OpenAI, OpenAIError, PermissionDeniedError

from app.config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMAuthError(LLMError):
    """Bad or missing API key. Surfaced to the user instead of degrading silently."""


class LLM(Protocol):
    def complete_json(self, model: str, system: str, user: str) -> dict: ...

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7) -> Any: ...


class DeepSeekClient:
    """Two call styles: JSON mode for extraction, and chat (optionally with tools) for replies.

    The SDK already retries 429/5xx with backoff (max_retries). JSON mode on
    DeepSeek can occasionally return empty content, so it is retried here.
    """

    def __init__(self, api_key: str, base_url: str | None = None, timeout: int | None = None):
        if not api_key:
            raise LLMAuthError("No API key configured. Set DEEPSEEK_API_KEY or enter a key in the UI.")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or settings.base_url,
            timeout=timeout or settings.llm_timeout,
            max_retries=2,
        )

    def complete_json(self, model: str, system: str, user: str, attempts: int = 3) -> dict:
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1200,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return json.loads(content)
                last_error = LLMError("empty JSON response")
            except (AuthenticationError, PermissionDeniedError) as e:
                raise LLMAuthError(f"API key rejected: {e}") from e
            except (OpenAIError, json.JSONDecodeError) as e:
                last_error = e
                log.warning("JSON call failed: %s", e)
        raise LLMError(f"JSON call failed after {attempts} attempts: {last_error}")

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.7) -> Any:
        kwargs: dict[str, Any] = {"model": model, "messages": messages,
                                  "temperature": temperature, "max_tokens": 900}
        if tools:
            kwargs["tools"] = tools
        try:
            return self._client.chat.completions.create(**kwargs).choices[0].message
        except (AuthenticationError, PermissionDeniedError) as e:
            raise LLMAuthError(f"API key rejected: {e}") from e
        except OpenAIError as e:
            raise LLMError(str(e)) from e


@lru_cache(maxsize=32)
def _cached_client(api_key: str) -> DeepSeekClient:
    return DeepSeekClient(api_key)


def make_llm(api_key: str | None) -> LLM:
    """One client per key, so HTTP connections are reused across turns."""
    key = api_key or settings.api_key
    if not key:
        raise LLMAuthError("No API key configured. Set DEEPSEEK_API_KEY or enter a key in the UI.")
    return _cached_client(key)
