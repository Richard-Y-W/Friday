from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError

from friday.llm._transport import Opener, post_json
from friday.llm.types import LLMRequest, LLMResponse, ProviderStatus

try:  # pragma: no cover - default transport
    from urllib.request import urlopen as _default_opener
except Exception:  # pragma: no cover
    _default_opener = None


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIProvider:
    """OpenAI-compatible chat-completions backend (also works for many local
    OpenAI-compatible servers via ``base_url``)."""

    name = "openai"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        opener: Opener | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.opener = opener or _default_opener
        self.timeout = timeout

    def check_availability(self) -> ProviderStatus:
        if not os.environ.get(self.api_key_env):
            return ProviderStatus(
                provider="openai",
                available=False,
                reason=f"missing API key env var: {self.api_key_env}",
            )
        return ProviderStatus(provider="openai", available=True)

    def generate(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return LLMResponse(
                provider="openai",
                model=model,
                success=False,
                error=f"missing API key env var: {self.api_key_env}",
            )
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "friday_response", "strict": True, "schema": request.response_schema},
            }
        start = time.monotonic()
        try:
            data = post_json(
                f"{self.base_url}/chat/completions",
                payload,
                {"Authorization": f"Bearer {api_key}"},
                opener=self.opener,
                timeout=self.timeout,
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return LLMResponse(
                provider="openai",
                model=model,
                success=False,
                latency_ms=_elapsed_ms(start),
                error=f"openai generation failed: {exc}",
            )
        return _parse_chat_response(data, model, start)


def _parse_chat_response(data: dict, model: str, start: float) -> LLMResponse:
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return LLMResponse(
            provider="openai",
            model=model,
            success=False,
            latency_ms=_elapsed_ms(start),
            error="openai response missing choices[0].message.content",
        )
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return LLMResponse(
        provider="openai",
        model=model,
        success=True,
        text=str(text or ""),
        latency_ms=_elapsed_ms(start),
        tokens_used=tokens if isinstance(tokens, int) else None,
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
