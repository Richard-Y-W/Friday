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


DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Anthropic Messages API backend. Good choice for the independent verifier
    or critic role so generation and verification do not share a model family."""

    name = "anthropic"

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
                provider="anthropic",
                available=False,
                reason=f"missing API key env var: {self.api_key_env}",
            )
        return ProviderStatus(provider="anthropic", available=True)

    def generate(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return LLMResponse(
                provider="anthropic",
                model=model,
                success=False,
                error=f"missing API key env var: {self.api_key_env}",
            )
        payload: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        start = time.monotonic()
        try:
            data = post_json(
                f"{self.base_url}/v1/messages",
                payload,
                headers,
                opener=self.opener,
                timeout=self.timeout,
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return LLMResponse(
                provider="anthropic",
                model=model,
                success=False,
                latency_ms=_elapsed_ms(start),
                error=f"anthropic generation failed: {exc}",
            )
        return _parse_messages_response(data, model, start)


def _parse_messages_response(data: dict, model: str, start: float) -> LLMResponse:
    content = data.get("content") if isinstance(data, dict) else None
    text_parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
    if not text_parts:
        return LLMResponse(
            provider="anthropic",
            model=model,
            success=False,
            latency_ms=_elapsed_ms(start),
            error="anthropic response missing text content",
        )
    usage = data.get("usage") if isinstance(data, dict) else None
    tokens = None
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            tokens = input_tokens + output_tokens
    return LLMResponse(
        provider="anthropic",
        model=model,
        success=True,
        text="".join(text_parts),
        latency_ms=_elapsed_ms(start),
        tokens_used=tokens,
    )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
