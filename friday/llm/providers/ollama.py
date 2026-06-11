from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError

from friday.llm._transport import Opener, get_text, post_json
from friday.llm.types import LLMRequest, LLMResponse, ProviderStatus

try:  # pragma: no cover - default transport
    from urllib.request import urlopen as _default_opener
except Exception:  # pragma: no cover
    _default_opener = None


DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    """Local Ollama backend. No API key, no per-token cost.

    This is the preferred provider for high-volume style and critic work, which
    keeps Friday's zero-cost posture even when LLM leaves are enabled.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        opener: Opener | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.opener = opener or _default_opener
        self.timeout = timeout

    def check_availability(self) -> ProviderStatus:
        try:
            body = get_text(self.base_url, opener=self.opener, timeout=min(self.timeout, 10.0))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            return ProviderStatus(
                provider="ollama",
                available=False,
                reason=f"cannot reach Ollama at {self.base_url}: {exc}",
            )
        if "Ollama is running" not in body:
            return ProviderStatus(
                provider="ollama",
                available=False,
                reason=f"unexpected response from {self.base_url}",
            )
        return ProviderStatus(provider="ollama", available=True, models=self._list_models())

    def _list_models(self) -> tuple[str, ...]:
        try:
            raw = get_text(f"{self.base_url}/api/tags", opener=self.opener, timeout=min(self.timeout, 10.0))
            data = json.loads(raw)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
            return ()
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return ()
        return tuple(str(item.get("name")) for item in models if isinstance(item, dict) and item.get("name"))

    def generate(self, request: LLMRequest, model: str) -> LLMResponse:
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        start = time.monotonic()
        try:
            data = post_json(
                f"{self.base_url}/api/generate",
                payload,
                {},
                opener=self.opener,
                timeout=self.timeout,
            )
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return LLMResponse(
                provider="ollama",
                model=model,
                success=False,
                latency_ms=_elapsed_ms(start),
                error=f"ollama generation failed: {exc}",
            )
        return LLMResponse(
            provider="ollama",
            model=model,
            success=True,
            text=str(data.get("response") or ""),
            latency_ms=_elapsed_ms(start),
            tokens_used=data.get("eval_count") if isinstance(data.get("eval_count"), int) else None,
        )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
