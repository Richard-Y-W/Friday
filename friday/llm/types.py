from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# Provider identifiers. "none" means "no model configured" — the default, which
# keeps Friday a zero-token deterministic tool unless a role is explicitly wired
# to a real provider. "claude_cli"/"codex_cli" run against the user's Claude /
# ChatGPT subscription via the local CLIs (no API tokens); "anthropic"/"openai"
# are the API-key-billed fallbacks.
ProviderName = str  # "ollama" | "claude_cli" | "codex_cli" | "openai" | "anthropic" | "none"

# Roles map a pipeline stage to a model. Keeping roles explicit lets a cheap
# local model screen while a stronger model composes and a *different* model
# verifies (the "independent model family" invariant from the build plan).
Role = str  # "screener" | "extractor" | "composer" | "verifier" | "critic" | "feedback"


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    # Optional JSON schema for providers that support structured output. Providers
    # that cannot enforce it should ignore it; callers must still validate.
    response_schema: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class LLMResponse:
    provider: ProviderName
    model: str
    success: bool
    text: str = ""
    latency_ms: int = 0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    role: Optional[Role] = None


@dataclass(frozen=True)
class ProviderStatus:
    provider: ProviderName
    available: bool
    reason: Optional[str] = None
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderName
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


@runtime_checkable
class Provider(Protocol):
    """A backend that can report availability and generate text.

    Implementations must never raise into the pipeline: failures are returned
    as ``LLMResponse(success=False, error=...)`` so the deterministic spine can
    fall back to template output.
    """

    name: ProviderName

    def check_availability(self) -> ProviderStatus: ...

    def generate(self, request: LLMRequest, model: str) -> LLMResponse: ...
