from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from friday.llm.providers import (
    AnthropicProvider,
    ClaudeCliProvider,
    CodexCliProvider,
    OllamaProvider,
    OpenAIProvider,
)
from friday.llm.types import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    Provider,
    ProviderName,
    ProviderStatus,
    Role,
)


@dataclass(frozen=True)
class RouterStatus:
    roles: dict[Role, dict[str, object]]
    providers: dict[ProviderName, ProviderStatus]


def default_providers() -> dict[ProviderName, Provider]:
    """Construct the built-in providers. No network or subprocess is touched here.

    ``claude_cli`` / ``codex_cli`` run against the user's Claude / ChatGPT
    *subscription* (rolling usage window) via the local CLIs, never per-token API
    credits — the preferred providers. ``anthropic`` / ``openai`` are the
    API-key (token-billed) fallbacks, off unless a role is explicitly wired to
    them.
    """
    return {
        "ollama": OllamaProvider(),
        "claude_cli": ClaudeCliProvider(),
        "codex_cli": CodexCliProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }


class ModelRouter:
    """Routes a pipeline role to its configured provider+model.

    Construction never calls a model. ``generate`` checks availability and
    returns a structured failure response (never raises) when a role is
    unconfigured or its provider is unavailable, so the deterministic spine can
    fall back to template output.
    """

    def __init__(
        self,
        roles: dict[Role, ModelConfig],
        *,
        providers: Optional[dict[ProviderName, Provider]] = None,
    ) -> None:
        self.roles = dict(roles)
        self.providers = providers if providers is not None else default_providers()
        self._status_cache: dict[ProviderName, ProviderStatus] = {}

    def configured_roles(self) -> list[Role]:
        return [role for role, config in self.roles.items() if config.provider not in (None, "none")]

    def role_config(self, role: Role) -> Optional[ModelConfig]:
        return self.roles.get(role)

    def provider_status(self, provider: ProviderName, *, refresh: bool = False) -> ProviderStatus:
        if not refresh and provider in self._status_cache:
            return self._status_cache[provider]
        impl = self.providers.get(provider)
        if impl is None:
            status = ProviderStatus(provider=provider, available=False, reason=f"unknown provider: {provider}")
        else:
            status = impl.check_availability()
        self._status_cache[provider] = status
        return status

    def is_available(self, role: Role) -> bool:
        config = self.roles.get(role)
        if config is None or config.provider in (None, "none"):
            return False
        return self.provider_status(config.provider).available

    def status(self) -> RouterStatus:
        roles: dict[Role, dict[str, object]] = {}
        seen: dict[ProviderName, ProviderStatus] = {}
        for role, config in self.roles.items():
            if config.provider in (None, "none"):
                roles[role] = {"provider": "none", "model": config.model, "available": False}
                continue
            status = self.provider_status(config.provider)
            seen[config.provider] = status
            roles[role] = {
                "provider": config.provider,
                "model": config.model,
                "available": status.available,
                "reason": status.reason,
            }
        return RouterStatus(roles=roles, providers=seen)

    def generate(self, role: Role, request: LLMRequest) -> LLMResponse:
        config = self.roles.get(role)
        if config is None or config.provider in (None, "none"):
            return LLMResponse(
                provider="none",
                model="none",
                success=False,
                error=f"no model configured for role: {role}",
                role=role,
            )
        impl = self.providers.get(config.provider)
        if impl is None:
            return LLMResponse(
                provider=config.provider,
                model=config.model,
                success=False,
                error=f"provider not found: {config.provider}",
                role=role,
            )
        status = self.provider_status(config.provider)
        if not status.available:
            return LLMResponse(
                provider=config.provider,
                model=config.model,
                success=False,
                error=f"provider unavailable: {status.reason}",
                role=role,
            )
        response = impl.generate(request, config.model)
        # Stamp the role onto the response (providers do not know about roles).
        return LLMResponse(
            provider=response.provider,
            model=response.model,
            success=response.success,
            text=response.text,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            error=response.error,
            role=role,
        )
