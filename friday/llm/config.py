from __future__ import annotations

from typing import Mapping, Optional

from friday.llm.router import ModelRouter
from friday.llm.types import ModelConfig, Provider, ProviderName, Role

# Pipeline roles, in the order they appear in the build plan (§11 / Phase 0).
ROLES: tuple[Role, ...] = ("screener", "extractor", "planner", "composer", "verifier", "critic")

# Named profiles keep provider/model pairs compatible when switching between
# local subscription CLIs. High-volume screening and extraction stay
# deterministic (``none``) so the thousands never hit a model.
LLM_PROFILE_CHOICES = ("codex", "claude")

LLM_PROFILES: dict[str, dict[Role, tuple[ProviderName, str]]] = {
    "codex": {
        "screener": ("none", ""),
        "extractor": ("none", ""),
        "planner": ("codex_cli", ""),
        "composer": ("codex_cli", ""),
        "verifier": ("codex_cli", ""),
        "critic": ("codex_cli", ""),
    },
    "claude": {
        "screener": ("none", ""),
        "extractor": ("none", ""),
        "planner": ("claude_cli", "sonnet"),
        "composer": ("claude_cli", "sonnet"),
        "verifier": ("codex_cli", ""),
        "critic": ("codex_cli", ""),
    },
}

# Default wiring. Use Codex for now because it is the provider currently logged
# in locally; ``friday llm use claude`` restores the Claude-writer/Codex-verifier
# split once Claude CLI auth is available. These are subscription CLIs, never
# token-billed API providers.
DEFAULT_ROLE_WIRING: dict[Role, tuple[ProviderName, str]] = {
    role: wiring for role, wiring in LLM_PROFILES["codex"].items()
}


def default_llm_settings() -> dict[str, object]:
    """The ``llm`` settings section: ``<role>_provider`` / ``<role>_model`` keys."""
    section: dict[str, object] = {"profile": "codex"}
    for role in ROLES:
        provider, model = DEFAULT_ROLE_WIRING[role]
        section[f"{role}_provider"] = provider
        section[f"{role}_model"] = model
    return section


def llm_profile_settings(profile: str) -> dict[str, object]:
    """Return a complete ``llm`` settings section for a named local-CLI profile."""
    if profile not in LLM_PROFILES:
        raise KeyError(f"unknown llm profile: {profile}")
    section: dict[str, object] = {"profile": profile}
    for role in ROLES:
        provider, model = LLM_PROFILES[profile][role]
        section[f"{role}_provider"] = provider
        section[f"{role}_model"] = model
    return section


def roles_from_settings(settings: Mapping[str, object]) -> dict[Role, ModelConfig]:
    """Read the ``llm`` settings section into per-role ``ModelConfig``."""
    section = settings.get("llm") if isinstance(settings, Mapping) else None
    if not isinstance(section, Mapping):
        section = {}
    roles: dict[Role, ModelConfig] = {}
    for role in ROLES:
        provider = str(section.get(f"{role}_provider", "none") or "none")
        model = str(section.get(f"{role}_model", "") or "")
        roles[role] = ModelConfig(provider=provider, model=model)
    return roles


def build_router(
    settings: Mapping[str, object],
    *,
    providers: Optional[dict[ProviderName, Provider]] = None,
) -> ModelRouter:
    """Construct a ``ModelRouter`` from settings. Touches no network/subprocess."""
    return ModelRouter(roles_from_settings(settings), providers=providers)
