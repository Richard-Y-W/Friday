from __future__ import annotations

from typing import Mapping, Optional

from friday.llm.router import ModelRouter
from friday.llm.types import ModelConfig, Provider, ProviderName, Role

# Pipeline roles, in the order they appear in the build plan (§11 / Phase 0).
ROLES: tuple[Role, ...] = ("screener", "extractor", "planner", "composer", "verifier", "critic")

# Default wiring. The two roles where a generative LLM actually belongs (PLAN §1)
# go to the *subscription* CLIs — composer on Claude, verifier/critic on Codex —
# which gives the "independent model family" the gate requires (§6) and bills the
# user's rolling usage window, never per-token API credits. High-volume screening
# and extraction stay deterministic (``none``) so the thousands never hit a model.
DEFAULT_ROLE_WIRING: dict[Role, tuple[ProviderName, str]] = {
    "screener": ("none", ""),
    "extractor": ("none", ""),
    "planner": ("claude_cli", "sonnet"),
    "composer": ("claude_cli", "sonnet"),
    "verifier": ("codex_cli", ""),
    "critic": ("codex_cli", ""),
}


def default_llm_settings() -> dict[str, object]:
    """The ``llm`` settings section: ``<role>_provider`` / ``<role>_model`` keys."""
    section: dict[str, object] = {}
    for role in ROLES:
        provider, model = DEFAULT_ROLE_WIRING[role]
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
