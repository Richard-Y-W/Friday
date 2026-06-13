import unittest

import friday.llm.config as llm_config
from friday.llm.config import (
    DEFAULT_ROLE_WIRING,
    ROLES,
    build_router,
    default_llm_settings,
    roles_from_settings,
)


class LlmConfigTests(unittest.TestCase):
    def test_default_wiring_uses_subscription_clis_for_generative_roles(self):
        # The roles where a generative LLM belongs default to subscription
        # CLIs, never the token-billed api providers.
        self.assertEqual(DEFAULT_ROLE_WIRING["planner"], ("codex_cli", ""))
        self.assertEqual(DEFAULT_ROLE_WIRING["composer"], ("codex_cli", ""))
        self.assertEqual(DEFAULT_ROLE_WIRING["verifier"][0], "codex_cli")
        self.assertEqual(DEFAULT_ROLE_WIRING["critic"][0], "codex_cli")
        self.assertEqual(DEFAULT_ROLE_WIRING["feedback"][0], "codex_cli")
        # High-volume screening/extraction stay deterministic.
        self.assertEqual(DEFAULT_ROLE_WIRING["screener"][0], "none")
        self.assertEqual(DEFAULT_ROLE_WIRING["extractor"][0], "none")

    def test_llm_profiles_switch_between_codex_and_claude_writing(self):
        self.assertEqual(getattr(llm_config, "LLM_PROFILE_CHOICES", None), ("codex", "claude"))

        profile_settings = getattr(llm_config, "llm_profile_settings", lambda _profile: {})

        codex = profile_settings("codex")
        self.assertEqual(codex["planner_provider"], "codex_cli")
        self.assertEqual(codex["planner_model"], "")
        self.assertEqual(codex["composer_provider"], "codex_cli")
        self.assertEqual(codex["composer_model"], "")
        self.assertEqual(codex["verifier_provider"], "codex_cli")
        self.assertEqual(codex["critic_provider"], "codex_cli")
        self.assertEqual(codex["feedback_provider"], "codex_cli")

        claude = profile_settings("claude")
        self.assertEqual(claude["planner_provider"], "claude_cli")
        self.assertEqual(claude["planner_model"], "sonnet")
        self.assertEqual(claude["composer_provider"], "claude_cli")
        self.assertEqual(claude["composer_model"], "sonnet")
        self.assertEqual(claude["verifier_provider"], "codex_cli")
        self.assertEqual(claude["verifier_model"], "")
        self.assertEqual(claude["feedback_provider"], "claude_cli")
        self.assertEqual(claude["feedback_model"], "sonnet")

    def test_no_default_role_uses_token_billed_providers(self):
        for role in ROLES:
            self.assertNotIn(DEFAULT_ROLE_WIRING[role][0], ("anthropic", "openai"))

    def test_default_llm_settings_has_provider_and_model_per_role(self):
        section = default_llm_settings()
        for role in ROLES:
            self.assertIn(f"{role}_provider", section)
            self.assertIn(f"{role}_model", section)

    def test_roles_from_settings_reads_section(self):
        settings = {"llm": {"composer_provider": "claude_cli", "composer_model": "opus"}}
        roles = roles_from_settings(settings)
        self.assertEqual(roles["composer"].provider, "claude_cli")
        self.assertEqual(roles["composer"].model, "opus")
        # Unspecified roles fall back to none.
        self.assertEqual(roles["screener"].provider, "none")

    def test_roles_from_settings_tolerates_missing_section(self):
        roles = roles_from_settings({})
        self.assertEqual(set(roles), set(ROLES))
        for role in ROLES:
            self.assertEqual(roles[role].provider, "none")

    def test_build_router_configures_generative_roles(self):
        router = build_router({"llm": default_llm_settings()})
        configured = set(router.configured_roles())
        self.assertEqual(configured, {"planner", "composer", "verifier", "critic", "feedback"})


if __name__ == "__main__":
    unittest.main()
