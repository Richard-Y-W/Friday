import unittest

from friday.llm.router import ModelRouter
from friday.llm.types import LLMRequest, LLMResponse, ModelConfig, ProviderStatus


class FakeProvider:
    def __init__(self, name, *, available=True, reason=None):
        self.name = name
        self._available = available
        self._reason = reason
        self.calls = []

    def check_availability(self):
        return ProviderStatus(provider=self.name, available=self._available, reason=self._reason)

    def generate(self, request, model):
        self.calls.append((request, model))
        return LLMResponse(provider=self.name, model=model, success=True, text=f"{self.name}:{model}")


class ModelRouterTests(unittest.TestCase):
    def _request(self):
        return LLMRequest(prompt="hi")

    def test_routes_role_to_configured_provider_and_model(self):
        ollama = FakeProvider("ollama")
        router = ModelRouter(
            {"composer": ModelConfig(provider="ollama", model="qwen2.5")},
            providers={"ollama": ollama},
        )
        response = router.generate("composer", self._request())
        self.assertTrue(response.success)
        self.assertEqual(response.text, "ollama:qwen2.5")
        self.assertEqual(response.role, "composer")
        self.assertEqual(ollama.calls[0][1], "qwen2.5")

    def test_unconfigured_role_fails_gracefully(self):
        router = ModelRouter({}, providers={"ollama": FakeProvider("ollama")})
        response = router.generate("composer", self._request())
        self.assertFalse(response.success)
        self.assertIn("no model configured", response.error)
        self.assertEqual(response.role, "composer")

    def test_none_provider_fails_gracefully(self):
        router = ModelRouter(
            {"composer": ModelConfig(provider="none", model="none")},
            providers={"ollama": FakeProvider("ollama")},
        )
        response = router.generate("composer", self._request())
        self.assertFalse(response.success)
        self.assertIn("no model configured", response.error)

    def test_unavailable_provider_fails_gracefully(self):
        router = ModelRouter(
            {"verifier": ModelConfig(provider="anthropic", model="claude")},
            providers={"anthropic": FakeProvider("anthropic", available=False, reason="missing key")},
        )
        response = router.generate("verifier", self._request())
        self.assertFalse(response.success)
        self.assertIn("provider unavailable", response.error)
        self.assertIn("missing key", response.error)

    def test_unknown_provider_name_fails_gracefully(self):
        router = ModelRouter(
            {"composer": ModelConfig(provider="mystery", model="x")},
            providers={},
        )
        response = router.generate("composer", self._request())
        self.assertFalse(response.success)
        self.assertIn("provider not found", response.error)

    def test_is_available_and_configured_roles(self):
        router = ModelRouter(
            {
                "composer": ModelConfig(provider="ollama", model="qwen"),
                "verifier": ModelConfig(provider="none", model="none"),
            },
            providers={"ollama": FakeProvider("ollama")},
        )
        self.assertTrue(router.is_available("composer"))
        self.assertFalse(router.is_available("verifier"))
        self.assertEqual(router.configured_roles(), ["composer"])

    def test_independent_models_per_role(self):
        ollama = FakeProvider("ollama")
        anthropic = FakeProvider("anthropic")
        router = ModelRouter(
            {
                "composer": ModelConfig(provider="ollama", model="qwen"),
                "verifier": ModelConfig(provider="anthropic", model="claude"),
            },
            providers={"ollama": ollama, "anthropic": anthropic},
        )
        self.assertEqual(router.generate("composer", self._request()).provider, "ollama")
        self.assertEqual(router.generate("verifier", self._request()).provider, "anthropic")

    def test_status_reports_per_role(self):
        router = ModelRouter(
            {"composer": ModelConfig(provider="ollama", model="qwen")},
            providers={"ollama": FakeProvider("ollama")},
        )
        status = router.status()
        self.assertTrue(status.roles["composer"]["available"])
        self.assertEqual(status.roles["composer"]["provider"], "ollama")


if __name__ == "__main__":
    unittest.main()
