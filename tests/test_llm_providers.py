import io
import json
import unittest

from friday.llm.providers import AnthropicProvider, OllamaProvider, OpenAIProvider
from friday.llm.types import LLMRequest


class FakeHTTPResponse:
    def __init__(self, body):
        self._buffer = io.BytesIO(body.encode("utf-8") if isinstance(body, str) else body)

    def read(self):
        return self._buffer.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_opener(handler):
    """handler(method, url, body) -> response string/bytes (or raises)."""
    def opener(request, timeout=None):
        body = None
        if getattr(request, "data", None):
            body = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse(handler(request.get_method(), request.full_url, body))
    return opener


class OllamaProviderTests(unittest.TestCase):
    def test_availability_true_when_running(self):
        def handler(method, url, body):
            if url.endswith("/api/tags"):
                return json.dumps({"models": [{"name": "qwen2.5-coder:7b"}]})
            return "Ollama is running"
        provider = OllamaProvider(opener=make_opener(handler))
        status = provider.check_availability()
        self.assertTrue(status.available)
        self.assertIn("qwen2.5-coder:7b", status.models)

    def test_availability_false_when_unreachable(self):
        def handler(method, url, body):
            raise OSError("connection refused")
        provider = OllamaProvider(opener=make_opener(handler))
        status = provider.check_availability()
        self.assertFalse(status.available)
        self.assertIn("cannot reach", status.reason)

    def test_generate_combines_system_and_prompt(self):
        captured = {}
        def handler(method, url, body):
            captured.update(body or {})
            return json.dumps({"response": "hello", "eval_count": 5})
        provider = OllamaProvider(opener=make_opener(handler))
        result = provider.generate(LLMRequest(prompt="world", system_prompt="be brief"), "qwen")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.tokens_used, 5)
        self.assertIn("be brief", captured["prompt"])
        self.assertIn("world", captured["prompt"])

    def test_generate_failure_is_structured(self):
        def handler(method, url, body):
            raise OSError("boom")
        provider = OllamaProvider(opener=make_opener(handler))
        result = provider.generate(LLMRequest(prompt="x"), "qwen")
        self.assertFalse(result.success)
        self.assertIn("ollama generation failed", result.error)


class OpenAIProviderTests(unittest.TestCase):
    def test_missing_key_fails_without_network(self):
        provider = OpenAIProvider(api_key_env="FRIDAY_TEST_MISSING_KEY", opener=make_opener(lambda *a: ""))
        self.assertFalse(provider.check_availability().available)
        result = provider.generate(LLMRequest(prompt="x"), "gpt-x")
        self.assertFalse(result.success)
        self.assertIn("missing API key", result.error)

    def test_generate_parses_chat_response(self):
        import os
        os.environ["FRIDAY_TEST_OPENAI_KEY"] = "sk-test"
        try:
            def handler(method, url, body):
                return json.dumps(
                    {"choices": [{"message": {"content": "grounded prose"}}], "usage": {"total_tokens": 12}}
                )
            provider = OpenAIProvider(api_key_env="FRIDAY_TEST_OPENAI_KEY", opener=make_opener(handler))
            result = provider.generate(LLMRequest(prompt="x"), "gpt-x")
            self.assertTrue(result.success)
            self.assertEqual(result.text, "grounded prose")
            self.assertEqual(result.tokens_used, 12)
        finally:
            del os.environ["FRIDAY_TEST_OPENAI_KEY"]


class AnthropicProviderTests(unittest.TestCase):
    def test_generate_parses_messages_response(self):
        import os
        os.environ["FRIDAY_TEST_ANTHROPIC_KEY"] = "sk-ant-test"
        try:
            def handler(method, url, body):
                return json.dumps(
                    {
                        "content": [{"type": "text", "text": "verified claim"}],
                        "usage": {"input_tokens": 4, "output_tokens": 6},
                    }
                )
            provider = AnthropicProvider(api_key_env="FRIDAY_TEST_ANTHROPIC_KEY", opener=make_opener(handler))
            result = provider.generate(LLMRequest(prompt="x", system_prompt="judge"), "claude")
            self.assertTrue(result.success)
            self.assertEqual(result.text, "verified claim")
            self.assertEqual(result.tokens_used, 10)
        finally:
            del os.environ["FRIDAY_TEST_ANTHROPIC_KEY"]

    def test_missing_key_fails_without_network(self):
        provider = AnthropicProvider(api_key_env="FRIDAY_TEST_MISSING_ANTHROPIC", opener=make_opener(lambda *a: ""))
        self.assertFalse(provider.check_availability().available)
        result = provider.generate(LLMRequest(prompt="x"), "claude")
        self.assertFalse(result.success)
        self.assertIn("missing API key", result.error)


if __name__ == "__main__":
    unittest.main()
