import json
import unittest
from pathlib import Path

from friday.llm._subprocess import CommandResult, child_env, run_command
from friday.llm.providers import ClaudeCliProvider, CodexCliProvider
from friday.llm.types import LLMRequest


class RecordingRunner:
    """A fake CommandRunner. Records the last call and returns a canned result.

    For codex, an optional ``writes_last_message`` simulates the CLI writing its
    final message to the path given after ``--output-last-message``.
    """

    def __init__(self, result, *, writes_last_message=None):
        self.result = result
        self.writes_last_message = writes_last_message
        self.calls = []

    def __call__(self, args, *, stdin=None, timeout=120.0):
        args = list(args)
        self.calls.append({"args": args, "stdin": stdin, "timeout": timeout})
        if self.writes_last_message is not None and "--output-last-message" in args:
            out_path = Path(args[args.index("--output-last-message") + 1])
            out_path.write_text(self.writes_last_message, encoding="utf-8")
        return self.result


def claude_json(result="ok", *, is_error=False, subtype="success", output_tokens=7):
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "is_error": is_error,
            "result": result,
            "usage": {"input_tokens": 3, "output_tokens": output_tokens},
        }
    )


class SubprocessHelpersTests(unittest.TestCase):
    def test_child_env_strips_billing_credentials(self):
        import os

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-be-stripped"
        os.environ["OPENAI_API_KEY"] = "sk-should-be-stripped"
        try:
            env = child_env()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", env)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["OPENAI_API_KEY"]

    def test_run_command_missing_executable_is_structured(self):
        result = run_command(["friday-no-such-binary-xyz"], timeout=5.0)
        self.assertTrue(result.not_found)
        self.assertEqual(result.returncode, 127)


class ClaudeCliProviderTests(unittest.TestCase):
    def test_generate_parses_print_json(self):
        runner = RecordingRunner(CommandResult(0, claude_json("grounded prose", output_tokens=9), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        result = provider.generate(LLMRequest(prompt="write", system_prompt="be terse"), "sonnet")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "grounded prose")
        self.assertEqual(result.tokens_used, 9)
        self.assertEqual(result.provider, "claude_cli")

    def test_prompt_goes_over_stdin_not_argv(self):
        runner = RecordingRunner(CommandResult(0, claude_json(), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        provider.generate(LLMRequest(prompt="SECRET-PROMPT-BODY"), "sonnet")
        call = runner.calls[-1]
        self.assertEqual(call["stdin"], "SECRET-PROMPT-BODY")
        self.assertNotIn("SECRET-PROMPT-BODY", call["args"])

    def test_print_mode_and_model_flags(self):
        runner = RecordingRunner(CommandResult(0, claude_json(), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        provider.generate(LLMRequest(prompt="x"), "haiku")
        args = runner.calls[-1]["args"]
        self.assertIn("-p", args)
        self.assertIn("--output-format", args)
        self.assertIn("json", args)
        self.assertIn("--model", args)
        self.assertIn("haiku", args)

    def test_system_prompt_appended(self):
        runner = RecordingRunner(CommandResult(0, claude_json(), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        provider.generate(LLMRequest(prompt="x", system_prompt="you are a verifier"), "sonnet")
        args = runner.calls[-1]["args"]
        self.assertIn("--append-system-prompt", args)
        self.assertIn("you are a verifier", args)

    def test_tools_are_denied_by_default(self):
        runner = RecordingRunner(CommandResult(0, claude_json(), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        provider.generate(LLMRequest(prompt="x"), "sonnet")
        args = runner.calls[-1]["args"]
        self.assertIn("--disallowedTools", args)
        denied = args[args.index("--disallowedTools") + 1]
        for tool in ("Bash", "Write", "WebFetch"):
            self.assertIn(tool, denied)

    def test_nonzero_exit_is_structured_failure(self):
        runner = RecordingRunner(CommandResult(1, "", "not logged in"))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        result = provider.generate(LLMRequest(prompt="x"), "sonnet")
        self.assertFalse(result.success)
        self.assertIn("not logged in", result.error)

    def test_is_error_payload_is_failure(self):
        runner = RecordingRunner(CommandResult(0, claude_json("usage limit reached", is_error=True), ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        result = provider.generate(LLMRequest(prompt="x"), "sonnet")
        self.assertFalse(result.success)
        self.assertIn("usage limit reached", result.error)

    def test_timeout_is_structured_failure(self):
        runner = RecordingRunner(CommandResult(124, "", "", timed_out=True))
        provider = ClaudeCliProvider(executable="claude", runner=runner, timeout=12.0)
        result = provider.generate(LLMRequest(prompt="x"), "sonnet")
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_garbage_output_is_structured_failure(self):
        runner = RecordingRunner(CommandResult(0, "not json at all", ""))
        provider = ClaudeCliProvider(executable="claude", runner=runner)
        result = provider.generate(LLMRequest(prompt="x"), "sonnet")
        self.assertFalse(result.success)
        self.assertIn("parse", result.error)

    def test_missing_executable_unavailable(self):
        provider = ClaudeCliProvider(executable="friday-no-such-claude-xyz")
        status = provider.check_availability()
        self.assertFalse(status.available)
        self.assertIn("not found", status.reason)


class CodexCliProviderTests(unittest.TestCase):
    def test_generate_reads_last_message_file(self):
        runner = RecordingRunner(CommandResult(0, "transcript noise", ""), writes_last_message="SUPPORT 0.91")
        provider = CodexCliProvider(executable="codex", runner=runner)
        result = provider.generate(LLMRequest(prompt="verify this"), "gpt-5")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "SUPPORT 0.91")
        self.assertEqual(result.provider, "codex_cli")

    def test_exec_runs_read_only_sandbox(self):
        runner = RecordingRunner(CommandResult(0, "", ""), writes_last_message="ok")
        provider = CodexCliProvider(executable="codex", runner=runner)
        provider.generate(LLMRequest(prompt="x"), "gpt-5")
        args = runner.calls[-1]["args"]
        self.assertIn("exec", args)
        self.assertIn("--sandbox", args)
        self.assertEqual(args[args.index("--sandbox") + 1], "read-only")

    def test_system_prompt_prepended_to_stdin(self):
        runner = RecordingRunner(CommandResult(0, "", ""), writes_last_message="ok")
        provider = CodexCliProvider(executable="codex", runner=runner)
        provider.generate(LLMRequest(prompt="claim", system_prompt="judge faithfully"), "gpt-5")
        self.assertIn("judge faithfully", runner.calls[-1]["stdin"])
        self.assertIn("claim", runner.calls[-1]["stdin"])

    def test_stdout_fallback_when_no_file(self):
        stdout = "codex\nthinking...\ncodex\nFINAL ANSWER\n"
        runner = RecordingRunner(CommandResult(0, stdout, ""))  # no file written
        provider = CodexCliProvider(executable="codex", runner=runner)
        result = provider.generate(LLMRequest(prompt="x"), "gpt-5")
        self.assertTrue(result.success)
        self.assertEqual(result.text, "FINAL ANSWER")

    def test_nonzero_exit_is_structured_failure(self):
        runner = RecordingRunner(CommandResult(1, "", "not authenticated"))
        provider = CodexCliProvider(executable="codex", runner=runner)
        result = provider.generate(LLMRequest(prompt="x"), "gpt-5")
        self.assertFalse(result.success)
        self.assertIn("not authenticated", result.error)


if __name__ == "__main__":
    unittest.main()
