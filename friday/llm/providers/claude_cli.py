from __future__ import annotations

import json
import shutil
import time
from typing import Optional, Sequence

from friday.llm._subprocess import CommandRunner, run_command
from friday.llm.types import LLMRequest, LLMResponse, ProviderStatus

# Built-in Claude Code tools denied to the generator. Friday's LLM leaves are
# pure text generators: they must never touch the filesystem, shell, or network.
# This is the CLI-level enforcement of PLAN §12 — "no tool definitions are ever
# passed, so injection has nothing to call" — and it becomes load-bearing the
# moment the composer reads untrusted paper body text (Phase 4).
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
    "TodoWrite",
)


class ClaudeCliProvider:
    """Anthropic models via the Claude Code CLI, billed against the user's
    Claude subscription (the rolling usage window) — never per-token API credits.

    ``run_command`` strips ``ANTHROPIC_API_KEY`` from the child environment, so a
    stray key can never silently switch Friday to API billing; auth is whatever
    ``claude`` is logged in with. Construction touches nothing; ``generate``
    never raises (failures return ``LLMResponse(success=False)``).
    """

    name = "claude_cli"

    def __init__(
        self,
        *,
        executable: str = "claude",
        runner: Optional[CommandRunner] = None,
        timeout: float = 180.0,
        disallowed_tools: Sequence[str] = DEFAULT_DISALLOWED_TOOLS,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.executable = executable
        self.runner: CommandRunner = runner or run_command
        self.timeout = timeout
        self.disallowed_tools = tuple(disallowed_tools)
        self.extra_args = tuple(extra_args)

    def _resolve(self) -> Optional[str]:
        found = shutil.which(self.executable)
        if found:
            return found
        # Allow an explicit path that shutil.which can't resolve from PATH.
        if "/" in self.executable or "\\" in self.executable:
            return self.executable
        return None

    def check_availability(self) -> ProviderStatus:
        exe = self._resolve()
        if exe is None:
            return ProviderStatus(self.name, False, reason=f"{self.executable!r} not found on PATH")
        result = self.runner([exe, "--version"], stdin=None, timeout=min(self.timeout, 30.0))
        if result.not_found:
            return ProviderStatus(self.name, False, reason=f"{self.executable!r} not found")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            return ProviderStatus(self.name, False, reason=f"`claude --version` failed: {detail}")
        return ProviderStatus(self.name, True, models=(result.stdout.strip(),))

    def generate(self, request: LLMRequest, model: str) -> LLMResponse:
        exe = self._resolve()
        if exe is None:
            return LLMResponse(self.name, model, success=False, error=f"{self.executable!r} not found on PATH")

        args: list[str] = [exe, "-p", "--output-format", "json"]
        if model:
            args += ["--model", model]
        if request.system_prompt:
            args += ["--append-system-prompt", request.system_prompt]
        if self.disallowed_tools:
            args += ["--disallowedTools", " ".join(self.disallowed_tools)]
        args += list(self.extra_args)

        start = time.monotonic()
        # The prompt goes over stdin, not argv: it avoids Windows command-line
        # length limits and quoting hazards on large composer prompts.
        result = self.runner(args, stdin=request.prompt, timeout=self.timeout)
        latency = int((time.monotonic() - start) * 1000)

        if result.not_found:
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"{self.executable!r} not found")
        if result.timed_out:
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"claude timed out after {self.timeout}s")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:500]
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"claude exited {result.returncode}: {detail}")
        return self._parse(result.stdout, model, latency)

    def _parse(self, stdout: str, model: str, latency: int) -> LLMResponse:
        raw = stdout.strip()
        if not raw:
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error="claude returned empty output")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"could not parse claude JSON: {exc}")
        if not isinstance(data, dict):
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error="unexpected claude JSON shape")
        if data.get("is_error") or data.get("subtype") not in (None, "success"):
            reason = data.get("result") or data.get("subtype") or "unknown error"
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"claude error: {reason}")
        text = data.get("result")
        if not isinstance(text, str):
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error="claude JSON missing 'result' text")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        out_tokens = usage.get("output_tokens")
        tokens = out_tokens if isinstance(out_tokens, int) else None
        return LLMResponse(self.name, model, success=True, text=text, latency_ms=latency, tokens_used=tokens)
