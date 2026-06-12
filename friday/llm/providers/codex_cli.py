from __future__ import annotations

import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from friday.llm._subprocess import CommandRunner, run_command
from friday.llm.types import LLMRequest, LLMResponse, ProviderStatus


class CodexCliProvider:
    """OpenAI models via the Codex CLI, billed against the user's ChatGPT
    subscription — never per-token API credits.

    This is Friday's *independent model family* (PLAN §6): with the composer on
    Claude, routing the verifier/critic here means no single model both writes
    and certifies a claim. ``run_command`` strips ``OPENAI_API_KEY`` from the
    child environment so auth stays on the subscription Codex is logged in with.

    Codex ``exec`` runs sandboxed (``read-only`` by default) and the final
    assistant message is captured via ``--output-last-message``, so Friday reads
    exactly the model's answer rather than scraping the agent transcript.
    """

    name = "codex_cli"

    def __init__(
        self,
        *,
        executable: str = "codex",
        runner: Optional[CommandRunner] = None,
        timeout: float = 240.0,
        sandbox: str = "read-only",
        extra_args: Sequence[str] = (),
    ) -> None:
        self.executable = executable
        self.runner: CommandRunner = runner or run_command
        self.timeout = timeout
        self.sandbox = sandbox
        self.extra_args = tuple(extra_args)

    def _resolve(self) -> Optional[str]:
        found = shutil.which(self.executable)
        if found:
            return found
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
            return ProviderStatus(self.name, False, reason=f"`codex --version` failed: {detail}")
        return ProviderStatus(self.name, True, models=(result.stdout.strip(),))

    def generate(self, request: LLMRequest, model: str) -> LLMResponse:
        exe = self._resolve()
        if exe is None:
            return LLMResponse(self.name, model, success=False, error=f"{self.executable!r} not found on PATH")

        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "last_message.txt"
            args: list[str] = [
                exe,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                self.sandbox,
                "--color",
                "never",
                "--output-last-message",
                str(out_path),
            ]
            if model:
                args += ["-m", model]
            args += list(self.extra_args)

            result = self.runner(args, stdin=prompt, timeout=self.timeout)
            latency = int((time.monotonic() - start) * 1000)

            if result.not_found:
                return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"{self.executable!r} not found")
            if result.timed_out:
                return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"codex timed out after {self.timeout}s")
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[:500]
                return LLMResponse(self.name, model, success=False, latency_ms=latency, error=f"codex exited {result.returncode}: {detail}")

            text = ""
            if out_path.exists():
                text = out_path.read_text(encoding="utf-8", errors="replace").strip()

        if not text:
            text = _last_message_from_stdout(result.stdout)
        if not text:
            return LLMResponse(self.name, model, success=False, latency_ms=latency, error="codex produced no final message")
        return LLMResponse(self.name, model, success=True, text=text, latency_ms=latency)


def _last_message_from_stdout(stdout: str) -> str:
    """Fallback when ``--output-last-message`` was not written: take the text
    after the last ``codex`` marker line in the human-readable transcript."""
    if not stdout.strip():
        return ""
    blocks = re.split(r"(?m)^\s*codex\s*$", stdout)
    candidate = blocks[-1] if len(blocks) > 1 else stdout
    return candidate.strip()
