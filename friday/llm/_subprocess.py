from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

# Stripped from every child process so the CLI providers authenticate with the
# user's subscription (OAuth / rolling usage window) and Friday can never
# silently fall back to per-token API billing. This is the enforcement behind
# the "use my subscription, not API tokens" requirement.
STRIPPED_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_API_KEY_PATH",
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    not_found: bool = False


class CommandRunner(Protocol):
    """Runs an argv and returns its result. Injectable so providers are tested
    without ever spawning a real subprocess."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        stdin: Optional[str] = None,
        timeout: float = 120.0,
    ) -> CommandResult: ...


def child_env() -> dict[str, str]:
    """A copy of the environment with token-billing credentials removed."""
    env = dict(os.environ)
    for key in STRIPPED_ENV_VARS:
        env.pop(key, None)
    return env


def _wrap_for_windows(args: list[str]) -> list[str]:
    # CreateProcess cannot execute .cmd/.bat shims directly (WinError 193);
    # route them through cmd.exe. Native .exe runs fine as-is.
    if os.name == "nt" and args and args[0].lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", *args]
    return args


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value or "")


def run_command(
    args: Sequence[str],
    *,
    stdin: Optional[str] = None,
    timeout: float = 120.0,
) -> CommandResult:
    """Run a command with subscription-only auth. Never raises: process-launch
    and timeout failures come back as a ``CommandResult`` with the relevant flag
    set, so providers can return a structured ``LLMResponse(success=False)``."""
    argv = _wrap_for_windows(list(args))
    try:
        proc = subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env(),
        )
    except FileNotFoundError as exc:
        return CommandResult(returncode=127, stdout="", stderr=str(exc), not_found=True)
    except OSError as exc:
        return CommandResult(returncode=126, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            timed_out=True,
        )
    return CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
