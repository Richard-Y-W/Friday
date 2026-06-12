# Friday — Implementation Progress

Tracks the build plan in [PLAN.md](PLAN.md). Phase numbers match §11 of the plan.

| Phase | Description | Status |
|---|---|---|
| — | Migrate JarvisResearch → Friday, rebrand, green baseline | ✅ done |
| — | Close leaked SQLite connections (`_connect` context manager) | ✅ done |
| **0** | LLM provider abstraction (`friday/llm/`) | ✅ done |
| **0.1** | Subscription-backed providers (`claude_cli`/`codex_cli`) + role config + `friday llm` | ✅ done |
| 1 | Sandbox the PDF parser | ◐ in progress |
| 2 | Evidence store + decompose/type | ▢ todo |
| 3 | Discourse planner + claim/connective split | ▢ todo |
| 4 | Style-aware composer + style packs (biomed first) | ▢ todo |
| 5 | Tier A + Tier B faithfulness gate | ▢ todo |
| 6 | Tier C critics (faithfulness + prose-quality) + revise loop | ▢ todo |
| 7 | Trust score + verdict→action | ▢ todo |
| 8 | Human feedback capture (`friday review-draft`) | ▢ todo |
| 9 | Feedback flywheel (eval-gated) | ▢ todo |
| 10 | Refactor `cli.py` monolith into a pipeline layer | ▢ ongoing |

## Phase 0 notes (done)

`friday/llm/` — a stdlib-only, dependency-free port of Tactician's provider/router design:

- `types.py` — `LLMRequest`, `LLMResponse`, `ProviderStatus`, `ModelConfig`, `Provider` protocol, `Role`.
- `parse.py` — `strip_markdown_fences` / `extract_json` (robust JSON from fenced/prose-wrapped model output).
- `providers/` — `OllamaProvider` (local, zero-token), `OpenAIProvider`, `AnthropicProvider`. HTTP transport is injectable (`opener`) so tests never touch the network.
- `router.py` — `ModelRouter`: role → provider+model, availability checks, and **graceful failure** (`success=False`, never raises) when a role is unconfigured or its provider is down. This is what lets later phases keep the "runs without an LLM" fallback.

**Default posture:** no roles configured → every `generate()` returns a structured failure → the deterministic spine is unaffected and zero-token. Local Ollama keeps style/critic work free; different providers per role serve the "independent model family" invariant (PLAN §6).

Tests: `tests/test_llm_parse.py`, `tests/test_llm_router.py`, `tests/test_llm_providers.py` (26 cases, no network, no SQLite).

## Phase 0.1 notes (done) — subscription providers, no API tokens

Friday's LLM leaves run against the user's **Claude / ChatGPT subscriptions** (the rolling usage window), never per-token API credits. New since Phase 0:

- `friday/llm/providers/claude_cli.py` — `ClaudeCliProvider`: shells out to the `claude` CLI in print mode (`claude -p --output-format json`), parses the `result` field. Prompt goes over **stdin** (no argv length/quoting limits). Built-in tools denied (`--disallowedTools`) per PLAN §12.
- `friday/llm/providers/codex_cli.py` — `CodexCliProvider`: `codex exec --sandbox read-only --output-last-message <file>`, the independent model family for verify/critique (§6).
- `friday/llm/_subprocess.py` — injectable `CommandRunner` (tests never spawn a real process) and `run_command`, which **strips `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` from the child env** — the hard guarantee that Friday can never silently fall back to token billing. Handles Windows `.cmd` shims via `cmd /c`.
- `friday/llm/config.py` — per-role wiring from settings. Default: composer→`claude_cli` (sonnet), verifier/critic→`codex_cli`, screener/extractor→`none` (high-volume work stays deterministic, §1).
- `friday/settings.py` — new `llm` section (`<role>_provider`/`<role>_model`).
- `friday llm status` / `friday llm test --role <role>` — inspect wiring + run one live generation to confirm subscription auth.

Tests: `tests/test_llm_cli_providers.py`, `tests/test_llm_config.py`, `LlmCommandTests` in `tests/test_cli.py`. Verified live: `friday llm test --role composer` generated on the Claude subscription with no API key set.

**Login:** `claude` is already authenticated (subscription). For the verifier/critic run `codex login` (ChatGPT subscription) when ready; until then those roles return a structured failure and the deterministic spine is unaffected.

## Phase 1 notes (in progress) — sandboxed PDF parser

`extract_pdf_text_pages` (`friday/pdf_ingestion.py`) now treats the PDF as
attacker-controlled input and parses it out-of-process under hard limits:

- **Scrubbed environment** — the parser inherits only a library/font allowlist
  (`_PARSER_ENV_ALLOWLIST`); Friday's API keys/tokens/network config never reach
  it. (`_scrubbed_parser_env`)
- **Wall-clock timeout** + **bounded page count** (`PDF_PARSE_MAX_PAGES`) +
  **capped output size** (`PDF_PARSE_MAX_OUTPUT_BYTES`, read via `_read_capped`)
  so a PDF that explodes into a huge text stream can't exhaust memory downstream.
- **POSIX memory/CPU rlimits** via `preexec_fn` (`RLIMIT_AS` + `RLIMIT_CPU`).
- **stdin closed** (`DEVNULL`) and crash → `RuntimeError` → caller records a
  blocked artifact.

Verified live: 13 pages extracted from `PLAN.pdf` through the hardened path.
Tests: `tests/test_pdf_sandbox.py`.

**Remaining for Phase 1:** hard memory caps on **Windows** (POSIX rlimits no-op
there) need a Job Object wrapper — the one platform-specific piece still open.
Network isolation is currently posture-only (pdftotext doesn't network, and the
env scrub removes proxy/credential vars); a namespace/Job-Object network block is
the stronger follow-up. Then Phase 2 (evidence store + `parse_confidence` +
decompose/type) can build on a trusted parse.
