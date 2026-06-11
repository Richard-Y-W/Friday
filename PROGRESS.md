# Friday — Implementation Progress

Tracks the build plan in [PLAN.md](PLAN.md). Phase numbers match §11 of the plan.

| Phase | Description | Status |
|---|---|---|
| — | Migrate JarvisResearch → Friday, rebrand, green baseline | ✅ done |
| — | Close leaked SQLite connections (`_connect` context manager) | ✅ done |
| **0** | LLM provider abstraction (`friday/llm/`) | ✅ done |
| 1 | Sandbox the PDF parser | ▢ todo |
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

## Next: Phase 1

Move the PDF parse into an isolated subprocess (no network, timeout, memory cap, crash isolation) before any LLM leaf reads body text. This is the safety prerequisite for Phase 2+.
