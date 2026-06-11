# LLM Auto Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional LLM-backed screening labels while keeping the no-token heuristic labeler as the default.

**Architecture:** Keep heuristic label logic in `screening.py`, add a focused `llm_labeling.py` provider module for prompt construction, strict JSON validation, and OpenAI Responses API transport. CLI and natural-query flows select the provider from settings and preserve human-label overrides.

**Tech Stack:** Python standard library, SQLite-backed existing storage, OpenAI Responses API via HTTPS when `auto_label.provider=llm` is selected.

---

### Task 1: Settings And CLI Provider Selection

**Files:**
- Modify: `friday/settings.py`
- Modify: `friday/cli.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_cli.py`

- [x] Add settings for `auto_label.model`, `auto_label.api_base_url`, and `auto_label.api_key_env`.
- [x] Add CLI flags for `auto-label --provider` and `auto-label --model`.
- [x] Pass provider/model settings into natural query auto-labeling.
- [x] Verify unknown providers are rejected before labels are written.

### Task 2: LLM Label Provider

**Files:**
- Create: `friday/llm_labeling.py`
- Modify: `friday/screening.py`
- Test: `tests/test_llm_labeling.py`
- Test: `tests/test_screening.py`

- [x] Build a metadata-only payload from `BatchItemRecord`.
- [x] Send strict JSON-schema Responses API requests only when provider is `llm`.
- [x] Parse output text/refusal safely and validate `label`, `confidence`, `rationale`, `evidence_terms`, and `exclusion_reason`.
- [x] Store model/provider metadata inside the label `signals` string.
- [x] Fall back to no label on client/config errors instead of overwriting human labels.

### Task 3: Docs And Verification

**Files:**
- Modify: `docs/specs/2026-06-07-friday-scanner-agent-design.md`

- [x] Document that LLM labeling is optional, token-using, JSON-only, metadata-only, and disabled by default.
- [x] Run `python3 -m unittest discover -v`.
- [x] Run `git diff --check`.
