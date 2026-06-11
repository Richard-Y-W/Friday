# Natural Corpus Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let natural Jarvis queries use imported Zotero/folder/Obsidian corpus JSON when local corpus matches are strong enough, while preserving the existing scholarly discovery fallback.

**Architecture:** Add a focused `jarvis_research.corpus_routing` module that loads configured corpus files, scores entries with deterministic metadata overlap, and renders a corpus route decision. Extend settings with corpus paths and routing thresholds, then branch `_handle_natural_language_query` to emit a corpus report when the router returns enough matches.

**Tech Stack:** Python standard library, existing JSON corpus artifacts, existing CLI/settings patterns, `unittest`.

---

### Task 1: Corpus Routing Module

**Files:**
- Create: `jarvis_research/corpus_routing.py`
- Test: `tests/test_corpus_routing.py`

- [x] **Step 1: Write failing router tests**

Add tests for loading corpus JSON files, ranking entries by query overlap, rejecting weak matches, and ignoring missing/invalid corpus paths.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_corpus_routing -v`
Expected: FAIL because `jarvis_research.corpus_routing` does not exist.

- [x] **Step 3: Implement router**

Create `CorpusRouteMatch`, `CorpusRouteResult`, and `route_corpus_query(query, corpus_paths, min_score, min_matches, limit)`. Score title, abstract, tags, venue, authors, DOI, and citation key with deterministic token overlap. Return `should_use_corpus=True` only when at least `min_matches` entries meet `min_score`.

- [x] **Step 4: Run router tests**

Run: `python3 -m unittest tests.test_corpus_routing -v`
Expected: PASS.

### Task 2: Settings and Natural Query CLI Integration

**Files:**
- Modify: `jarvis_research/settings.py`
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI/settings tests**

Add tests showing `/settings` exposes corpus routing defaults, `settings set corpus.paths ...` persists corpus paths, and a natural query routes to corpus output without calling the live discoverer when local corpus matches are strong.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_settings tests.test_cli.CliTests.test_natural_query_routes_to_configured_corpus_when_relevant -v`
Expected: FAIL because corpus settings and natural routing are not implemented.

- [x] **Step 3: Implement settings**

Add a `corpus` settings section with `paths`, `min_score`, `min_matches`, and `limit`. Keep values string-compatible with the existing `settings set` command by parsing `paths` as a path-list string inside the router call.

- [x] **Step 4: Implement natural routing**

In `_handle_natural_language_query`, load corpus settings first. If `route_corpus_query` returns `should_use_corpus`, print `Natural query route: corpus`, render a concise Markdown/JSON/text corpus report, write `--output` if supplied, and skip discovery/deep-read. If the corpus route does not qualify, continue through existing discovery behavior.

- [x] **Step 5: Run focused CLI/settings tests**

Run: `python3 -m unittest tests.test_settings tests.test_cli.CliTests.test_natural_query_routes_to_configured_corpus_when_relevant tests.test_cli.CliTests.test_unknown_command_routes_to_natural_research_using_settings -v`
Expected: PASS.

### Task 3: Verification and Merge

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-natural-corpus-routing.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 2: Update checklist**

Mark completed tasks in this plan.

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `natural-corpus-routing`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
