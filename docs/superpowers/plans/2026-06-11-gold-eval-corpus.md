# Gold Eval Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated JSON-backed `gold` eval suite for realistic scanner behavior checks.

**Architecture:** Create `friday.eval_corpus` to load `eval_corpus/gold_cases.json` and convert each case into existing `EvalCase` objects. Extend `friday.eval_suite` with the `gold` suite and update CLI/CI tests to prove the suite is available and gated.

**Tech Stack:** Python standard library JSON/pathlib/tempfile/unittest, existing Friday query planning, source policy, relevance, screening, and storage APIs.

---

### Task 1: Corpus Loader Tests

**Files:**
- Create: `tests/test_eval_corpus.py`
- Create later: `friday/eval_corpus.py`
- Create later: `eval_corpus/gold_cases.json`

- [x] **Step 1: Write failing tests**

Create tests that import `load_gold_eval_cases`, `build_gold_eval_cases`, and `GOLD_CORPUS_PATH`. Assert the JSON corpus exists, has `schema_version: 1.0`, has at least seven cases, has stable `case_id` values starting with `gold.`, and converts to `EvalCase` objects with `suite == "gold"`.

- [x] **Step 2: Run red test**

Run: `python3 -m unittest tests.test_eval_corpus -v`
Expected: fail because `friday.eval_corpus` does not exist.

- [x] **Step 3: Implement corpus file and loader**

Add `eval_corpus/gold_cases.json` with query, source, ranking, and screening label cases. Add `friday/eval_corpus.py` with `load_gold_eval_cases()`, `build_gold_eval_cases()`, and evaluator helpers.

- [x] **Step 4: Run green test**

Run: `python3 -m unittest tests.test_eval_corpus -v`
Expected: pass.

### Task 2: Eval Suite and CLI Integration

**Files:**
- Modify: `friday/eval_suite.py`
- Modify: `tests/test_eval_suite.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing tests**

Add tests proving `run_eval_suite("gold")` passes, cases are all suite `gold`, CLI list prints `gold`, and `friday eval-suite run --suite gold --format json` returns a passing `eval_suite_report`.

- [x] **Step 2: Run red tests**

Run focused eval-suite and CLI tests. Expected: fail because `gold` is not an available suite.

- [x] **Step 3: Implement suite integration**

Update `available_eval_suites()` to include `gold`; import and append `build_gold_eval_cases()` to default cases.

- [x] **Step 4: Run green tests**

Run the focused eval-suite, eval-corpus, and CLI tests. Expected: pass.

### Task 3: CI Gate Update

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_ci_workflow.py`

- [x] **Step 1: Write failing workflow test**

Assert the CI workflow includes `python3 -m friday eval-suite run --suite gold`.

- [x] **Step 2: Run red test**

Run: `python3 -m unittest tests.test_ci_workflow -v`
Expected: fail because CI does not run the gold suite yet.

- [x] **Step 3: Add CI command**

Add the gold suite command to `.github/workflows/ci.yml`.

- [x] **Step 4: Run green test and real gold command**

Run `python3 -m unittest tests.test_ci_workflow -v` and `python3 -m friday eval-suite run --suite gold`.
Expected: pass.

### Task 4: Final Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-gold-eval-corpus.md`

- [x] **Step 1: Run full test suite**

Run: `python3 -m unittest discover -v`
Expected: pass.

- [x] **Step 2: Commit, merge, push, cleanup**

Commit on `gold-eval-corpus`, fast-forward `main`, rerun the full suite on `main`, push, remove the temporary worktree, and delete the local branch.
