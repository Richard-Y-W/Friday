# Run Summary Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `run-summary` command that summarizes the latest research run or batch and recommends next actions.

**Architecture:** Add a focused `friday.run_summary` module for target resolution, structured dashboard construction, and text rendering. Extend `friday.cli` with a thin `run-summary` command that supports `--latest`, `--run-id`, `--batch-id`, `--format text|json`, and `--limit`.

**Tech Stack:** Python standard library, existing SQLite-backed `FridayStore`, existing `label_eval`, `label_review`, `research_artifacts`, `screening`, and `unittest`.

---

### Task 1: Dashboard Builder

**Files:**
- Create: `friday/run_summary.py`
- Test: `tests/test_run_summary.py`

- [x] **Step 1: Write failing builder tests**

Add tests that create a research run, batch items, agent/human labels, a failed PDF artifact, a stored PDF artifact, and evidence. Assert the dashboard includes target IDs, counts, high-relevance unlabeled items, failed PDFs, label evaluation, and next commands.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_run_summary -v`
Expected: FAIL because `friday.run_summary` does not exist.

- [x] **Step 3: Implement dashboard builder**

Add `build_run_summary_dashboard(store, latest=False, run_id=None, batch_id=None, limit=5)` and target/count/attention helpers.

- [x] **Step 4: Run builder tests**

Run: `python3 -m unittest tests.test_run_summary -v`
Expected: PASS.

### Task 2: Text Renderer and CLI

**Files:**
- Modify: `friday/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Add tests for:
- `friday run-summary --latest`
- `friday run-summary --latest --format json`
- `friday run-summary --latest` when no run or batch exists

- [x] **Step 2: Run CLI tests to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_run_summary_latest_outputs_dashboard tests.test_cli.CliTests.test_run_summary_json_outputs_structured_dashboard tests.test_cli.CliTests.test_run_summary_latest_reports_missing_target -v`
Expected: FAIL because `run-summary` is not implemented.

- [x] **Step 3: Implement parser and handler**

Add the command name, parser, `_handle_run_summary`, JSON output, text output, and clean errors.

- [x] **Step 4: Run focused CLI tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_run_summary_latest_outputs_dashboard tests.test_cli.CliTests.test_run_summary_json_outputs_structured_dashboard tests.test_cli.CliTests.test_run_summary_latest_reports_missing_target -v`
Expected: PASS.

### Task 3: Verification and Integration

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-run-summary-dashboard.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [ ] **Step 2: Commit, merge, push, cleanup**

Commit on `run-summary-dashboard`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
