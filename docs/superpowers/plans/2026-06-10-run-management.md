# Run Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make research runs easier to find, resume, and audit by adding run listing, latest-run resume, and default run artifact folders.

**Architecture:** Extend the existing `research-run` CLI orchestration without changing discovery, source gating, ranking, labeling, or PDF ingestion behavior. Reuse the existing SQLite run ledger and artifact builders, adding only CLI routing and deterministic output path helpers.

**Tech Stack:** Python standard library, SQLite, `unittest`, existing JarvisResearch modules.

---

### Task 1: List Research Runs

**Files:**
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write the failing test**

Add a CLI test that creates two research runs, calls `research-runs`, and verifies it prints recent `run_...` IDs with status, query, screened/deep-read counts, and batch IDs.

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_runs_lists_recent_run_ledger -v`
Expected: FAIL because `research-runs` is not a recognized command.

- [x] **Step 3: Write minimal implementation**

Add `research-runs` to `COMMAND_NAMES`, parser setup, dispatch, and a `_handle_research_runs(store)` function that uses `store.list_research_runs()`.

- [x] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_runs_lists_recent_run_ledger -v`
Expected: PASS.

### Task 2: Resume Latest Run

**Files:**
- Modify: `jarvis_research/storage.py`
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing tests**

Add storage coverage for `latest_research_run()`, then add a CLI test showing `research-run --latest --deep-read-limit 2` resumes the newest run and deep-reads the next candidate without needing a run ID.

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_storage.StorageTests.test_latest_research_run_returns_newest_run tests.test_cli.CliTests.test_research_run_latest_resumes_newest_run -v`
Expected: FAIL because `latest_research_run` and `--latest` do not exist.

- [x] **Step 3: Write minimal implementation**

Add `JarvisStore.latest_research_run()`, add `--latest` to `research-run`, and route it through the same resume path as `--resume-run`.

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_storage.StorageTests.test_latest_research_run_returns_newest_run tests.test_cli.CliTests.test_research_run_latest_resumes_newest_run -v`
Expected: PASS.

### Task 3: Automatic Run Artifact Folder

**Files:**
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write the failing test**

Add a CLI test that runs `research-run` without `--output`, `--passport`, `--rejection-log`, or `--run-summary`, then verifies files are written under `.jarvis/runs/<run_id>/`.

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_run_writes_default_artifact_folder -v`
Expected: FAIL because the command prints to stdout and does not create a run folder.

- [x] **Step 3: Write minimal implementation**

Add a helper that builds default paths after the run ID exists:
- `.jarvis/runs/<run_id>/report.<md|txt|json>`
- `.jarvis/runs/<run_id>/passport.json`
- `.jarvis/runs/<run_id>/rejection-log.json`
- `.jarvis/runs/<run_id>/run-summary.json`

Use explicit CLI paths when provided.

- [x] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_run_writes_default_artifact_folder -v`
Expected: PASS.

### Task 4: Verification and Integration

**Files:**
- Modify: `docs/superpowers/plans/2026-06-10-run-management.md`

- [x] **Step 1: Run focused tests**

Run: `python3 -m unittest tests.test_storage tests.test_cli -v`
Expected: PASS.

- [x] **Step 2: Run full test suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 3: Update this checklist**

Mark completed tasks.

- [ ] **Step 4: Commit, merge, push, and clean up**

Commit on `run-management`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
