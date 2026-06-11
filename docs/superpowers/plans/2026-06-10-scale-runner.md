# Scale Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable `research-run` command that screens many scholarly records, labels candidates, deep-reads only the selected top papers, and exports run-level artifacts.

**Architecture:** Reuse the existing scholarly discovery, source gate, ranking, auto-labeling, deep-read, report, passport, and rejection-log modules. Add a small persistent run ledger in SQLite and a CLI orchestration layer that records status transitions and can resume an existing run by ID.

**Tech Stack:** Python standard library, SQLite, `unittest`, existing Friday modules.

---

### Task 1: Research Run Storage

**Files:**
- Modify: `friday/storage.py`
- Test: `tests/test_storage.py`

- [x] **Step 1: Write the failing test**

Add a storage test that creates a research run, attaches a batch, updates status/count fields, lists runs newest-first, and loads the same run by ID.

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_storage.StorageTests.test_creates_and_updates_research_run_ledger -v`
Expected: FAIL because `FridayStore.create_research_run` does not exist.

- [x] **Step 3: Write minimal implementation**

Add `ResearchRunRecord`, a `research_runs` SQLite table, row conversion, and store methods for create/get/list/update/sync counts.

- [x] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_storage.StorageTests.test_creates_and_updates_research_run_ledger -v`
Expected: PASS.

### Task 2: Run Summary Artifact

**Files:**
- Modify: `friday/research_artifacts.py`
- Test: `tests/test_research_artifacts.py`

- [x] **Step 1: Write the failing test**

Add a test that builds a run summary and verifies run config, status, batch counts, screening labels, artifact counts, source policy, and repro lock.

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_research_artifacts.ResearchArtifactsTests.test_research_run_summary_records_status_counts_and_policy -v`
Expected: FAIL because `build_research_run_summary` does not exist.

- [x] **Step 3: Write minimal implementation**

Add `build_research_run_summary(store, run_id, data_dir=None)` using existing batch passport ingredients where possible.

- [x] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_research_artifacts.ResearchArtifactsTests.test_research_run_summary_records_status_counts_and_policy -v`
Expected: PASS.

### Task 3: `research-run` CLI

**Files:**
- Modify: `friday/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write the failing tests**

Add tests that:
- `research-run "MALDI AMR"` creates a run and batch, screens candidates, applies agent labels, deep-reads the top paper, and writes report/passport/rejection-log/run-summary artifacts.
- `research-run --resume-run <run_id> --deep-read-limit 2` reuses the existing batch, avoids duplicate screening rows, and deep-reads the next ranked paper.

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_run_creates_ledger_labels_deep_reads_and_writes_artifacts tests.test_cli.CliTests.test_research_run_resume_deep_reads_next_ranked_candidate -v`
Expected: FAIL because the parser does not know `research-run`.

- [x] **Step 3: Write minimal implementation**

Add `research-run` parser options, dispatch, status transitions, run creation/resume, idempotent discovery, heuristic labels, optional LLM review, deep reads, report/passport/rejection-log/run-summary output, and concise CLI status output.

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_run_creates_ledger_labels_deep_reads_and_writes_artifacts tests.test_cli.CliTests.test_research_run_resume_deep_reads_next_ranked_candidate -v`
Expected: PASS.

### Task 4: Verification and Integration

**Files:**
- Modify: `docs/superpowers/plans/2026-06-10-scale-runner.md`

- [x] **Step 1: Run focused tests**

Run: `python3 -m unittest tests.test_storage tests.test_research_artifacts tests.test_cli -v`
Expected: PASS.

- [x] **Step 2: Run full test suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 3: Update this checklist**

Mark completed tasks and leave the plan as a record of the implementation.

- [ ] **Step 4: Commit, merge, and push**

Commit on `scale-runner`, merge into `main`, push `main`, then remove the temporary worktree and feature branch.
