# Label Export Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export screening labels and paper metadata as a reusable training/evaluation dataset for future relevance classifiers.

**Architecture:** Add a focused `friday.label_export` module that flattens batch items, current screening labels, source-gate data, relevance metadata, and smart review-queue rationale into row dictionaries. Extend `friday labels` with an `export` action supporting `--latest`, `--batch-id`, or `--all`, plus JSONL and CSV renderers.

**Tech Stack:** Python standard library, existing SQLite-backed `FridayStore`, existing screening/review-queue utilities, `unittest`.

---

### Task 1: Export Row Builder

**Files:**
- Create: `friday/label_export.py`
- Test: `tests/test_label_export.py`

- [x] **Step 1: Write failing exporter tests**

Add tests that create batches with agent and human labels and assert rows include query, paper metadata, source-gate status, relevance metadata, gold vs weak labels, confidence/rationale/signals, and review queue reason for agent-labeled candidates.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_label_export -v`
Expected: FAIL because `friday.label_export` does not exist.

- [x] **Step 3: Implement exporter module**

Add `build_label_export_rows(store, batch_ids=None)`, `render_label_export_jsonl(rows)`, and `render_label_export_csv(rows)`.

- [x] **Step 4: Run exporter tests**

Run: `python3 -m unittest tests.test_label_export -v`
Expected: PASS.

### Task 2: CLI Export Command

**Files:**
- Modify: `friday/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Add tests for:
- `friday labels export --latest --format jsonl --output labels.jsonl`
- `friday labels export --all --format csv --output labels.csv`

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_export_latest_writes_jsonl tests.test_cli.CliTests.test_labels_export_all_writes_csv -v`
Expected: FAIL because `labels export` is not implemented.

- [x] **Step 3: Implement CLI parser and handler**

Extend the existing `labels` command with optional `export` action, `--all`, `--format jsonl|csv`, and `--output`. Preserve the current `labels --latest` and `labels --batch-id` listing behavior.

- [x] **Step 4: Run focused CLI tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_export_latest_writes_jsonl tests.test_cli.CliTests.test_labels_export_all_writes_csv tests.test_cli.CliTests.test_review_queue_command_lists_llm_candidates -v`
Expected: PASS.

### Task 3: Verification and Merge

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-label-export-loop.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 2: Update checklist**

Mark completed tasks in this plan.

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `label-export-loop`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
