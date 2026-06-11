# Label Review Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI-first label review workflow for quickly triaging agent labels, unlabeled high-relevance papers, and human overrides.

**Architecture:** Add a focused `friday.label_review` module that joins batch items, current screening labels, and smart review-queue rationale into ranked review rows. Extend `friday labels` with `review` and `set` actions while preserving existing list/export behavior.

**Tech Stack:** Python standard library, existing `FridayStore`, existing screening review queue, `unittest`.

---

### Task 1: Review Row Builder

**Files:**
- Create: `friday/label_review.py`
- Test: `tests/test_label_review.py`

- [x] **Step 1: Write failing review-row tests**

Add tests showing review rows include labeled and unlabeled allowed papers, prioritize smart queue candidates, expose queue reason/score, filter by `maybe`, `agent`, `unlabeled`, and `min_relevance`.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_label_review -v`
Expected: FAIL because `friday.label_review` does not exist.

- [x] **Step 3: Implement review module**

Add `build_label_review_rows(items, labels, only=None, min_relevance=0, limit=20)` returning dictionaries with label/source/confidence/relevance/title/source/queue fields.

- [x] **Step 4: Run review-row tests**

Run: `python3 -m unittest tests.test_label_review -v`
Expected: PASS.

### Task 2: CLI Review and Set Actions

**Files:**
- Modify: `friday/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Add tests for:
- `friday labels review --latest --only maybe`
- `friday labels review --latest --only unlabeled --min-relevance 60`
- `friday labels set --latest --source ... --label relevant --note ...`

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_review_filters_maybe_and_unlabeled tests.test_cli.CliTests.test_labels_set_applies_human_override -v`
Expected: FAIL because `labels review` and `labels set` are not implemented.

- [x] **Step 3: Implement parser and handlers**

Extend the existing `labels` command with `review` and `set` actions, filter options, and compact review output. Make `labels set` call `store.set_screening_label(... label_source="human")`.

- [x] **Step 4: Run focused CLI tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_review_filters_maybe_and_unlabeled tests.test_cli.CliTests.test_labels_set_applies_human_override tests.test_cli.CliTests.test_labels_export_latest_writes_jsonl -v`
Expected: PASS.

### Task 3: Verification and Merge

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-label-review-workflow.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 2: Update checklist**

Mark completed tasks in this plan.

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `label-review-workflow`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
