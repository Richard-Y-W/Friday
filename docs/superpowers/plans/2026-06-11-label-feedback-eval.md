# Label Feedback Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add label evaluation, threshold recommendations, and stronger human-feedback weighting.

**Architecture:** Create `jarvis_research.label_eval` for pure evaluation logic. Extend `jarvis labels` with an `eval` action for text/JSON output. Update screening recommendation and deep-read ordering to prefer human feedback while preserving existing source-gate and label safety behavior.

**Tech Stack:** Python standard library, existing SQLite-backed `JarvisStore`, existing screening label records, `unittest`.

---

### Task 1: Label Evaluation Module

**Files:**
- Create: `jarvis_research/label_eval.py`
- Test: `tests/test_label_eval.py`

- [x] **Step 1: Write failing eval tests**

Add tests that build human labels with notes like `agent=relevant confidence=0.91`, then assert human counts, confusion matrix, precision/recall, top disagreements, and threshold recommendations.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_label_eval -v`
Expected: FAIL because `jarvis_research.label_eval` does not exist.

- [x] **Step 3: Implement eval module**

Add `build_label_evaluation(items, labels)` and helper functions for parsing prior agent metadata, computing metrics, and building recommendations.

- [x] **Step 4: Run eval tests**

Run: `python3 -m unittest tests.test_label_eval -v`
Expected: PASS.

### Task 2: CLI Eval Action

**Files:**
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Add tests for:
- `jarvis labels eval --latest`
- `jarvis labels eval --latest --format json`

- [x] **Step 2: Run CLI tests to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_eval_outputs_feedback_summary tests.test_cli.CliTests.test_labels_eval_json_outputs_structured_report -v`
Expected: FAIL because `labels eval` is not implemented.

- [x] **Step 3: Implement parser and handler**

Add `eval` to labels action choices, route to `_handle_labels_eval`, and print compact text or JSON.

- [x] **Step 4: Run focused CLI tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_labels_eval_outputs_feedback_summary tests.test_cli.CliTests.test_labels_eval_json_outputs_structured_report tests.test_cli.CliTests.test_labels_review_filters_maybe_and_unlabeled -v`
Expected: PASS.

### Task 3: Feedback Weighting

**Files:**
- Modify: `jarvis_research/screening.py`
- Test: `tests/test_screening.py`

- [x] **Step 1: Write failing screening tests**

Add tests proving human relevant feedback outranks agent-only relevant feedback in recommendations and human maybe deep-read ordering outranks agent maybe ordering.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_screening.ScreeningTests.test_recommendations_weight_human_feedback_more_than_agent_feedback tests.test_screening.ScreeningTests.test_deep_read_order_prefers_human_maybe_over_agent_maybe -v`
Expected: FAIL under current equal-weight label behavior.

- [x] **Step 3: Implement feedback weighting**

Update recommendation profiles and label buckets so human labels have stronger influence and human maybe records rank ahead of agent maybe records.

- [x] **Step 4: Run screening tests**

Run: `python3 -m unittest tests.test_screening -v`
Expected: PASS.

### Task 4: Verification and Integration

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-label-feedback-eval.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [ ] **Step 2: Commit, merge, push, cleanup**

Commit on `label-feedback-eval`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
