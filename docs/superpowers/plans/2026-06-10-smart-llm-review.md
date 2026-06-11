# Smart LLM Review Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spend LLM labeling calls on the papers most worth reviewing instead of blindly reviewing the first N batch items.

**Architecture:** Add a deterministic review-queue scorer in `jarvis_research.screening` that ranks allowed non-human-labeled papers by label uncertainty, high-relevance conflicts, and source diversity. Reuse that queue in a new `jarvis review-queue` CLI command and in `research-run --auto-label-provider llm --llm-review-limit N`, then export queue rationale into run artifacts.

**Tech Stack:** Python standard library, SQLite-backed existing storage, existing LLM label client abstraction, `unittest`.

---

### Task 1: Screening Review Queue

**Files:**
- Modify: `jarvis_research/screening.py`
- Test: `tests/test_screening.py`

- [x] **Step 1: Write failing queue tests**

Add tests that build a mixed batch and verify the queue:
- excludes blocked sources
- excludes human-labeled items
- prioritizes high-relevance `maybe`
- prioritizes high-relevance `irrelevant`
- includes top unlabeled candidates
- avoids filling the queue with duplicate domains/providers when there are diverse alternatives
- returns rationale text and scores

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_screening.ScreeningTests.test_llm_review_queue_prioritizes_borderline_conflicts_and_diversity -v`
Expected: FAIL because `build_llm_review_queue` does not exist.

- [x] **Step 3: Implement queue builder**

Add `LlmReviewQueueItem` and `build_llm_review_queue(items, labels, limit)` to `screening.py`. The implementation should be deterministic and metadata-only.

- [x] **Step 4: Run queue tests**

Run: `python3 -m unittest tests.test_screening -v`
Expected: PASS.

### Task 2: Queue-Aware Auto Labeling

**Files:**
- Modify: `jarvis_research/screening.py`
- Test: `tests/test_screening.py`

- [x] **Step 1: Write failing LLM selection test**

Add a test showing `auto_label_batch_items(... provider="llm", review_queue=[...])` sends only queued items to the LLM client, in queue order.

- [x] **Step 2: Run test to verify failure**

Run: `python3 -m unittest tests.test_screening.ScreeningTests.test_llm_auto_label_uses_review_queue_order -v`
Expected: FAIL because `auto_label_batch_items` does not accept a queue.

- [x] **Step 3: Implement queue input**

Extend `auto_label_batch_items` with optional `review_queue`. For `provider="llm"`, use queue item sources to select batch items. Keep current first-N behavior as fallback when no queue is passed.

- [x] **Step 4: Run screening tests**

Run: `python3 -m unittest tests.test_screening -v`
Expected: PASS.

### Task 3: CLI Preview

**Files:**
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI preview test**

Add a test for `jarvis review-queue --latest --limit 3` that prints queued papers with score, reason, label, confidence, source, and title.

- [x] **Step 2: Run test to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_review_queue_command_lists_llm_candidates -v`
Expected: FAIL because `review-queue` is not a command.

- [x] **Step 3: Implement command**

Add `review-queue` to `COMMAND_NAMES`, parser setup with `--batch-id`, `--latest`, `--limit`, and a handler that calls `build_llm_review_queue`.

- [x] **Step 4: Run CLI preview test**

Run: `python3 -m unittest tests.test_cli.CliTests.test_review_queue_command_lists_llm_candidates -v`
Expected: PASS.

### Task 4: Research Run Integration and Artifact Export

**Files:**
- Modify: `jarvis_research/cli.py`
- Modify: `jarvis_research/research_artifacts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_research_artifacts.py`

- [x] **Step 1: Write failing integration tests**

Add tests that:
- `research-run --auto-label-provider llm --llm-review-limit 2` reviews queued candidates, not first N items.
- run summary and batch passport include `llm_review_queue` rows with score, reason, source, title, label, and confidence.

- [x] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_cli.CliTests.test_research_run_llm_uses_smart_review_queue tests.test_research_artifacts.ResearchArtifactTests.test_run_artifacts_include_llm_review_queue -v`
Expected: FAIL because integration/export is missing.

- [x] **Step 3: Implement integration/export**

In `research-run`, build the queue after heuristic labels and pass it to the LLM auto-label call. Add queue export helpers in `research_artifacts.py` so run summary and passport can include queue rationale.

- [x] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_screening tests.test_cli tests.test_research_artifacts -v`
Expected: PASS.

### Task 5: Verification and Merge

**Files:**
- Modify: `docs/superpowers/plans/2026-06-10-smart-llm-review.md`

- [x] **Step 1: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS.

- [x] **Step 2: Update checklist**

Mark completed tasks.

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `smart-llm-review`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
