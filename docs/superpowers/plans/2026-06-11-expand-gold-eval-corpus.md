# Expand Gold Eval Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the offline gold eval corpus to at least 25 realistic cases and fix any scoped behavior gaps those cases expose.

**Architecture:** Reuse the existing JSON corpus format and `jarvis_research.eval_corpus` runner. Add cases first, strengthen tests around corpus size/type coverage, then update query planning, ranking, or screening only when gold cases fail.

**Tech Stack:** Python standard library, existing JarvisResearch eval corpus, query planning, source policy, relevance, screening, and unittest.

---

### Task 1: Red Tests for Expanded Corpus

**Files:**
- Modify: `tests/test_eval_corpus.py`

- [x] **Step 1: Strengthen corpus tests**

Require at least 25 gold cases and require all supported case types: `query_plan`, `source_policy`, `ranking`, and `screening_label`.

- [x] **Step 2: Run red test**

Run: `python3 -m unittest tests.test_eval_corpus -v`
Expected: fail because the current corpus has 8 cases.

### Task 2: Expand Cases and Fix Gaps

**Files:**
- Modify: `eval_corpus/gold_cases.json`
- Modify as needed: `jarvis_research/query_planning.py`
- Modify as needed: `jarvis_research/relevance.py`
- Modify as needed: `jarvis_research/screening.py`
- Modify as needed: focused tests under `tests/`

- [x] **Step 1: Add gold cases**

Expand to at least 25 cases covering biomedical acronyms, source safety, ranking collisions, natural-language prompts, and label edge cases.

- [x] **Step 2: Run gold corpus tests**

Run: `python3 -m unittest tests.test_eval_corpus -v`
Expected: failures only where new cases expose current behavior gaps.

- [x] **Step 3: Add focused regression tests for behavior fixes**

For each code behavior change, add a focused test in the relevant existing test module before changing production code.

- [x] **Step 4: Implement scoped fixes**

Update the local module needed for each failure. Keep all behavior offline and deterministic.

- [x] **Step 5: Run focused green tests**

Run the relevant focused tests plus `python3 -m jarvis_research eval-suite run --suite gold`.

### Task 3: Verification and Finish

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-expand-gold-eval-corpus.md`

- [x] **Step 1: Run focused suite**

Run:

```bash
python3 -m unittest tests.test_eval_corpus tests.test_eval_suite tests.test_relevance tests.test_screening tests.test_query_planning tests.test_source_policy -v
python3 -m jarvis_research eval-suite run --suite gold
```

- [x] **Step 2: Run full suite**

Run: `python3 -m unittest discover -v`

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `expand-gold-eval-corpus`, fast-forward `main`, rerun the full suite on `main`, push, remove the temporary worktree, and delete the local branch.
