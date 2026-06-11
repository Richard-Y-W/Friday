# Acronym Resolver V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-off acronym conditionals with a generic, registry-backed acronym resolver that can resolve known acronyms and audit unknown acronyms without hallucinating meanings.

**Architecture:** Add `friday/acronyms.py` as the acronym registry and resolver. Refactor `friday/query_planning.py` to consume resolver output, preserve unknown acronyms, and keep existing query expansion behavior. Add gold eval cases and focused unit tests for biomedical, ML/NLP, computational, and unknown acronyms.

**Tech Stack:** Python standard library dataclasses, existing `unittest` suite, existing Friday gold eval corpus.

---

### Task 1: Registry Resolver Red Tests

**Files:**
- Create: `tests/test_acronyms.py`
- Modify: `tests/test_query_planning.py`

- [x] **Step 1: Add failing resolver tests**

Create tests asserting:
- `detect_acronyms("PCR CNN xyz")` returns `("PCR", "CNN")` but not lowercase `xyz`.
- `resolve_acronyms("PCR assay")` resolves PCR to polymerase chain reaction.
- `resolve_acronyms("CNN image classification")` resolves CNN to convolutional neural network.
- `resolve_acronyms("ABC novel biomarker")` returns ABC as unresolved.
- `resolve_acronyms("AMR parsing")` still chooses abstract meaning representation.
- `resolve_acronyms("MALDI AMR")` still chooses antimicrobial resistance.

- [x] **Step 2: Add failing query-planning tests**

Add tests asserting:
- `plan_query("PCR assay diagnostic sensitivity")` has intent `biomedical` and expands PCR.
- `plan_query("CNN image classification")` has intent `ml` and expands CNN.
- `plan_query("SVM classifier feature selection")` has intent `ml` and expands SVM.
- `plan_query("XYZ biomarker discovery")` records unresolved XYZ while keeping the original query.

- [x] **Step 3: Run red tests**

Run: `python3 -m unittest tests.test_acronyms tests.test_query_planning -v`

Expected: fail because `friday.acronyms` does not exist and query planning has no generic resolver yet.

### Task 2: Resolver Implementation

**Files:**
- Create: `friday/acronyms.py`
- Modify: `friday/query_planning.py`

- [x] **Step 1: Implement acronym registry**

Create dataclasses `AcronymSense` and `AcronymResolution`, a registry with AMR, AST, CNN, CRE, ESBL, LLM, MDR, MIC, NLP, PCR, and SVM, and helpers `detect_acronyms`, `resolve_acronyms`.

- [x] **Step 2: Refactor query planning**

Replace acronym-specific conditionals with resolver output. Convert resolver records to existing `ResolvedAcronym` records so downstream code remains compatible.

- [x] **Step 3: Keep expansion behavior stable**

Continue direct acronym replacement for known meanings. Keep AMR-specific biomedical/NLP/computational expansion templates. Add generic registry expansion templates for PCR, CNN, SVM, LLM, and NLP.

- [x] **Step 4: Run focused tests**

Run: `python3 -m unittest tests.test_acronyms tests.test_query_planning -v`

Expected: pass.

### Task 3: Gold Eval Coverage

**Files:**
- Modify: `eval_corpus/gold_cases.json`
- Modify: `tests/test_eval_corpus.py` if needed

- [x] **Step 1: Add gold cases**

Add query-plan cases for PCR, CNN, SVM, LLM/NLP, and an unknown acronym. The unknown case should expect `intent="unknown"` and `unresolved_acronyms_contains`.

- [x] **Step 2: Extend gold evaluator if needed**

If the gold case needs to assert unresolved acronyms, add support for `unresolved_acronyms_contains` in `friday/eval_corpus.py`.

- [x] **Step 3: Run gold eval**

Run: `python3 -m friday eval-suite run --suite gold`

Expected: all cases pass.

### Task 4: Verification and Finish

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-acronym-resolver-v2.md`

- [x] **Step 1: Run focused suite**

Run:

```bash
python3 -m unittest tests.test_acronyms tests.test_query_planning tests.test_eval_corpus tests.test_eval_suite -v
python3 -m friday eval-suite run --suite gold
```

- [x] **Step 2: Run full suite**

Run: `python3 -m unittest discover -v`

- [ ] **Step 3: Commit, merge, push, cleanup**

Commit on `acronym-resolver-v2`, fast-forward `main`, rerun the full suite on `main`, push, remove the temporary worktree, and delete the local branch.
