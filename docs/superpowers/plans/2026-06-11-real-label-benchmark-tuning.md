# Real Label Benchmark Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-smoke human-label benchmark and tune heuristic screening against the 30 reviewed smoke-run labels.

**Architecture:** Extend the existing JSON-backed eval corpus loader with a second fixture file and suite name, reusing the existing `screening_label` evaluator. Tune `friday.screening` with small, query-aware scoring rules that preserve the current metadata-only, non-LLM safety model.

**Tech Stack:** Python standard library, JSON fixtures, existing `EvalCase`, `FridayStore`, `auto_label_batch_items`, and `unittest`.

---

### Task 1: Real-Smoke Fixture Loader

**Files:**
- Modify: `friday/eval_corpus.py`
- Modify: `friday/eval_suite.py`
- Modify: `tests/test_eval_corpus.py`
- Modify: `tests/test_eval_suite.py`
- Create: `eval_corpus/real_smoke_labels.json`

- [x] **Step 1: Write failing loader and suite tests**

Add tests that import `REAL_SMOKE_CORPUS_PATH`, `load_real_smoke_eval_cases`, and `build_real_smoke_eval_cases`. Assert the fixture exists, has 30 cases, uses `real_smoke.*` case IDs, converts to suite `real-smoke`, and `run_eval_suite("real-smoke")` is available.

- [x] **Step 2: Run failing tests**

Run:

```bash
python3 -m unittest tests.test_eval_corpus.GoldEvalCorpusTests.test_real_smoke_corpus_file_loads_human_labels tests.test_eval_corpus.GoldEvalCorpusTests.test_real_smoke_corpus_converts_to_eval_cases tests.test_eval_suite.EvalSuiteTests.test_real_smoke_suite_runs_human_label_cases -v
```

Expected: FAIL because real-smoke corpus loading and suite wiring do not exist.

- [x] **Step 3: Add fixture and loader**

Add `eval_corpus/real_smoke_labels.json` and extend `eval_corpus.py` with real-smoke load/build helpers. Reuse the existing screening-label case runner and candidate mapping.

- [x] **Step 4: Run loader tests**

Run the focused loader/suite tests again. Expected: tests load the suite, but real-smoke cases may still fail before heuristic tuning.

### Task 2: Heuristic Screening Tuning

**Files:**
- Modify: `friday/screening.py`
- Modify: `tests/test_screening.py`

- [x] **Step 1: Write focused failing screening tests**

Add tests for:

- MALDI AMR off-domain susceptibility physics demotes to `irrelevant`
- ESBL CRE non-biomedical surveillance demotes to `irrelevant`
- clinical PubMed sparse metadata promotes to `relevant`
- math-language statistical/applied-linguistics metadata promotes to `relevant`

- [x] **Step 2: Run focused tests and real-smoke suite red**

Run:

```bash
python3 -m unittest tests.test_screening.ScreeningTests.test_real_smoke_tuning_demotes_off_domain_biomedical_maybe tests.test_screening.ScreeningTests.test_real_smoke_tuning_promotes_math_language_methods -v
python3 -m friday eval-suite run --suite real-smoke
```

Expected: FAIL before heuristic changes.

- [x] **Step 3: Tune rules**

Add small helper token sets and query-aware branches in `_auto_label_item` for:

- biomedical wrong-domain physics/simulation/surveillance terms
- clinical source/query boosts for PubMed/DOI clinical rows
- math-language methodology boosts

- [x] **Step 4: Run focused tests and real-smoke suite green**

Run the same focused tests and `python3 -m friday eval-suite run --suite real-smoke`. Expected: PASS.

### Task 3: CI, Docs, and Verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/superpowers/plans/2026-06-11-real-label-benchmark-tuning.md`

- [x] **Step 1: Add CI gate check**

Update CI to run:

```bash
python3 -m friday eval-suite run --suite real-smoke
```

- [x] **Step 2: Run full verification**

Run:

```bash
python3 -m unittest discover -v
python3 -m friday eval-suite run --suite real-smoke
python3 -m friday eval-suite run --suite gold
```

Expected: all pass.

- [x] **Step 3: Commit, merge, push, cleanup**

Commit on `real-label-benchmark-tuning`, fast-forward `main`, rerun full verification on main, push, remove the temporary worktree, and delete the local branch.
