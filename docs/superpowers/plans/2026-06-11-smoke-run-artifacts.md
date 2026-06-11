# Smoke Run Artifact Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `jarvis smoke-run` so real research smoke queries produce a complete review, export, eval, and manifest artifact pack.

**Architecture:** Keep `research-run` as the single owner of discovery, source gating, auto-labeling, deep reads, and report/passport/rejection/summary artifacts. Add a thin CLI wrapper that supplies smoke defaults, points `research-run` at a smoke directory, then derives label review/export/eval files and a manifest from the same batch.

**Tech Stack:** Python standard library, existing `JarvisStore`, existing CLI handlers, `label_review`, `label_export`, `label_eval`, `research_artifacts`, and `unittest`.

---

### Task 1: Smoke Run CLI Contract

**Files:**
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write the failing CLI test**

Add `test_smoke_run_writes_dogfood_artifact_pack` near the existing `research-run` artifact tests. The test should call:

```python
code = main(
    [
        "smoke-run",
        "MALDI AMR",
        "--limit",
        "2",
        "--deep-read-limit",
        "0",
        "--output-dir",
        str(smoke_dir),
        "--data-dir",
        str(data_dir),
    ],
    discoverer=fake_discoverer,
)
```

Assert the command writes `report.md`, `passport.json`, `rejection-log.json`, `run-summary.json`, `labels-review.json`, `labels-export.jsonl`, `label-eval.json`, and `smoke-manifest.json`. Assert the manifest contains `artifact_type="smoke_run_manifest"`, the query, limits, run ID, batch ID, artifact paths, and next commands.

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_cli.CliTests.test_smoke_run_writes_dogfood_artifact_pack -v`

Expected: FAIL because `smoke-run` is not a known command.

### Task 2: Smoke Run Implementation

**Files:**
- Modify: `jarvis_research/cli.py`

- [x] **Step 1: Add parser and command dispatch**

Add `smoke-run` to `COMMAND_NAMES`, register a subparser with smoke defaults, and dispatch to `_handle_smoke_run`.

- [x] **Step 2: Wrap `research-run`**

Implement `_handle_smoke_run(args, store, data_dir, discoverer, pdf_ingestor, llm_label_client)` so it creates the smoke output directory, builds a `research-run` namespace with explicit artifact paths inside that directory, and calls `_handle_research_run`.

- [x] **Step 3: Write review/export/eval artifacts**

After the wrapped run completes, load the latest run and linked batch, then write:

```text
labels-review.json
labels-export.jsonl
label-eval.json
smoke-manifest.json
```

Use `build_label_review_rows`, `build_label_export_rows`, `render_label_export_jsonl`, `build_label_evaluation`, and `write_json_artifact`.

- [x] **Step 4: Print next commands**

Print `Smoke run directory: ...` and the concrete `run-summary`, `labels review`, `labels export`, and `labels eval` commands using the run and batch IDs.

### Task 3: Verification and Merge

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-smoke-run-artifacts.md`

- [x] **Step 1: Run the focused smoke-run test**

Run: `python3 -m unittest tests.test_cli.CliTests.test_smoke_run_writes_dogfood_artifact_pack -v`

Expected: PASS.

- [x] **Step 2: Run related CLI artifact tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_smoke_run_writes_dogfood_artifact_pack tests.test_cli.CliTests.test_research_run_writes_default_artifact_folder -v`

Expected: PASS.

- [x] **Step 3: Run the full suite**

Run: `python3 -m unittest discover -v`

Expected: PASS.

- [x] **Step 4: Commit, merge, push, cleanup**

Commit on `smoke-run-artifacts`, fast-forward `main`, push `main`, remove the temporary worktree, and delete the local feature branch.
