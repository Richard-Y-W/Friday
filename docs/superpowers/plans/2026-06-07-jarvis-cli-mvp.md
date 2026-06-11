# Jarvis CLI MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working `jarvis` CLI slice: safe source gating, persistent scan/batch IDs, batch lists, scan lists, and evidence-constrained reports.

**Architecture:** Implement a small Python package using only the standard library. Keep source-policy logic, SQLite persistence, reporting, and CLI parsing in separate modules so later PDF parsing and scholarly API discovery can plug in without rewriting the interface.

**Tech Stack:** Python 3.11+, `argparse`, `sqlite3`, `unittest`, standard-library URL/CSV/JSON utilities.

---

## File Structure

- Create `pyproject.toml`: package metadata and console entry point.
- Create `jarvis_research/__init__.py`: package marker.
- Create `jarvis_research/__main__.py`: allows `python -m jarvis_research`.
- Create `jarvis_research/source_policy.py`: allowlist/blocklist, identifier normalization, source gate decisions.
- Create `jarvis_research/storage.py`: SQLite schema, scan/batch creation, list/latest lookups.
- Create `jarvis_research/reporting.py`: text reports for scans and batches.
- Create `jarvis_research/cli.py`: argparse command routing for `scan`, `batches`, `scans`, and `report`.
- Create tests under `tests/` for each behavior.

## Task 1: Source Gate

**Files:**
- Create: `jarvis_research/source_policy.py`
- Test: `tests/test_source_policy.py`

- [ ] **Step 1: Write failing tests**

```python
from jarvis_research.source_policy import evaluate_source


def test_allows_arxiv_pdf_url():
    decision = evaluate_source("https://arxiv.org/pdf/2401.12345")
    assert decision.allowed is True
    assert decision.kind == "url"
    assert decision.normalized == "https://arxiv.org/pdf/2401.12345"


def test_allows_bare_doi():
    decision = evaluate_source("10.1038/s41586-020-2649-2")
    assert decision.allowed is True
    assert decision.kind == "doi"
    assert decision.normalized == "10.1038/s41586-020-2649-2"


def test_blocks_github_even_when_file_looks_scholarly():
    decision = evaluate_source("https://github.com/example/repo/blob/main/paper.pdf")
    assert decision.allowed is False
    assert decision.reason == "blocked_domain"


def test_blocks_archives_and_code_artifacts():
    decision = evaluate_source("https://arxiv.org/e-print/2401.12345")
    assert decision.allowed is False
    assert decision.reason == "blocked_extension_or_artifact"
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_source_policy -v`

Expected: import failure because `jarvis_research.source_policy` does not exist.

- [ ] **Step 3: Implement source gate**

Create a `SourceDecision` dataclass and `evaluate_source()` with allowlisted scholarly domains, blocked domains, and blocked artifact extensions.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_source_policy -v`

Expected: all source-policy tests pass.

## Task 2: SQLite Storage And IDs

**Files:**
- Create: `jarvis_research/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


def test_creates_scan_id_and_lists_scan(tmp_path):
    store = JarvisStore(tmp_path / "jarvis.db")
    scan = store.create_scan("https://arxiv.org/pdf/2401.12345", evaluate_source("https://arxiv.org/pdf/2401.12345"))
    assert scan.scan_id.startswith("scan_")
    assert store.list_scans()[0].scan_id == scan.scan_id


def test_creates_batch_id_and_tracks_counts(tmp_path):
    store = JarvisStore(tmp_path / "jarvis.db")
    batch = store.create_batch(query="test query", limit=1000, mode="query")
    store.add_batch_item(batch.batch_id, "https://github.com/example/repo", evaluate_source("https://github.com/example/repo"))
    loaded = store.get_batch(batch.batch_id)
    assert loaded.batch_id == batch.batch_id
    assert loaded.blocked_count == 1
    assert loaded.screened_count == 1
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_storage -v`

Expected: import failure because `jarvis_research.storage` does not exist.

- [ ] **Step 3: Implement storage**

Use SQLite tables for `scans`, `batches`, and `batch_items`. Generate IDs with UTC timestamp plus short random suffix.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_storage -v`

Expected: all storage tests pass.

## Task 3: Reporting

**Files:**
- Create: `jarvis_research/reporting.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

```python
from jarvis_research.reporting import render_batch_report, render_scan_report
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


def test_scan_report_includes_source_decision(tmp_path):
    store = JarvisStore(tmp_path / "jarvis.db")
    scan = store.create_scan("https://github.com/example/repo", evaluate_source("https://github.com/example/repo"))
    report = render_scan_report(store, scan.scan_id)
    assert scan.scan_id in report
    assert "blocked_domain" in report


def test_batch_report_includes_coverage_counts(tmp_path):
    store = JarvisStore(tmp_path / "jarvis.db")
    batch = store.create_batch(query="test query", limit=1000, mode="query")
    store.add_batch_item(batch.batch_id, "https://arxiv.org/pdf/2401.12345", evaluate_source("https://arxiv.org/pdf/2401.12345"))
    report = render_batch_report(store, batch.batch_id)
    assert "Screened: 1" in report
    assert "Deep-scanned: 1" in report
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_reporting -v`

Expected: import failure because `jarvis_research.reporting` does not exist.

- [ ] **Step 3: Implement reporting**

Render plain-text scan and batch reports from stored records. Reports must not invent evidence; they describe stored scan/batch state only.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_reporting -v`

Expected: all reporting tests pass.

## Task 4: CLI

**Files:**
- Create: `jarvis_research/cli.py`
- Create: `jarvis_research/__main__.py`
- Create: `jarvis_research/__init__.py`
- Create: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

```python
import io
from contextlib import redirect_stdout

from jarvis_research.cli import main


def run_cli(args, tmp_path):
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([*args, "--data-dir", str(tmp_path / ".jarvis")])
    return code, out.getvalue()


def test_scan_prints_scan_id_and_report_command(tmp_path):
    code, output = run_cli(["scan", "https://arxiv.org/pdf/2401.12345"], tmp_path)
    assert code == 0
    assert "Scan ID: scan_" in output
    assert "jarvis report scan_" in output


def test_query_scan_prints_batch_id(tmp_path):
    code, output = run_cli(["scan", "--query", "MALDI AMR", "--limit", "1000"], tmp_path)
    assert code == 0
    assert "Batch ID: batch_" in output
    assert "jarvis report batch_" in output


def test_report_latest_uses_latest_batch(tmp_path):
    run_cli(["scan", "--query", "MALDI AMR", "--limit", "1000"], tmp_path)
    code, output = run_cli(["report", "--latest"], tmp_path)
    assert code == 0
    assert "Batch ID:" in output
    assert "MALDI AMR" in output
```

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_cli -v`

Expected: import failure because `jarvis_research.cli` does not exist.

- [ ] **Step 3: Implement CLI**

Use `argparse` to route:

- `scan <source>`
- `scan --query ... --limit ...`
- `scan --manifest ...`
- `batches`
- `scans`
- `report <id>`
- `report --latest`

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_cli -v`

Expected: all CLI tests pass.

## Task 5: Final Verification

**Files:**
- Modify as needed based on test results.

- [ ] **Step 1: Run full test suite**

Run: `python -m unittest discover -v`

Expected: all tests pass.

- [ ] **Step 2: Run CLI smoke commands**

Run:

```bash
python -m jarvis_research scan https://arxiv.org/pdf/2401.12345
python -m jarvis_research scan --query "MALDI AMR" --limit 1000
python -m jarvis_research batches
python -m jarvis_research report --latest
```

Expected: commands exit 0 and print scan/batch IDs and coverage reports.
