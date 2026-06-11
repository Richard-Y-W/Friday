# Scholarly Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `jarvis scan --query "<topic>" --limit N` query scholarly indexes and persist screened candidate records in the batch.

**Architecture:** Add a `discovery` module that converts OpenAlex, arXiv, and PubMed API responses into a common `Candidate` record. Extend storage to save candidate metadata on batch items. Keep discovery HTTP isolated behind injectable fetch functions so unit tests use fixed API fixtures and never require live network.

**Tech Stack:** Python standard library, `urllib.request`, `urllib.parse`, `json`, `xml.etree.ElementTree`, `sqlite3`, `unittest`.

---

## File Structure

- Create `jarvis_research/discovery.py`: API URL construction, HTTP helpers, response parsers, candidate dedupe.
- Modify `jarvis_research/storage.py`: add nullable candidate metadata columns to `batch_items`, add migration for existing DBs, persist `Candidate` values.
- Modify `jarvis_research/cli.py`: call discovery for `scan --query`, store candidates, print real counts.
- Modify `jarvis_research/reporting.py`: show discovered candidate metadata in batch reports.
- Add `tests/test_discovery.py`: parser and URL behavior using fake HTTP responses.
- Update `tests/test_storage.py`, `tests/test_cli.py`, and `tests/test_reporting.py`: assert persisted candidate metadata and non-empty query batches.

## Task 1: Discovery Parsers

**Files:**
- Create: `jarvis_research/discovery.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

Add tests that parse one OpenAlex JSON response, one arXiv Atom response, and one PubMed ESummary JSON response into `Candidate` records. The tests should assert title, provider, DOI/arXiv/PMID identifiers, year, URL, and source strings.

- [ ] **Step 2: Verify red**

Run: `python3 -m unittest tests.test_discovery -v`

Expected: import failure because `jarvis_research.discovery` does not exist.

- [ ] **Step 3: Implement discovery parsers**

Create:

- `Candidate` dataclass
- `parse_openalex(payload)`
- `parse_arxiv(atom_xml)`
- `parse_pubmed_summary(payload)`

Each parser returns a list of `Candidate` objects with a `source_for_gate` field used by the source policy.

- [ ] **Step 4: Verify green**

Run: `python3 -m unittest tests.test_discovery -v`

Expected: discovery parser tests pass.

## Task 2: Discovery Client

**Files:**
- Modify: `jarvis_research/discovery.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

Add a fake fetcher test for `discover_candidates("maldi amr", limit=3)`. It should verify the client calls OpenAlex, arXiv, PubMed ESearch, and PubMed ESummary, deduplicates repeated DOI/source records, and returns no more than the requested limit.

- [ ] **Step 2: Verify red**

Run: `python3 -m unittest tests.test_discovery -v`

Expected: failure because `discover_candidates` is not implemented.

- [ ] **Step 3: Implement discovery client**

Add:

- `discover_candidates(query, limit, fetch_json=None, fetch_text=None)`
- OpenAlex URL: `https://api.openalex.org/works?search=<query>&per-page=<n>`
- arXiv URL: `https://export.arxiv.org/api/query?search_query=all:<query>&start=0&max_results=<n>`
- PubMed ESearch URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=<query>&retmode=json&retmax=<n>`
- PubMed ESummary URL: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=<ids>&retmode=json`

Use one request per provider in this slice and cap each provider at the remaining limit. Later pagination can extend this without changing the storage interface.

- [ ] **Step 4: Verify green**

Run: `python3 -m unittest tests.test_discovery -v`

Expected: discovery client tests pass.

## Task 3: Persist Candidate Metadata

**Files:**
- Modify: `jarvis_research/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

Add a test that creates a `Candidate`, passes it to `add_batch_item(..., candidate=candidate)`, then verifies `list_batch_items()` returns provider, title, DOI, PMID/arXiv ID, year, and URL.

- [ ] **Step 2: Verify red**

Run: `python3 -m unittest tests.test_storage -v`

Expected: failure because `add_batch_item` does not accept candidate metadata.

- [ ] **Step 3: Implement storage metadata**

Add nullable columns to `batch_items`:

- `provider`
- `title`
- `doi`
- `pmid`
- `arxiv_id`
- `year`
- `url`

Update `BatchItemRecord`, `add_batch_item`, `list_batch_items`, and schema migration to include those fields.

- [ ] **Step 4: Verify green**

Run: `python3 -m unittest tests.test_storage -v`

Expected: storage tests pass.

## Task 4: Query CLI Uses Discovery

**Files:**
- Modify: `jarvis_research/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add a CLI test that calls `main(["scan", "--query", "MALDI AMR", "--limit", "2", "--data-dir", ...], discoverer=fake_discoverer)`, where `fake_discoverer` returns one allowed arXiv candidate and one blocked GitHub candidate. Assert output says `Screened: 2`, `Blocked: 1`, and `Deep-scanned: 1`.

- [ ] **Step 2: Verify red**

Run: `python3 -m unittest tests.test_cli -v`

Expected: failure because `main` does not accept a `discoverer` and query scans still store no candidates.

- [ ] **Step 3: Implement CLI discovery integration**

Update `main(argv=None, discoverer=discover_candidates)` and `_handle_scan()` so query mode discovers candidates, evaluates each `candidate.source_for_gate`, stores candidate metadata, and prints real coverage counts.

- [ ] **Step 4: Verify green**

Run: `python3 -m unittest tests.test_cli -v`

Expected: CLI tests pass.

## Task 5: Reports Show Candidates

**Files:**
- Modify: `jarvis_research/reporting.py`
- Test: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Add a batch report test with stored candidate metadata. Assert the report includes provider, title, and DOI/PMID/arXiv ID.

- [ ] **Step 2: Verify red**

Run: `python3 -m unittest tests.test_reporting -v`

Expected: failure because reports only show source and reason.

- [ ] **Step 3: Implement richer batch report items**

Render each item with status, provider, title, identifier, source, and reason.

- [ ] **Step 4: Verify green**

Run: `python3 -m unittest tests.test_reporting -v`

Expected: reporting tests pass.

## Task 6: Final Verification

**Files:**
- Modify as needed based on test failures.

- [ ] **Step 1: Run full unit suite**

Run: `python3 -m unittest discover -v`

Expected: all tests pass.

- [ ] **Step 2: Run offline CLI smoke**

Run:

```bash
python3 -m jarvis_research scan --query "MALDI AMR" --limit 5 --data-dir /tmp/jarvis-discovery-smoke
python3 -m jarvis_research batches --data-dir /tmp/jarvis-discovery-smoke
python3 -m jarvis_research report --latest --data-dir /tmp/jarvis-discovery-smoke
```

Expected: commands exit 0. If network is unavailable, the query command should still create a batch and print an API error message instead of crashing.
