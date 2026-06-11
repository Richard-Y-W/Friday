# Jarvis Scanner Agent Design

Date: 2026-06-07
Status: Approved direction, awaiting user review before implementation planning

## Summary

Jarvis is the local umbrella name for a safe research assistant. The first build is `jarvis scan`, which calls Scanner Agent V1: a scholarly-only ingestion and evidence-extraction agent. It reads papers, not arbitrary web pages or code repositories. It treats all paper content as untrusted evidence, extracts auditable claims, and stores provenance for later research workflows.

The later `jarvis research` mode will search broad scholarly indexes, screen hundreds or thousands of candidate papers, deep-read a selected subset through `jarvis scan`, and write cited evidence reports. That mode depends on Scanner Agent V1 because broad search without a trustworthy scanner would produce summaries without reliable provenance.

## Goals

- Ingest scholarly papers safely from allowlisted sources.
- Block GitHub, arbitrary downloads, scripts, ZIPs, and supplementary code by default.
- Parse PDFs in a restricted environment with no code execution.
- Extract paper metadata, sections, tables, captions, references, methods, results, limitations, and claim-level evidence.
- Store each extracted claim with source, page, section, text span, and confidence.
- Provide batch scanning that can screen thousands of papers while deep-reading only selected PDFs unless explicitly requested.
- Make prompt injection ineffective by treating document text as untrusted data, never as tool instructions.

## Non-Goals

- Jarvis Scanner V1 will not clone repositories, inspect GitHub code, execute paper-provided scripts, or ingest arbitrary supplementary files.
- It will not claim to deeply read every candidate discovered in a large query unless explicitly run in an all-deep mode.
- It will not replace human judgment for paper quality, novelty, or final scientific claims.
- It will not initially implement autonomous hypothesis generation or multi-day discovery loops.

## User Interface

Primary commands:

```bash
jarvis scan <paper-url-or-doi>
jarvis scan --query "<scholarly query>" --limit 1000
jarvis scan --manifest papers.csv
jarvis scan --all-deep --manifest papers.csv
jarvis label --batch-id <batch-id> --source <paper-url-or-doi> --label relevant
jarvis labels --batch-id <batch-id> --recommend
jarvis auto-label --batch-id <batch-id> --apply
jarvis auto-label --batch-id <batch-id> --provider llm --model gpt-5.5 --dry-run
jarvis
jarvis> tell me about MALDI AMR
jarvis> jarvis tell me about the importance of math in language
jarvis> /settings
jarvis /settings
jarvis /settings set auto_label.provider llm
jarvis /settings set auto_label.model gpt-5.5
jarvis /settings set research.limit 250
jarvis what is the importance of math in language
jarvis tell me about MALDI AMR --deep-read-limit 5 --write
jarvis tell me about MALDI AMR --write --write-output draft.md
jarvis report <scan-id-or-batch-id>
jarvis write --latest --mode literature-review
```

Command meanings:

- `jarvis scan <paper-url-or-doi>` scans one allowed scholarly source.
- `jarvis scan --query ...` discovers and screens many candidates, then deep-scans a selected subset.
- `jarvis scan --manifest ...` scans or screens a supplied list of DOIs, arXiv IDs, PubMed IDs, or allowlisted PDF URLs.
- `jarvis scan --all-deep ...` deep-scans every safe PDF in a manifest and requires explicit user intent because it can be slow and expensive.
- `jarvis label ...` stores a human screening decision for a batch item.
- `jarvis labels ... --recommend` lists label counts and recommends unlabeled papers using prior relevant/irrelevant decisions.
- `jarvis auto-label ...` applies metadata-only agent labels with confidence, rationale, and signals. The default `heuristic` provider is no-token. The optional `llm` provider uses a strict JSON response contract and should be run in `--dry-run` mode first when token cost matters.
- Plain `jarvis` opens an interactive shell when run in a real terminal. It keeps `jarvis --help` and non-interactive invocations script-safe.
- Inside the shell, natural lines such as `tell me about MALDI AMR` run the scholarly flow with saved settings and write a report package automatically. The shell also accepts the redundant prefix, so `jarvis tell me about MALDI AMR` works at the prompt.
- Successful shell research runs also copy the reader-facing report PDF to `~/Desktop/JarvisReports/<query-slug>-<timestamp>.pdf` while keeping the complete package under `.jarvis/reports/...`.
- `jarvis /settings ...` shows or updates saved defaults used by natural-language research runs.
- `jarvis <natural language question>` routes unknown commands into the scholarly-only research flow.
- `jarvis <natural language question> --write` runs the same scholarly scan and then drafts an evidence-bound literature-review note from the page-anchored batch report.
- `jarvis report ...` emits a cited summary from the evidence database.
- `jarvis write ...` drafts from an existing batch report without repeating discovery or PDF parsing.

The natural-language fallback is not a general chatbot. It is a wrapper over the same scholarly-only scanner, auto-labeler, safe PDF reader, and cited report renderer.
For recognized casual research prompts, Jarvis rewrites conversational wording into safer scholarly query variants before discovery. For example, `what is the importance of math in language` becomes searches such as `mathematical linguistics`, `formal language theory natural language`, and `information theory language`, instead of searching primarily for weak words like `what` or `importance`.
With `--write` or `--draft`, the natural fallback feeds the generated batch report into the writing copilot. The draft is still evidence-bound: it uses only extracted page-level evidence and marks unsupported areas as `MATERIAL GAP`. `--output` remains the scanner report path; `--write-output` writes the draft or writing package separately.
Writing packages now include both machine-readable handoff files and a reader-facing report: `report.md`, `report.pdf`, `literature_table.csv`, `evidence_table.csv`, `citation_audit.json`, raw writing JSON, paper references, screening labels, supported paragraphs, blocked paragraphs, and material gaps. The generated PDF is a simple local rendering of the evidence-bound Markdown report, not a free-form web summary.

## Source Policy

Allowed by default:

- arXiv metadata and PDFs.
- PubMed and PubMed Central metadata/full text where available.
- DOI metadata through Crossref.
- OpenAlex and Semantic Scholar metadata.
- Publisher PDFs from an explicit allowlist such as Nature, Springer, ScienceDirect, Wiley, IEEE, ACM, PLOS, Cell Press, Science, Oxford Academic, Cambridge, BMJ, JAMA, NEJM, and ASM.

Blocked by default:

- GitHub, GitLab, Bitbucket, Gist, and arbitrary code-hosting links.
- ZIP, TAR, RAR, DMG, executable, notebook, script, binary, and supplementary-code downloads.
- Google Drive, Dropbox, personal websites, unknown file hosts, and arbitrary HTML downloads.
- Any source that fails domain, MIME type, file signature, size, or redirect checks.

Manual override is not part of V1. If a source is blocked, the output records the reason and continues.

## Pipeline

### 1. Discover

For query or batch mode, Jarvis queries scholarly indexes rather than general web search. The first candidate set is metadata-only: title, abstract, DOI, arXiv ID, PMID/PMCID, authors, venue, year, citation counts where available, source URL, and available full-text links.

### 2. Gate

Each candidate passes through the source gate before any full-text download:

- domain allowlist check
- redirect chain check
- MIME type check
- PDF magic-byte check for full-text files
- maximum file-size check
- content hash recording
- source identifier normalization

Blocked candidates are stored with exclusion reasons so the final report can describe coverage honestly.

### 3. Screen

Large queries screen all candidates by metadata first. Screening stores relevance score, paper type estimate, source quality flags, year, venue, dedupe status, and optional human labels. The default deep-read set combines:

- top relevance matches
- human `relevant` labels
- highly cited papers
- recent papers
- review papers
- benchmark or dataset papers
- contradictory or negative evidence where detectable

Default batch behavior screens hundreds or thousands and deep-reads only a selected subset. Human `irrelevant` labels are excluded from resumed deep reads, while human `relevant` labels can override the minimum relevance threshold.

Agent auto-labeling is conservative and auditable. It ignores conversational prompt filler, does not count a candidate's stored query variant as evidence for relevance, and records overlap signals used for each label. Domain-specific prompt rewrites can add stricter checks; for mathematical-language prompts, generic language-model or clinical-language papers are not marked `relevant` unless the metadata also contains mathematical or formal-language signals.

LLM-backed auto-labeling is optional and disabled by default. When enabled through `auto_label.provider=llm` or `jarvis auto-label --provider llm`, the model receives only metadata and query-plan fields: title, abstract, journal, MeSH terms, OpenAlex concepts, identifiers, year, relevance score, and query variants. It does not receive PDF files, parsed paper text, local paths, API keys, commands, or tools. The response must be strict JSON with `label`, `confidence`, `rationale`, `evidence_terms`, and `exclusion_reason`; invalid or failed responses are skipped rather than written. Human labels continue to override all agent labels.

### 4. Parse

PDF parsing runs in a restricted subprocess or container:

- no network access during parsing
- no execution of embedded content
- timeout limit
- memory limit
- file-size limit
- parser crash isolation

The parser extracts inert artifacts: text blocks, sections, page numbers, tables, figure captions, references, and document metadata.

### 5. Extract Evidence

The evidence extractor treats parsed document text as untrusted source material. It extracts:

- paper type: empirical, review, dataset, benchmark, method, protocol, opinion, or unclassified
- datasets and cohorts
- methods and experimental conditions
- metrics and main results
- limitations and caveats
- claims with page, section, and text-span provenance
- cited references
- source-quality flags such as retraction status, missing methods, unusually old evidence, weak source type, or inaccessible full text

Every generated claim must link back to extracted evidence. Uncited claims are marked as unsupported and excluded from final reports by default.

### 6. Store

The evidence database stores:

- `papers`: DOI/arXiv/PMID, title, authors, venue, year, source URLs, hashes, retrieval time
- `documents`: parser version, text blocks, section map, tables, captions, references
- `claims`: claim text, claim type, paper ID, page, section, span, confidence, extraction model/version
- `screening`: query ID, relevance score, include/exclude decision, exclusion reason
- `source_flags`: retraction, publisher status, source type, old evidence, blocked source, parser warnings
- `batches`: query, candidate count, screened count, deep-read count, blocked count, created outputs

The storage format can start as SQLite plus local files for PDFs and parsed artifacts.

## Prompt-Injection Controls

- The model receives paper text only inside an explicit untrusted-evidence wrapper.
- Document text is never interpreted as developer, system, user, or tool instructions.
- The scanner cannot call tools based on instructions found inside a paper.
- Extraction prompts require page/section/span citations for claims.
- Any text that resembles prompt-injection content is flagged as document content, not executed or followed.
- Final reports separate evidence from agent reasoning and include coverage counts.

## Batch Semantics

`jarvis scan --query ... --limit 1000` means:

- discover up to 1,000 scholarly candidates
- screen all candidates by metadata
- block unsafe or unsupported sources
- deep-read a selected subset, usually dozens to low hundreds
- produce a coverage report

It does not mean every one of the 1,000 candidates is deeply parsed. Deep-reading every candidate requires `--all-deep` and still respects the source gate.

## Outputs

Single-paper scan output:

- paper metadata
- scan status
- parsed section/table/caption summary
- extracted claim table
- source-quality flags
- citation-ready evidence snippets

Batch output:

- candidate count
- duplicate count
- blocked count and reasons
- screened count
- deep-read count
- top evidence clusters
- papers requiring manual review
- cited summary constrained to extracted evidence

## Error Handling

- Blocked source: record source and reason, continue.
- Missing PDF: store metadata-only record, continue.
- Parser failure: record parser warning and preserve metadata.
- Model extraction failure: retry once with smaller chunks, then mark extraction failed.
- Incomplete provenance: exclude claim from final evidence report by default.

## Testing Strategy

V1 tests should cover:

- allowlist and blocklist behavior
- DOI/arXiv/PubMed identifier normalization
- MIME and PDF signature validation
- redirect rejection for blocked domains
- parser timeout and failure isolation
- prompt-injection fixture inside a fake PDF text block
- claim extraction requiring page/section/span provenance
- batch coverage accounting
- no network or code execution in parser tests

## Open Implementation Decisions

These are design choices for the implementation plan, not unresolved product requirements:

- exact CLI framework
- exact parser stack
- exact embedding/indexing library
- exact LLM/provider abstraction
- exact publisher allowlist versioning format

The product behavior above is fixed for V1 unless the user revises the spec.
