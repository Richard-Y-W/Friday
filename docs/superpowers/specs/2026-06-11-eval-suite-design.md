# Eval Suite Design

## Goal

Add a local evaluation suite for JarvisResearch so we can run a deterministic scorecard over the research agent's core safety and quality behaviors before scaling to larger paper runs.

## Scope

The first version is CLI-first and offline. It does not discover live papers, browse the web, call an LLM, download PDFs, or mutate normal user research state. It evaluates existing local modules with curated fixture cases and returns a text or JSON report.

Commands:

```bash
jarvis eval-suite run
jarvis eval-suite run --suite biomedical
jarvis eval-suite run --suite natural-language
jarvis eval-suite run --format json
jarvis eval-suite list
```

## Architecture

Create a focused `jarvis_research.eval_suite` module that owns suite definitions, deterministic fixture setup, case execution, score aggregation, and text rendering. Extend `jarvis_research.cli` with a thin `eval-suite` command that delegates to the module and prints text or JSON.

Each case has a stable `case_id`, suite name, category, expected behavior, and evaluator function. Evaluators call real project APIs:

- `plan_query` for acronym and natural-language query planning.
- `evaluate_source` for source-gate behavior.
- `rank_candidates` for metadata relevance ordering.
- `auto_label_batch_items` for heuristic screening labels.
- `resolve_candidate_pdf_url` for safe PDF resolution without downloading.
- `build_claim_support_audit` for page-anchored evidence auditing.

Cases that need storage use an in-memory SQLite store and fixture batch records, so they do not depend on `.jarvis`.

## Suites

`core` runs every case. Named suite filters run subsets:

- `biomedical`: MALDI/AMR acronym resolution, biomedical ranking, heuristic labels, and PMC/OA PDF resolution.
- `natural-language`: conversational math-language query planning and screening labels.
- `safety`: GitHub/code artifact blocking and evidence-support audit gaps.

## Output

Text output is short and operational:

- Suite name.
- Overall status: `pass` or `fail`.
- Passed/failed counts and percentage.
- One line per case with status, category, and message.

JSON output exposes the same data as a structured artifact:

- `artifact_type: eval_suite_report`
- `schema_version`
- `suite`
- `status`
- `counts`
- `cases`

The command exits `0` only when all selected cases pass. Unknown suites or actions exit `1` with a clear message.

## Error Handling

Evaluator exceptions are caught and reported as failed cases with `error:<ExceptionType>` messages. This keeps the suite useful even if one subsystem regresses hard.

## Testing

Add pure module tests for:

- Core report shape and pass counts.
- Suite filtering.
- Failure reporting when an injected case returns `False`.

Add CLI tests for:

- `jarvis eval-suite list`.
- `jarvis eval-suite run`.
- `jarvis eval-suite run --suite biomedical --format json`.
- Unknown suite error.

