# Gold Eval Corpus Design

## Goal

Add a curated, JSON-backed gold eval corpus so the offline eval suite can check realistic scanner behavior, not only synthetic hard-coded fixture cases.

## Scope

This feature adds:

- `eval_corpus/gold_cases.json`
- `friday.eval_corpus`
- `friday eval-suite run --suite gold`
- CI coverage for the gold suite

The gold suite remains offline. It does not browse the web, call scholarly APIs, download PDFs, execute files, or call an LLM. It evaluates local fixture metadata through existing Friday modules.

## Corpus Format

The corpus file is JSON:

```json
{
  "schema_version": "1.0",
  "cases": [
    {
      "case_id": "gold.query.maldi_amr",
      "type": "query_plan",
      "description": "MALDI AMR means antimicrobial resistance.",
      "query": "MALDI AMR",
      "expected": {
        "intent": "biomedical",
        "expanded_contains": ["MALDI antimicrobial resistance"],
        "rejected_meanings_contains": ["abstract meaning representation"]
      }
    }
  ]
}
```

Supported case types:

- `query_plan`: checks `plan_query`.
- `source_policy`: checks `evaluate_source`.
- `ranking`: checks `rank_candidates` on fixture candidate metadata.
- `screening_label`: checks `auto_label_batch_items` on fixture candidate metadata.

## Eval Integration

`friday.eval_corpus` loads the JSON file, validates a minimal schema, and converts corpus entries into `EvalCase` instances. `friday.eval_suite` adds `gold` to `available_eval_suites()` and appends gold cases to the default case list.

`core` remains the full suite and includes gold cases. `gold` filters to only corpus-backed cases.

## Initial Gold Cases

The first corpus includes realistic examples for:

- MALDI AMR query expansion.
- Conversational math-language query planning.
- GitHub paper-looking PDF blocking.
- Biomedical MALDI AMR ranking over NLP AMR collisions.
- AMR parsing not becoming a biomedical inclusion.
- Clear biomedical relevant label.
- Clear off-topic irrelevant label.

## Testing

Add tests that prove:

- The corpus file exists and loads.
- Every corpus case has stable IDs, supported type, description, and expected data.
- `run_eval_suite("gold")` passes and includes only gold cases.
- CLI list includes `gold`.
- CLI JSON output works for `--suite gold`.
- CI workflow runs the gold suite.

