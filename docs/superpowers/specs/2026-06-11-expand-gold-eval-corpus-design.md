# Expand Gold Eval Corpus Design

## Goal

Expand the JSON-backed gold eval corpus from starter coverage to realistic scanner coverage while keeping the suite offline, deterministic, and fast enough for CI.

## Scope

This expands `eval_corpus/gold_cases.json` to at least 25 cases across:

- Query planning: AMR ambiguity, biomedical acronyms, MALDI variants, natural math-language prompts.
- Source policy: blocked code/artifact URLs and allowed scholarly URLs/DOIs.
- Ranking: biomedical-vs-NLP AMR collisions, MALDI AST/susceptibility, math-language ranking.
- Screening labels: relevant, maybe, and irrelevant edge cases.

The corpus format stays unchanged. `friday.eval_corpus` should still load and execute cases without network, browser, LLM, or PDF downloads.

## Expected Behavior Fixes

If new gold cases expose behavior gaps, keep fixes scoped to existing local modules. Likely areas:

- Query planning for biomedical acronyms such as AST, ESBL, and CRE.
- Relevance ranking for MALDI AST/susceptibility and math-language queries.
- Heuristic labels for borderline maybe cases.

## Testing

Raise the minimum gold corpus size assertion to 25 cases and require coverage across all supported case types. Keep the existing guarantee that every gold case passes through the current local pipeline.

Run:

```bash
python3 -m unittest tests.test_eval_corpus tests.test_eval_suite tests.test_relevance tests.test_screening tests.test_query_planning tests.test_source_policy -v
python3 -m friday eval-suite run --suite gold
python3 -m unittest discover -v
```

