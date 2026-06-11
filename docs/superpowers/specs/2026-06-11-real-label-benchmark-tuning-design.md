# Real Label Benchmark Tuning Design

## Goal

Turn the 30 human-reviewed smoke-run labels into an offline benchmark and tune the heuristic screening rules so real mistakes become measurable regressions.

## Scope

This bundle covers the first three remaining scanner quality items:

- real smoke label benchmark fixture
- auto-label rule tuning
- clinical versus arXiv/ML filtering improvements

It does not change live discovery providers, PDF resolution, writing-copilot prose, or optional LLM labeling.

## Benchmark Fixture

Add `eval_corpus/real_smoke_labels.json` with 30 `screening_label` cases derived from the human review sheets:

- MALDI AMR: 6 reviewed `maybe` rows
- ESBL CRE surveillance: 4 reviewed `maybe` rows
- importance of math in language: 10 reviewed `maybe` rows
- sepsis procalcitonin antibiotic stewardship: 10 reviewed `maybe` rows

The fixture is offline and stores enough metadata for deterministic `auto_label_batch_items` evaluation. It must not depend on `.friday`, Desktop files, live APIs, PDFs, or an LLM.

## Eval Suite

Add a `real-smoke` eval suite. `core` should include `real-smoke` cases so the full local suite and CI gate catch regressions.

Expected commands:

```bash
python3 -m friday eval-suite run --suite real-smoke
python3 -m friday eval-suite run --suite core
```

The real-smoke suite should fail before tuning, because current heuristics label these rows as `maybe`. After tuning, the suite should pass.

## Heuristic Tuning

Keep `maybe` as a review bucket, but improve the obvious cases:

- demote non-AMR uses of `susceptibility`, including physics, superconductors, QSM, and lattice/topological papers
- demote non-biomedical uses of `surveillance`, including communications, tactical video, and air mobility
- demote generic simulation/transmission papers for clinical stewardship queries unless they include strong clinical/stewardship/procalcitonin/sepsis signals
- promote true clinical/PubMed papers for clinical queries even when abstracts are sparse
- promote math-language/applied-linguistics/statistical-linguistics papers that are broad but genuinely about language methodology

## Data Flow

```text
real_smoke_labels.json
  -> eval_corpus loader
  -> EvalCase objects in suite=real-smoke
  -> auto_label_batch_items
  -> expected human label comparison
```

## Testing

Add tests for:

- loading and validating the real-smoke fixture
- converting real-smoke cases into `EvalCase` objects
- real-smoke suite availability
- real-smoke suite passing after tuning
- focused screening tests for off-domain biomedical false positives and math-language promotion

Run the focused tests first, then full `python3 -m unittest discover -v`, then `python3 -m friday eval-suite run --suite real-smoke`.
