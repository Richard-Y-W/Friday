# Label Feedback Evaluation Design

## Goal

Add a CLI-first feedback loop that measures agent screening labels against human overrides, recommends safer confidence thresholds, and uses human label feedback to improve future unlabeled review and deep-read ordering.

## Scope

This feature stays inside the scholarly scanner. It does not train a model, call an LLM, browse the web, or execute downloaded artifacts. It uses labels already stored in Friday batches.

## Design

Create a focused `friday.label_eval` module that joins batch items with screening labels and reads prior agent label evidence from each human-labeled row. Because a human override replaces the current row, the evaluator will parse previous agent values from either label signals or the human label note when present. The supported note format is simple and explicit: `agent=relevant confidence=0.84`. If no prior agent value exists, the human label still contributes to human label counts but not to agent-vs-human accuracy.

The evaluator returns a dictionary with:

- human label counts
- comparable count
- confusion matrix keyed by human label and agent label
- per-label precision and recall
- disagreement rows ordered by confidence
- high-confidence mistakes
- threshold recommendations derived from confidence distributions

Extend `friday labels` with an `eval` action:

```bash
friday labels eval --latest
friday labels eval --latest --format json
```

Text output should be compact and operational: counts, comparable totals, threshold recommendations, and top disagreements. JSON output should expose the full structured report for later dashboards.

For feedback into ranking, keep the existing recommendation profile approach but sharpen it:

- Human relevant labels are stronger positive examples than agent relevant labels.
- Human irrelevant labels are stronger negative examples than agent irrelevant labels.
- Human `maybe` labels are weak positive examples.
- `rank_deep_read_items` should prioritize human relevant labels over agent relevant labels, then human maybe, then agent maybe, then unlabeled, while still excluding irrelevant labels.

## Error Handling

If a batch has no labels, evaluation returns empty counts and prints a clear no-comparisons message. If human labels lack prior agent metadata, they are counted but excluded from confusion/accuracy metrics. Malformed note metadata is ignored instead of failing the command.

## Testing

Add unit tests for `label_eval` and focused CLI tests for text and JSON output. Add screening tests proving human labels have stronger influence than agent labels in recommendations and deep-read ordering.

