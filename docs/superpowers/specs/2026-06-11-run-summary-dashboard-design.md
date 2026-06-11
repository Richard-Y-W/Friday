# Run Summary Dashboard Design

## Goal

Add a CLI-first `run-summary` command that tells the user what happened in the latest research run or batch, what needs attention, and which Jarvis command to run next.

## Scope

This is an operational summary over existing local JarvisResearch state. It does not browse the web, call an LLM, retry PDFs, train a model, or change labels. It reads the SQLite store and reuses existing artifact builders, label evaluation, review queue, and rejection log logic.

## Command

```bash
jarvis run-summary --latest
jarvis run-summary --latest --format json
jarvis run-summary --run-id run_...
jarvis run-summary --batch-id batch_...
```

`--latest` prefers the latest research run. If no research run exists, it falls back to the latest batch. `--run-id` uses the run and its linked batch. `--batch-id` summarizes a batch without run-level metadata.

## Summary Contents

The structured report should include:

- target: target type, run ID, batch ID, query, status
- counts: screened, blocked, allowed, labeled, human labels, agent labels, unlabeled allowed, stored PDFs, failed PDFs, extracted evidence
- label evaluation: comparable overrides, accuracy, top disagreements, high-confidence mistakes
- attention items: maybe labels, high-relevance unlabeled papers, failed PDF attempts, source-gate blocks, label disagreements
- next commands: concrete CLI commands the user can run next

The text output should be short and scannable. JSON output should expose the full structured report for future UI/dashboard work.

## Design

Create `jarvis_research.run_summary` with pure functions:

- `build_run_summary_dashboard(store, latest=False, run_id=None, batch_id=None, limit=5)`
- `render_run_summary_text(summary)`

Keep CLI routing thin. The builder resolves the target, gathers batch/run data, calls `build_label_evaluation`, `build_rejection_log`, and `build_llm_review_queue`, and computes attention items from local records.

## Error Handling

If no target exists, return a clear CLI error. If a run has no batch yet, summarize the run metadata and report missing batch state without crashing. If a batch has no labels or no PDF artifacts, show zero counts and next commands rather than failing.

## Testing

Add pure module tests for the structured dashboard, including failed PDF attempts, high-relevance unlabeled papers, label disagreements, and command recommendations. Add CLI tests for text and JSON output plus the no-target error case.
