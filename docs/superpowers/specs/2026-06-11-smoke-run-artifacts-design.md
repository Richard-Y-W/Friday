# Smoke Run Artifact Pack Design

## Goal

Add a CLI-first `smoke-run` command for real research dogfooding. A smoke run should execute one scholarly-only research query with practical defaults, then write every artifact needed for review, threshold tuning, and later benchmark snapshots.

## Scope

This command is an orchestration wrapper around the existing `research-run` pipeline. It does not add new discovery providers, browse the open web, execute external artifacts, train a classifier, or change the scholarly-only source policy.

## Command

```bash
friday smoke-run "MALDI AMR" --limit 100 --deep-read-limit 5
friday smoke-run "ESBL CRE surveillance" --limit 100 --deep-read-limit 5 --output-dir .friday/smoke-runs/esbl-cre
```

Defaults:

- `--limit 100`
- `--deep-read-limit 5`
- `--page-size 200`
- `--request-delay 0`
- `--min-relevance 25`
- `--deep-read-workers 1`
- `--format markdown`

## Artifact Pack

The smoke run directory should contain:

- `report.md`, `report.txt`, or `report.json`
- `passport.json`
- `rejection-log.json`
- `run-summary.json`
- `labels-review.json`
- `labels-export.jsonl`
- `label-eval.json`
- `smoke-manifest.json`

`smoke-manifest.json` records the query, limits, run ID, batch ID, artifact paths, and next commands. This gives each real smoke query a stable handoff into the human review loop.

## Review Loop

After a smoke run completes, the CLI should print the directory and concrete follow-up commands:

```bash
friday run-summary --run-id run_...
friday labels review --batch-id batch_...
friday labels export --batch-id batch_... --output ...
friday labels eval --batch-id batch_...
```

The exported artifacts make it possible to review maybe/high-confidence labels, apply human overrides, run label evaluation, and tune thresholds from real messy searches.

## Safety

The wrapper inherits the existing source gate and PDF ingestion rules:

- scholarly sources only
- GitHub/code/supplementary artifacts blocked by default
- paper text treated as untrusted input
- no command execution from discovered content

## Testing

Add a CLI test that runs `smoke-run` against a fake discoverer with one allowed arXiv result and one blocked GitHub result. Assert the command exits cleanly, writes the complete artifact pack, records run and batch IDs in the manifest, and prints the next review commands.
