# Eval Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline `friday eval-suite` scorecard for core research-agent quality and safety behavior.

**Architecture:** Add `friday.eval_suite` for suite definitions, fixture execution, aggregation, and text rendering. Keep `friday.cli` thin by parsing `eval-suite list` and `eval-suite run`, then delegating to the module.

**Tech Stack:** Python standard library, `unittest`, existing Friday modules and SQLite store helpers.

---

### Task 1: Module Tests

**Files:**
- Create: `tests/test_eval_suite.py`
- Create later: `friday/eval_suite.py`

- [x] **Step 1: Write failing module tests**

```python
import unittest

from friday.eval_suite import EvalCase, run_eval_suite, render_eval_report_text


class EvalSuiteTests(unittest.TestCase):
    def test_core_suite_runs_all_default_cases(self):
        report = run_eval_suite("core")

        self.assertEqual(report["artifact_type"], "eval_suite_report")
        self.assertEqual(report["suite"], "core")
        self.assertEqual(report["status"], "pass")
        self.assertGreaterEqual(report["counts"]["total"], 6)
        self.assertEqual(report["counts"]["failed"], 0)
        case_ids = {case["case_id"] for case in report["cases"]}
        self.assertIn("biomedical.maldi_amr_query_plan", case_ids)
        self.assertIn("safety.github_pdf_blocked", case_ids)

    def test_named_suite_filters_cases(self):
        report = run_eval_suite("natural-language")

        self.assertEqual(report["suite"], "natural-language")
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["cases"])
        self.assertEqual({case["suite"] for case in report["cases"]}, {"natural-language"})

    def test_failure_case_is_reported_without_raising(self):
        report = run_eval_suite(
            "core",
            cases=[
                EvalCase(
                    case_id="fixture.fail",
                    suite="core",
                    category="fixture",
                    description="intentional failure",
                    run=lambda: (False, "expected failure"),
                )
            ],
        )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["counts"], {"total": 1, "passed": 0, "failed": 1, "pass_rate": 0.0})
        self.assertEqual(report["cases"][0]["message"], "expected failure")

    def test_text_renderer_includes_summary_and_cases(self):
        report = run_eval_suite("safety")
        text = render_eval_report_text(report)

        self.assertIn("Friday Eval Suite", text)
        self.assertIn("Suite: safety", text)
        self.assertIn("Status: pass", text)
        self.assertIn("safety.github_pdf_blocked", text)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_eval_suite -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'friday.eval_suite'`.

- [x] **Step 3: Implement `friday.eval_suite`**

Implement:

```python
@dataclass(frozen=True)
class EvalCase:
    case_id: str
    suite: str
    category: str
    description: str
    run: Callable[[], tuple[bool, str]]


def available_eval_suites() -> tuple[str, ...]:
    return ("core", "biomedical", "natural-language", "safety")


def run_eval_suite(suite: str = "core", cases: Sequence[EvalCase] | None = None) -> dict[str, Any]:
    ...


def render_eval_report_text(report: dict[str, Any]) -> str:
    ...
```

Include deterministic default cases that exercise query planning, source policy, relevance ranking, heuristic auto-labels, PDF URL resolution, and claim-support audit gaps.

- [x] **Step 4: Run module tests to verify they pass**

Run: `python3 -m unittest tests.test_eval_suite -v`
Expected: PASS.

### Task 2: CLI Integration

**Files:**
- Modify: `friday/cli.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing CLI tests**

Add tests that assert:

```python
code, output = self.run_cli(["eval-suite", "list"], tmp_path)
self.assertEqual(code, 0)
self.assertIn("core", output)
self.assertIn("biomedical", output)

code, output = self.run_cli(["eval-suite", "run"], tmp_path)
self.assertEqual(code, 0)
self.assertIn("Friday Eval Suite", output)
self.assertIn("Status: pass", output)

code, output = self.run_cli(["eval-suite", "run", "--suite", "biomedical", "--format", "json"], tmp_path)
self.assertEqual(code, 0)
report = json.loads(output)
self.assertEqual(report["suite"], "biomedical")
self.assertEqual(report["status"], "pass")

code, output = self.run_cli(["eval-suite", "run", "--suite", "unknown"], tmp_path)
self.assertEqual(code, 1)
self.assertIn("Unknown eval suite: unknown", output)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_cli.CliTests.test_eval_suite_list_outputs_available_suites tests.test_cli.CliTests.test_eval_suite_run_outputs_scorecard tests.test_cli.CliTests.test_eval_suite_run_json_outputs_structured_report tests.test_cli.CliTests.test_eval_suite_unknown_suite_returns_error -v`
Expected: FAIL because `eval-suite` is not routed by the CLI.

- [x] **Step 3: Implement CLI command**

Add imports from `friday.eval_suite`, include `"eval-suite"` in `COMMAND_NAMES`, add parser subcommands:

```python
eval_suite = subparsers.add_parser("eval-suite", help="Run offline Friday quality and safety evaluations.")
eval_suite.add_argument("action", choices=("list", "run"))
eval_suite.add_argument("--suite", default="core")
eval_suite.add_argument("--format", choices=("text", "json"), default="text")
```

Add `_handle_eval_suite(args)` and route it before store-backed commands because it does not need `.friday`.

- [x] **Step 4: Run CLI tests to verify they pass**

Run: `python3 -m unittest tests.test_cli.CliTests.test_eval_suite_list_outputs_available_suites tests.test_cli.CliTests.test_eval_suite_run_outputs_scorecard tests.test_cli.CliTests.test_eval_suite_run_json_outputs_structured_report tests.test_cli.CliTests.test_eval_suite_unknown_suite_returns_error -v`
Expected: PASS.

### Task 3: Verification and Finish

**Files:**
- Modify: `docs/superpowers/plans/2026-06-11-eval-suite.md`

- [x] **Step 1: Run focused eval suite tests**

Run: `python3 -m unittest tests.test_eval_suite -v`
Expected: PASS.

- [x] **Step 2: Run focused CLI tests**

Run: `python3 -m unittest tests.test_cli.CliTests.test_eval_suite_list_outputs_available_suites tests.test_cli.CliTests.test_eval_suite_run_outputs_scorecard tests.test_cli.CliTests.test_eval_suite_run_json_outputs_structured_report tests.test_cli.CliTests.test_eval_suite_unknown_suite_returns_error -v`
Expected: PASS.

- [x] **Step 3: Run full suite**

Run: `python3 -m unittest discover -v`
Expected: PASS with all tests.

- [x] **Step 4: Commit, merge, push, cleanup**

Commit on `eval-suite`, fast-forward `main`, run the full suite again on merged `main`, push `main`, remove the temporary worktree, and delete the local feature branch.

