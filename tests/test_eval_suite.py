import unittest

from jarvis_research.eval_suite import EvalCase, run_eval_suite, render_eval_report_text


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

    def test_gold_suite_runs_json_backed_cases(self):
        report = run_eval_suite("gold")

        self.assertEqual(report["suite"], "gold")
        self.assertEqual(report["status"], "pass")
        self.assertGreaterEqual(report["counts"]["total"], 7)
        self.assertEqual({case["suite"] for case in report["cases"]}, {"gold"})
        self.assertIn("gold.query.maldi_amr_biomedical", {case["case_id"] for case in report["cases"]})

    def test_real_smoke_suite_runs_human_label_cases(self):
        report = run_eval_suite("real-smoke")

        self.assertEqual(report["suite"], "real-smoke")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["counts"]["total"], 30)
        self.assertEqual({case["suite"] for case in report["cases"]}, {"real-smoke"})
        self.assertIn("real_smoke.maldi_amr.01", {case["case_id"] for case in report["cases"]})

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

        self.assertIn("Jarvis Eval Suite", text)
        self.assertIn("Suite: safety", text)
        self.assertIn("Status: pass", text)
        self.assertIn("safety.github_pdf_blocked", text)
