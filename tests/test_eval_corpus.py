import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.eval_corpus import (
    GOLD_CORPUS_PATH,
    REAL_SMOKE_CORPUS_PATH,
    build_gold_eval_cases,
    build_real_smoke_eval_cases,
    load_gold_eval_cases,
    load_real_smoke_eval_cases,
)


class GoldEvalCorpusTests(unittest.TestCase):
    def test_gold_corpus_file_loads_with_stable_supported_cases(self):
        cases = load_gold_eval_cases()

        self.assertTrue(GOLD_CORPUS_PATH.exists())
        self.assertGreaterEqual(len(cases), 25)
        self.assertEqual(
            {"query_plan", "source_policy", "ranking", "screening_label"},
            {case["type"] for case in cases},
        )
        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        for case in cases:
            self.assertTrue(case["case_id"].startswith("gold."))
            self.assertIn(case["type"], {"query_plan", "source_policy", "ranking", "screening_label"})
            self.assertIsInstance(case["description"], str)
            self.assertTrue(case["description"])
            self.assertIsInstance(case["expected"], dict)

    def test_gold_corpus_converts_to_eval_cases(self):
        eval_cases = build_gold_eval_cases()

        self.assertGreaterEqual(len(eval_cases), 25)
        self.assertEqual({case.suite for case in eval_cases}, {"gold"})
        self.assertTrue(all(case.case_id.startswith("gold.") for case in eval_cases))
        self.assertIn("gold.query.maldi_amr_biomedical", {case.case_id for case in eval_cases})

    def test_gold_eval_cases_pass_against_current_pipeline(self):
        eval_cases = build_gold_eval_cases()
        failed = []
        for case in eval_cases:
            passed, message = case.run()
            if not passed:
                failed.append(f"{case.case_id}: {message}")

        self.assertEqual(failed, [])

    def test_query_plan_gold_case_can_assert_unresolved_acronyms(self):
        payload = {
            "schema_version": "1.0",
            "cases": [
                {
                    "case_id": "gold.query.unresolved_negative",
                    "type": "query_plan",
                    "description": "A query-plan case fails when an expected unresolved acronym is absent.",
                    "query": "PCR assay",
                    "expected": {
                        "intent": "biomedical",
                        "unresolved_acronyms_contains": ["XYZ"],
                    },
                }
            ],
        }
        with TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "gold_cases.json"
            corpus_path.write_text(json.dumps(payload), encoding="utf-8")
            eval_case = build_gold_eval_cases(corpus_path)[0]

            passed, message = eval_case.run()

        self.assertFalse(passed)
        self.assertIn("missing_unresolved_acronym='XYZ'", message)

    def test_real_smoke_corpus_file_loads_human_labels(self):
        cases = load_real_smoke_eval_cases()

        self.assertTrue(REAL_SMOKE_CORPUS_PATH.exists())
        self.assertGreaterEqual(len(cases), 36)
        self.assertEqual({"screening_label", "topic_curation"}, {case["type"] for case in cases})
        self.assertEqual(sum(1 for case in cases if case["type"] == "screening_label"), 30)
        self.assertGreaterEqual(sum(1 for case in cases if case["type"] == "topic_curation"), 6)
        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertIn("real_smoke.maldi_amr.01", case_ids)
        self.assertIn("real_smoke.sepsis_procalcitonin.10", case_ids)
        self.assertIn("real_smoke.topic_curation.maldi_amr.broad_amr_blocked", case_ids)
        self.assertIn("real_smoke.topic_curation.protein_folding.generic_protein_blocked", case_ids)
        for case in cases:
            self.assertTrue(case["case_id"].startswith("real_smoke."))
            self.assertIsInstance(case["candidate"], dict)
            self.assertIsInstance(case["query"], str)
            self.assertTrue(case["query"])
            if case["type"] == "screening_label":
                self.assertIn(case["expected"]["label"], {"relevant", "maybe", "irrelevant"})
            if case["type"] == "topic_curation":
                self.assertIn("eligible_for_deep_read", case["expected"])

    def test_real_smoke_corpus_converts_to_eval_cases(self):
        eval_cases = build_real_smoke_eval_cases()

        self.assertGreaterEqual(len(eval_cases), 36)
        self.assertEqual({case.suite for case in eval_cases}, {"real-smoke"})
        self.assertTrue(all(case.case_id.startswith("real_smoke.") for case in eval_cases))

    def test_real_smoke_topic_curation_cases_pass_against_current_pipeline(self):
        eval_cases = [
            case
            for case in build_real_smoke_eval_cases()
            if case.category == "topic_curation"
        ]
        failed = []
        for case in eval_cases:
            passed, message = case.run()
            if not passed:
                failed.append(f"{case.case_id}: {message}")

        self.assertGreaterEqual(len(eval_cases), 6)
        self.assertEqual(failed, [])
