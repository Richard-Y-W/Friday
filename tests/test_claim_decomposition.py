import json
import unittest

from friday.claim_decomposition import build_report_claim_units


class ClaimDecompositionTests(unittest.TestCase):
    def test_builds_typed_claim_units_with_support_metadata(self):
        report = (
            "# Friday Research Report\n\n"
            "Source: Batch `batch_test`; query `MALDI AMR`; screened `100`; deep-read `5`\n\n"
            "## Executive Summary\n\n"
            "- **Results:** Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
            "---\n\n"
            "## Results\n\n"
            "Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
            "---\n\n"
            "## Limitations\n\n"
            "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
            "---\n\n"
            "## Evidence Table\n\n"
            "| Section | Evidence | Citations |\n"
            "| --- | --- | --- |\n"
            "| result | AUROC 0.91 | 1, p. 2 |\n"
        )
        package = {
            "source_report.json": {"batch_id": "batch_test", "query": "MALDI AMR"},
            "material_gaps.json": [
                {
                    "reason": "evidence_gap",
                    "message": "No page-anchored limitation evidence is available in this batch.",
                }
            ],
            "evidence_tables.json": {
                "all_rows": [
                    {
                        "row_id": "R1",
                        "paper": "P1",
                        "evidence_type": "result",
                        "support_status": "SUPPORTED",
                        "quality_label": "clean",
                        "quality_score": 0.95,
                        "parse_confidence": 0.88,
                        "trust_label": "trusted",
                        "trust_score": 0.9,
                        "citation": "P1 p2",
                        "text": "AUROC 0.91",
                    },
                    {
                        "row_id": "R2",
                        "paper": "P2",
                        "evidence_type": "result",
                        "support_status": "SUPPORTED",
                        "quality_label": "clean",
                        "quality_score": 0.92,
                        "parse_confidence": 0.82,
                        "trust_label": "trusted",
                        "trust_score": 0.85,
                        "citation": "P2 p2",
                        "text": "sensitivity 88 percent",
                    },
                ]
            },
        }

        artifact = build_report_claim_units(report, package)

        self.assertEqual(artifact["artifact_type"], "report_claim_units")
        units = artifact["claim_units"]
        self.assertGreaterEqual(len(units), 2)
        supported = next(unit for unit in units if "AUROC 0.91" in unit["text"])
        self.assertEqual(supported["claim_type"], "synthesis")
        self.assertEqual(supported["citations"], ["P1 p2", "P2 p2"])
        self.assertEqual(supported["support_status"], "supported")
        self.assertEqual(supported["evidence_count"], 2)
        self.assertEqual(supported["evidence_types"], ["result"])
        self.assertEqual(supported["min_parse_confidence"], 0.82)
        self.assertEqual(supported["min_trust_score"], 0.85)
        gap = next(unit for unit in units if unit["claim_type"] == "material_gap")
        self.assertEqual(gap["support_status"], "material_gap")
        self.assertEqual(gap["citations"], [])

    def test_marks_uncited_factual_sentences_and_unknown_citations(self):
        report = (
            "# Friday Research Report\n\n"
            "## Results\n\n"
            "The method reduced mortality across hospitals.\n\n"
            "One paper reported AUROC 0.91 [9, p. 9].\n"
        )
        package = {"source_report.json": {}, "evidence_tables.json": {"all_rows": []}, "material_gaps.json": []}

        artifact = build_report_claim_units(report, package)

        by_status = {unit["support_status"]: unit for unit in artifact["claim_units"]}
        self.assertIn("uncited", by_status)
        self.assertEqual(by_status["uncited"]["claim_type"], "factual")
        self.assertIn("unknown_citation", by_status)
        self.assertEqual(by_status["unknown_citation"]["citations"], ["P9 p9"])

    def test_artifact_is_json_serializable(self):
        artifact = build_report_claim_units("# Friday Research Report\n\n## Results\n\nNo supported findings were available.\n", {})

        encoded = json.dumps(artifact, sort_keys=True)

        self.assertIn("report_claim_units", encoded)


if __name__ == "__main__":
    unittest.main()
