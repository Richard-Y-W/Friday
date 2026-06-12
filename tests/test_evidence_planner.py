import json
import unittest

from friday.evidence_planner import build_evidence_plan, build_llm_evidence_plan
from friday.llm.types import LLMResponse


class EvidencePlannerTests(unittest.TestCase):
    def test_deterministic_plan_excludes_front_matter_and_formula_rows(self):
        package = _planner_package(
            rows=[
                {
                    "row_id": "R1",
                    "evidence_type": "result",
                    "citation": "P1 p1",
                    "text": "Keywords: MALDI-TOF MS, antimicrobial resistance screening, AMR.",
                    "trust_label": "trusted",
                },
                {
                    "row_id": "R2",
                    "evidence_type": "method",
                    "citation": "P1 p2",
                    "text": "Formally, the learning objective is defined as: L = CE(z,y) + mu L_SNP.",
                    "trust_label": "trusted",
                },
                {
                    "row_id": "R3",
                    "evidence_type": "result",
                    "citation": "P2 p3",
                    "text": "The classifier achieved an AUROC of 0.91 in validation isolates.",
                    "trust_label": "trusted",
                },
            ]
        )

        plan = build_evidence_plan(package, section="all")
        rows = {row["row_id"]: row for row in plan["rows"]}

        self.assertEqual(rows["R1"]["action"], "exclude")
        self.assertEqual(rows["R1"]["role"], "front_matter")
        self.assertEqual(rows["R2"]["action"], "appendix")
        self.assertEqual(rows["R2"]["role"], "formula_detail")
        self.assertEqual(rows["R3"]["action"], "include")
        self.assertEqual(rows["R3"]["role"], "result")
        self.assertEqual(plan["included_row_ids"], ["R3"])
        self.assertEqual(plan["appendix_row_ids"], ["R2"])
        self.assertEqual(plan["excluded_row_ids"], ["R1"])
        self.assertEqual(plan["included_citations"], ["P2 p3"])
        self.assertEqual(plan["appendix_citations"], ["P1 p2"])
        self.assertEqual(plan["excluded_citations"], ["P1 p1"])

    def test_llm_plan_uses_planner_role_and_rejects_unknown_citations(self):
        package = _planner_package(
            rows=[
                {
                    "row_id": "R1",
                    "evidence_type": "result",
                    "citation": "P1 p2",
                    "text": "The classifier achieved an AUROC of 0.91.",
                    "trust_label": "trusted",
                }
            ]
        )
        router = FakeRouter(
            {
                "planner": LLMResponse(
                    provider="claude_cli",
                    model="sonnet",
                    success=True,
                    text=json.dumps(
                        {
                            "rows": [
                                {
                                    "row_id": "R1",
                                    "citation": "P1 p2",
                                    "role": "result",
                                    "action": "include",
                                    "reason": "usable result evidence",
                                },
                                {
                                    "row_id": "made-up",
                                    "citation": "P9 p9",
                                    "role": "result",
                                    "action": "include",
                                    "reason": "not in package",
                                },
                            ]
                        }
                    ),
                )
            }
        )

        plan = build_llm_evidence_plan(package, section="results", router=router)

        self.assertEqual([role for role, _request in router.calls], ["planner"])
        self.assertEqual(plan["provider"], "claude_cli")
        self.assertEqual(plan["planner_status"], "pass")
        self.assertEqual(plan["included_citations"], ["P1 p2"])
        self.assertEqual([row["row_id"] for row in plan["rows"]], ["R1"])
        self.assertIn("trusted evidence rows", router.calls[0][1].prompt)
        self.assertIn("Do not browse", router.calls[0][1].system_prompt)


class FakeRouter:
    def __init__(self, responses):
        self.responses = {
            role: list(response) if isinstance(response, list) else [response]
            for role, response in responses.items()
        }
        self.calls = []

    def generate(self, role, request):
        self.calls.append((role, request))
        return self.responses[role].pop(0)


def _planner_package(*, rows):
    return {
        "source_report.json": {
            "batch_id": "batch_test",
            "query": "MALDI AMR",
        },
        "paper_references.json": [
            {
                "label": "P1",
                "title": "MALDI AMR paper",
                "year": 2024,
                "journal": "Clinical Microbiology",
            },
            {
                "label": "P2",
                "title": "MALDI classifier paper",
                "year": 2024,
                "journal": "Nature Medicine",
            },
        ],
        "material_gaps.json": [],
        "evidence_tables.json": {
            "schema_version": "1.0",
            "artifact_type": "writing_evidence_tables",
            "all_rows": rows,
            "tables": {},
        },
    }


if __name__ == "__main__":
    unittest.main()
