import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.compose_agent import (
    ComposePackageError,
    build_compose_package_files,
)


class ComposeAgentTests(unittest.TestCase):
    def test_missing_required_package_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "source_report.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ComposePackageError) as raised:
                build_compose_package_files(package_dir, section="results")

            self.assertIn("supported_paragraphs.json", str(raised.exception))

    def test_results_compose_uses_only_supported_result_paragraphs(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="results")

            self.assertEqual(
                sorted(files),
                [
                    "claim_audit.json",
                    "conflicts.json",
                    "draft.md",
                    "outline.json",
                    "refused_claims.json",
                    "used_evidence.json",
                ],
            )
            draft = files["draft.md"]
            self.assertIn("# Evidence-Bound Results Draft", draft)
            self.assertIn("Across 2 papers, result evidence includes AUROC 0.91; sensitivity 88 percent [P1 p2; P2 p2].", draft)
            self.assertNotIn("method evidence includes", draft)
            self.assertNotIn("unsupported generated result", draft)

            audit = json.loads(files["claim_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["supported_paragraph_count"], 1)
            self.assertEqual(audit["paragraphs"][0]["citations"], ["P1 p2", "P2 p2"])
            used = json.loads(files["used_evidence.json"])
            self.assertEqual(used["used_evidence"][0]["evidence_type"], "result")
            refused = json.loads(files["refused_claims.json"])
            self.assertEqual(refused["refused_claims"][0]["reason"], "unknown_page_citation")
            conflicts = json.loads(files["conflicts.json"])
            self.assertEqual(conflicts["conflict_count"], 0)

    def test_no_section_evidence_emits_material_gap_without_used_evidence(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="limitations")

            self.assertIn("MATERIAL GAP: No supported limitation evidence is available in this writing package.", files["draft.md"])
            audit = json.loads(files["claim_audit.json"])
            self.assertEqual(audit["status"], "material_gap")
            self.assertEqual(audit["supported_paragraph_count"], 0)
            used = json.loads(files["used_evidence.json"])
            self.assertEqual(used["used_evidence"], [])

    def test_results_compose_groups_outline_and_exports_conflicts(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_grouped_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="results")

            outline = json.loads(files["outline.json"])
            self.assertEqual(outline["groups"][0]["group_label"], "Resistance detection")
            self.assertEqual(outline["groups"][0]["paragraph_count"], 2)
            self.assertEqual(outline["groups"][0]["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(outline["groups"][1]["group_label"], "Model performance")
            self.assertEqual(outline["items"][0]["group_label"], "Resistance detection")

            draft = files["draft.md"]
            self.assertLess(draft.index("## Resistance detection"), draft.index("## Model performance"))
            self.assertIn("MALDI-TOF improved resistant-isolate detection [P1 p2].", draft)
            self.assertIn("MALDI-TOF showed no improvement for resistant-isolate detection [P2 p2].", draft)

            conflicts = json.loads(files["conflicts.json"])
            self.assertEqual(conflicts["artifact_type"], "compose_conflicts")
            self.assertEqual(conflicts["conflict_count"], 1)
            self.assertEqual(conflicts["conflicts"][0]["group_label"], "Resistance detection")
            self.assertEqual(conflicts["conflicts"][0]["stance_set"], ["negative", "positive"])
            self.assertEqual(conflicts["conflicts"][0]["citations"], ["P1 p2", "P2 p2"])

    def test_compose_attaches_evidence_table_rows_when_available(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "schema_version": "1.0",
                    "artifact_type": "writing_evidence_tables",
                    "tables": {
                        "results": [
                            {
                                "row_id": "E10",
                                "evidence_type": "result",
                                "paper": "P1",
                                "citation": "P1 p2",
                                "page_number": 2,
                                "text": "The model achieved an AUROC of 0.91.",
                            },
                            {
                                "row_id": "E11",
                                "evidence_type": "result",
                                "paper": "P2",
                                "citation": "P2 p2",
                                "page_number": 2,
                                "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            },
                        ]
                    },
                    "counts": {"results": 2},
                },
            )

            files = build_compose_package_files(package_dir, section="results")

            used = json.loads(files["used_evidence.json"])
            evidence = used["used_evidence"][0]
            self.assertEqual([row["row_id"] for row in evidence["table_rows"]], ["E10", "E11"])
            outline = json.loads(files["outline.json"])
            self.assertEqual(outline["items"][0]["table_row_ids"], ["E10", "E11"])


def _write_fixture_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "source_report.json",
        {
            "batch_id": "batch_test",
            "query": "MALDI AMR",
            "screened_count": 1000,
            "blocked_count": 25,
            "deep_read_count": 50,
        },
    )
    _write_json(
        package_dir / "paper_references.json",
        [
            {
                "label": "P1",
                "title": "MALDI antimicrobial resistance prediction",
                "year": 2024,
                "journal": "Nature Medicine",
                "doi": "10.1038/example-a",
                "evidence_count": 2,
            },
            {
                "label": "P2",
                "title": "MALDI-TOF antimicrobial susceptibility testing",
                "year": 2023,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-b",
                "evidence_count": 2,
            },
        ],
    )
    _write_json(
        package_dir / "supported_paragraphs.json",
        [
            {
                "paragraph_id": "S1.1",
                "block_id": "S1",
                "section": "Method",
                "evidence_type": "method",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "Across 2 papers, method evidence includes spectra classifiers [P1 p1; P2 p1].",
                "citations": ["P1 p1", "P2 p1"],
                "evidence_count": 2,
            },
            {
                "paragraph_id": "S2.1",
                "block_id": "S2",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "Across 2 papers, result evidence includes AUROC 0.91; sensitivity 88 percent [P1 p2; P2 p2].",
                "citations": ["P1 p2", "P2 p2"],
                "evidence_count": 2,
            },
        ],
    )
    _write_json(
        package_dir / "blocked_paragraphs.json",
        [
            {
                "paragraph_id": "S99.1",
                "block_id": "S99",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "MATERIAL_GAP",
                "reason": "unknown_page_citation",
                "paragraph": "unsupported generated result [P9 p9]",
                "citations": ["P9 p9"],
                "evidence_count": 0,
            }
        ],
    )
    _write_json(
        package_dir / "material_gaps.json",
        [
            {
                "reason": "evidence_gap",
                "message": "No page-anchored limitation evidence is available in this batch.",
            }
        ],
    )


def _write_grouped_fixture_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "source_report.json",
        {
            "batch_id": "batch_grouped",
            "query": "MALDI AMR",
            "screened_count": 1000,
            "blocked_count": 25,
            "deep_read_count": 50,
        },
    )
    _write_json(
        package_dir / "paper_references.json",
        [
            {
                "label": "P1",
                "title": "Positive MALDI detection result",
                "year": 2024,
                "journal": "Nature Medicine",
                "doi": "10.1038/example-positive",
                "evidence_count": 1,
            },
            {
                "label": "P2",
                "title": "Negative MALDI detection result",
                "year": 2024,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-negative",
                "evidence_count": 1,
            },
            {
                "label": "P3",
                "title": "MALDI classifier performance",
                "year": 2023,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-performance",
                "evidence_count": 1,
            },
        ],
    )
    _write_json(
        package_dir / "supported_paragraphs.json",
        [
            {
                "paragraph_id": "S1.1",
                "block_id": "S1",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "MALDI-TOF improved resistant-isolate detection [P1 p2].",
                "citations": ["P1 p2"],
                "evidence_count": 1,
            },
            {
                "paragraph_id": "S2.1",
                "block_id": "S2",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "MALDI-TOF showed no improvement for resistant-isolate detection [P2 p2].",
                "citations": ["P2 p2"],
                "evidence_count": 1,
            },
            {
                "paragraph_id": "S3.1",
                "block_id": "S3",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "The classifier achieved AUROC 0.91 [P3 p4].",
                "citations": ["P3 p4"],
                "evidence_count": 1,
            },
        ],
    )
    _write_json(package_dir / "blocked_paragraphs.json", [])
    _write_json(package_dir / "material_gaps.json", [])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
