import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_research.discovery import Candidate
from jarvis_research.evidence import EvidenceItem
from jarvis_research.reporting import render_batch_report_json
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore
from jarvis_research.writing_copilot import (
    build_writing_audit_summary,
    build_writing_payload,
    render_writing_markdown,
)


class WritingCopilotTests(unittest.TestCase):
    def test_claim_table_uses_only_page_anchored_supported_claims(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_evidence(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            payload = build_writing_payload(report_data, mode="claim-table")
            markdown = render_writing_markdown(payload)

            self.assertEqual(payload["artifact_type"], "writing_copilot_output")
            self.assertEqual(payload["mode"], "claim-table")
            self.assertEqual(len(payload["claims"]), 3)
            self.assertIn("| C1 | SUPPORTED | method | P1 p1 | We used MALDI-TOF spectra to train a classifier. |", markdown)
            self.assertIn("| C2 | SUPPORTED | result | P1 p2 | The model achieved an AUROC of 0.91. |", markdown)
            self.assertNotIn("unsupported", markdown.lower())

    def test_literature_review_preserves_citations_and_material_gaps(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_evidence(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            payload = build_writing_payload(report_data, mode="literature-review")
            markdown = render_writing_markdown(payload)

            self.assertIn("# Evidence-Bound Literature Review Draft", markdown)
            self.assertIn("Method evidence: We used MALDI-TOF spectra to train a classifier. [P1 p1]", markdown)
            self.assertIn("Result evidence: The model achieved an AUROC of 0.91. [P1 p2]", markdown)
            self.assertIn("Limitation evidence: The cohort was from a single hospital. [P1 p3]", markdown)
            self.assertIn("This draft uses only page-anchored extracted evidence.", markdown)

    def test_outline_and_limitations_surface_gaps_when_no_evidence_exists(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            report_data = render_batch_report_json(store, batch.batch_id)

            outline = render_writing_markdown(build_writing_payload(report_data, mode="outline"))
            limitations = render_writing_markdown(build_writing_payload(report_data, mode="limitations"))

            self.assertIn("MATERIAL GAP: No page-anchored extracted evidence is available", outline)
            self.assertIn("# Evidence-Bound Limitations", limitations)
            self.assertIn("MATERIAL GAP", limitations)

    def test_payload_can_be_built_from_report_json_file(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_evidence(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            report_path = Path(tmp) / "report.json"
            report_path.write_text(json.dumps(report_data), encoding="utf-8")

            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            payload = build_writing_payload(loaded, mode="claim-table")

            self.assertEqual(payload["source_report"]["batch_id"], batch_id)
            self.assertEqual(payload["claims"][0]["citation"], "P1 p1")

    def test_payload_clusters_evidence_and_builds_paper_reference_table(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            payload = build_writing_payload(report_data, mode="methods-summary")

            self.assertEqual(payload["paper_references"][0]["label"], "P1")
            self.assertEqual(payload["paper_references"][0]["evidence_count"], 2)
            self.assertEqual(payload["paper_references"][1]["label"], "P2")
            method_cluster = next(item for item in payload["evidence_clusters"] if item["evidence_type"] == "method")
            result_cluster = next(item for item in payload["evidence_clusters"] if item["evidence_type"] == "result")
            self.assertEqual(method_cluster["paper_count"], 2)
            self.assertEqual(result_cluster["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(payload["citation_check"]["status"], "pass")

    def test_methods_and_results_summary_render_cross_paper_synthesis_with_citations(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            methods = render_writing_markdown(build_writing_payload(report_data, mode="methods-summary"))
            results = render_writing_markdown(build_writing_payload(report_data, mode="results-summary"))

            self.assertIn("# Evidence-Bound Methods Summary", methods)
            self.assertIn("Across 2 papers, method evidence includes", methods)
            self.assertIn("[P1 p1; P2 p1]", methods)
            self.assertIn("## Paper References", methods)
            self.assertIn("| P1 | MALDI antimicrobial resistance prediction | 2024 | Nature Medicine | 10.1038/example-a | 2 |", methods)
            self.assertIn("# Evidence-Bound Results Summary", results)
            self.assertIn("Across 2 papers, result evidence includes", results)
            self.assertIn("[P1 p2; P2 p2]", results)

    def test_background_and_research_gaps_modes_remain_evidence_bound(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            background = render_writing_markdown(build_writing_payload(report_data, mode="background"))
            gaps = render_writing_markdown(build_writing_payload(report_data, mode="research-gaps"))

            self.assertIn("# Evidence-Bound Background", background)
            self.assertIn("Across 2 papers, method evidence includes", background)
            self.assertIn("[P1 p1; P2 p1]", background)
            self.assertIn("# Evidence-Bound Research Gaps", gaps)
            self.assertIn("MATERIAL GAP: No page-anchored limitation evidence is available in this batch.", gaps)

    def test_citation_check_flags_missing_citations_in_synthesis_blocks(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="results-summary")

            broken = dict(payload)
            broken["synthesis_blocks"] = [
                {
                    "section": "Results",
                    "evidence_type": "result",
                    "paragraph": "This paragraph has no page citation.",
                    "citations": [],
                    "status": "SUPPORTED",
                }
            ]

            from jarvis_research.writing_copilot import validate_citation_coverage

            check = validate_citation_coverage(broken)

            self.assertEqual(check["status"], "fail")
            self.assertEqual(check["uncited_blocks"][0]["section"], "Results")

    def test_payload_records_paragraph_level_claim_audit(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)

            payload = build_writing_payload(report_data, mode="methods-summary")

            self.assertIn("paragraph_claim_audit", payload)
            audit = payload.get("paragraph_claim_audit", [])
            method_entry = next(item for item in audit if item["evidence_type"] == "method")
            self.assertEqual(method_entry["support_status"], "SUPPORTED")
            self.assertEqual(method_entry["citations"], ["P1 p1", "P2 p1"])
            self.assertEqual(method_entry["evidence_count"], 2)
            self.assertIn("Across 2 papers, method evidence includes", method_entry["paragraph"])
            self.assertEqual(payload["citation_check"]["audited_paragraph_count"], len(audit))
            self.assertEqual(payload["citation_check"]["unsupported_paragraphs"], [])

    def test_paragraph_audit_rejects_citations_not_backed_by_block_evidence(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="results-summary")

            broken = dict(payload)
            broken["synthesis_blocks"] = [
                {
                    "section": "Results",
                    "evidence_type": "result",
                    "paragraph": "This paragraph cites an unavailable page. [P9 p9]",
                    "citations": ["P1 p2"],
                    "status": "SUPPORTED",
                }
            ]

            from jarvis_research.writing_copilot import validate_citation_coverage

            check = validate_citation_coverage(broken)

            self.assertEqual(check["status"], "fail")
            self.assertEqual(check["unsupported_paragraphs"][0]["reason"], "unknown_page_citation")

    def test_payload_includes_audit_summary_for_downstream_tools(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            store.set_screening_label(batch_id, "10.1038/example-a", "relevant", note="human include")
            store.set_screening_label(batch_id, "10.1038/example-b", "maybe")
            report_data = render_batch_report_json(store, batch_id)

            payload = build_writing_payload(report_data, mode="results-summary")

            summary = payload["audit_summary"]
            self.assertEqual(summary["citation_check_status"], "pass")
            self.assertEqual(summary["supported_paragraph_count"], 2)
            self.assertEqual(summary["blocked_paragraph_count"], 0)
            self.assertEqual(summary["supported_paragraphs"][1]["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(summary["blocked_paragraphs"], [])
            self.assertEqual(payload["screening_labels"]["counts"]["relevant"], 1)
            self.assertEqual(payload["source_report"]["screening_label_counts"]["maybe"], 1)
            self.assertEqual(payload["screening_labels"]["labels"][0]["note"], "human include")

    def test_audit_summary_exposes_blocked_paragraphs_without_markdown_parsing(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="results-summary")
            payload["synthesis_blocks"] = [
                {
                    "block_id": "S99",
                    "section": "Results",
                    "evidence_type": "result",
                    "paragraph": "This paragraph cites an unavailable page. [P9 p9]",
                    "citations": ["P1 p2"],
                    "status": "SUPPORTED",
                }
            ]

            summary = build_writing_audit_summary(payload)

            self.assertEqual(summary["citation_check_status"], "fail")
            self.assertEqual(summary["supported_paragraph_count"], 0)
            self.assertEqual(summary["blocked_paragraph_count"], 1)
            self.assertEqual(summary["blocked_paragraphs"][0]["reason"], "unknown_page_citation")
            self.assertEqual(summary["blocked_paragraphs"][0]["paragraph_id"], "S99.1")

    def test_build_writing_package_files_splits_safe_handoff_artifacts(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="results-summary")

            import jarvis_research.writing_copilot as writing_copilot

            self.assertTrue(hasattr(writing_copilot, "build_writing_package_files"))
            package = writing_copilot.build_writing_package_files(payload)

            self.assertEqual(
                sorted(package),
                [
                    "blocked_paragraphs.json",
                    "citation_audit.json",
                    "claims.csv",
                    "draft.md",
                    "evidence_table.csv",
                    "evidence_tables.json",
                    "limitations.csv",
                    "literature_table.csv",
                    "material_gaps.json",
                    "methods.csv",
                    "paper_references.json",
                    "populations.csv",
                    "report.md",
                    "report.pdf",
                    "results.csv",
                    "screening_labels.json",
                    "source_report.json",
                    "supported_paragraphs.json",
                    "writing.json",
                ],
            )
            self.assertIn("# Evidence-Bound Results Summary", package["draft.md"])
            self.assertIn("# Jarvis Evidence Report", package["report.md"])
            self.assertIn("## Executive Summary", package["report.md"])
            self.assertIn("## Background", package["report.md"])
            self.assertIn("## Key Findings", package["report.md"])
            self.assertIn("## Methods And Evidence Base", package["report.md"])
            self.assertIn("## Limitations And Gaps", package["report.md"])
            self.assertIn("## Literature Table", package["report.md"])
            self.assertTrue(package["report.pdf"].startswith(b"%PDF-1.4"))
            self.assertEqual(json.loads(package["writing.json"])["artifact_type"], "writing_copilot_output")
            self.assertEqual(json.loads(package["paper_references.json"])[0]["label"], "P1")
            self.assertEqual(json.loads(package["supported_paragraphs.json"])[0]["citations"], ["P1 p1", "P2 p1"])
            self.assertEqual(json.loads(package["blocked_paragraphs.json"]), [])
            self.assertEqual(json.loads(package["source_report.json"])["batch_id"], batch_id)
            self.assertEqual(json.loads(package["screening_labels.json"])["counts"]["relevant"], 0)
            self.assertEqual(json.loads(package["writing.json"])["evidence_tables"]["counts"]["results"], 2)
            self.assertEqual(json.loads(package["citation_audit.json"])["citation_check_status"], "pass")

    def test_build_writing_package_files_exports_structured_evidence_tables(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_table_evidence(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="literature-review")

            from jarvis_research.writing_copilot import build_writing_package_files

            package = build_writing_package_files(payload)
            tables = json.loads(package["evidence_tables.json"])

            self.assertEqual(tables["artifact_type"], "writing_evidence_tables")
            self.assertEqual(tables["counts"]["methods"], 1)
            self.assertEqual(tables["counts"]["results"], 1)
            self.assertEqual(tables["counts"]["limitations"], 1)
            self.assertEqual(tables["counts"]["populations"], 1)
            result = tables["tables"]["results"][0]
            self.assertEqual(result["row_id"], "E2")
            self.assertEqual(result["evidence_type"], "result")
            self.assertEqual(result["paper"], "P1")
            self.assertEqual(result["citation"], "P1 p2")
            self.assertEqual(result["page_number"], 2)
            self.assertEqual(result["paper_title"], "MALDI antimicrobial resistance prediction")
            self.assertEqual(result["journal"], "Nature Medicine")
            self.assertEqual(result["doi"], "10.1038/example-table")
            self.assertEqual(result["text"], "The model achieved an AUROC of 0.91.")

            rows = list(csv.DictReader(io.StringIO(package["evidence_table.csv"])))
            self.assertEqual([row["row_id"] for row in rows], ["E1", "E2", "E3", "E4"])
            result_rows = list(csv.DictReader(io.StringIO(package["results.csv"])))
            self.assertEqual(result_rows[0]["citation"], "P1 p2")
            population_rows = list(csv.DictReader(io.StringIO(package["populations.csv"])))
            self.assertEqual(population_rows[0]["evidence_type"], "dataset_population")
            literature_rows = list(csv.DictReader(io.StringIO(package["literature_table.csv"])))
            self.assertEqual(literature_rows[0]["paper"], "P1")
            self.assertEqual(literature_rows[0]["title"], "MALDI antimicrobial resistance prediction")
            self.assertEqual(list(csv.DictReader(io.StringIO(package["claims.csv"]))), [])

    def test_render_blocks_unsupported_synthesis_paragraphs(self):
        with TemporaryDirectory() as tmp:
            store, batch_id = _store_with_two_papers(Path(tmp))
            report_data = render_batch_report_json(store, batch_id)
            payload = build_writing_payload(report_data, mode="results-summary")
            payload["synthesis_blocks"] = [
                {
                    "section": "Results",
                    "evidence_type": "result",
                    "paragraph": "This unsupported paragraph should never be emitted.",
                    "citations": ["P1 p2"],
                    "status": "SUPPORTED",
                }
            ]

            markdown = render_writing_markdown(payload)

            self.assertIn("MATERIAL GAP: Unsupported synthesis paragraph blocked in Results", markdown)
            self.assertNotIn("This unsupported paragraph should never be emitted.", markdown)


def _store_with_evidence(root: Path) -> tuple[JarvisStore, str]:
    store = JarvisStore(root / "jarvis.db")
    batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
    source = "10.1038/example"
    candidate = Candidate(
        provider="openalex",
        title="MALDI antimicrobial resistance prediction",
        source_for_gate=source,
        doi=source,
        journal="Nature Medicine",
        year=2024,
    )
    store.add_batch_item(batch.batch_id, source, evaluate_source(source), candidate)
    artifact = store.add_pdf_artifact(
        batch.batch_id,
        source=source,
        pdf_url="https://www.nature.com/articles/example.pdf",
        final_url="https://www.nature.com/articles/example.pdf",
        sha256="a" * 64,
        byte_count=123,
        content_type="application/pdf",
        local_path="artifacts/paper.pdf",
        status="stored",
        reason="pdf_text_extracted",
    )
    store.add_evidence_records(
        artifact.artifact_id,
        [
            EvidenceItem(
                evidence_type="method",
                text="We used MALDI-TOF spectra to train a classifier.",
                page_number=1,
            ),
            EvidenceItem(
                evidence_type="result",
                text="The model achieved an AUROC of 0.91.",
                page_number=2,
            ),
            EvidenceItem(
                evidence_type="limitation",
                text="The cohort was from a single hospital.",
                page_number=3,
            ),
        ],
    )
    return store, batch.batch_id


def _store_with_two_papers(root: Path) -> tuple[JarvisStore, str]:
    store = JarvisStore(root / "jarvis.db")
    batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
    papers = [
        (
            "10.1038/example-a",
            "MALDI antimicrobial resistance prediction",
            "We used MALDI-TOF spectra to train a classifier.",
            "The model achieved an AUROC of 0.91.",
        ),
        (
            "10.1038/example-b",
            "MALDI-TOF antimicrobial susceptibility testing",
            "Investigators trained a MALDI-TOF classifier on clinical isolates.",
            "The classifier detected resistant isolates with 88 percent sensitivity.",
        ),
    ]
    for index, (source, title, method, result) in enumerate(papers, start=1):
        candidate = Candidate(
            provider="openalex",
            title=title,
            source_for_gate=source,
            doi=source,
            journal="Nature Medicine" if index == 1 else "Clinical Microbiology",
            year=2024 if index == 1 else 2023,
        )
        store.add_batch_item(batch.batch_id, source, evaluate_source(source), candidate)
        artifact = store.add_pdf_artifact(
            batch.batch_id,
            source=source,
            pdf_url=f"https://example.org/paper-{index}.pdf",
            final_url=f"https://example.org/paper-{index}.pdf",
            sha256=str(index) * 64,
            byte_count=123,
            content_type="application/pdf",
            local_path=f"artifacts/paper-{index}.pdf",
            status="stored",
            reason="pdf_text_extracted",
        )
        store.add_evidence_records(
            artifact.artifact_id,
            [
                EvidenceItem(evidence_type="method", text=method, page_number=1),
                EvidenceItem(evidence_type="result", text=result, page_number=2),
            ],
        )
    return store, batch.batch_id


def _store_with_table_evidence(root: Path) -> tuple[JarvisStore, str]:
    store = JarvisStore(root / "jarvis.db")
    batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
    source = "10.1038/example-table"
    candidate = Candidate(
        provider="openalex",
        title="MALDI antimicrobial resistance prediction",
        source_for_gate=source,
        doi=source,
        journal="Nature Medicine",
        year=2024,
    )
    store.add_batch_item(batch.batch_id, source, evaluate_source(source), candidate)
    artifact = store.add_pdf_artifact(
        batch.batch_id,
        source=source,
        pdf_url="https://www.nature.com/articles/example-table.pdf",
        final_url="https://www.nature.com/articles/example-table.pdf",
        sha256="b" * 64,
        byte_count=456,
        content_type="application/pdf",
        local_path="artifacts/paper-table.pdf",
        status="stored",
        reason="pdf_text_extracted",
    )
    store.add_evidence_records(
        artifact.artifact_id,
        [
            EvidenceItem(
                evidence_type="method",
                text="We used MALDI-TOF spectra to train a classifier.",
                page_number=1,
            ),
            EvidenceItem(
                evidence_type="result",
                text="The model achieved an AUROC of 0.91.",
                page_number=2,
            ),
            EvidenceItem(
                evidence_type="dataset_population",
                text="The cohort included 220 clinical isolates.",
                page_number=3,
            ),
            EvidenceItem(
                evidence_type="limitation",
                text="The cohort was from a single hospital.",
                page_number=4,
            ),
        ],
    )
    return store, batch.batch_id


if __name__ == "__main__":
    unittest.main()
