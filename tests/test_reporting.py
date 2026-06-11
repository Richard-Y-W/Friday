import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.evidence import EvidenceItem
from friday.reporting import (
    build_batch_report_data,
    render_batch_report,
    render_batch_report_json,
    render_batch_report_markdown,
    render_scan_report_json,
    render_scan_report_markdown,
    render_scan_report,
)
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class ReportingTests(unittest.TestCase):
    def test_scan_report_includes_source_decision(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            scan = store.create_scan(
                "https://github.com/example/repo",
                evaluate_source("https://github.com/example/repo"),
            )
            report = render_scan_report(store, scan.scan_id)
            self.assertIn(scan.scan_id, report)
            self.assertIn("blocked_domain", report)

    def test_batch_report_includes_coverage_counts(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1000, mode="query")
            store.add_batch_item(
                batch.batch_id,
                "https://arxiv.org/pdf/2401.12345",
                evaluate_source("https://arxiv.org/pdf/2401.12345"),
            )
            report = render_batch_report(store, batch.batch_id)
            self.assertIn("Screened: 1", report)
            self.assertIn("Allowed: 1", report)
            self.assertIn("Deep-scanned: 0", report)

    def test_batch_report_includes_candidate_metadata(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1000, mode="query")
            candidate = Candidate(
                provider="arxiv",
                title="Low SNR drone RF fingerprinting",
                source_for_gate="https://arxiv.org/pdf/2401.12345",
                doi="10.48550/arXiv.2401.12345",
                arxiv_id="2401.12345",
                pmcid="PMC1234567",
                year=2024,
                url="https://arxiv.org/abs/2401.12345",
                relevance_score=17,
                relevance_reason="not_biomedical",
                query_variant="MALDI antimicrobial resistance",
                query_intent="biomedical",
                acronym_expansions="AMR=antimicrobial resistance",
                journal="Journal of Clinical Microbiology",
                concepts="Mass spectrometry; Antimicrobial resistance",
                mesh_terms="Drug Resistance, Microbial",
                oa_status="gold",
                open_access_pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf/example.pdf",
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            report = render_batch_report(store, batch.batch_id)

            self.assertIn("arxiv", report)
            self.assertIn("Low SNR drone RF fingerprinting", report)
            self.assertIn("10.48550/arXiv.2401.12345", report)
            self.assertIn("pmcid=PMC1234567", report)
            self.assertIn("relevance=17", report)
            self.assertIn("not_biomedical", report)
            self.assertIn("query=MALDI antimicrobial resistance", report)
            self.assertIn("AMR=antimicrobial resistance", report)
            self.assertIn("journal=Journal of Clinical Microbiology", report)
            self.assertIn("mesh=Drug Resistance, Microbial", report)
            self.assertIn("concepts=Mass spectrometry; Antimicrobial resistance", report)
            self.assertIn("oa=gold", report)

    def test_batch_report_includes_pdf_artifacts_and_page_counts(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source="https://arxiv.org/pdf/2401.12345",
                pdf_url="https://arxiv.org/pdf/2401.12345",
                final_url="https://arxiv.org/pdf/2401.12345",
                sha256="b" * 64,
                byte_count=456,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page one text", "page two text"])
            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="The model achieved an AUROC of 0.91.",
                        page_number=2,
                    )
                ],
            )

            report = render_batch_report(store, batch.batch_id)

            self.assertIn("Parsed PDFs:", report)
            self.assertIn("stored: https://arxiv.org/pdf/2401.12345", report)
            self.assertIn("pages=2", report)
            self.assertIn("Parsed page-level paper text is stored.", report)
            self.assertIn("Extracted evidence:", report)
            self.assertIn("result p2: The model achieved an AUROC of 0.91.", report)
            self.assertIn("Cited Evidence Report", report)
            self.assertIn("- [P1 p2] The model achieved an AUROC of 0.91.", report)

    def test_scan_report_exports_markdown_and_json(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            scan = store.create_scan(
                "https://github.com/example/repo",
                evaluate_source("https://github.com/example/repo"),
            )

            markdown = render_scan_report_markdown(store, scan.scan_id)
            data = render_scan_report_json(store, scan.scan_id)

            self.assertIn("# Friday Scan Report", markdown)
            self.assertIn(f"- Scan ID: `{scan.scan_id}`", markdown)
            self.assertEqual(data["report_type"], "scan")
            self.assertEqual(data["scan"]["scan_id"], scan.scan_id)
            self.assertEqual(data["scan"]["status"], "blocked")
            self.assertEqual(data["scan"]["reason"], "blocked_domain")

    def test_batch_report_exports_markdown_and_json(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Direct AMR prediction from MALDI-TOF spectra",
                source_for_gate="10.1038/example",
                doi="10.1038/example",
                pmid="12345678",
                journal="Nature Medicine",
                year=2024,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source=candidate.source_for_gate,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="b" * 64,
                byte_count=456,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page one", "page two"])
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
                ],
            )

            markdown = render_batch_report_markdown(store, batch.batch_id)
            data = render_batch_report_json(store, batch.batch_id)

            self.assertIn("# Friday Batch Report", markdown)
            self.assertIn("## Cited Evidence", markdown)
            self.assertIn("- [P1 p2] The model achieved an AUROC of 0.91.", markdown)
            self.assertEqual(data["report_type"], "batch")
            self.assertEqual(data["batch"]["batch_id"], batch.batch_id)
            self.assertEqual(data["batch"]["screened_count"], 1)
            self.assertEqual(data["batch"]["allowed_count"], 1)
            self.assertEqual(data["cited_evidence"]["coverage"]["stored_pdfs"], 1)
            self.assertEqual(
                data["cited_evidence"]["paper_references"][0]["title"],
                "Direct AMR prediction from MALDI-TOF spectra",
            )
            self.assertEqual(
                data["cited_evidence"]["evidence"]["result"][0]["citation"],
                "P1 p2",
            )

    def test_batch_report_exports_screening_label_audit(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            relevant = Candidate(
                provider="openalex",
                title="MALDI-TOF antimicrobial resistance prediction",
                source_for_gate="10.1038/relevant",
                doi="10.1038/relevant",
                relevance_score=42,
            )
            irrelevant = Candidate(
                provider="arxiv",
                title="Abstract Meaning Representation parser",
                source_for_gate="https://arxiv.org/pdf/2201.11111",
                arxiv_id="2201.11111",
                relevance_score=12,
            )
            for candidate in (relevant, irrelevant):
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(
                batch.batch_id,
                relevant.source_for_gate,
                "relevant",
                note="human include",
            )
            store.set_screening_label(batch.batch_id, irrelevant.source_for_gate, "irrelevant")

            text = render_batch_report(store, batch.batch_id)
            markdown = render_batch_report_markdown(store, batch.batch_id)
            data = render_batch_report_json(store, batch.batch_id)

            self.assertEqual(data["screening_labels"]["counts"]["relevant"], 1)
            self.assertEqual(data["screening_labels"]["counts"]["irrelevant"], 1)
            self.assertEqual(data["screening_labels"]["labels"][0]["title"], relevant.title)
            self.assertEqual(data["screening_labels"]["labels"][0]["note"], "human include")
            self.assertEqual(data["items"][0]["screening_label"]["label"], "relevant")
            self.assertIn("Screening labels:", text)
            self.assertIn("relevant=1", text)
            self.assertIn("irrelevant=1", text)
            self.assertIn("## Screening Labels", markdown)
            self.assertIn("human include", markdown)

    def test_batch_report_data_filters_noisy_evidence(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            source = "10.1038/example"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source=source,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="c" * 64,
                byte_count=456,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="TOF MS produces singly charged ions TABLE 1 | Microbial detection methods.",
                        page_number=1,
                    ),
                    EvidenceItem(
                        evidence_type="result",
                        text="The model achieved an AUROC of 0.91.",
                        page_number=2,
                    ),
                ],
            )

            data = build_batch_report_data(store, batch.batch_id)

            result_texts = [
                item["text"]
                for item in data["cited_evidence"]["evidence"]["result"]
            ]
            self.assertEqual(result_texts, ["The model achieved an AUROC of 0.91."])


if __name__ == "__main__":
    unittest.main()
