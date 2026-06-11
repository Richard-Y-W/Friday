import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_research.cited_report import render_cited_evidence_report
from jarvis_research.discovery import Candidate
from jarvis_research.evidence import EvidenceItem
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


class CitedEvidenceReportTests(unittest.TestCase):
    def test_renders_paper_references_and_page_citations(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Direct AMR prediction from MALDI-TOF spectra",
                source_for_gate="10.1038/example",
                doi="10.1038/example",
                pmid="12345678",
                pmcid="PMC1234567",
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
                sha256="d" * 64,
                byte_count=789,
                content_type="application/pdf",
                local_path="artifacts/batch_1/example.pdf",
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
                    EvidenceItem(
                        evidence_type="limitation",
                        text="A limitation is that isolates came from one center.",
                        page_number=2,
                    ),
                ],
            )

            report = render_cited_evidence_report(store, batch.batch_id)

            self.assertIn("Cited Evidence Report", report)
            self.assertIn("Coverage: screened=1; allowed=1; blocked=0; deep-read=1; stored-pdfs=1", report)
            self.assertIn(
                "[P1] Direct AMR prediction from MALDI-TOF spectra; doi=10.1038/example; pmid=12345678; pmcid=PMC1234567; journal=Nature Medicine; year=2024",
                report,
            )
            self.assertIn("Methods:", report)
            self.assertIn("- [P1 p1] We used MALDI-TOF spectra to train a classifier.", report)
            self.assertIn("Results:", report)
            self.assertIn("- [P1 p2] The model achieved an AUROC of 0.91.", report)
            self.assertIn("Limitations:", report)
            self.assertIn("- [P1 p2] A limitation is that isolates came from one center.", report)
            self.assertIn("Evidence Gaps:", report)
            self.assertIn("- Deep-read coverage: 1 of 1 screened records produced stored PDFs.", report)

    def test_reports_when_no_evidence_is_available(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            store.add_batch_item(
                batch.batch_id,
                "10.1038/example",
                evaluate_source("10.1038/example"),
            )

            report = render_cited_evidence_report(store, batch.batch_id)

            self.assertIn("Cited Evidence Report", report)
            self.assertIn("No extracted evidence is available for this batch yet.", report)
            self.assertIn("Coverage: screened=1; allowed=1; blocked=0; deep-read=0; stored-pdfs=0", report)

    def test_filters_noisy_legacy_evidence_from_report(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="MALDI AMR paper",
                source_for_gate="10.1038/example",
                doi="10.1038/example",
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
                sha256="e" * 64,
                byte_count=789,
                content_type="application/pdf",
                local_path="artifacts/batch_1/example.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="TOF MS produces singly charged ions, thus interpretation of However, automation TABLE 1 | Microbial detection methods.",
                        page_number=4,
                    ),
                    EvidenceItem(
                        evidence_type="result",
                        text="The model achieved an AUROC of 0.91 in validation.",
                        page_number=5,
                    ),
                ],
            )

            report = render_cited_evidence_report(store, batch.batch_id)

            self.assertNotIn("TABLE 1", report)
            self.assertNotIn("interpretation of However", report)
            self.assertIn("- [P1 p5] The model achieved an AUROC of 0.91 in validation.", report)


if __name__ == "__main__":
    unittest.main()
