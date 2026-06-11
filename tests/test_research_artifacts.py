import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.evidence import EvidenceItem
from friday.research_artifacts import (
    build_batch_passport,
    build_rejection_log,
    build_research_run_summary,
)
from friday.reporting import render_batch_report_json, render_batch_report_markdown
from friday.screening import build_llm_review_queue
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class ResearchArtifactTests(unittest.TestCase):
    def test_batch_passport_records_query_policy_and_repro_lock(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=1000, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="MALDI-TOF antimicrobial resistance prediction",
                source_for_gate="10.1038/example",
                doi="10.1038/example",
                abstract="Antimicrobial resistance prediction from MALDI spectra.",
                query_variant="MALDI antimicrobial resistance",
                query_intent="biomedical",
                acronym_expansions="AMR=antimicrobial resistance",
            )
            store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "relevant",
                note="human include",
            )

            passport = build_batch_passport(store, batch.batch_id, data_dir=Path(tmp) / ".friday")

            self.assertEqual(passport["artifact_type"], "batch_passport")
            self.assertEqual(passport["batch"]["batch_id"], batch.batch_id)
            self.assertEqual(passport["batch"]["query"], "MALDI AMR")
            self.assertIn("MALDI antimicrobial resistance", passport["query_plan"]["expanded_queries"])
            self.assertEqual(passport["source_policy"]["blocked_by_default"], ["github", "code", "archives"])
            self.assertEqual(passport["repro_lock"]["stochasticity_declaration"], "Live scholarly APIs and LLM outputs are not byte-reproducible. This lock documents configuration, not deterministic replay.")
            self.assertIn("friday_commit", passport["repro_lock"])
            self.assertEqual(passport["screening_labels"]["counts"]["relevant"], 1)
            self.assertEqual(passport["screening_labels"]["labels"][0]["source"], candidate.source_for_gate)
            self.assertEqual(passport["screening_labels"]["labels"][0]["note"], "human include")

    def test_rejection_log_records_blocked_sources_and_failed_pdfs(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            blocked_source = "https://github.com/example/repo/blob/main/paper.pdf"
            safe_source = "10.1038/example"
            store.add_batch_item(batch.batch_id, blocked_source, evaluate_source(blocked_source))
            store.add_batch_item(batch.batch_id, safe_source, evaluate_source(safe_source))
            store.add_pdf_artifact(
                batch.batch_id,
                source=safe_source,
                pdf_url=None,
                final_url=None,
                sha256=None,
                byte_count=None,
                content_type=None,
                local_path=None,
                status="blocked",
                reason="no_safe_pdf_url",
            )

            rejection_log = build_rejection_log(store, batch.batch_id)

            reasons = {(item["source"], item["stage"], item["reason"]) for item in rejection_log["rejected"]}
            self.assertIn((blocked_source, "source_gate", "blocked_domain"), reasons)
            self.assertIn((safe_source, "pdf_ingestion", "no_safe_pdf_url"), reasons)

    def test_research_run_summary_records_status_counts_and_policy(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            run = store.create_research_run(
                query="MALDI AMR",
                limit=1000,
                deep_read_limit=50,
                min_relevance=25,
                auto_label_provider="heuristic",
                llm_review_limit=0,
            )
            batch = store.create_batch(query="MALDI AMR", limit=1000, mode="research_run")
            candidate = Candidate(
                provider="openalex",
                title="MALDI-TOF antimicrobial resistance prediction",
                source_for_gate="10.1038/example",
                doi="10.1038/example",
                abstract="Antimicrobial resistance prediction from MALDI spectra.",
            )
            store.update_research_run(run.run_id, batch_id=batch.batch_id, status="labeling")
            store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "relevant",
                label_source="agent",
                confidence=0.88,
                rationale="metadata strongly matches query",
                signals="query_overlap=maldi,amr",
            )
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source=candidate.source_for_gate,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="b" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/example.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="The method identified resistant isolates.",
                        page_number=4,
                    )
                ],
            )
            store.sync_research_run_counts(run.run_id)
            store.update_research_run(run.run_id, status="complete")

            summary = build_research_run_summary(store, run.run_id, data_dir=data_dir)

            self.assertEqual(summary["artifact_type"], "research_run_summary")
            self.assertEqual(summary["run"]["run_id"], run.run_id)
            self.assertEqual(summary["run"]["status"], "complete")
            self.assertEqual(summary["run"]["deep_read_limit"], 50)
            self.assertEqual(summary["batch"]["batch_id"], batch.batch_id)
            self.assertEqual(summary["batch"]["screened_count"], 1)
            self.assertEqual(summary["batch"]["deep_read_count"], 1)
            self.assertEqual(summary["artifacts"]["stored_pdf_count"], 1)
            self.assertEqual(summary["screening_labels"]["counts"]["relevant"], 1)
            self.assertEqual(summary["source_policy"]["blocked_by_default"], ["github", "code", "archives"])
            self.assertEqual(summary["repro_lock"]["local_state"]["data_dir"], str(data_dir))

    def test_run_artifacts_include_llm_review_queue(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            run = store.create_research_run(
                query="language math",
                limit=2,
                deep_read_limit=0,
                min_relevance=25,
                auto_label_provider="llm",
                llm_review_limit=1,
            )
            batch = store.create_batch(query="language math", limit=2, mode="research_run")
            store.update_research_run(run.run_id, batch_id=batch.batch_id, status="labeling")
            candidate = Candidate(
                provider="openalex",
                title="Language and cognition",
                source_for_gate="10.1000/queue",
                abstract="Language learning and symbolic reasoning.",
                relevance_score=70,
            )
            store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.62,
                rationale="metadata partially matches query",
                signals="provider=heuristic",
            )
            queue = build_llm_review_queue(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=1,
            )

            passport = build_batch_passport(
                store,
                batch.batch_id,
                data_dir=data_dir,
                llm_review_queue=queue,
            )
            summary = build_research_run_summary(
                store,
                run.run_id,
                data_dir=data_dir,
                llm_review_queue=queue,
            )

            self.assertEqual(passport["llm_review_queue"]["items"][0]["source"], candidate.source_for_gate)
            self.assertEqual(summary["llm_review_queue"]["items"][0]["source"], candidate.source_for_gate)
            self.assertIn(
                "heuristic_maybe_high_relevance",
                summary["llm_review_queue"]["items"][0]["reason"],
            )

    def test_batch_report_contains_claim_support_audit(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            source = "10.1038/example"
            candidate = Candidate(
                provider="openalex",
                title="MALDI AMR paper",
                source_for_gate=source,
                doi=source,
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
                        evidence_type="result",
                        text="The model achieved an AUROC of 0.91.",
                        page_number=2,
                    )
                ],
            )

            data = render_batch_report_json(store, batch.batch_id)
            markdown = render_batch_report_markdown(store, batch.batch_id)

            self.assertEqual(data["claim_support_audit"]["counts"]["supported"], 1)
            self.assertEqual(data["claim_support_audit"]["supported_claims"][0]["citation"], "P1 p2")
            self.assertIn("## Claim Support Audit", markdown)
            self.assertIn("SUPPORTED [P1 p2]", markdown)


if __name__ == "__main__":
    unittest.main()
