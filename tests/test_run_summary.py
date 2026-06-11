import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.evidence import EvidenceItem
from friday.run_summary import build_run_summary_dashboard, render_run_summary_text
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class RunSummaryDashboardTests(unittest.TestCase):
    def test_builds_dashboard_with_attention_items_and_next_commands(self):
        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            run = store.create_research_run(
                query="MALDI AMR",
                limit=1000,
                deep_read_limit=50,
                min_relevance=25,
                auto_label_provider="heuristic",
                llm_review_limit=10,
            )
            batch = store.create_batch(query="MALDI AMR", limit=1000, mode="research_run")
            store.update_research_run(run.run_id, batch_id=batch.batch_id, status="review")
            disagreement = Candidate(
                provider="pubmed",
                title="AMR semantic graph parsing",
                source_for_gate="10.1000/disagreement",
                abstract="Abstract Meaning Representation graph parsing.",
                relevance_score=88,
            )
            maybe = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/maybe",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=70,
            )
            unlabeled = Candidate(
                provider="openalex",
                title="MALDI high relevance unlabeled",
                source_for_gate="10.1000/unlabeled",
                abstract="MALDI antibiotic resistance metadata.",
                relevance_score=82,
            )
            blocked = Candidate(
                provider="openalex",
                title="Blocked repository paper",
                source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                relevance_score=90,
            )
            for candidate in [disagreement, maybe, unlabeled, blocked]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(
                batch.batch_id,
                disagreement.source_for_gate,
                "irrelevant",
                note="agent=relevant confidence=0.91",
            )
            store.set_screening_label(
                batch.batch_id,
                maybe.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.61,
            )
            stored = store.add_pdf_artifact(
                batch.batch_id,
                source=maybe.source_for_gate,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/example.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_evidence_records(
                stored.artifact_id,
                [EvidenceItem(evidence_type="result", text="The method detected resistance.", page_number=4)],
            )
            store.add_pdf_artifact(
                batch.batch_id,
                source=unlabeled.source_for_gate,
                pdf_url=None,
                final_url=None,
                sha256=None,
                byte_count=None,
                content_type=None,
                local_path=None,
                status="blocked",
                reason="no_safe_pdf_url",
            )
            store.sync_research_run_counts(run.run_id)

            summary = build_run_summary_dashboard(store, latest=True, limit=3)
            text = render_run_summary_text(summary)
            commands = [item["command"] for item in summary["next_commands"]]

            self.assertEqual(summary["artifact_type"], "run_summary_dashboard")
            self.assertEqual(summary["target"]["target_type"], "research_run")
            self.assertEqual(summary["target"]["run_id"], run.run_id)
            self.assertEqual(summary["target"]["batch_id"], batch.batch_id)
            self.assertEqual(summary["counts"]["screened"], 4)
            self.assertEqual(summary["counts"]["blocked"], 1)
            self.assertEqual(summary["counts"]["allowed"], 3)
            self.assertEqual(summary["counts"]["labeled"], 2)
            self.assertEqual(summary["counts"]["human_labels"], 1)
            self.assertEqual(summary["counts"]["agent_labels"], 1)
            self.assertEqual(summary["counts"]["unlabeled_allowed"], 1)
            self.assertEqual(summary["counts"]["stored_pdfs"], 1)
            self.assertEqual(summary["counts"]["failed_pdfs"], 1)
            self.assertEqual(summary["counts"]["evidence_items"], 1)
            self.assertEqual(summary["label_evaluation"]["comparable_count"], 1)
            self.assertGreaterEqual(summary["review_queue"]["queued_count"], 1)
            self.assertEqual(summary["attention"]["label_disagreements"][0]["source"], disagreement.source_for_gate)
            self.assertEqual(summary["attention"]["high_relevance_unlabeled"][0]["source"], unlabeled.source_for_gate)
            self.assertEqual(summary["attention"]["failed_pdfs"][0]["reason"], "no_safe_pdf_url")
            self.assertIn("friday labels review --latest --only maybe", commands)
            self.assertIn("friday labels eval --latest", commands)
            self.assertIn(f"friday research-run --resume-run {run.run_id}", commands)
            self.assertIn("Friday Run Summary", text)
            self.assertIn("Attention", text)
            self.assertIn("Next commands", text)
