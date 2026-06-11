import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_research.discovery import Candidate
from jarvis_research.label_review import build_label_review_rows
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


class LabelReviewTests(unittest.TestCase):
    def test_review_rows_prioritize_queue_candidates_and_filter_labels(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=5, mode="query")
            maybe = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/maybe",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=82,
            )
            unlabeled = Candidate(
                provider="openalex",
                title="MALDI antimicrobial resistance unlabeled",
                source_for_gate="10.1000/unlabeled",
                abstract="MALDI antimicrobial resistance metadata.",
                relevance_score=79,
            )
            human = Candidate(
                provider="arxiv",
                title="Human-reviewed antimicrobial resistance paper",
                source_for_gate="https://arxiv.org/pdf/2401.01001",
                abstract="MALDI antimicrobial resistance.",
                relevance_score=95,
            )
            blocked = Candidate(
                provider="openalex",
                title="Repository artifact",
                source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                abstract="Repository artifact.",
                relevance_score=99,
            )
            for candidate in (maybe, unlabeled, human, blocked):
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(
                batch.batch_id,
                maybe.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.61,
                rationale="uncertain metadata match",
            )
            store.set_screening_label(
                batch.batch_id,
                human.source_for_gate,
                "relevant",
                label_source="human",
                note="gold include",
            )

            rows = build_label_review_rows(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=10,
            )
            maybe_rows = build_label_review_rows(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                only="maybe",
                limit=10,
            )
            agent_rows = build_label_review_rows(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                only="agent",
                limit=10,
            )
            unlabeled_rows = build_label_review_rows(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                only="unlabeled",
                min_relevance=60,
                limit=10,
            )

            self.assertEqual(rows[0]["source"], maybe.source_for_gate)
            self.assertIn("heuristic_maybe_high_relevance", rows[0]["review_queue_reason"])
            self.assertEqual(rows[0]["label"], "maybe")
            self.assertEqual(rows[0]["label_source"], "agent")
            self.assertEqual(rows[0]["confidence"], 0.61)
            self.assertEqual(rows[1]["source"], unlabeled.source_for_gate)
            self.assertEqual(rows[1]["label_source"], "unlabeled")
            self.assertEqual(rows[-1]["label_source"], "human")
            self.assertNotIn(blocked.source_for_gate, {row["source"] for row in rows})
            self.assertEqual([row["source"] for row in maybe_rows], [maybe.source_for_gate])
            self.assertEqual([row["source"] for row in agent_rows], [maybe.source_for_gate])
            self.assertEqual([row["source"] for row in unlabeled_rows], [unlabeled.source_for_gate])


if __name__ == "__main__":
    unittest.main()
