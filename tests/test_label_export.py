import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_research.discovery import Candidate
from jarvis_research.label_export import (
    build_label_export_rows,
    render_label_export_csv,
    render_label_export_jsonl,
)
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


class LabelExportTests(unittest.TestCase):
    def test_builds_training_rows_with_gold_and_weak_labels(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            agent_candidate = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance classifier",
                source_for_gate="10.1000/agent",
                doi="10.1000/agent",
                pmid="12345",
                abstract="MALDI antimicrobial resistance prediction from spectra.",
                relevance_score=81,
                relevance_reason="query terms overlap",
                query_variant="MALDI antimicrobial resistance",
                query_intent="biomedical",
                journal="Clinical Microbiology",
                concepts="antimicrobial resistance;MALDI-TOF",
                mesh_terms="Drug Resistance, Microbial",
            )
            human_candidate = Candidate(
                provider="openalex",
                title="Abstract meaning representation parsing",
                source_for_gate="10.1000/human",
                doi="10.1000/human",
                abstract="Semantic parsing for natural language.",
                relevance_score=12,
                relevance_reason="weak metadata overlap",
            )
            blocked_candidate = Candidate(
                provider="openalex",
                title="Repository supplement",
                source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                abstract="Unsafe repository-hosted artifact.",
                relevance_score=5,
            )
            for candidate in (agent_candidate, human_candidate, blocked_candidate):
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(
                batch.batch_id,
                agent_candidate.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.62,
                rationale="metadata is promising but uncertain",
                signals="label_provider=heuristic",
            )
            store.set_screening_label(
                batch.batch_id,
                human_candidate.source_for_gate,
                "irrelevant",
                note="AMR means abstract meaning representation here",
                label_source="human",
            )
            store.set_screening_label(
                batch.batch_id,
                blocked_candidate.source_for_gate,
                "irrelevant",
                label_source="agent",
                confidence=0.99,
                rationale="blocked source",
                signals="source_policy=blocked",
            )

            rows = build_label_export_rows(store, batch_ids=[batch.batch_id])

            self.assertEqual(len(rows), 3)
            agent_row = next(row for row in rows if row["source"] == "10.1000/agent")
            human_row = next(row for row in rows if row["source"] == "10.1000/human")
            blocked_row = next(row for row in rows if row["source"].startswith("https://github.com"))
            self.assertEqual(agent_row["query"], "MALDI AMR")
            self.assertEqual(agent_row["title"], "MALDI antimicrobial resistance classifier")
            self.assertEqual(agent_row["abstract"], "MALDI antimicrobial resistance prediction from spectra.")
            self.assertEqual(agent_row["doi"], "10.1000/agent")
            self.assertEqual(agent_row["pmid"], "12345")
            self.assertEqual(agent_row["journal"], "Clinical Microbiology")
            self.assertEqual(agent_row["concepts"], "antimicrobial resistance;MALDI-TOF")
            self.assertEqual(agent_row["mesh_terms"], "Drug Resistance, Microbial")
            self.assertEqual(agent_row["relevance_score"], 81)
            self.assertEqual(agent_row["weak_label"], "maybe")
            self.assertIsNone(agent_row["gold_label"])
            self.assertEqual(agent_row["label_confidence"], 0.62)
            self.assertEqual(agent_row["label_rationale"], "metadata is promising but uncertain")
            self.assertIn("heuristic_maybe_high_relevance", agent_row["review_queue_reason"])
            self.assertGreater(agent_row["review_queue_score"], 0)
            self.assertEqual(human_row["gold_label"], "irrelevant")
            self.assertIsNone(human_row["weak_label"])
            self.assertEqual(human_row["label_note"], "AMR means abstract meaning representation here")
            self.assertIsNone(human_row["review_queue_reason"])
            self.assertFalse(blocked_row["allowed"])
            self.assertEqual(blocked_row["source_reason"], "blocked_domain")

    def test_renders_jsonl_and_csv(self):
        rows = [
            {
                "batch_id": "batch_a",
                "query": "language math",
                "source": "10.1000/a",
                "title": "Mathematical language",
                "label": "relevant",
                "label_source": "human",
                "gold_label": "relevant",
                "weak_label": None,
            }
        ]

        jsonl = render_label_export_jsonl(rows)
        csv_text = render_label_export_csv(rows)

        self.assertEqual(json.loads(jsonl)["gold_label"], "relevant")
        csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(csv_rows[0]["source"], "10.1000/a")
        self.assertEqual(csv_rows[0]["gold_label"], "relevant")
        self.assertEqual(csv_rows[0]["weak_label"], "")


if __name__ == "__main__":
    unittest.main()
