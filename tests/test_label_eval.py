import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_research.discovery import Candidate
from jarvis_research.label_eval import build_label_evaluation
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


class LabelEvaluationTests(unittest.TestCase):
    def test_evaluates_agent_labels_against_human_overrides(self):
        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="MALDI AMR", limit=4, mode="query")
            correct = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance spectra",
                source_for_gate="10.1000/correct",
                abstract="MALDI spectra predict antimicrobial resistance.",
                relevance_score=78,
            )
            false_positive = Candidate(
                provider="openalex",
                title="AMR semantic graph parsing",
                source_for_gate="10.1000/false-positive",
                abstract="Abstract Meaning Representation graph parsing.",
                relevance_score=62,
            )
            false_negative = Candidate(
                provider="pubmed",
                title="MALDI antibiotic susceptibility testing",
                source_for_gate="10.1000/false-negative",
                abstract="Antibiotic susceptibility from mass spectrometry.",
                relevance_score=71,
            )
            no_agent_history = Candidate(
                provider="arxiv",
                title="Unreviewed human decision",
                source_for_gate="https://arxiv.org/pdf/2401.12345",
                relevance_score=40,
            )
            for candidate in [correct, false_positive, false_negative, no_agent_history]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(
                batch.batch_id,
                correct.source_for_gate,
                "relevant",
                note="agent=relevant confidence=0.74",
            )
            store.set_screening_label(
                batch.batch_id,
                false_positive.source_for_gate,
                "irrelevant",
                note="agent=relevant confidence=0.91",
            )
            store.set_screening_label(
                batch.batch_id,
                false_negative.source_for_gate,
                "relevant",
                note="agent=irrelevant confidence=0.83",
            )
            store.set_screening_label(batch.batch_id, no_agent_history.source_for_gate, "maybe")

            report = build_label_evaluation(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
            )

            self.assertEqual(report["human_label_counts"], {"relevant": 2, "maybe": 1, "irrelevant": 1})
            self.assertEqual(report["comparable_count"], 3)
            self.assertEqual(report["confusion_matrix"]["relevant"]["relevant"], 1)
            self.assertEqual(report["confusion_matrix"]["irrelevant"]["relevant"], 1)
            self.assertEqual(report["confusion_matrix"]["relevant"]["irrelevant"], 1)
            self.assertAlmostEqual(report["accuracy"], 1 / 3)
            self.assertAlmostEqual(report["precision"]["relevant"], 0.5)
            self.assertAlmostEqual(report["recall"]["relevant"], 0.5)
            self.assertEqual(report["disagreements"][0]["source"], false_positive.source_for_gate)
            self.assertEqual(report["disagreements"][0]["agent_confidence"], 0.91)
            self.assertEqual(report["high_confidence_mistakes"][0]["source"], false_positive.source_for_gate)
            self.assertEqual(report["recommendations"][0]["confidence"], 0.91)
            self.assertIn("review", report["recommendations"][0]["message"])

