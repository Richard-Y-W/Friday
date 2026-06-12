import unittest

from friday.discovery import Candidate
from friday.evidence import EvidenceItem
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class StorageTests(unittest.TestCase):
    def test_creates_scan_id_and_lists_scan(self):
        with self.subTest("scan persists and lists"):
            from tempfile import TemporaryDirectory
            from pathlib import Path

            with TemporaryDirectory() as tmp:
                store = FridayStore(Path(tmp) / "friday.db")
                scan = store.create_scan(
                    "https://arxiv.org/pdf/2401.12345",
                    evaluate_source("https://arxiv.org/pdf/2401.12345"),
                )
                self.assertTrue(scan.scan_id.startswith("scan_"))
                self.assertEqual(store.list_scans()[0].scan_id, scan.scan_id)

    def test_creates_batch_id_and_tracks_counts(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1000, mode="query")
            store.add_batch_item(
                batch.batch_id,
                "https://github.com/example/repo",
                evaluate_source("https://github.com/example/repo"),
            )
            loaded = store.get_batch(batch.batch_id)
            self.assertEqual(loaded.batch_id, batch.batch_id)
            self.assertEqual(loaded.blocked_count, 1)
            self.assertEqual(loaded.screened_count, 1)

    def test_persists_candidate_metadata_on_batch_items(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=10, mode="query")
            candidate = Candidate(
                provider="arxiv",
                title="Low SNR drone RF fingerprinting",
                source_for_gate="https://arxiv.org/pdf/2401.12345",
                doi="10.48550/arXiv.2401.12345",
                arxiv_id="2401.12345",
                pmcid="PMC1234567",
                year=2024,
                url="https://arxiv.org/abs/2401.12345",
                abstract="Drone RF fingerprinting from wireless signal data.",
                relevance_score=12,
                relevance_reason="not_biomedical",
                query_variant="MALDI antimicrobial resistance",
                query_intent="biomedical",
                acronym_expansions="AMR=antimicrobial resistance",
                journal="Journal of Clinical Microbiology",
                concepts="Mass spectrometry; Antimicrobial resistance",
                mesh_terms="Drug Resistance, Microbial; Mass Spectrometry",
                oa_status="gold",
                open_access_pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf/example.pdf",
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            item = store.list_batch_items(batch.batch_id)[0]

            self.assertEqual(item.provider, "arxiv")
            self.assertEqual(item.title, "Low SNR drone RF fingerprinting")
            self.assertEqual(item.doi, "10.48550/arXiv.2401.12345")
            self.assertEqual(item.arxiv_id, "2401.12345")
            self.assertEqual(item.pmcid, "PMC1234567")
            self.assertEqual(item.year, 2024)
            self.assertEqual(item.url, "https://arxiv.org/abs/2401.12345")
            self.assertEqual(item.abstract, "Drone RF fingerprinting from wireless signal data.")
            self.assertEqual(item.relevance_score, 12)
            self.assertEqual(item.relevance_reason, "not_biomedical")
            self.assertEqual(item.query_variant, "MALDI antimicrobial resistance")
            self.assertEqual(item.query_intent, "biomedical")
            self.assertEqual(item.acronym_expansions, "AMR=antimicrobial resistance")
            self.assertEqual(item.journal, "Journal of Clinical Microbiology")
            self.assertEqual(item.concepts, "Mass spectrometry; Antimicrobial resistance")
            self.assertEqual(item.mesh_terms, "Drug Resistance, Microbial; Mass Spectrometry")
            self.assertEqual(item.oa_status, "gold")
            self.assertEqual(
                item.open_access_pdf_url,
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/pdf/example.pdf",
            )

    def test_add_batch_item_if_new_does_not_increment_duplicate_sources(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=10, mode="query")
            decision = evaluate_source("https://arxiv.org/pdf/2401.12345")

            first = store.add_batch_item_if_new(batch.batch_id, "https://arxiv.org/pdf/2401.12345", decision)
            duplicate = store.add_batch_item_if_new(batch.batch_id, "https://arxiv.org/pdf/2401.12345", decision)
            loaded = store.get_batch(batch.batch_id)

            self.assertIsNotNone(first)
            self.assertIsNone(duplicate)
            self.assertEqual(loaded.screened_count, 1)
            self.assertEqual(len(store.list_batch_items(batch.batch_id)), 1)

    def test_persists_screening_labels_for_batch_items(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=10, mode="query")
            source = "https://arxiv.org/pdf/2401.12345"
            item = store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            label = store.set_screening_label(
                batch.batch_id,
                "HTTPS://ARXIV.ORG/pdf/2401.12345",
                "Relevant",
                note="matches MALDI AMR intent",
            )
            updated = store.set_screening_label(batch.batch_id, item.normalized, "irrelevant")

            labels = store.list_screening_labels(batch.batch_id)
            counts = store.screening_label_counts(batch.batch_id)

            self.assertEqual(label.normalized, item.normalized)
            self.assertEqual(label.source, source)
            self.assertEqual(label.label, "relevant")
            self.assertEqual(label.label_source, "human")
            self.assertEqual(labels[0].label, "irrelevant")
            self.assertEqual(labels[0].label_source, "human")
            self.assertEqual(updated.created_at, label.created_at)
            self.assertIsNone(labels[0].note)
            self.assertEqual(counts, {"irrelevant": 1})

            with self.assertRaises(KeyError):
                store.set_screening_label(batch.batch_id, "https://arxiv.org/pdf/9999.99999", "relevant")
            with self.assertRaises(ValueError):
                store.set_screening_label(batch.batch_id, source, "include")

    def test_agent_screening_labels_preserve_human_override(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="language math", limit=10, mode="query")
            source = "10.1038/example"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            agent = store.set_screening_label(
                batch.batch_id,
                source,
                "relevant",
                label_source="agent",
                confidence=0.82,
                rationale="query terms overlap with title",
                signals="language,math",
            )
            human = store.set_screening_label(batch.batch_id, source, "irrelevant", label_source="human")
            skipped = store.set_screening_label(
                batch.batch_id,
                source,
                "relevant",
                label_source="agent",
                confidence=0.95,
                overwrite_human=False,
            )

            labels = store.list_screening_labels(batch.batch_id)

            self.assertEqual(agent.label_source, "agent")
            self.assertEqual(agent.confidence, 0.82)
            self.assertEqual(agent.rationale, "query terms overlap with title")
            self.assertEqual(agent.signals, "language,math")
            self.assertEqual(human.label, "irrelevant")
            self.assertEqual(human.label_source, "human")
            self.assertIsNone(skipped)
            self.assertEqual(labels[0].label, "irrelevant")
            self.assertEqual(labels[0].label_source, "human")

    def test_creates_and_updates_research_run_ledger(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            run = store.create_research_run(
                query="MALDI AMR",
                limit=1000,
                deep_read_limit=50,
                min_relevance=25,
                auto_label_provider="heuristic",
                llm_review_limit=0,
            )
            batch = store.create_batch(query="MALDI AMR", limit=1000, mode="research_run")
            safe_source = "10.1038/example"
            blocked_source = "https://github.com/example/repo/blob/main/paper.pdf"
            store.add_batch_item(batch.batch_id, safe_source, evaluate_source(safe_source))
            store.add_batch_item(batch.batch_id, blocked_source, evaluate_source(blocked_source))
            store.add_pdf_artifact(
                batch.batch_id,
                source=safe_source,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/example.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )

            attached = store.update_research_run(
                run.run_id,
                batch_id=batch.batch_id,
                status="deep_reading",
            )
            synced = store.sync_research_run_counts(run.run_id)
            completed = store.update_research_run(run.run_id, status="complete")
            loaded = store.get_research_run(run.run_id)
            listed = store.list_research_runs()

            self.assertTrue(run.run_id.startswith("run_"))
            self.assertEqual(run.status, "created")
            self.assertEqual(attached.batch_id, batch.batch_id)
            self.assertEqual(synced.screened_count, 2)
            self.assertEqual(synced.blocked_count, 1)
            self.assertEqual(synced.allowed_count, 1)
            self.assertEqual(synced.deep_read_count, 1)
            self.assertEqual(completed.status, "complete")
            self.assertEqual(loaded.run_id, run.run_id)
            self.assertEqual(listed[0].run_id, run.run_id)

    def test_latest_research_run_returns_newest_run(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            first = store.create_research_run(
                query="first query",
                limit=10,
                deep_read_limit=1,
                min_relevance=25,
                auto_label_provider="heuristic",
                llm_review_limit=0,
            )
            second = store.create_research_run(
                query="second query",
                limit=20,
                deep_read_limit=2,
                min_relevance=30,
                auto_label_provider="heuristic",
                llm_review_limit=0,
            )

            latest = store.latest_research_run()

            self.assertEqual(latest.run_id, second.run_id)
            self.assertNotEqual(latest.run_id, first.run_id)

    def test_persists_pdf_artifacts_and_pages(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source="https://arxiv.org/pdf/2401.12345",
                pdf_url="https://arxiv.org/pdf/2401.12345",
                final_url="https://arxiv.org/pdf/2401.12345",
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page one", "page two"])

            loaded_batch = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)
            pages = store.list_pdf_pages(artifact.artifact_id)

            self.assertEqual(loaded_batch.deep_read_count, 1)
            self.assertEqual(artifacts[0].artifact_id, artifact.artifact_id)
            self.assertEqual(artifacts[0].sha256, "a" * 64)
            self.assertEqual(pages[0].page_number, 1)
            self.assertEqual(pages[0].char_count, 8)
            self.assertEqual(pages[1].text, "page two")

    def test_persists_evidence_records(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source="https://arxiv.org/pdf/2401.12345",
                pdf_url="https://arxiv.org/pdf/2401.12345",
                final_url="https://arxiv.org/pdf/2401.12345",
                sha256="c" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )

            records = store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="The model achieved an AUROC of 0.91.",
                        page_number=3,
                        parse_confidence=0.82,
                        parse_flags=("wide_spacing",),
                    )
                ],
            )
            loaded = store.list_evidence_records(artifact.artifact_id)

            self.assertEqual(records[0].artifact_id, artifact.artifact_id)
            self.assertEqual(loaded[0].evidence_type, "result")
            self.assertEqual(loaded[0].text, "The model achieved an AUROC of 0.91.")
            self.assertEqual(loaded[0].page_number, 3)
            self.assertEqual(loaded[0].char_count, 36)
            self.assertEqual(loaded[0].quality_label, "clean")
            self.assertEqual(loaded[0].quality_score, 1.0)
            self.assertEqual(loaded[0].quality_flags, ())
            self.assertEqual(loaded[0].parse_confidence, 0.82)
            self.assertEqual(loaded[0].parse_flags, ("wide_spacing",))

    def test_persists_blocked_evidence_quality_metadata(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source="https://arxiv.org/pdf/2401.12345",
                pdf_url="https://arxiv.org/pdf/2401.12345",
                final_url="https://arxiv.org/pdf/2401.12345",
                sha256="d" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/batch_1/paper.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )

            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="method",
                        text="Defense University of Malaysia), searches were carried out using Unfortunately, within 50 years.",
                        page_number=1,
                        quality_label="blocked",
                        quality_score=0.2,
                        quality_flags=("column_stitching",),
                    )
                ],
            )
            loaded = store.list_evidence_records(artifact.artifact_id)

            self.assertEqual(loaded[0].quality_label, "blocked")
            self.assertEqual(loaded[0].quality_score, 0.2)
            self.assertEqual(loaded[0].quality_flags, ("column_stitching",))


if __name__ == "__main__":
    unittest.main()
