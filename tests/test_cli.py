import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from friday.discovery import Candidate
from friday.cli import main
from friday.llm_labeling import LlmLabelResult
from friday.pdf_ingestion import PdfIngestionResult


class CliTests(unittest.TestCase):
    def run_cli(self, args, tmp_path):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([*args, "--data-dir", str(tmp_path / ".friday")])
        return code, out.getvalue()

    def test_scan_prints_scan_id_and_report_command(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["scan", "https://arxiv.org/pdf/2401.12345"], Path(tmp))
            self.assertEqual(code, 0)
            self.assertIn("Scan ID: scan_", output)
            self.assertIn("friday report scan_", output)

    def test_query_scan_prints_batch_id(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1000"], Path(tmp))
            self.assertEqual(code, 0)
            self.assertIn("Batch ID: batch_", output)
            self.assertIn("friday report batch_", output)

    def test_settings_show_and_set_research_defaults(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            code, output = self.run_cli(["/settings"], tmp_path)
            set_code, set_output = self.run_cli(["settings", "set", "research.limit", "250"], tmp_path)
            updated_code, updated_output = self.run_cli(["/setting"], tmp_path)

            self.assertEqual(code, 0)
            self.assertIn("research.limit: 100", output)
            self.assertEqual(set_code, 0)
            self.assertIn("research.limit: 250", set_output)
            self.assertEqual(updated_code, 0)
            self.assertIn("research.limit: 250", updated_output)
            self.assertIn("research.deep_read_limit: 10", updated_output)

    def test_llm_status_lists_role_wiring(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["llm", "status"], Path(tmp))

        self.assertEqual(code, 0)
        self.assertIn("Friday LLM role wiring", output)
        self.assertIn("composer", output)
        self.assertIn("claude_cli", output)
        self.assertIn("verifier", output)
        self.assertIn("codex_cli", output)

    def test_llm_test_requires_wired_role(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["settings", "set", "llm.composer_provider", "none"], tmp_path)

            code, output = self.run_cli(["llm", "test", "--role", "composer"], tmp_path)

        self.assertEqual(code, 2)
        self.assertIn("role 'composer' is not wired to a provider", output)

    def test_plain_friday_non_interactive_keeps_help_output(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([], force_interactive=False)

        output = out.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("usage: friday", output)
        self.assertNotIn("Friday Research", output)

    def test_plain_friday_interactive_shell_handles_settings_and_exit(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    ["--data-dir", str(data_dir)],
                    input_stream=io.StringIO("/settings\n/exit\n"),
                    force_interactive=True,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("\x1b[38;2;130;200;229m", output)
            self.assertIn("\x1b[0m        friday research 1.0.0", output)
            self.assertIn("Paper scanner - cited PDF reports", output)
            self.assertIn("friday>", output)
            self.assertIn("Friday settings", output)
            self.assertIn("research.limit: 100", output)

    def test_plain_friday_interactive_shell_prints_cyan_lens_splash_with_plain_title(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    ["--data-dir", str(Path(tmp) / ".friday")],
                    input_stream=io.StringIO("/exit\n"),
                    force_interactive=True,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("\x1b[38;2;130;200;229m", output)
            self.assertIn("▄███████▄", output)
            self.assertIn("\x1b[0m        friday research 1.0.0", output)
            self.assertNotIn("Friday Research v0.1.0", output)
            self.assertIn("Scholarly-only evidence assistant", output)
            self.assertIn("\x1b[0m", output)

    def test_interactive_shell_natural_query_writes_report_package(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "tell me about MALDI AMR")
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="MALDI-TOF spectra identify antimicrobial resistance patterns.",
                    relevance_score=80,
                )
            ][:limit]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="1" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="MALDI-TOF spectra identified antimicrobial resistance patterns.",
                        page_number=2,
                    )
                ],
            )
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            desktop_report_dir = Path(tmp) / "Desktop" / "FridayReports"
            out = io.StringIO()
            with patch.dict(os.environ, {"FRIDAY_DESKTOP_REPORT_DIR": str(desktop_report_dir)}), redirect_stdout(out):
                code = main(
                    ["--data-dir", str(data_dir)],
                    input_stream=io.StringIO("friday tell me about MALDI AMR\n/exit\n"),
                    force_interactive=True,
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            package_line = next(
                line for line in output.splitlines() if line.startswith("Report package: ")
            )
            package_dir = Path(package_line.split(": ", 1)[1])
            self.assertEqual(code, 0)
            self.assertIn("Natural query route: scholarly", output)
            self.assertTrue((package_dir / "report.md").exists())
            self.assertTrue((package_dir / "report.pdf").exists())
            self.assertTrue((package_dir / "evidence_table.csv").exists())
            self.assertIn("## Executive Summary", (package_dir / "report.md").read_text(encoding="utf-8"))
            desktop_pdf = desktop_report_dir / f"{package_dir.name}.pdf"
            self.assertTrue(desktop_pdf.exists())
            self.assertEqual(desktop_pdf.read_bytes(), (package_dir / "report.pdf").read_bytes())
            self.assertIn(f"Desktop PDF: {desktop_pdf}", output)

    def test_query_scan_stores_discovered_candidates(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "MALDI AMR")
            self.assertEqual(limit, 2)
            return [
                Candidate(
                    provider="arxiv",
                    title="Safe scholarly paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    year=2024,
                    url="https://arxiv.org/abs/2401.12345",
                ),
                Candidate(
                    provider="openalex",
                    title="Blocked repo paper",
                    source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                    url="https://github.com/example/repo/blob/main/paper.pdf",
                ),
            ]

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                )
            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Screened: 2", output)
            self.assertIn("Blocked: 1", output)
            self.assertIn("Allowed: 1", output)
            self.assertIn("Deep-scanned: 0", output)

    def test_unknown_command_routes_to_natural_research_using_settings(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "what is the importance of math in language")
            self.assertEqual(limit, 2)
            return [
                Candidate(
                    provider="openalex",
                    title="Mathematical structures in formal language theory",
                    source_for_gate="10.1038/language-math",
                    abstract="Language syntax can be modeled with algebra and formal grammars.",
                    relevance_score=12,
                )
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url="https://www.nature.com/articles/language-math.pdf",
                final_url="https://www.nature.com/articles/language-math.pdf",
                sha256="d" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url="https://www.nature.com/articles/language-math.pdf",
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["/settings", "set", "research.limit", "2"], tmp_path)
            self.run_cli(["/settings", "set", "research.deep_read_limit", "1"], tmp_path)

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "what",
                        "is",
                        "the",
                        "importance",
                        "of",
                        "math",
                        "in",
                        "language",
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertEqual(deep_read_sources, ["10.1038/language-math"])
            self.assertIn("Natural query: what is the importance of math in language", output)
            self.assertIn("# Friday Batch Report", output)
            self.assertIn("Screening Labels", output)

    def test_query_scan_deep_reads_limited_safe_candidates(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="First MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.11111",
                    arxiv_id="2401.11111",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
                Candidate(
                    provider="arxiv",
                    title="Second MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.22222",
                    arxiv_id="2401.22222",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="c" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertEqual(deep_read_sources, ["https://arxiv.org/pdf/2401.11111"])
            self.assertIn("Screened: 2", output)
            self.assertIn("Allowed: 2", output)
            self.assertIn("Deep-scanned: 1", output)

    def test_query_scan_does_not_deep_read_below_relevance_threshold(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="Pushing the Limits of AMR Parsing with Self-Learning",
                    source_for_gate="https://arxiv.org/pdf/2010.10673v1",
                    arxiv_id="2010.10673v1",
                    abstract="Abstract Meaning Representation parsing for natural language.",
                )
            ]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            raise AssertionError("low-relevance candidate should not be deep-read")

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Deep-scanned: 0", output)

    def test_query_scan_deep_reads_highest_relevance_candidate_first(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="Pushing the Limits of AMR Parsing with Self-Learning",
                    source_for_gate="https://arxiv.org/pdf/2010.10673v1",
                    arxiv_id="2010.10673v1",
                    abstract="Abstract Meaning Representation parsing for natural language.",
                ),
                Candidate(
                    provider="arxiv",
                    title="MALDI-TOF antimicrobial resistance prediction in Pseudomonas aeruginosa",
                    source_for_gate="https://arxiv.org/pdf/2401.55555v1",
                    arxiv_id="2401.55555v1",
                    abstract="Antimicrobial resistance prediction from MALDI-TOF spectra.",
                ),
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="d" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            self.assertEqual(code, 0)
            self.assertEqual(deep_read_sources, ["https://arxiv.org/pdf/2401.55555v1"])

    def test_query_scan_blocked_deep_read_does_not_consume_success_limit(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="pubmed",
                    title="MALDI antimicrobial resistance DOI-only result",
                    source_for_gate="10.1038/example",
                    doi="10.1038/example",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance PDF result",
                    source_for_gate="https://arxiv.org/pdf/2401.77777v1",
                    arxiv_id="2401.77777v1",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            if source.startswith("10."):
                artifact = store.add_pdf_artifact(
                    batch_id,
                    source=source,
                    pdf_url=None,
                    final_url=None,
                    sha256=None,
                    byte_count=None,
                    content_type=None,
                    local_path=None,
                    status="blocked",
                    reason="no_safe_pdf_url",
                )
                return PdfIngestionResult(
                    status="blocked",
                    reason="no_safe_pdf_url",
                    artifact_id=artifact.artifact_id,
                    pdf_url=None,
                    page_count=0,
                )
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="e" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertEqual(deep_read_sources, ["10.1038/example", "https://arxiv.org/pdf/2401.77777v1"])
            self.assertIn("Deep-scanned: 1", output)

    def test_query_scan_resume_batch_adds_only_new_candidates(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        candidates = [
            Candidate(
                provider="arxiv",
                title=f"MALDI antimicrobial resistance paper {index}",
                source_for_gate=f"https://arxiv.org/pdf/2401.9000{index}v1",
                arxiv_id=f"2401.9000{index}v1",
                abstract="Antimicrobial resistance prediction from MALDI spectra.",
            )
            for index in range(1, 4)
        ]

        def fake_discoverer(query, limit):
            self.assertEqual(query, "MALDI AMR")
            return candidates[:limit]

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            first_out = io.StringIO()
            with redirect_stdout(first_out):
                first_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            self.assertEqual(first_code, 0)
            batch_id = next(
                line.split(": ", 1)[1]
                for line in first_out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )

            second_out = io.StringIO()
            with redirect_stdout(second_out):
                second_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "3",
                        "--resume-batch",
                        batch_id,
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )

            output = second_out.getvalue()
            self.assertEqual(second_code, 0)
            self.assertIn(f"Resumed: {batch_id}", output)
            self.assertIn("Screened: 3", output)
            self.assertIn("Allowed: 3", output)

    def test_query_scan_resume_deep_reads_next_ranked_candidate_to_target(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        candidates = [
            Candidate(
                provider="arxiv",
                title=f"MALDI-TOF antimicrobial resistance paper {index}",
                source_for_gate=f"https://arxiv.org/pdf/2401.9100{index}v1",
                arxiv_id=f"2401.9100{index}v1",
                abstract="Antimicrobial resistance prediction from MALDI-TOF spectra.",
            )
            for index in range(1, 4)
        ]

        def fake_discoverer(query, limit):
            return candidates[:limit]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="f" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            first_out = io.StringIO()
            with redirect_stdout(first_out):
                first_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "3",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )
            self.assertEqual(first_code, 0)
            batch_id = next(
                line.split(": ", 1)[1]
                for line in first_out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )

            second_out = io.StringIO()
            with redirect_stdout(second_out):
                second_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "3",
                        "--deep-read-limit",
                        "2",
                        "--resume-batch",
                        batch_id,
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            self.assertEqual(second_code, 0)
            self.assertEqual(
                deep_read_sources,
                [
                    "https://arxiv.org/pdf/2401.91001v1",
                    "https://arxiv.org/pdf/2401.91002v1",
                ],
            )
            self.assertIn("Deep-scanned: 2", second_out.getvalue())

    def test_query_scan_resume_uses_screening_labels_for_deep_read_order(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        candidates = [
            Candidate(
                provider="arxiv",
                title="Pushing the Limits of AMR Parsing with Self-Learning",
                source_for_gate="https://arxiv.org/pdf/2010.10673v1",
                arxiv_id="2010.10673v1",
                abstract="Abstract Meaning Representation parsing for natural language.",
            ),
            Candidate(
                provider="arxiv",
                title="MALDI-TOF antimicrobial resistance prediction in Pseudomonas aeruginosa",
                source_for_gate="https://arxiv.org/pdf/2401.55555v1",
                arxiv_id="2401.55555v1",
                abstract="Antimicrobial resistance prediction from MALDI-TOF spectra.",
            ),
        ]

        def fake_discoverer(query, limit):
            return candidates[:limit]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="b" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            first_out = io.StringIO()
            with redirect_stdout(first_out):
                first_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            self.assertEqual(first_code, 0)
            batch_id = next(
                line.split(": ", 1)[1]
                for line in first_out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )

            relevant_code, relevant_output = self.run_cli(
                [
                    "label",
                    "--batch-id",
                    batch_id,
                    "--source",
                    "https://arxiv.org/pdf/2010.10673v1",
                    "--label",
                    "relevant",
                    "--note",
                    "human override",
                ],
                Path(tmp),
            )
            irrelevant_code, _ = self.run_cli(
                [
                    "label",
                    "--batch-id",
                    batch_id,
                    "--source",
                    "https://arxiv.org/pdf/2401.55555v1",
                    "--label",
                    "irrelevant",
                ],
                Path(tmp),
            )
            labels_code, labels_output = self.run_cli(["labels", "--batch-id", batch_id], Path(tmp))

            second_out = io.StringIO()
            with redirect_stdout(second_out):
                second_code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "1",
                        "--resume-batch",
                        batch_id,
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            self.assertEqual(relevant_code, 0)
            self.assertIn("Labeled: relevant", relevant_output)
            self.assertEqual(irrelevant_code, 0)
            self.assertEqual(labels_code, 0)
            self.assertIn("relevant=1", labels_output)
            self.assertIn("irrelevant=1", labels_output)
            self.assertEqual(second_code, 0)
            self.assertEqual(deep_read_sources, ["https://arxiv.org/pdf/2010.10673v1"])
            self.assertIn("Deep-scanned: 1", second_out.getvalue())

    def test_auto_label_command_applies_agent_labels_without_overwriting_human_labels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        candidates = [
            Candidate(
                provider="openalex",
                title="Mathematical structures in formal language theory",
                source_for_gate="10.1038/language-math",
                abstract="Language syntax can be modeled with algebra and formal grammars.",
                relevance_score=18,
            ),
            Candidate(
                provider="arxiv",
                title="Abstract Meaning Representation parser",
                source_for_gate="https://arxiv.org/pdf/2201.11111",
                abstract="Semantic graph parsing for natural language.",
                relevance_score=8,
            ),
        ]

        def fake_discoverer(query, limit):
            return candidates[:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "language math",
                        "--limit",
                        "2",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            self.assertEqual(code, 0)
            batch_id = next(
                line.split(": ", 1)[1]
                for line in out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )

            self.run_cli(
                [
                    "label",
                    "--batch-id",
                    batch_id,
                    "--source",
                    "https://arxiv.org/pdf/2201.11111",
                    "--label",
                    "irrelevant",
                ],
                tmp_path,
            )
            auto_code, auto_output = self.run_cli(["auto-label", "--batch-id", batch_id, "--apply"], tmp_path)
            report_path = tmp_path / "report.json"
            self.run_cli(["report", batch_id, "--format", "json", "--output", str(report_path)], tmp_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))

            labels = {label["source"]: label for label in report["screening_labels"]["labels"]}
            self.assertEqual(auto_code, 0)
            self.assertIn("Applied: 1", auto_output)
            self.assertIn("Skipped human labels: 1", auto_output)
            self.assertEqual(labels["10.1038/language-math"]["label_source"], "agent")
            self.assertGreaterEqual(labels["10.1038/language-math"]["confidence"], 0.65)
            self.assertEqual(labels["https://arxiv.org/pdf/2201.11111"]["label_source"], "human")

    def test_auto_label_command_can_use_llm_provider_with_injected_client(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FakeLlmClient:
            def __init__(self):
                self.calls = []

            def label(self, *, query, item, model):
                self.calls.append((query, item.source, model))
                return LlmLabelResult(
                    label="maybe",
                    confidence=0.72,
                    rationale="LLM sees partial query alignment.",
                    evidence_terms=("language",),
                    exclusion_reason="Missing explicit mathematical structure.",
                )

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="openalex",
                    title="Language and mathematics education",
                    source_for_gate="10.1000/llm-cli",
                    abstract="Language models appear in mathematics classrooms.",
                    relevance_score=17,
                )
            ]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "math in language",
                        "--limit",
                        "1",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            self.assertEqual(code, 0)
            batch_id = next(
                line.split(": ", 1)[1]
                for line in out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )
            fake_client = FakeLlmClient()
            auto_out = io.StringIO()
            with redirect_stdout(auto_out):
                auto_code = main(
                    [
                        "auto-label",
                        "--batch-id",
                        batch_id,
                        "--provider",
                        "llm",
                        "--model",
                        "gpt-test",
                        "--apply",
                        "--data-dir",
                        str(data_dir),
                    ],
                    llm_label_client=fake_client,
                )
            report_path = tmp_path / "report.json"
            self.run_cli(["report", batch_id, "--format", "json", "--output", str(report_path)], tmp_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            labels = report["screening_labels"]["labels"]

            self.assertEqual(auto_code, 0)
            self.assertIn("Provider: llm", auto_out.getvalue())
            self.assertIn("Model: gpt-test", auto_out.getvalue())
            self.assertIn("Applied: 1", auto_out.getvalue())
            self.assertEqual(fake_client.calls, [("math in language", "10.1000/llm-cli", "gpt-test")])
            self.assertEqual(labels[0]["label"], "maybe")
            self.assertIn("label_provider=llm", labels[0]["signals"])

    def test_natural_query_uses_saved_llm_auto_label_settings(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FakeLlmClient:
            def __init__(self):
                self.calls = []

            def label(self, *, query, item, model):
                self.calls.append((query, item.source, model))
                return LlmLabelResult(
                    label="relevant",
                    confidence=0.9,
                    rationale="LLM sees strong metadata alignment.",
                    evidence_terms=("mathematical linguistics",),
                    exclusion_reason=None,
                )

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="openalex",
                    title="Mathematical Linguistics",
                    source_for_gate="10.1000/natural-llm",
                    abstract="Formal language theory and syntax.",
                    relevance_score=17,
                )
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            self.run_cli(["/settings", "set", "auto_label.provider", "llm"], tmp_path)
            self.run_cli(["/settings", "set", "auto_label.model", "gpt-test"], tmp_path)
            fake_client = FakeLlmClient()
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "what",
                        "is",
                        "the",
                        "importance",
                        "of",
                        "math",
                        "in",
                        "language",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "0",
                    ],
                    discoverer=fake_discoverer,
                    llm_label_client=fake_client,
                )
            batch_id = next(
                line.split(": ", 1)[1]
                for line in out.getvalue().splitlines()
                if line.startswith("Batch ID:")
            )
            report_path = tmp_path / "natural-report.json"
            self.run_cli(["report", batch_id, "--format", "json", "--output", str(report_path)], tmp_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            labels = report["screening_labels"]["labels"]

            self.assertEqual(code, 0)
            self.assertEqual(fake_client.calls, [("what is the importance of math in language", "10.1000/natural-llm", "gpt-test")])
            self.assertEqual(labels[0]["label"], "relevant")
            self.assertIn("label_provider=llm", labels[0]["signals"])

    def test_natural_query_routes_to_configured_corpus_when_relevant(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            corpus_path = tmp_path / "corpus.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "literature_corpus",
                        "literature_corpus": [
                            {
                                "citation_key": "smith2024mathlang",
                                "title": "Mathematical structure in natural language",
                                "abstract": "Formal grammars and algebraic syntax model language.",
                                "venue": "Linguistics and Philosophy",
                                "tags": ["formal language", "mathematics"],
                                "source_pointer": "https://doi.org/10.1000/math-lang",
                                "source_type": "zotero",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            set_code, set_output = self.run_cli(["settings", "set", "corpus.paths", str(corpus_path)], tmp_path)

            discoverer_calls = []

            def fake_discoverer(query, limit):
                discoverer_calls.append((query, limit))
                return []

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "friday",
                        "tell",
                        "me",
                        "about",
                        "math",
                        "in",
                        "language",
                    ],
                    discoverer=fake_discoverer,
                )

            output = out.getvalue()
            self.assertEqual(set_code, 0, set_output)
            self.assertEqual(code, 0)
            self.assertEqual(discoverer_calls, [])
            self.assertIn("Natural query route: corpus", output)
            self.assertIn("# Friday Corpus Report", output)
            self.assertIn("Mathematical structure in natural language", output)
            self.assertNotIn("Batch ID:", output)
            self.assertFalse((data_dir / "friday.db").exists())

    def test_natural_query_write_outputs_evidence_bound_literature_review(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "friday tell me about MALDI AMR")
            self.assertEqual(limit, 1)
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI-TOF antimicrobial resistance prediction",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="MALDI-TOF spectra identify antimicrobial resistance patterns.",
                    relevance_score=80,
                )
            ]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="d" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="MALDI-TOF spectra identified antimicrobial resistance patterns.",
                        page_number=2,
                    )
                ],
            )
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "friday",
                        "tell",
                        "me",
                        "about",
                        "MALDI",
                        "AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--min-relevance",
                        "0",
                        "--write",
                        "--write-mode",
                        "literature-review",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("Natural query route: scholarly", output)
            self.assertIn("# Friday Batch Report", output)
            self.assertIn("Writing mode: literature-review", output)
            self.assertIn("# Evidence-Bound Literature Review Draft", output)
            self.assertIn("This draft uses only page-anchored extracted evidence.", output)
            self.assertIn(
                "Result evidence: MALDI-TOF spectra identified antimicrobial resistance patterns. [P1 p2]",
                output,
            )

    def test_natural_query_write_output_keeps_report_output_separate(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI-TOF antimicrobial resistance prediction",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="MALDI-TOF spectra identify antimicrobial resistance patterns.",
                    relevance_score=80,
                )
            ][:limit]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="e" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

            store.add_evidence_records(
                artifact.artifact_id,
                [
                    EvidenceItem(
                        evidence_type="result",
                        text="MALDI-TOF spectra identified antimicrobial resistance patterns.",
                        page_number=2,
                    )
                ],
            )
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.json"
            draft_path = tmp_path / "draft.md"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "tell",
                        "me",
                        "about",
                        "MALDI",
                        "AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--min-relevance",
                        "0",
                        "--format",
                        "json",
                        "--output",
                        str(report_path),
                        "--write",
                        "--write-output",
                        str(draft_path),
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            draft = draft_path.read_text(encoding="utf-8")
            self.assertEqual(code, 0)
            self.assertIn(f"Wrote report: {report_path}", output)
            self.assertIn(f"Wrote writing draft: {draft_path}", output)
            self.assertEqual(report["batch"]["query"], "tell me about MALDI AMR")
            self.assertIn("# Evidence-Bound Literature Review Draft", draft)
            self.assertIn("[P1 p2]", draft)

    def test_natural_query_write_package_requires_output_directory(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        discoverer_calls = []

        def fake_discoverer(query, limit):
            discoverer_calls.append((query, limit))
            return []

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "friday",
                        "tell",
                        "me",
                        "about",
                        "MALDI",
                        "AMR",
                        "--write",
                        "--write-format",
                        "package",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                )

            self.assertEqual(code, 2)
            self.assertEqual(discoverer_calls, [])
            self.assertIn("natural --write-format package requires --write-output directory.", out.getvalue())

    def test_query_scan_deep_read_workers_do_not_exceed_target(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title=f"MALDI-TOF antimicrobial resistance paper {index}",
                    source_for_gate=f"https://arxiv.org/pdf/2401.9200{index}v1",
                    arxiv_id=f"2401.9200{index}v1",
                    abstract="Antimicrobial resistance prediction from MALDI-TOF spectra.",
                )
                for index in range(1, 4)
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path=f"artifacts/{source.rsplit('/', 1)[-1]}.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "scan",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "3",
                        "--deep-read-limit",
                        "2",
                        "--deep-read-workers",
                        "2",
                        "--data-dir",
                        str(Path(tmp) / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            self.assertEqual(code, 0)
            self.assertEqual(len(deep_read_sources), 2)
            self.assertIn("Deep-scanned: 2", out.getvalue())

    def test_report_latest_uses_latest_batch(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1000"], tmp_path)
            code, output = self.run_cli(["report", "--latest"], tmp_path)
            self.assertEqual(code, 0)
            self.assertIn("Batch ID:", output)
            self.assertIn("MALDI AMR", output)

    def test_report_supports_markdown_format(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1"], tmp_path)
            code, output = self.run_cli(["report", "--latest", "--format", "markdown"], tmp_path)

            self.assertEqual(code, 0)
            self.assertIn("# Friday Batch Report", output)
            self.assertIn("## Cited Evidence", output)

    def test_report_supports_json_format(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1"], tmp_path)
            code, output = self.run_cli(["report", "--latest", "--format", "json"], tmp_path)

            self.assertEqual(code, 0)
            data = json.loads(output)
            self.assertEqual(data["report_type"], "batch")
            self.assertEqual(data["batch"]["query"], "MALDI AMR")
            self.assertIn("cited_evidence", data)

    def test_report_output_writes_file(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.md"
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1"], tmp_path)
            code, output = self.run_cli(
                ["report", "--latest", "--format", "markdown", "--output", str(report_path)],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote report: {report_path}", output)
            self.assertIn("# Friday Batch Report", report_path.read_text(encoding="utf-8"))

    def test_research_runs_query_and_writes_report_passport_and_rejection_log(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
                Candidate(
                    provider="openalex",
                    title="Blocked code artifact",
                    source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                ),
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.json"
            passport_path = tmp_path / "passport.json"
            rejection_path = tmp_path / "rejections.json"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--format",
                        "json",
                        "--output",
                        str(report_path),
                        "--passport",
                        str(passport_path),
                        "--rejection-log",
                        str(rejection_path),
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                )

            self.assertEqual(code, 0)
            output = out.getvalue()
            self.assertIn("Batch ID: batch_", output)
            self.assertIn(f"Wrote report: {report_path}", output)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["batch"]["query"], "MALDI AMR")
            self.assertEqual(json.loads(passport_path.read_text(encoding="utf-8"))["artifact_type"], "batch_passport")
            rejected = json.loads(rejection_path.read_text(encoding="utf-8"))["rejected"]
            self.assertEqual(rejected[0]["reason"], "blocked_domain")

    def test_research_run_creates_ledger_labels_deep_reads_and_writes_artifacts(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "MALDI AMR")
            self.assertEqual(limit, 2)
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI-TOF antimicrobial resistance prediction",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
                Candidate(
                    provider="openalex",
                    title="Blocked code artifact",
                    source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                ),
            ]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.json"
            passport_path = tmp_path / "passport.json"
            rejection_path = tmp_path / "rejections.json"
            summary_path = tmp_path / "run-summary.json"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research-run",
                        "MALDI",
                        "AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "1",
                        "--format",
                        "json",
                        "--output",
                        str(report_path),
                        "--passport",
                        str(passport_path),
                        "--rejection-log",
                        str(rejection_path),
                        "--run-summary",
                        str(summary_path),
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            output = out.getvalue()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            passport = json.loads(passport_path.read_text(encoding="utf-8"))
            rejections = json.loads(rejection_path.read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(deep_read_sources, ["https://arxiv.org/pdf/2401.12345"])
            self.assertIn("Run ID: run_", output)
            self.assertIn("Status: complete", output)
            self.assertIn("Auto-labeled: 1", output)
            self.assertIn(f"Wrote run summary: {summary_path}", output)
            self.assertEqual(report["batch"]["query"], "MALDI AMR")
            self.assertEqual(summary["artifact_type"], "research_run_summary")
            self.assertEqual(summary["run"]["status"], "complete")
            self.assertEqual(summary["batch"]["screened_count"], 2)
            self.assertEqual(summary["batch"]["blocked_count"], 1)
            self.assertEqual(summary["screening_labels"]["counts"]["relevant"], 1)
            self.assertEqual(passport["artifact_type"], "batch_passport")
            self.assertEqual(rejections["counts"]["source_gate"], 1)

    def test_research_run_resume_deep_reads_next_ranked_candidate(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.storage import FridayStore

        candidates = [
            Candidate(
                provider="arxiv",
                title=f"MALDI-TOF antimicrobial resistance paper {index}",
                source_for_gate=f"https://arxiv.org/pdf/2401.9300{index}v1",
                arxiv_id=f"2401.9300{index}v1",
                abstract="Antimicrobial resistance prediction from MALDI-TOF spectra.",
            )
            for index in range(1, 4)
        ]
        discover_calls = []

        def fake_discoverer(query, limit):
            discover_calls.append((query, limit))
            return candidates[:limit]

        deep_read_sources = []

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="b" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path=f"artifacts/{source.rsplit('/', 1)[-1]}.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            first_out = io.StringIO()
            with redirect_stdout(first_out):
                first_code = main(
                    [
                        "research-run",
                        "MALDI AMR",
                        "--limit",
                        "3",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )
            self.assertEqual(first_code, 0)
            run_id = next(
                line.split(": ", 1)[1]
                for line in first_out.getvalue().splitlines()
                if line.startswith("Run ID:")
            )

            second_out = io.StringIO()
            with redirect_stdout(second_out):
                second_code = main(
                    [
                        "research-run",
                        "--resume-run",
                        run_id,
                        "--deep-read-limit",
                        "2",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            store = FridayStore(data_dir / "friday.db")
            run = store.get_research_run(run_id)
            batch = store.get_batch(run.batch_id)

            self.assertEqual(second_code, 0)
            self.assertEqual(discover_calls, [("MALDI AMR", 3)])
            self.assertEqual(
                deep_read_sources,
                [
                    "https://arxiv.org/pdf/2401.93001v1",
                    "https://arxiv.org/pdf/2401.93002v1",
                ],
            )
            self.assertIn(f"Resumed run: {run_id}", second_out.getvalue())
            self.assertEqual(batch.screened_count, 3)
            self.assertEqual(batch.deep_read_count, 2)
            self.assertEqual(run.status, "complete")

    def test_research_runs_lists_recent_run_ledger(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title=f"{query} paper",
                    source_for_gate=f"https://arxiv.org/pdf/2401.{len(query):05d}",
                    arxiv_id=f"2401.{len(query):05d}",
                    abstract=f"Scholarly abstract for {query}.",
                )
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            for query in ("first query", "second query"):
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(
                        [
                            "research-run",
                            query,
                            "--limit",
                            "1",
                            "--deep-read-limit",
                            "0",
                            "--data-dir",
                            str(data_dir),
                        ],
                        discoverer=fake_discoverer,
                    )
                self.assertEqual(code, 0)

            code, output = self.run_cli(["research-runs"], tmp_path)

            self.assertEqual(code, 0)
            lines = [line for line in output.splitlines() if line.startswith("run_")]
            self.assertEqual(len(lines), 2)
            self.assertIn("second query", lines[0])
            self.assertIn("status=complete", lines[0])
            self.assertIn("screened=1", lines[0])
            self.assertIn("deep=0", lines[0])
            self.assertIn("batch=batch_", lines[0])
            self.assertIn("first query", lines[1])

    def test_eval_suite_list_outputs_available_suites(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["eval-suite", "list"], Path(tmp))

            self.assertEqual(code, 0)
            self.assertIn("core", output)
            self.assertIn("biomedical", output)
            self.assertIn("natural-language", output)
            self.assertIn("safety", output)
            self.assertIn("gold", output)

    def test_eval_suite_run_outputs_scorecard(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["eval-suite", "run"], Path(tmp))

            self.assertEqual(code, 0)
            self.assertIn("Friday Eval Suite", output)
            self.assertIn("Suite: core", output)
            self.assertIn("Status: pass", output)
            self.assertIn("biomedical.maldi_amr_query_plan", output)
            self.assertIn("safety.github_pdf_blocked", output)

    def test_eval_suite_run_json_outputs_structured_report(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(
                ["eval-suite", "run", "--suite", "biomedical", "--format", "json"],
                Path(tmp),
            )
            report = json.loads(output)

            self.assertEqual(code, 0)
            self.assertEqual(report["artifact_type"], "eval_suite_report")
            self.assertEqual(report["suite"], "biomedical")
            self.assertEqual(report["status"], "pass")
            self.assertEqual({case["suite"] for case in report["cases"]}, {"biomedical"})

    def test_eval_suite_run_gold_json_outputs_structured_report(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(
                ["eval-suite", "run", "--suite", "gold", "--format", "json"],
                Path(tmp),
            )
            report = json.loads(output)

            self.assertEqual(code, 0)
            self.assertEqual(report["artifact_type"], "eval_suite_report")
            self.assertEqual(report["suite"], "gold")
            self.assertEqual(report["status"], "pass")
            self.assertGreaterEqual(report["counts"]["total"], 7)
            self.assertEqual({case["suite"] for case in report["cases"]}, {"gold"})

    def test_eval_suite_unknown_suite_returns_error(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["eval-suite", "run", "--suite", "unknown"], Path(tmp))

            self.assertEqual(code, 1)
            self.assertIn("Unknown eval suite: unknown", output)

    def test_run_summary_latest_outputs_dashboard(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.evidence import EvidenceItem
        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
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
            maybe = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/run-summary-maybe",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=70,
            )
            unlabeled = Candidate(
                provider="openalex",
                title="MALDI high relevance unlabeled",
                source_for_gate="10.1000/run-summary-unlabeled",
                abstract="MALDI antibiotic resistance metadata.",
                relevance_score=82,
            )
            for candidate in [maybe, unlabeled]:
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
            )
            artifact = store.add_pdf_artifact(
                batch.batch_id,
                source=maybe.source_for_gate,
                pdf_url="https://www.nature.com/articles/example.pdf",
                final_url="https://www.nature.com/articles/example.pdf",
                sha256="c" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/example.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_evidence_records(
                artifact.artifact_id,
                [EvidenceItem(evidence_type="result", text="The method detected resistance.", page_number=4)],
            )
            store.sync_research_run_counts(run.run_id)

            code, output = self.run_cli(["run-summary", "--latest"], tmp_path)

            self.assertEqual(code, 0)
            self.assertIn("Friday Run Summary", output)
            self.assertIn(f"Run: {run.run_id}", output)
            self.assertIn(f"Batch: {batch.batch_id}", output)
            self.assertIn("screened=2", output)
            self.assertIn("unlabeled_allowed=1", output)
            self.assertIn("Attention", output)
            self.assertIn(unlabeled.source_for_gate, output)
            self.assertIn("friday labels review --latest --only maybe", output)

    def test_run_summary_json_outputs_structured_dashboard(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="language math", limit=25, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Formal language mathematics",
                source_for_gate="10.1000/run-summary-json",
                abstract="Formal grammars and algebraic language theory.",
                relevance_score=71,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            code, output = self.run_cli(["run-summary", "--latest", "--format", "json"], tmp_path)
            summary = json.loads(output)

            self.assertEqual(code, 0)
            self.assertEqual(summary["artifact_type"], "run_summary_dashboard")
            self.assertEqual(summary["target"]["target_type"], "batch")
            self.assertEqual(summary["target"]["batch_id"], batch.batch_id)
            self.assertEqual(summary["counts"]["screened"], 1)
            self.assertEqual(summary["attention"]["high_relevance_unlabeled"][0]["source"], candidate.source_for_gate)

    def test_run_summary_latest_reports_missing_target(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            code, output = self.run_cli(["run-summary", "--latest"], Path(tmp))

            self.assertEqual(code, 1)
            self.assertIn("No research runs or batches found.", output)

    def test_research_run_latest_resumes_newest_run(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        candidates_by_query = {
            "first query": [
                Candidate(
                    provider="arxiv",
                    title="First query paper",
                    source_for_gate="https://arxiv.org/pdf/2401.94001v1",
                    arxiv_id="2401.94001v1",
                    abstract="First scholarly abstract.",
                )
            ],
            "second query": [
                Candidate(
                    provider="arxiv",
                    title="Second query paper",
                    source_for_gate="https://arxiv.org/pdf/2401.94002v1",
                    arxiv_id="2401.94002v1",
                    abstract="Second scholarly abstract.",
                )
            ],
        }
        discover_calls = []
        deep_read_sources = []

        def fake_discoverer(query, limit):
            discover_calls.append((query, limit))
            return candidates_by_query[query][:limit]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            deep_read_sources.append(source)
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="c" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            store.add_pdf_pages(artifact.artifact_id, ["page text"])
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            for query in ("first query", "second query"):
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(
                        [
                            "research-run",
                            query,
                            "--limit",
                            "1",
                            "--deep-read-limit",
                            "0",
                            "--data-dir",
                            str(data_dir),
                        ],
                        discoverer=fake_discoverer,
                        pdf_ingestor=fake_pdf_ingestor,
                    )
                self.assertEqual(code, 0)

            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research-run",
                        "--latest",
                        "--deep-read-limit",
                        "1",
                        "--min-relevance",
                        "0",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )

            self.assertEqual(code, 0)
            self.assertEqual(discover_calls, [("first query", 1), ("second query", 1)])
            self.assertEqual(deep_read_sources, ["https://arxiv.org/pdf/2401.94002v1"])
            self.assertIn("Resumed run: run_", out.getvalue())
            self.assertIn("Query: second query", out.getvalue())
            self.assertIn("Deep-scanned: 1", out.getvalue())

    def test_research_run_writes_default_artifact_folder(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.95001v1",
                    arxiv_id="2401.95001v1",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                ),
                Candidate(
                    provider="openalex",
                    title="Blocked code artifact",
                    source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                ),
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research-run",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "0",
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            output = out.getvalue()
            run_id = next(line.split(": ", 1)[1] for line in output.splitlines() if line.startswith("Run ID:"))
            run_dir = data_dir / "runs" / run_id

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote report: {run_dir / 'report.md'}", output)
            self.assertIn(f"Wrote passport: {run_dir / 'passport.json'}", output)
            self.assertIn(f"Wrote rejection log: {run_dir / 'rejection-log.json'}", output)
            self.assertIn(f"Wrote run summary: {run_dir / 'run-summary.json'}", output)
            self.assertIn("# Friday Batch Report", (run_dir / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((run_dir / "passport.json").read_text(encoding="utf-8"))["artifact_type"], "batch_passport")
            self.assertEqual(json.loads((run_dir / "rejection-log.json").read_text(encoding="utf-8"))["counts"]["source_gate"], 1)
            self.assertEqual(
                json.loads((run_dir / "run-summary.json").read_text(encoding="utf-8"))["run"]["run_id"],
                run_id,
            )

    def test_smoke_run_writes_dogfood_artifact_pack(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            self.assertEqual(query, "MALDI AMR")
            self.assertEqual(limit, 2)
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.96001v1",
                    arxiv_id="2401.96001v1",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                    year=2024,
                ),
                Candidate(
                    provider="openalex",
                    title="Blocked code artifact",
                    source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                    url="https://github.com/example/repo/blob/main/paper.pdf",
                ),
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            smoke_dir = tmp_path / "smoke-pack"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "smoke-run",
                        "MALDI AMR",
                        "--limit",
                        "2",
                        "--deep-read-limit",
                        "0",
                        "--output-dir",
                        str(smoke_dir),
                        "--data-dir",
                        str(data_dir),
                    ],
                    discoverer=fake_discoverer,
                )
            output = out.getvalue()

            self.assertEqual(code, 0)
            self.assertIn(f"Smoke run directory: {smoke_dir}", output)
            self.assertIn("Next commands:", output)

            expected_files = [
                "report.md",
                "passport.json",
                "rejection-log.json",
                "run-summary.json",
                "labels-review.json",
                "labels-export.jsonl",
                "label-eval.json",
                "smoke-manifest.json",
            ]
            for filename in expected_files:
                self.assertTrue((smoke_dir / filename).exists(), filename)

            manifest = json.loads((smoke_dir / "smoke-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_type"], "smoke_run_manifest")
            self.assertEqual(manifest["query"], "MALDI AMR")
            self.assertEqual(manifest["limit"], 2)
            self.assertEqual(manifest["deep_read_limit"], 0)
            self.assertTrue(manifest["run_id"].startswith("run_"))
            self.assertTrue(manifest["batch_id"].startswith("batch_"))
            self.assertEqual(
                set(manifest["artifacts"]),
                {
                    "report",
                    "passport",
                    "rejection_log",
                    "run_summary",
                    "labels_review",
                    "labels_export",
                    "label_eval",
                },
            )
            self.assertIn(f"friday run-summary --run-id {manifest['run_id']}", manifest["next_commands"])
            self.assertIn(f"friday labels review --batch-id {manifest['batch_id']}", manifest["next_commands"])
            self.assertIn(f"# Friday Batch Report", (smoke_dir / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads((smoke_dir / "rejection-log.json").read_text(encoding="utf-8"))["counts"]["source_gate"],
                1,
            )
            label_eval = json.loads((smoke_dir / "label-eval.json").read_text(encoding="utf-8"))
            self.assertEqual(label_eval["batch_id"], manifest["batch_id"])
            review_payload = json.loads((smoke_dir / "labels-review.json").read_text(encoding="utf-8"))
            self.assertEqual(review_payload["artifact_type"], "labels_review")
            self.assertEqual(review_payload["batch_id"], manifest["batch_id"])
            self.assertTrue((smoke_dir / "labels-export.jsonl").read_text(encoding="utf-8").strip())

    def test_review_queue_command_lists_llm_candidates(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=3, mode="query")
            maybe = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/maybe",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=78,
            )
            conflict = Candidate(
                provider="openalex",
                title="MALDI antimicrobial resistance conflict",
                source_for_gate="10.1000/conflict",
                abstract="MALDI antimicrobial resistance metadata.",
                relevance_score=74,
            )
            human = Candidate(
                provider="arxiv",
                title="Human-reviewed paper",
                source_for_gate="https://arxiv.org/pdf/2401.99001",
                abstract="MALDI antimicrobial resistance.",
                relevance_score=90,
            )
            for candidate in [maybe, conflict, human]:
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
            )
            store.set_screening_label(
                batch.batch_id,
                conflict.source_for_gate,
                "irrelevant",
                label_source="agent",
                confidence=0.89,
            )
            store.set_screening_label(batch.batch_id, human.source_for_gate, "relevant")

            code, output = self.run_cli(["review-queue", "--latest", "--limit", "2"], tmp_path)

            self.assertEqual(code, 0)
            self.assertIn(f"LLM review queue for batch: {batch.batch_id}", output)
            self.assertIn("rank=1", output)
            self.assertIn("source=10.1000/maybe", output)
            self.assertIn("reason=heuristic_maybe_high_relevance", output)
            self.assertIn("label=maybe", output)
            self.assertIn("confidence=0.610", output)
            self.assertIn("source=10.1000/conflict", output)
            self.assertNotIn(human.source_for_gate, output)

    def test_labels_export_latest_writes_jsonl(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="language math", limit=5, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Mathematical structure in language",
                source_for_gate="10.1000/lang-math",
                doi="10.1000/lang-math",
                abstract="Formal grammars model natural language.",
                relevance_score=77,
                relevance_reason="strong query overlap",
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "relevant",
                label_source="human",
                note="gold include",
            )
            output_path = tmp_path / "labels.jsonl"

            code, output = self.run_cli(
                [
                    "labels",
                    "export",
                    "--latest",
                    "--format",
                    "jsonl",
                    "--output",
                    str(output_path),
                ],
                tmp_path,
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote label export: {output_path}", output)
            self.assertIn("Rows: 1", output)
            self.assertEqual(rows[0]["batch_id"], batch.batch_id)
            self.assertEqual(rows[0]["gold_label"], "relevant")
            self.assertIsNone(rows[0]["weak_label"])
            self.assertEqual(rows[0]["label_note"], "gold include")

    def test_labels_export_all_writes_csv(self):
        import csv
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            first_batch = store.create_batch(query="MALDI AMR", limit=5, mode="query")
            second_batch = store.create_batch(query="language math", limit=5, mode="query")
            first_candidate = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance",
                source_for_gate="10.1000/maldi",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=84,
            )
            second_candidate = Candidate(
                provider="openalex",
                title="Mathematical language",
                source_for_gate="10.1000/language",
                abstract="Formal language theory.",
                relevance_score=71,
            )
            for batch_record, candidate, label in [
                (first_batch, first_candidate, "maybe"),
                (second_batch, second_candidate, "relevant"),
            ]:
                store.add_batch_item(
                    batch_record.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
                store.set_screening_label(
                    batch_record.batch_id,
                    candidate.source_for_gate,
                    label,
                    label_source="agent",
                    confidence=0.74,
                    rationale="metadata match",
                    signals="label_provider=heuristic",
                )
            output_path = tmp_path / "labels.csv"

            code, output = self.run_cli(
                [
                    "labels",
                    "export",
                    "--all",
                    "--format",
                    "csv",
                    "--output",
                    str(output_path),
                ],
                tmp_path,
            )
            rows = list(csv.DictReader(output_path.read_text(encoding="utf-8").splitlines()))

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote label export: {output_path}", output)
            self.assertIn("Rows: 2", output)
            self.assertEqual({row["query"] for row in rows}, {"MALDI AMR", "language math"})
            self.assertEqual({row["weak_label"] for row in rows}, {"maybe", "relevant"})
            self.assertEqual({row["gold_label"] for row in rows}, {""})

    def test_labels_eval_outputs_feedback_summary(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=2, mode="query")
            candidate = Candidate(
                provider="pubmed",
                title="MALDI antibiotic susceptibility",
                source_for_gate="10.1000/eval-cli",
                abstract="MALDI spectra can support antibiotic susceptibility testing.",
                relevance_score=76,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.66,
            )
            self.run_cli(
                [
                    "labels",
                    "set",
                    "--latest",
                    "--source",
                    candidate.source_for_gate,
                    "--label",
                    "relevant",
                    "--note",
                    "human checked",
                ],
                tmp_path,
            )

            code, output = self.run_cli(["labels", "eval", "--latest"], tmp_path)

            self.assertEqual(code, 0)
            self.assertIn(f"Label evaluation for batch: {batch.batch_id}", output)
            self.assertIn("Human labels: relevant=1 maybe=0 irrelevant=0", output)
            self.assertIn("Comparable overrides: 1", output)
            self.assertIn("Accuracy: 0.000", output)
            self.assertIn("Recommendation:", output)
            self.assertIn("human=relevant", output)
            self.assertIn("agent=maybe", output)
            self.assertIn("confidence=0.660", output)

    def test_labels_eval_json_outputs_structured_report(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="language math", limit=1, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Formal language mathematics",
                source_for_gate="10.1000/eval-json",
                abstract="Formal grammars and algebraic language theory.",
                relevance_score=71,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "relevant",
                note="agent=relevant confidence=0.77",
            )

            code, output = self.run_cli(["labels", "eval", "--latest", "--format", "json"], tmp_path)
            report = json.loads(output)

            self.assertEqual(code, 0)
            self.assertEqual(report["batch_id"], batch.batch_id)
            self.assertEqual(report["human_label_counts"]["relevant"], 1)
            self.assertEqual(report["comparable_count"], 1)
            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["confusion_matrix"]["relevant"]["relevant"], 1)

    def test_labels_review_filters_maybe_and_unlabeled(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=5, mode="query")
            maybe = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/maybe-review",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=82,
            )
            unlabeled = Candidate(
                provider="openalex",
                title="MALDI antimicrobial resistance unlabeled",
                source_for_gate="10.1000/unlabeled-review",
                abstract="MALDI antimicrobial resistance metadata.",
                relevance_score=79,
            )
            low_unlabeled = Candidate(
                provider="openalex",
                title="Low relevance MALDI note",
                source_for_gate="10.1000/low-review",
                abstract="Weak metadata.",
                relevance_score=18,
            )
            human = Candidate(
                provider="arxiv",
                title="Human-reviewed antimicrobial resistance paper",
                source_for_gate="https://arxiv.org/pdf/2401.01002",
                abstract="MALDI antimicrobial resistance.",
                relevance_score=95,
            )
            for candidate in (maybe, unlabeled, low_unlabeled, human):
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
            )
            store.set_screening_label(batch.batch_id, human.source_for_gate, "relevant")

            maybe_code, maybe_output = self.run_cli(
                ["labels", "review", "--latest", "--only", "maybe"],
                tmp_path,
            )
            unlabeled_code, unlabeled_output = self.run_cli(
                ["labels", "review", "--latest", "--only", "unlabeled", "--min-relevance", "60"],
                tmp_path,
            )

            self.assertEqual(maybe_code, 0)
            self.assertIn(f"Label review for batch: {batch.batch_id}", maybe_output)
            self.assertIn("rank=1", maybe_output)
            self.assertIn("label=maybe", maybe_output)
            self.assertIn("source_label=agent", maybe_output)
            self.assertIn("confidence=0.610", maybe_output)
            self.assertIn("queue_reason=heuristic_maybe_high_relevance", maybe_output)
            self.assertIn(maybe.source_for_gate, maybe_output)
            self.assertNotIn(unlabeled.source_for_gate, maybe_output)
            self.assertEqual(unlabeled_code, 0)
            self.assertIn("label=-", unlabeled_output)
            self.assertIn("source_label=unlabeled", unlabeled_output)
            self.assertIn(unlabeled.source_for_gate, unlabeled_output)
            self.assertNotIn(low_unlabeled.source_for_gate, unlabeled_output)
            self.assertNotIn(human.source_for_gate, unlabeled_output)

    def test_labels_set_applies_human_override(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from friday.source_policy import evaluate_source
        from friday.storage import FridayStore

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="language math", limit=5, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Mathematical structure in language",
                source_for_gate="10.1000/review-set",
                abstract="Formal grammars model natural language.",
                relevance_score=77,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )
            store.set_screening_label(
                batch.batch_id,
                candidate.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.67,
            )

            code, output = self.run_cli(
                [
                    "labels",
                    "set",
                    "--latest",
                    "--source",
                    candidate.source_for_gate,
                    "--label",
                    "relevant",
                    "--note",
                    "human checked",
                ],
                tmp_path,
            )
            labels = store.list_screening_labels(batch.batch_id)

            self.assertEqual(code, 0)
            self.assertIn("Labeled: relevant", output)
            self.assertIn("Note: human checked", output)
            self.assertEqual(labels[0].label, "relevant")
            self.assertEqual(labels[0].label_source, "human")
            self.assertEqual(labels[0].note, "human checked")

    def test_research_run_llm_uses_smart_review_queue(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FakeLlmClient:
            def __init__(self):
                self.calls = []

            def label(self, *, query, item, model):
                self.calls.append(item.source)
                return LlmLabelResult(
                    label="maybe",
                    confidence=0.76,
                    rationale="LLM reviewed smart queue item.",
                    evidence_terms=("reviewed",),
                    exclusion_reason=None,
                )

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="openalex",
                    title="Mathematical structures in formal language theory",
                    source_for_gate="10.1000/first",
                    abstract="Formal grammars, algebra, and language theory.",
                    relevance_score=60,
                ),
                Candidate(
                    provider="openalex",
                    title="General mathematics education",
                    source_for_gate="10.1000/conflict",
                    abstract="Students use mathematics classroom tools.",
                    relevance_score=80,
                ),
                Candidate(
                    provider="arxiv",
                    title="Information theory and language",
                    source_for_gate="https://arxiv.org/pdf/2401.99002",
                    abstract="Entropy and coding theory for language.",
                    relevance_score=20,
                ),
            ][:limit]

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_path = tmp_path / "summary.json"
            passport_path = tmp_path / "passport.json"
            fake_client = FakeLlmClient()
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research-run",
                        "language math",
                        "--limit",
                        "3",
                        "--deep-read-limit",
                        "0",
                        "--auto-label-provider",
                        "llm",
                        "--llm-review-limit",
                        "1",
                        "--auto-label-model",
                        "gpt-test",
                        "--run-summary",
                        str(summary_path),
                        "--passport",
                        str(passport_path),
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    llm_label_client=fake_client,
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            passport = json.loads(passport_path.read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(fake_client.calls, ["https://arxiv.org/pdf/2401.99002"])
            self.assertIn("LLM-reviewed: 1", out.getvalue())
            self.assertEqual(summary["llm_review_queue"]["items"][0]["source"], "https://arxiv.org/pdf/2401.99002")
            self.assertIn("heuristic_maybe", summary["llm_review_queue"]["items"][0]["reason"])
            self.assertIn("low_confidence_label", summary["llm_review_queue"]["items"][0]["reason"])
            self.assertEqual(passport["llm_review_queue"]["items"][0]["source"], "https://arxiv.org/pdf/2401.99002")

    def test_import_corpus_writes_folder_adapter_outputs(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "papers"
            source_dir.mkdir()
            (source_dir / "MALDI AMR paper.pdf").write_bytes(b"%PDF-1.4")
            output_path = tmp_path / "corpus.json"
            rejection_path = tmp_path / "rejections.json"

            code, output = self.run_cli(
                [
                    "import-corpus",
                    "--folder",
                    str(source_dir),
                    "--output",
                    str(output_path),
                    "--rejection-log",
                    str(rejection_path),
                ],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote corpus: {output_path}", output)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["literature_corpus"][0]["title"], "MALDI AMR paper")

    def test_write_latest_outputs_claim_table_from_supported_evidence(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                )
            ]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="a" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

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
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )
            self.assertEqual(code, 0)

            output_path = tmp_path / "claims.md"
            code, output = self.run_cli(
                ["write", "--latest", "--mode", "claim-table", "--output", str(output_path)],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote writing draft: {output_path}", output)
            self.assertIn("| C1 | SUPPORTED | result | P1 p2 | The model achieved an AUROC of 0.91. |", output_path.read_text(encoding="utf-8"))

    def test_write_from_report_json_file_outputs_literature_review(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1"], tmp_path)
            report_path = tmp_path / "report.json"
            self.run_cli(["report", "--latest", "--format", "json", "--output", str(report_path)], tmp_path)

            code, output = self.run_cli(
                ["write", "--report", str(report_path), "--mode", "literature-review"],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn("# Evidence-Bound Literature Review Draft", output)
            self.assertIn("MATERIAL GAP", output)

    def test_write_accepts_results_summary_mode(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.run_cli(["scan", "--query", "MALDI AMR", "--limit", "1"], tmp_path)
            output_path = tmp_path / "results.md"

            code, output = self.run_cli(
                ["write", "--latest", "--mode", "results-summary", "--output", str(output_path)],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote writing draft: {output_path}", output)
            self.assertIn("# Evidence-Bound Results Summary", output_path.read_text(encoding="utf-8"))

    def test_write_json_exposes_audit_summary_for_downstream_tools(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                )
            ]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="b" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

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
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )
            self.assertEqual(code, 0)

            output_path = tmp_path / "writing.json"
            code, output = self.run_cli(
                [
                    "write",
                    "--latest",
                    "--mode",
                    "results-summary",
                    "--format",
                    "json",
                    "--output",
                    str(output_path),
                ],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote writing draft: {output_path}", output)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["artifact_type"], "writing_copilot_output")
            self.assertEqual(data["audit_summary"]["citation_check_status"], "pass")
            self.assertEqual(data["audit_summary"]["supported_paragraph_count"], 1)
            self.assertEqual(data["audit_summary"]["blocked_paragraph_count"], 0)
            self.assertEqual(data["audit_summary"]["supported_paragraphs"][0]["citations"], ["P1 p2"])

    def test_write_package_exports_handoff_folder(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        def fake_discoverer(query, limit):
            return [
                Candidate(
                    provider="arxiv",
                    title="MALDI antimicrobial resistance paper",
                    source_for_gate="https://arxiv.org/pdf/2401.12345",
                    arxiv_id="2401.12345",
                    abstract="Antimicrobial resistance prediction from MALDI spectra.",
                )
            ]

        def fake_pdf_ingestor(store, data_dir, batch_id, source, candidate=None):
            artifact = store.add_pdf_artifact(
                batch_id,
                source=source,
                pdf_url=source,
                final_url=source,
                sha256="c" * 64,
                byte_count=123,
                content_type="application/pdf",
                local_path="artifacts/test.pdf",
                status="stored",
                reason="pdf_text_extracted",
            )
            from friday.evidence import EvidenceItem

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
            return PdfIngestionResult(
                status="stored",
                reason="pdf_text_extracted",
                artifact_id=artifact.artifact_id,
                pdf_url=source,
                page_count=1,
            )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "research",
                        "--query",
                        "MALDI AMR",
                        "--limit",
                        "1",
                        "--deep-read-limit",
                        "1",
                        "--data-dir",
                        str(tmp_path / ".friday"),
                    ],
                    discoverer=fake_discoverer,
                    pdf_ingestor=fake_pdf_ingestor,
                )
            self.assertEqual(code, 0)

            output_dir = tmp_path / "writing-package"
            out = io.StringIO()
            with redirect_stdout(out):
                try:
                    code = main(
                        [
                            "write",
                            "--latest",
                            "--mode",
                            "results-summary",
                            "--format",
                            "package",
                            "--output",
                            str(output_dir),
                            "--data-dir",
                            str(tmp_path / ".friday"),
                        ]
                    )
                except SystemExit as exc:
                    code = exc.code
            output = out.getvalue()

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote writing package: {output_dir}", output)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
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
            self.assertIn("# Evidence-Bound Results Summary", (output_dir / "draft.md").read_text(encoding="utf-8"))
            self.assertIn("# Friday Evidence Report", (output_dir / "report.md").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "report.pdf").read_bytes().startswith(b"%PDF-1.4"))
            writing = json.loads((output_dir / "writing.json").read_text(encoding="utf-8"))
            screening_labels = json.loads((output_dir / "screening_labels.json").read_text(encoding="utf-8"))
            supported = json.loads((output_dir / "supported_paragraphs.json").read_text(encoding="utf-8"))
            self.assertEqual(writing["audit_summary"]["supported_paragraph_count"], 1)
            self.assertEqual(screening_labels["artifact_type"], "screening_label_summary")
            self.assertEqual(supported[0]["citations"], ["P1 p2"])

    def test_compose_exports_evidence_bound_draft_package(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_dir = tmp_path / "writing-package"
            _write_compose_fixture_package(package_dir)
            output_dir = tmp_path / "compose-output"

            code, output = self.run_cli(
                [
                    "compose",
                    "--package",
                    str(package_dir),
                    "--section",
                    "results",
                    "--output",
                    str(output_dir),
                ],
                tmp_path,
            )

            self.assertEqual(code, 0)
            self.assertIn(f"Wrote compose package: {output_dir}", output)
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "claim_audit.json",
                    "conflicts.json",
                    "draft.md",
                    "outline.json",
                    "refused_claims.json",
                    "used_evidence.json",
                ],
            )
            draft = (output_dir / "draft.md").read_text(encoding="utf-8")
            self.assertIn("# Evidence-Bound Results Draft", draft)
            self.assertIn("result evidence includes AUROC 0.91", draft)
            self.assertNotIn("unsupported generated result", draft)
            used = json.loads((output_dir / "used_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(used["used_evidence"][0]["citations"], ["P1 p2", "P2 p2"])

    def test_compose_llm_writes_planner_and_composer_audit_without_real_provider(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / ".friday"
            package_dir = tmp_path / "writing-package"
            _write_compose_fixture_package(package_dir)
            output_dir = tmp_path / "compose-output"
            self.run_cli(["settings", "set", "llm.composer_provider", "none"], tmp_path)
            out = io.StringIO()

            with redirect_stdout(out):
                code = main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "compose",
                        "--package",
                        str(package_dir),
                        "--section",
                        "results",
                        "--llm",
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "discourse_plan.json").exists())
            self.assertTrue((output_dir / "composer_audit.json").exists())
            self.assertTrue((output_dir / "verifier_audit.json").exists())
            audit = json.loads((output_dir / "composer_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["reason"], "model_unavailable")
            verifier_audit = json.loads((output_dir / "verifier_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(verifier_audit["status"], "skipped")
            self.assertEqual(verifier_audit["reason"], "composer_not_trusted")

    def test_compose_does_not_create_friday_store(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_dir = tmp_path / "writing-package"
            _write_compose_fixture_package(package_dir)
            output_dir = tmp_path / "compose-output"
            data_dir = tmp_path / ".friday"
            out = io.StringIO()

            with redirect_stdout(out):
                code = main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "compose",
                        "--package",
                        str(package_dir),
                        "--section",
                        "results",
                        "--output",
                        str(output_dir),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertFalse((data_dir / "friday.db").exists())

    def test_global_data_dir_before_subcommand_is_honored(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--data-dir",
                        str(data_dir),
                        "scan",
                        "https://arxiv.org/pdf/2401.12345",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((data_dir / "friday.db").exists())

def _write_compose_fixture_package(package_dir):
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
            }
        ],
    )
    _write_json(
        package_dir / "supported_paragraphs.json",
        [
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
            }
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
    _write_json(package_dir / "material_gaps.json", [])


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


class LlmCommandTests(unittest.TestCase):
    def run_cli(self, args, tmp_path):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([*args, "--data-dir", str(tmp_path / ".friday")])
        return code, out.getvalue()

    def _disable_all_roles(self, tmp_path):
        # Set every role to 'none' so `llm status` never spawns a CLI subprocess.
        for role in ("screener", "extractor", "composer", "verifier", "critic"):
            self.run_cli(["settings", "set", f"llm.{role}_provider", "none"], tmp_path)

    def test_llm_status_lists_roles_without_spawning(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._disable_all_roles(tmp_path)
            code, output = self.run_cli(["llm", "status"], tmp_path)
            self.assertEqual(code, 0)
            for role in ("screener", "composer", "verifier"):
                self.assertIn(role, output)

    def test_llm_test_requires_wired_role(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._disable_all_roles(tmp_path)
            code, output = self.run_cli(["llm", "test", "--role", "composer"], tmp_path)
            self.assertEqual(code, 2)
            self.assertIn("not wired", output)


if __name__ == "__main__":
    unittest.main()
