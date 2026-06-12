import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from friday import pdf_parser
from friday.pdf_parser import (
    PDF_PARSE_MAX_PAGES,
    _PARSER_ENV_ALLOWLIST,
    _parser_preexec,
    _read_capped,
    _scrubbed_parser_env,
    ParsedPdfPage,
    PdfParseResult,
    PdfParserFailure,
    parse_pdf_with_fallback,
    parse_with_pdftotext_raw,
)


class PdfParserTests(unittest.TestCase):
    def test_parse_pdf_with_fallback_prefers_high_confidence_parser(self):
        calls = []

        def low_confidence_parser(path):
            calls.append("low")
            return PdfParseResult(
                parser_name="low-parser",
                parser_version="1",
                pages=[
                    ParsedPdfPage(
                        page_number=1,
                        text="Articles seTo design suitable local and global interventions.",
                        confidence=0.2,
                        flags=("column_stitching",),
                    )
                ],
                confidence=0.2,
                flags=("low_confidence",),
            )

        def high_confidence_parser(path):
            calls.append("high")
            return PdfParseResult(
                parser_name="high-parser",
                parser_version="1",
                pages=[
                    ParsedPdfPage(
                        page_number=1,
                        text="This review used a structured search strategy across PubMed and EMBASE.",
                        confidence=0.92,
                        flags=(),
                    )
                ],
                confidence=0.92,
                flags=(),
            )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.7\nfake\n%%EOF")

            result = parse_pdf_with_fallback(
                path,
                parsers=[low_confidence_parser, high_confidence_parser],
                min_confidence=0.8,
            )

        self.assertEqual(calls, ["low", "high"])
        self.assertEqual(result.parser_name, "high-parser")
        self.assertEqual(result.confidence, 0.92)
        self.assertEqual(result.pages[0].confidence, 0.92)

    def test_parse_pdf_with_fallback_evaluates_all_acceptable_parsers_before_choosing(self):
        calls = []

        def acceptable_layout_parser(path):
            calls.append("layout")
            return PdfParseResult(
                parser_name="layout-parser",
                parser_version="1",
                pages=[
                    ParsedPdfPage(
                        page_number=1,
                        text="The model achieved an AUROC of 0.91 in validation.",
                        confidence=0.75,
                        flags=("wide_spacing",),
                    )
                ],
                confidence=0.75,
                flags=("wide_spacing",),
            )

        def cleaner_default_parser(path):
            calls.append("default")
            return PdfParseResult(
                parser_name="default-parser",
                parser_version="1",
                pages=[
                    ParsedPdfPage(
                        page_number=1,
                        text="The model achieved an AUROC of 0.91 in validation.",
                        confidence=0.95,
                        flags=(),
                    )
                ],
                confidence=0.95,
                flags=(),
            )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.7\nfake\n%%EOF")

            result = parse_pdf_with_fallback(
                path,
                parsers=[acceptable_layout_parser, cleaner_default_parser],
                min_confidence=0.6,
            )

        self.assertEqual(calls, ["layout", "default"])
        self.assertEqual(result.parser_name, "default-parser")
        self.assertEqual(result.confidence, 0.95)

    def test_parse_pdf_with_fallback_raises_when_all_parsers_fail(self):
        def raising_parser(path):
            raise RuntimeError("parser failed")

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.7\nfake\n%%EOF")

            with self.assertRaises(PdfParserFailure):
                parse_pdf_with_fallback(path, parsers=[raising_parser])

    def test_pdftotext_raw_parser_uses_poppler_raw_mode(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            Path(command[-1]).write_text("Results\nThe model achieved an AUROC of 0.91.\f", encoding="utf-8")

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pdf"
            path.write_bytes(b"%PDF-1.7\nfake\n%%EOF")

            with patch("friday.pdf_parser.shutil.which", return_value="/usr/bin/pdftotext"), patch(
                "friday.pdf_parser.subprocess.run",
                side_effect=fake_run,
            ):
                result = parse_with_pdftotext_raw(path)

        self.assertIn("-raw", commands[0])
        self.assertEqual(result.parser_name, "pdftotext-raw")
        self.assertEqual(result.pages[0].text, "Results\nThe model achieved an AUROC of 0.91.")

    def test_pdftotext_runs_with_scrubbed_env_and_caps(self):
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = list(command)
            captured["kwargs"] = dict(kwargs)
            Path(command[-1]).write_text("PAGE ONE\fPAGE TWO", encoding="utf-8")

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-not-reach-parser"
        try:
            with patch("friday.pdf_parser.shutil.which", return_value="/usr/bin/pdftotext"), patch(
                "friday.pdf_parser.subprocess.run",
                side_effect=fake_run,
            ):
                result = parse_with_pdftotext_raw(Path("paper.pdf"))
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

        self.assertEqual([page.text for page in result.pages], ["PAGE ONE", "PAGE TWO"])
        self.assertEqual(captured["kwargs"]["timeout"], pdf_parser.PDF_TEXT_TIMEOUT_SECONDS)
        self.assertTrue(captured["kwargs"]["check"])
        self.assertEqual(captured["kwargs"]["stdin"], pdf_parser.subprocess.DEVNULL)
        self.assertNotIn("ANTHROPIC_API_KEY", captured["kwargs"]["env"])
        self.assertIn("-l", captured["command"])
        self.assertIn(str(PDF_PARSE_MAX_PAGES), captured["command"])

    def test_parser_env_only_keeps_allowlisted_keys(self):
        os.environ["OPENAI_API_KEY"] = "sk-should-not-leak"
        os.environ["FRIDAY_RANDOM_VAR"] = "leak-me"
        os.environ["PATH"] = os.environ.get("PATH", "/usr/bin")
        try:
            env = _scrubbed_parser_env()
        finally:
            del os.environ["OPENAI_API_KEY"]
            del os.environ["FRIDAY_RANDOM_VAR"]

        self.assertIn("PATH", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("FRIDAY_RANDOM_VAR", env)
        for key in env:
            self.assertIn(key, _PARSER_ENV_ALLOWLIST)

    def test_parser_preexec_ignores_unsupported_rlimit_calls(self):
        class RejectingResource:
            RLIMIT_AS = 1
            RLIMIT_CPU = 2

            def setrlimit(self, _name, _limits):
                raise OSError("unsupported rlimit")

        with patch("friday.pdf_parser._resource", RejectingResource()):
            preexec = _parser_preexec(1024, 1)

            preexec()

    def test_read_capped_truncates_oversized_output(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.txt"
            path.write_bytes(b"x" * 5000)

            text = _read_capped(path, max_bytes=1000)

        self.assertEqual(len(text), 1000)


if __name__ == "__main__":
    unittest.main()
