import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday import pdf_ingestion
from friday.pdf_ingestion import (
    _PARSER_ENV_ALLOWLIST,
    _read_capped,
    _scrubbed_parser_env,
    extract_pdf_text_pages,
)


class ScrubbedEnvTests(unittest.TestCase):
    def test_secrets_are_dropped(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-secret"
        os.environ["OPENAI_API_KEY"] = "sk-secret"
        os.environ["PATH"] = os.environ.get("PATH", "/usr/bin")
        try:
            env = _scrubbed_parser_env()
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertIn("PATH", env)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
            del os.environ["OPENAI_API_KEY"]

    def test_only_allowlisted_keys_survive(self):
        os.environ["FRIDAY_RANDOM_VAR"] = "leak-me"
        try:
            env = _scrubbed_parser_env()
            self.assertNotIn("FRIDAY_RANDOM_VAR", env)
            for key in env:
                self.assertIn(key, _PARSER_ENV_ALLOWLIST)
        finally:
            del os.environ["FRIDAY_RANDOM_VAR"]


class ReadCappedTests(unittest.TestCase):
    def test_truncates_oversized_output(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.txt"
            path.write_bytes(b"x" * 5000)
            text = _read_capped(path, max_bytes=1000)
            self.assertEqual(len(text), 1000)

    def test_returns_full_text_when_small(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.txt"
            path.write_text("page one\fpage two", encoding="utf-8")
            text = _read_capped(path, max_bytes=10_000)
            self.assertEqual(text, "page one\fpage two")


class ExtractPdfTextPagesTests(unittest.TestCase):
    def test_missing_pdftotext_raises(self):
        original = pdf_ingestion.shutil.which
        pdf_ingestion.shutil.which = lambda name: None
        try:
            with self.assertRaises(RuntimeError) as ctx:
                extract_pdf_text_pages(Path("nope.pdf"))
            self.assertIn("pdftotext_not_found", str(ctx.exception))
        finally:
            pdf_ingestion.shutil.which = original

    def test_runs_sandboxed_with_scrubbed_env_and_caps(self):
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = list(args)
            captured["kwargs"] = kwargs
            # Simulate pdftotext writing the output file.
            out_path = Path(args[-1])
            out_path.write_text("PAGE ONE\fPAGE TWO", encoding="utf-8")

            class _Completed:
                returncode = 0

            return _Completed()

        original_run = pdf_ingestion.subprocess.run
        original_which = pdf_ingestion.shutil.which
        pdf_ingestion.subprocess.run = fake_run
        pdf_ingestion.shutil.which = lambda name: "/usr/bin/pdftotext"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-not-reach-parser"
        try:
            pages = extract_pdf_text_pages(Path("paper.pdf"), timeout_seconds=12.0)
        finally:
            pdf_ingestion.subprocess.run = original_run
            pdf_ingestion.shutil.which = original_which
            del os.environ["ANTHROPIC_API_KEY"]

        self.assertEqual(pages, ["PAGE ONE", "PAGE TWO"])
        self.assertEqual(captured["kwargs"]["timeout"], 12.0)
        self.assertTrue(captured["kwargs"]["check"])
        self.assertNotIn("ANTHROPIC_API_KEY", captured["kwargs"]["env"])
        self.assertIn("-l", captured["args"])
        self.assertIn(str(pdf_ingestion.PDF_PARSE_MAX_PAGES), captured["args"])


if __name__ == "__main__":
    unittest.main()
