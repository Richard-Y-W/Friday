import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.corpus_adapters import (
    import_folder_corpus,
    import_obsidian_corpus,
    import_zotero_corpus,
)


class CorpusAdapterTests(unittest.TestCase):
    def test_folder_adapter_imports_pdfs_and_rejects_unsupported_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MALDI AMR paper.pdf").write_bytes(b"%PDF-1.4")
            (root / "notes.txt").write_text("not a paper", encoding="utf-8")

            corpus, rejection_log = import_folder_corpus(root)

            self.assertEqual(corpus["artifact_type"], "literature_corpus")
            self.assertEqual(corpus["literature_corpus"][0]["title"], "MALDI AMR paper")
            self.assertEqual(corpus["literature_corpus"][0]["source_type"], "pdf")
            self.assertEqual(rejection_log["rejected"][0]["reason"], "unsupported_file_type")

    def test_zotero_adapter_imports_csl_json(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "zotero.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "smith2024maldi",
                            "title": "MALDI antimicrobial resistance prediction",
                            "author": [{"family": "Smith", "given": "A."}],
                            "issued": {"date-parts": [[2024]]},
                            "DOI": "10.1038/example",
                            "container-title": "Nature Medicine",
                            "abstract": "A study about antimicrobial resistance.",
                            "URL": "https://doi.org/10.1038/example",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            corpus, rejection_log = import_zotero_corpus(path)

            self.assertEqual(rejection_log["rejected"], [])
            item = corpus["literature_corpus"][0]
            self.assertEqual(item["citation_key"], "smith2024maldi")
            self.assertEqual(item["doi"], "10.1038/example")
            self.assertEqual(item["authors"][0]["family"], "Smith")

    def test_obsidian_adapter_reads_frontmatter_and_rejects_missing_title(self):
        with TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / "paper.md").write_text(
                "---\ntitle: MALDI AMR paper\nyear: 2024\ndoi: 10.1038/example\nauthors: Smith; Jones\n---\nNotes",
                encoding="utf-8",
            )
            (vault / "untitled.md").write_text("---\nyear: 2024\n---\nNo heading", encoding="utf-8")

            corpus, rejection_log = import_obsidian_corpus(vault)

            self.assertEqual(corpus["literature_corpus"][0]["title"], "MALDI AMR paper")
            self.assertEqual(corpus["literature_corpus"][0]["authors"][0]["family"], "Smith")
            self.assertEqual(rejection_log["rejected"][0]["reason"], "missing_required_field")


if __name__ == "__main__":
    unittest.main()
