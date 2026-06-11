import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.corpus_routing import route_corpus_query


class CorpusRoutingTests(unittest.TestCase):
    def test_routes_to_corpus_when_enough_entries_match_query(self):
        with TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "corpus.json"
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
                            },
                            {
                                "citation_key": "noise2024",
                                "title": "Clinical speech assessment",
                                "abstract": "A clinical study about speech pathology.",
                                "source_pointer": "https://doi.org/10.1000/noise",
                                "source_type": "zotero",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = route_corpus_query(
                "what is the importance of math in language",
                [corpus_path],
                min_score=12,
                min_matches=1,
                limit=5,
            )

            self.assertTrue(result.should_use_corpus)
            self.assertEqual(result.loaded_count, 2)
            self.assertEqual(len(result.matches), 1)
            self.assertEqual(result.matches[0].title, "Mathematical structure in natural language")
            self.assertGreaterEqual(result.matches[0].score, 12)
            self.assertIn("math", result.matches[0].matched_terms)
            self.assertIn("language", result.matches[0].matched_terms)

    def test_falls_back_when_corpus_matches_are_too_weak(self):
        with TemporaryDirectory() as tmp:
            corpus_path = Path(tmp) / "corpus.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "artifact_type": "literature_corpus",
                        "literature_corpus": [
                            {
                                "citation_key": "clinical2024",
                                "title": "Clinical speech assessment",
                                "abstract": "A trial about speech outcomes.",
                                "source_pointer": "https://doi.org/10.1000/clinical",
                                "source_type": "zotero",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = route_corpus_query(
                "what is the importance of math in language",
                [corpus_path],
                min_score=12,
                min_matches=1,
                limit=5,
            )

            self.assertFalse(result.should_use_corpus)
            self.assertEqual(result.matches, [])

    def test_ignores_missing_and_invalid_corpus_paths(self):
        with TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing.json"
            invalid_path = Path(tmp) / "invalid.json"
            invalid_path.write_text("{not json", encoding="utf-8")

            result = route_corpus_query(
                "MALDI antimicrobial resistance",
                [missing_path, invalid_path],
                min_score=8,
                min_matches=1,
                limit=5,
            )

            self.assertFalse(result.should_use_corpus)
            self.assertEqual(result.loaded_count, 0)
            self.assertEqual(len(result.rejected_paths), 2)


if __name__ == "__main__":
    unittest.main()
