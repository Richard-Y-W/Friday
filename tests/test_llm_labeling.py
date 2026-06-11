import json
import unittest

from jarvis_research.discovery import Candidate
from jarvis_research.llm_labeling import (
    LlmLabelingError,
    build_llm_label_payload,
    build_openai_responses_request,
    parse_llm_label_response,
)
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


class LlmLabelingTests(unittest.TestCase):
    def test_builds_metadata_only_payload_without_tool_or_file_fields(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = JarvisStore(Path(tmp) / "jarvis.db")
            batch = store.create_batch(query="what is the importance of math in language", limit=1, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Mathematical Linguistics",
                source_for_gate="10.1007/978-1-84628-986-6",
                abstract="Formal language theory and syntax are central topics.",
                doi="10.1007/978-1-84628-986-6",
                journal="Springer",
                concepts="Mathematical linguistics; formal grammar",
                relevance_score=17,
            )
            item = store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            payload = build_llm_label_payload(batch.query, item)

            self.assertEqual(payload["query_plan"]["intent"], "mathematical_linguistics")
            self.assertIn("mathematical linguistics", payload["query_plan"]["expanded_queries"])
            self.assertEqual(payload["candidate"]["title"], "Mathematical Linguistics")
            self.assertEqual(payload["candidate"]["doi"], "10.1007/978-1-84628-986-6")
            self.assertNotIn("local_path", payload["candidate"])
            self.assertNotIn("pdf_text", payload["candidate"])
            self.assertNotIn("tools", json.dumps(payload).lower())

    def test_builds_strict_json_schema_openai_request(self):
        request = build_openai_responses_request(
            model="gpt-test",
            payload={
                "query": "math in language",
                "query_plan": {"intent": "mathematical_linguistics", "expanded_queries": []},
                "candidate": {"title": "Mathematical Linguistics"},
            },
        )

        self.assertEqual(request["model"], "gpt-test")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertIn("Do not browse", request["input"][0]["content"])
        self.assertIn("untrusted", request["input"][0]["content"].lower())

    def test_parses_and_validates_llm_response(self):
        result = parse_llm_label_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "label": "relevant",
                                        "confidence": 0.91,
                                        "rationale": "Title and abstract match mathematical linguistics.",
                                        "evidence_terms": ["mathematical linguistics", "formal grammar"],
                                        "exclusion_reason": None,
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(result.label, "relevant")
        self.assertEqual(result.confidence, 0.91)
        self.assertEqual(result.evidence_terms, ("mathematical linguistics", "formal grammar"))
        self.assertIsNone(result.exclusion_reason)

    def test_rejects_invalid_llm_label(self):
        with self.assertRaises(LlmLabelingError):
            parse_llm_label_response(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "label": "definitely",
                                            "confidence": 2,
                                            "rationale": "",
                                            "evidence_terms": "not-list",
                                            "exclusion_reason": None,
                                        }
                                    ),
                                }
                            ],
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
