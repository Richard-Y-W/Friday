import unittest

from friday.llm.parse import extract_json, strip_markdown_fences


class StripMarkdownFencesTests(unittest.TestCase):
    def test_returns_plain_text_unchanged(self):
        self.assertEqual(strip_markdown_fences("hello world"), "hello world")

    def test_strips_language_tagged_fence(self):
        text = "```json\n{\"a\": 1}\n```"
        self.assertEqual(strip_markdown_fences(text), '{"a": 1}')

    def test_strips_fence_with_leading_prose(self):
        text = "Here is the answer:\n```json\n{\"a\": 1}\n```"
        self.assertEqual(strip_markdown_fences(text), '{"a": 1}')

    def test_handles_crlf(self):
        text = "```\r\n{\"a\": 1}\r\n```"
        self.assertEqual(strip_markdown_fences(text), '{"a": 1}')


class ExtractJsonTests(unittest.TestCase):
    def test_parses_fenced_object(self):
        self.assertEqual(extract_json("```json\n{\"label\": \"relevant\"}\n```"), {"label": "relevant"})

    def test_parses_unfenced_object(self):
        self.assertEqual(extract_json('{"label": "relevant"}'), {"label": "relevant"})

    def test_parses_object_after_prose(self):
        self.assertEqual(extract_json('Sure! {"n": 2}'), {"n": 2})

    def test_parses_array(self):
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_returns_none_on_garbage(self):
        self.assertIsNone(extract_json("not json at all"))

    def test_returns_none_on_empty(self):
        self.assertIsNone(extract_json(""))


if __name__ == "__main__":
    unittest.main()
