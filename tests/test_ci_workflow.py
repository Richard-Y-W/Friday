import unittest
from pathlib import Path


class CiWorkflowTests(unittest.TestCase):
    def test_github_actions_runs_tests_and_eval_suite_on_main_changes(self):
        workflow_path = Path(".github/workflows/ci.yml")
        self.assertTrue(workflow_path.exists(), "CI workflow must exist at .github/workflows/ci.yml")
        workflow = workflow_path.read_text(encoding="utf-8")

        self.assertIn("name: CI", workflow)
        self.assertIn("push:", workflow)
        self.assertIn("pull_request:", workflow)
        self.assertIn("branches: [main]", workflow)
        self.assertIn("actions/checkout@v4", workflow)
        self.assertIn("actions/setup-python@v5", workflow)
        self.assertIn("python3 -m unittest discover -v", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run --suite biomedical", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run --suite natural-language", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run --suite safety", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run --suite gold", workflow)
        self.assertIn("python3 -m jarvis_research eval-suite run --suite real-smoke", workflow)
