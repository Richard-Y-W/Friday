import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.draft_review import build_draft_feedback_files


class DraftReviewTests(unittest.TestCase):
    def test_build_draft_feedback_prefills_review_items_from_trust_and_audits(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_review_fixture_package(package_dir)

            files = build_draft_feedback_files(package_dir)

            feedback = json.loads(files["draft_feedback.json"])
            self.assertEqual(feedback["artifact_type"], "draft_feedback")
            self.assertEqual(feedback["review_status"], "pending")
            self.assertEqual(feedback["trust_verdict"], "needs_review")
            self.assertEqual(feedback["trust_action"], "human_review")
            self.assertEqual(feedback["human_feedback"], {})
            self.assertEqual(feedback["review_items"][0]["rule"], "critic_not_run")
            self.assertIn("global deployment", json.dumps(feedback["review_items"]))

            queue = files["review_queue.md"]
            self.assertIn("# Draft Review Queue", queue)
            self.assertIn("Trust verdict: needs_review", queue)
            self.assertIn("R1", queue)

    def test_build_draft_feedback_captures_human_decision(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_review_fixture_package(package_dir)

            files = build_draft_feedback_files(
                package_dir,
                decision="bad_evidence",
                note="The deployment sentence is unsupported.",
                reviewer="byung",
            )

            feedback = json.loads(files["draft_feedback.json"])
            self.assertEqual(feedback["review_status"], "captured")
            self.assertEqual(feedback["human_feedback"]["decision"], "bad_evidence")
            self.assertEqual(feedback["human_feedback"]["reviewer"], "byung")
            self.assertIn("unsupported", feedback["human_feedback"]["note"])
            self.assertIn("Human decision: bad_evidence", files["review_queue.md"])


def _write_review_fixture_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "report_trust_score.json",
        {
            "schema_version": "1.0",
            "artifact_type": "report_trust_score",
            "score": 82,
            "verdict": "needs_review",
            "action": "human_review",
            "reasons": ["critic_not_run"],
            "components": {"critic": "not_run"},
        },
    )
    _write_json(package_dir / "report_manifest.json", {"trust_verdict": "needs_review", "trust_action": "human_review"})
    _write_json(package_dir / "report_prose_quality.json", {"status": "pass", "issues": []})
    _write_json(
        package_dir / "report_faithfulness_audit.json",
        {
            "status": "fallback",
            "issues": [
                {
                    "tier": "B",
                    "rule": "weak_evidence_overlap",
                    "sentence": "One paper proved global deployment [1, p. 2].",
                    "missing_terms": ["global", "deployment"],
                }
            ],
        },
    )
    _write_json(package_dir / "citation_audit.json", {"status": "pass", "unknown_citations": []})
    (package_dir / "report.md").write_text("# Friday Research Report\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
