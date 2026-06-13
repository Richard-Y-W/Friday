import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.feedback_review import (
    FeedbackReviewError,
    build_feedback_decision_files,
    build_feedback_proposal_review,
)


class FeedbackReviewTests(unittest.TestCase):
    def test_build_feedback_proposal_review_summarizes_proposal_and_actions(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_tuning_package(package_dir)

            review = build_feedback_proposal_review(package_dir)

            self.assertIn("# Feedback Tuning Proposal Review", review)
            self.assertIn("Status: proposed", review)
            self.assertIn("report_prose_quality", review)
            self.assertIn("friday feedback approve", review)

    def test_build_feedback_decision_files_approves_proposal_without_applying(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_tuning_package(package_dir)

            files = build_feedback_decision_files(
                package_dir,
                decision="approved",
                note="This matches the failure.",
                reviewer="byung",
            )

            decision = json.loads(files["tuning_decision.json"])
            self.assertEqual(decision["artifact_type"], "tuning_decision")
            self.assertEqual(decision["decision"], "approved")
            self.assertEqual(decision["proposal_status"], "approved")
            self.assertEqual(decision["apply_status"], "not_applied")
            self.assertEqual(decision["required_eval_gates"], ["gold", "real-smoke"])
            self.assertEqual(decision["human_review"]["reviewer"], "byung")
            self.assertIn("Eval gates before apply", files["tuning_decision.md"])

    def test_build_feedback_decision_files_rejects_proposal(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_tuning_package(package_dir)

            files = build_feedback_decision_files(
                package_dir,
                decision="rejected",
                note="Too broad.",
            )

            decision = json.loads(files["tuning_decision.json"])
            self.assertEqual(decision["decision"], "rejected")
            self.assertEqual(decision["proposal_status"], "rejected")
            self.assertEqual(decision["apply_status"], "not_applicable")
            self.assertIn("Decision: rejected", files["tuning_decision.md"])

    def test_build_feedback_review_requires_proposal(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FeedbackReviewError):
                build_feedback_proposal_review(Path(tmp) / "missing-package")


def _write_tuning_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "tuning_proposal.json",
        {
            "schema_version": "1.0",
            "artifact_type": "tuning_proposal",
            "status": "proposed",
            "package_dir": str(package_dir),
            "interpreter_provider": "codex_cli",
            "interpreter_model": "",
            "summary": "The user flagged readability.",
            "signal_types": ["prose_quality"],
            "proposed_actions": [
                {
                    "target": "report_prose_quality",
                    "action": "add_blocked_phrase",
                    "value": "evidence includes",
                    "reason": "The report read like stitched evidence.",
                }
            ],
            "benchmark_recommendation": "save_failure",
            "required_eval_gates": ["gold", "real-smoke"],
            "auto_apply": False,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
