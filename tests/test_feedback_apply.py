import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.feedback_apply import FeedbackApplyError, apply_feedback_tuning


class FeedbackApplyTests(unittest.TestCase):
    def test_apply_feedback_tuning_runs_eval_gates_and_writes_local_rules(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "report-package"
            data_dir = root / ".friday"
            _write_approved_tuning_package(package_dir)
            seen_suites = []

            def eval_runner(suite):
                seen_suites.append(suite)
                return _eval_report(suite, "pass")

            files = apply_feedback_tuning(package_dir, data_dir, eval_runner=eval_runner)

            self.assertEqual(seen_suites, ["gold", "real-smoke"])
            apply = json.loads(files["tuning_apply.json"])
            self.assertEqual(apply["artifact_type"], "tuning_apply")
            self.assertEqual(apply["status"], "applied")
            self.assertEqual(apply["eval_status"], "pass")
            self.assertEqual(apply["applied_changes"][0]["target"], "report_prose_quality")
            rules = json.loads((data_dir / "feedback" / "rules" / "prose_quality.json").read_text(encoding="utf-8"))
            self.assertEqual(rules["artifact_type"], "feedback_rule_store")
            self.assertEqual(rules["rules"][0]["value"], "evidence includes")

    def test_apply_feedback_tuning_blocks_when_eval_gate_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "report-package"
            data_dir = root / ".friday"
            _write_approved_tuning_package(package_dir)

            def eval_runner(suite):
                return _eval_report(suite, "fail" if suite == "gold" else "pass")

            files = apply_feedback_tuning(package_dir, data_dir, eval_runner=eval_runner)

            apply = json.loads(files["tuning_apply.json"])
            self.assertEqual(apply["status"], "blocked")
            self.assertEqual(apply["eval_status"], "fail")
            self.assertFalse((data_dir / "feedback" / "rules" / "prose_quality.json").exists())
            self.assertIn("not applied", files["tuning_apply.md"])

    def test_apply_feedback_tuning_requires_approved_decision(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            data_dir = Path(tmp) / ".friday"
            _write_approved_tuning_package(package_dir, decision="rejected", apply_status="not_applicable")

            with self.assertRaises(FeedbackApplyError):
                apply_feedback_tuning(package_dir, data_dir, eval_runner=lambda suite: _eval_report(suite, "pass"))


def _write_approved_tuning_package(
    package_dir: Path,
    *,
    decision: str = "approved",
    apply_status: str = "not_applied",
) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "tuning_proposal.json",
        {
            "schema_version": "1.0",
            "artifact_type": "tuning_proposal",
            "status": "proposed",
            "package_dir": str(package_dir),
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
            "required_eval_gates": ["gold", "real-smoke"],
            "auto_apply": False,
        },
    )
    _write_json(
        package_dir / "tuning_decision.json",
        {
            "schema_version": "1.0",
            "artifact_type": "tuning_decision",
            "decision": decision,
            "proposal_status": decision,
            "apply_status": apply_status,
            "required_eval_gates": ["gold", "real-smoke"],
            "human_review": {"note": "Looks right.", "reviewer": "byung"},
        },
    )


def _eval_report(suite: str, status: str) -> dict:
    return {
        "schema_version": "1.0",
        "artifact_type": "eval_suite_report",
        "suite": suite,
        "status": status,
        "counts": {"total": 1, "passed": 1 if status == "pass" else 0, "failed": 0 if status == "pass" else 1},
        "cases": [],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
