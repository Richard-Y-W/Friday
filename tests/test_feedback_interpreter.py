import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.feedback_interpreter import (
    FeedbackInterpretationError,
    build_feedback_tuning_proposal_files,
)
from friday.llm.types import LLMRequest, LLMResponse


class FeedbackInterpreterTests(unittest.TestCase):
    def test_build_feedback_tuning_proposal_uses_feedback_role(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_feedback_package(package_dir)
            router = RecordingRouter(
                LLMResponse(
                    provider="codex_cli",
                    model="",
                    success=True,
                    text=json.dumps(
                        {
                            "summary": "The user flagged readability and weak evidence.",
                            "signal_types": ["prose_quality", "faithfulness"],
                            "proposed_actions": [
                                {
                                    "target": "report_prose_quality",
                                    "action": "add_blocked_phrase",
                                    "value": "evidence includes",
                                    "reason": "The user said the report was hard to read.",
                                }
                            ],
                            "benchmark_recommendation": "save_failure",
                        }
                    ),
                )
            )

            files = build_feedback_tuning_proposal_files(package_dir, router=router)

            self.assertEqual(router.calls[0][0], "feedback")
            self.assertIsInstance(router.calls[0][1], LLMRequest)
            proposal = json.loads(files["tuning_proposal.json"])
            self.assertEqual(proposal["artifact_type"], "tuning_proposal")
            self.assertEqual(proposal["interpreter_provider"], "codex_cli")
            self.assertEqual(proposal["status"], "proposed")
            self.assertEqual(proposal["signal_types"], ["prose_quality", "faithfulness"])
            self.assertEqual(proposal["proposed_actions"][0]["target"], "report_prose_quality")
            self.assertIn("Eval gates required", files["tuning_proposal.md"])
            self.assertIn("feedback_interpreter_prompt.json", files)

    def test_build_feedback_tuning_proposal_requires_available_interpreter(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "report-package"
            _write_feedback_package(package_dir)
            router = RecordingRouter(
                LLMResponse(
                    provider="codex_cli",
                    model="",
                    success=False,
                    error="provider unavailable: codex not logged in",
                )
            )

            with self.assertRaises(FeedbackInterpretationError) as caught:
                build_feedback_tuning_proposal_files(package_dir, router=router)

            self.assertIn("provider unavailable", str(caught.exception))


class RecordingRouter:
    def __init__(self, response: LLMResponse):
        self.response = response
        self.calls = []

    def generate(self, role: str, request: LLMRequest) -> LLMResponse:
        self.calls.append((role, request))
        return self.response


def _write_feedback_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "draft_feedback.json",
        {
            "artifact_type": "draft_feedback",
            "review_status": "captured",
            "trust_verdict": "needs_review",
            "trust_action": "human_review",
            "human_feedback": {
                "decision": "needs_rewrite",
                "note": "The summary is hard to read and the evidence feels broad.",
                "reviewer": "byung",
            },
            "review_items": [
                {
                    "id": "R1",
                    "source": "trust_score",
                    "rule": "critic_not_run",
                    "severity": "important",
                }
            ],
        },
    )
    _write_json(
        package_dir / "report_trust_score.json",
        {
            "score": 82,
            "verdict": "needs_review",
            "action": "human_review",
            "reasons": ["critic_not_run"],
        },
    )
    _write_json(
        package_dir / "report_prose_quality.json",
        {
            "status": "fallback",
            "issues": [
                {
                    "rule": "raw_evidence_dump_phrase",
                    "examples": ["Across 3 papers, evidence includes..."],
                }
            ],
        },
    )
    _write_json(
        package_dir / "report_faithfulness_audit.json",
        {
            "status": "fallback",
            "issues": [
                {
                    "rule": "weak_evidence_overlap",
                    "sentence": "One paper proved deployment [1, p. 2].",
                }
            ],
        },
    )
    (package_dir / "report.md").write_text("# Friday Research Report\n", encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
