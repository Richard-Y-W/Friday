from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEEDBACK_PROPOSAL_DECISIONS = ("approved", "rejected")


class FeedbackReviewError(ValueError):
    """Raised when a feedback tuning proposal cannot be reviewed."""


def build_feedback_proposal_review(package_dir: Path) -> str:
    proposal = _load_required_proposal(package_dir)
    decision = _load_json(package_dir / "tuning_decision.json")
    lines = [
        "# Feedback Tuning Proposal Review",
        "",
        f"Package: {proposal.get('package_dir') or str(package_dir)}",
        f"Status: {_display_value(proposal.get('status'))}",
        f"Interpreter: {_display_value(proposal.get('interpreter_provider'))}/{_display_value(proposal.get('interpreter_model') or '(default)')}",
    ]
    if decision:
        lines.append(f"Decision: {_display_value(decision.get('decision'))}")
    summary = str(proposal.get("summary") or "").strip()
    if summary:
        lines.extend(["", "## Summary", "", summary])
    signals = proposal.get("signal_types", [])
    if isinstance(signals, list) and signals:
        lines.extend(["", "## Signals", ""])
        for signal in signals:
            lines.append(f"- {signal}")
    lines.extend(["", "## Proposed Actions", ""])
    actions = proposal.get("proposed_actions", [])
    if isinstance(actions, list) and actions:
        for index, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                continue
            lines.append(f"{index}. {action.get('target', 'unknown')}: {action.get('action', 'review')}")
            lines.append(f"   Value: {_display_value(action.get('value'))}")
            reason = str(action.get("reason") or "").strip()
            if reason:
                lines.append(f"   Reason: {reason}")
    else:
        lines.append("- No concrete tuning action proposed.")
    gates = proposal.get("required_eval_gates", [])
    if isinstance(gates, list) and gates:
        lines.extend(["", "## Required Gates", ""])
        for gate in gates:
            lines.append(f"- {gate}")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"Approve: friday feedback approve --package {package_dir}",
            f"Reject: friday feedback reject --package {package_dir}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_feedback_decision_files(
    package_dir: Path,
    *,
    decision: str,
    note: str | None = None,
    reviewer: str | None = None,
) -> dict[str, str]:
    if decision not in FEEDBACK_PROPOSAL_DECISIONS:
        raise FeedbackReviewError(f"unsupported feedback proposal decision: {decision}")
    proposal = _load_required_proposal(package_dir)
    apply_status = "not_applied" if decision == "approved" else "not_applicable"
    payload = {
        "schema_version": "1.0",
        "artifact_type": "tuning_decision",
        "package_dir": str(package_dir),
        "proposal_artifact": str(package_dir / "tuning_proposal.json"),
        "proposal_status": decision,
        "decision": decision,
        "apply_status": apply_status,
        "auto_apply": False,
        "required_eval_gates": _as_string_list(proposal.get("required_eval_gates")),
        "proposal_summary": str(proposal.get("summary") or ""),
        "proposed_action_count": _list_len(proposal.get("proposed_actions")),
        "human_review": {
            "note": note or "",
            "reviewer": reviewer or "",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        },
        "next_actions": _decision_next_actions(decision, proposal),
    }
    return {
        "tuning_decision.json": _json_text(payload),
        "tuning_decision.md": _render_decision_markdown(payload),
    }


def _decision_next_actions(decision: str, proposal: dict[str, Any]) -> list[str]:
    if decision == "approved":
        gates = ", ".join(_as_string_list(proposal.get("required_eval_gates"))) or "configured eval gates"
        return [
            f"Run eval gates before applying: {gates}.",
            "Apply only through the eval-gated feedback apply command.",
        ]
    return ["No tuning will be applied from this proposal unless it is revised and approved later."]


def _render_decision_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Feedback Tuning Decision",
        "",
        f"Decision: {payload.get('decision')}",
        f"Proposal status: {payload.get('proposal_status')}",
        f"Apply status: {payload.get('apply_status')}",
        f"Package: {payload.get('package_dir')}",
        "",
    ]
    human_review = payload.get("human_review", {})
    if isinstance(human_review, dict):
        lines.extend(
            [
                "## Human Review",
                "",
                f"Reviewer: {_display_value(human_review.get('reviewer'))}",
                f"Note: {_display_value(human_review.get('note'))}",
                f"Reviewed at: {_display_value(human_review.get('reviewed_at'))}",
                "",
            ]
        )
    lines.extend(["## Eval gates before apply", ""])
    gates = payload.get("required_eval_gates", [])
    if isinstance(gates, list) and gates:
        for gate in gates:
            lines.append(f"- {gate}")
    else:
        lines.append("- No eval gates listed.")
    next_actions = payload.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.extend(["", "## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _load_required_proposal(package_dir: Path) -> dict[str, Any]:
    if not package_dir.exists() or not package_dir.is_dir():
        raise FeedbackReviewError(f"package directory not found: {package_dir}")
    path = package_dir / "tuning_proposal.json"
    proposal = _load_json(path)
    if not proposal:
        raise FeedbackReviewError(f"tuning proposal not found: {path}")
    if proposal.get("artifact_type") != "tuning_proposal":
        raise FeedbackReviewError(f"invalid tuning proposal artifact: {path}")
    return proposal


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
