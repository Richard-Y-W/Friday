from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from friday.llm.parse import extract_json
from friday.llm.types import LLMRequest


REQUIRED_EVAL_GATES = ("gold", "real-smoke")


class FeedbackInterpretationError(RuntimeError):
    """Raised when the selected feedback interpreter cannot produce a proposal."""


def build_feedback_tuning_proposal_files(package_dir: Path, *, router: Any) -> dict[str, str]:
    feedback = _load_required_json(package_dir / "draft_feedback.json")
    context = _feedback_context(package_dir, feedback)
    system_prompt, prompt = _feedback_interpreter_prompts(context)
    response = router.generate(
        "feedback",
        LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.0,
        ),
    )
    if not getattr(response, "success", False):
        raise FeedbackInterpretationError(str(getattr(response, "error", "") or "feedback interpreter failed"))

    parsed = extract_json(str(getattr(response, "text", "") or ""))
    if not isinstance(parsed, dict):
        raise FeedbackInterpretationError("feedback interpreter returned unparseable output")

    proposal = _normalize_proposal(parsed, response, context)
    return {
        "feedback_interpreter_prompt.json": _json_text(
            {
                "schema_version": "1.0",
                "artifact_type": "feedback_interpreter_prompt",
                "role": "feedback",
                "system_prompt": system_prompt,
                "prompt": prompt,
            }
        ),
        "tuning_proposal.json": _json_text(proposal),
        "tuning_proposal.md": _render_tuning_proposal_markdown(proposal),
    }


def _feedback_context(package_dir: Path, feedback: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "feedback_interpreter_context",
        "package_dir": str(package_dir),
        "draft_feedback": feedback,
        "trust_score": _load_json(package_dir / "report_trust_score.json"),
        "prose_quality": _load_json(package_dir / "report_prose_quality.json"),
        "faithfulness": _load_json(package_dir / "report_faithfulness_audit.json"),
        "critic": _load_json(package_dir / "report_critic_audit.json"),
        "revision": _load_json(package_dir / "report_revision_audit.json"),
        "citation": _load_json(package_dir / "citation_audit.json"),
    }


def _feedback_interpreter_prompts(context: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are Friday's feedback interpreter. Convert a human draft review and "
        "existing report audits into a small tuning proposal. Do not browse, do "
        "not call tools, and do not invent evidence. Return JSON only."
    )
    prompt = (
        "Interpret this review packet into JSON with keys: summary, signal_types, "
        "proposed_actions, benchmark_recommendation. signal_types must be a list "
        "using values such as prose_quality, faithfulness, citation, evidence_quality, "
        "topic_routing, or benchmark. proposed_actions must be a list of objects "
        "with target, action, value, and reason. The proposal is advisory only; "
        "Friday will require human approval and eval gates before applying changes.\n\n"
        + json.dumps(context, indent=2, sort_keys=True)
    )
    return system_prompt, prompt


def _normalize_proposal(parsed: dict[str, Any], response: Any, context: dict[str, Any]) -> dict[str, Any]:
    signal_types = parsed.get("signal_types", [])
    if not isinstance(signal_types, list):
        signal_types = [str(signal_types)]
    proposed_actions = parsed.get("proposed_actions", [])
    if not isinstance(proposed_actions, list):
        proposed_actions = []
    normalized_actions = [
        {
            "target": str(action.get("target") or "unknown"),
            "action": str(action.get("action") or "review"),
            "value": action.get("value", ""),
            "reason": str(action.get("reason") or ""),
        }
        for action in proposed_actions
        if isinstance(action, dict)
    ]
    feedback = context.get("draft_feedback", {})
    human_feedback = feedback.get("human_feedback") if isinstance(feedback, dict) else {}
    if not normalized_actions and isinstance(human_feedback, dict):
        decision = str(human_feedback.get("decision") or "")
        normalized_actions = _fallback_actions_for_decision(decision)

    return {
        "schema_version": "1.0",
        "artifact_type": "tuning_proposal",
        "status": "proposed",
        "package_dir": context.get("package_dir"),
        "interpreter_provider": getattr(response, "provider", "unknown"),
        "interpreter_model": getattr(response, "model", ""),
        "summary": str(parsed.get("summary") or "").strip(),
        "signal_types": [str(signal).strip() for signal in signal_types if str(signal).strip()],
        "proposed_actions": normalized_actions,
        "benchmark_recommendation": str(parsed.get("benchmark_recommendation") or "none"),
        "required_eval_gates": list(REQUIRED_EVAL_GATES),
        "auto_apply": False,
        "raw_interpreter_output": parsed,
    }


def _fallback_actions_for_decision(decision: str) -> list[dict[str, str]]:
    if decision in {"needs_rewrite", "too_confusing"}:
        return [
            {
                "target": "report_prose_quality",
                "action": "review_style_rules",
                "value": decision,
                "reason": "Human feedback flagged report readability.",
            }
        ]
    if decision in {"bad_evidence", "citation_issue", "reject"}:
        return [
            {
                "target": "report_faithfulness_audit",
                "action": "review_evidence_rules",
                "value": decision,
                "reason": "Human feedback flagged evidence or citation reliability.",
            }
        ]
    if decision == "approve":
        return [
            {
                "target": "benchmark_pack",
                "action": "save_positive_example",
                "value": "approved_report",
                "reason": "Human feedback approved this draft.",
            }
        ]
    return []


def _render_tuning_proposal_markdown(proposal: dict[str, Any]) -> str:
    lines = [
        "# Feedback Tuning Proposal",
        "",
        f"Status: {proposal.get('status', '-')}",
        f"Interpreter: {proposal.get('interpreter_provider', '-')}/{proposal.get('interpreter_model', '') or '(default)'}",
        f"Package: {proposal.get('package_dir', '-')}",
        "",
    ]
    summary = str(proposal.get("summary") or "").strip()
    if summary:
        lines.extend(["## Summary", "", summary, ""])
    signals = proposal.get("signal_types", [])
    if isinstance(signals, list) and signals:
        lines.extend(["## Signals", ""])
        for signal in signals:
            lines.append(f"- {signal}")
        lines.append("")
    actions = proposal.get("proposed_actions", [])
    lines.extend(["## Proposed Actions", ""])
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
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "Eval gates required before applying:",
        ]
    )
    for gate in proposal.get("required_eval_gates", []):
        lines.append(f"- {gate}")
    lines.append("")
    lines.append("Auto-apply: false")
    return "\n".join(lines).rstrip() + "\n"


def _load_required_json(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not value:
        raise FeedbackInterpretationError(f"required feedback artifact not found: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
