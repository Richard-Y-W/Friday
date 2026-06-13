from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from friday.eval_suite import run_eval_suite


DEFAULT_REQUIRED_EVAL_GATES = ("gold", "real-smoke")


class FeedbackApplyError(ValueError):
    """Raised when a tuning proposal is not eligible for safe apply."""


def apply_feedback_tuning(
    package_dir: Path,
    data_dir: Path,
    *,
    eval_runner: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, str]:
    proposal = _load_required_json(package_dir / "tuning_proposal.json", artifact_type="tuning_proposal")
    decision = _load_required_json(package_dir / "tuning_decision.json", artifact_type="tuning_decision")
    _validate_decision_for_apply(decision)
    gates = _required_gates(decision, proposal)
    runner = eval_runner or run_eval_suite
    eval_reports = [_run_eval_gate(gate, runner) for gate in gates]
    eval_status = "pass" if all(report.get("status") == "pass" for report in eval_reports) else "fail"
    if eval_status != "pass":
        payload = _apply_payload(
            package_dir,
            data_dir,
            proposal,
            decision,
            eval_reports,
            status="blocked",
            applied_changes=[],
            blocked_reason="eval_gate_failed",
        )
        return {
            "tuning_apply.json": _json_text(payload),
            "tuning_apply.md": _render_apply_markdown(payload),
        }

    applied_changes = _apply_rule_changes(package_dir, data_dir, proposal)
    payload = _apply_payload(
        package_dir,
        data_dir,
        proposal,
        decision,
        eval_reports,
        status="applied",
        applied_changes=applied_changes,
        blocked_reason=None,
    )
    return {
        "tuning_apply.json": _json_text(payload),
        "tuning_apply.md": _render_apply_markdown(payload),
    }


def _validate_decision_for_apply(decision: dict[str, Any]) -> None:
    if decision.get("decision") != "approved":
        raise FeedbackApplyError("tuning proposal must be approved before apply")
    if decision.get("apply_status") != "not_applied":
        raise FeedbackApplyError(f"tuning proposal is not applyable: apply_status={decision.get('apply_status')}")


def _required_gates(decision: dict[str, Any], proposal: dict[str, Any]) -> list[str]:
    gates = _as_string_list(decision.get("required_eval_gates")) or _as_string_list(proposal.get("required_eval_gates"))
    return gates or list(DEFAULT_REQUIRED_EVAL_GATES)


def _run_eval_gate(suite: str, runner: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    try:
        report = runner(suite)
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "artifact_type": "eval_suite_report",
            "suite": suite,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "counts": {"total": 0, "passed": 0, "failed": 1},
            "cases": [],
        }
    if not isinstance(report, dict):
        return {
            "schema_version": "1.0",
            "artifact_type": "eval_suite_report",
            "suite": suite,
            "status": "fail",
            "error": "eval runner returned non-dict report",
            "counts": {"total": 0, "passed": 0, "failed": 1},
            "cases": [],
        }
    return report


def _apply_rule_changes(package_dir: Path, data_dir: Path, proposal: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    actions = proposal.get("proposed_actions", [])
    if not isinstance(actions, list):
        return changes
    for action in actions:
        if not isinstance(action, dict):
            continue
        target = str(action.get("target") or "general")
        rule_path = _rule_path(data_dir, target)
        rule_record = {
            "source_package": str(package_dir),
            "target": target,
            "action": str(action.get("action") or "review"),
            "value": action.get("value", ""),
            "reason": str(action.get("reason") or ""),
            "proposal_summary": str(proposal.get("summary") or ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_rule(rule_path, target, rule_record)
        changes.append(
            {
                "target": target,
                "action": rule_record["action"],
                "value": rule_record["value"],
                "reason": rule_record["reason"],
                "file": str(rule_path),
            }
        )
    return changes


def _append_rule(path: Path, target: str, rule: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = _load_json(path)
    if not store:
        store = {
            "schema_version": "1.0",
            "artifact_type": "feedback_rule_store",
            "target": target,
            "rules": [],
        }
    rules = store.get("rules")
    if not isinstance(rules, list):
        rules = []
    rules.append(rule)
    store["rules"] = rules
    path.write_text(_json_text(store) + "\n", encoding="utf-8")


def _rule_path(data_dir: Path, target: str) -> Path:
    filename = {
        "report_prose_quality": "prose_quality.json",
        "report_faithfulness_audit": "faithfulness.json",
        "citation_audit": "citation.json",
        "topic_routing": "topic_routing.json",
        "topic_profile": "topic_routing.json",
        "benchmark_pack": "benchmark.json",
    }.get(target, "general.json")
    return data_dir / "feedback" / "rules" / filename


def _apply_payload(
    package_dir: Path,
    data_dir: Path,
    proposal: dict[str, Any],
    decision: dict[str, Any],
    eval_reports: list[dict[str, Any]],
    *,
    status: str,
    applied_changes: list[dict[str, Any]],
    blocked_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "tuning_apply",
        "status": status,
        "eval_status": "pass" if status == "applied" else "fail",
        "blocked_reason": blocked_reason,
        "package_dir": str(package_dir),
        "data_dir": str(data_dir),
        "proposal_artifact": str(package_dir / "tuning_proposal.json"),
        "decision_artifact": str(package_dir / "tuning_decision.json"),
        "decision": decision.get("decision"),
        "required_eval_gates": [report.get("suite") for report in eval_reports],
        "eval_gates": [
            {
                "suite": report.get("suite"),
                "status": report.get("status"),
                "counts": report.get("counts", {}),
                "error": report.get("error"),
            }
            for report in eval_reports
        ],
        "applied_changes": applied_changes,
        "auto_apply": False,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "proposal_summary": proposal.get("summary", ""),
    }


def _render_apply_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Feedback Tuning Apply",
        "",
        f"Status: {payload.get('status')}",
        f"Eval status: {payload.get('eval_status')}",
        f"Package: {payload.get('package_dir')}",
        "",
        "## Eval Gates",
        "",
    ]
    for gate in payload.get("eval_gates", []):
        if not isinstance(gate, dict):
            continue
        lines.append(f"- {gate.get('suite')}: {gate.get('status')}")
    if payload.get("status") != "applied":
        lines.extend(["", "Tuning was not applied because an eval gate failed."])
    else:
        lines.extend(["", "## Applied Changes", ""])
        changes = payload.get("applied_changes", [])
        if isinstance(changes, list) and changes:
            for change in changes:
                if not isinstance(change, dict):
                    continue
                lines.append(f"- {change.get('target')}: {change.get('action')} -> {change.get('value')} ({change.get('file')})")
        else:
            lines.append("- No concrete local rule changes were present in the proposal.")
    return "\n".join(lines).rstrip() + "\n"


def _load_required_json(path: Path, *, artifact_type: str) -> dict[str, Any]:
    value = _load_json(path)
    if not value:
        raise FeedbackApplyError(f"required artifact not found: {path}")
    if value.get("artifact_type") != artifact_type:
        raise FeedbackApplyError(f"invalid artifact type in {path}: {value.get('artifact_type')}")
    return value


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


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
