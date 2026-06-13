from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REVIEW_DECISIONS = (
    "approve",
    "reject",
    "needs_rewrite",
    "bad_evidence",
    "citation_issue",
    "too_confusing",
)


class DraftReviewError(ValueError):
    """Raised when a draft review packet cannot be built."""


def build_draft_feedback_files(
    package_dir: Path,
    *,
    decision: str | None = None,
    note: str | None = None,
    reviewer: str | None = None,
) -> dict[str, str]:
    if not package_dir.exists() or not package_dir.is_dir():
        raise DraftReviewError(f"package directory not found: {package_dir}")
    if decision is not None and decision not in REVIEW_DECISIONS:
        raise DraftReviewError(f"unsupported review decision: {decision}")

    trust_score = _load_json(package_dir / "report_trust_score.json")
    manifest = _load_json(package_dir / "report_manifest.json")
    review_items = _build_review_items(package_dir, trust_score)
    human_feedback = _human_feedback(decision=decision, note=note, reviewer=reviewer)
    feedback = {
        "schema_version": "1.0",
        "artifact_type": "draft_feedback",
        "package_dir": str(package_dir),
        "report_path": str(package_dir / "report.md") if (package_dir / "report.md").exists() else None,
        "trust_score": _first_present(trust_score.get("score"), manifest.get("trust_score")),
        "trust_verdict": _first_present(trust_score.get("verdict"), manifest.get("trust_verdict"), "unknown"),
        "trust_action": _first_present(trust_score.get("action"), manifest.get("trust_action"), "unknown"),
        "review_status": "captured" if human_feedback else "pending",
        "human_feedback": human_feedback,
        "review_items": review_items,
        "next_actions": _next_actions(review_items, human_feedback, trust_score),
    }
    return {
        "draft_feedback.json": _json_text(feedback),
        "review_queue.md": _render_review_queue(feedback),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unreadable",
            "issues": [
                {
                    "rule": "unreadable_artifact",
                    "detail": f"Could not read {path.name}.",
                }
            ],
        }
    return parsed if isinstance(parsed, dict) else {}


def _build_review_items(package_dir: Path, trust_score: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items: list[dict[str, Any]] = []
    for reason in _as_string_list(trust_score.get("reasons")):
        raw_items.append(
            {
                "source": "trust_score",
                "rule": reason,
                "severity": _severity_for_rule(reason),
                "prompt": _prompt_for_rule(reason),
            }
        )

    raw_items.extend(
        _issue_items(
            _load_json(package_dir / "report_prose_quality.json"),
            source="report_prose_quality",
            default_severity="important",
        )
    )
    raw_items.extend(
        _issue_items(
            _load_json(package_dir / "report_faithfulness_audit.json"),
            source="report_faithfulness_audit",
            default_severity="blocking",
        )
    )
    raw_items.extend(
        _issue_items(
            _load_json(package_dir / "report_critic_audit.json"),
            source="report_critic_audit",
            default_severity="important",
        )
    )
    raw_items.extend(
        _issue_items(
            _load_json(package_dir / "report_revision_audit.json"),
            source="report_revision_audit",
            default_severity="important",
        )
    )
    raw_items.extend(
        _issue_items(
            _load_json(package_dir / "report_revision_critic_audit.json"),
            source="report_revision_critic_audit",
            default_severity="important",
        )
    )
    raw_items.extend(_citation_items(_load_json(package_dir / "citation_audit.json")))

    if not raw_items:
        raw_items.append(
            {
                "source": "draft_review",
                "rule": "final_human_check",
                "severity": "low",
                "prompt": "Skim the report for readability, citation clarity, and unsupported claims before sharing.",
            }
        )
    return [{**item, "id": f"R{index}"} for index, item in enumerate(raw_items, start=1)]


def _issue_items(audit: dict[str, Any], *, source: str, default_severity: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    issues = audit.get("issues", [])
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            rule = str(issue.get("rule") or issue.get("reason") or "audit_issue")
            item = {
                "source": source,
                "rule": rule,
                "severity": _severity_for_rule(rule, default=default_severity),
                "prompt": _prompt_for_rule(rule),
            }
            for key in (
                "tier",
                "detail",
                "summary",
                "sentence",
                "citations",
                "unknown_citations",
                "missing_terms",
                "examples",
                "claim",
                "recommendation",
            ):
                if key in issue:
                    item[key] = issue[key]
            items.append(item)
    status = str(audit.get("status") or "")
    reason = str(audit.get("reason") or "")
    if status and status not in {"pass", "unknown"} and reason and not items:
        items.append(
            {
                "source": source,
                "rule": reason,
                "severity": _severity_for_rule(reason, default=default_severity),
                "prompt": _prompt_for_rule(reason),
                "status": status,
            }
        )
    return items


def _citation_items(audit: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    unknown = audit.get("unknown_citations", [])
    if isinstance(unknown, list):
        for citation in unknown:
            items.append(
                {
                    "source": "citation_audit",
                    "rule": "unknown_citation",
                    "severity": "blocking",
                    "prompt": "Check whether the citation marker points to real evidence before trusting this sentence.",
                    "citation": citation,
                }
            )
    items.extend(_issue_items(audit, source="citation_audit", default_severity="blocking"))
    return items


def _human_feedback(
    *,
    decision: str | None,
    note: str | None,
    reviewer: str | None,
) -> dict[str, str]:
    if decision is None and note is None and reviewer is None:
        return {}
    return {
        "decision": decision or "",
        "note": note or "",
        "reviewer": reviewer or "",
    }


def _next_actions(
    review_items: list[dict[str, Any]],
    human_feedback: dict[str, str],
    trust_score: dict[str, Any],
) -> list[str]:
    if human_feedback:
        decision = human_feedback.get("decision")
        if decision == "approve":
            return ["Record the approved package as a benchmark candidate."]
        if decision in {"reject", "bad_evidence", "citation_issue"}:
            return ["Do not publish this draft; fix evidence/citation issues and rerun the report gates."]
        if decision in {"needs_rewrite", "too_confusing"}:
            return ["Revise the draft for readability, then rerun prose and faithfulness gates."]
        return ["Use the captured feedback to decide whether to revise, block, or approve the draft."]

    action = str(trust_score.get("action") or "")
    if action == "block" or any(item.get("severity") == "blocking" for item in review_items):
        return ["Review blocking evidence/citation issues before sharing this report."]
    if action == "human_review":
        return ["Complete the review queue, then capture a decision with friday review-draft."]
    return ["Complete a final human skim and capture the decision if the draft will be reused."]


def _render_review_queue(feedback: dict[str, Any]) -> str:
    lines = [
        "# Draft Review Queue",
        "",
        f"Package: {feedback.get('package_dir')}",
        f"Trust score: {_display_value(feedback.get('trust_score'))}",
        f"Trust verdict: {_display_value(feedback.get('trust_verdict'))}",
        f"Trust action: {_display_value(feedback.get('trust_action'))}",
        f"Review status: {_display_value(feedback.get('review_status'))}",
    ]
    human_feedback = feedback.get("human_feedback")
    if isinstance(human_feedback, dict) and human_feedback:
        lines.extend(
            [
                "",
                f"Human decision: {_display_value(human_feedback.get('decision'))}",
                f"Reviewer: {_display_value(human_feedback.get('reviewer'))}",
                f"Note: {_display_value(human_feedback.get('note'))}",
            ]
        )
    lines.extend(["", "## Review Items", ""])
    review_items = feedback.get("review_items", [])
    if not isinstance(review_items, list) or not review_items:
        lines.append("- No review items.")
    else:
        for item in review_items:
            if not isinstance(item, dict):
                continue
            lines.append(f"### {item.get('id', '-')}: {item.get('rule', 'review_item')}")
            lines.append("")
            lines.append(f"- Source: {_display_value(item.get('source'))}")
            lines.append(f"- Severity: {_display_value(item.get('severity'))}")
            prompt = str(item.get("prompt") or "").strip()
            if prompt:
                lines.append(f"- Review prompt: {prompt}")
            for key in ("tier", "detail", "summary", "sentence", "citations", "missing_terms", "examples"):
                if key in item:
                    lines.append(f"- {key.replace('_', ' ').title()}: {_display_value(item.get(key))}")
            lines.append("")
    next_actions = feedback.get("next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.extend(["## Next Actions", ""])
        for action in next_actions:
            lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def _severity_for_rule(rule: str, *, default: str = "important") -> str:
    if rule in {
        "citation_audit_failed",
        "faithfulness_failed",
        "tier_a_failed",
        "unknown_citation",
        "uncited_factual_sentence",
    }:
        return "blocking"
    if rule in {"tier_b_failed", "weak_evidence_overlap", "prose_quality_failed"}:
        return "important"
    if rule in {"critic_not_run", "critic_not_passed", "critic_rejected", "critic_unavailable"}:
        return "important"
    return default


def _prompt_for_rule(rule: str) -> str:
    prompts = {
        "critic_not_run": "No independent critic approved this report; review readability and faithfulness before publishing.",
        "critic_not_passed": "The critic did not approve this report; inspect the critic findings before reuse.",
        "critic_rejected": "The critic rejected the draft; decide whether the issue is evidence, prose, or both.",
        "citation_audit_failed": "Citation checks failed; verify every factual sentence has a valid paper/page marker.",
        "unknown_citation": "A citation marker was not found in the evidence map; repair or remove the sentence.",
        "faithfulness_failed": "The report contains claims that may not be supported by cited evidence.",
        "tier_a_failed": "A hard citation or material-gap rule failed; do not publish without repair.",
        "tier_b_failed": "A cited sentence has weak overlap with its evidence; verify the claim manually.",
        "weak_evidence_overlap": "Check whether this sentence is actually supported by the cited evidence.",
        "prose_quality_failed": "The report failed readability/prose-quality checks; rewrite before sharing.",
        "raw_evidence_dump_phrase": "Rewrite table-like evidence fragments into reader-facing synthesis.",
        "internal_citation_syntax": "Replace internal P/page syntax with reader-facing citations.",
        "source_author_voice": "Rewrite extracted paper voice into neutral report prose.",
        "awkward_report_phrase": "Revise the sentence so it reads naturally.",
    }
    return prompts.get(rule, "Inspect this item and decide whether the draft should be approved, revised, or blocked.")


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return "-"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
