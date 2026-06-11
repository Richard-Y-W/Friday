from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from friday.label_eval import build_label_evaluation
from friday.research_artifacts import build_rejection_log
from friday.screening import build_llm_review_queue
from friday.storage import (
    BatchItemRecord,
    BatchRecord,
    FridayStore,
    PdfArtifactRecord,
    ResearchRunRecord,
    ScreeningLabelRecord,
)


HIGH_RELEVANCE_UNLABELED_THRESHOLD = 60


class RunSummaryTargetError(RuntimeError):
    pass


def build_run_summary_dashboard(
    store: FridayStore,
    *,
    latest: bool = False,
    run_id: str | None = None,
    batch_id: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    run, batch, target_type = _resolve_target(store, latest=latest, run_id=run_id, batch_id=batch_id)
    items = store.list_batch_items(batch.batch_id) if batch else []
    labels = store.list_screening_labels(batch.batch_id) if batch else []
    artifacts = store.list_pdf_artifacts(batch.batch_id) if batch else []
    label_evaluation = build_label_evaluation(items, labels) if batch else _empty_label_evaluation()
    rejection_log = build_rejection_log(store, batch.batch_id) if batch else _empty_rejection_log()
    review_queue = build_llm_review_queue(items, labels, limit=max(0, limit)) if batch else []

    summary = {
        "schema_version": "1.0",
        "artifact_type": "run_summary_dashboard",
        "generated_at": _now(),
        "target": _target(run, batch, target_type),
        "counts": _counts(store, items, labels, artifacts),
        "label_evaluation": {
            "comparable_count": label_evaluation["comparable_count"],
            "correct_count": label_evaluation["correct_count"],
            "accuracy": label_evaluation["accuracy"],
            "human_label_counts": label_evaluation["human_label_counts"],
            "precision": label_evaluation["precision"],
            "recall": label_evaluation["recall"],
        },
        "review_queue": _review_queue_rows(review_queue),
        "attention": _attention(items, labels, artifacts, label_evaluation, rejection_log, limit=max(0, limit)),
    }
    summary["next_commands"] = _next_commands(summary)
    return summary


def render_run_summary_text(summary: dict[str, Any]) -> str:
    target = summary["target"]
    counts = summary["counts"]
    label_eval = summary["label_evaluation"]
    attention = summary["attention"]
    lines = [
        "Friday Run Summary",
        "",
        f"Target: {target['target_type']}",
        f"Run: {target['run_id'] or '-'}",
        f"Batch: {target['batch_id'] or '-'}",
        f"Query: {target['query'] or '-'}",
        f"Status: {target['status'] or '-'}",
        "",
        "Counts:",
        (
            f"screened={counts['screened']} blocked={counts['blocked']} allowed={counts['allowed']} "
            f"labeled={counts['labeled']} human={counts['human_labels']} agent={counts['agent_labels']} "
            f"unlabeled_allowed={counts['unlabeled_allowed']} stored_pdfs={counts['stored_pdfs']} "
            f"failed_pdfs={counts['failed_pdfs']} evidence={counts['evidence_items']}"
        ),
        "",
        "Label evaluation:",
        f"comparable={label_eval['comparable_count']} accuracy={label_eval['accuracy']:.3f}",
        "",
        "Attention:",
    ]
    _append_attention(lines, "Label disagreements", attention["label_disagreements"], "human_label", "agent_label")
    _append_attention(lines, "High-confidence mistakes", attention["high_confidence_mistakes"], "human_label", "agent_label")
    _append_attention(lines, "Maybe labels", attention["maybe_labels"], "label", "confidence")
    _append_attention(lines, "High-relevance unlabeled", attention["high_relevance_unlabeled"], "relevance_score", "provider")
    _append_attention(lines, "Failed PDFs", attention["failed_pdfs"], "status", "reason")
    _append_attention(lines, "Source-gate blocks", attention["source_gate_blocks"], "stage", "reason")
    lines.append(f"- Smart review queue: {summary['review_queue']['queued_count']}")
    lines.extend(["", "Next commands:"])
    if summary["next_commands"]:
        for command in summary["next_commands"]:
            lines.append(f"- {command['command']} ({command['reason']})")
    else:
        lines.append("- No immediate next command recommended.")
    return "\n".join(lines)


def _resolve_target(
    store: FridayStore,
    *,
    latest: bool,
    run_id: str | None,
    batch_id: str | None,
) -> tuple[ResearchRunRecord | None, BatchRecord | None, str]:
    if run_id:
        run = store.get_research_run(run_id)
        batch = store.get_batch(run.batch_id) if run.batch_id else None
        return run, batch, "research_run"
    if batch_id:
        return None, store.get_batch(batch_id), "batch"
    if latest:
        run = store.latest_research_run()
        if run is not None:
            batch = store.get_batch(run.batch_id) if run.batch_id else None
            return run, batch, "research_run"
        batch = store.latest_batch()
        if batch is not None:
            return None, batch, "batch"
        raise RunSummaryTargetError("No research runs or batches found.")
    raise RunSummaryTargetError("run-summary requires --latest, --run-id, or --batch-id")


def _target(
    run: ResearchRunRecord | None,
    batch: BatchRecord | None,
    target_type: str,
) -> dict[str, Any]:
    return {
        "target_type": target_type,
        "run_id": run.run_id if run else None,
        "batch_id": batch.batch_id if batch else run.batch_id if run else None,
        "query": run.query if run else batch.query if batch else None,
        "status": run.status if run else None,
        "created_at": run.created_at if run else batch.created_at if batch else None,
        "updated_at": run.updated_at if run else None,
    }


def _counts(
    store: FridayStore,
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    artifacts: list[PdfArtifactRecord],
) -> dict[str, int]:
    allowed_items = [item for item in items if item.allowed]
    labels_by_normalized = {label.normalized: label for label in labels}
    evidence_count = sum(len(store.list_evidence_records(artifact.artifact_id)) for artifact in artifacts)
    return {
        "screened": len(items),
        "blocked": len([item for item in items if not item.allowed]),
        "allowed": len(allowed_items),
        "labeled": len(labels),
        "human_labels": len([label for label in labels if label.label_source == "human"]),
        "agent_labels": len([label for label in labels if label.label_source == "agent"]),
        "unlabeled_allowed": len([item for item in allowed_items if item.normalized not in labels_by_normalized]),
        "stored_pdfs": len([artifact for artifact in artifacts if artifact.status == "stored"]),
        "failed_pdfs": len([artifact for artifact in artifacts if artifact.status != "stored"]),
        "evidence_items": evidence_count,
    }


def _attention(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    artifacts: list[PdfArtifactRecord],
    label_evaluation: dict[str, Any],
    rejection_log: dict[str, Any],
    *,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    labels_by_normalized = {label.normalized: label for label in labels}
    maybe_labels = [
        _label_attention_row(label, _item_by_normalized(items).get(label.normalized))
        for label in labels
        if label.label == "maybe"
    ]
    high_relevance_unlabeled = [
        _item_attention_row(item)
        for item in sorted(
            items,
            key=lambda item: (-(item.relevance_score or 0), item.source),
        )
        if item.allowed
        and item.normalized not in labels_by_normalized
        and (item.relevance_score or 0) >= HIGH_RELEVANCE_UNLABELED_THRESHOLD
    ]
    failed_pdfs = [
        {
            "source": artifact.source,
            "artifact_id": artifact.artifact_id,
            "status": artifact.status,
            "reason": artifact.reason,
            "pdf_url": artifact.pdf_url,
            "final_url": artifact.final_url,
        }
        for artifact in artifacts
        if artifact.status != "stored"
    ]
    source_gate_blocks = [
        rejection
        for rejection in rejection_log.get("rejected", [])
        if rejection.get("stage") == "source_gate"
    ]
    return {
        "label_disagreements": label_evaluation["disagreements"][:limit],
        "high_confidence_mistakes": label_evaluation["high_confidence_mistakes"][:limit],
        "maybe_labels": maybe_labels[:limit],
        "high_relevance_unlabeled": high_relevance_unlabeled[:limit],
        "failed_pdfs": failed_pdfs[:limit],
        "source_gate_blocks": source_gate_blocks[:limit],
    }


def _next_commands(summary: dict[str, Any]) -> list[dict[str, str]]:
    target = summary["target"]
    attention = summary["attention"]
    commands: list[dict[str, str]] = []
    if attention["maybe_labels"]:
        commands.append(
            {
                "command": "friday labels review --latest --only maybe",
                "reason": "review maybe labels first",
            }
        )
    if attention["high_relevance_unlabeled"]:
        commands.append(
            {
                "command": "friday labels review --latest --only unlabeled --min-relevance 60",
                "reason": "triage high-relevance unlabeled papers",
            }
        )
    if summary["counts"]["human_labels"]:
        commands.append({"command": "friday labels eval --latest", "reason": "inspect agent-vs-human label quality"})
    if target["batch_id"]:
        commands.append({"command": f"friday report {target['batch_id']}", "reason": "open the cited batch report"})
    if target["run_id"]:
        commands.append(
            {
                "command": f"friday research-run --resume-run {target['run_id']}",
                "reason": "continue screening or deep reading this run",
            }
        )
    return commands


def _review_queue_rows(review_queue: list[Any]) -> dict[str, Any]:
    return {
        "queued_count": len(review_queue),
        "items": [
            {
                "rank": rank,
                "score": entry.score,
                "reason": entry.reason,
                "source": entry.item.source,
                "title": entry.item.title,
                "label": entry.label,
                "label_source": entry.label_source,
                "confidence": entry.confidence,
                "relevance_score": entry.item.relevance_score,
            }
            for rank, entry in enumerate(review_queue, start=1)
        ],
    }


def _append_attention(
    lines: list[str],
    label: str,
    rows: list[dict[str, Any]],
    first_field: str,
    second_field: str,
) -> None:
    if not rows:
        lines.append(f"- {label}: none")
        return
    lines.append(f"- {label}: {len(rows)}")
    for row in rows[:3]:
        title = row.get("title") or row.get("source")
        first = row.get(first_field)
        second = row.get(second_field)
        lines.append(f"  - {row.get('source')} | {title} | {first_field}={first} {second_field}={second}")


def _label_attention_row(
    label: ScreeningLabelRecord,
    item: BatchItemRecord | None,
) -> dict[str, Any]:
    return {
        "source": label.source,
        "normalized": label.normalized,
        "title": item.title if item else None,
        "label": label.label,
        "label_source": label.label_source,
        "confidence": label.confidence,
        "relevance_score": item.relevance_score if item else None,
        "provider": item.provider if item else None,
    }


def _item_attention_row(item: BatchItemRecord) -> dict[str, Any]:
    return {
        "source": item.source,
        "normalized": item.normalized,
        "title": item.title,
        "provider": item.provider,
        "relevance_score": item.relevance_score or 0,
        "relevance_reason": item.relevance_reason,
        "doi": item.doi,
        "pmid": item.pmid,
        "pmcid": item.pmcid,
        "arxiv_id": item.arxiv_id,
    }


def _item_by_normalized(items: list[BatchItemRecord]) -> dict[str, BatchItemRecord]:
    return {item.normalized: item for item in items}


def _empty_label_evaluation() -> dict[str, Any]:
    return {
        "comparable_count": 0,
        "correct_count": 0,
        "accuracy": 0.0,
        "human_label_counts": {"relevant": 0, "maybe": 0, "irrelevant": 0},
        "precision": {"relevant": 0.0, "maybe": 0.0, "irrelevant": 0.0},
        "recall": {"relevant": 0.0, "maybe": 0.0, "irrelevant": 0.0},
        "disagreements": [],
        "high_confidence_mistakes": [],
    }


def _empty_rejection_log() -> dict[str, Any]:
    return {"rejected": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
