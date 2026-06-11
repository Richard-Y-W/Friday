from __future__ import annotations

from typing import Any

from friday.screening import build_llm_review_queue
from friday.storage import BatchItemRecord, ScreeningLabelRecord


LABEL_REVIEW_FILTERS = ("relevant", "maybe", "irrelevant", "human", "agent", "unlabeled")


def build_label_review_rows(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    *,
    only: str | None = None,
    min_relevance: int = 0,
    limit: int = 20,
) -> list[dict[str, Any]]:
    labels_by_normalized = {label.normalized: label for label in labels}
    queue_by_normalized = {
        entry.item.normalized: (rank, entry.score, entry.reason)
        for rank, entry in enumerate(
            build_llm_review_queue(items, labels, limit=len(items)),
            start=1,
        )
    }
    rows = []
    for item in items:
        if not item.allowed:
            continue
        relevance = item.relevance_score or 0
        if relevance < min_relevance:
            continue
        label = labels_by_normalized.get(item.normalized)
        row = _review_row(item, label, queue_by_normalized.get(item.normalized))
        if _matches_filter(row, only):
            rows.append(row)
    rows.sort(key=_review_sort_key)
    return rows[: max(0, limit)]


def _review_row(
    item: BatchItemRecord,
    label: ScreeningLabelRecord | None,
    queue_entry: tuple[int, int, str] | None,
) -> dict[str, Any]:
    queue_rank = queue_score = queue_reason = None
    if queue_entry is not None:
        queue_rank, queue_score, queue_reason = queue_entry
    return {
        "source": item.source,
        "normalized": item.normalized,
        "title": item.title or item.source,
        "provider": item.provider,
        "allowed": item.allowed,
        "source_reason": item.reason,
        "label": label.label if label else None,
        "label_source": label.label_source if label else "unlabeled",
        "confidence": label.confidence if label else None,
        "note": label.note if label else None,
        "rationale": label.rationale if label else None,
        "signals": label.signals if label else None,
        "relevance_score": item.relevance_score or 0,
        "relevance_reason": item.relevance_reason,
        "review_queue_rank": queue_rank,
        "review_queue_score": queue_score,
        "review_queue_reason": queue_reason,
        "doi": item.doi,
        "pmid": item.pmid,
        "pmcid": item.pmcid,
        "arxiv_id": item.arxiv_id,
    }


def _matches_filter(row: dict[str, Any], only: str | None) -> bool:
    if only is None:
        return True
    normalized = only.strip().lower()
    if normalized in {"human", "agent", "unlabeled"}:
        return row["label_source"] == normalized
    return row["label"] == normalized


def _review_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str, str]:
    queue_rank = row["review_queue_rank"]
    queue_bucket = 0 if queue_rank is not None else 1
    source_bucket = {"agent": 0, "unlabeled": 1, "human": 2}.get(row["label_source"], 3)
    return (
        queue_bucket,
        queue_rank if queue_rank is not None else 999_999,
        source_bucket,
        str(row["title"]).lower(),
        str(row["source"]),
    )
