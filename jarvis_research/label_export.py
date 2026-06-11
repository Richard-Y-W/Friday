from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

from jarvis_research.screening import build_llm_review_queue
from jarvis_research.storage import (
    BatchItemRecord,
    BatchRecord,
    JarvisStore,
    ScreeningLabelRecord,
)


LABEL_EXPORT_FIELDS = [
    "schema_version",
    "batch_id",
    "batch_mode",
    "query",
    "batch_limit",
    "batch_created_at",
    "source",
    "normalized",
    "allowed",
    "source_reason",
    "source_domain",
    "provider",
    "title",
    "abstract",
    "doi",
    "pmid",
    "pmcid",
    "arxiv_id",
    "year",
    "url",
    "journal",
    "concepts",
    "mesh_terms",
    "oa_status",
    "open_access_pdf_url",
    "relevance_score",
    "relevance_reason",
    "query_variant",
    "query_intent",
    "acronym_expansions",
    "label",
    "label_source",
    "gold_label",
    "weak_label",
    "label_confidence",
    "label_rationale",
    "label_signals",
    "label_note",
    "label_created_at",
    "label_updated_at",
    "review_queue_rank",
    "review_queue_score",
    "review_queue_reason",
]


def build_label_export_rows(
    store: JarvisStore,
    *,
    batch_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    batches = _selected_batches(store, batch_ids)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        items = store.list_batch_items(batch.batch_id)
        labels = store.list_screening_labels(batch.batch_id)
        items_by_normalized = {item.normalized: item for item in items}
        queue_by_normalized = _review_queue_by_normalized(items, labels)
        for label in labels:
            item = items_by_normalized.get(label.normalized)
            if item is None:
                continue
            rows.append(_export_row(batch, item, label, queue_by_normalized.get(item.normalized)))
    return rows


def render_label_export_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows)


def render_label_export_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=LABEL_EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().rstrip("\r\n")


def _selected_batches(store: JarvisStore, batch_ids: Iterable[str] | None) -> list[BatchRecord]:
    if batch_ids is None:
        return store.list_batches()
    return [store.get_batch(batch_id) for batch_id in batch_ids]


def _review_queue_by_normalized(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
) -> dict[str, tuple[int, int, str]]:
    queue = build_llm_review_queue(items, labels, limit=len(items))
    return {
        entry.item.normalized: (rank, entry.score, entry.reason)
        for rank, entry in enumerate(queue, start=1)
    }


def _export_row(
    batch: BatchRecord,
    item: BatchItemRecord,
    label: ScreeningLabelRecord,
    queue_entry: tuple[int, int, str] | None,
) -> dict[str, Any]:
    review_rank = review_score = review_reason = None
    if queue_entry is not None:
        review_rank, review_score, review_reason = queue_entry
    gold_label = label.label if label.label_source == "human" else None
    weak_label = label.label if label.label_source == "agent" else None
    return {
        "schema_version": "1.0",
        "batch_id": batch.batch_id,
        "batch_mode": batch.mode,
        "query": batch.query,
        "batch_limit": batch.limit,
        "batch_created_at": batch.created_at,
        "source": item.source,
        "normalized": item.normalized,
        "allowed": item.allowed,
        "source_reason": item.reason,
        "source_domain": item.domain,
        "provider": item.provider,
        "title": item.title,
        "abstract": item.abstract,
        "doi": item.doi,
        "pmid": item.pmid,
        "pmcid": item.pmcid,
        "arxiv_id": item.arxiv_id,
        "year": item.year,
        "url": item.url,
        "journal": item.journal,
        "concepts": item.concepts,
        "mesh_terms": item.mesh_terms,
        "oa_status": item.oa_status,
        "open_access_pdf_url": item.open_access_pdf_url,
        "relevance_score": item.relevance_score,
        "relevance_reason": item.relevance_reason,
        "query_variant": item.query_variant,
        "query_intent": item.query_intent,
        "acronym_expansions": item.acronym_expansions,
        "label": label.label,
        "label_source": label.label_source,
        "gold_label": gold_label,
        "weak_label": weak_label,
        "label_confidence": label.confidence,
        "label_rationale": label.rationale,
        "label_signals": label.signals,
        "label_note": label.note,
        "label_created_at": label.created_at,
        "label_updated_at": label.updated_at,
        "review_queue_rank": review_rank,
        "review_queue_score": review_score,
        "review_queue_reason": review_reason,
    }
