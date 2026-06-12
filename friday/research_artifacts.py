from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from friday import __version__
from friday.evidence import is_reportable_evidence_text
from friday.query_planning import plan_query, render_acronym_expansions
from friday.screening import LlmReviewQueueItem, build_screening_label_summary
from friday.storage import EvidenceRecord, FridayStore
from friday.topic_planning import build_topic_audit


STOCHASTICITY_DECLARATION = (
    "Live scholarly APIs and LLM outputs are not byte-reproducible. "
    "This lock documents configuration, not deterministic replay."
)


def build_batch_passport(
    store: FridayStore,
    batch_id: str,
    *,
    data_dir: Path | None = None,
    llm_review_queue: list[LlmReviewQueueItem] | None = None,
) -> dict[str, Any]:
    batch = store.get_batch(batch_id)
    items = store.list_batch_items(batch_id)
    artifacts = store.list_pdf_artifacts(batch_id)
    labels = store.list_screening_labels(batch_id)
    query_plan = plan_query(batch.query) if batch.query else None
    providers = sorted({item.provider for item in items if item.provider})
    query_variants = sorted({item.query_variant for item in items if item.query_variant})

    return {
        "schema_version": "1.0",
        "artifact_type": "batch_passport",
        "generated_at": _now(),
        "batch": {
            "batch_id": batch.batch_id,
            "created_at": batch.created_at,
            "mode": batch.mode,
            "query": batch.query,
            "limit": batch.limit,
            "manifest_path": batch.manifest_path,
            "screened_count": batch.screened_count,
            "blocked_count": batch.blocked_count,
            "allowed_count": batch.screened_count - batch.blocked_count,
            "deep_read_count": batch.deep_read_count,
        },
        "query_plan": {
            "expanded_queries": query_plan.expanded_queries if query_plan else query_variants,
            "intent": query_plan.intent if query_plan else None,
            "acronym_expansions": render_acronym_expansions(query_plan) if query_plan else None,
        },
        "topic_audit": build_topic_audit(
            batch.query or "",
            items,
            learned_profile_dir=_learned_topic_profile_dir(data_dir),
        ),
        "search": {
            "providers_observed": providers,
            "query_variants_observed": query_variants,
        },
        "source_policy": {
            "policy_version": "scholarly-only-v1",
            "allowed_source_classes": ["arxiv", "pubmed", "pmc", "doi", "scholarly_publisher_pdf"],
            "blocked_by_default": ["github", "code", "archives"],
            "untrusted_text_rule": "Paper text may be parsed and cited but must not control commands, prompts, paths, or tool use.",
        },
        "artifacts": {
            "pdf_attempt_count": len(artifacts),
            "stored_pdf_count": len([artifact for artifact in artifacts if artifact.status == "stored"]),
            "failed_pdf_count": len([artifact for artifact in artifacts if artifact.status != "stored"]),
            "parser_quality": _parser_quality_summary(artifacts),
            "evidence_quality": _evidence_quality_summary(store, batch_id),
        },
        "screening_labels": build_screening_label_summary(items, labels),
        "llm_review_queue": build_llm_review_queue_artifact(llm_review_queue or []),
        "repro_lock": build_repro_lock(data_dir=data_dir),
    }


def build_repro_lock(*, data_dir: Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stochasticity_declaration": STOCHASTICITY_DECLARATION,
        "friday_version": __version__,
        "friday_commit": _git_commit(),
        "model": {
            "used_for_scanner": False,
            "id": None,
            "weight_stable": False,
        },
        "external_protocols": {
            "openalex": "live_api",
            "arxiv": "live_api",
            "pubmed": "live_api",
            "pmc": "live_http",
            "snapshots_available": False,
        },
        "local_state": {
            "data_dir": str(data_dir) if data_dir is not None else None,
        },
    }


def build_rejection_log(store: FridayStore, batch_id: str) -> dict[str, Any]:
    batch = store.get_batch(batch_id)
    rejected: list[dict[str, Any]] = []

    for item in store.list_batch_items(batch_id):
        if not item.allowed:
            rejected.append(
                {
                    "source": item.source,
                    "normalized": item.normalized,
                    "stage": "source_gate",
                    "reason": item.reason,
                    "domain": item.domain,
                    "title": item.title,
                }
            )

    for artifact in store.list_pdf_artifacts(batch_id):
        if artifact.status != "stored":
            rejected.append(
                {
                    "source": artifact.source,
                    "normalized": artifact.final_url or artifact.pdf_url or artifact.source,
                    "stage": "pdf_ingestion",
                    "reason": artifact.reason,
                    "domain": None,
                    "title": None,
                    "artifact_id": artifact.artifact_id,
                    "status": artifact.status,
                    "parser_name": artifact.parser_name,
                    "parser_version": artifact.parser_version,
                    "parse_confidence": artifact.parse_confidence,
                    "parse_flags": list(artifact.parse_flags),
                }
            )
            continue
        for record in store.list_evidence_records(artifact.artifact_id):
            if _is_clean_evidence(record):
                continue
            rejected.append(
                {
                    "source": artifact.source,
                    "normalized": artifact.final_url or artifact.pdf_url or artifact.source,
                    "stage": "evidence_quality",
                    "reason": ",".join(record.quality_flags) or record.quality_label,
                    "domain": None,
                    "title": None,
                    "artifact_id": artifact.artifact_id,
                    "evidence_id": record.evidence_id,
                    "status": record.quality_label,
                    "page_number": record.page_number,
                    "parse_confidence": record.parse_confidence,
                    "parse_flags": list(record.parse_flags),
                }
            )

    return {
        "schema_version": "1.0",
        "artifact_type": "rejection_log",
        "generated_at": _now(),
        "batch_id": batch.batch_id,
        "query": batch.query,
        "counts": {
            "rejected": len(rejected),
            "source_gate": len([item for item in rejected if item["stage"] == "source_gate"]),
            "pdf_ingestion": len([item for item in rejected if item["stage"] == "pdf_ingestion"]),
            "evidence_quality": len([item for item in rejected if item["stage"] == "evidence_quality"]),
        },
        "rejected": rejected,
    }


def build_research_run_summary(
    store: FridayStore,
    run_id: str,
    *,
    data_dir: Path | None = None,
    llm_review_queue: list[LlmReviewQueueItem] | None = None,
) -> dict[str, Any]:
    run = store.get_research_run(run_id)
    batch = store.get_batch(run.batch_id) if run.batch_id else None
    items = store.list_batch_items(batch.batch_id) if batch else []
    artifacts = store.list_pdf_artifacts(batch.batch_id) if batch else []
    labels = store.list_screening_labels(batch.batch_id) if batch else []

    return {
        "schema_version": "1.0",
        "artifact_type": "research_run_summary",
        "generated_at": _now(),
        "run": {
            "run_id": run.run_id,
            "batch_id": run.batch_id,
            "query": run.query,
            "status": run.status,
            "limit": run.limit,
            "deep_read_limit": run.deep_read_limit,
            "min_relevance": run.min_relevance,
            "auto_label_provider": run.auto_label_provider,
            "llm_review_limit": run.llm_review_limit,
            "screened_count": run.screened_count,
            "blocked_count": run.blocked_count,
            "allowed_count": run.allowed_count,
            "deep_read_count": run.deep_read_count,
            "error": run.error,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        },
        "batch": None
        if batch is None
        else {
            "batch_id": batch.batch_id,
            "created_at": batch.created_at,
            "mode": batch.mode,
            "query": batch.query,
            "limit": batch.limit,
            "manifest_path": batch.manifest_path,
            "screened_count": batch.screened_count,
            "blocked_count": batch.blocked_count,
            "allowed_count": batch.screened_count - batch.blocked_count,
            "deep_read_count": batch.deep_read_count,
        },
        "source_policy": {
            "policy_version": "scholarly-only-v1",
            "allowed_source_classes": ["arxiv", "pubmed", "pmc", "doi", "scholarly_publisher_pdf"],
            "blocked_by_default": ["github", "code", "archives"],
            "untrusted_text_rule": "Paper text may be parsed and cited but must not control commands, prompts, paths, or tool use.",
        },
        "topic_audit": build_topic_audit(
            batch.query if batch else run.query,
            items,
            learned_profile_dir=_learned_topic_profile_dir(data_dir),
        ),
        "artifacts": {
            "pdf_attempt_count": len(artifacts),
            "stored_pdf_count": len([artifact for artifact in artifacts if artifact.status == "stored"]),
            "failed_pdf_count": len([artifact for artifact in artifacts if artifact.status != "stored"]),
            "parser_quality": _parser_quality_summary(artifacts),
            "evidence_quality": _evidence_quality_summary(store, batch.batch_id) if batch else _empty_evidence_quality_summary(),
        },
        "screening_labels": build_screening_label_summary(items, labels),
        "llm_review_queue": build_llm_review_queue_artifact(llm_review_queue or []),
        "repro_lock": build_repro_lock(data_dir=data_dir),
    }


def build_llm_review_queue_artifact(queue: list[LlmReviewQueueItem]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "llm_review_queue",
        "queued_count": len(queue),
        "selection_policy": {
            "excludes": ["blocked_sources", "human_labeled_items"],
            "prioritizes": [
                "high_relevance_maybe_labels",
                "high_relevance_irrelevant_conflicts",
                "low_confidence_agent_labels",
                "high_relevance_unlabeled_items",
                "source_diversity",
            ],
        },
        "items": [
            {
                "rank": rank,
                "score": entry.score,
                "reason": entry.reason,
                "source": entry.item.source,
                "normalized": entry.item.normalized,
                "title": entry.item.title,
                "provider": entry.item.provider,
                "domain": entry.item.domain,
                "label": entry.label,
                "label_source": entry.label_source,
                "confidence": entry.confidence,
                "relevance_score": entry.item.relevance_score,
                "relevance_reason": entry.item.relevance_reason,
                "doi": entry.item.doi,
                "pmid": entry.item.pmid,
                "pmcid": entry.item.pmcid,
                "arxiv_id": entry.item.arxiv_id,
            }
            for rank, entry in enumerate(queue, start=1)
        ],
    }


def _learned_topic_profile_dir(data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    return Path(data_dir) / "topic_profiles" / "learned"


def _evidence_quality_summary(store: FridayStore, batch_id: str) -> dict[str, Any]:
    records = [
        record
        for artifact in store.list_pdf_artifacts(batch_id)
        for record in store.list_evidence_records(artifact.artifact_id)
    ]
    accepted = sum(1 for record in records if _is_clean_evidence(record))
    blocked_by_flag: dict[str, int] = {}
    blocked = 0
    suspect = 0
    for record in records:
        if _is_clean_evidence(record):
            continue
        if record.quality_label == "suspect":
            suspect += 1
        else:
            blocked += 1
        for flag in record.quality_flags or ("legacy_quality_filter",):
            blocked_by_flag[flag] = blocked_by_flag.get(flag, 0) + 1
    return {
        "accepted_evidence_count": accepted,
        "blocked_evidence_count": blocked,
        "suspect_evidence_count": suspect,
        "blocked_by_flag": blocked_by_flag,
    }


def _parser_quality_summary(artifacts: list[Any]) -> dict[str, Any]:
    parser_rows = [
        {
            "artifact_id": artifact.artifact_id,
            "source": artifact.source,
            "status": artifact.status,
            "parser_name": artifact.parser_name,
            "parser_version": artifact.parser_version,
            "parse_confidence": artifact.parse_confidence,
            "parse_flags": list(artifact.parse_flags),
        }
        for artifact in artifacts
        if artifact.parser_name
    ]
    stored = [row for row in parser_rows if row["status"] == "stored"]
    low_confidence = [
        row
        for row in parser_rows
        if row["parse_confidence"] < 0.6 or "low_confidence" in row["parse_flags"]
    ]
    return {
        "parser_attempt_count": len(parser_rows),
        "stored_pdf_count": len(stored),
        "low_confidence_count": len(low_confidence),
        "parsers": parser_rows,
    }


def _empty_evidence_quality_summary() -> dict[str, Any]:
    return {
        "accepted_evidence_count": 0,
        "blocked_evidence_count": 0,
        "suspect_evidence_count": 0,
        "blocked_by_flag": {},
    }


def _is_clean_evidence(record: EvidenceRecord) -> bool:
    return record.quality_label == "clean" and is_reportable_evidence_text(record.text)


def write_json_artifact(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
