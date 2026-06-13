from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import secrets
from typing import Iterator

from friday.discovery import Candidate
from friday.evidence import EvidenceItem, assess_evidence_trust
from friday.source_policy import SourceDecision, evaluate_source


SCREENING_LABEL_CHOICES = ("relevant", "maybe", "irrelevant")
SCREENING_LABEL_SOURCE_CHOICES = ("human", "agent")


@dataclass(frozen=True)
class ScanRecord:
    scan_id: str
    source: str
    normalized: str
    kind: str
    allowed: bool
    reason: str
    domain: str | None
    created_at: str


@dataclass(frozen=True)
class BatchRecord:
    batch_id: str
    query: str | None
    limit: int | None
    mode: str
    manifest_path: str | None
    screened_count: int
    blocked_count: int
    deep_read_count: int
    created_at: str


@dataclass(frozen=True)
class ResearchRunRecord:
    run_id: str
    batch_id: str | None
    query: str
    status: str
    limit: int
    deep_read_limit: int
    min_relevance: int
    auto_label_provider: str
    llm_review_limit: int
    screened_count: int
    blocked_count: int
    allowed_count: int
    deep_read_count: int
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class BatchItemRecord:
    batch_id: str
    source: str
    normalized: str
    allowed: bool
    reason: str
    domain: str | None
    provider: str | None
    title: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    arxiv_id: str | None
    year: int | None
    url: str | None
    abstract: str | None
    relevance_score: int | None
    relevance_reason: str | None
    query_variant: str | None
    query_intent: str | None
    acronym_expansions: str | None
    journal: str | None
    concepts: str | None
    mesh_terms: str | None
    oa_status: str | None
    open_access_pdf_url: str | None
    created_at: str


@dataclass(frozen=True)
class PdfArtifactRecord:
    artifact_id: str
    batch_id: str
    source: str
    pdf_url: str | None
    final_url: str | None
    sha256: str | None
    byte_count: int | None
    content_type: str | None
    local_path: str | None
    status: str
    reason: str
    parser_name: str | None
    parser_version: str | None
    parse_confidence: float
    parse_flags: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class PdfPageRecord:
    artifact_id: str
    page_number: int
    text: str
    char_count: int
    parse_confidence: float
    parse_flags: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    artifact_id: str
    evidence_type: str
    page_number: int
    text: str
    char_count: int
    quality_label: str
    quality_score: float
    quality_flags: tuple[str, ...]
    parse_confidence: float
    parse_flags: tuple[str, ...]
    trust_label: str
    trust_score: float
    trust_reasons: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class ReportClaimUnitRecord:
    report_package_path: str
    claim_unit_id: str
    source_batch_id: str | None
    source_query: str | None
    section: str
    claim_type: str
    text: str
    source_sentence: str
    citations: tuple[str, ...]
    support_status: str
    evidence_count: int
    evidence_types: tuple[str, ...]
    evidence_row_ids: tuple[str, ...]
    min_quality_score: float | None
    min_parse_confidence: float | None
    min_trust_score: float | None
    support_details: dict[str, object]
    created_at: str


@dataclass(frozen=True)
class ScreeningLabelRecord:
    batch_id: str
    source: str
    normalized: str
    label: str
    note: str | None
    label_source: str
    confidence: float | None
    rationale: str | None
    signals: str | None
    created_at: str
    updated_at: str


class FridayStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_scan(self, source: str, decision: SourceDecision) -> ScanRecord:
        scan_id = _make_id("scan")
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into scans (
                    scan_id, source, normalized, kind, allowed, reason, domain, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    source,
                    decision.normalized,
                    decision.kind,
                    int(decision.allowed),
                    decision.reason,
                    decision.domain,
                    created_at,
                ),
            )
        return ScanRecord(
            scan_id=scan_id,
            source=source,
            normalized=decision.normalized,
            kind=decision.kind,
            allowed=decision.allowed,
            reason=decision.reason,
            domain=decision.domain,
            created_at=created_at,
        )

    def get_scan(self, scan_id: str) -> ScanRecord:
        with self._connect() as conn:
            row = conn.execute("select * from scans where scan_id = ?", (scan_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown scan id: {scan_id}")
        return _scan_from_row(row)

    def list_scans(self) -> list[ScanRecord]:
        with self._connect() as conn:
            rows = conn.execute("select * from scans order by created_at desc, scan_id desc").fetchall()
        return [_scan_from_row(row) for row in rows]

    def create_batch(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
        mode: str,
        manifest_path: str | None = None,
    ) -> BatchRecord:
        batch_id = _make_id("batch")
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into batches (
                    batch_id, query, limit_value, mode, manifest_path,
                    screened_count, blocked_count, deep_read_count, created_at
                ) values (?, ?, ?, ?, ?, 0, 0, 0, ?)
                """,
                (batch_id, query, limit, mode, manifest_path, created_at),
            )
        return BatchRecord(batch_id, query, limit, mode, manifest_path, 0, 0, 0, created_at)

    def create_research_run(
        self,
        *,
        query: str,
        limit: int,
        deep_read_limit: int,
        min_relevance: int,
        auto_label_provider: str,
        llm_review_limit: int,
        batch_id: str | None = None,
        status: str = "created",
    ) -> ResearchRunRecord:
        run_id = _make_id("run")
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into research_runs (
                    run_id, batch_id, query, status, limit_value, deep_read_limit,
                    min_relevance, auto_label_provider, llm_review_limit,
                    screened_count, blocked_count, allowed_count, deep_read_count,
                    error, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, null, ?, ?)
                """,
                (
                    run_id,
                    batch_id,
                    query,
                    status,
                    limit,
                    deep_read_limit,
                    min_relevance,
                    auto_label_provider,
                    llm_review_limit,
                    created_at,
                    created_at,
                ),
            )
        return ResearchRunRecord(
            run_id=run_id,
            batch_id=batch_id,
            query=query,
            status=status,
            limit=limit,
            deep_read_limit=deep_read_limit,
            min_relevance=min_relevance,
            auto_label_provider=auto_label_provider,
            llm_review_limit=llm_review_limit,
            screened_count=0,
            blocked_count=0,
            allowed_count=0,
            deep_read_count=0,
            error=None,
            created_at=created_at,
            updated_at=created_at,
        )

    def get_research_run(self, run_id: str) -> ResearchRunRecord:
        with self._connect() as conn:
            row = conn.execute("select * from research_runs where run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown research run id: {run_id}")
        return _research_run_from_row(row)

    def list_research_runs(self) -> list[ResearchRunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from research_runs order by created_at desc, rowid desc"
            ).fetchall()
        return [_research_run_from_row(row) for row in rows]

    def latest_research_run(self) -> ResearchRunRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from research_runs order by created_at desc, rowid desc limit 1"
            ).fetchone()
        return _research_run_from_row(row) if row is not None else None

    def update_research_run(self, run_id: str, **fields: object) -> ResearchRunRecord:
        allowed_fields = {
            "batch_id": "batch_id",
            "status": "status",
            "limit": "limit_value",
            "deep_read_limit": "deep_read_limit",
            "min_relevance": "min_relevance",
            "auto_label_provider": "auto_label_provider",
            "llm_review_limit": "llm_review_limit",
            "screened_count": "screened_count",
            "blocked_count": "blocked_count",
            "allowed_count": "allowed_count",
            "deep_read_count": "deep_read_count",
            "error": "error",
        }
        unknown = sorted(set(fields) - set(allowed_fields))
        if unknown:
            raise ValueError(f"unknown research run fields: {', '.join(unknown)}")
        if not fields:
            return self.get_research_run(run_id)
        updated_at = _now()
        assignments = [f"{allowed_fields[name]} = ?" for name in fields]
        values = [fields[name] for name in fields]
        assignments.append("updated_at = ?")
        values.append(updated_at)
        values.append(run_id)
        with self._connect() as conn:
            result = conn.execute(
                f"update research_runs set {', '.join(assignments)} where run_id = ?",
                values,
            )
        if result.rowcount == 0:
            raise KeyError(f"unknown research run id: {run_id}")
        return self.get_research_run(run_id)

    def sync_research_run_counts(self, run_id: str) -> ResearchRunRecord:
        run = self.get_research_run(run_id)
        if not run.batch_id:
            return self.update_research_run(
                run_id,
                screened_count=0,
                blocked_count=0,
                allowed_count=0,
                deep_read_count=0,
            )
        batch = self.get_batch(run.batch_id)
        return self.update_research_run(
            run_id,
            screened_count=batch.screened_count,
            blocked_count=batch.blocked_count,
            allowed_count=batch.screened_count - batch.blocked_count,
            deep_read_count=batch.deep_read_count,
        )

    def add_batch_item(
        self,
        batch_id: str,
        source: str,
        decision: SourceDecision,
        candidate: Candidate | None = None,
        deep_read: bool = False,
    ) -> BatchItemRecord:
        created_at = _now()
        blocked_delta = 0 if decision.allowed else 1
        deep_delta = 1 if deep_read and decision.allowed else 0
        with self._connect() as conn:
            existing = conn.execute("select batch_id from batches where batch_id = ?", (batch_id,)).fetchone()
            if existing is None:
                raise KeyError(f"unknown batch id: {batch_id}")
            conn.execute(
                """
                insert into batch_items (
                    batch_id, source, normalized, allowed, reason, domain,
                    provider, title, doi, pmid, pmcid, arxiv_id, year, url, abstract,
                    relevance_score, relevance_reason, query_variant, query_intent,
                    acronym_expansions, journal, concepts, mesh_terms, oa_status,
                    open_access_pdf_url, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    source,
                    decision.normalized,
                    int(decision.allowed),
                    decision.reason,
                    decision.domain,
                    candidate.provider if candidate else None,
                    candidate.title if candidate else None,
                    candidate.doi if candidate else None,
                    candidate.pmid if candidate else None,
                    candidate.pmcid if candidate else None,
                    candidate.arxiv_id if candidate else None,
                    candidate.year if candidate else None,
                    candidate.url if candidate else None,
                    candidate.abstract if candidate else None,
                    candidate.relevance_score if candidate else None,
                    candidate.relevance_reason if candidate else None,
                    candidate.query_variant if candidate else None,
                    candidate.query_intent if candidate else None,
                    candidate.acronym_expansions if candidate else None,
                    candidate.journal if candidate else None,
                    candidate.concepts if candidate else None,
                    candidate.mesh_terms if candidate else None,
                    candidate.oa_status if candidate else None,
                    candidate.open_access_pdf_url if candidate else None,
                    created_at,
                ),
            )
            conn.execute(
                """
                update batches
                   set screened_count = screened_count + 1,
                       blocked_count = blocked_count + ?,
                       deep_read_count = deep_read_count + ?
                 where batch_id = ?
                """,
                (blocked_delta, deep_delta, batch_id),
            )
        return BatchItemRecord(
            batch_id=batch_id,
            source=source,
            normalized=decision.normalized,
            allowed=decision.allowed,
            reason=decision.reason,
            domain=decision.domain,
            provider=candidate.provider if candidate else None,
            title=candidate.title if candidate else None,
            doi=candidate.doi if candidate else None,
            pmid=candidate.pmid if candidate else None,
            pmcid=candidate.pmcid if candidate else None,
            arxiv_id=candidate.arxiv_id if candidate else None,
            year=candidate.year if candidate else None,
            url=candidate.url if candidate else None,
            abstract=candidate.abstract if candidate else None,
            relevance_score=candidate.relevance_score if candidate else None,
            relevance_reason=candidate.relevance_reason if candidate else None,
            query_variant=candidate.query_variant if candidate else None,
            query_intent=candidate.query_intent if candidate else None,
            acronym_expansions=candidate.acronym_expansions if candidate else None,
            journal=candidate.journal if candidate else None,
            concepts=candidate.concepts if candidate else None,
            mesh_terms=candidate.mesh_terms if candidate else None,
            oa_status=candidate.oa_status if candidate else None,
            open_access_pdf_url=candidate.open_access_pdf_url if candidate else None,
            created_at=created_at,
        )

    def add_batch_item_if_new(
        self,
        batch_id: str,
        source: str,
        decision: SourceDecision,
        candidate: Candidate | None = None,
        deep_read: bool = False,
    ) -> BatchItemRecord | None:
        with self._connect() as conn:
            existing = conn.execute(
                """
                select rowid from batch_items
                 where batch_id = ? and normalized = ?
                 limit 1
                """,
                (batch_id, decision.normalized),
            ).fetchone()
        if existing is not None:
            return None
        return self.add_batch_item(
            batch_id,
            source,
            decision,
            candidate=candidate,
            deep_read=deep_read,
        )

    def get_batch(self, batch_id: str) -> BatchRecord:
        with self._connect() as conn:
            row = conn.execute("select * from batches where batch_id = ?", (batch_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown batch id: {batch_id}")
        return _batch_from_row(row)

    def latest_batch(self) -> BatchRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "select * from batches order by created_at desc, batch_id desc limit 1"
            ).fetchone()
        return _batch_from_row(row) if row is not None else None

    def update_batch_limit(self, batch_id: str, limit: int) -> None:
        with self._connect() as conn:
            result = conn.execute(
                "update batches set limit_value = ? where batch_id = ?",
                (limit, batch_id),
            )
        if result.rowcount == 0:
            raise KeyError(f"unknown batch id: {batch_id}")

    def list_batches(self) -> list[BatchRecord]:
        with self._connect() as conn:
            rows = conn.execute("select * from batches order by created_at desc, batch_id desc").fetchall()
        return [_batch_from_row(row) for row in rows]

    def list_batch_items(self, batch_id: str) -> list[BatchItemRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from batch_items where batch_id = ? order by created_at, rowid",
                (batch_id,),
            ).fetchall()
        return [_batch_item_from_row(row) for row in rows]

    def set_screening_label(
        self,
        batch_id: str,
        source: str,
        label: str,
        note: str | None = None,
        *,
        label_source: str = "human",
        confidence: float | None = None,
        rationale: str | None = None,
        signals: str | None = None,
        overwrite_human: bool = True,
    ) -> ScreeningLabelRecord | None:
        normalized_label = label.strip().lower()
        if normalized_label not in SCREENING_LABEL_CHOICES:
            raise ValueError(f"screening label must be one of: {', '.join(SCREENING_LABEL_CHOICES)}")
        normalized_source = label_source.strip().lower()
        if normalized_source not in SCREENING_LABEL_SOURCE_CHOICES:
            raise ValueError(f"label source must be one of: {', '.join(SCREENING_LABEL_SOURCE_CHOICES)}")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        lookup = _label_lookup_source(source)
        cleaned_source = source.strip()
        now = _now()
        with self._connect() as conn:
            batch = conn.execute("select batch_id from batches where batch_id = ?", (batch_id,)).fetchone()
            if batch is None:
                raise KeyError(f"unknown batch id: {batch_id}")
            item = conn.execute(
                """
                select source, normalized from batch_items
                 where batch_id = ? and (source = ? or normalized = ?)
                 order by rowid
                 limit 1
                """,
                (batch_id, cleaned_source, lookup),
            ).fetchone()
            if item is None:
                raise KeyError(f"unknown batch item for label source: {source}")
            existing = conn.execute(
                """
                select created_at, label_source from screening_labels
                 where batch_id = ? and normalized = ?
                """,
                (batch_id, item["normalized"]),
            ).fetchone()
            if (
                existing is not None
                and normalized_source == "agent"
                and not overwrite_human
                and existing["label_source"] == "human"
            ):
                return None
            created_at = existing["created_at"] if existing is not None else now
            conn.execute(
                """
                insert into screening_labels (
                    batch_id, source, normalized, label, note, label_source,
                    confidence, rationale, signals, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(batch_id, normalized) do update set
                    source = excluded.source,
                    label = excluded.label,
                    note = excluded.note,
                    label_source = excluded.label_source,
                    confidence = excluded.confidence,
                    rationale = excluded.rationale,
                    signals = excluded.signals,
                    updated_at = excluded.updated_at
                """,
                (
                    batch_id,
                    item["source"],
                    item["normalized"],
                    normalized_label,
                    note,
                    normalized_source,
                    confidence,
                    rationale,
                    signals,
                    created_at,
                    now,
                ),
            )
            row = conn.execute(
                """
                select * from screening_labels
                 where batch_id = ? and normalized = ?
                """,
                (batch_id, item["normalized"]),
            ).fetchone()
        return _screening_label_from_row(row)

    def list_screening_labels(self, batch_id: str) -> list[ScreeningLabelRecord]:
        with self._connect() as conn:
            batch = conn.execute("select batch_id from batches where batch_id = ?", (batch_id,)).fetchone()
            if batch is None:
                raise KeyError(f"unknown batch id: {batch_id}")
            rows = conn.execute(
                """
                select * from screening_labels
                 where batch_id = ?
                 order by created_at, rowid
                """,
                (batch_id,),
            ).fetchall()
        return [_screening_label_from_row(row) for row in rows]

    def screening_label_counts(self, batch_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.list_screening_labels(batch_id):
            counts[label.label] = counts.get(label.label, 0) + 1
        return counts

    def add_pdf_artifact(
        self,
        batch_id: str,
        *,
        source: str,
        pdf_url: str | None,
        final_url: str | None,
        sha256: str | None,
        byte_count: int | None,
        content_type: str | None,
        local_path: str | None,
        status: str,
        reason: str,
        parser_name: str | None = None,
        parser_version: str | None = None,
        parse_confidence: float = 0.0,
        parse_flags: tuple[str, ...] = (),
    ) -> PdfArtifactRecord:
        artifact_id = _make_id("pdf")
        created_at = _now()
        deep_delta = 1 if status == "stored" else 0
        with self._connect() as conn:
            existing = conn.execute("select batch_id from batches where batch_id = ?", (batch_id,)).fetchone()
            if existing is None:
                raise KeyError(f"unknown batch id: {batch_id}")
            conn.execute(
                """
                insert into pdf_artifacts (
                    artifact_id, batch_id, source, pdf_url, final_url, sha256,
                    byte_count, content_type, local_path, status, reason,
                    parser_name, parser_version, parse_confidence, parse_flags, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    batch_id,
                    source,
                    pdf_url,
                    final_url,
                    sha256,
                    byte_count,
                    content_type,
                    local_path,
                    status,
                    reason,
                    parser_name,
                    parser_version,
                    parse_confidence,
                    json.dumps(list(parse_flags), sort_keys=True),
                    created_at,
                ),
            )
            if deep_delta:
                conn.execute(
                    "update batches set deep_read_count = deep_read_count + ? where batch_id = ?",
                    (deep_delta, batch_id),
                )
        return PdfArtifactRecord(
            artifact_id=artifact_id,
            batch_id=batch_id,
            source=source,
            pdf_url=pdf_url,
            final_url=final_url,
            sha256=sha256,
            byte_count=byte_count,
            content_type=content_type,
            local_path=local_path,
            status=status,
            reason=reason,
            parser_name=parser_name,
            parser_version=parser_version,
            parse_confidence=parse_confidence,
            parse_flags=parse_flags,
            created_at=created_at,
        )

    def add_pdf_pages(
        self,
        artifact_id: str,
        pages: list[str],
        *,
        page_confidences: list[float] | None = None,
        page_flags: list[tuple[str, ...]] | None = None,
    ) -> list[PdfPageRecord]:
        created_at = _now()
        records = [
            PdfPageRecord(
                artifact_id=artifact_id,
                page_number=index,
                text=text,
                char_count=len(text),
                parse_confidence=(page_confidences[index - 1] if page_confidences and index - 1 < len(page_confidences) else 1.0),
                parse_flags=(page_flags[index - 1] if page_flags and index - 1 < len(page_flags) else ()),
                created_at=created_at,
            )
            for index, text in enumerate(pages, start=1)
        ]
        with self._connect() as conn:
            existing = conn.execute(
                "select artifact_id from pdf_artifacts where artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown pdf artifact id: {artifact_id}")
            conn.executemany(
                """
                insert into pdf_pages (
                    artifact_id, page_number, text, char_count, parse_confidence, parse_flags, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.artifact_id,
                        record.page_number,
                        record.text,
                        record.char_count,
                        record.parse_confidence,
                        json.dumps(list(record.parse_flags), sort_keys=True),
                        record.created_at,
                    )
                    for record in records
                ],
            )
        return records

    def list_pdf_artifacts(self, batch_id: str) -> list[PdfArtifactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from pdf_artifacts where batch_id = ? order by created_at, rowid",
                (batch_id,),
            ).fetchall()
        return [_pdf_artifact_from_row(row) for row in rows]

    def list_pdf_pages(self, artifact_id: str) -> list[PdfPageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from pdf_pages where artifact_id = ? order by page_number",
                (artifact_id,),
            ).fetchall()
        return [_pdf_page_from_row(row) for row in rows]

    def add_evidence_records(self, artifact_id: str, items: list[EvidenceItem]) -> list[EvidenceRecord]:
        created_at = _now()
        records = []
        for item in items:
            trust = assess_evidence_trust(item)
            records.append(
                EvidenceRecord(
                    evidence_id=_make_id("evidence"),
                    artifact_id=artifact_id,
                    evidence_type=item.evidence_type,
                    page_number=item.page_number,
                    text=item.text,
                    char_count=len(item.text),
                    quality_label=item.quality_label,
                    quality_score=item.quality_score,
                    quality_flags=item.quality_flags,
                    parse_confidence=item.parse_confidence,
                    parse_flags=item.parse_flags,
                    trust_label=trust.label,
                    trust_score=trust.score,
                    trust_reasons=trust.reasons,
                    created_at=created_at,
                )
            )
        with self._connect() as conn:
            existing = conn.execute(
                "select artifact_id from pdf_artifacts where artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown pdf artifact id: {artifact_id}")
            conn.executemany(
                """
                insert into evidence_records (
                    evidence_id, artifact_id, evidence_type, page_number, text, char_count,
                    quality_label, quality_score, quality_flags, parse_confidence, parse_flags,
                    trust_label, trust_score, trust_reasons, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.evidence_id,
                        record.artifact_id,
                        record.evidence_type,
                        record.page_number,
                        record.text,
                        record.char_count,
                        record.quality_label,
                        record.quality_score,
                        json.dumps(list(record.quality_flags), sort_keys=True),
                        record.parse_confidence,
                        json.dumps(list(record.parse_flags), sort_keys=True),
                        record.trust_label,
                        record.trust_score,
                        json.dumps(list(record.trust_reasons), sort_keys=True),
                        record.created_at,
                    )
                    for record in records
                ],
            )
        return records

    def list_evidence_records(self, artifact_id: str) -> list[EvidenceRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from evidence_records
                 where artifact_id = ?
                 order by page_number, rowid
                """,
                (artifact_id,),
            ).fetchall()
        return [_evidence_record_from_row(row) for row in rows]

    def replace_report_claim_units(
        self,
        report_package_path: str,
        claim_units_artifact: dict[str, object],
    ) -> list[ReportClaimUnitRecord]:
        created_at = _now()
        source_report = claim_units_artifact.get("source_report")
        if not isinstance(source_report, dict):
            source_report = {}
        units = claim_units_artifact.get("claim_units")
        if not isinstance(units, list):
            units = []
        records = [
            _report_claim_unit_record_from_payload(
                report_package_path,
                unit,
                source_batch_id=_optional_str(source_report.get("batch_id")),
                source_query=_optional_str(source_report.get("query")),
                created_at=created_at,
            )
            for unit in units
            if isinstance(unit, dict)
        ]
        with self._connect() as conn:
            conn.execute(
                "delete from report_claim_units where report_package_path = ?",
                (report_package_path,),
            )
            conn.executemany(
                """
                insert into report_claim_units (
                    report_package_path, claim_unit_id, source_batch_id, source_query,
                    section, claim_type, text, source_sentence, citations, support_status,
                    evidence_count, evidence_types, evidence_row_ids, min_quality_score,
                    min_parse_confidence, min_trust_score, support_details, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.report_package_path,
                        record.claim_unit_id,
                        record.source_batch_id,
                        record.source_query,
                        record.section,
                        record.claim_type,
                        record.text,
                        record.source_sentence,
                        json.dumps(list(record.citations), sort_keys=True),
                        record.support_status,
                        record.evidence_count,
                        json.dumps(list(record.evidence_types), sort_keys=True),
                        json.dumps(list(record.evidence_row_ids), sort_keys=True),
                        record.min_quality_score,
                        record.min_parse_confidence,
                        record.min_trust_score,
                        json.dumps(record.support_details, sort_keys=True),
                        record.created_at,
                    )
                    for record in records
                ],
            )
        return records

    def list_report_claim_units(self, report_package_path: str) -> list[ReportClaimUnitRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select * from report_claim_units
                 where report_package_path = ?
                 order by rowid
                """,
                (report_package_path,),
            ).fetchall()
        return [_report_claim_unit_from_row(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists scans (
                    scan_id text primary key,
                    source text not null,
                    normalized text not null,
                    kind text not null,
                    allowed integer not null,
                    reason text not null,
                    domain text,
                    created_at text not null
                );

                create table if not exists batches (
                    batch_id text primary key,
                    query text,
                    limit_value integer,
                    mode text not null,
                    manifest_path text,
                    screened_count integer not null,
                    blocked_count integer not null,
                    deep_read_count integer not null,
                    created_at text not null
                );

                create table if not exists research_runs (
                    run_id text primary key,
                    batch_id text,
                    query text not null,
                    status text not null,
                    limit_value integer not null,
                    deep_read_limit integer not null,
                    min_relevance integer not null,
                    auto_label_provider text not null,
                    llm_review_limit integer not null,
                    screened_count integer not null,
                    blocked_count integer not null,
                    allowed_count integer not null,
                    deep_read_count integer not null,
                    error text,
                    created_at text not null,
                    updated_at text not null,
                    foreign key(batch_id) references batches(batch_id)
                );

                create table if not exists batch_items (
                    batch_id text not null,
                    source text not null,
                    normalized text not null,
                    allowed integer not null,
                    reason text not null,
                    domain text,
                    provider text,
                    title text,
                    doi text,
                    pmid text,
                    pmcid text,
                    arxiv_id text,
                    year integer,
                    url text,
                    abstract text,
                    relevance_score integer,
                    relevance_reason text,
                    query_variant text,
                    query_intent text,
                    acronym_expansions text,
                    journal text,
                    concepts text,
                    mesh_terms text,
                    oa_status text,
                    open_access_pdf_url text,
                    created_at text not null,
                    foreign key(batch_id) references batches(batch_id)
                );

                create table if not exists pdf_artifacts (
                    artifact_id text primary key,
                    batch_id text not null,
                    source text not null,
                    pdf_url text,
                    final_url text,
                    sha256 text,
                    byte_count integer,
                    content_type text,
                    local_path text,
                    status text not null,
                    reason text not null,
                    parser_name text,
                    parser_version text,
                    parse_confidence real not null default 0.0,
                    parse_flags text not null default '[]',
                    created_at text not null,
                    foreign key(batch_id) references batches(batch_id)
                );

                create table if not exists pdf_pages (
                    artifact_id text not null,
                    page_number integer not null,
                    text text not null,
                    char_count integer not null,
                    parse_confidence real not null default 1.0,
                    parse_flags text not null default '[]',
                    created_at text not null,
                    foreign key(artifact_id) references pdf_artifacts(artifact_id)
                );

                create table if not exists evidence_records (
                    evidence_id text primary key,
                    artifact_id text not null,
                    evidence_type text not null,
                    page_number integer not null,
                    text text not null,
                    char_count integer not null,
                    quality_label text not null default 'clean',
                    quality_score real not null default 1.0,
                    quality_flags text not null default '[]',
                    parse_confidence real not null default 1.0,
                    parse_flags text not null default '[]',
                    trust_label text not null default 'trusted',
                    trust_score real not null default 1.0,
                    trust_reasons text not null default '[]',
                    created_at text not null,
                    foreign key(artifact_id) references pdf_artifacts(artifact_id)
                );

                create table if not exists report_claim_units (
                    report_package_path text not null,
                    claim_unit_id text not null,
                    source_batch_id text,
                    source_query text,
                    section text not null,
                    claim_type text not null,
                    text text not null,
                    source_sentence text not null,
                    citations text not null default '[]',
                    support_status text not null,
                    evidence_count integer not null,
                    evidence_types text not null default '[]',
                    evidence_row_ids text not null default '[]',
                    min_quality_score real,
                    min_parse_confidence real,
                    min_trust_score real,
                    support_details text not null default '{}',
                    created_at text not null,
                    primary key(report_package_path, claim_unit_id)
                );

                create table if not exists screening_labels (
                    batch_id text not null,
                    source text not null,
                    normalized text not null,
                    label text not null,
                    note text,
                    label_source text not null default 'human',
                    confidence real,
                    rationale text,
                    signals text,
                    created_at text not null,
                    updated_at text not null,
                    primary key(batch_id, normalized),
                    foreign key(batch_id) references batches(batch_id)
                );
                """
            )
            self._ensure_batch_item_columns(conn)
            self._ensure_pdf_artifact_columns(conn)
            self._ensure_pdf_page_columns(conn)
            self._ensure_evidence_record_columns(conn)
            self._ensure_screening_label_columns(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Close the connection on exit so the SQLite file handle is released.
        # A bare `with sqlite3.connect(...)` only scopes the transaction, not
        # the handle, which leaks connections and blocks file cleanup on
        # Windows (and invites "database is locked" under concurrent writers).
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_batch_item_columns(self, conn: sqlite3.Connection) -> None:
        existing = _column_names(conn, "batch_items")
        migrations = {
            "provider": "alter table batch_items add column provider text",
            "title": "alter table batch_items add column title text",
            "doi": "alter table batch_items add column doi text",
            "pmid": "alter table batch_items add column pmid text",
            "pmcid": "alter table batch_items add column pmcid text",
            "arxiv_id": "alter table batch_items add column arxiv_id text",
            "year": "alter table batch_items add column year integer",
            "url": "alter table batch_items add column url text",
            "abstract": "alter table batch_items add column abstract text",
            "relevance_score": "alter table batch_items add column relevance_score integer",
            "relevance_reason": "alter table batch_items add column relevance_reason text",
            "query_variant": "alter table batch_items add column query_variant text",
            "query_intent": "alter table batch_items add column query_intent text",
            "acronym_expansions": "alter table batch_items add column acronym_expansions text",
            "journal": "alter table batch_items add column journal text",
            "concepts": "alter table batch_items add column concepts text",
            "mesh_terms": "alter table batch_items add column mesh_terms text",
            "oa_status": "alter table batch_items add column oa_status text",
            "open_access_pdf_url": "alter table batch_items add column open_access_pdf_url text",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)

    def _ensure_screening_label_columns(self, conn: sqlite3.Connection) -> None:
        existing = _column_names(conn, "screening_labels")
        migrations = {
            "label_source": "alter table screening_labels add column label_source text not null default 'human'",
            "confidence": "alter table screening_labels add column confidence real",
            "rationale": "alter table screening_labels add column rationale text",
            "signals": "alter table screening_labels add column signals text",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)

    def _ensure_evidence_record_columns(self, conn: sqlite3.Connection) -> None:
        existing = _column_names(conn, "evidence_records")
        migrations = {
            "quality_label": "alter table evidence_records add column quality_label text not null default 'clean'",
            "quality_score": "alter table evidence_records add column quality_score real not null default 1.0",
            "quality_flags": "alter table evidence_records add column quality_flags text not null default '[]'",
            "parse_confidence": "alter table evidence_records add column parse_confidence real not null default 1.0",
            "parse_flags": "alter table evidence_records add column parse_flags text not null default '[]'",
            "trust_label": "alter table evidence_records add column trust_label text not null default 'trusted'",
            "trust_score": "alter table evidence_records add column trust_score real not null default 1.0",
            "trust_reasons": "alter table evidence_records add column trust_reasons text not null default '[]'",
        }
        added_columns = set()
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)
                added_columns.add(column)
        if {"trust_label", "trust_score", "trust_reasons"} & added_columns:
            self._backfill_evidence_trust(conn)

    def _backfill_evidence_trust(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            select evidence_id, evidence_type, page_number, text, quality_label, quality_score,
                   quality_flags, parse_confidence, parse_flags
              from evidence_records
            """
        ).fetchall()
        for row in rows:
            item = EvidenceItem(
                evidence_type=row["evidence_type"],
                text=row["text"],
                page_number=row["page_number"],
                quality_label=row["quality_label"],
                quality_score=float(row["quality_score"]),
                quality_flags=_json_tuple(row["quality_flags"]),
                parse_confidence=float(row["parse_confidence"]),
                parse_flags=_json_tuple(row["parse_flags"]),
            )
            trust = assess_evidence_trust(item)
            conn.execute(
                """
                update evidence_records
                   set trust_label = ?, trust_score = ?, trust_reasons = ?
                 where evidence_id = ?
                """,
                (
                    trust.label,
                    trust.score,
                    json.dumps(list(trust.reasons), sort_keys=True),
                    row["evidence_id"],
                ),
            )

    def _ensure_pdf_artifact_columns(self, conn: sqlite3.Connection) -> None:
        existing = _column_names(conn, "pdf_artifacts")
        migrations = {
            "parser_name": "alter table pdf_artifacts add column parser_name text",
            "parser_version": "alter table pdf_artifacts add column parser_version text",
            "parse_confidence": "alter table pdf_artifacts add column parse_confidence real not null default 0.0",
            "parse_flags": "alter table pdf_artifacts add column parse_flags text not null default '[]'",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)

    def _ensure_pdf_page_columns(self, conn: sqlite3.Connection) -> None:
        existing = _column_names(conn, "pdf_pages")
        migrations = {
            "parse_confidence": "alter table pdf_pages add column parse_confidence real not null default 1.0",
            "parse_flags": "alter table pdf_pages add column parse_flags text not null default '[]'",
        }
        for column, statement in migrations.items():
            if column not in existing:
                conn.execute(statement)


def _scan_from_row(row: sqlite3.Row) -> ScanRecord:
    return ScanRecord(
        scan_id=row["scan_id"],
        source=row["source"],
        normalized=row["normalized"],
        kind=row["kind"],
        allowed=bool(row["allowed"]),
        reason=row["reason"],
        domain=row["domain"],
        created_at=row["created_at"],
    )


def _batch_from_row(row: sqlite3.Row) -> BatchRecord:
    return BatchRecord(
        batch_id=row["batch_id"],
        query=row["query"],
        limit=row["limit_value"],
        mode=row["mode"],
        manifest_path=row["manifest_path"],
        screened_count=row["screened_count"],
        blocked_count=row["blocked_count"],
        deep_read_count=row["deep_read_count"],
        created_at=row["created_at"],
    )


def _research_run_from_row(row: sqlite3.Row) -> ResearchRunRecord:
    return ResearchRunRecord(
        run_id=row["run_id"],
        batch_id=row["batch_id"],
        query=row["query"],
        status=row["status"],
        limit=row["limit_value"],
        deep_read_limit=row["deep_read_limit"],
        min_relevance=row["min_relevance"],
        auto_label_provider=row["auto_label_provider"],
        llm_review_limit=row["llm_review_limit"],
        screened_count=row["screened_count"],
        blocked_count=row["blocked_count"],
        allowed_count=row["allowed_count"],
        deep_read_count=row["deep_read_count"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _batch_item_from_row(row: sqlite3.Row) -> BatchItemRecord:
    return BatchItemRecord(
        batch_id=row["batch_id"],
        source=row["source"],
        normalized=row["normalized"],
        allowed=bool(row["allowed"]),
        reason=row["reason"],
        domain=row["domain"],
        provider=row["provider"],
        title=row["title"],
        doi=row["doi"],
        pmid=row["pmid"],
        pmcid=row["pmcid"],
        arxiv_id=row["arxiv_id"],
        year=row["year"],
        url=row["url"],
        abstract=row["abstract"],
        relevance_score=row["relevance_score"],
        relevance_reason=row["relevance_reason"],
        query_variant=row["query_variant"],
        query_intent=row["query_intent"],
        acronym_expansions=row["acronym_expansions"],
        journal=row["journal"],
        concepts=row["concepts"],
        mesh_terms=row["mesh_terms"],
        oa_status=row["oa_status"],
        open_access_pdf_url=row["open_access_pdf_url"],
        created_at=row["created_at"],
    )


def _pdf_artifact_from_row(row: sqlite3.Row) -> PdfArtifactRecord:
    return PdfArtifactRecord(
        artifact_id=row["artifact_id"],
        batch_id=row["batch_id"],
        source=row["source"],
        pdf_url=row["pdf_url"],
        final_url=row["final_url"],
        sha256=row["sha256"],
        byte_count=row["byte_count"],
        content_type=row["content_type"],
        local_path=row["local_path"],
        status=row["status"],
        reason=row["reason"],
        parser_name=row["parser_name"],
        parser_version=row["parser_version"],
        parse_confidence=float(row["parse_confidence"]),
        parse_flags=_json_tuple(row["parse_flags"]),
        created_at=row["created_at"],
    )


def _pdf_page_from_row(row: sqlite3.Row) -> PdfPageRecord:
    return PdfPageRecord(
        artifact_id=row["artifact_id"],
        page_number=row["page_number"],
        text=row["text"],
        char_count=row["char_count"],
        parse_confidence=float(row["parse_confidence"]),
        parse_flags=_json_tuple(row["parse_flags"]),
        created_at=row["created_at"],
    )


def _evidence_record_from_row(row: sqlite3.Row) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=row["evidence_id"],
        artifact_id=row["artifact_id"],
        evidence_type=row["evidence_type"],
        page_number=row["page_number"],
        text=row["text"],
        char_count=row["char_count"],
        quality_label=row["quality_label"],
        quality_score=float(row["quality_score"]),
        quality_flags=_json_tuple(row["quality_flags"]),
        parse_confidence=float(row["parse_confidence"]),
        parse_flags=_json_tuple(row["parse_flags"]),
        trust_label=row["trust_label"],
        trust_score=float(row["trust_score"]),
        trust_reasons=_json_tuple(row["trust_reasons"]),
        created_at=row["created_at"],
    )


def _report_claim_unit_record_from_payload(
    report_package_path: str,
    unit: dict[str, object],
    *,
    source_batch_id: str | None,
    source_query: str | None,
    created_at: str,
) -> ReportClaimUnitRecord:
    support_details = unit.get("support_details")
    return ReportClaimUnitRecord(
        report_package_path=report_package_path,
        claim_unit_id=str(unit.get("claim_unit_id") or ""),
        source_batch_id=source_batch_id,
        source_query=source_query,
        section=str(unit.get("section") or ""),
        claim_type=str(unit.get("claim_type") or ""),
        text=str(unit.get("text") or ""),
        source_sentence=str(unit.get("source_sentence") or ""),
        citations=tuple(_string_list_payload(unit.get("citations"))),
        support_status=str(unit.get("support_status") or ""),
        evidence_count=_int_or_zero(unit.get("evidence_count")),
        evidence_types=tuple(_string_list_payload(unit.get("evidence_types"))),
        evidence_row_ids=tuple(_string_list_payload(unit.get("evidence_row_ids"))),
        min_quality_score=_optional_float(unit.get("min_quality_score")),
        min_parse_confidence=_optional_float(unit.get("min_parse_confidence")),
        min_trust_score=_optional_float(unit.get("min_trust_score")),
        support_details=support_details if isinstance(support_details, dict) else {},
        created_at=created_at,
    )


def _report_claim_unit_from_row(row: sqlite3.Row) -> ReportClaimUnitRecord:
    return ReportClaimUnitRecord(
        report_package_path=row["report_package_path"],
        claim_unit_id=row["claim_unit_id"],
        source_batch_id=row["source_batch_id"],
        source_query=row["source_query"],
        section=row["section"],
        claim_type=row["claim_type"],
        text=row["text"],
        source_sentence=row["source_sentence"],
        citations=_json_tuple(row["citations"]),
        support_status=row["support_status"],
        evidence_count=row["evidence_count"],
        evidence_types=_json_tuple(row["evidence_types"]),
        evidence_row_ids=_json_tuple(row["evidence_row_ids"]),
        min_quality_score=_optional_float(row["min_quality_score"]),
        min_parse_confidence=_optional_float(row["min_parse_confidence"]),
        min_trust_score=_optional_float(row["min_trust_score"]),
        support_details=_json_dict(row["support_details"]),
        created_at=row["created_at"],
    )


def _screening_label_from_row(row: sqlite3.Row) -> ScreeningLabelRecord:
    return ScreeningLabelRecord(
        batch_id=row["batch_id"],
        source=row["source"],
        normalized=row["normalized"],
        label=row["label"],
        note=row["note"],
        label_source=row["label_source"],
        confidence=row["confidence"],
        rationale=row["rationale"],
        signals=row["signals"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_tuple(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if str(item))


def _json_dict(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_list_payload(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _make_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{secrets.token_hex(3)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _label_lookup_source(source: str) -> str:
    return evaluate_source(source).normalized


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()}
