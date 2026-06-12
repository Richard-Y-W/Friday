from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from friday.evidence import is_reportable_evidence_text
from friday.storage import BatchItemRecord, EvidenceRecord, FridayStore, PdfArtifactRecord


TYPE_HEADINGS = {
    "claim": "Claims",
    "method": "Methods",
    "result": "Results",
    "limitation": "Limitations",
    "dataset_population": "Datasets / Populations",
}


@dataclass(frozen=True)
class PaperReference:
    label: str
    artifact: PdfArtifactRecord
    item: BatchItemRecord | None


def render_cited_evidence_report(store: FridayStore, batch_id: str) -> str:
    data = build_cited_evidence_data(store, batch_id)
    batch = data["batch"]
    evidence_by_type = data["evidence"]
    evidence_count = sum(len(records) for records in evidence_by_type.values())
    lines = [
        "Cited Evidence Report",
        f"Batch ID: {batch['batch_id']}",
    ]
    if batch["query"]:
        lines.append(f"Query: {batch['query']}")
    lines.append(
        "Coverage: "
        f"screened={data['coverage']['screened']}; "
        f"allowed={data['coverage']['allowed']}; "
        f"blocked={data['coverage']['blocked']}; "
        f"deep-read={data['coverage']['deep_read']}; "
        f"stored-pdfs={data['coverage']['stored_pdfs']}"
    )

    if not evidence_count:
        lines.extend(["", "No extracted evidence is available for this batch yet.", "", "Evidence Gaps:"])
        lines.extend(f"- {gap}" for gap in data["evidence_gaps"])
        return "\n".join(lines)

    lines.extend(["", "Paper References:"])
    for reference in data["paper_references"]:
        lines.append(_paper_reference_data_line(reference))

    lines.extend(["", "Evidence:"])
    for evidence_type in ("claim", "method", "result", "dataset_population", "limitation"):
        records = evidence_by_type.get(evidence_type, [])
        if not records:
            continue
        lines.append(f"{TYPE_HEADINGS[evidence_type]}:")
        for record in records[:8]:
            lines.append(f"- [{record['citation']}] {record['text']}")

    lines.extend(["", "Evidence Gaps:"])
    lines.extend(f"- {gap}" for gap in data["evidence_gaps"])
    return "\n".join(lines)


def build_cited_evidence_data(store: FridayStore, batch_id: str) -> dict[str, Any]:
    batch = store.get_batch(batch_id)
    items_by_source = {item.source: item for item in store.list_batch_items(batch.batch_id)}
    artifacts = store.list_pdf_artifacts(batch.batch_id)
    stored_artifacts = [artifact for artifact in artifacts if artifact.status == "stored"]
    blocked_artifacts = [artifact for artifact in artifacts if artifact.status != "stored"]
    references = [
        PaperReference(label=f"P{index}", artifact=artifact, item=items_by_source.get(artifact.source))
        for index, artifact in enumerate(stored_artifacts, start=1)
    ]
    evidence_by_type = _evidence_by_type(store, references)
    allowed_count = batch.screened_count - batch.blocked_count
    evidence_gaps = [
        f"Deep-read coverage: {len(stored_artifacts)} of {batch.screened_count} screened records produced stored PDFs."
    ]
    if blocked_artifacts:
        evidence_gaps.append(f"PDF failures: {len(blocked_artifacts)} blocked or failed PDF attempts.")
    blocked_evidence_count = sum(
        1
        for artifact in stored_artifacts
        for record in store.list_evidence_records(artifact.artifact_id)
        if not _is_clean_evidence(record)
    )
    low_trust_evidence_count = sum(
        1
        for artifact in stored_artifacts
        for record in store.list_evidence_records(artifact.artifact_id)
        if _is_clean_evidence(record) and not _is_trusted_evidence(record)
    )
    if blocked_evidence_count:
        evidence_gaps.append(f"Evidence quality gate blocked {blocked_evidence_count} extracted fragments.")
    if low_trust_evidence_count:
        evidence_gaps.append(f"Evidence trust gate withheld {low_trust_evidence_count} low-trust extracted fragments.")
    if not evidence_by_type.get("limitation"):
        evidence_gaps.append("No extracted limitation evidence found.")

    return {
        "batch": {
            "batch_id": batch.batch_id,
            "query": batch.query,
        },
        "coverage": {
            "screened": batch.screened_count,
            "allowed": allowed_count,
            "blocked": batch.blocked_count,
            "deep_read": batch.deep_read_count,
            "stored_pdfs": len(stored_artifacts),
        },
        "paper_references": [_paper_reference_data(reference) for reference in references],
        "evidence": _evidence_data_by_type(evidence_by_type),
        "evidence_gaps": evidence_gaps,
    }


def _evidence_by_type(
    store: FridayStore,
    references: list[PaperReference],
) -> dict[str, list[tuple[PaperReference, EvidenceRecord]]]:
    evidence_by_type: dict[str, list[tuple[PaperReference, EvidenceRecord]]] = {}
    for reference in references:
        for record in store.list_evidence_records(reference.artifact.artifact_id):
            if not _is_trusted_evidence(record):
                continue
            evidence_by_type.setdefault(record.evidence_type, []).append((reference, record))
    return evidence_by_type


def _evidence_data_by_type(
    evidence_by_type: dict[str, list[tuple[PaperReference, EvidenceRecord]]],
) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {evidence_type: [] for evidence_type in TYPE_HEADINGS}
    for evidence_type, records in evidence_by_type.items():
        data[evidence_type] = [
            {
                "paper": reference.label,
                "page_number": record.page_number,
                "citation": f"{reference.label} p{record.page_number}",
                "text": record.text,
                "quality_label": record.quality_label,
                "quality_score": record.quality_score,
                "quality_flags": list(record.quality_flags),
                "parse_confidence": record.parse_confidence,
                "parse_flags": list(record.parse_flags),
                "trust_label": record.trust_label,
                "trust_score": record.trust_score,
                "trust_reasons": list(record.trust_reasons),
            }
            for reference, record in records
        ]
    return data


def _is_clean_evidence(record: EvidenceRecord) -> bool:
    return record.quality_label == "clean" and is_reportable_evidence_text(record.text)


def _is_trusted_evidence(record: EvidenceRecord) -> bool:
    return _is_clean_evidence(record) and record.trust_label == "trusted"


def _paper_reference_data(reference: PaperReference) -> dict[str, Any]:
    item = reference.item
    return {
        "label": reference.label,
        "source": reference.artifact.source,
        "title": item.title if item else None,
        "doi": item.doi if item else None,
        "pmid": item.pmid if item else None,
        "pmcid": item.pmcid if item else None,
        "arxiv_id": item.arxiv_id if item else None,
        "journal": item.journal if item else None,
        "year": item.year if item else None,
        "pdf_url": reference.artifact.pdf_url,
        "artifact_id": reference.artifact.artifact_id,
    }


def _paper_reference_data_line(reference: dict[str, Any]) -> str:
    parts = [f"[{reference['label']}] {reference['title'] or reference['source']}"]
    identifiers = []
    if reference["doi"]:
        identifiers.append(f"doi={reference['doi']}")
    if reference["pmid"]:
        identifiers.append(f"pmid={reference['pmid']}")
    if reference["pmcid"]:
        identifiers.append(f"pmcid={reference['pmcid']}")
    if reference["arxiv_id"]:
        identifiers.append(f"arxiv={reference['arxiv_id']}")
    if reference["journal"]:
        identifiers.append(f"journal={reference['journal']}")
    if reference["year"]:
        identifiers.append(f"year={reference['year']}")
    if identifiers:
        parts.append("; ".join(identifiers))
    return "; ".join(parts)
