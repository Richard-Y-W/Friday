from __future__ import annotations

from typing import Any

from friday.cited_report import build_cited_evidence_data, render_cited_evidence_report
from friday.claim_audit import build_claim_support_audit
from friday.evidence import is_reportable_evidence_text
from friday.screening import build_screening_label_summary
from friday.storage import BatchItemRecord, EvidenceRecord, FridayStore, PdfArtifactRecord


def render_scan_report(store: FridayStore, scan_id: str) -> str:
    data = render_scan_report_json(store, scan_id)
    scan = data["scan"]
    lines = [
        f"Scan ID: {scan['scan_id']}",
        f"Created: {scan['created_at']}",
        f"Source: {scan['source']}",
        f"Normalized: {scan['normalized']}",
        f"Kind: {scan['kind']}",
        f"Status: {scan['status']}",
        f"Reason: {scan['reason']}",
    ]
    if scan["domain"]:
        lines.append(f"Domain: {scan['domain']}")
    lines.extend(
        [
            "",
            "Evidence status:",
            "No parsed paper evidence is stored yet in this CLI slice.",
            "This report is constrained to source-gate and run metadata.",
        ]
    )
    return "\n".join(lines)


def render_scan_report_markdown(store: FridayStore, scan_id: str) -> str:
    data = render_scan_report_json(store, scan_id)
    scan = data["scan"]
    lines = [
        "# Friday Scan Report",
        "",
        f"- Scan ID: `{scan['scan_id']}`",
        f"- Created: {scan['created_at']}",
        f"- Source: {scan['source']}",
        f"- Normalized: {scan['normalized']}",
        f"- Kind: {scan['kind']}",
        f"- Status: {scan['status']}",
        f"- Reason: {scan['reason']}",
    ]
    if scan["domain"]:
        lines.append(f"- Domain: {scan['domain']}")
    lines.extend(
        [
            "",
            "## Evidence Status",
            "",
            "No parsed paper evidence is stored yet in this CLI slice.",
            "This report is constrained to source-gate and run metadata.",
        ]
    )
    return "\n".join(lines)


def render_scan_report_json(store: FridayStore, scan_id: str) -> dict[str, Any]:
    scan = store.get_scan(scan_id)
    return {
        "report_type": "scan",
        "scan": {
            "scan_id": scan.scan_id,
            "created_at": scan.created_at,
            "source": scan.source,
            "normalized": scan.normalized,
            "kind": scan.kind,
            "status": "allowed" if scan.allowed else "blocked",
            "allowed": scan.allowed,
            "reason": scan.reason,
            "domain": scan.domain,
        },
        "evidence_status": {
            "parsed_paper_evidence": False,
            "message": "No parsed paper evidence is stored yet in this CLI slice.",
        },
    }


def render_batch_report(store: FridayStore, batch_id: str) -> str:
    data = build_batch_report_data(store, batch_id)
    batch = data["batch"]
    lines = [
        f"Batch ID: {batch['batch_id']}",
        f"Created: {batch['created_at']}",
        f"Mode: {batch['mode']}",
    ]
    if batch["query"]:
        lines.append(f"Query: {batch['query']}")
    if batch["limit"] is not None:
        lines.append(f"Limit: {batch['limit']}")
    if batch["manifest_path"]:
        lines.append(f"Manifest: {batch['manifest_path']}")
    lines.extend(
        [
            f"Screened: {batch['screened_count']}",
            f"Blocked: {batch['blocked_count']}",
            f"Allowed: {batch['allowed_count']}",
            f"Deep-scanned: {batch['deep_read_count']}",
        ]
    )
    lines.extend(_render_screening_labels_from_data(data))
    if data["items"]:
        lines.extend(["", "Items:"])
        for item in data["items"]:
            lines.append(_render_batch_item_data(item))
    else:
        lines.extend(["", "Items:", "- No candidate items are stored for this batch yet."])
    lines.extend(_render_evidence_status_from_data(data))
    lines.extend(_render_claim_support_audit(data["claim_support_audit"]))
    lines.extend(["", render_cited_evidence_report(store, batch_id)])
    return "\n".join(lines)


def render_batch_report_markdown(store: FridayStore, batch_id: str) -> str:
    data = build_batch_report_data(store, batch_id)
    batch = data["batch"]
    cited = data["cited_evidence"]
    lines = [
        "# Friday Batch Report",
        "",
        "## Coverage",
        "",
        f"- Batch ID: `{batch['batch_id']}`",
        f"- Created: {batch['created_at']}",
        f"- Mode: {batch['mode']}",
    ]
    if batch["query"]:
        lines.append(f"- Query: {batch['query']}")
    if batch["limit"] is not None:
        lines.append(f"- Limit: {batch['limit']}")
    if batch["manifest_path"]:
        lines.append(f"- Manifest: {batch['manifest_path']}")
    lines.extend(
        [
            f"- Screened: {batch['screened_count']}",
            f"- Blocked: {batch['blocked_count']}",
            f"- Allowed: {batch['allowed_count']}",
            f"- Deep-scanned: {batch['deep_read_count']}",
            "",
            "## Screening Labels",
            "",
            _render_markdown_screening_label_counts(data["screening_labels"]),
            "",
        ]
    )
    if data["screening_labels"]["labels"]:
        for label in data["screening_labels"]["labels"]:
            lines.append(_render_markdown_screening_label(label))
    else:
        lines.append("- No human screening labels are stored for this batch yet.")
    lines.extend(
        [
            "",
            "## Items",
            "",
        ]
    )
    if data["items"]:
        lines.extend(_render_markdown_item(item) for item in data["items"])
    else:
        lines.append("- No candidate items are stored for this batch yet.")

    lines.extend(["", "## Parsed PDFs", ""])
    if data["pdf_artifacts"]:
        for artifact in data["pdf_artifacts"]:
            parser = _render_parser_summary_data(artifact)
            lines.append(
                f"- {artifact['status']}: {artifact['pdf_url'] or artifact['source']} "
                f"(reason={artifact['reason']}; pages={artifact['page_count']}; "
                f"bytes={artifact['byte_count'] or 0}{parser})"
            )
    else:
        lines.append("- No parsed paper evidence is stored yet in this CLI slice.")

    lines.extend(["", "## Cited Evidence", ""])
    if cited["paper_references"]:
        lines.append("### Paper References")
        lines.append("")
        for reference in cited["paper_references"]:
            lines.append(_markdown_paper_reference(reference))
        lines.extend(["", "### Evidence", ""])
        for evidence_type in ("claim", "method", "result", "dataset_population", "limitation"):
            records = cited["evidence"].get(evidence_type, [])
            if not records:
                continue
            lines.append(f"#### {evidence_type.replace('_', ' ').title()}")
            for record in records[:8]:
                lines.append(f"- [{record['citation']}] {record['text']}")
            lines.append("")
    else:
        lines.append("No extracted evidence is available for this batch yet.")

    lines.extend(["", "### Evidence Gaps", ""])
    lines.extend(f"- {gap}" for gap in cited["evidence_gaps"])

    quality = data["evidence_status"]["quality_summary"]
    lines.extend(["", "## Evidence Quality", ""])
    lines.append(f"- Accepted evidence: {quality['accepted_evidence_count']}")
    lines.append(f"- Blocked evidence: {quality['blocked_evidence_count']}")
    lines.append(f"- Suspect evidence: {quality['suspect_evidence_count']}")
    if quality["blocked_by_flag"]:
        lines.append("- Blocked by flag:")
        for flag, count in sorted(quality["blocked_by_flag"].items()):
            lines.append(f"  - {flag}: {count}")

    audit = data["claim_support_audit"]
    lines.extend(["", "## Claim Support Audit", ""])
    lines.append(
        f"- Supported: {audit['counts']['supported']}; "
        f"missing page anchor: {audit['counts']['missing_page_anchor']}; "
        f"material gaps: {audit['counts']['material_gaps']}"
    )
    if audit["supported_claims"]:
        for item in audit["supported_claims"][:10]:
            lines.append(f"- SUPPORTED [{item['citation']}] {item['text']}")
    if audit["material_gaps"]:
        for gap in audit["material_gaps"]:
            lines.append(f"- MATERIAL GAP: {gap['message']}")
    return "\n".join(lines).rstrip()


def render_batch_report_json(store: FridayStore, batch_id: str) -> dict[str, Any]:
    return build_batch_report_data(store, batch_id)


def build_batch_report_data(store: FridayStore, batch_id: str) -> dict[str, Any]:
    batch = store.get_batch(batch_id)
    items = store.list_batch_items(batch.batch_id)
    labels = store.list_screening_labels(batch.batch_id)
    screening_labels = build_screening_label_summary(items, labels)
    labels_by_normalized = {
        label["normalized"]: label
        for label in screening_labels["labels"]
    }
    artifacts = store.list_pdf_artifacts(batch.batch_id)
    return {
        "report_type": "batch",
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
        "items": [_batch_item_data(item, labels_by_normalized.get(item.normalized)) for item in items],
        "screening_labels": screening_labels,
        "pdf_artifacts": [_pdf_artifact_data(store, artifact) for artifact in artifacts],
        "evidence_status": _evidence_status_data(store, artifacts),
        "cited_evidence": build_cited_evidence_data(store, batch.batch_id),
        "claim_support_audit": build_claim_support_audit(store, batch.batch_id),
    }


def _render_batch_item(item: BatchItemRecord) -> str:
    return _render_batch_item_data(_batch_item_data(item))


def _render_batch_item_data(item: dict[str, Any]) -> str:
    status = "allowed" if item["allowed"] else "blocked"
    provider = item["provider"] or "manual"
    title = item["title"] or item["source"]
    identifier = _item_identifier_data(item)
    suffix = f"; {identifier}" if identifier else ""
    relevance = _item_relevance_data(item)
    query_context = _item_query_context_data(item)
    metadata = _item_metadata_data(item)
    return (
        f"- {status}: [{provider}] {title}{suffix}{relevance}{query_context}{metadata}; "
        f"source={item['source']}; reason={item['reason']}"
    )


def _batch_item_data(item: BatchItemRecord, screening_label: dict[str, object] | None = None) -> dict[str, Any]:
    return {
        "batch_id": item.batch_id,
        "source": item.source,
        "normalized": item.normalized,
        "allowed": item.allowed,
        "reason": item.reason,
        "domain": item.domain,
        "provider": item.provider,
        "title": item.title,
        "doi": item.doi,
        "pmid": item.pmid,
        "pmcid": item.pmcid,
        "arxiv_id": item.arxiv_id,
        "year": item.year,
        "url": item.url,
        "abstract": item.abstract,
        "relevance_score": item.relevance_score,
        "relevance_reason": item.relevance_reason,
        "query_variant": item.query_variant,
        "query_intent": item.query_intent,
        "acronym_expansions": item.acronym_expansions,
        "journal": item.journal,
        "concepts": item.concepts,
        "mesh_terms": item.mesh_terms,
        "oa_status": item.oa_status,
        "open_access_pdf_url": item.open_access_pdf_url,
        "screening_label": screening_label,
        "created_at": item.created_at,
    }


def _pdf_artifact_data(store: FridayStore, artifact: PdfArtifactRecord) -> dict[str, Any]:
    records = store.list_evidence_records(artifact.artifact_id)
    quality = _evidence_quality_summary(records)
    return {
        "artifact_id": artifact.artifact_id,
        "batch_id": artifact.batch_id,
        "source": artifact.source,
        "pdf_url": artifact.pdf_url,
        "final_url": artifact.final_url,
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_count,
        "content_type": artifact.content_type,
        "local_path": artifact.local_path,
        "status": artifact.status,
        "reason": artifact.reason,
        "parser_name": artifact.parser_name,
        "parser_version": artifact.parser_version,
        "parse_confidence": artifact.parse_confidence,
        "parse_flags": list(artifact.parse_flags),
        "created_at": artifact.created_at,
        "page_count": len(store.list_pdf_pages(artifact.artifact_id)),
        "evidence_count": quality["accepted_evidence_count"],
        "accepted_evidence_count": quality["accepted_evidence_count"],
        "blocked_evidence_count": quality["blocked_evidence_count"],
        "suspect_evidence_count": quality["suspect_evidence_count"],
        "blocked_by_flag": quality["blocked_by_flag"],
    }


def _evidence_status_data(
    store: FridayStore,
    artifacts: list[PdfArtifactRecord],
) -> dict[str, Any]:
    stored_count = 0
    evidence_preview = []
    evidence_type_counts: dict[str, int] = {}
    all_records: list[EvidenceRecord] = []
    for artifact in artifacts:
        if artifact.status == "stored":
            stored_count += 1
        for record in store.list_evidence_records(artifact.artifact_id):
            all_records.append(record)
            if not _is_clean_evidence_record(record):
                continue
            if evidence_type_counts.get(record.evidence_type, 0) >= 2:
                continue
            evidence_type_counts[record.evidence_type] = evidence_type_counts.get(record.evidence_type, 0) + 1
            evidence_preview.append(
                {
                    "evidence_type": record.evidence_type,
                    "page_number": record.page_number,
                    "text": record.text,
                    "artifact_id": record.artifact_id,
                }
            )
    if not artifacts:
        message = "No parsed paper evidence is stored yet in this CLI slice."
    elif stored_count:
        message = "Parsed page-level paper text is stored."
    else:
        message = "No parsed paper evidence is stored yet for this batch."
    return {
        "parsed_page_text": bool(stored_count),
        "stored_pdf_count": stored_count,
        "message": message,
        "evidence_preview": evidence_preview[:12],
        "quality_summary": _evidence_quality_summary(all_records),
    }


def _render_evidence_status_from_data(data: dict[str, Any]) -> list[str]:
    artifacts = data["pdf_artifacts"]
    if not artifacts:
        return [
            "",
            "Evidence status:",
            data["evidence_status"]["message"],
            "This report is constrained to source-gate and run metadata.",
        ]

    lines = ["", "Parsed PDFs:"]
    for artifact in artifacts:
        parser = _render_parser_summary_data(artifact)
        lines.append(
            f"- {artifact['status']}: {artifact['pdf_url'] or artifact['source']}; "
            f"reason={artifact['reason']}; pages={artifact['page_count']}; "
            f"bytes={artifact['byte_count'] or 0}{parser}"
        )
    lines.extend(["", "Evidence status:", data["evidence_status"]["message"]])
    quality = data["evidence_status"]["quality_summary"]
    lines.extend(
        [
            (
                "Evidence quality: "
                f"accepted={quality['accepted_evidence_count']} "
                f"blocked={quality['blocked_evidence_count']} "
                f"suspect={quality['suspect_evidence_count']}"
            )
        ]
    )
    preview = data["evidence_status"]["evidence_preview"]
    if preview:
        lines.extend(["", "Extracted evidence:"])
        lines.extend(
            f"- {item['evidence_type']} p{item['page_number']}: {item['text']}"
            for item in preview
        )
    return lines


def _render_parser_summary_data(artifact: dict[str, Any]) -> str:
    if not artifact.get("parser_name"):
        return ""
    suffix = (
        f"; parser={artifact['parser_name']}"
        f"; confidence={artifact['parse_confidence']:.2f}"
    )
    if artifact.get("parse_flags"):
        suffix += f"; flags={','.join(artifact['parse_flags'])}"
    return suffix


def _is_clean_evidence_record(record: EvidenceRecord) -> bool:
    return record.quality_label == "clean" and is_reportable_evidence_text(record.text)


def _evidence_quality_summary(records: list[EvidenceRecord]) -> dict[str, Any]:
    accepted_count = 0
    blocked_count = 0
    suspect_count = 0
    blocked_by_flag: dict[str, int] = {}
    for record in records:
        if _is_clean_evidence_record(record):
            accepted_count += 1
            continue
        if record.quality_label == "suspect":
            suspect_count += 1
        else:
            blocked_count += 1
        flags = record.quality_flags or ("legacy_quality_filter",)
        for flag in flags:
            blocked_by_flag[flag] = blocked_by_flag.get(flag, 0) + 1
    return {
        "accepted_evidence_count": accepted_count,
        "blocked_evidence_count": blocked_count,
        "suspect_evidence_count": suspect_count,
        "blocked_by_flag": blocked_by_flag,
    }


def _render_screening_labels_from_data(data: dict[str, Any]) -> list[str]:
    summary = data["screening_labels"]
    counts = summary["counts"]
    lines = [
        "",
        "Screening labels:",
        f"relevant={counts['relevant']} maybe={counts['maybe']} irrelevant={counts['irrelevant']}",
    ]
    if not summary["labels"]:
        lines.append("- No human screening labels are stored for this batch yet.")
        return lines
    for label in summary["labels"]:
        note = f"; note={label['note']}" if label["note"] else ""
        title = label["title"] or label["source"]
        lines.append(f"- {label['label']}: {title}; source={label['source']}{note}")
    return lines


def _render_claim_support_audit(audit: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "Claim support audit:",
        (
            f"supported={audit['counts']['supported']}; "
            f"missing-page-anchor={audit['counts']['missing_page_anchor']}; "
            f"material-gaps={audit['counts']['material_gaps']}"
        ),
    ]
    if audit["material_gaps"]:
        lines.extend(f"- MATERIAL GAP: {gap['message']}" for gap in audit["material_gaps"])
    return lines


def _item_identifier_data(item: dict[str, Any]) -> str | None:
    if item["doi"]:
        return f"doi={item['doi']}"
    if item["pmid"]:
        return f"pmid={item['pmid']}"
    if item["arxiv_id"]:
        return f"arxiv={item['arxiv_id']}"
    return None


def _item_relevance_data(item: dict[str, Any]) -> str:
    if item["relevance_score"] is None:
        return ""
    reason = f" ({item['relevance_reason']})" if item["relevance_reason"] else ""
    return f"; relevance={item['relevance_score']}{reason}"


def _item_query_context_data(item: dict[str, Any]) -> str:
    parts = []
    if item["query_variant"]:
        parts.append(f"query={item['query_variant']}")
    if item["acronym_expansions"]:
        parts.append(f"acronyms={item['acronym_expansions']}")
    if item["query_intent"] and item["query_intent"] != "unknown":
        parts.append(f"intent={item['query_intent']}")
    return "; " + "; ".join(parts) if parts else ""


def _item_metadata_data(item: dict[str, Any]) -> str:
    parts = []
    if item["pmcid"]:
        parts.append(f"pmcid={item['pmcid']}")
    if item["journal"]:
        parts.append(f"journal={item['journal']}")
    if item["mesh_terms"]:
        parts.append(f"mesh={item['mesh_terms']}")
    if item["concepts"]:
        parts.append(f"concepts={item['concepts']}")
    if item["oa_status"]:
        parts.append(f"oa={item['oa_status']}")
    return "; " + "; ".join(parts) if parts else ""


def _render_markdown_screening_label_counts(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return (
        f"- Relevant: {counts['relevant']}; "
        f"maybe: {counts['maybe']}; "
        f"irrelevant: {counts['irrelevant']}"
    )


def _render_markdown_screening_label(label: dict[str, Any]) -> str:
    note = f"; note={label['note']}" if label["note"] else ""
    title = label["title"] or label["source"]
    return f"- **{label['label']}** {title}; source={label['source']}{note}"


def _render_markdown_item(item: dict[str, Any]) -> str:
    status = "allowed" if item["allowed"] else "blocked"
    provider = item["provider"] or "manual"
    title = item["title"] or item["source"]
    identifier = _item_identifier_data(item)
    parts = [f"- **{status}** [{provider}] {title}"]
    if identifier:
        parts.append(identifier)
    if item["relevance_score"] is not None:
        parts.append(f"relevance={item['relevance_score']}")
    if item["screening_label"]:
        parts.append(f"label={item['screening_label']['label']}")
    parts.append(f"source={item['source']}")
    parts.append(f"reason={item['reason']}")
    return "; ".join(parts)


def _markdown_paper_reference(reference: dict[str, Any]) -> str:
    parts = [f"- [{reference['label']}] {reference['title'] or reference['source']}"]
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
