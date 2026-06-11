from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Union


PackageFileContent = Union[str, bytes]


MODE_CHOICES = (
    "claim-table",
    "outline",
    "literature-review",
    "limitations",
    "background",
    "methods-summary",
    "results-summary",
    "research-gaps",
)

TYPE_LABELS = {
    "claim": "Claim",
    "method": "Method",
    "result": "Result",
    "dataset_population": "Dataset/population",
    "limitation": "Limitation",
}

EVIDENCE_TABLES = {
    "claim": ("claims", "claims.csv"),
    "method": ("methods", "methods.csv"),
    "result": ("results", "results.csv"),
    "dataset_population": ("populations", "populations.csv"),
    "limitation": ("limitations", "limitations.csv"),
}

EVIDENCE_TABLE_COLUMNS = (
    "row_id",
    "claim_id",
    "support_status",
    "table",
    "evidence_type",
    "paper",
    "citation",
    "page_number",
    "text",
    "paper_title",
    "year",
    "journal",
    "doi",
    "source",
)


def build_writing_payload(report_data: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if mode not in MODE_CHOICES:
        raise ValueError(f"unsupported writing mode: {mode}")
    audit = report_data.get("claim_support_audit") or {}
    cited = report_data.get("cited_evidence") or {}
    batch = report_data.get("batch") or {}
    screening_labels = report_data.get("screening_labels") or _empty_screening_label_summary()
    claims = list(audit.get("supported_claims") or [])
    material_gaps = _material_gaps(audit, cited)
    paper_references = _paper_references(cited, claims)
    claims_by_type = _claims_by_type(claims)
    evidence_clusters = _evidence_clusters(claims_by_type)
    synthesis_blocks = _synthesis_blocks(evidence_clusters)
    paragraph_claim_audit = _paragraph_claim_audit(synthesis_blocks)
    payload = {
        "schema_version": "1.0",
        "artifact_type": "writing_copilot_output",
        "mode": mode,
        "safety_policy": {
            "evidence_bound": True,
            "llm_used": False,
            "rule": "Use only page-anchored supported claims from the scanner report; mark missing support as MATERIAL GAP.",
        },
        "source_report": {
            "report_type": report_data.get("report_type"),
            "batch_id": batch.get("batch_id"),
            "query": batch.get("query"),
            "screened_count": batch.get("screened_count"),
            "blocked_count": batch.get("blocked_count"),
            "deep_read_count": batch.get("deep_read_count"),
            "screening_label_counts": screening_labels.get("counts", {}),
        },
        "screening_labels": screening_labels,
        "claims": claims,
        "claims_by_type": claims_by_type,
        "paper_references": paper_references,
        "evidence_clusters": evidence_clusters,
        "synthesis_blocks": synthesis_blocks,
        "paragraph_claim_audit": paragraph_claim_audit,
        "material_gaps": material_gaps,
    }
    payload["evidence_tables"] = build_evidence_tables(payload)
    payload["citation_check"] = validate_citation_coverage(payload)
    payload["audit_summary"] = build_writing_audit_summary(payload)
    return payload


def build_writing_audit_summary(payload: dict[str, Any]) -> dict[str, Any]:
    paragraph_audit = _paragraph_claim_audit(payload.get("synthesis_blocks", []))
    citation_check = validate_citation_coverage(payload)
    supported = [
        _audit_summary_entry(entry)
        for entry in paragraph_audit
        if entry["support_status"] == "SUPPORTED"
    ]
    blocked = [
        _audit_summary_entry(entry)
        for entry in paragraph_audit
        if entry["support_status"] != "SUPPORTED"
    ]
    return {
        "schema_version": "1.0",
        "citation_check_status": citation_check["status"],
        "total_paragraph_count": len(paragraph_audit),
        "supported_paragraph_count": len(supported),
        "blocked_paragraph_count": len(blocked),
        "material_gap_count": len(payload.get("material_gaps", [])),
        "supported_paragraphs": supported,
        "blocked_paragraphs": blocked,
    }


def build_writing_package_files(payload: dict[str, Any]) -> dict[str, PackageFileContent]:
    writing_payload = dict(payload)
    writing_payload["evidence_tables"] = build_evidence_tables(writing_payload)
    writing_payload["citation_check"] = validate_citation_coverage(writing_payload)
    writing_payload["audit_summary"] = build_writing_audit_summary(writing_payload)
    audit_summary = writing_payload["audit_summary"]
    evidence_tables = writing_payload["evidence_tables"]
    report_markdown = render_evidence_report_markdown(writing_payload)
    files = {
        "writing.json": _json_text(writing_payload),
        "draft.md": render_writing_markdown(writing_payload).rstrip() + "\n",
        "report.md": report_markdown.rstrip() + "\n",
        "report.pdf": render_report_pdf_bytes(report_markdown),
        "paper_references.json": _json_text(writing_payload.get("paper_references", [])),
        "supported_paragraphs.json": _json_text(audit_summary["supported_paragraphs"]),
        "blocked_paragraphs.json": _json_text(audit_summary["blocked_paragraphs"]),
        "material_gaps.json": _json_text(writing_payload.get("material_gaps", [])),
        "source_report.json": _json_text(writing_payload.get("source_report", {})),
        "screening_labels.json": _json_text(writing_payload.get("screening_labels", _empty_screening_label_summary())),
        "citation_audit.json": _json_text(audit_summary),
        "literature_table.csv": _literature_csv_text(writing_payload.get("paper_references", [])),
        "evidence_tables.json": _json_text(evidence_tables),
        "evidence_table.csv": _csv_text(evidence_tables["all_rows"]),
    }
    for _evidence_type, (table_name, filename) in EVIDENCE_TABLES.items():
        files[filename] = _csv_text(evidence_tables["tables"][table_name])
    return files


def render_evidence_report_markdown(payload: dict[str, Any]) -> str:
    source = payload.get("source_report", {})
    references = payload.get("paper_references", [])
    claims_by_type = payload.get("claims_by_type", {})
    audit = payload.get("audit_summary") or build_writing_audit_summary(payload)
    lines = [
        "# Jarvis Evidence Report",
        "",
        _source_line(payload),
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(_executive_summary_lines(payload, audit))
    lines.extend(
        [
            "",
            "## Background",
            "",
        ]
    )
    lines.extend(_report_claim_lines(claims_by_type.get("dataset_population", []), empty="No page-anchored background or population evidence was extracted."))
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )
    lines.extend(_report_claim_lines(claims_by_type.get("result", []), empty="No page-anchored result evidence was extracted."))
    lines.extend(
        [
            "",
            "## Methods And Evidence Base",
            "",
        ]
    )
    lines.extend(_report_claim_lines(claims_by_type.get("method", []), empty="No page-anchored method evidence was extracted."))
    lines.extend(
        [
            "",
            "## Limitations And Gaps",
            "",
        ]
    )
    limitation_lines = _report_claim_lines(claims_by_type.get("limitation", []), empty="")
    if limitation_lines:
        lines.extend(limitation_lines)
    else:
        lines.append("- No page-anchored limitation evidence was extracted.")
    for gap in payload.get("material_gaps", []):
        lines.append(f"- MATERIAL GAP: {gap['message']}")
    lines.extend(
        [
            "",
            "## Literature Table",
            "",
            "| Paper | Title | Year | Venue | Evidence Count |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if references:
        for reference in references:
            lines.append(
                "| "
                f"{_cell(reference.get('label', ''))} | "
                f"{_cell(reference.get('title') or '')} | "
                f"{_cell(reference.get('year') or '')} | "
                f"{_cell(reference.get('journal') or '')} | "
                f"{_cell(reference.get('evidence_count') or 0)} |"
            )
    else:
        lines.append("| MATERIAL GAP | No parsed paper references available. |  |  | 0 |")
    lines.extend(
        [
            "",
            "## Evidence Table",
            "",
            f"- Supported extracted claims: {len(payload.get('claims', []))}",
            f"- Parsed paper references: {len(references)}",
            f"- Screened records: {source.get('screened_count', 0) or 0}",
            f"- Blocked records: {source.get('blocked_count', 0) or 0}",
            f"- Deep-read PDFs: {source.get('deep_read_count', 0) or 0}",
            "",
            "See `evidence_table.csv` for the full page-anchored evidence table.",
            "",
            "## Citation Audit",
            "",
            f"- Citation check: {audit.get('citation_check_status', 'unknown')}",
            f"- Supported paragraphs: {audit.get('supported_paragraph_count', 0)}",
            f"- Blocked paragraphs: {audit.get('blocked_paragraph_count', 0)}",
            f"- Material gaps: {audit.get('material_gap_count', 0)}",
        ]
    )
    return "\n".join(lines).rstrip()


def build_evidence_tables(payload: dict[str, Any]) -> dict[str, Any]:
    references = {reference["label"]: reference for reference in payload.get("paper_references", [])}
    tables: dict[str, list[dict[str, Any]]] = {
        table_name: []
        for table_name, _filename in EVIDENCE_TABLES.values()
    }
    all_rows = []
    for index, claim in enumerate(payload.get("claims", []), start=1):
        evidence_type = str(claim.get("evidence_type") or "claim")
        table_name = EVIDENCE_TABLES.get(evidence_type, EVIDENCE_TABLES["claim"])[0]
        reference = references.get(claim.get("paper"), {})
        row = _evidence_table_row(
            row_id=f"E{index}",
            table_name=table_name,
            claim=claim,
            reference=reference,
        )
        tables.setdefault(table_name, []).append(row)
        all_rows.append(row)
    return {
        "schema_version": "1.0",
        "artifact_type": "writing_evidence_tables",
        "source_report": payload.get("source_report", {}),
        "columns": list(EVIDENCE_TABLE_COLUMNS),
        "counts": {
            table_name: len(rows)
            for table_name, rows in tables.items()
        },
        "tables": tables,
        "all_rows": all_rows,
    }


def validate_citation_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    paragraph_audit = _paragraph_claim_audit(payload.get("synthesis_blocks", []))
    unsupported_paragraphs = [
        entry for entry in paragraph_audit if entry["support_status"] != "SUPPORTED"
    ]
    uncited_blocks = [
        {
            "section": entry["section"],
            "evidence_type": entry["evidence_type"],
            "paragraph": entry["paragraph"],
            "reason": entry["reason"],
        }
        for entry in unsupported_paragraphs
    ]
    if unsupported_paragraphs:
        return {
            "status": "fail",
            "rule": "Every generated synthesis paragraph must contain known page citations from extracted evidence.",
            "audited_paragraph_count": len(paragraph_audit),
            "unsupported_paragraphs": unsupported_paragraphs,
            "uncited_blocks": uncited_blocks,
        }
    return {
        "status": "pass",
        "rule": "Every generated synthesis paragraph contains known page citations from extracted evidence.",
        "audited_paragraph_count": len(paragraph_audit),
        "unsupported_paragraphs": [],
        "uncited_blocks": [],
    }


def render_writing_markdown(payload: dict[str, Any]) -> str:
    mode = payload["mode"]
    if mode == "claim-table":
        return _render_claim_table(payload)
    if mode == "outline":
        return _render_outline(payload)
    if mode == "literature-review":
        return _render_literature_review(payload)
    if mode == "limitations":
        return _render_limitations(payload)
    if mode == "background":
        return _render_background(payload)
    if mode == "methods-summary":
        return _render_synthesis_mode(payload, title="# Evidence-Bound Methods Summary", evidence_types=("method",))
    if mode == "results-summary":
        return _render_synthesis_mode(payload, title="# Evidence-Bound Results Summary", evidence_types=("result",))
    if mode == "research-gaps":
        return _render_research_gaps(payload)
    raise ValueError(f"unsupported writing mode: {mode}")


def _render_claim_table(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Bound Claim Table",
        "",
        _source_line(payload),
        "",
        "| Claim ID | Status | Type | Citation | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    if payload["claims"]:
        for claim in payload["claims"]:
            lines.append(
                "| "
                f"{_cell(claim['claim_id'])} | "
                f"{_cell(claim['support_status'])} | "
                f"{_cell(claim['evidence_type'])} | "
                f"{_cell(claim['citation'])} | "
                f"{_cell(claim['text'])} |"
            )
    else:
        lines.append("| MATERIAL GAP | NO_SUPPORTED_CLAIMS |  |  | No page-anchored supported claims are available. |")
    lines.extend(_render_gap_lines(payload))
    return "\n".join(lines)


def _render_outline(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Bound Outline",
        "",
        _source_line(payload),
        "",
        "This outline uses only page-anchored extracted evidence.",
        "",
        "## Sections",
        "",
    ]
    section_order = ("method", "result", "dataset_population", "limitation", "claim")
    for evidence_type in section_order:
        claims = payload["claims_by_type"].get(evidence_type, [])
        if not claims:
            continue
        lines.append(f"### {TYPE_LABELS[evidence_type]}")
        for claim in claims:
            lines.append(f"- {_sentence(claim)}")
        lines.append("")
    if not payload["claims"]:
        lines.append("- MATERIAL GAP: No page-anchored extracted evidence is available for an evidence-bound outline.")
    lines.extend(_render_gap_lines(payload))
    return "\n".join(lines).rstrip()


def _render_literature_review(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Bound Literature Review Draft",
        "",
        _source_line(payload),
        "",
        "This draft uses only page-anchored extracted evidence.",
        "",
    ]
    if payload["claims"]:
        for evidence_type in ("method", "result", "dataset_population", "limitation", "claim"):
            for claim in payload["claims_by_type"].get(evidence_type, []):
                label = TYPE_LABELS[evidence_type]
                lines.append(f"{label} evidence: {claim['text']} [{claim['citation']}]")
                lines.append("")
    else:
        lines.append("MATERIAL GAP: No page-anchored extracted evidence is available for a literature-review draft.")
        lines.append("")
    lines.extend(_render_gap_lines(payload))
    return "\n".join(lines).rstrip()


def _render_limitations(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Bound Limitations",
        "",
        _source_line(payload),
        "",
    ]
    limitation_claims = payload["claims_by_type"].get("limitation", [])
    if limitation_claims:
        for claim in limitation_claims:
            lines.append(f"- {_sentence(claim)}")
    else:
        lines.append("- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.")
    lines.extend(_render_gap_lines(payload))
    return "\n".join(lines).rstrip()


def _render_background(payload: dict[str, Any]) -> str:
    return _render_synthesis_mode(
        payload,
        title="# Evidence-Bound Background",
        evidence_types=("method", "result", "dataset_population", "claim"),
        intro="This background section is assembled only from page-anchored scanner evidence.",
    )


def _render_synthesis_mode(
    payload: dict[str, Any],
    *,
    title: str,
    evidence_types: tuple[str, ...],
    intro: str | None = None,
) -> str:
    lines = [
        title,
        "",
        _source_line(payload),
        "",
    ]
    if intro:
        lines.extend([intro, ""])
    blocks = [
        block
        for block in payload["synthesis_blocks"]
        if block["evidence_type"] in evidence_types
    ]
    if blocks:
        for block in blocks:
            for entry in _paragraph_claim_audit([block]):
                if entry["support_status"] == "SUPPORTED":
                    lines.append(entry["paragraph"])
                else:
                    lines.append(
                        "MATERIAL GAP: Unsupported synthesis paragraph blocked in "
                        f"{entry['section']}: {entry['reason'].replace('_', ' ')}."
                    )
                lines.append("")
    else:
        labels = ", ".join(TYPE_LABELS[evidence_type].lower() for evidence_type in evidence_types)
        lines.append(f"MATERIAL GAP: No page-anchored {labels} evidence is available for controlled synthesis.")
        lines.append("")
    lines.extend(_render_paper_reference_lines(payload))
    lines.extend(_render_gap_lines(payload))
    lines.extend(_render_citation_check_lines(payload))
    return "\n".join(lines).rstrip()


def _render_research_gaps(payload: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Bound Research Gaps",
        "",
        _source_line(payload),
        "",
    ]
    if not payload["claims_by_type"].get("limitation"):
        lines.append("- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.")
    for gap in payload["material_gaps"]:
        lines.append(f"- MATERIAL GAP: {gap['message']}")
    if len(lines) == 4:
        lines.append("- No material gaps were detected by the current scanner report.")
    lines.extend(_render_paper_reference_lines(payload))
    return "\n".join(lines).rstrip()


def _claims_by_type(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {evidence_type: [] for evidence_type in TYPE_LABELS}
    for claim in claims:
        grouped.setdefault(claim["evidence_type"], []).append(claim)
    return grouped


def _paper_references(cited: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_counts: dict[str, int] = {}
    for claim in claims:
        evidence_counts[claim["paper"]] = evidence_counts.get(claim["paper"], 0) + 1
    references = []
    for reference in cited.get("paper_references") or []:
        references.append(
            {
                "label": reference["label"],
                "title": reference.get("title") or reference.get("source"),
                "year": reference.get("year"),
                "journal": reference.get("journal"),
                "doi": reference.get("doi"),
                "source": reference.get("source"),
                "evidence_count": evidence_counts.get(reference["label"], 0),
            }
        )
    return references


def _evidence_clusters(claims_by_type: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    clusters = []
    for evidence_type in ("method", "result", "dataset_population", "limitation", "claim"):
        claims = claims_by_type.get(evidence_type, [])
        if not claims:
            continue
        papers = sorted({claim["paper"] for claim in claims})
        clusters.append(
            {
                "topic": evidence_type.replace("_", " "),
                "evidence_type": evidence_type,
                "label": TYPE_LABELS[evidence_type],
                "claim_count": len(claims),
                "paper_count": len(papers),
                "papers": papers,
                "citations": [claim["citation"] for claim in claims],
                "claims": claims,
            }
        )
    return clusters


def _synthesis_blocks(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": f"S{index}",
            "section": cluster["label"],
            "evidence_type": cluster["evidence_type"],
            "paragraph": _synthesis_paragraph(cluster),
            "citations": cluster["citations"],
            "status": "SUPPORTED",
        }
        for index, cluster in enumerate(clusters, start=1)
    ]


def _synthesis_paragraph(cluster: dict[str, Any]) -> str:
    paper_label = "paper" if cluster["paper_count"] == 1 else "papers"
    snippets = "; ".join(claim["text"] for claim in cluster["claims"][:3])
    return (
        f"Across {cluster['paper_count']} {paper_label}, {cluster['evidence_type']} evidence includes "
        f"{snippets} [{'; '.join(cluster['citations'])}]."
    )


def _audit_summary_entry(entry: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "paragraph_id": entry["paragraph_id"],
        "block_id": entry["block_id"],
        "section": entry["section"],
        "evidence_type": entry["evidence_type"],
        "support_status": entry["support_status"],
        "reason": entry["reason"],
        "paragraph": entry["paragraph"],
        "citations": entry["citations"],
        "evidence_count": entry["evidence_count"],
    }
    if entry["support_status"] != "SUPPORTED":
        summary["available_citations"] = entry["available_citations"]
        summary["unknown_citations"] = entry["unknown_citations"]
    return summary


def _evidence_table_row(
    *,
    row_id: str,
    table_name: str,
    claim: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "claim_id": claim.get("claim_id"),
        "support_status": claim.get("support_status"),
        "table": table_name,
        "evidence_type": claim.get("evidence_type"),
        "paper": claim.get("paper"),
        "citation": claim.get("citation"),
        "page_number": claim.get("page_number"),
        "text": claim.get("text"),
        "paper_title": reference.get("title"),
        "year": reference.get("year"),
        "journal": reference.get("journal"),
        "doi": reference.get("doi"),
        "source": reference.get("source"),
    }


def _executive_summary_lines(payload: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    source = payload.get("source_report", {})
    claims = payload.get("claims", [])
    references = payload.get("paper_references", [])
    lines = [
        (
            f"- Jarvis screened {source.get('screened_count', 0) or 0} records, "
            f"blocked {source.get('blocked_count', 0) or 0}, and deep-read "
            f"{source.get('deep_read_count', 0) or 0} safe scholarly PDFs."
        ),
        (
            f"- The report uses {len(claims)} page-anchored evidence rows from "
            f"{len(references)} parsed paper references."
        ),
        (
            f"- Citation audit status: {audit.get('citation_check_status', 'unknown')}; "
            f"{audit.get('material_gap_count', 0)} material gaps recorded."
        ),
    ]
    top_result = _first_claim(payload.get("claims_by_type", {}).get("result", []))
    if top_result:
        lines.append(f"- Most direct result evidence: {top_result['text']} [{top_result['citation']}].")
    elif not claims:
        lines.append("- MATERIAL GAP: no page-anchored evidence was available for synthesis.")
    return lines


def _report_claim_lines(claims: list[dict[str, Any]], *, empty: str) -> list[str]:
    selected = _select_report_claims(claims)
    if not selected:
        return [f"- MATERIAL GAP: {empty}"] if empty else []
    return [f"- {claim['text']} [{claim['citation']}]" for claim in selected]


def _select_report_claims(claims: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    clean = [claim for claim in claims if _claim_text_quality(str(claim.get("text", ""))) >= 0]
    clean.sort(key=lambda claim: _claim_text_quality(str(claim.get("text", ""))), reverse=True)
    return clean[:limit]


def _first_claim(claims: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _select_report_claims(claims, limit=1)
    return selected[0] if selected else None


def _claim_text_quality(text: str) -> int:
    words = text.split()
    if len(words) < 6:
        return -1
    penalty = 0
    if text.count("[") != text.count("]"):
        penalty += 2
    if sum(1 for char in text if char.isupper()) > max(12, len(text) // 3):
        penalty += 1
    reward = min(len(words), 40)
    if any(token in text.lower() for token in ("achieved", "show", "identified", "agreement", "sensitivity")):
        reward += 6
    return reward - penalty


def _literature_csv_text(references: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = ("paper", "title", "year", "journal", "doi", "source", "evidence_count")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for reference in references:
        writer.writerow(
            {
                "paper": reference.get("label") or "",
                "title": reference.get("title") or "",
                "year": reference.get("year") or "",
                "journal": reference.get("journal") or "",
                "doi": reference.get("doi") or "",
                "source": reference.get("source") or "",
                "evidence_count": reference.get("evidence_count") or 0,
            }
        )
    return output.getvalue()


def render_report_pdf_bytes(markdown: str) -> bytes:
    lines = _pdf_lines_from_markdown(markdown)
    page_chunks = [lines[index : index + 42] for index in range(0, len(lines), 42)] or [[]]
    objects: list[bytes] = []
    catalog_id = 1
    pages_id = 2
    font_id = 3
    page_ids = []
    for page_index, page_lines in enumerate(page_chunks):
        page_id = 4 + page_index * 2
        content_id = page_id + 1
        page_ids.append(page_id)
        stream = _pdf_page_stream(page_lines)
        objects.append(
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>\n"
            "endobj\n".encode("latin-1")
        )
        objects.append(
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream\nendobj\n"
        )
    first_objects = [
        f"{catalog_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n".encode("latin-1"),
        (
            f"{pages_id} 0 obj\n<< /Type /Pages /Kids "
            f"[{' '.join(f'{page_id} 0 R' for page_id in page_ids)}] /Count {len(page_ids)} >>\nendobj\n"
        ).encode("latin-1"),
        f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin-1"),
    ]
    all_objects = first_objects + objects
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in all_objects:
        offsets.append(len(content))
        content.extend(obj)
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(all_objects) + 1}\n".encode("latin-1"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    content.extend(
        (
            f"trailer\n<< /Size {len(all_objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(content)


def _pdf_lines_from_markdown(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        text = raw_line.strip()
        if not text:
            lines.append("")
            continue
        text = re.sub(r"^#+\s*", "", text)
        text = text.replace("`", "")
        lines.extend(_wrap_pdf_line(text, width=86))
    return lines


def _wrap_pdf_line(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}"
    lines.append(current)
    return lines


def _pdf_page_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        parts.append(f"({_pdf_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def _pdf_escape(text: str) -> str:
    encoded = text.encode("latin-1", errors="replace").decode("latin-1")
    return encoded.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _empty_screening_label_summary() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "screening_label_summary",
        "counts": {
            "relevant": 0,
            "maybe": 0,
            "irrelevant": 0,
        },
        "labeled_count": 0,
        "rules": {
            "relevant": "Prioritized for resumed deep reads and allowed to bypass the minimum relevance threshold.",
            "maybe": "Kept eligible after relevant labels and before unlabeled papers with the same score.",
            "irrelevant": "Excluded from resumed deep reads.",
        },
        "labels": [],
    }


def _csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=EVIDENCE_TABLE_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: "" if row.get(column) is None else row.get(column)
                for column in EVIDENCE_TABLE_COLUMNS
            }
        )
    return output.getvalue()


def _paragraph_claim_audit(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit = []
    for block_index, block in enumerate(blocks, start=1):
        block_id = block.get("block_id") or f"S{block_index}"
        expected_citations = _normal_citations(block.get("citations") or [])
        paragraphs = _split_paragraphs(block.get("paragraph", ""))
        if not paragraphs:
            paragraphs = [""]
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            observed_citations = _page_citations_in_text(paragraph)
            unknown_citations = [
                citation for citation in observed_citations if citation not in expected_citations
            ]
            support_status, reason = _paragraph_support_status(
                block=block,
                paragraph=paragraph,
                expected_citations=expected_citations,
                observed_citations=observed_citations,
                unknown_citations=unknown_citations,
            )
            audit.append(
                {
                    "paragraph_id": f"{block_id}.{paragraph_index}",
                    "block_id": block_id,
                    "section": block.get("section"),
                    "evidence_type": block.get("evidence_type"),
                    "support_status": support_status,
                    "reason": reason,
                    "paragraph": paragraph,
                    "citations": observed_citations,
                    "available_citations": expected_citations,
                    "unknown_citations": unknown_citations,
                    "evidence_count": len(observed_citations) if support_status == "SUPPORTED" else 0,
                }
            )
    return audit


def _paragraph_support_status(
    *,
    block: dict[str, Any],
    paragraph: str,
    expected_citations: list[str],
    observed_citations: list[str],
    unknown_citations: list[str],
) -> tuple[str, str]:
    if block.get("status") != "SUPPORTED":
        return "MATERIAL_GAP", "unsupported_block_status"
    if not paragraph.strip():
        return "MATERIAL_GAP", "empty_paragraph"
    if not expected_citations:
        return "MATERIAL_GAP", "no_page_anchored_evidence"
    if not observed_citations:
        return "MATERIAL_GAP", "missing_page_citation"
    if unknown_citations:
        return "MATERIAL_GAP", "unknown_page_citation"
    return "SUPPORTED", "page_anchored"


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _normal_citations(citations: list[object]) -> list[str]:
    normalized = []
    for citation in citations:
        text = " ".join(str(citation).split())
        if text:
            normalized.append(text)
    return normalized


def _material_gaps(audit: dict[str, Any], cited: dict[str, Any]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for gap in audit.get("material_gaps") or []:
        gaps.append({"reason": gap.get("reason", "material_gap"), "message": gap.get("message", "")})
    for message in cited.get("evidence_gaps") or []:
        gaps.append({"reason": "evidence_gap", "message": str(message)})
    return _dedupe_gaps(gaps)


def _render_gap_lines(payload: dict[str, Any]) -> list[str]:
    if not payload["material_gaps"]:
        return []
    lines = ["", "## Material Gaps", ""]
    for gap in payload["material_gaps"]:
        lines.append(f"- MATERIAL GAP: {gap['message']}")
    return lines


def _render_paper_reference_lines(payload: dict[str, Any]) -> list[str]:
    if not payload["paper_references"]:
        return []
    lines = [
        "",
        "## Paper References",
        "",
        "| Paper | Title | Year | Venue | DOI | Evidence Count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for reference in payload["paper_references"]:
        lines.append(
            "| "
            f"{_cell(reference['label'])} | "
            f"{_cell(reference['title'] or '')} | "
            f"{_cell(reference['year'] or '')} | "
            f"{_cell(reference['journal'] or '')} | "
            f"{_cell(reference['doi'] or '')} | "
            f"{_cell(reference['evidence_count'])} |"
        )
    return lines


def _render_citation_check_lines(payload: dict[str, Any]) -> list[str]:
    check = validate_citation_coverage(payload)
    if check["status"] == "pass":
        return ["", "Citation check: pass."]
    lines = ["", "Citation check: fail."]
    for paragraph in check["unsupported_paragraphs"]:
        lines.append(
            "- Unsupported synthesis paragraph blocked in "
            f"{paragraph['section']}: {paragraph['reason'].replace('_', ' ')}."
        )
    return lines


def _source_line(payload: dict[str, Any]) -> str:
    source = payload["source_report"]
    parts = []
    if source.get("batch_id"):
        parts.append(f"Batch `{source['batch_id']}`")
    if source.get("query"):
        parts.append(f"query `{source['query']}`")
    if not parts:
        return "Source: scanner report JSON"
    return "Source: " + "; ".join(parts)


def _sentence(claim: dict[str, Any]) -> str:
    return f"{claim['text']} [{claim['citation']}]"


def _cell(value: object) -> str:
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def _dedupe_gaps(gaps: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = []
    seen = set()
    for gap in gaps:
        message = gap["message"]
        if not message or message in seen:
            continue
        seen.add(message)
        unique.append(gap)
    return unique


def _has_page_citation(text: str) -> bool:
    return bool(_page_citations_in_text(text))


def _page_citations_in_text(text: str) -> list[str]:
    citations = []
    for bracketed in re.findall(r"\[([^\]]+)\]", text):
        for match in re.finditer(r"\bP\d+\s+p\d+\b", bracketed):
            citation = " ".join(match.group(0).split())
            if citation not in citations:
                citations.append(citation)
    return citations
