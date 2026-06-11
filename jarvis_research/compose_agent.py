from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SECTION_CHOICES = ("background", "methods", "results", "limitations", "all")

REQUIRED_PACKAGE_FILES = (
    "supported_paragraphs.json",
    "blocked_paragraphs.json",
    "material_gaps.json",
    "paper_references.json",
    "source_report.json",
)

SECTION_CONFIG = {
    "background": {
        "title": "# Evidence-Bound Background Draft",
        "label": "background",
        "evidence_types": ("method", "result", "dataset_population", "claim"),
    },
    "methods": {
        "title": "# Evidence-Bound Methods Draft",
        "label": "method",
        "evidence_types": ("method",),
    },
    "results": {
        "title": "# Evidence-Bound Results Draft",
        "label": "result",
        "evidence_types": ("result",),
    },
    "limitations": {
        "title": "# Evidence-Bound Limitations Draft",
        "label": "limitation",
        "evidence_types": ("limitation",),
    },
    "all": {
        "title": "# Evidence-Bound Composite Draft",
        "label": "section",
        "evidence_types": ("method", "result", "dataset_population", "limitation", "claim"),
    },
}


class ComposePackageError(ValueError):
    """Raised when a writing package cannot be safely composed."""


def build_compose_package_files(package_dir: Path, *, section: str) -> dict[str, str]:
    package = load_writing_package(package_dir)
    payload = build_compose_payload(package, section=section)
    return {
        "draft.md": payload["draft_markdown"].rstrip() + "\n",
        "outline.json": _json_text(payload["outline"]),
        "claim_audit.json": _json_text(payload["claim_audit"]),
        "used_evidence.json": _json_text(payload["used_evidence"]),
        "refused_claims.json": _json_text(payload["refused_claims"]),
        "conflicts.json": _json_text(payload["conflicts"]),
    }


def load_writing_package(package_dir: Path) -> dict[str, Any]:
    package: dict[str, Any] = {}
    for filename in REQUIRED_PACKAGE_FILES:
        path = package_dir / filename
        if not path.is_file():
            raise ComposePackageError(f"Missing writing package file: {filename}")
        try:
            package[filename] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComposePackageError(f"Invalid JSON in {filename}: {exc.msg}") from exc
    _require_type(package, "supported_paragraphs.json", list)
    _require_type(package, "blocked_paragraphs.json", list)
    _require_type(package, "material_gaps.json", list)
    _require_type(package, "paper_references.json", list)
    _require_type(package, "source_report.json", dict)
    evidence_tables_path = package_dir / "evidence_tables.json"
    if evidence_tables_path.is_file():
        try:
            package["evidence_tables.json"] = json.loads(evidence_tables_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ComposePackageError(f"Invalid JSON in evidence_tables.json: {exc.msg}") from exc
        _require_type(package, "evidence_tables.json", dict)
    return package


def build_compose_payload(package: dict[str, Any], *, section: str) -> dict[str, Any]:
    if section not in SECTION_CONFIG:
        raise ComposePackageError(f"Unsupported compose section: {section}")
    config = SECTION_CONFIG[section]
    evidence_types = tuple(config["evidence_types"])
    supported_entries = package["supported_paragraphs.json"]
    blocked_entries = package["blocked_paragraphs.json"]
    material_gaps = package["material_gaps.json"]
    paper_references = package["paper_references.json"]
    source_report = package["source_report.json"]
    table_rows_by_citation = _table_rows_by_citation(
        package.get("evidence_tables.json", {}),
        evidence_types=evidence_types,
    )

    used_evidence = [
        _with_table_rows(_used_entry(entry), table_rows_by_citation)
        for entry in supported_entries
        if _matches_section(entry, evidence_types) and _is_supported_paragraph(entry)
    ]
    refused = [
        _refused_entry(entry)
        for entry in blocked_entries
        if _matches_section(entry, evidence_types)
    ]
    refused.extend(
        _refused_entry(entry, reason="not_usable_supported_paragraph")
        for entry in supported_entries
        if _matches_section(entry, evidence_types) and not _is_supported_paragraph(entry)
    )
    if not used_evidence:
        refused.append(
            {
                "reason": "no_supported_section_evidence",
                "section": section,
                "evidence_type": ",".join(evidence_types),
                "message": f"No supported {config['label']} evidence is available in this writing package.",
            }
        )

    used_evidence, evidence_groups = _group_used_evidence(used_evidence)
    conflicts = _detect_conflicts(section, evidence_groups)
    outline = _build_outline(
        section=section,
        config=config,
        source_report=source_report,
        paper_references=paper_references,
        used_evidence=used_evidence,
        evidence_groups=evidence_groups,
        material_gaps=material_gaps,
    )
    claim_audit = _build_claim_audit(section, used_evidence, refused)
    payload = {
        "schema_version": "1.0",
        "artifact_type": "compose_agent_output",
        "section": section,
        "safety_policy": {
            "evidence_bound": True,
            "llm_used": False,
            "rule": "Use only paragraphs already marked SUPPORTED by the writing package audit.",
        },
        "outline": outline,
        "claim_audit": claim_audit,
        "used_evidence": {
            "schema_version": "1.0",
            "artifact_type": "compose_used_evidence",
            "section": section,
            "used_evidence": used_evidence,
        },
        "refused_claims": {
            "schema_version": "1.0",
            "artifact_type": "compose_refused_claims",
            "section": section,
            "refused_claims": refused,
        },
        "conflicts": conflicts,
    }
    payload["draft_markdown"] = _render_draft(
        config=config,
        source_report=source_report,
        used_evidence=used_evidence,
        evidence_groups=evidence_groups,
        material_gaps=material_gaps,
        paper_references=paper_references,
        audit=claim_audit,
        conflicts=conflicts,
    )
    return payload


def _build_outline(
    *,
    section: str,
    config: dict[str, Any],
    source_report: dict[str, Any],
    paper_references: list[dict[str, Any]],
    used_evidence: list[dict[str, Any]],
    evidence_groups: list[dict[str, Any]],
    material_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "compose_outline",
        "section": section,
        "title": config["title"].lstrip("# "),
        "evidence_types": list(config["evidence_types"]),
        "source_report": source_report,
        "paper_reference_count": len(paper_references),
        "material_gap_count": len(material_gaps),
        "groups": [
            {
                "group_id": group["group_id"],
                "group_label": group["group_label"],
                "evidence_type": group["evidence_type"],
                "paragraph_count": group["paragraph_count"],
                "citation_count": group["citation_count"],
                "citations": group["citations"],
                "papers": group["papers"],
                "source_paragraph_ids": group["source_paragraph_ids"],
            }
            for group in evidence_groups
        ],
        "items": [
            {
                "outline_id": f"O{index}",
                "source_paragraph_id": entry["source_paragraph_id"],
                "group_id": entry["group_id"],
                "group_label": entry["group_label"],
                "table_row_ids": entry["table_row_ids"],
                "evidence_type": entry["evidence_type"],
                "citations": entry["citations"],
                "paragraph": entry["paragraph"],
            }
            for index, entry in enumerate(used_evidence, start=1)
        ],
    }


def _build_claim_audit(
    section: str,
    used_evidence: list[dict[str, Any]],
    refused: list[dict[str, Any]],
) -> dict[str, Any]:
    paragraphs = [
        {
            "compose_paragraph_id": f"C{index}",
            "source_paragraph_id": entry["source_paragraph_id"],
            "support_status": "SUPPORTED",
            "reason": "page_anchored",
            "evidence_type": entry["evidence_type"],
            "paragraph": entry["paragraph"],
            "citations": entry["citations"],
            "table_row_ids": entry["table_row_ids"],
            "evidence_count": len(entry["citations"]),
        }
        for index, entry in enumerate(used_evidence, start=1)
    ]
    return {
        "schema_version": "1.0",
        "artifact_type": "compose_claim_audit",
        "section": section,
        "status": "pass" if paragraphs else "material_gap",
        "audited_paragraph_count": len(paragraphs),
        "supported_paragraph_count": len(paragraphs),
        "refused_claim_count": len(refused),
        "paragraphs": paragraphs,
    }


def _render_draft(
    *,
    config: dict[str, Any],
    source_report: dict[str, Any],
    used_evidence: list[dict[str, Any]],
    evidence_groups: list[dict[str, Any]],
    material_gaps: list[dict[str, Any]],
    paper_references: list[dict[str, Any]],
    audit: dict[str, Any],
    conflicts: dict[str, Any],
) -> str:
    lines = [
        config["title"],
        "",
        _source_line(source_report),
        "",
        "This draft uses only paragraphs marked SUPPORTED in the writing package audit.",
        "",
    ]
    if evidence_groups:
        for group in evidence_groups:
            lines.append(f"## {group['group_label']}")
            lines.append("")
            for entry in group["paragraphs"]:
                lines.append(entry["paragraph"])
                lines.append("")
    else:
        lines.append(f"MATERIAL GAP: No supported {config['label']} evidence is available in this writing package.")
        lines.append("")

    if conflicts["conflicts"]:
        lines.extend(["## Conflicts Requiring Review", ""])
        for conflict in conflicts["conflicts"]:
            lines.append(
                "- "
                f"{conflict['group_label']}: {', '.join(conflict['stance_set'])} "
                f"evidence across [{'; '.join(conflict['citations'])}]."
            )
        lines.append("")

    if material_gaps:
        lines.extend(["## Material Gaps", ""])
        for gap in material_gaps:
            message = str(gap.get("message") or "").strip()
            if message:
                lines.append(f"- MATERIAL GAP: {message}")
        lines.append("")

    if paper_references:
        lines.extend(_render_paper_reference_lines(paper_references))
        lines.append("")

    lines.append(f"Claim audit: {audit['status']}.")
    return "\n".join(lines).rstrip()


def _used_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_paragraph_id": str(entry.get("paragraph_id") or ""),
        "source_block_id": str(entry.get("block_id") or ""),
        "section": entry.get("section"),
        "evidence_type": str(entry.get("evidence_type") or ""),
        "paragraph": str(entry.get("paragraph") or "").strip(),
        "citations": _normal_citations(entry.get("citations") or []),
        "evidence_count": int(entry.get("evidence_count") or len(entry.get("citations") or [])),
    }


def _with_table_rows(
    entry: dict[str, Any],
    table_rows_by_citation: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    table_rows = []
    for citation in entry["citations"]:
        table_rows.extend(table_rows_by_citation.get(citation, []))
    return {
        **entry,
        "table_rows": table_rows,
        "table_row_ids": [row["row_id"] for row in table_rows if row.get("row_id")],
    }


def _group_used_evidence(
    used_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for entry in used_evidence:
        group_label = _group_label(entry)
        group_key = group_label.casefold()
        labels[group_key] = group_label
        buckets.setdefault(group_key, []).append(dict(entry))

    raw_groups = [
        _group_entry(labels[group_key], entries)
        for group_key, entries in buckets.items()
    ]
    raw_groups.sort(
        key=lambda group: (
            -group["paragraph_count"],
            -group["citation_count"],
            group["group_label"],
        )
    )

    grouped_evidence: list[dict[str, Any]] = []
    evidence_groups: list[dict[str, Any]] = []
    for index, group in enumerate(raw_groups, start=1):
        group_id = f"G{index}"
        paragraphs = []
        for entry in group["paragraphs"]:
            grouped = dict(entry)
            grouped["group_id"] = group_id
            grouped["group_label"] = group["group_label"]
            paragraphs.append(grouped)
            grouped_evidence.append(grouped)
        evidence_groups.append({**group, "group_id": group_id, "paragraphs": paragraphs})
    return grouped_evidence, evidence_groups


def _group_entry(group_label: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    citations = _ordered_unique(
        citation
        for entry in entries
        for citation in entry["citations"]
    )
    return {
        "group_id": "",
        "group_label": group_label,
        "evidence_type": entries[0]["evidence_type"] if entries else "",
        "paragraph_count": len(entries),
        "citation_count": len(citations),
        "citations": citations,
        "papers": _ordered_unique(_paper_label(citation) for citation in citations),
        "source_paragraph_ids": [entry["source_paragraph_id"] for entry in entries],
        "paragraphs": entries,
    }


def _detect_conflicts(section: str, evidence_groups: list[dict[str, Any]]) -> dict[str, Any]:
    conflicts = []
    for group in evidence_groups:
        stances: dict[str, list[dict[str, Any]]] = {}
        for entry in group["paragraphs"]:
            stance = _evidence_stance(entry["paragraph"])
            if stance != "neutral":
                stances.setdefault(stance, []).append(entry)
        if "positive" not in stances or "negative" not in stances:
            continue
        conflict_entries = [
            entry
            for entry in group["paragraphs"]
            if _evidence_stance(entry["paragraph"]) in {"negative", "positive"}
        ]
        conflicts.append(
            {
                "conflict_id": f"K{len(conflicts) + 1}",
                "group_id": group["group_id"],
                "group_label": group["group_label"],
                "evidence_type": group["evidence_type"],
                "reason": "mixed_directional_evidence",
                "stance_set": sorted(stances),
                "citations": _ordered_unique(
                    citation
                    for entry in conflict_entries
                    for citation in entry["citations"]
                ),
                "source_paragraph_ids": [entry["source_paragraph_id"] for entry in conflict_entries],
            }
        )
    return {
        "schema_version": "1.0",
        "artifact_type": "compose_conflicts",
        "section": section,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def _group_label(entry: dict[str, Any]) -> str:
    topic = str(entry.get("topic") or "").strip()
    if topic:
        return topic
    evidence_type = entry["evidence_type"]
    text = entry["paragraph"].casefold()
    if evidence_type == "result":
        if any(token in text for token in ("detect", "resistant-isolate", "resistant isolate", "resistance")):
            return "Resistance detection"
        if any(token in text for token in ("auroc", "auc", "accuracy", "sensitivity", "specificity", "performance")):
            return "Model performance"
        if any(token in text for token in ("susceptibility", "antibiotic", "antimicrobial")):
            return "Antimicrobial susceptibility"
        return "Result evidence"
    if evidence_type == "method":
        return "Methods"
    if evidence_type == "dataset_population":
        return "Dataset and population"
    if evidence_type == "limitation":
        return "Limitations"
    return "Claims"


def _evidence_stance(paragraph: str) -> str:
    text = paragraph.casefold()
    negative_phrases = (
        "no improvement",
        "not improve",
        "did not improve",
        "failed to improve",
        "decreased",
        "reduced",
        "lower",
        "worse",
    )
    positive_phrases = (
        "improved",
        "increased",
        "higher",
        "outperformed",
        "achieved",
        "detected",
    )
    if any(phrase in text for phrase in negative_phrases):
        return "negative"
    if any(phrase in text for phrase in positive_phrases):
        return "positive"
    return "neutral"


def _refused_entry(entry: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    return {
        "source_paragraph_id": str(entry.get("paragraph_id") or ""),
        "source_block_id": str(entry.get("block_id") or ""),
        "section": entry.get("section"),
        "evidence_type": str(entry.get("evidence_type") or ""),
        "support_status": str(entry.get("support_status") or "MATERIAL_GAP"),
        "reason": reason or str(entry.get("reason") or "material_gap"),
        "paragraph": str(entry.get("paragraph") or "").strip(),
        "citations": _normal_citations(entry.get("citations") or []),
        "evidence_count": int(entry.get("evidence_count") or 0),
    }


def _matches_section(entry: dict[str, Any], evidence_types: tuple[str, ...]) -> bool:
    return str(entry.get("evidence_type") or "") in evidence_types


def _is_supported_paragraph(entry: dict[str, Any]) -> bool:
    paragraph = str(entry.get("paragraph") or "").strip()
    citations = _normal_citations(entry.get("citations") or [])
    if entry.get("support_status") != "SUPPORTED":
        return False
    if not paragraph or not citations:
        return False
    return all(citation in paragraph for citation in citations)


def _normal_citations(citations: list[object]) -> list[str]:
    normalized = []
    for citation in citations:
        text = " ".join(str(citation).split())
        if text:
            normalized.append(text)
    return normalized


def _table_rows_by_citation(
    evidence_tables: dict[str, Any],
    *,
    evidence_types: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_citation: dict[str, list[dict[str, Any]]] = {}
    tables = evidence_tables.get("tables") if isinstance(evidence_tables, dict) else {}
    if not isinstance(tables, dict):
        return rows_by_citation
    for rows in tables.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("evidence_type") or "") not in evidence_types:
                continue
            citation = " ".join(str(row.get("citation") or "").split())
            if not citation:
                continue
            rows_by_citation.setdefault(citation, []).append(row)
    return rows_by_citation


def _ordered_unique(values) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _paper_label(citation: str) -> str:
    return citation.split(" ", 1)[0]


def _source_line(source_report: dict[str, Any]) -> str:
    parts = []
    if source_report.get("batch_id"):
        parts.append(f"Batch `{source_report['batch_id']}`")
    if source_report.get("query"):
        parts.append(f"query `{source_report['query']}`")
    if source_report.get("screened_count") is not None:
        parts.append(f"screened `{source_report['screened_count']}`")
    if source_report.get("deep_read_count") is not None:
        parts.append(f"deep-read `{source_report['deep_read_count']}`")
    return "Source: " + "; ".join(parts) if parts else "Source: writing package"


def _render_paper_reference_lines(paper_references: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Paper References",
        "",
        "| Paper | Title | Year | Venue | DOI | Evidence Count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for reference in paper_references:
        lines.append(
            "| "
            f"{_cell(reference.get('label') or '')} | "
            f"{_cell(reference.get('title') or '')} | "
            f"{_cell(reference.get('year') or '')} | "
            f"{_cell(reference.get('journal') or '')} | "
            f"{_cell(reference.get('doi') or '')} | "
            f"{_cell(reference.get('evidence_count') or 0)} |"
        )
    return lines


def _require_type(
    package: dict[str, Any],
    filename: str,
    expected_type: type,
) -> None:
    if not isinstance(package[filename], expected_type):
        raise ComposePackageError(f"{filename} must contain {expected_type.__name__}")


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _cell(value: object) -> str:
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")
