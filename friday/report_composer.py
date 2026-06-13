from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from friday.compose_agent import (
    ComposePackageError,
    build_compose_package_files,
    build_llm_compose_package_files,
    load_writing_package,
)
from friday.writing_copilot import render_report_pdf_bytes


REPORT_SECTIONS = ("background", "methods", "results", "limitations")


def build_full_report_package_files(
    package_dir: Path,
    *,
    router: Any | None = None,
    use_llm: bool = False,
) -> dict[str, str | bytes]:
    if use_llm and router is None:
        raise ComposePackageError("LLM full report compose requires a configured router.")
    package = load_writing_package(package_dir)
    sections = _compose_sections(package_dir, router=router, use_llm=use_llm)
    report_markdown = render_full_report_markdown(package, sections)
    citation_audit = build_full_report_citation_audit(report_markdown, sections)
    files: dict[str, str | bytes] = {}
    for section, section_files in sections.items():
        for filename, content in section_files.items():
            files[f"sections/{section}/{filename}"] = content
    files.update(
        {
            "report.md": report_markdown.rstrip() + "\n",
            "report.pdf": render_report_pdf_bytes(report_markdown),
            "citation_audit.json": _json_text(citation_audit),
            "report_manifest.json": _json_text(_report_manifest(package, sections, citation_audit)),
            "evidence_table.md": _evidence_table_markdown(package),
            "evidence_table.csv": _evidence_table_csv(package),
            "literature_table.md": _literature_table_markdown(package),
            "literature_table.csv": _literature_table_csv(package),
            "paper_references.json": _json_text(package.get("paper_references.json", [])),
            "source_report.json": _json_text(package.get("source_report.json", {})),
            "material_gaps.json": _json_text(package.get("material_gaps.json", [])),
        }
    )
    return files


def render_full_report_markdown(package: dict[str, Any], sections: dict[str, dict[str, str]]) -> str:
    source = package.get("source_report.json", {})
    bodies = {section: _section_body(section, files) for section, files in sections.items()}
    lines = [
        "# Friday Research Report",
        "",
        _source_line(source),
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(_executive_summary_lines(bodies))
    for title, section in (
        ("Background", "background"),
        ("Methods", "methods"),
        ("Results", "results"),
        ("Limitations", "limitations"),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(bodies[section])
    lines.extend(["", "## Evidence Table", ""])
    lines.extend(_evidence_table_markdown(package).splitlines())
    lines.extend(["", "## Literature", ""])
    lines.extend(_literature_table_markdown(package).splitlines())
    lines.extend(["", "## Citation Audit", ""])
    audit = build_full_report_citation_audit("\n".join(lines), sections)
    lines.extend(
        [
            f"- Status: {audit['status']}",
            f"- Used citations: {len(audit['used_citations'])}",
            f"- Unknown citations: {len(audit['unknown_citations'])}",
        ]
    )
    return "\n".join(lines).rstrip()


def build_full_report_citation_audit(
    report_markdown: str,
    sections: dict[str, dict[str, str]],
) -> dict[str, Any]:
    section_audits = {
        section: _section_audit(section_files)
        for section, section_files in sections.items()
    }
    known = _ordered_unique(
        citation
        for audit in section_audits.values()
        for citation in audit["required_citations"]
    )
    used = _extract_citations(report_markdown)
    unknown = [citation for citation in used if citation not in set(known)]
    section_statuses = [audit["status"] for audit in section_audits.values()]
    return {
        "schema_version": "1.0",
        "artifact_type": "full_report_citation_audit",
        "status": "pass" if not unknown and all(status in {"pass", "material_gap", "fallback"} for status in section_statuses) else "fallback",
        "used_citations": used,
        "required_citations": known,
        "unknown_citations": unknown,
        "sections": section_audits,
    }


def _compose_sections(
    package_dir: Path,
    *,
    router: Any | None,
    use_llm: bool,
) -> dict[str, dict[str, str]]:
    sections = {}
    for section in REPORT_SECTIONS:
        if use_llm:
            sections[section] = build_llm_compose_package_files(package_dir, section=section, router=router)
        else:
            sections[section] = build_compose_package_files(package_dir, section=section)
    return sections


def _section_body(section: str, section_files: dict[str, str]) -> list[str]:
    if section == "background":
        return _background_body(section_files)
    draft_markdown = section_files.get("draft.md", "")
    duplicate_heading = {
        "methods": "## Methods",
        "results": "## Results",
        "limitations": "## Limitations",
    }.get(section)
    lines = []
    skip_rest = False
    for raw_line in draft_markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("# "):
            continue
        if duplicate_heading and stripped == duplicate_heading:
            continue
        if stripped.startswith("## "):
            line = "### " + stripped.removeprefix("## ").strip()
        if stripped.startswith("Source:"):
            continue
        if stripped == "This draft uses only paragraphs marked SUPPORTED in the writing package audit.":
            continue
        if stripped.startswith("Claim audit:"):
            continue
        if stripped in {"## Paper References", "## Conflicts Requiring Review"}:
            skip_rest = True
            continue
        if stripped == "## Material Gaps" and section != "limitations":
            skip_rest = True
            continue
        if skip_rest:
            continue
        lines.append(line)
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return [f"- MATERIAL GAP: No composed {section} section was available."]
    return lines


def _background_body(section_files: dict[str, str]) -> list[str]:
    used = _load_json(section_files.get("used_evidence.json"))
    entries = [
        entry
        for entry in used.get("used_evidence", [])
        if isinstance(entry, dict) and str(entry.get("evidence_type") or "") in {"claim", "dataset_population"}
    ]
    if not entries:
        return ["- MATERIAL GAP: No dedicated background evidence was available in this writing package."]
    lines = []
    current_group = ""
    for entry in entries:
        group = str(entry.get("group_label") or _background_group_label(entry)).strip()
        if group != current_group:
            if lines:
                lines.append("")
            lines.extend([f"### {group}", ""])
            current_group = group
        paragraph = str(entry.get("paragraph") or "").strip()
        if paragraph:
            lines.extend([paragraph, ""])
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _background_group_label(entry: dict[str, Any]) -> str:
    evidence_type = str(entry.get("evidence_type") or "")
    if evidence_type == "dataset_population":
        return "Dataset and population"
    return "Claims"


def _executive_summary_lines(bodies: dict[str, list[str]]) -> list[str]:
    bullets = []
    for section in ("background", "methods", "results", "limitations"):
        sentence = _first_cited_or_gap_line(bodies.get(section, []))
        if sentence:
            bullets.append(f"- {sentence}")
    return bullets or ["- MATERIAL GAP: No cited section evidence was available for summary."]


def _first_cited_or_gap_line(lines: list[str]) -> str:
    for line in lines:
        stripped = line.strip().lstrip("- ").strip()
        if not stripped or stripped.startswith("##"):
            continue
        if "MATERIAL GAP:" in stripped or _extract_citations(stripped):
            return stripped
    return ""


def _section_audit(section_files: dict[str, str]) -> dict[str, Any]:
    claim = _load_json(section_files.get("claim_audit.json"))
    composer = _load_json(section_files.get("composer_audit.json"))
    verifier = _load_json(section_files.get("verifier_audit.json"))
    required = _ordered_unique(
        [
            *_string_list(composer.get("required_citations")),
            *_string_list(verifier.get("required_citations")),
            *[
                citation
                for paragraph in claim.get("paragraphs", [])
                for citation in _string_list(paragraph.get("citations"))
            ],
        ]
    )
    claim_status = str(claim.get("status") or "")
    composer_status = str(composer.get("status") or "") or None
    verifier_status = str(verifier.get("status") or "") or None
    return {
        "status": verifier_status or composer_status or claim_status or "unknown",
        "claim_audit_status": claim_status or None,
        "composer_status": composer_status,
        "verifier_status": verifier_status,
        "required_citations": required,
        "used_citations": _extract_citations(section_files.get("draft.md", "")),
    }


def _report_manifest(
    package: dict[str, Any],
    sections: dict[str, dict[str, str]],
    citation_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "full_report_manifest",
        "source_report": package.get("source_report.json", {}),
        "sections": {
            section: {
                "files": sorted(section_files),
                "draft_citations": _extract_citations(section_files.get("draft.md", "")),
            }
            for section, section_files in sections.items()
        },
        "citation_audit_status": citation_audit.get("status"),
    }


def _evidence_table_markdown(package: dict[str, Any]) -> str:
    rows = _evidence_rows(package)
    lines = ["| Section | Evidence | Citations |", "| --- | --- | --- |"]
    for row in rows[:30]:
        lines.append(
            f"| {_markdown_cell(row['section'])} | {_markdown_cell(row['evidence'])} | {_markdown_cell('; '.join(row['citations']))} |"
        )
    if not rows:
        lines.append("| - | No supported evidence rows were available. | - |")
    return "\n".join(lines) + "\n"


def _literature_table_markdown(package: dict[str, Any]) -> str:
    references = package.get("paper_references.json", [])
    lines = ["| Paper | Title | Year | Venue | DOI |", "| --- | --- | --- | --- | --- |"]
    for reference in references:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(str(reference.get(key) or ""))
                for key in ("label", "title", "year", "journal", "doi")
            )
            + " |"
        )
    if not references:
        lines.append("| - | No paper references were available. | - | - | - |")
    return "\n".join(lines) + "\n"


def _evidence_table_csv(package: dict[str, Any]) -> str:
    rows = _evidence_rows(package)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("section", "evidence", "citations"), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"section": row["section"], "evidence": row["evidence"], "citations": "; ".join(row["citations"])})
    return output.getvalue()


def _literature_table_csv(package: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("paper", "title", "year", "journal", "doi"), lineterminator="\n")
    writer.writeheader()
    for reference in package.get("paper_references.json", []):
        writer.writerow(
            {
                "paper": reference.get("label") or "",
                "title": reference.get("title") or "",
                "year": reference.get("year") or "",
                "journal": reference.get("journal") or "",
                "doi": reference.get("doi") or "",
            }
        )
    return output.getvalue()


def _evidence_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for paragraph in package.get("supported_paragraphs.json", []):
        rows.append(
            {
                "section": str(paragraph.get("section") or paragraph.get("evidence_type") or ""),
                "evidence": str(paragraph.get("paragraph") or ""),
                "citations": _string_list(paragraph.get("citations")),
            }
        )
    return rows


def _source_line(source_report: dict[str, Any]) -> str:
    return (
        f"Source: Batch `{source_report.get('batch_id', '')}`; "
        f"query `{source_report.get('query', '')}`; "
        f"screened `{source_report.get('screened_count', 0)}`; "
        f"deep-read `{source_report.get('deep_read_count', 0)}`"
    )


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_citations(text: str) -> list[str]:
    citations = []
    for bracket in re.findall(r"\[([^\]]+)\]", text):
        for part in bracket.split(";"):
            citation = " ".join(part.strip().split())
            if re.fullmatch(r"P\d+\s+p\d+", citation):
                citations.append(citation)
    return _ordered_unique(citations)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ordered_unique(values: list[str] | Any) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _markdown_cell(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
