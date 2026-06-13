from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any

from friday.report_composer import (
    _allowed_material_gap_messages,
    _citation_sort_key,
    _citations_from_bracket_content,
    _extract_citations,
    _looks_like_factual_report_sentence,
    _sentence_evidence_support,
    _strip_page_citation_brackets,
    _strip_report_bullet_label,
)


@dataclass(frozen=True)
class ClaimUnit:
    claim_unit_id: str
    section: str
    claim_type: str
    text: str
    source_sentence: str
    citations: list[str]
    support_status: str
    evidence_count: int
    evidence_types: list[str]
    evidence_row_ids: list[str]
    min_quality_score: float | None
    min_parse_confidence: float | None
    min_trust_score: float | None
    support_details: dict[str, Any]


def build_report_claim_units(report_markdown: str, package: dict[str, Any]) -> dict[str, Any]:
    evidence_index = _evidence_text_index(package)
    evidence_metadata = _evidence_metadata_by_citation(package)
    allowed_gaps = _allowed_material_gap_messages(package)
    units = []
    for record in _report_sentence_records(report_markdown):
        sentence = record["sentence"]
        text = _claim_text(sentence)
        if not text:
            continue
        claim_type = _claim_type(sentence, record["section"])
        if claim_type == "structural":
            continue
        citations = sorted(_extract_citations(sentence), key=_citation_sort_key)
        support_status, support_details = _claim_support_status(
            sentence,
            citations,
            claim_type=claim_type,
            evidence_index=evidence_index,
            allowed_gaps=allowed_gaps,
        )
        if support_status == "non_factual":
            continue
        metadata = _merge_evidence_metadata(citations, evidence_metadata)
        units.append(
            ClaimUnit(
                claim_unit_id=f"C{len(units) + 1}",
                section=record["section"] or "unknown",
                claim_type=claim_type,
                text=text,
                source_sentence=sentence,
                citations=citations,
                support_status=support_status,
                evidence_count=metadata["evidence_count"],
                evidence_types=metadata["evidence_types"],
                evidence_row_ids=metadata["evidence_row_ids"],
                min_quality_score=metadata["min_quality_score"],
                min_parse_confidence=metadata["min_parse_confidence"],
                min_trust_score=metadata["min_trust_score"],
                support_details=support_details,
            )
        )
    counts_by_status = Counter(unit.support_status for unit in units)
    counts_by_type = Counter(unit.claim_type for unit in units)
    issue_count = sum(
        count
        for status, count in counts_by_status.items()
        if status in {"uncited", "unknown_citation", "weak_support"}
    )
    return {
        "schema_version": "1.0",
        "artifact_type": "report_claim_units",
        "source_report": package.get("source_report.json", {}),
        "status": "pass" if issue_count == 0 else "fallback",
        "claim_unit_count": len(units),
        "issue_count": issue_count,
        "counts": {
            "by_support_status": dict(sorted(counts_by_status.items())),
            "by_claim_type": dict(sorted(counts_by_type.items())),
        },
        "claim_units": [asdict(unit) for unit in units],
    }


def _report_sentence_records(report_markdown: str) -> list[dict[str, str]]:
    records = []
    section = ""
    in_skipped_section = False
    for raw_line in report_markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped == "---":
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
            in_skipped_section = section.casefold() in {"evidence table", "literature", "citation audit"}
            continue
        if in_skipped_section:
            continue
        if stripped.startswith("|") or re.fullmatch(r"[|:\-\s]+", stripped):
            continue
        normalized = " ".join(stripped.split())
        if not normalized:
            continue
        for sentence in _split_report_sentences(normalized):
            records.append({"section": section, "sentence": sentence})
    return records


def _split_report_sentences(line: str) -> list[str]:
    if _extract_citations(line) or line.startswith("- MATERIAL GAP:") or line.startswith("MATERIAL GAP:"):
        return [line]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z*-])", line)
    return [part.strip() for part in parts if part.strip()]


def _claim_text(sentence: str) -> str:
    text = _strip_report_bullet_label(sentence)
    if text.startswith("MATERIAL GAP:"):
        text = text.removeprefix("MATERIAL GAP:").strip()
    text = _strip_page_citation_brackets(text).strip()
    text = text.rstrip(".")
    return " ".join(text.split())


def _claim_type(sentence: str, section: str) -> str:
    text = _strip_report_bullet_label(sentence)
    lowered = text.casefold()
    section_key = section.casefold()
    if text.startswith("Source:"):
        return "structural"
    if text.startswith("MATERIAL GAP:") or _looks_like_material_gap(text):
        return "material_gap"
    citations = _extract_citations(sentence)
    if not citations and not _looks_like_factual_report_sentence(sentence):
        return "structural"
    if not citations:
        return "factual"
    if len(citations) > 1 or _has_synthesis_cue(text):
        return "synthesis"
    if "limitation" in section_key or re.search(r"\blimitations?\b", lowered):
        return "limitation"
    if "method" in section_key or re.search(r"\b(method|using|used|classifier|model|assay|protocol)\b", lowered):
        return "method"
    if "result" in section_key or re.search(r"\b(reported|found|showed|achieved|sensitivity|specificity|auc|auroc)\b", lowered):
        return "result"
    if citations:
        return "factual"
    return "factual"


def _has_synthesis_cue(text: str) -> bool:
    return bool(
        re.search(
            r"\b(across|both|combined|collectively|multiple|several|two|three|four|five|\d+\s+papers?)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_material_gap(text: str) -> bool:
    lowered = text.casefold()
    return lowered.startswith("no ") and "evidence" in lowered and "available" in lowered


def _claim_support_status(
    sentence: str,
    citations: list[str],
    *,
    claim_type: str,
    evidence_index: dict[str, str],
    allowed_gaps: set[str],
) -> tuple[str, dict[str, Any]]:
    text = _strip_report_bullet_label(sentence)
    if claim_type == "material_gap":
        gap_text = _claim_text(sentence)
        allowed = gap_text in allowed_gaps or _looks_like_material_gap(gap_text)
        return "material_gap", {"allowed": allowed}
    if not citations:
        if _looks_like_factual_report_sentence(sentence):
            return "uncited", {}
        return "non_factual", {}
    unknown = [citation for citation in citations if citation not in evidence_index]
    if unknown:
        return "unknown_citation", {"unknown_citations": unknown}
    support = _sentence_evidence_support(text, citations, evidence_index)
    if support["status"] == "pass":
        return "supported", support
    return "weak_support", support


def _evidence_text_index(package: dict[str, Any]) -> dict[str, str]:
    metadata = _evidence_metadata_by_citation(package)
    return {
        citation: " ".join(_ordered_unique(str(text) for text in values["texts"] if str(text).strip()))
        for citation, values in metadata.items()
    }


def _evidence_metadata_by_citation(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in _all_atomic_rows(package):
        if str(row.get("support_status") or "SUPPORTED") not in {"", "SUPPORTED"}:
            continue
        if str(row.get("quality_label") or "clean") == "blocked":
            continue
        citation = str(row.get("citation") or "").strip()
        text = str(row.get("text") or "").strip()
        if not citation or not text:
            continue
        _add_evidence_metadata(
            metadata,
            citation,
            text=text,
            evidence_type=str(row.get("evidence_type") or ""),
            row_id=str(row.get("row_id") or ""),
            quality_score=_optional_float(row.get("quality_score")),
            parse_confidence=_optional_float(row.get("parse_confidence")),
            trust_score=_optional_float(row.get("trust_score")),
        )
    for paragraph in package.get("supported_paragraphs.json", []):
        if not isinstance(paragraph, dict):
            continue
        if str(paragraph.get("support_status") or "SUPPORTED") not in {"", "SUPPORTED"}:
            continue
        text = str(paragraph.get("paragraph") or "").strip()
        if not text:
            continue
        for citation in _string_list(paragraph.get("citations")):
            _add_evidence_metadata(
                metadata,
                citation,
                text=text,
                evidence_type=str(paragraph.get("evidence_type") or paragraph.get("section") or ""),
                row_id=str(paragraph.get("paragraph_id") or ""),
                quality_score=None,
                parse_confidence=None,
                trust_score=None,
            )
    return metadata


def _add_evidence_metadata(
    metadata: dict[str, dict[str, Any]],
    citation: str,
    *,
    text: str,
    evidence_type: str,
    row_id: str,
    quality_score: float | None,
    parse_confidence: float | None,
    trust_score: float | None,
) -> None:
    citations = _citations_from_bracket_content(citation)
    if not citations and re.fullmatch(r"P\d+\s+p\d+", citation):
        citations = [citation]
    for normalized_citation in citations:
        entry = metadata.setdefault(
            normalized_citation,
            {
                "texts": [],
                "evidence_types": [],
                "evidence_row_ids": [],
                "quality_scores": [],
                "parse_confidences": [],
                "trust_scores": [],
            },
        )
        entry["texts"].append(text)
        if evidence_type:
            entry["evidence_types"].append(evidence_type)
        if row_id:
            entry["evidence_row_ids"].append(row_id)
        if quality_score is not None:
            entry["quality_scores"].append(quality_score)
        if parse_confidence is not None:
            entry["parse_confidences"].append(parse_confidence)
        if trust_score is not None:
            entry["trust_scores"].append(trust_score)


def _merge_evidence_metadata(citations: list[str], metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    texts = []
    evidence_types = []
    row_ids = []
    quality_scores = []
    parse_confidences = []
    trust_scores = []
    for citation in citations:
        entry = metadata.get(citation, {})
        texts.extend(entry.get("texts", []))
        evidence_types.extend(entry.get("evidence_types", []))
        row_ids.extend(entry.get("evidence_row_ids", []))
        quality_scores.extend(entry.get("quality_scores", []))
        parse_confidences.extend(entry.get("parse_confidences", []))
        trust_scores.extend(entry.get("trust_scores", []))
    return {
        "evidence_count": len(citations),
        "evidence_types": _ordered_unique(evidence_types),
        "evidence_row_ids": _ordered_unique(row_ids),
        "min_quality_score": _minimum_or_none(quality_scores),
        "min_parse_confidence": _minimum_or_none(parse_confidences),
        "min_trust_score": _minimum_or_none(trust_scores),
    }


def _all_atomic_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_tables = package.get("evidence_tables.json", {})
    if not isinstance(evidence_tables, dict):
        return []
    all_rows = evidence_tables.get("all_rows")
    if isinstance(all_rows, list):
        return [row for row in all_rows if isinstance(row, dict)]
    rows = []
    tables = evidence_tables.get("tables")
    if isinstance(tables, dict):
        for table_rows in tables.values():
            if isinstance(table_rows, list):
                rows.extend(row for row in table_rows if isinstance(row, dict))
    return rows


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


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minimum_or_none(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(min(cleaned), 6)
