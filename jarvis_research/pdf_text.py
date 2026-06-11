from __future__ import annotations

import re
from collections import Counter


SECTION_HEADINGS = {
    "abstract",
    "background",
    "objective",
    "methods",
    "method",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "limitation",
}

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def clean_pdf_pages(pages: list[str]) -> list[str]:
    normalized_pages = [_normalize_text(page) for page in pages]
    repeated_edge_lines = _repeated_edge_lines(normalized_pages)
    cleaned_pages = []
    for page in normalized_pages:
        lines = []
        for raw_line in page.splitlines():
            line = _normalize_line(raw_line)
            if not line:
                continue
            if _should_drop_line(line, repeated_edge_lines):
                continue
            lines.append(line)
        joined = _join_hyphenated_line_breaks("\n".join(lines))
        cleaned_pages.append(joined.strip())
    return [page for page in cleaned_pages if page]


def _normalize_text(text: str) -> str:
    normalized = text
    for source, target in LIGATURES.items():
        normalized = normalized.replace(source, target)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def _repeated_edge_lines(pages: list[str]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [_normalize_line(line) for line in page.splitlines() if _normalize_line(line)]
        edge_lines = set(lines[:2] + lines[-2:])
        for line in edge_lines:
            if len(line) >= 8 and not _is_section_heading(line):
                counts[line.lower()] += 1
    return {line for line, count in counts.items() if count >= 2}


def _should_drop_line(line: str, repeated_edge_lines: set[str]) -> bool:
    lowered = line.lower()
    if lowered in repeated_edge_lines:
        return True
    if _is_page_number(line):
        return True
    if "www." in lowered and ("journal" in lowered or "frontiers" in lowered):
        return True
    if re.search(r"\b(?:doi|issn)\s*[:/]", lowered):
        return True
    return False


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}", line, flags=re.IGNORECASE))


def _is_section_heading(line: str) -> bool:
    normalized = line.strip().rstrip(":").lower()
    return normalized in SECTION_HEADINGS


def _join_hyphenated_line_breaks(text: str) -> str:
    return re.sub(r"([A-Za-z]+)-\n([A-Za-z]+)", _join_hyphen_match, text)


def _join_hyphen_match(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if left.isupper() or right.isupper():
        return f"{left}-{right}"
    return f"{left}{right}"
