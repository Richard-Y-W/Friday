from __future__ import annotations

from dataclasses import dataclass
import re


DEFAULT_MAX_EVIDENCE_ITEMS = 40
DEFAULT_MAX_ITEMS_PER_TYPE = 8
DEFAULT_MAX_TEXT_CHARS = 500

EVIDENCE_TYPES = {
    "claim",
    "method",
    "result",
    "limitation",
    "dataset_population",
}

INSTRUCTION_INJECTION_PATTERNS = (
    re.compile(r"\bignore (all )?(previous|prior|above) instructions\b", re.IGNORECASE),
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper message\b", re.IGNORECASE),
    re.compile(r"\breveal .*secret", re.IGNORECASE),
    re.compile(r"\bexfiltrat", re.IGNORECASE),
    re.compile(r"\bexecute (a )?(command|script)\b", re.IGNORECASE),
    re.compile(r"\brun (bash|sh|python|curl|wget)\b", re.IGNORECASE),
)

NON_EVIDENCE_PATTERNS = (
    re.compile(r"\bdepartment of\b", re.IGNORECASE),
    re.compile(r"\bkeywords?:\b", re.IGNORECASE),
    re.compile(r"\bspecialty section\b", re.IGNORECASE),
    re.compile(r"\bcorrespondence\b", re.IGNORECASE),
    re.compile(r"\breceived:\b|\baccepted:\b|\bpublished:\b", re.IGNORECASE),
    re.compile(r"\bdetection method advantages disadvantages\b", re.IGNORECASE),
    re.compile(r"\badvantages disadvantages\b", re.IGNORECASE),
)

LAYOUT_NOISE_PATTERNS = (
    re.compile(r"\btable\s+\d+\s*\|", re.IGNORECASE),
    re.compile(r"\bfigure\s+\d+\s*\|", re.IGNORECASE),
    re.compile(r"\bfrontiers in\b.*\bwww\.", re.IGNORECASE),
    re.compile(r"\bwww\.[^\s]+", re.IGNORECASE),
    re.compile(r"\binterpretation of however\b", re.IGNORECASE),
    re.compile(r"\bthus interpretation of however\b", re.IGNORECASE),
    re.compile(r"\b(?:the|this|a|an|we|they)\s+[a-z]+\s+(?:both|however|the|they|when|thus)\b", re.IGNORECASE),
    re.compile(r"\bthey this\b", re.IGNORECASE),
    re.compile(r"\b(?:maldi|tof)-in\b", re.IGNORECASE),
    re.compile(r"\bthe sample microbial\b", re.IGNORECASE),
    re.compile(r"\bavailability of many within\b", re.IGNORECASE),
    re.compile(r"\bin the a number of\b", re.IGNORECASE),
    re.compile(r"\ba number of .* matrices sample\b", re.IGNORECASE),
)

SECTION_ALIASES = {
    "abstract": "abstract",
    "background": "abstract",
    "objective": "abstract",
    "objectives": "abstract",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "methodology": "methods",
    "dataset": "methods",
    "population": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
    "limitation": "limitations",
    "references": "references",
    "acknowledgements": "references",
    "acknowledgments": "references",
}

EVIDENCE_SECTIONS = {
    "claim": {"unknown", "abstract", "discussion", "conclusion"},
    "method": {"unknown", "abstract", "methods"},
    "result": {"unknown", "abstract", "results", "discussion", "conclusion"},
    "dataset_population": {"unknown", "abstract", "methods", "results"},
    "limitation": {"unknown", "abstract", "discussion", "conclusion", "limitations"},
}

TYPE_PATTERNS = {
    "limitation": (
        re.compile(r"\blimitations?\b", re.IGNORECASE),
        re.compile(r"\blimited by\b", re.IGNORECASE),
        re.compile(r"\bsingle[- ]center\b", re.IGNORECASE),
        re.compile(r"\bsmall sample\b", re.IGNORECASE),
        re.compile(r"\bfurther validation\b", re.IGNORECASE),
    ),
    "result": (
        re.compile(r"\bresults?\b", re.IGNORECASE),
        re.compile(r"\bachieved\b", re.IGNORECASE),
        re.compile(r"\b(improved|increased|decreased|reduced)\b", re.IGNORECASE),
        re.compile(r"\b(accuracy|sensitivity|specificity|auc|auroc|f1[- ]score)\b", re.IGNORECASE),
        re.compile(r"\bp\s*[<=>]\s*0?\.\d+", re.IGNORECASE),
        re.compile(r"\bsignificant(?:ly)?\b", re.IGNORECASE),
    ),
    "dataset_population": (
        re.compile(r"\bdatasets?\b", re.IGNORECASE),
        re.compile(r"\bcohort\b", re.IGNORECASE),
        re.compile(r"\bpopulation\b", re.IGNORECASE),
        re.compile(r"\bpatients?\b", re.IGNORECASE),
        re.compile(r"\bparticipants?\b", re.IGNORECASE),
        re.compile(r"\bclinical isolates?\b", re.IGNORECASE),
        re.compile(r"\bisolates?\b", re.IGNORECASE),
        re.compile(r"\bsamples?\b", re.IGNORECASE),
        re.compile(r"\bspecimens?\b", re.IGNORECASE),
        re.compile(r"\bn\s*=\s*\d+", re.IGNORECASE),
    ),
    "method": (
        re.compile(r"\bwe (used|use|analyzed|analysed|trained|evaluated|performed)\b", re.IGNORECASE),
        re.compile(r"\busing\b", re.IGNORECASE),
        re.compile(r"\b(classifier|model|regression|machine learning|random forest)\b", re.IGNORECASE),
        re.compile(r"\bspectra were\b", re.IGNORECASE),
    ),
    "claim": (
        re.compile(r"\bwe (show|demonstrate|present|propose|develop|developed|report)\b", re.IGNORECASE),
        re.compile(r"\bthis study (shows|demonstrates|presents|reports)\b", re.IGNORECASE),
        re.compile(r"\bour findings\b", re.IGNORECASE),
        re.compile(r"\bobjective\b", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class EvidenceItem:
    evidence_type: str
    text: str
    page_number: int


def extract_evidence_from_pages(
    pages: list[str],
    *,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
    max_items_per_type: int = DEFAULT_MAX_ITEMS_PER_TYPE,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    type_counts = {evidence_type: 0 for evidence_type in EVIDENCE_TYPES}

    for page_number, page_text in enumerate(pages, start=1):
        for section, sentence in _sectioned_sentences(page_text):
            text = _sanitize_text(sentence, max_text_chars=max_text_chars)
            if not is_reportable_evidence_text(text):
                continue
            evidence_type = _classify(text)
            if evidence_type is None:
                continue
            if not _section_supports_evidence_type(section, evidence_type):
                continue
            if type_counts[evidence_type] >= max_items_per_type:
                continue
            items.append(EvidenceItem(evidence_type=evidence_type, text=text, page_number=page_number))
            type_counts[evidence_type] += 1
            if len(items) >= max_items:
                return items
    return items


def is_reportable_evidence_text(text: str) -> bool:
    if len(text) < 20:
        return False
    if _looks_like_instruction_injection(text):
        return False
    if _looks_like_non_evidence(text):
        return False
    if _looks_like_layout_noise(text):
        return False
    return True


def _sectioned_sentences(text: str) -> list[tuple[str, str]]:
    blocks = _section_blocks(text)
    sectioned = []
    for section, block_text in blocks:
        sectioned.extend((section, sentence) for sentence in _sentences(block_text))
    return sectioned


def _section_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    if not lines:
        return [("unknown", text)]

    section = "unknown"
    current: list[str] = []
    blocks: list[tuple[str, str]] = []
    saw_heading = False
    for line in lines:
        heading = _section_heading(line)
        if heading:
            if current:
                blocks.append((section, " ".join(current)))
                current = []
            section = heading
            saw_heading = True
            continue
        current.append(line)
    if current:
        blocks.append((section, " ".join(current)))
    if not saw_heading:
        return [("unknown", text)]
    return blocks


def _section_heading(line: str) -> str | None:
    normalized = " ".join(line.strip().strip(":").lower().split())
    return SECTION_ALIASES.get(normalized)


def _sentences(text: str) -> list[str]:
    prepared = re.sub(
        r"\b(Abstract|Background|Objective|Methods?|Results?|Conclusion|Limitations?|Dataset|Population):",
        r". \1:",
        text,
        flags=re.IGNORECASE,
    )
    prepared = re.sub(r"[\r\n]+", " ", prepared)
    return [
        sentence.strip(" .")
        for sentence in re.split(r"(?<=[.!?])\s+", prepared)
        if sentence.strip(" .")
    ]


def _sanitize_text(text: str, *, max_text_chars: int) -> str:
    cleaned = "".join(char if char.isprintable() else " " for char in text)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= max_text_chars:
        return cleaned
    return cleaned[: max(0, max_text_chars - 3)].rstrip() + "..."


def _looks_like_instruction_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INSTRUCTION_INJECTION_PATTERNS)


def _looks_like_non_evidence(text: str) -> bool:
    return any(pattern.search(text) for pattern in NON_EVIDENCE_PATTERNS)


def _looks_like_layout_noise(text: str) -> bool:
    if "|" in text:
        return True
    if any(pattern.search(text) for pattern in LAYOUT_NOISE_PATTERNS):
        return True
    if re.fullmatch(r"(?:table|figure)\s+\d+.*", text, flags=re.IGNORECASE):
        return True
    return False


def _classify(text: str) -> str | None:
    for evidence_type in ("limitation", "result", "method", "dataset_population", "claim"):
        if any(pattern.search(text) for pattern in TYPE_PATTERNS[evidence_type]):
            return evidence_type
    return None


def _section_supports_evidence_type(section: str, evidence_type: str) -> bool:
    if section == "references":
        return False
    return section in EVIDENCE_SECTIONS[evidence_type]
