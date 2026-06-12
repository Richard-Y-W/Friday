from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


DEFAULT_MAX_EVIDENCE_ITEMS = 40
DEFAULT_MAX_ITEMS_PER_TYPE = 8
DEFAULT_MAX_TEXT_CHARS = 500
MIN_PAGE_PARSE_CONFIDENCE = 0.6

EVIDENCE_TYPES = {
    "claim",
    "method",
    "result",
    "limitation",
    "dataset_population",
}

QUALITY_LABELS = {"clean", "suspect", "blocked"}
TRUST_LABELS = {"trusted", "review", "quarantined"}
MIN_TRUSTED_EVIDENCE_SCORE = 0.8
MIN_REVIEW_EVIDENCE_SCORE = 0.55

TRUST_QUARANTINE_FLAGS = {
    "column_stitching",
    "document_parse_quality",
    "embedded_heading",
    "formula_fragment",
    "front_matter",
    "hyphenation_break",
    "instruction_injection",
    "low_page_parse_confidence",
    "page_parse_quality",
    "reference_section",
    "sentence_fragment",
    "short_fragment",
    "symbol_loss",
    "table_fragment",
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
    re.compile(r"\bfaculty of\b", re.IGNORECASE),
    re.compile(r"\borcid\.org\b", re.IGNORECASE),
    re.compile(r"\bcorresponding author\b", re.IGNORECASE),
    re.compile(r"\bavailability of data and materials\b", re.IGNORECASE),
    re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE),
    re.compile(r"\bkeywords?\s*:", re.IGNORECASE),
    re.compile(r"\bspecialty section\b", re.IGNORECASE),
    re.compile(r"\bcorrespondence\b", re.IGNORECASE),
    re.compile(r"\breceived\s*:|\baccepted\s*:|\bpublished\s*:", re.IGNORECASE),
    re.compile(r"\boriginal research\s+published\s*:", re.IGNORECASE),
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
    re.compile(r"\btables?\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bfigures?\s+\d+\b", re.IGNORECASE),
    re.compile(r"\busing unfortunately\b", re.IGNORECASE),
    re.compile(r"\bof understanding leads\b", re.IGNORECASE),
    re.compile(r"\bfrom are manifold\b", re.IGNORECASE),
    re.compile(r"\barticles\s+seTo\b", re.IGNORECASE),
    re.compile(r"\breSearch strategy\b"),
    re.compile(r"\breregion\b", re.IGNORECASE),
    re.compile(r"\bmutationinduced\b", re.IGNORECASE),
    re.compile(r"\bregularizagating\b", re.IGNORECASE),
    re.compile(r"^(?:[A-Z][a-z]+ [A-Z],\s*){2,}"),
    re.compile(r"\braising awareness of malaysia\b", re.IGNORECASE),
    re.compile(r"\bthe discovery resistance\b", re.IGNORECASE),
    re.compile(r"\brelatable and appropriate journals\b", re.IGNORECASE),
    re.compile(r"\bdespite the more\b", re.IGNORECASE),
    re.compile(r"\btheir mass aminoglycoside\b", re.IGNORECASE),
    re.compile(r"\bresulted in the caused\b", re.IGNORECASE),
    re.compile(r"\bgramsingle\b", re.IGNORECASE),
    re.compile(r"\bfda two significant areas\b", re.IGNORECASE),
    re.compile(r"\buse formulary\b", re.IGNORECASE),
    re.compile(r"\bsuccessfully studies reported\b", re.IGNORECASE),
    re.compile(r"\bprotect highly seems\b", re.IGNORECASE),
    re.compile(r"\bmost over the patient\b", re.IGNORECASE),
    re.compile(r"\btight plasmids\b", re.IGNORECASE),
    re.compile(r"\bit is bank\b", re.IGNORECASE),
    re.compile(r"\bwrote the introis\b", re.IGNORECASE),
    re.compile(r"\bthe marized using\b", re.IGNORECASE),
    re.compile(r"\bescheriand\b", re.IGNORECASE),
    re.compile(r"\b(?:β|beta|-)?lactam lactamases\b", re.IGNORECASE),
    re.compile(r"\blower respiratory tract\s+\d+$", re.IGNORECASE),
    re.compile(r"^secondary of microorganisms\b", re.IGNORECASE),
    re.compile(r"\ba previous study mic range\b.*\bin parallel\b", re.IGNORECASE),
    re.compile(r"\bminimum the use\b", re.IGNORECASE),
    re.compile(r"\bpathogen\s+ance\b", re.IGNORECASE),
    re.compile(r"\bin only gentamicin\b", re.IGNORECASE),
    re.compile(r"\bresistance pattern addition\b", re.IGNORECASE),
    re.compile(r"^the resistant was\b", re.IGNORECASE),
    re.compile(r",\s+the most critical group\b", re.IGNORECASE),
    re.compile(r"\bhigh prevalence bacteriological profile\b", re.IGNORECASE),
    re.compile(r"\bcurrent microbial isolates from wound swabs\b", re.IGNORECASE),
    re.compile(r"\bone study compared mic results using broth in susceptibility\b", re.IGNORECASE),
    re.compile(r"\boverall median phenotypic results\b", re.IGNORECASE),
    re.compile(r"\bbruker\s+after exposure\b.*\bdaltonics\b", re.IGNORECASE),
    re.compile(r"\bclinprottools\b.*\bbruker\s+after exposure\b", re.IGNORECASE),
    re.compile(r"\bafter exposure of\s+[-–−]\s*lactam antibiotics to\s+[-–−]\s*lactamase producing\s+daltonics\b", re.IGNORECASE),
)

EMBEDDED_SECTION_HEADINGS = (
    "article quality assessment",
    "availability of data and materials",
    "data extraction",
    "limitations of this study",
    "microbial resistance patterns",
    "search strategy",
)

MID_SENTENCE_SECTION_STARTERS = (
    "Article",
    "Articles",
    "Background",
    "Both",
    "By",
    "Coding",
    "Data",
    "Each",
    "For",
    "If",
    "Information",
    "Initial",
    "Interviews",
    "Methods",
    "Microbial",
    "No",
    "Objective",
    "Poultry",
    "Results",
    "Secondly",
    "The",
    "This",
    "Thus",
    "Unfortunately",
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
        re.compile(r"\bthis (review|study) used\b", re.IGNORECASE),
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
class EvidenceQuality:
    label: str
    score: float
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceTrust:
    label: str
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    evidence_type: str
    text: str
    page_number: int
    quality_label: str = "clean"
    quality_score: float = 1.0
    quality_flags: tuple[str, ...] = ()
    parse_confidence: float = 1.0
    parse_flags: tuple[str, ...] = ()
    trust_label: str | None = None
    trust_score: float | None = None
    trust_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceCurationResult:
    accepted: list[EvidenceItem]
    blocked: list[EvidenceItem]
    blocked_by_flag: dict[str, int]


def extract_evidence_from_pages(
    pages: list[str],
    *,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
    max_items_per_type: int = DEFAULT_MAX_ITEMS_PER_TYPE,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> list[EvidenceItem]:
    return curate_evidence_from_pages(
        pages,
        max_items=max_items,
        max_items_per_type=max_items_per_type,
        max_text_chars=max_text_chars,
    ).accepted


def curate_evidence_from_pages(
    pages: list[str],
    *,
    page_parse_confidences: list[float] | None = None,
    page_parse_flags: list[tuple[str, ...]] | None = None,
    min_page_parse_confidence: float = MIN_PAGE_PARSE_CONFIDENCE,
    max_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
    max_items_per_type: int = DEFAULT_MAX_ITEMS_PER_TYPE,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_blocked_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
) -> EvidenceCurationResult:
    items: list[EvidenceItem] = []
    blocked: list[EvidenceItem] = []
    blocked_by_flag: Counter[str] = Counter()
    type_counts = {evidence_type: 0 for evidence_type in EVIDENCE_TYPES}

    for page_number, page_text in enumerate(pages, start=1):
        page_parse_confidence = (
            page_parse_confidences[page_number - 1]
            if page_parse_confidences and page_number - 1 < len(page_parse_confidences)
            else 1.0
        )
        page_parse_flags_for_page = (
            page_parse_flags[page_number - 1]
            if page_parse_flags and page_number - 1 < len(page_parse_flags)
            else ()
        )
        page_quality_flags = _page_parse_quality_flags(
            page_parse_confidence,
            min_page_parse_confidence=min_page_parse_confidence,
        )
        page_accepted: list[EvidenceItem] = []
        page_blocked: list[EvidenceItem] = []
        body_text, reference_text = _split_reference_tail(page_text)
        reference_blocked = _reference_blocked_items(
            reference_text,
            page_number=page_number,
            max_text_chars=max_text_chars,
        )
        if not reference_text and _page_looks_like_reference_list(page_text):
            body_text = ""
            reference_blocked = _reference_blocked_items(
                page_text,
                page_number=page_number,
                max_text_chars=max_text_chars,
            )
        if not body_text.strip():
            for item in page_blocked:
                blocked_by_flag.update(item.quality_flags)
                if len(blocked) < max_blocked_items:
                    blocked.append(item)
            for item in reference_blocked:
                blocked_by_flag.update(item.quality_flags)
                if len(blocked) < max_blocked_items:
                    blocked.append(item)
            continue

        for section, sentence in _sectioned_sentences(body_text):
            text = _sanitize_text(sentence, max_text_chars=max_text_chars)
            quality = assess_evidence_quality(text)
            if quality.label != "clean":
                evidence_type = _classify(text)
                if evidence_type is None:
                    continue
                page_blocked.append(
                    EvidenceItem(
                        evidence_type=evidence_type,
                        text=text,
                        page_number=page_number,
                        quality_label=quality.label,
                        quality_score=quality.score,
                        quality_flags=tuple(_ordered_unique([*quality.flags, *page_quality_flags])),
                        parse_confidence=page_parse_confidence,
                        parse_flags=page_parse_flags_for_page,
                    )
                )
                continue
            evidence_type = _classify(text)
            if evidence_type is None:
                continue
            text = _strip_leading_inline_section_heading(text)
            if not text:
                continue
            if not _section_supports_evidence_type(section, evidence_type):
                continue
            if page_quality_flags:
                page_blocked.append(
                    EvidenceItem(
                        evidence_type=evidence_type,
                        text=text,
                        page_number=page_number,
                        quality_label="blocked",
                        quality_score=min(quality.score, page_parse_confidence),
                        quality_flags=page_quality_flags,
                        parse_confidence=page_parse_confidence,
                        parse_flags=page_parse_flags_for_page,
                    )
                )
                continue
            page_accepted.append(
                EvidenceItem(
                    evidence_type=evidence_type,
                    text=text,
                    page_number=page_number,
                    quality_label=quality.label,
                    quality_score=quality.score,
                    quality_flags=quality.flags,
                    parse_confidence=page_parse_confidence,
                    parse_flags=page_parse_flags_for_page,
                )
            )

        if _page_parse_quality_is_poor(page_accepted, page_blocked):
            page_accepted, page_quality_blocked = _split_page_accepted_by_parse_quality(page_accepted)
            page_blocked.extend(page_quality_blocked)
        page_blocked.extend(reference_blocked)

        for item in page_blocked:
            blocked_by_flag.update(item.quality_flags)
            if len(blocked) < max_blocked_items:
                blocked.append(item)

        for item in page_accepted:
            if type_counts[item.evidence_type] >= max_items_per_type:
                continue
            items.append(item)
            type_counts[item.evidence_type] += 1
            if len(items) >= max_items:
                return EvidenceCurationResult(
                    accepted=items,
                    blocked=blocked,
                    blocked_by_flag=dict(blocked_by_flag),
                )
    return EvidenceCurationResult(
        accepted=items,
        blocked=blocked,
        blocked_by_flag=dict(blocked_by_flag),
    )


def is_reportable_evidence_text(text: str) -> bool:
    return assess_evidence_quality(text).label == "clean"


def assess_evidence_trust(item: EvidenceItem) -> EvidenceTrust:
    reasons: list[str] = []
    quality_score = _clamp_score(item.quality_score)
    parse_confidence = _clamp_score(item.parse_confidence)
    score = quality_score * parse_confidence
    quarantine = False

    if item.quality_label != "clean":
        reasons.append(f"quality_label:{item.quality_label}")
        score = min(score, 0.2)
        quarantine = True
    else:
        current_quality = assess_evidence_quality(item.text)
        if current_quality.label != "clean":
            reasons.append("legacy_quality_filter")
            reasons.extend(current_quality.flags)
            score = min(score, 0.2)
            quarantine = True
    if quality_score < 0.9:
        reasons.append("lower_quality_score")
    if parse_confidence < 0.9:
        reasons.append("lower_parse_confidence")
    if parse_confidence < MIN_PAGE_PARSE_CONFIDENCE:
        reasons.append("low_parse_confidence")
        score = min(score, 0.3)
        quarantine = True
    if item.page_number <= 0:
        reasons.append("missing_page_anchor")
        score = min(score, 0.4)
        quarantine = True
    if _word_count(item.text) <= 4:
        reasons.append("sentence_incomplete")
        score = min(score, 0.45)
        quarantine = True

    for flag in item.quality_flags:
        if flag in TRUST_QUARANTINE_FLAGS:
            reasons.append(flag)
            score = min(score, 0.2)
            quarantine = True
        else:
            reasons.append(f"quality_flag:{flag}")
            score = min(score, 0.78)

    for flag in item.parse_flags:
        reasons.append(f"parse_flag:{flag}")
        score = min(score, 0.78)

    score = round(_clamp_score(score), 3)
    if quarantine or score < MIN_REVIEW_EVIDENCE_SCORE:
        label = "quarantined"
    elif score < MIN_TRUSTED_EVIDENCE_SCORE or reasons:
        label = "review"
    else:
        label = "trusted"
    return EvidenceTrust(label=label, score=score, reasons=tuple(_ordered_unique(reasons)))


def is_trusted_evidence_item(item: EvidenceItem) -> bool:
    if item.trust_label is not None:
        return item.trust_label == "trusted"
    return assess_evidence_trust(item).label == "trusted"


def apply_document_parse_quality_gate(
    curation: EvidenceCurationResult,
    *,
    min_blocked_items: int = 20,
    min_blocked_ratio: float = 0.6,
) -> EvidenceCurationResult:
    parse_quality_blocked = [
        item for item in curation.blocked if "reference_section" not in item.quality_flags
    ]
    if not _document_parse_quality_is_poor(
        curation.accepted,
        parse_quality_blocked,
        min_blocked_items=min_blocked_items,
        min_blocked_ratio=min_blocked_ratio,
    ):
        return curation

    accepted = []
    document_blocked = []
    for item in curation.accepted:
        if _is_document_quality_vulnerable_evidence(item):
            document_blocked.extend(_block_accepted_items([item], "document_parse_quality"))
            continue
        accepted.append(item)

    if not document_blocked:
        return curation

    blocked_by_flag = Counter(curation.blocked_by_flag)
    for item in document_blocked:
        blocked_by_flag.update(item.quality_flags)
    return EvidenceCurationResult(
        accepted=accepted,
        blocked=[*curation.blocked, *document_blocked],
        blocked_by_flag=dict(blocked_by_flag),
    )


def assess_evidence_quality(text: str) -> EvidenceQuality:
    flags: list[str] = []
    if len(text) < 20 or _word_count(text) <= 4:
        flags.append("short_fragment")
    if _looks_like_instruction_injection(text):
        flags.append("instruction_injection")
    if _looks_like_non_evidence(text):
        flags.append("front_matter")
    layout_flags = _layout_noise_flags(text)
    flags.extend(layout_flags)
    if flags:
        return EvidenceQuality(label="blocked", score=0.2, flags=tuple(_ordered_unique(flags)))
    return EvidenceQuality(label="clean", score=1.0, flags=())


def _clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def _page_parse_quality_is_poor(accepted: list[EvidenceItem], blocked: list[EvidenceItem]) -> bool:
    evidence_like_count = len(accepted) + len(blocked)
    if evidence_like_count == 0:
        return False
    return len(blocked) >= 2 and (len(blocked) / evidence_like_count) >= 0.5


def _page_parse_quality_flags(
    parse_confidence: float,
    *,
    min_page_parse_confidence: float,
) -> tuple[str, ...]:
    if parse_confidence < min_page_parse_confidence:
        return ("low_page_parse_confidence",)
    return ()


def _document_parse_quality_is_poor(
    accepted: list[EvidenceItem],
    blocked: list[EvidenceItem],
    *,
    min_blocked_items: int,
    min_blocked_ratio: float,
) -> bool:
    evidence_like_count = len(accepted) + len(blocked)
    if evidence_like_count == 0 or len(blocked) < min_blocked_items:
        return False
    return (len(blocked) / evidence_like_count) >= min_blocked_ratio


def _split_page_accepted_by_parse_quality(items: list[EvidenceItem]) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    accepted = []
    blocked = []
    for item in items:
        if _is_page_quality_vulnerable_evidence(item):
            blocked.extend(_block_accepted_items([item], "page_parse_quality"))
            continue
        accepted.append(item)
    return accepted, blocked


def _is_document_quality_vulnerable_evidence(item: EvidenceItem) -> bool:
    if _is_page_quality_vulnerable_evidence(item):
        return True
    return not _has_strong_document_quality_anchor(item.text)


def _is_page_quality_vulnerable_evidence(item: EvidenceItem) -> bool:
    text = item.text.strip()
    lowered = text.lower()
    word_count = _word_count(text)
    if word_count <= 8:
        return True
    if word_count <= 14 and re.match(r"^(using|by|for|from|in|with|based on)\b", lowered):
        return True
    if re.search(r"\b(?:and|or|of|for|with|using)\s+\d{2,4}$", lowered):
        return True
    return False


def _has_strong_document_quality_anchor(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(this review|this study|our findings|article retrieval and screening)\b", lowered):
        return True
    if re.search(r"\b(structured search|inclusion/exclusion|clinical isolates|validation cohort)\b", lowered):
        return True
    if re.search(r"\b(auroc|auc|sensitivity|specificity)\b", lowered):
        return True
    if re.search(r"\b(minimum inhibitory|mic)\b", lowered) and re.search(
        r"(?:[<>]=?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:ug|µg|μg|mg|mm|%))",
        lowered,
    ):
        return True
    if re.search(r"\b(pubmed|embase|cochrane)\b", lowered) and re.search(r"\b(search|databases?|studies)\b", lowered):
        return True
    return False


def _block_accepted_items(items: list[EvidenceItem], flag: str) -> list[EvidenceItem]:
    blocked = []
    for item in items:
        flags = tuple(_ordered_unique([*item.quality_flags, flag]))
        blocked.append(
            EvidenceItem(
                evidence_type=item.evidence_type,
                text=item.text,
                page_number=item.page_number,
                quality_label="blocked",
                quality_score=min(item.quality_score, 0.4),
                quality_flags=flags,
                parse_confidence=item.parse_confidence,
                parse_flags=item.parse_flags,
            )
        )
    return blocked


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
    cleaned = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= max_text_chars:
        return cleaned
    return cleaned[: max(0, max_text_chars - 3)].rstrip() + "..."


def _strip_leading_inline_section_heading(text: str) -> str:
    heading_pattern = "|".join(
        re.escape(heading)
        for heading in sorted(
            [
                "abstract",
                "background",
                "objective",
                "objectives",
                "introduction",
                "methods",
                "method",
                "materials and methods",
                "methodology",
                "results and discussion",
                "result and discussion",
                "results",
                "result",
                "findings",
                "discussion",
                "conclusion",
                "conclusions",
                "limitations",
                "limitation",
                "dataset",
                "population",
            ],
            key=len,
            reverse=True,
        )
    )
    cleaned = re.sub(
        rf"^(?:{heading_pattern})\s*(?::|\.|-)?\s+",
        "",
        text.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" .")


def _looks_like_instruction_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INSTRUCTION_INJECTION_PATTERNS)


def _looks_like_non_evidence(text: str) -> bool:
    return any(pattern.search(text) for pattern in NON_EVIDENCE_PATTERNS)


def _looks_like_layout_noise(text: str) -> bool:
    return bool(_layout_noise_flags(text))


def _layout_noise_flags(text: str) -> tuple[str, ...]:
    flags: list[str] = []
    if "|" in text:
        flags.append("table_fragment")
    if "•" in text:
        flags.append("table_fragment")
    if _looks_like_dense_table_row(text):
        flags.append("table_fragment")
    if _starts_like_fragment(text):
        flags.append("sentence_fragment")
    if _has_spaced_hyphenation_break(text):
        flags.append("hyphenation_break")
    if any(pattern.search(text) for pattern in LAYOUT_NOISE_PATTERNS):
        flags.append("column_stitching")
    flags.extend(_symbol_loss_flags(text))
    if _has_embedded_all_caps_heading(text):
        flags.append("embedded_heading")
    if _has_embedded_section_headings(text):
        flags.append("embedded_heading")
    if _has_single_embedded_section_heading(text):
        flags.append("embedded_heading")
    if _has_mid_sentence_section_starter(text):
        flags.append("column_stitching")
    if _ends_with_dangling_connector(text):
        flags.append("sentence_fragment")
    if _has_formula_fragment(text):
        flags.append("formula_fragment")
    if _has_ocr_spaced_words(text):
        flags.append("ocr_spacing")
    if re.fullmatch(r"(?:table|figure)\s+\d+.*", text, flags=re.IGNORECASE):
        flags.append("table_fragment")
    return tuple(_ordered_unique(flags))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9-]*\b", text))


def _looks_like_dense_table_row(text: str) -> bool:
    if len(re.findall(r"\b(?:NA|NS|HAI|UTI|AGE|WI|BSI|CoNS)\b", text)) < 3:
        return False
    if len(re.findall(r"\d+(?:\.\d+)?\s*\(\d+(?:\.\d+)?\)", text)) < 2:
        return False
    return True


def _starts_like_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r"^(?:\(?\d+\)?|[:;,])", stripped):
        return True
    return stripped[0].islower()


def _has_spaced_hyphenation_break(text: str) -> bool:
    return re.search(r"\b[A-Za-z]{3,}-\s+[A-Za-z]{2,}\b", text) is not None


def _has_embedded_all_caps_heading(text: str) -> bool:
    for match in re.finditer(
        r"\b(?:[A-Z][A-Z-]{2,}|AND|OF|FOR|THE|IN|TO)(?:\s+(?:[A-Z][A-Z-]{2,}|AND|OF|FOR|THE|IN|TO)){2,}\b",
        text,
    ):
        words = match.group(0).split()
        if any(len(word.strip("-")) >= 7 for word in words):
            return True
    return False


def _has_embedded_section_headings(text: str) -> bool:
    lowered = text.lower()
    matches = [heading for heading in EMBEDDED_SECTION_HEADINGS if _heading_match_indices(lowered, heading)]
    if len(matches) >= 2:
        return True
    return any(len(_heading_match_indices(lowered, heading)) >= 2 for heading in matches)


def _has_single_embedded_section_heading(text: str) -> bool:
    lowered = text.lower()
    for heading in EMBEDDED_SECTION_HEADINGS:
        indices = _heading_match_indices(lowered, heading)
        if not indices:
            continue
        index = indices[0]
        if heading == "search strategy" and lowered[max(0, index - 16) : index].strip().endswith("structured"):
            continue
        if index > 20:
            return True
    return False


def _heading_match_indices(lowered_text: str, heading: str) -> list[int]:
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(heading)}(?![a-z0-9])")
    return [match.start() for match in pattern.finditer(lowered_text)]


def _has_mid_sentence_section_starter(text: str) -> bool:
    starters = "|".join(re.escape(starter) for starter in MID_SENTENCE_SECTION_STARTERS)
    return re.search(
        rf"\b[a-z][a-z]+\s+(?:{starters})\b",
        text,
    ) is not None


def _ends_with_dangling_connector(text: str) -> bool:
    stripped = text.strip()
    if re.search(r"\b(?:and|or|of|for|to|with|using|as|in|by|from|through|a|an|the)$", stripped, re.IGNORECASE):
        return True
    if re.search(r"\b(?:i\.e|e\.g)$", stripped, re.IGNORECASE):
        return True
    if re.search(r"\(\[\d+\s*,\s*(?:thm|theorem|lemma|prop|proposition)$", stripped, re.IGNORECASE):
        return True
    return False


def _has_formula_fragment(text: str) -> bool:
    stripped = text.strip()
    math_chars = set("~∈∞∗∑∫√≤≥<>⊂⊕⊖⨏̃τ−")
    math_char_count = sum(1 for char in stripped if char in math_chars)
    bracket_count = sum(1 for char in stripped if char in "()[]{}")
    operator_count = len(re.findall(r"(?:[A-Za-z]\s*[=<>]\s*[A-Za-z0-9~]|[A-Za-z]\s*[-+]\s*~|~\s*[A-Za-z0-9])", stripped))
    word_count = _word_count(stripped)
    if math_char_count + bracket_count >= 18 and operator_count >= 2:
        return True
    if word_count <= 20 and math_char_count + bracket_count >= 6 and operator_count >= 2:
        return True
    if re.match(r"^[<>=~({\[]", stripped) and (",," in stripped or "~" in stripped) and word_count <= 16:
        return True
    if math_char_count >= 4 and re.search(r"[:=]\s*[\{\[\(]?\s*\d+\s*,\s*$", stripped):
        return True
    if re.search(r"\b[A-Za-z][A-Za-z0-9]*\([^)]{0,20}~[^)]*\)", stripped) and math_char_count >= 3:
        return True
    return False


def _has_ocr_spaced_words(text: str) -> bool:
    spaced_words = re.findall(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", text)
    if len(spaced_words) >= 2:
        return True
    if spaced_words and sum(1 for char in text if char in "~=()[]{}") >= 8:
        return True
    return False


def _symbol_loss_flags(text: str) -> list[str]:
    flags: list[str] = []
    if "\ufffd" in text:
        flags.append("symbol_loss")
    if re.search(r"(?<![A-Za-zβ])[-–−]\s*lactam(?:ase)?\b", text, flags=re.IGNORECASE):
        flags.append("symbol_loss")
    return flags


def _classify(text: str) -> str | None:
    for evidence_type in ("limitation", "result", "method", "dataset_population", "claim"):
        if any(pattern.search(text) for pattern in TYPE_PATTERNS[evidence_type]):
            return evidence_type
    return None


def _section_supports_evidence_type(section: str, evidence_type: str) -> bool:
    if section == "references":
        return False
    return section in EVIDENCE_SECTIONS[evidence_type]


def _page_looks_like_reference_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    numbered_references = 0
    citation_markers = 0
    for line in lines:
        if re.match(r"^\d{1,4}\.\s+[A-Z][A-Za-z'’-]+", line):
            numbered_references += 1
        if re.search(
            r"\(\d{4}\)|\b(?:J|Journal|Clin|Clinical|Microbiol|Antimicrob|Infect|Med|Pharm|PLoS|Front|Lancet|BMJ)\b",
            line,
        ):
            citation_markers += 1
    return numbered_references >= 3 and citation_markers >= 3


def _split_reference_tail(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().rstrip(":") == "references":
            return "\n".join(lines[:index]), "\n".join(lines[index + 1 :])
    return text, ""


def _reference_blocked_items(
    text: str,
    *,
    page_number: int,
    max_text_chars: int,
) -> list[EvidenceItem]:
    blocked = []
    if not text.strip():
        return blocked
    for sentence in _sentences(text):
        cleaned = _sanitize_text(sentence, max_text_chars=max_text_chars)
        evidence_type = _classify(cleaned)
        if evidence_type is None:
            continue
        blocked.append(
            EvidenceItem(
                evidence_type=evidence_type,
                text=_strip_leading_inline_section_heading(cleaned),
                page_number=page_number,
                quality_label="blocked",
                quality_score=0.2,
                quality_flags=("reference_section",),
            )
        )
    return blocked


def _ordered_unique(values: list[str]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
