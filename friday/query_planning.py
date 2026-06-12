from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from friday.acronyms import resolve_acronyms
from friday.topic_planning import plan_topic


@dataclass(frozen=True)
class ResolvedAcronym:
    acronym: str
    meaning: str
    intent: str
    reason: str
    rejected_meanings: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    intent: str
    expanded_queries: tuple[str, ...]
    resolved_acronyms: tuple[ResolvedAcronym, ...]


AMR_MEANINGS = {
    "biomedical": "antimicrobial resistance",
    "nlp": "abstract meaning representation",
    "computational": "adaptive mesh refinement",
}

MATH_LANGUAGE_CONTEXT = {
    "math",
    "mathematics",
    "mathematical",
    "computation",
    "computational",
    "formal",
    "algebra",
    "automata",
    "information",
    "entropy",
}

LANGUAGE_CONTEXT = {
    "language",
    "languages",
    "linguistic",
    "linguistics",
    "grammar",
    "grammars",
    "syntax",
    "semantic",
    "semantics",
}

MATH_LANGUAGE_EXPANSIONS = (
    "mathematical linguistics",
    "formal language theory natural language",
    "information theory language",
    "mathematical models of language acquisition",
)

LEADING_ASSISTANT_PREFIX = re.compile(r"^\s*(?:friday|jarvis)\b[\s,:-]*", re.IGNORECASE)

CONVERSATIONAL_QUERY_PREFIXES = (
    re.compile(r"^\s*tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^\s*tell\s+me\s+", re.IGNORECASE),
    re.compile(r"^\s*can\s+you\s+tell\s+me\s+about\s+", re.IGNORECASE),
    re.compile(r"^\s*can\s+you\s+tell\s+me\s+", re.IGNORECASE),
    re.compile(r"^\s*give\s+me\s+(?:a\s+)?(?:report|summary|overview)\s+(?:about|on|of)\s+", re.IGNORECASE),
    re.compile(r"^\s*write\s+(?:a\s+)?(?:report|summary|overview)\s+(?:about|on|of)\s+", re.IGNORECASE),
    re.compile(r"^\s*summarize\s+", re.IGNORECASE),
    re.compile(r"^\s*explain\s+", re.IGNORECASE),
    re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(?:the\s+)?", re.IGNORECASE),
    re.compile(r"^\s*what['’]s\s+(?:the\s+)?", re.IGNORECASE),
)


def plan_query(query: str, *, learned_profile_dir: Path | None = None) -> QueryPlan:
    original = normalize_research_query(query)
    normalized = _normalize(original)
    resolved = [
        ResolvedAcronym(
            acronym=item.acronym,
            meaning=item.meaning,
            intent=item.intent,
            reason=item.reason,
            rejected_meanings=item.rejected_meanings,
        )
        for item in resolve_acronyms(original)
    ]

    if not resolved and _is_math_language_query(normalized):
        return QueryPlan(original, "mathematical_linguistics", MATH_LANGUAGE_EXPANSIONS, ())

    if not resolved:
        topic_profile = plan_topic(original, learned_profile_dir=learned_profile_dir)
        if topic_profile.domain != "unknown":
            return QueryPlan(original, topic_profile.domain, topic_profile.search_queries, ())
        return QueryPlan(original, "unknown", (original,), ())

    intent = _dominant_intent(resolved)
    expanded = _expanded_queries(original, resolved, intent)
    return QueryPlan(original, intent, expanded, tuple(resolved))


def render_acronym_expansions(plan: QueryPlan) -> str | None:
    if not plan.resolved_acronyms:
        return None
    return "; ".join(
        f"{item.acronym}=unresolved" if item.intent == "unknown" else f"{item.acronym}={item.meaning}"
        for item in plan.resolved_acronyms
    )


def _expanded_queries(original: str, resolved: list[ResolvedAcronym], intent: str) -> tuple[str, ...]:
    direct = original
    for item in resolved:
        if item.intent == "unknown":
            continue
        direct = _replace_acronym(direct, item.acronym, item.meaning)

    expansions = [direct]
    meanings = {item.acronym: item.meaning for item in resolved}
    if intent == "biomedical" and meanings.get("AMR") == AMR_MEANINGS["biomedical"]:
        if _has_token(original, "MALDI"):
            expansions.extend(
                [
                    "MALDI-TOF antibiotic resistance",
                    "MALDI-TOF antimicrobial susceptibility",
                ]
            )
        else:
            expansions.extend(["antimicrobial resistance", "antibiotic resistance"])
    elif intent == "nlp" and meanings.get("AMR") == AMR_MEANINGS["nlp"]:
        expansions.append(_replace_acronym(original, "AMR", "abstract meaning representation"))
    elif intent == "computational" and meanings.get("AMR") == AMR_MEANINGS["computational"]:
        expansions.append(_replace_acronym(original, "AMR", "adaptive mesh refinement"))

    return tuple(_dedupe(expansions))


def normalize_research_query(query: str) -> str:
    cleaned = " ".join(query.split())
    if not cleaned:
        return ""
    cleaned = LEADING_ASSISTANT_PREFIX.sub("", cleaned).strip()
    for pattern in CONVERSATIONAL_QUERY_PREFIXES:
        updated = pattern.sub("", cleaned).strip()
        if updated != cleaned:
            cleaned = updated
            break
    return cleaned or " ".join(query.split())


def _dominant_intent(resolved: list[ResolvedAcronym]) -> str:
    for intent in ("biomedical", "nlp", "ml", "computational"):
        if any(item.intent == intent for item in resolved):
            return intent
    return "unknown"


def _replace_acronym(text: str, acronym: str, replacement: str) -> str:
    return re.sub(rf"\b{re.escape(acronym)}\b", replacement, text, flags=re.IGNORECASE)


def _contains_acronym(text: str, acronym: str) -> bool:
    return re.search(rf"\b{re.escape(acronym)}\b", text, flags=re.IGNORECASE) is not None


def _has_token(text: str, token: str) -> bool:
    return _contains_acronym(text, token)


def _has_any(normalized_query: str, terms: set[str]) -> bool:
    return any(_contains_normalized(normalized_query, term) for term in terms)


def _is_math_language_query(normalized_query: str) -> bool:
    return _has_any(normalized_query, MATH_LANGUAGE_CONTEXT) and _has_any(normalized_query, LANGUAGE_CONTEXT)


def _contains_normalized(normalized_query: str, term: str) -> bool:
    normalized_term = _normalize(term)
    return f" {normalized_term} " in f" {normalized_query} "


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique
