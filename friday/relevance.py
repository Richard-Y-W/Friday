from __future__ import annotations

from dataclasses import replace
import re

from friday.discovery import Candidate
from friday.query_planning import plan_query


BIOMEDICAL_TERMS = {
    "antimicrobial": 18,
    "antibiotic": 14,
    "antifungal": 14,
    "resistance": 16,
    "resistant": 14,
    "susceptibility": 14,
    "pathogen": 10,
    "bacterial": 10,
    "bacteria": 10,
    "clinical": 8,
    "isolate": 8,
    "isolates": 8,
    "infection": 8,
    "pseudomonas": 10,
    "aeruginosa": 6,
    "staphylococcus": 8,
    "enterobacteriaceae": 8,
    "klebsiella": 8,
    "escherichia": 8,
    "fungal": 8,
}

MALDI_TERMS = {
    "maldi": 16,
    "maldi tof": 18,
    "maldi-tof": 18,
    "mass spectrometry": 12,
    "spectrometry": 8,
    "spectra": 6,
}

NLP_AMR_TERMS = {
    "abstract meaning representation": 70,
    "amr parsing": 50,
    "amr parser": 50,
    "amr-to-text": 50,
    "semantic graph": 32,
    "semantic graphs": 32,
    "natural language": 26,
    "text generation": 22,
    "neural machine translation": 22,
}

MATH_LANGUAGE_TERMS = {
    "mathematical linguistics": 50,
    "formal language theory": 44,
    "formal language": 28,
    "automata": 24,
    "grammar": 16,
    "grammars": 16,
    "syntax": 16,
    "semantics": 12,
    "algebra": 14,
    "algebraic": 14,
    "information theory": 8,
    "language acquisition": 12,
    "natural language": 10,
}

MATH_LANGUAGE_WRONG_DOMAIN_TERMS = {
    "biomedical": 20,
    "clinical": 22,
    "disease": 16,
    "health": 16,
    "medical": 16,
    "medicine": 16,
    "patient": 16,
    "patients": 16,
}

GENERIC_LANGUAGE_MODEL_TERMS = {
    "chatgpt": 20,
    "large language model": 18,
    "large language models": 18,
    "llm": 16,
    "llms": 16,
    "homework": 14,
    "classroom": 14,
    "education": 12,
}

PROVIDER_BONUS = {
    "pubmed": 14,
    "openalex": 8,
    "arxiv": 0,
}

PROVIDER_RANK = {
    "pubmed": 0,
    "openalex": 1,
    "arxiv": 2,
}


def score_candidate(query: str, candidate: Candidate) -> Candidate:
    query_intent = plan_query(query).intent
    text = _candidate_text(candidate)
    normalized = _normalize(text)
    score = 5 + PROVIDER_BONUS.get(candidate.provider, 0)
    reasons: list[str] = [f"provider={candidate.provider}"]

    biomedical_hits = _weighted_hits(normalized, BIOMEDICAL_TERMS)
    if biomedical_hits:
        score += sum(weight for _, weight in biomedical_hits)
        reasons.append("biomedical_terms=" + ",".join(term for term, _ in biomedical_hits[:5]))

    metadata_hits = _metadata_hits(candidate)
    if metadata_hits:
        score += sum(weight for _, weight in metadata_hits)
        reasons.append("metadata_terms=" + ",".join(term for term, _ in metadata_hits[:5]))

    maldi_hits = _weighted_hits(normalized, MALDI_TERMS)
    if maldi_hits:
        score += sum(weight for _, weight in maldi_hits)
        reasons.append("maldi_context=" + ",".join(term for term, _ in maldi_hits[:3]))

    if _query_asks_for_amr(query) and _has_amr_biomedical_context(normalized):
        score += 18
        reasons.append("amr_context")

    math_language_hits = _weighted_hits(normalized, MATH_LANGUAGE_TERMS)
    if query_intent == "mathematical_linguistics":
        if math_language_hits:
            score += sum(weight for _, weight in math_language_hits)
            reasons.append("math_language_context=" + ",".join(term for term, _ in math_language_hits[:5]))
        wrong_domain_hits = _weighted_hits(normalized, MATH_LANGUAGE_WRONG_DOMAIN_TERMS)
        generic_llm_hits = _weighted_hits(normalized, GENERIC_LANGUAGE_MODEL_TERMS)
        if wrong_domain_hits and not math_language_hits:
            score -= min(80, sum(weight for _, weight in wrong_domain_hits) + 18)
            reasons.append("math_language_wrong_domain_penalty")
        if generic_llm_hits and not math_language_hits:
            score -= min(60, sum(weight for _, weight in generic_llm_hits))
            reasons.append("generic_language_model_penalty")

    if candidate.doi or candidate.pmid:
        score += 4
        reasons.append("identifier_bonus")

    nlp_hits = _weighted_hits(normalized, NLP_AMR_TERMS)
    if query_intent == "nlp":
        if nlp_hits:
            score += sum(weight for _, weight in nlp_hits)
            reasons.append("nlp_amr_context=" + ",".join(term for term, _ in nlp_hits[:4]))
        collision_penalty = sum(weight for _, weight in biomedical_hits + metadata_hits + maldi_hits)
        if collision_penalty:
            score -= min(200, collision_penalty + (80 if not nlp_hits else 20))
            reasons.append("non_nlp_collision_penalty")
    elif _query_asks_for_amr(query) and nlp_hits:
        score -= sum(weight for _, weight in nlp_hits)
        reasons.append("nlp_amr_penalty=" + ",".join(term for term, _ in nlp_hits[:4]))

    if not biomedical_hits and not maldi_hits and not metadata_hits:
        reasons.append("not_biomedical")

    return replace(
        candidate,
        relevance_score=max(0, min(100, score)),
        relevance_reason="; ".join(reasons),
    )


def rank_candidates(query: str, candidates: list[Candidate]) -> list[Candidate]:
    scored = [score_candidate(query, candidate) for candidate in candidates]
    return [
        candidate
        for _, candidate in sorted(
            enumerate(scored),
            key=lambda item: (
                -(item[1].relevance_score or 0),
                PROVIDER_RANK.get(item[1].provider, 99),
                -(item[1].year or 0),
                item[0],
            ),
        )
    ]


def _candidate_text(candidate: Candidate) -> str:
    return " ".join(
        value
        for value in [
            candidate.title,
            candidate.abstract or "",
            candidate.journal or "",
            candidate.concepts or "",
            candidate.mesh_terms or "",
            candidate.doi or "",
            candidate.provider,
        ]
        if value
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _weighted_hits(normalized_text: str, weighted_terms: dict[str, int]) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for term, weight in weighted_terms.items():
        if _contains_term(normalized_text, term):
            hits.append((term, weight))
    return hits


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    return f" {normalized_term} " in f" {normalized_text} "


def _query_asks_for_amr(query: str) -> bool:
    normalized = _normalize(query)
    return _contains_term(normalized, "amr") or _contains_term(normalized, "antimicrobial resistance")


def _has_amr_biomedical_context(normalized_text: str) -> bool:
    return any(
        _contains_term(normalized_text, term)
        for term in [
            "antimicrobial",
            "antibiotic",
            "antifungal",
            "resistance",
            "resistant",
            "susceptibility",
        ]
    )


def _metadata_hits(candidate: Candidate) -> list[tuple[str, int]]:
    metadata = _normalize(" ".join([candidate.concepts or "", candidate.mesh_terms or "", candidate.journal or ""]))
    if not metadata:
        return []
    weighted_terms = {
        "drug resistance microbial": 28,
        "antimicrobial resistance": 28,
        "mass spectrometry": 18,
        "microbiology": 12,
        "clinical microbiology": 16,
        "anti bacterial agents": 12,
    }
    return _weighted_hits(metadata, weighted_terms)
