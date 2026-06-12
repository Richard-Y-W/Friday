from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_SEED_PROFILE_DIR = Path(__file__).resolve().parent / "topic_profiles" / "seed"
LEARNED_SCHEMA_VERSION = "1.0"
MAX_PROFILE_TERMS = 12


@dataclass(frozen=True)
class TopicProfile:
    query: str
    domain: str
    core_terms: tuple[str, ...]
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    search_queries: tuple[str, ...]
    reason: str
    profile_id: str | None = None
    topic_ids: tuple[str, ...] = ()
    source_preferences: tuple[str, ...] = ()
    evidence_policy_hint: str | None = None
    match_terms: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()


def plan_topic(query: str, *, learned_profile_dir: Path | None = None) -> TopicProfile:
    cleaned = " ".join(query.split())
    profiles = [*load_seed_profiles()]
    if learned_profile_dir is not None:
        profiles.extend(load_learned_profiles(learned_profile_dir))

    matches = [
        profile
        for profile in profiles
        if _profile_match_score(cleaned, profile) > 0
    ]
    if not matches:
        return _unknown_profile(cleaned)
    return compose_topic_profiles(cleaned, matches)


def load_seed_profiles(profile_dir: Path = DEFAULT_SEED_PROFILE_DIR) -> list[TopicProfile]:
    return _load_profile_dir(profile_dir, default_reason="seed_profile")


def load_learned_profiles(memory_dir: Path) -> list[TopicProfile]:
    return _load_profile_dir(memory_dir, default_reason="learned_profile")


def compose_topic_profiles(query: str, profiles: Iterable[TopicProfile]) -> TopicProfile:
    selected = list(profiles)
    if not selected:
        return _unknown_profile(query)

    topic_ids = tuple(profile.profile_id for profile in selected if profile.profile_id)
    domains = _dedupe([profile.domain for profile in selected if profile.domain and profile.domain != "unknown"])
    hints = _dedupe([profile.evidence_policy_hint for profile in selected if profile.evidence_policy_hint])
    search_queries = _dedupe(
        [
            query,
            *[
                search_query
                for profile in selected
                for search_query in profile.search_queries
            ],
        ]
    )
    return TopicProfile(
        query=query,
        profile_id="composed." + "+".join(topic_ids) if topic_ids else "composed",
        topic_ids=topic_ids,
        domain=domains[0] if len(domains) == 1 else "+".join(domains),
        core_terms=tuple(_dedupe([term for profile in selected for term in profile.core_terms])),
        positive_terms=tuple(_dedupe([term for profile in selected for term in profile.positive_terms])),
        negative_terms=tuple(_dedupe([term for profile in selected for term in profile.negative_terms])),
        search_queries=tuple(search_queries),
        source_preferences=tuple(_dedupe([source for profile in selected for source in profile.source_preferences])),
        evidence_policy_hint=hints[0] if len(hints) == 1 else "+".join(hints) if hints else None,
        match_terms=tuple(_dedupe([term for profile in selected for term in profile.match_terms])),
        required_terms=tuple(_dedupe([term for profile in selected for term in profile.required_terms])),
        reason="composed_profiles:" + ",".join(topic_ids),
    )


def mine_metadata_profile(
    query: str,
    records: Iterable[Any],
    *,
    profile_id: str | None = None,
    domain: str = "session",
    min_count: int = 2,
) -> TopicProfile:
    cleaned = " ".join(query.split())
    term_counts = _metadata_term_counts(records)
    positive_terms = [
        term
        for term, count in term_counts.most_common(MAX_PROFILE_TERMS)
        if count >= min_count
    ]
    search_queries = _metadata_search_queries(cleaned, positive_terms)
    return TopicProfile(
        query=cleaned,
        profile_id=profile_id or "session." + _slug(cleaned),
        topic_ids=(profile_id or "session." + _slug(cleaned),),
        domain=domain if positive_terms else "unknown",
        core_terms=(cleaned,) if cleaned else (),
        positive_terms=tuple(positive_terms),
        negative_terms=(),
        search_queries=tuple(search_queries),
        source_preferences=(),
        evidence_policy_hint=None,
        match_terms=(cleaned,) if cleaned else (),
        reason="metadata_mined_profile" if positive_terms else "no_repeated_metadata_terms",
    )


def update_topic_memory(
    memory_dir: Path,
    query: str,
    *,
    relevant_records: Iterable[Any],
    irrelevant_records: Iterable[Any] = (),
) -> TopicProfile:
    memory_dir.mkdir(parents=True, exist_ok=True)
    cleaned = " ".join(query.split())
    profile_id = "learned." + _slug(cleaned)
    relevant_counts = _metadata_term_counts(relevant_records)
    irrelevant_counts = _metadata_term_counts(irrelevant_records)
    positive_terms = {
        term: count
        for term, count in relevant_counts.most_common(MAX_PROFILE_TERMS)
        if count > 0
    }
    negative_terms = {
        term: count
        for term, count in irrelevant_counts.most_common(MAX_PROFILE_TERMS)
        if count > 0 and term not in positive_terms
    }
    payload = {
        "schema_version": LEARNED_SCHEMA_VERSION,
        "profile_id": profile_id,
        "profile_type": "learned",
        "domain": "learned",
        "match_terms": [cleaned],
        "required_terms": [],
        "core_terms": [cleaned],
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "search_queries": _metadata_search_queries(cleaned, list(positive_terms)),
        "source_preferences": [],
        "evidence_policy_hint": "learned",
    }
    path = memory_dir / f"{profile_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _profile_from_mapping(payload, default_reason="learned_profile")


def _load_profile_dir(profile_dir: Path, *, default_reason: str) -> list[TopicProfile]:
    if not profile_dir.exists():
        return []
    profiles = []
    for path in sorted(profile_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles.append(_profile_from_mapping(payload, default_reason=default_reason))
    return profiles


def _profile_from_mapping(payload: dict[str, Any], *, default_reason: str) -> TopicProfile:
    profile_id = str(payload["profile_id"])
    return TopicProfile(
        query="",
        profile_id=profile_id,
        topic_ids=(profile_id,),
        domain=str(payload.get("domain") or "unknown"),
        core_terms=tuple(_terms(payload.get("core_terms"))),
        positive_terms=tuple(_terms(payload.get("positive_terms"))),
        negative_terms=tuple(_terms(payload.get("negative_terms"))),
        search_queries=tuple(_terms(payload.get("search_queries"))),
        source_preferences=tuple(_terms(payload.get("source_preferences"))),
        evidence_policy_hint=str(payload["evidence_policy_hint"]) if payload.get("evidence_policy_hint") else None,
        match_terms=tuple(_terms(payload.get("match_terms"))),
        required_terms=tuple(_terms(payload.get("required_terms"))),
        reason=str(payload.get("reason") or default_reason),
    )


def _terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key in value if str(key).strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _profile_match_score(query: str, profile: TopicProfile) -> int:
    normalized = _normalize(query)
    score = 0
    if profile.required_terms and all(_contains_term(normalized, term) for term in profile.required_terms):
        score += 4
    for term in profile.match_terms or profile.core_terms:
        if _contains_term(normalized, term):
            score += 2
    return score


def _metadata_term_counts(records: Iterable[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for term in _controlled_metadata_terms(record):
            counts[term] += 2
        for term in _text_metadata_terms(record):
            counts[term] += 1
    return counts


def _controlled_metadata_terms(record: Any) -> list[str]:
    terms: list[str] = []
    for field in ("concepts", "mesh_terms"):
        value = getattr(record, field, None) or ""
        for term in re.split(r"[;,]", value):
            cleaned = " ".join(term.split())
            if cleaned and len(cleaned) >= 3:
                terms.append(cleaned)
    journal = " ".join((getattr(record, "journal", None) or "").split())
    if journal:
        terms.append(journal)
    return terms


def _text_metadata_terms(record: Any) -> list[str]:
    text = " ".join(
        value
        for value in [
            getattr(record, "title", None),
            getattr(record, "abstract", None),
        ]
        if value
    )
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
        if len(token) > 2 and token.lower() not in _STOP_TERMS
    ]
    terms = []
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase_tokens = tokens[index : index + size]
            if any(token.lower() in _STOP_TERMS for token in phrase_tokens):
                continue
            terms.append(" ".join(phrase_tokens))
    return terms


def _metadata_search_queries(query: str, positive_terms: list[str]) -> list[str]:
    queries = [query] if query else []
    for term in positive_terms[:5]:
        if _normalize(term) == _normalize(query):
            continue
        queries.append(f"{query} {term}".strip())
    return _dedupe(queries)


def _unknown_profile(query: str) -> TopicProfile:
    return TopicProfile(
        query=query,
        profile_id="unknown",
        topic_ids=(),
        domain="unknown",
        core_terms=(query,) if query else (),
        positive_terms=(),
        negative_terms=(),
        search_queries=(query,) if query else (),
        source_preferences=(),
        evidence_policy_hint=None,
        match_terms=(),
        required_terms=(),
        reason="no_topic_profile",
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    return f" {normalized_term} " in f" {normalized_text} "


def _dedupe(values: Iterable[str | None]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        cleaned = " ".join(str(value).split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "topic"


_STOP_TERMS = {
    "about",
    "after",
    "analysis",
    "and",
    "are",
    "based",
    "for",
    "from",
    "how",
    "into",
    "paper",
    "study",
    "the",
    "this",
    "using",
    "with",
}
