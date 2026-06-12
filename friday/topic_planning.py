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
    component_terms: tuple[tuple[str, tuple[str, ...]], ...] = ()
    required_component_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicCurationDecision:
    source: str | None
    title: str | None
    status: str
    eligible_for_deep_read: bool
    curation_score: int
    matched_query_terms: tuple[str, ...]
    matched_core_terms: tuple[str, ...]
    matched_positive_terms: tuple[str, ...]
    matched_negative_terms: tuple[str, ...]
    matched_topic_components: tuple[str, ...]
    missing_topic_components: tuple[str, ...]
    reason: str


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


def plan_topic_for_records(
    query: str,
    records: Iterable[Any],
    *,
    learned_profile_dir: Path | None = None,
) -> TopicProfile:
    profile = plan_topic(query, learned_profile_dir=learned_profile_dir)
    if profile.domain != "unknown":
        return profile
    mined = mine_metadata_profile(query, records)
    if mined.domain != "unknown":
        return mined
    return profile


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
        component_terms=tuple(
            _dedupe_components(
                component
                for profile in selected
                for component in (profile.component_terms or _component_terms(profile))
            )
        ),
        required_component_ids=tuple(
            profile.profile_id
            for profile in selected
            if profile.profile_id and _explicitly_activates_component(query, profile)
        ),
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
        component_terms=(
            (profile_id or "session." + _slug(cleaned), tuple(_dedupe([cleaned, *positive_terms]))),
        )
        if cleaned or positive_terms
        else (),
        required_component_ids=(),
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


def evaluate_topic_curation(record: Any, topic_profile: TopicProfile) -> TopicCurationDecision:
    if topic_profile.domain == "unknown":
        return TopicCurationDecision(
            source=getattr(record, "source", None) or getattr(record, "source_for_gate", None),
            title=getattr(record, "title", None),
            status="unknown_topic",
            eligible_for_deep_read=True,
            curation_score=0,
            matched_query_terms=(),
            matched_core_terms=(),
            matched_positive_terms=(),
            matched_negative_terms=(),
            matched_topic_components=(),
            missing_topic_components=(),
            reason="no_topic_profile",
        )

    text = _record_text(record)
    normalized = _normalize(text)
    query_terms = _query_focus_terms(topic_profile.query)
    matched_query_terms = tuple(term for term in query_terms if _contains_term(normalized, term))
    matched_core_terms = tuple(term for term in topic_profile.core_terms if _contains_term(normalized, term))
    matched_positive_terms = tuple(
        term for term in topic_profile.positive_terms if _contains_term(normalized, term)
    )
    matched_negative_terms = tuple(
        term for term in topic_profile.negative_terms if _contains_term(normalized, term)
    )
    matched_topic_components = _matched_topic_components(normalized, topic_profile)
    missing_topic_components = tuple(
        component_id
        for component_id in topic_profile.required_component_ids
        if component_id not in matched_topic_components
    )
    curation_score = (
        12 * len(matched_query_terms)
        + 10 * len(matched_core_terms)
        + 4 * len(matched_positive_terms)
        + 6 * len(matched_topic_components)
        - 20 * len(missing_topic_components)
        - 10 * len(matched_negative_terms)
    )

    has_topic_evidence = bool(matched_core_terms or matched_positive_terms)
    has_query_focus = bool(matched_query_terms)
    if missing_topic_components:
        status = "topic_mismatch"
        eligible = False
        reason = "missing_topic_component"
    elif matched_negative_terms and not has_topic_evidence and not has_query_focus:
        status = "topic_mismatch"
        eligible = False
        reason = "negative_terms_without_topic_evidence"
    elif not has_query_focus:
        status = "topic_mismatch"
        eligible = False
        reason = "missing_query_focus"
    elif not has_topic_evidence:
        status = "weak_topic_match"
        eligible = False
        reason = "missing_profile_terms"
    else:
        status = "topic_match"
        eligible = True
        reason = "query_and_profile_terms_matched"

    return TopicCurationDecision(
        source=getattr(record, "source", None) or getattr(record, "source_for_gate", None),
        title=getattr(record, "title", None),
        status=status,
        eligible_for_deep_read=eligible,
        curation_score=curation_score,
        matched_query_terms=matched_query_terms,
        matched_core_terms=matched_core_terms,
        matched_positive_terms=matched_positive_terms,
        matched_negative_terms=matched_negative_terms,
        matched_topic_components=matched_topic_components,
        missing_topic_components=missing_topic_components,
        reason=reason,
    )


def build_topic_audit(
    query: str,
    records: Iterable[Any],
    *,
    learned_profile_dir: Path | None = None,
    topic_profile: TopicProfile | None = None,
) -> dict[str, Any]:
    items = list(records)
    profile = topic_profile or plan_topic_for_records(query, items, learned_profile_dir=learned_profile_dir)
    decisions = [evaluate_topic_curation(item, profile) for item in items]
    return {
        "schema_version": "1.0",
        "artifact_type": "topic_audit",
        "profile": _topic_profile_artifact(profile),
        "curation": {
            "item_count": len(decisions),
            "eligible_for_deep_read_count": sum(1 for decision in decisions if decision.eligible_for_deep_read),
            "blocked_by_topic_count": sum(
                1 for decision in decisions if not decision.eligible_for_deep_read
            ),
            "status_counts": _status_counts(decisions),
            "policy": {
                "requires": [
                    "at least one non-stopword query-focus term",
                    "at least one core or positive topic-profile term",
                    "all explicitly activated topic components for composite queries",
                ],
                "human_override": "Human relevant labels may still force deep-read eligibility.",
            },
        },
        "items": [_topic_curation_artifact(decision) for decision in decisions],
    }


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
        component_terms=((profile_id, tuple(_profile_component_terms(payload))),),
        required_component_ids=(profile_id,),
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


def _profile_component_terms(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _dedupe(
            [
                *_terms(payload.get("core_terms")),
                *_terms(payload.get("positive_terms")),
                *_terms(payload.get("match_terms")),
            ]
        )
    )


def _component_terms(profile: TopicProfile) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not profile.profile_id:
        return ()
    return (
        (
            profile.profile_id,
            tuple(_dedupe([*profile.core_terms, *profile.positive_terms, *profile.match_terms])),
        ),
    )


def _dedupe_components(
    components: Iterable[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
    deduped: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for component_id, terms in components:
        if not component_id or component_id in seen:
            continue
        cleaned_terms = tuple(_dedupe(terms))
        if not cleaned_terms:
            continue
        seen.add(component_id)
        deduped.append((component_id, cleaned_terms))
    return deduped


def _explicitly_activates_component(query: str, profile: TopicProfile) -> bool:
    normalized = _normalize(query)
    for term in [*profile.match_terms, *profile.core_terms]:
        if _contains_term(normalized, term):
            return True
    return False


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
            if _is_useful_metadata_term(term):
                counts[term] += 2
        for term in _text_metadata_terms(record):
            if _is_useful_metadata_term(term):
                counts[term] += 1
    return counts


def _is_useful_metadata_term(term: str) -> bool:
    normalized = _normalize(term)
    if not normalized or normalized in _GENERIC_METADATA_TERMS:
        return False
    tokens = normalized.split()
    if len(tokens) == 1 and tokens[0] in _GENERIC_METADATA_TERMS:
        return False
    return True


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


def _matched_topic_components(normalized_text: str, profile: TopicProfile) -> tuple[str, ...]:
    matched: list[str] = []
    for component_id, terms in profile.component_terms:
        if any(_contains_term(normalized_text, term) for term in terms):
            matched.append(component_id)
    return tuple(matched)


def _topic_profile_artifact(profile: TopicProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "topic_ids": list(profile.topic_ids),
        "domain": profile.domain,
        "reason": profile.reason,
        "core_terms": list(profile.core_terms),
        "positive_terms": list(profile.positive_terms),
        "negative_terms": list(profile.negative_terms),
        "search_queries": list(profile.search_queries),
        "source_preferences": list(profile.source_preferences),
        "evidence_policy_hint": profile.evidence_policy_hint,
        "match_terms": list(profile.match_terms),
        "required_terms": list(profile.required_terms),
        "component_terms": [
            {"component_id": component_id, "terms": list(terms)}
            for component_id, terms in profile.component_terms
        ],
        "required_component_ids": list(profile.required_component_ids),
    }


def _topic_curation_artifact(decision: TopicCurationDecision) -> dict[str, Any]:
    return {
        "source": decision.source,
        "title": decision.title,
        "status": decision.status,
        "eligible_for_deep_read": decision.eligible_for_deep_read,
        "curation_score": decision.curation_score,
        "matched_query_terms": list(decision.matched_query_terms),
        "matched_core_terms": list(decision.matched_core_terms),
        "matched_positive_terms": list(decision.matched_positive_terms),
        "matched_negative_terms": list(decision.matched_negative_terms),
        "matched_topic_components": list(decision.matched_topic_components),
        "missing_topic_components": list(decision.missing_topic_components),
        "reason": decision.reason,
    }


def _status_counts(decisions: list[TopicCurationDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    return counts


def _record_text(record: Any) -> str:
    return " ".join(
        value
        for value in [
            getattr(record, "title", None),
            getattr(record, "abstract", None),
            getattr(record, "journal", None),
            getattr(record, "concepts", None),
            getattr(record, "mesh_terms", None),
        ]
        if value
    )


def _query_focus_terms(query: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", query)
        if len(token) >= 3 and token.lower() not in _STOP_TERMS
    ]
    phrase_terms = []
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase_terms.append(" ".join(tokens[index : index + size]))
    return tuple(_dedupe([*phrase_terms, *tokens]))


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

_GENERIC_METADATA_TERMS = {
    "agricultural and biological sciences",
    "biology",
    "biochemistry",
    "chemistry",
    "computer science",
    "earth and planetary sciences",
    "economics",
    "electrical engineering",
    "engineering",
    "environmental science",
    "materials science",
    "mathematics",
    "medicine",
    "physics",
    "psychology",
    "social sciences",
    "statistics",
}
