from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re

from jarvis_research.llm_labeling import (
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    LlmLabelResult,
    OpenAIResponsesLabelClient,
)
from jarvis_research.query_planning import plan_query
from jarvis_research.storage import BatchItemRecord, JarvisStore, SCREENING_LABEL_CHOICES, ScreeningLabelRecord


STOP_WORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "as",
    "at",
    "by",
    "can",
    "describe",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "important",
    "importance",
    "in",
    "into",
    "is",
    "jarvis",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "the",
    "to",
    "using",
    "what",
    "why",
    "with",
}

MATH_LANGUAGE_MATH_TOKENS = {
    "algebra",
    "algebraic",
    "automata",
    "coding",
    "entropy",
    "formal",
    "information",
    "math",
    "mathematical",
    "mathematics",
    "theory",
}

MATH_LANGUAGE_STRONG_MATH_TOKENS = {
    "algebra",
    "algebraic",
    "automata",
    "formal",
    "math",
    "mathematical",
    "mathematics",
}

MATH_LANGUAGE_LANGUAGE_TOKENS = {
    "acquisition",
    "grammar",
    "grammars",
    "language",
    "languages",
    "linguistic",
    "linguistics",
    "natural",
    "semantic",
    "semantics",
    "syntax",
}

MATH_LANGUAGE_STRUCTURAL_TOKENS = {
    "algebra",
    "algebraic",
    "automata",
    "formal",
    "grammar",
    "grammars",
    "linguistic",
    "linguistics",
    "semantic",
    "semantics",
    "syntax",
}

GENERIC_LANGUAGE_MODEL_TOKENS = {
    "chatgpt",
    "classroom",
    "education",
    "educational",
    "gpt",
    "homework",
    "llm",
    "llms",
    "model",
    "models",
    "student",
    "students",
}

WRONG_DOMAIN_FOR_MATH_LANGUAGE = {
    "biomedical",
    "clinical",
    "clinician",
    "clinicians",
    "disease",
    "health",
    "medical",
    "medicine",
    "patient",
    "patients",
}

MATH_LANGUAGE_METHOD_TOKENS = {
    "applied",
    "bayesian",
    "computational",
    "computer",
    "frequentist",
    "generation",
    "hypothesis",
    "linguistic",
    "linguistics",
    "machine",
    "methods",
    "morpheme",
    "morphological",
    "processing",
    "program",
    "psycholinguistics",
    "question",
    "relies",
    "statistical",
    "statistics",
    "techniques",
}

MALDI_AMR_OFF_DOMAIN_TOKENS = {
    "chemical",
    "chiral",
    "clusters",
    "condensate",
    "fermions",
    "lattice",
    "magnetic",
    "mapping",
    "mri",
    "oxychloride",
    "quantitative",
    "quantum",
    "superconducting",
    "superconductor",
    "tantalum",
    "topological",
    "wilson",
}

BIOMEDICAL_SURVEILLANCE_OFF_DOMAIN_TOKENS = {
    "5g",
    "air",
    "aviation",
    "color",
    "communications",
    "frames",
    "histogram",
    "keyframes",
    "mobility",
    "navigation",
    "rgb",
    "tactical",
    "traffic",
    "video",
}

ESBL_CRE_CLINICAL_TOKENS = {
    "care",
    "clinical",
    "critically",
    "developing",
    "hospital",
    "implications",
    "patients",
    "patterns",
    "prevalence",
    "stewardship",
    "surveillance",
    "tertiary",
    "uropathogens",
}

ESBL_CRE_RESISTANCE_TOKENS = {
    "antibiotic",
    "antibiotics",
    "antimicrobial",
    "carbapenem",
    "carbapenemase",
    "cre",
    "drug",
    "esbl",
    "lactamase",
    "resistance",
    "resistant",
}

CLINICAL_STEWARDSHIP_RELEVANCE_TOKENS = {
    "antibiotic",
    "antimicrobial",
    "bacterial",
    "biomarker",
    "biomarkers",
    "care",
    "clinical",
    "critical",
    "decision",
    "guidelines",
    "icu",
    "infection",
    "infections",
    "intensive",
    "management",
    "neonatal",
    "neonates",
    "pediatric",
    "procalcitonin",
    "pyelonephritis",
    "sepsis",
    "septic",
    "shock",
    "stewardship",
    "treatment",
}

SIMULATION_PACKAGE_TOKENS = {
    "api",
    "environment",
    "functions",
    "gymnasium",
    "optimization",
    "package",
    "policy",
    "python",
    "reinforcement",
    "reward",
    "simulation",
    "simulator",
}


@dataclass(frozen=True)
class ScreeningRecommendation:
    item: BatchItemRecord
    score: int
    base_score: int
    relevant_overlap: tuple[str, ...]
    irrelevant_overlap: tuple[str, ...]


@dataclass(frozen=True)
class AutoLabelDecision:
    item: BatchItemRecord
    label: str
    confidence: float
    rationale: str
    signals: str


@dataclass(frozen=True)
class AutoLabelBatchResult:
    decisions: list[AutoLabelDecision]
    applied_count: int
    skipped_human_count: int
    skipped_low_confidence_count: int
    skipped_error_count: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class LlmReviewQueueItem:
    item: BatchItemRecord
    score: int
    reason: str
    label: str | None
    label_source: str | None
    confidence: float | None


def build_screening_label_summary(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
) -> dict[str, object]:
    items_by_normalized = {item.normalized: item for item in items}
    counts = {choice: 0 for choice in SCREENING_LABEL_CHOICES}
    label_rows = []
    for label in labels:
        counts[label.label] = counts.get(label.label, 0) + 1
        item = items_by_normalized.get(label.normalized)
        label_rows.append(_screening_label_row(label, item))
    return {
        "schema_version": "1.0",
        "artifact_type": "screening_label_summary",
        "counts": counts,
        "labeled_count": len(label_rows),
        "rules": {
            "relevant": "Prioritized for resumed deep reads and allowed to bypass the minimum relevance threshold.",
            "maybe": "Kept eligible after relevant labels and before unlabeled papers with the same score.",
            "irrelevant": "Excluded from resumed deep reads.",
        },
        "labels": label_rows,
    }


def rank_deep_read_items(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    *,
    min_relevance: int,
) -> list[BatchItemRecord]:
    labels_by_normalized = {label.normalized: label for label in labels}
    eligible: list[tuple[int, BatchItemRecord]] = []
    for index, item in enumerate(items):
        if not item.allowed:
            continue
        label_record = labels_by_normalized.get(item.normalized)
        label = label_record.label if label_record else None
        if label == "irrelevant":
            continue
        score = item.relevance_score or 0
        if label != "relevant" and min_relevance > 0 and score < min_relevance:
            continue
        eligible.append((index, item))

    return [
        item
        for _, item in sorted(
            eligible,
            key=lambda indexed: (
                _label_bucket(labels_by_normalized.get(indexed[1].normalized)),
                -(indexed[1].relevance_score or 0),
                indexed[0],
            ),
        )
    ]


def build_llm_review_queue(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    *,
    limit: int = 50,
) -> list[LlmReviewQueueItem]:
    labels_by_normalized = {label.normalized: label for label in labels}
    scored: list[tuple[int, LlmReviewQueueItem]] = []
    for index, item in enumerate(items):
        if not item.allowed:
            continue
        label_record = labels_by_normalized.get(item.normalized)
        if label_record and label_record.label_source == "human":
            continue
        queue_item = _score_llm_review_candidate(item, label_record)
        if queue_item.score <= 0:
            continue
        scored.append((index, queue_item))

    ordered = [
        queue_item
        for _index, queue_item in sorted(
            scored,
            key=lambda indexed: (
                -indexed[1].score,
                -(indexed[1].item.relevance_score or 0),
                indexed[0],
            ),
        )
    ]
    return _select_diverse_queue(ordered, limit=max(0, limit))


def auto_label_batch_items(
    store: JarvisStore,
    batch_id: str,
    *,
    query: str | None = None,
    limit: int = 1000,
    apply: bool = False,
    min_confidence: float = 0.0,
    provider: str = "heuristic",
    model: str | None = None,
    llm_client: object | None = None,
    api_base_url: str = DEFAULT_OPENAI_BASE_URL,
    api_key_env: str = DEFAULT_OPENAI_API_KEY_ENV,
    review_queue: list[LlmReviewQueueItem] | None = None,
) -> AutoLabelBatchResult:
    if provider not in {"heuristic", "llm"}:
        raise ValueError(f"unsupported auto-label provider: {provider}")
    batch = store.get_batch(batch_id)
    items = [item for item in store.list_batch_items(batch_id) if item.allowed]
    labels_by_normalized = {label.normalized: label for label in store.list_screening_labels(batch_id)}
    query_tokens = _query_tokens(query or batch.query or "")
    decisions: list[AutoLabelDecision] = []
    applied_count = 0
    skipped_human_count = 0
    skipped_low_confidence_count = 0
    skipped_error_count = 0
    errors: list[str] = []
    selected_model = model or DEFAULT_OPENAI_MODEL
    queue_by_normalized = {entry.item.normalized: entry for entry in review_queue or []}
    if provider == "llm" and review_queue is not None:
        items_by_normalized = {item.normalized: item for item in items}
        selected_items = [
            items_by_normalized[entry.item.normalized]
            for entry in review_queue[: max(0, limit)]
            if entry.item.normalized in items_by_normalized
        ]
    else:
        selected_items = items[: max(0, limit)]
    if provider == "llm" and llm_client is None:
        try:
            llm_client = OpenAIResponsesLabelClient(
                model=selected_model,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
            )
        except Exception as exc:
            remaining = sum(
                1
                for item in selected_items
                if not (
                    (existing := labels_by_normalized.get(item.normalized))
                    and existing.label_source == "human"
                )
            )
            return AutoLabelBatchResult(
                decisions=[],
                applied_count=0,
                skipped_human_count=len(selected_items) - remaining,
                skipped_low_confidence_count=0,
                skipped_error_count=remaining,
                errors=(str(exc),),
            )

    for item in selected_items:
        existing = labels_by_normalized.get(item.normalized)
        if existing and existing.label_source == "human":
            skipped_human_count += 1
            continue
        try:
            if provider == "llm":
                decision = _llm_auto_label_item(
                    item,
                    query=query or batch.query,
                    model=selected_model,
                    llm_client=llm_client,
                )
            else:
                decision = _auto_label_item(item, query_tokens)
        except Exception as exc:
            skipped_error_count += 1
            errors.append(str(exc))
            continue
        if provider == "llm" and item.normalized in queue_by_normalized:
            decision = _with_queue_signals(decision, queue_by_normalized[item.normalized])
        decisions.append(decision)
        if decision.confidence < min_confidence:
            skipped_low_confidence_count += 1
            continue
        if apply:
            stored = store.set_screening_label(
                batch_id,
                item.source,
                decision.label,
                label_source="agent",
                confidence=decision.confidence,
                rationale=decision.rationale,
                signals=decision.signals,
                overwrite_human=False,
            )
            if stored is None:
                skipped_human_count += 1
            else:
                applied_count += 1

    return AutoLabelBatchResult(
        decisions=decisions,
        applied_count=applied_count,
        skipped_human_count=skipped_human_count,
        skipped_low_confidence_count=skipped_low_confidence_count,
        skipped_error_count=skipped_error_count,
        errors=tuple(errors[:10]),
    )


def recommend_unlabeled_items(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
    *,
    limit: int = 10,
) -> list[ScreeningRecommendation]:
    labels_by_normalized = {label.normalized: label for label in labels}
    relevant_profile = _label_profile(items, labels_by_normalized, {"relevant": 2, "maybe": 1})
    irrelevant_profile = _label_profile(items, labels_by_normalized, {"irrelevant": 2})
    recommendations: list[ScreeningRecommendation] = []

    for item in items:
        if not item.allowed or item.normalized in labels_by_normalized:
            continue
        tokens = _item_tokens(item)
        relevant_overlap = tuple(sorted(tokens.intersection(relevant_profile)))
        irrelevant_overlap = tuple(sorted(tokens.intersection(irrelevant_profile)))
        base_score = item.relevance_score or 0
        score = (
            base_score
            + 6 * sum(relevant_profile[token] for token in relevant_overlap)
            - 6 * sum(irrelevant_profile[token] for token in irrelevant_overlap)
        )
        recommendations.append(
            ScreeningRecommendation(
                item=item,
                score=score,
                base_score=base_score,
                relevant_overlap=relevant_overlap,
                irrelevant_overlap=irrelevant_overlap,
            )
        )

    return sorted(
        recommendations,
        key=lambda recommendation: (
            -recommendation.score,
            -recommendation.base_score,
            recommendation.item.source,
        ),
    )[: max(0, limit)]


def _score_llm_review_candidate(
    item: BatchItemRecord,
    label: ScreeningLabelRecord | None,
) -> LlmReviewQueueItem:
    relevance = item.relevance_score or 0
    reasons: list[str] = []
    score = relevance
    label_value = label.label if label else None
    confidence = label.confidence if label else None
    label_source = label.label_source if label else None

    if label_value == "maybe":
        score += 120
        if relevance >= 40:
            reasons.append("heuristic_maybe_high_relevance")
        else:
            reasons.append("heuristic_maybe")
    elif label_value == "irrelevant" and relevance >= 40:
        score += 110
        reasons.append("heuristic_irrelevant_high_relevance")
    elif label is None:
        score += 80
        if relevance >= 40:
            reasons.append("unlabeled_high_relevance")
        else:
            reasons.append("unlabeled")
    elif label_value == "relevant" and confidence is not None and confidence < 0.75:
        score += 70
        reasons.append("low_confidence_relevant")
    else:
        score += 5
        reasons.append(f"agent_{label_value or 'unlabeled'}")

    if confidence is not None and confidence < 0.7:
        score += 30
        reasons.append("low_confidence_label")

    return LlmReviewQueueItem(
        item=item,
        score=score,
        reason="+".join(dict.fromkeys(reasons)),
        label=label_value,
        label_source=label_source,
        confidence=confidence,
    )


def _select_diverse_queue(items: list[LlmReviewQueueItem], *, limit: int) -> list[LlmReviewQueueItem]:
    if limit <= 0:
        return []
    selected: list[LlmReviewQueueItem] = []
    deferred: list[LlmReviewQueueItem] = []
    seen_keys: set[str] = set()
    for item in items:
        key = _diversity_key(item.item)
        if key not in seen_keys:
            selected.append(item)
            seen_keys.add(key)
        else:
            deferred.append(item)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for item in deferred:
            if item not in selected:
                selected.append(item)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _diversity_key(item: BatchItemRecord) -> str:
    return item.domain or item.provider or item.normalized.split("/", 1)[0]


def _with_queue_signals(decision: AutoLabelDecision, queue_item: LlmReviewQueueItem) -> AutoLabelDecision:
    signals = (
        f"{decision.signals};"
        f"review_queue_score={queue_item.score};"
        f"review_queue_reason={queue_item.reason}"
    )
    return replace(decision, signals=signals)


def _screening_label_row(
    label: ScreeningLabelRecord,
    item: BatchItemRecord | None,
) -> dict[str, object]:
    return {
        "source": label.source,
        "normalized": label.normalized,
        "label": label.label,
        "note": label.note,
        "label_source": label.label_source,
        "confidence": label.confidence,
        "rationale": label.rationale,
        "signals": label.signals,
        "created_at": label.created_at,
        "updated_at": label.updated_at,
        "title": item.title if item else None,
        "provider": item.provider if item else None,
        "allowed": item.allowed if item else None,
        "reason": item.reason if item else None,
        "doi": item.doi if item else None,
        "pmid": item.pmid if item else None,
        "pmcid": item.pmcid if item else None,
        "arxiv_id": item.arxiv_id if item else None,
        "year": item.year if item else None,
        "relevance_score": item.relevance_score if item else None,
        "relevance_reason": item.relevance_reason if item else None,
    }


def _auto_label_item(item: BatchItemRecord, query_tokens: set[str]) -> AutoLabelDecision:
    item_tokens = _item_tokens(item)
    overlap = sorted(item_tokens.intersection(query_tokens))
    relevance = item.relevance_score or 0
    math_language_query = _is_math_language_query_tokens(query_tokens)
    math_overlap = sorted(item_tokens.intersection(MATH_LANGUAGE_MATH_TOKENS))
    strong_math_overlap = sorted(item_tokens.intersection(MATH_LANGUAGE_STRONG_MATH_TOKENS))
    language_overlap = sorted(item_tokens.intersection(MATH_LANGUAGE_LANGUAGE_TOKENS))
    structural_overlap = sorted(item_tokens.intersection(MATH_LANGUAGE_STRUCTURAL_TOKENS))
    generic_language_model_overlap = sorted(item_tokens.intersection(GENERIC_LANGUAGE_MODEL_TOKENS))
    wrong_domain_overlap = sorted(item_tokens.intersection(WRONG_DOMAIN_FOR_MATH_LANGUAGE))
    math_language_method_overlap = sorted(item_tokens.intersection(MATH_LANGUAGE_METHOD_TOKENS))
    maldi_amr_off_domain_overlap = sorted(item_tokens.intersection(MALDI_AMR_OFF_DOMAIN_TOKENS))
    surveillance_off_domain_overlap = sorted(
        item_tokens.intersection(BIOMEDICAL_SURVEILLANCE_OFF_DOMAIN_TOKENS)
    )
    clinical_stewardship_overlap = sorted(
        item_tokens.intersection(CLINICAL_STEWARDSHIP_RELEVANCE_TOKENS)
    )
    simulation_package_overlap = sorted(item_tokens.intersection(SIMULATION_PACKAGE_TOKENS))

    if math_language_query:
        generic_llm_penalty = bool(generic_language_model_overlap) and not structural_overlap and relevance < 20
        if wrong_domain_overlap and not (
            math_overlap
            and structural_overlap
            and not generic_llm_penalty
        ):
            label = "irrelevant"
            confidence = min(0.9, 0.7 + relevance / 300)
            rationale = "metadata is wrong-domain for mathematical language query"
        elif _is_math_language_method_match(item_tokens):
            label = "relevant"
            confidence = min(0.95, 0.68 + 0.04 * len(math_language_method_overlap) + relevance / 300)
            rationale = "metadata matches computational or statistical language methods"
        elif (
            math_overlap
            and language_overlap
            and (strong_math_overlap or structural_overlap)
            and (relevance >= 20 or strong_math_overlap)
            and not generic_llm_penalty
        ):
            label = "relevant"
            confidence = min(0.95, 0.62 + 0.05 * len(overlap) + relevance / 250)
            rationale = "metadata matches mathematical language query terms"
        elif language_overlap and not wrong_domain_overlap:
            label = "maybe"
            confidence = min(0.78, 0.54 + 0.04 * len(overlap) + relevance / 350)
            rationale = "metadata partially matches language query terms"
        elif math_overlap and language_overlap:
            label = "maybe"
            confidence = min(0.78, 0.56 + 0.04 * len(overlap) + relevance / 350)
            rationale = "metadata partially matches mathematical language query terms"
        else:
            label = "irrelevant"
            confidence = min(0.9, 0.7 + max(0, 25 - relevance) / 250)
            rationale = "metadata lacks mathematical language evidence"
    elif _is_maldi_amr_query_tokens(query_tokens):
        if _is_maldi_amr_biomedical_match(item_tokens):
            label = "relevant"
            confidence = min(0.95, 0.72 + 0.05 * len(overlap) + relevance / 300)
            rationale = "metadata matches biomedical antimicrobial resistance evidence"
        elif maldi_amr_off_domain_overlap or _is_general_antibiotic_prescribing_noise(item_tokens):
            label = "irrelevant"
            confidence = min(0.9, 0.7 + 0.04 * len(maldi_amr_off_domain_overlap) + relevance / 350)
            rationale = "metadata matches off-domain MALDI/AMR noise"
        elif _is_maldi_clinical_spectra_context(item_tokens):
            label = "maybe"
            confidence = min(0.78, 0.62 + 0.04 * len(overlap) + relevance / 350)
            rationale = "metadata has MALDI clinical spectra context without direct AMR evidence"
        elif not _has_antimicrobial_resistance_context(item_tokens):
            label = "irrelevant"
            confidence = min(0.88, 0.76 + relevance / 400)
            rationale = "metadata lacks biomedical antimicrobial resistance context"
        else:
            label = "maybe"
            confidence = min(0.78, 0.55 + 0.06 * len(overlap) + relevance / 350)
            rationale = "metadata partially matches biomedical antimicrobial resistance"
    elif _is_esbl_cre_query_tokens(query_tokens):
        if _is_esbl_cre_clinical_surveillance_match(item_tokens):
            label = "relevant"
            confidence = min(0.95, 0.72 + 0.05 * len(overlap) + relevance / 300)
            rationale = "metadata matches clinical resistance surveillance"
        elif surveillance_off_domain_overlap or _is_basic_beta_lactamase_without_surveillance(item_tokens):
            label = "irrelevant"
            confidence = min(0.9, 0.7 + 0.04 * len(surveillance_off_domain_overlap) + relevance / 350)
            rationale = "metadata lacks clinical ESBL/CRE surveillance context"
        elif not item_tokens.intersection(ESBL_CRE_RESISTANCE_TOKENS):
            label = "irrelevant"
            confidence = min(0.88, 0.68 + relevance / 350)
            rationale = "metadata lacks ESBL/CRE resistance evidence"
        else:
            label = "maybe"
            confidence = min(0.78, 0.55 + 0.06 * len(overlap) + relevance / 350)
            rationale = "metadata partially matches ESBL/CRE surveillance"
    elif _is_clinical_stewardship_query_tokens(query_tokens):
        if _is_clinical_stewardship_noise(item, item_tokens):
            label = "irrelevant"
            confidence = min(0.9, 0.7 + 0.03 * len(simulation_package_overlap) + relevance / 350)
            rationale = "metadata matches clinical-query noise rather than patient evidence"
        elif _is_clinical_stewardship_match(item_tokens):
            label = "relevant"
            confidence = min(0.95, 0.72 + 0.04 * len(clinical_stewardship_overlap) + relevance / 350)
            rationale = "metadata matches clinical infection, biomarker, or stewardship evidence"
        elif item_tokens.intersection({"antibiotic", "antimicrobial", "sepsis", "procalcitonin", "stewardship"}):
            label = "maybe"
            confidence = min(0.78, 0.55 + 0.06 * len(overlap) + relevance / 350)
            rationale = "metadata partially matches clinical stewardship query"
        else:
            label = "irrelevant"
            confidence = min(0.88, 0.68 + relevance / 350)
            rationale = "metadata lacks clinical stewardship evidence"
    elif len(overlap) >= 3 or relevance >= 65:
        label = "relevant"
        confidence = min(0.95, 0.62 + 0.08 * len(overlap) + relevance / 250)
        rationale = "metadata strongly matches query terms"
    elif len(overlap) >= 2 or (overlap and relevance >= 15) or relevance >= 25:
        label = "maybe"
        confidence = min(0.8, 0.55 + 0.07 * len(overlap) + relevance / 300)
        rationale = "metadata partially matches query terms"
    else:
        label = "irrelevant"
        confidence = min(0.9, 0.68 + max(0, 25 - relevance) / 200)
        rationale = "metadata has little query overlap"
    signals = (
        f"query_overlap={','.join(overlap) or '-'};"
        f"math_overlap={','.join(math_overlap) or '-'};"
        f"strong_math_overlap={','.join(strong_math_overlap) or '-'};"
        f"language_overlap={','.join(language_overlap) or '-'};"
        f"structural_overlap={','.join(structural_overlap) or '-'};"
        f"generic_language_model_overlap={','.join(generic_language_model_overlap) or '-'};"
        f"wrong_domain_overlap={','.join(wrong_domain_overlap) or '-'};"
        f"math_language_method_overlap={','.join(math_language_method_overlap) or '-'};"
        f"maldi_amr_off_domain_overlap={','.join(maldi_amr_off_domain_overlap) or '-'};"
        f"surveillance_off_domain_overlap={','.join(surveillance_off_domain_overlap) or '-'};"
        f"clinical_stewardship_overlap={','.join(clinical_stewardship_overlap) or '-'};"
        f"simulation_package_overlap={','.join(simulation_package_overlap) or '-'};"
        f"relevance_score={relevance};"
        f"provider={item.provider or 'unknown'}"
    )
    return AutoLabelDecision(
        item=item,
        label=label,
        confidence=round(confidence, 3),
        rationale=rationale,
        signals=signals,
    )


def _llm_auto_label_item(
    item: BatchItemRecord,
    *,
    query: str | None,
    model: str,
    llm_client: object | None,
) -> AutoLabelDecision:
    if llm_client is None or not hasattr(llm_client, "label"):
        raise RuntimeError("LLM auto-label provider requires a label client")
    result = llm_client.label(query=query, item=item, model=model)
    if not isinstance(result, LlmLabelResult):
        raise RuntimeError("LLM label client returned an invalid result type")
    signals = (
        "label_provider=llm;"
        f"model={model};"
        f"evidence_terms={','.join(result.evidence_terms) or '-'};"
        f"exclusion_reason={result.exclusion_reason or '-'};"
        f"provider={item.provider or 'unknown'}"
    )
    return AutoLabelDecision(
        item=item,
        label=result.label,
        confidence=result.confidence,
        rationale=result.rationale,
        signals=signals,
    )


def _label_bucket(label: ScreeningLabelRecord | None) -> int:
    if label and label.label == "relevant" and label.label_source == "human":
        return 0
    if label and label.label == "relevant":
        return 1
    if label and label.label == "maybe" and label.label_source == "human":
        return 2
    if label and label.label == "maybe":
        return 3
    return 4


def _label_profile(
    items: list[BatchItemRecord],
    labels_by_normalized: dict[str, ScreeningLabelRecord],
    label_weights: dict[str, int],
) -> Counter[str]:
    profile: Counter[str] = Counter()
    for item in items:
        label = labels_by_normalized.get(item.normalized)
        if label is None or label.label not in label_weights:
            continue
        weight = label_weights[label.label] * (4 if label.label_source == "human" else 1)
        profile.update({token: weight for token in _item_tokens(item)})
    return profile


def _item_tokens(item: BatchItemRecord) -> set[str]:
    text = " ".join(
        value
        for value in [
            item.title,
            item.abstract,
            item.journal,
            item.concepts,
            item.mesh_terms,
        ]
        if value
    )
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _query_tokens(query: str) -> set[str]:
    plan = plan_query(query)
    text = " ".join((query, *plan.expanded_queries))
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }
    expanded = set(tokens)
    if {"math", "mathematics", "mathematical"}.intersection(tokens):
        expanded.update({"math", "mathematics", "mathematical", "algebra", "formal"})
    if "language" in tokens:
        expanded.update({"language", "linguistic", "syntax", "grammar", "grammars"})
    return expanded


def _is_math_language_query_tokens(query_tokens: set[str]) -> bool:
    return bool(query_tokens.intersection(MATH_LANGUAGE_MATH_TOKENS)) and bool(
        query_tokens.intersection(MATH_LANGUAGE_LANGUAGE_TOKENS)
    )


def _is_math_language_method_match(item_tokens: set[str]) -> bool:
    if {"natural", "language", "processing"}.issubset(item_tokens) and item_tokens.intersection(
        {"computational", "question", "generation", "program", "computer", "relies", "techniques"}
    ):
        return True
    if {"computer", "program", "language"}.issubset(item_tokens):
        return True
    if "morphological" in item_tokens and item_tokens.intersection(
        {"corpus", "morpheme", "processing", "unsupervised"}
    ):
        return True
    if {"applied", "linguistics", "machine", "learning"}.issubset(item_tokens):
        return True
    if item_tokens.intersection(
        {"bayesian", "frequentist", "hypothesis", "statistical", "statistics"}
    ) and item_tokens.intersection({"linguistic", "linguistics", "psycholinguistics"}):
        return True
    return False


def _is_maldi_amr_query_tokens(query_tokens: set[str]) -> bool:
    return "maldi" in query_tokens and bool(
        query_tokens.intersection({"antibiotic", "antimicrobial", "resistance", "susceptibility"})
    )


def _has_antimicrobial_resistance_context(item_tokens: set[str]) -> bool:
    has_resistance = bool(item_tokens.intersection({"resistance", "resistant", "susceptibility"}))
    has_antimicrobial = bool(
        item_tokens.intersection({"antibiotic", "antibiotics", "antimicrobial", "antimicrobials"})
    )
    return has_resistance and has_antimicrobial


def _is_maldi_amr_biomedical_match(item_tokens: set[str]) -> bool:
    if "antimicrobial" in item_tokens and "resistance" in item_tokens:
        return True
    if "maldi" in item_tokens and _has_antimicrobial_resistance_context(item_tokens):
        return True
    if "resistance" in item_tokens and item_tokens.intersection({"proteins", "protein"}):
        return bool(item_tokens.intersection({"antimicrobial", "antibiotic", "antibiotics"}))
    return False


def _is_maldi_clinical_spectra_context(item_tokens: set[str]) -> bool:
    return bool(item_tokens.intersection({"maldi", "spectra", "spectrometry"})) and bool(
        item_tokens.intersection(
            {"bacterial", "clinical", "isolates", "laboratories", "mass", "microbiology"}
        )
    )


def _is_general_antibiotic_prescribing_noise(item_tokens: set[str]) -> bool:
    if "prescribing" not in item_tokens:
        return False
    return bool(item_tokens.intersection({"policy", "policies", "primary", "urinary"}))


def _is_esbl_cre_query_tokens(query_tokens: set[str]) -> bool:
    return bool(query_tokens.intersection({"cre", "esbl"})) and "surveillance" in query_tokens


def _is_esbl_cre_clinical_surveillance_match(item_tokens: set[str]) -> bool:
    return bool(item_tokens.intersection(ESBL_CRE_RESISTANCE_TOKENS)) and bool(
        item_tokens.intersection(ESBL_CRE_CLINICAL_TOKENS)
    )


def _is_basic_beta_lactamase_without_surveillance(item_tokens: set[str]) -> bool:
    if not {"beta", "lactamase"}.issubset(item_tokens):
        return False
    return not bool(item_tokens.intersection(ESBL_CRE_CLINICAL_TOKENS))


def _is_clinical_stewardship_query_tokens(query_tokens: set[str]) -> bool:
    return bool(query_tokens.intersection({"procalcitonin", "sepsis"})) and bool(
        query_tokens.intersection({"antibiotic", "antimicrobial", "stewardship"})
    )


def _is_clinical_stewardship_noise(item: BatchItemRecord, item_tokens: set[str]) -> bool:
    if len(item_tokens.intersection(SIMULATION_PACKAGE_TOKENS)) >= 2:
        return True
    if {"transmission", "disease"}.issubset(item_tokens) and not item_tokens.intersection(
        {"biomarker", "biomarkers", "clinical", "icu", "intensive", "procalcitonin", "sepsis", "stewardship"}
    ):
        return True
    if (
        (item.provider or "").lower() == "arxiv"
        and "sepsis" in item_tokens
        and {"deep", "learning"}.issubset(item_tokens)
        and not item_tokens.intersection(
            {
                "antibiotic",
                "antimicrobial",
                "biomarker",
                "biomarkers",
                "guideline",
                "guidelines",
                "procalcitonin",
                "stewardship",
                "treatment",
            }
        )
    ):
        return True
    return False


def _is_clinical_stewardship_match(item_tokens: set[str]) -> bool:
    if "procalcitonin" in item_tokens and item_tokens.intersection(
        {"antimicrobial", "bacterial", "infection", "infections", "sepsis"}
    ):
        return True
    if item_tokens.intersection({"sepsis", "septic"}) and item_tokens.intersection(
        {
            "antibiotic",
            "biomarker",
            "biomarkers",
            "care",
            "critical",
            "guidelines",
            "icu",
            "intensive",
            "management",
            "shock",
            "treatment",
        }
    ):
        return True
    if "stewardship" in item_tokens and item_tokens.intersection(
        {"antibiotic", "antimicrobial"}
    ) and item_tokens.intersection({"care", "clinical", "decision", "icu", "intensive", "pediatric"}):
        return True
    if "clinical" in item_tokens and item_tokens.intersection(
        {"bacterial", "infection", "infections"}
    ) and item_tokens.intersection({"gram", "negative", "neonatal", "neonates"}):
        return True
    if item_tokens.intersection({"biomarker", "biomarkers"}) and item_tokens.intersection(
        {"clinical", "critical", "infection", "microbiological", "pyelonephritis", "sepsis"}
    ):
        return True
    return False
