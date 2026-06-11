from __future__ import annotations

import re
from typing import Any

from jarvis_research.storage import BatchItemRecord, SCREENING_LABEL_CHOICES, ScreeningLabelRecord


HIGH_CONFIDENCE_MISTAKE_THRESHOLD = 0.8


def build_label_evaluation(
    items: list[BatchItemRecord],
    labels: list[ScreeningLabelRecord],
) -> dict[str, Any]:
    items_by_normalized = {item.normalized: item for item in items}
    choices = list(SCREENING_LABEL_CHOICES)
    human_counts = {choice: 0 for choice in choices}
    confusion = {human: {agent: 0 for agent in choices} for human in choices}
    comparable_rows: list[dict[str, Any]] = []

    for label in labels:
        if label.label_source != "human":
            continue
        human_counts[label.label] = human_counts.get(label.label, 0) + 1
        prior_agent = _parse_prior_agent_label(label)
        if prior_agent is None:
            continue
        item = items_by_normalized.get(label.normalized)
        agent_label, agent_confidence = prior_agent
        if agent_label not in choices:
            continue
        confusion[label.label][agent_label] += 1
        comparable_rows.append(
            {
                "source": label.source,
                "normalized": label.normalized,
                "title": item.title if item else None,
                "human_label": label.label,
                "agent_label": agent_label,
                "agent_confidence": agent_confidence,
                "relevance_score": item.relevance_score if item else None,
                "provider": item.provider if item else None,
                "note": label.note,
            }
        )

    comparable_count = len(comparable_rows)
    correct_count = sum(1 for row in comparable_rows if row["human_label"] == row["agent_label"])
    disagreements = sorted(
        [row for row in comparable_rows if row["human_label"] != row["agent_label"]],
        key=lambda row: (-(row["agent_confidence"] or -1), str(row["source"])),
    )
    high_confidence_mistakes = [
        row
        for row in disagreements
        if row["agent_confidence"] is not None
        and row["agent_confidence"] >= HIGH_CONFIDENCE_MISTAKE_THRESHOLD
    ]

    return {
        "schema_version": "1.0",
        "artifact_type": "label_evaluation",
        "human_label_counts": human_counts,
        "comparable_count": comparable_count,
        "correct_count": correct_count,
        "accuracy": correct_count / comparable_count if comparable_count else 0.0,
        "confusion_matrix": confusion,
        "precision": _precision(confusion, choices),
        "recall": _recall(confusion, choices),
        "disagreements": disagreements,
        "high_confidence_mistakes": high_confidence_mistakes,
        "recommendations": _threshold_recommendations(comparable_rows, disagreements),
    }


def _parse_prior_agent_label(label: ScreeningLabelRecord) -> tuple[str, float | None] | None:
    text = " ".join(value for value in [label.signals, label.note] if value)
    if not text:
        return None
    label_match = re.search(
        r"(?:previous_)?agent(?:_label)?\s*=\s*(relevant|maybe|irrelevant)\b",
        text,
        flags=re.IGNORECASE,
    )
    if label_match is None:
        return None
    confidence = None
    confidence_match = re.search(
        r"(?:previous_agent_confidence|agent_confidence|confidence)\s*=\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if confidence_match is not None:
        try:
            parsed = float(confidence_match.group(1))
        except ValueError:
            parsed = None
        if parsed is not None and 0 <= parsed <= 1:
            confidence = parsed
    return label_match.group(1).lower(), confidence


def _precision(confusion: dict[str, dict[str, int]], choices: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for label in choices:
        predicted = sum(confusion[human][label] for human in choices)
        values[label] = confusion[label][label] / predicted if predicted else 0.0
    return values


def _recall(confusion: dict[str, dict[str, int]], choices: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for label in choices:
        actual = sum(confusion[label][agent] for agent in choices)
        values[label] = confusion[label][label] / actual if actual else 0.0
    return values


def _threshold_recommendations(
    comparable_rows: list[dict[str, Any]],
    disagreements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not comparable_rows:
        return [
            {
                "type": "need_human_overrides",
                "confidence": None,
                "message": "Add human overrides with prior agent metadata before tuning thresholds.",
            }
        ]
    mistake_confidences = [
        row["agent_confidence"]
        for row in disagreements
        if row["agent_confidence"] is not None
    ]
    if mistake_confidences:
        threshold = round(max(mistake_confidences), 3)
        return [
            {
                "type": "review_confidence_at_or_below",
                "confidence": threshold,
                "message": f"Route agent labels with confidence <= {threshold:.2f} to review before auto-accepting.",
            }
        ]
    correct_confidences = [
        row["agent_confidence"]
        for row in comparable_rows
        if row["agent_confidence"] is not None
    ]
    if correct_confidences:
        threshold = round(min(correct_confidences), 3)
        return [
            {
                "type": "no_disagreements_observed",
                "confidence": threshold,
                "message": f"No disagreements observed; current comparable labels are correct down to {threshold:.2f} confidence.",
            }
        ]
    return [
        {
            "type": "need_confidence_metadata",
            "confidence": None,
            "message": "Human overrides exist, but prior agent confidence metadata is missing.",
        }
    ]

