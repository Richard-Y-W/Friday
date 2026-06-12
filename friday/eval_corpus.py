from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import json

from friday.discovery import Candidate
from friday.query_planning import plan_query
from friday.relevance import rank_candidates
from friday.screening import auto_label_batch_items
from friday.source_policy import evaluate_source
from friday.storage import FridayStore
from friday.topic_planning import evaluate_topic_curation, plan_topic_for_records


GOLD_CORPUS_PATH = Path(__file__).resolve().parent.parent / "eval_corpus" / "gold_cases.json"
REAL_SMOKE_CORPUS_PATH = Path(__file__).resolve().parent.parent / "eval_corpus" / "real_smoke_labels.json"
SUPPORTED_CASE_TYPES = {"query_plan", "source_policy", "ranking", "screening_label", "topic_curation"}


def load_gold_eval_cases(path: Path = GOLD_CORPUS_PATH) -> list[dict[str, Any]]:
    return _load_eval_cases(path, id_prefix="gold.", corpus_name="gold")


def load_real_smoke_eval_cases(path: Path = REAL_SMOKE_CORPUS_PATH) -> list[dict[str, Any]]:
    return _load_eval_cases(path, id_prefix="real_smoke.", corpus_name="real smoke")


def _load_eval_cases(path: Path, *, id_prefix: str, corpus_name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError(f"{corpus_name} eval corpus schema_version must be 1.0")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{corpus_name} eval corpus must contain a cases list")
    _validate_cases(cases, id_prefix=id_prefix, corpus_name=corpus_name)
    return cases


def build_gold_eval_cases(path: Path = GOLD_CORPUS_PATH):
    return _build_eval_cases(path, suite="gold", loader=load_gold_eval_cases)


def build_real_smoke_eval_cases(path: Path = REAL_SMOKE_CORPUS_PATH):
    return _build_eval_cases(path, suite="real-smoke", loader=load_real_smoke_eval_cases)


def _build_eval_cases(path: Path, *, suite: str, loader):
    from friday.eval_suite import EvalCase

    return [
        EvalCase(
            case_id=case["case_id"],
            suite=suite,
            category=case["type"],
            description=case["description"],
            run=_case_runner(case),
        )
        for case in loader(path)
    ]


def _validate_cases(cases: list[dict[str, Any]], *, id_prefix: str, corpus_name: str) -> None:
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"{corpus_name} eval case at index {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.startswith(id_prefix):
            raise ValueError(f"{corpus_name} eval case at index {index} must have a {id_prefix}* case_id")
        if case_id in seen:
            raise ValueError(f"duplicate {corpus_name} eval case_id: {case_id}")
        seen.add(case_id)
        case_type = case.get("type")
        if case_type not in SUPPORTED_CASE_TYPES:
            raise ValueError(f"unsupported {corpus_name} eval case type for {case_id}: {case_type}")
        description = case.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{corpus_name} eval case {case_id} must have a description")
        if not isinstance(case.get("expected"), dict):
            raise ValueError(f"{corpus_name} eval case {case_id} must have expected data")


def _case_runner(case: dict[str, Any]):
    case_type = case["type"]
    if case_type == "query_plan":
        return lambda: _run_query_plan_case(case)
    if case_type == "source_policy":
        return lambda: _run_source_policy_case(case)
    if case_type == "ranking":
        return lambda: _run_ranking_case(case)
    if case_type == "screening_label":
        return lambda: _run_screening_label_case(case)
    if case_type == "topic_curation":
        return lambda: _run_topic_curation_case(case)
    return lambda: (False, f"unsupported gold case type: {case_type}")


def _run_query_plan_case(case: dict[str, Any]) -> tuple[bool, str]:
    plan = plan_query(case["query"])
    expected = case["expected"]
    rejected_meanings = {
        meaning
        for acronym in plan.resolved_acronyms
        for meaning in acronym.rejected_meanings
    }
    unresolved_acronyms = {
        acronym.acronym
        for acronym in plan.resolved_acronyms
        if acronym.reason == "unresolved_acronym"
    }
    failures: list[str] = []
    if expected.get("intent") and plan.intent != expected["intent"]:
        failures.append(f"intent={plan.intent!r}")
    for value in expected.get("expanded_contains", []):
        if value not in plan.expanded_queries:
            failures.append(f"missing_expansion={value!r}")
    for value in expected.get("rejected_meanings_contains", []):
        if value not in rejected_meanings:
            failures.append(f"missing_rejected_meaning={value!r}")
    for value in expected.get("unresolved_acronyms_contains", []):
        if value not in unresolved_acronyms:
            failures.append(f"missing_unresolved_acronym={value!r}")
    if failures:
        return False, "; ".join(failures)
    return True, f"query plan matched intent {plan.intent}"


def _run_source_policy_case(case: dict[str, Any]) -> tuple[bool, str]:
    decision = evaluate_source(case["source"])
    expected = case["expected"]
    checks = {
        "allowed": decision.allowed,
        "reason": decision.reason,
        "domain": decision.domain,
        "normalized": decision.normalized,
    }
    failures = [
        f"{key}={checks.get(key)!r}"
        for key, expected_value in expected.items()
        if checks.get(key) != expected_value
    ]
    if failures:
        return False, "; ".join(failures)
    return True, f"source policy returned {decision.reason}"


def _run_ranking_case(case: dict[str, Any]) -> tuple[bool, str]:
    ranked = rank_candidates(
        case["query"],
        [_candidate_from_mapping(candidate) for candidate in case.get("candidates", [])],
    )
    expected_top = case["expected"].get("top_source")
    actual_top = ranked[0].source_for_gate if ranked else None
    if actual_top != expected_top:
        ordered = [candidate.source_for_gate for candidate in ranked]
        return False, f"expected top_source={expected_top!r}; ranked={ordered!r}"
    return True, f"ranked {actual_top} first"


def _run_screening_label_case(case: dict[str, Any]) -> tuple[bool, str]:
    with TemporaryDirectory() as tmp:
        store = FridayStore(Path(tmp) / "friday.db")
        batch = store.create_batch(query=case["query"], limit=1, mode="gold_eval")
        candidate = _candidate_from_mapping(case["candidate"])
        store.add_batch_item(
            batch.batch_id,
            candidate.source_for_gate,
            evaluate_source(candidate.source_for_gate),
            candidate,
        )
        result = auto_label_batch_items(store, batch.batch_id, query=case["query"])
    decision = result.decisions[0] if result.decisions else None
    expected = case["expected"]
    if decision is None:
        return False, "no screening decision returned"
    min_confidence = expected.get("min_confidence", 0.0)
    if decision.label != expected.get("label"):
        return False, f"expected label={expected.get('label')!r}; got {decision.label!r}"
    if decision.confidence < min_confidence:
        return False, f"expected confidence>={min_confidence}; got {decision.confidence}"
    return True, f"labeled as {decision.label} with confidence {decision.confidence}"


def _run_topic_curation_case(case: dict[str, Any]) -> tuple[bool, str]:
    candidate = _candidate_from_mapping(case["candidate"])
    profile_records = [_candidate_from_mapping(record) for record in case.get("profile_records", [])]
    profile = plan_topic_for_records(case["query"], profile_records or [candidate])
    decision = evaluate_topic_curation(candidate, profile)
    expected = case["expected"]

    failures: list[str] = []
    if "eligible_for_deep_read" in expected and decision.eligible_for_deep_read != expected["eligible_for_deep_read"]:
        failures.append(f"eligible_for_deep_read={decision.eligible_for_deep_read!r}")
    if expected.get("status") and decision.status != expected["status"]:
        failures.append(f"status={decision.status!r}")
    if expected.get("reason") and decision.reason != expected["reason"]:
        failures.append(f"reason={decision.reason!r}")
    for component_id in expected.get("missing_topic_components_contains", []):
        if component_id not in decision.missing_topic_components:
            failures.append(f"missing_component_absent={component_id!r}")
    for component_id in expected.get("matched_topic_components_contains", []):
        if component_id not in decision.matched_topic_components:
            failures.append(f"matched_component_absent={component_id!r}")
    for topic_id in expected.get("profile_topic_ids_contains", []):
        if topic_id not in profile.topic_ids:
            failures.append(f"profile_topic_absent={topic_id!r}")

    if failures:
        return False, "; ".join(failures)
    return True, f"topic curation {decision.status}: {decision.reason}"


def _candidate_from_mapping(data: dict[str, Any]) -> Candidate:
    return Candidate(
        provider=data["provider"],
        title=data["title"],
        source_for_gate=data["source_for_gate"],
        url=data.get("url"),
        pdf_url=data.get("pdf_url"),
        doi=data.get("doi"),
        pmid=data.get("pmid"),
        pmcid=data.get("pmcid"),
        arxiv_id=data.get("arxiv_id"),
        year=data.get("year"),
        abstract=data.get("abstract"),
        relevance_score=data.get("relevance_score"),
        relevance_reason=data.get("relevance_reason"),
        query_variant=data.get("query_variant"),
        query_intent=data.get("query_intent"),
        acronym_expansions=data.get("acronym_expansions"),
        journal=data.get("journal"),
        concepts=data.get("concepts"),
        mesh_terms=data.get("mesh_terms"),
        oa_status=data.get("oa_status"),
        open_access_pdf_url=data.get("open_access_pdf_url"),
    )
