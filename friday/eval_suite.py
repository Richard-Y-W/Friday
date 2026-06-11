from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence

from friday.claim_audit import build_claim_support_audit
from friday.discovery import Candidate
from friday.evidence import EvidenceItem
from friday.pdf_ingestion import resolve_candidate_pdf_url
from friday.query_planning import plan_query
from friday.relevance import rank_candidates
from friday.screening import auto_label_batch_items
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


EvalCaseRunner = Callable[[], tuple[bool, str]]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    suite: str
    category: str
    description: str
    run: EvalCaseRunner


def available_eval_suites() -> tuple[str, ...]:
    return ("core", "biomedical", "natural-language", "safety", "gold", "real-smoke")


def run_eval_suite(
    suite: str = "core",
    *,
    cases: Sequence[EvalCase] | None = None,
) -> dict[str, Any]:
    if suite not in available_eval_suites():
        raise ValueError(f"Unknown eval suite: {suite}")

    selected_cases = _select_cases(suite, list(cases) if cases is not None else _default_cases())
    case_results = [_run_case(case) for case in selected_cases]
    passed = sum(1 for result in case_results if result["status"] == "pass")
    total = len(case_results)
    failed = total - passed
    pass_rate = round(passed / total, 3) if total else 0.0

    return {
        "schema_version": "1.0",
        "artifact_type": "eval_suite_report",
        "suite": suite,
        "status": "pass" if failed == 0 else "fail",
        "counts": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
        },
        "cases": case_results,
    }


def render_eval_report_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    pass_rate = int(round(counts["pass_rate"] * 100))
    lines = [
        "Friday Eval Suite",
        f"Suite: {report['suite']}",
        f"Status: {report['status']}",
        (
            f"Cases: {counts['passed']}/{counts['total']} passed "
            f"({pass_rate}%)"
        ),
        "",
        "Cases",
    ]
    for case in report["cases"]:
        lines.append(
            f"- [{case['status']}] {case['case_id']} "
            f"({case['category']}): {case['message']}"
        )
    return "\n".join(lines)


def _select_cases(suite: str, cases: list[EvalCase]) -> list[EvalCase]:
    if suite == "core":
        return cases
    return [case for case in cases if case.suite == suite]


def _run_case(case: EvalCase) -> dict[str, Any]:
    try:
        passed, message = case.run()
    except Exception as exc:
        passed = False
        message = f"error:{type(exc).__name__}: {exc}"
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "category": case.category,
        "description": case.description,
        "status": "pass" if passed else "fail",
        "message": message,
    }


def _default_cases() -> list[EvalCase]:
    from friday.eval_corpus import build_gold_eval_cases, build_real_smoke_eval_cases

    built_in_cases = [
        EvalCase(
            case_id="biomedical.maldi_amr_query_plan",
            suite="biomedical",
            category="query_planning",
            description="MALDI AMR resolves to antimicrobial resistance and rejects other AMR meanings.",
            run=_case_maldi_amr_query_plan,
        ),
        EvalCase(
            case_id="natural_language.math_language_query_plan",
            suite="natural-language",
            category="query_planning",
            description="Conversational math-language prompts become scholarly query variants.",
            run=_case_math_language_query_plan,
        ),
        EvalCase(
            case_id="safety.github_pdf_blocked",
            suite="safety",
            category="source_gate",
            description="GitHub-hosted PDFs are blocked even when the path looks scholarly.",
            run=_case_github_pdf_blocked,
        ),
        EvalCase(
            case_id="biomedical.ranking_prefers_biomedical_amr",
            suite="biomedical",
            category="ranking",
            description="Biomedical MALDI AMR metadata ranks above NLP AMR collisions.",
            run=_case_biomedical_ranking,
        ),
        EvalCase(
            case_id="biomedical.heuristic_label_relevant_maldi_amr",
            suite="biomedical",
            category="screening_label",
            description="Heuristic auto-labeling marks clear MALDI antimicrobial resistance papers relevant.",
            run=_case_biomedical_auto_label,
        ),
        EvalCase(
            case_id="natural_language.heuristic_label_math_language",
            suite="natural-language",
            category="screening_label",
            description="Heuristic auto-labeling handles math-language papers from a natural query.",
            run=_case_math_language_auto_label,
        ),
        EvalCase(
            case_id="biomedical.safe_pmc_pdf_resolution",
            suite="biomedical",
            category="pdf_resolution",
            description="PubMed Central candidates resolve to safe PMC PDF URLs without downloader execution.",
            run=_case_safe_pmc_pdf_resolution,
        ),
        EvalCase(
            case_id="safety.claim_audit_requires_page_evidence",
            suite="safety",
            category="claim_support",
            description="Claim support auditing reports a material gap when no page-anchored evidence exists.",
            run=_case_claim_audit_requires_evidence,
        ),
        EvalCase(
            case_id="safety.claim_audit_accepts_page_evidence",
            suite="safety",
            category="claim_support",
            description="Claim support auditing passes when evidence is page anchored and cited.",
            run=_case_claim_audit_accepts_page_evidence,
        ),
    ]
    return [*built_in_cases, *build_gold_eval_cases(), *build_real_smoke_eval_cases()]


def _case_maldi_amr_query_plan() -> tuple[bool, str]:
    plan = plan_query("MALDI AMR")
    rejected = plan.resolved_acronyms[0].rejected_meanings if plan.resolved_acronyms else ()
    checks = [
        plan.intent == "biomedical",
        "MALDI antimicrobial resistance" in plan.expanded_queries,
        "MALDI-TOF antimicrobial susceptibility" in plan.expanded_queries,
        "abstract meaning representation" in rejected,
        "adaptive mesh refinement" in rejected,
    ]
    return _case_result(
        all(checks),
        "resolved AMR as antimicrobial resistance with MALDI-safe expansions",
        f"unexpected plan: {plan}",
    )


def _case_math_language_query_plan() -> tuple[bool, str]:
    plan = plan_query("friday tell me about how language is math")
    checks = [
        plan.intent == "mathematical_linguistics",
        "mathematical linguistics" in plan.expanded_queries,
        "formal language theory natural language" in plan.expanded_queries,
    ]
    return _case_result(
        all(checks),
        "rewrote conversational prompt to mathematical linguistics queries",
        f"unexpected plan: {plan}",
    )


def _case_github_pdf_blocked() -> tuple[bool, str]:
    decision = evaluate_source("https://github.com/example/repo/blob/main/paper.pdf")
    passed = not decision.allowed and decision.reason == "blocked_domain" and decision.domain == "github.com"
    return _case_result(
        passed,
        "blocked github.com before any ingestion",
        f"unexpected source decision: {decision}",
    )


def _case_biomedical_ranking() -> tuple[bool, str]:
    biomedical = Candidate(
        provider="pubmed",
        title="MALDI-TOF antimicrobial resistance prediction in clinical isolates",
        source_for_gate="10.1000/maldi-amr",
        doi="10.1000/maldi-amr",
        abstract="MALDI spectra were used for antimicrobial susceptibility testing.",
        journal="Clinical Microbiology and Infection",
        mesh_terms="Drug Resistance, Microbial; Mass Spectrometry",
        concepts="antimicrobial resistance; microbiology",
        year=2025,
    )
    nlp = Candidate(
        provider="arxiv",
        title="AMR parsing with semantic graph generation",
        source_for_gate="https://arxiv.org/pdf/2401.00001",
        arxiv_id="2401.00001",
        abstract="Abstract meaning representation parsing for natural language text generation.",
        year=2025,
    )
    ranked = rank_candidates("MALDI AMR", [nlp, biomedical])
    passed = ranked[0].title == biomedical.title and (ranked[0].relevance_score or 0) > (ranked[1].relevance_score or 0)
    return _case_result(
        passed,
        f"ranked biomedical candidate first at score {ranked[0].relevance_score}",
        f"unexpected ranking: {[item.title for item in ranked]}",
    )


def _case_biomedical_auto_label() -> tuple[bool, str]:
    with TemporaryDirectory() as tmp:
        store = FridayStore(Path(tmp) / "friday.db")
        batch = store.create_batch(query="MALDI AMR", limit=1, mode="eval")
        candidate = Candidate(
            provider="pubmed",
            title="MALDI-TOF antimicrobial resistance prediction",
            source_for_gate="10.1000/eval-maldi-amr",
            doi="10.1000/eval-maldi-amr",
            abstract="Antimicrobial resistance and antibiotic susceptibility from MALDI spectra.",
            relevance_score=82,
            mesh_terms="Drug Resistance, Microbial; Mass Spectrometry",
        )
        store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
        result = auto_label_batch_items(store, batch.batch_id, query="MALDI AMR")
    decision = result.decisions[0] if result.decisions else None
    passed = decision is not None and decision.label == "relevant" and decision.confidence >= 0.75
    return _case_result(
        passed,
        f"labeled clear MALDI AMR paper as {decision.label if decision else '-'}",
        f"unexpected label result: {result}",
    )


def _case_math_language_auto_label() -> tuple[bool, str]:
    with TemporaryDirectory() as tmp:
        store = FridayStore(Path(tmp) / "friday.db")
        batch = store.create_batch(query="what is the importance of math in language", limit=1, mode="eval")
        candidate = Candidate(
            provider="openalex",
            title="Formal grammar and mathematical models of language acquisition",
            source_for_gate="10.1000/eval-language-math",
            doi="10.1000/eval-language-math",
            abstract="Mathematical linguistics links algebra, formal grammar, syntax, and language learning.",
            relevance_score=70,
            concepts="mathematical linguistics; formal language theory",
        )
        store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
        result = auto_label_batch_items(
            store,
            batch.batch_id,
            query="what is the importance of math in language",
        )
    decision = result.decisions[0] if result.decisions else None
    passed = decision is not None and decision.label == "relevant" and decision.confidence >= 0.75
    return _case_result(
        passed,
        f"labeled math-language paper as {decision.label if decision else '-'}",
        f"unexpected label result: {result}",
    )


def _case_safe_pmc_pdf_resolution() -> tuple[bool, str]:
    candidate = Candidate(
        provider="pubmed",
        title="Open PubMed Central antimicrobial resistance paper",
        source_for_gate="10.1000/eval-pmc",
        doi="10.1000/eval-pmc",
        pmcid="PMC1234567",
    )
    resolution = resolve_candidate_pdf_url(candidate)
    passed = (
        resolution.pdf_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/"
        and resolution.reason == "resolved_pmc_pdf"
    )
    return _case_result(
        passed,
        "resolved PMCID to allowlisted PMC PDF URL",
        f"unexpected PDF resolution: {resolution}",
    )


def _case_claim_audit_requires_evidence() -> tuple[bool, str]:
    with TemporaryDirectory() as tmp:
        store = FridayStore(Path(tmp) / "friday.db")
        batch = store.create_batch(query="MALDI AMR", limit=1, mode="eval")
        audit = build_claim_support_audit(store, batch.batch_id)
    passed = audit["status"] == "gaps" and audit["counts"]["material_gaps"] == 1
    return _case_result(
        passed,
        "reported material gap when no extracted evidence exists",
        f"unexpected audit: {audit}",
    )


def _case_claim_audit_accepts_page_evidence() -> tuple[bool, str]:
    with TemporaryDirectory() as tmp:
        store = FridayStore(Path(tmp) / "friday.db")
        batch = store.create_batch(query="MALDI AMR", limit=1, mode="eval")
        candidate = Candidate(
            provider="pubmed",
            title="MALDI evidence paper",
            source_for_gate="10.1000/eval-evidence",
            doi="10.1000/eval-evidence",
        )
        store.add_batch_item(batch.batch_id, candidate.source_for_gate, evaluate_source(candidate.source_for_gate), candidate)
        artifact = store.add_pdf_artifact(
            batch.batch_id,
            source=candidate.source_for_gate,
            pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
            final_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
            sha256="e" * 64,
            byte_count=1200,
            content_type="application/pdf",
            local_path="artifacts/eval.pdf",
            status="stored",
            reason="pdf_text_extracted",
        )
        store.add_evidence_records(
            artifact.artifact_id,
            [
                EvidenceItem(
                    evidence_type="result",
                    text="The MALDI-TOF assay identified resistant isolates.",
                    page_number=4,
                )
            ],
        )
        audit = build_claim_support_audit(store, batch.batch_id)
    passed = audit["status"] == "pass" and audit["counts"]["supported"] == 1
    return _case_result(
        passed,
        "accepted one page-anchored supported evidence claim",
        f"unexpected audit: {audit}",
    )


def _case_result(passed: bool, pass_message: str, fail_message: str) -> tuple[bool, str]:
    return passed, pass_message if passed else fail_message
