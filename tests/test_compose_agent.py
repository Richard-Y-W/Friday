import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.compose_agent import (
    ComposePackageError,
    build_discourse_plan,
    build_compose_package_files,
    build_llm_compose_package_files,
    load_writing_package,
)
from friday.llm.types import LLMResponse


class ComposeAgentTests(unittest.TestCase):
    def test_missing_required_package_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "source_report.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(ComposePackageError) as raised:
                build_compose_package_files(package_dir, section="results")

            self.assertIn("supported_paragraphs.json", str(raised.exception))

    def test_results_compose_uses_only_supported_result_paragraphs(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="results")

            self.assertEqual(
                sorted(files),
                [
                    "claim_audit.json",
                    "conflicts.json",
                    "draft.md",
                    "outline.json",
                    "refused_claims.json",
                    "used_evidence.json",
                ],
            )
            draft = files["draft.md"]
            self.assertIn("# Evidence-Bound Results Draft", draft)
            self.assertIn("Across 2 papers, result evidence includes AUROC 0.91; sensitivity 88 percent [P1 p2; P2 p2].", draft)
            self.assertNotIn("method evidence includes", draft)
            self.assertNotIn("unsupported generated result", draft)

            audit = json.loads(files["claim_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["supported_paragraph_count"], 1)
            self.assertEqual(audit["paragraphs"][0]["citations"], ["P1 p2", "P2 p2"])
            used = json.loads(files["used_evidence.json"])
            self.assertEqual(used["used_evidence"][0]["evidence_type"], "result")
            refused = json.loads(files["refused_claims.json"])
            self.assertEqual(refused["refused_claims"][0]["reason"], "unknown_page_citation")
            conflicts = json.loads(files["conflicts.json"])
            self.assertEqual(conflicts["conflict_count"], 0)

    def test_no_section_evidence_emits_material_gap_without_used_evidence(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="limitations")

            self.assertIn("MATERIAL GAP: No supported limitation evidence is available in this writing package.", files["draft.md"])
            audit = json.loads(files["claim_audit.json"])
            self.assertEqual(audit["status"], "material_gap")
            self.assertEqual(audit["supported_paragraph_count"], 0)
            used = json.loads(files["used_evidence.json"])
            self.assertEqual(used["used_evidence"], [])

    def test_results_compose_groups_outline_and_exports_conflicts(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_grouped_fixture_package(package_dir)

            files = build_compose_package_files(package_dir, section="results")

            outline = json.loads(files["outline.json"])
            self.assertEqual(outline["groups"][0]["group_label"], "Resistance detection")
            self.assertEqual(outline["groups"][0]["paragraph_count"], 2)
            self.assertEqual(outline["groups"][0]["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(outline["groups"][1]["group_label"], "Model performance")
            self.assertEqual(outline["items"][0]["group_label"], "Resistance detection")

            draft = files["draft.md"]
            self.assertLess(draft.index("## Resistance detection"), draft.index("## Model performance"))
            self.assertIn("MALDI-TOF improved resistant-isolate detection [P1 p2].", draft)
            self.assertIn("MALDI-TOF showed no improvement for resistant-isolate detection [P2 p2].", draft)

            conflicts = json.loads(files["conflicts.json"])
            self.assertEqual(conflicts["artifact_type"], "compose_conflicts")
            self.assertEqual(conflicts["conflict_count"], 1)
            self.assertEqual(conflicts["conflicts"][0]["group_label"], "Resistance detection")
            self.assertEqual(conflicts["conflicts"][0]["stance_set"], ["negative", "positive"])
            self.assertEqual(conflicts["conflicts"][0]["citations"], ["P1 p2", "P2 p2"])

    def test_compose_attaches_evidence_table_rows_when_available(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "schema_version": "1.0",
                    "artifact_type": "writing_evidence_tables",
                    "tables": {
                        "results": [
                            {
                                "row_id": "E10",
                                "evidence_type": "result",
                                "paper": "P1",
                                "citation": "P1 p2",
                                "page_number": 2,
                                "text": "The model achieved an AUROC of 0.91.",
                            },
                            {
                                "row_id": "E11",
                                "evidence_type": "result",
                                "paper": "P2",
                                "citation": "P2 p2",
                                "page_number": 2,
                                "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            },
                        ]
                    },
                    "counts": {"results": 2},
                },
            )

            files = build_compose_package_files(package_dir, section="results")

            used = json.loads(files["used_evidence.json"])
            evidence = used["used_evidence"][0]
            self.assertEqual([row["row_id"] for row in evidence["table_rows"]], ["E10", "E11"])
            outline = json.loads(files["outline.json"])
            self.assertEqual(outline["items"][0]["table_row_ids"], ["E10", "E11"])

    def test_discourse_plan_uses_only_supported_evidence_and_required_citations(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            package = load_writing_package(package_dir)

            plan = build_discourse_plan(package, section="results")

            self.assertEqual(plan["artifact_type"], "discourse_plan")
            self.assertEqual(plan["section"], "results")
            self.assertEqual(plan["required_citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(plan["moves"][0]["kind"], "source_context")
            self.assertEqual(plan["moves"][1]["kind"], "evidence_synthesis")
            self.assertNotIn("unsupported generated result", json.dumps(plan))

    def test_llm_compose_accepts_evidence_bound_draft(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text=(
                            "# Results\n\n"
                            "The supported result evidence reports AUROC 0.91 and 88 percent sensitivity [P1 p2; P2 p2]."
                        ),
                    ),
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "Draft uses only supported result evidence.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("The supported result evidence reports", files["draft.md"])
            self.assertEqual(files["verified_draft.md"], files["draft.md"])
            self.assertIn("llm_draft.md", files)
            self.assertIn("discourse_plan.json", files)
            self.assertIn("composer_prompt.json", files)
            self.assertIn("verifier_prompt.json", files)
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "pass")
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "pass")
            self.assertEqual(router.calls[0][0], "composer")
            self.assertEqual(router.calls[1][0], "verifier")
            self.assertIn("AUROC 0.91", router.calls[0][1].prompt)
            self.assertIn("The supported result evidence reports", router.calls[1][1].prompt)
            self.assertNotIn("unsupported generated result", router.calls[0][1].prompt)
            composer_payload = json.loads(router.calls[0][1].prompt)
            self.assertIn("atomic_evidence_rows", composer_payload)
            self.assertNotIn("supported_evidence", composer_payload)
            self.assertEqual(composer_payload["atomic_evidence_rows"][0]["citation"], "P1 p2")

    def test_llm_compose_inserts_source_context_line_before_verifier(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text="# Results\n\nThe supported result evidence reports AUROC 0.91 [P1 p2].",
                    ),
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "Source context is present and the claim is cited.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                                "missing_material_gaps": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`", files["draft.md"])
            self.assertIn("Source: Batch `batch_test`", router.calls[1][1].prompt)
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "pass")

    def test_llm_compose_repairs_uncited_factual_sentences_before_verifier(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The supported result evidence reports AUROC 0.91 [P1 p2]. "
                                "This extra factual sentence has no citation."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text="# Results\n\nThe supported result evidence reports AUROC 0.91 [P1 p2].",
                        ),
                    ],
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "Revision removed the uncited sentence.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("The supported result evidence reports", files["draft.md"])
            self.assertNotIn("This extra factual sentence", files["draft.md"])
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["reason"], "uncited_factual_sentence")
            self.assertIn("This extra factual sentence has no citation.", audit["local_policy_issues"][0]["sentence"])
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "pass")
            revision_audit = json.loads(files["revision_audit.json"])
            self.assertEqual(revision_audit["status"], "pass")
            self.assertEqual(revision_audit["reason"], "revision_verified")
            self.assertEqual([role for role, _request in router.calls], ["composer", "composer", "verifier"])

    def test_llm_compose_does_not_split_cited_species_initials(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text="# Results\n\nRelebactam improved activity against *P. aeruginosa* [P1 p2].",
                    ),
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "The cited species sentence is treated as one sentence.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("*P. aeruginosa* [P1 p2].", files["draft.md"])
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual([role for role, _request in router.calls], ["composer", "verifier"])

    def test_llm_compose_repairs_material_gap_expansion_with_exact_gap_messages(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The supported result evidence reports AUROC 0.91 [P1 p2].\n\n"
                                "## Material Gaps\n\n"
                                "> No page-anchored limitation evidence is available in this batch."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The supported result evidence reports AUROC 0.91 [P1 p2].\n\n"
                                "## Material Gaps\n\n"
                                "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch."
                            ),
                        ),
                    ],
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "The material gap is copied exactly.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                                "missing_material_gaps": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.", files["draft.md"])
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["reason"], "uncited_factual_sentence")
            revision_payload = json.loads(json.loads(files["revision_prompt.json"])["prompt"])
            self.assertEqual(
                revision_payload["repair_context"]["material_gaps"],
                ["No page-anchored limitation evidence is available in this batch."],
            )
            revision_audit = json.loads(files["revision_audit.json"])
            self.assertEqual(revision_audit["status"], "pass")
            self.assertEqual([role for role, _request in router.calls], ["composer", "composer", "verifier"])

    def test_llm_compose_accepts_revised_draft_after_verifier_feedback(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The draft overstates the evidence as clinically validated [P1 p2; P2 p2]."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The supported result evidence reports AUROC 0.91 and 88 percent sensitivity [P1 p2; P2 p2]."
                            ),
                        ),
                    ],
                    "verifier": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "Clinically validated is unsupported.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "pass",
                                    "summary": "Revision removed unsupported wording.",
                                    "unsupported_claims": [],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                    ],
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("The supported result evidence reports", files["draft.md"])
            self.assertEqual(files["verified_draft.md"], files["draft.md"])
            self.assertIn("clinically validated", files["llm_draft.md"])
            self.assertNotIn("clinically validated", files["draft.md"])
            self.assertIn("revised_llm_draft.md", files)
            self.assertIn("revision_prompt.json", files)
            self.assertIn("initial_verifier_audit.json", files)
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "pass")
            revision_audit = json.loads(files["revision_audit.json"])
            self.assertEqual(revision_audit["status"], "pass")
            self.assertEqual(revision_audit["reason"], "revision_verified")
            self.assertEqual([role for role, _request in router.calls], ["composer", "verifier", "composer", "verifier"])
            self.assertIn("Clinically validated is unsupported", router.calls[2][1].prompt)
            revision_prompt = json.loads(files["revision_prompt.json"])["prompt"]
            revision_payload = json.loads(revision_prompt)
            self.assertIn("repair_context", revision_payload)
            self.assertNotIn("supported_evidence", revision_payload)
            self.assertIn("Do not expand acronyms unless the exact expansion appears in a matching evidence row.", revision_payload["output_rules"])
            self.assertEqual(revision_payload["repair_context"]["failed_claims"], ["clinically validated"])
            self.assertLessEqual(len(revision_payload["repair_context"]["evidence_rows"]), 2)

    def test_llm_compose_allows_second_revision_after_local_repair_reaches_verifier(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The supported result evidence reports AUROC 0.91 [P1 p2]. "
                                "This uncited bridge sentence should be removed."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text="# Results\n\nThe result was clinically validated [P1 p2].",
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text="# Results\n\nThe supported result evidence reports AUROC 0.91 [P1 p2].",
                        ),
                    ],
                    "verifier": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "Clinically validated is unsupported.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                    "missing_material_gaps": [],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "pass",
                                    "summary": "Second revision removed unsupported wording.",
                                    "unsupported_claims": [],
                                    "citation_errors": [],
                                    "missing_material_gaps": [],
                                }
                            ),
                        ),
                    ],
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("AUROC 0.91", files["draft.md"])
            self.assertNotIn("clinically validated", files["draft.md"])
            self.assertIn("revision_2_prompt.json", files)
            self.assertIn("revised_llm_draft_2.md", files)
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "pass")
            revision_audit = json.loads(files["revision_audit.json"])
            self.assertEqual(revision_audit["status"], "pass")
            self.assertEqual(revision_audit["attempt"], 2)
            self.assertEqual([role for role, _request in router.calls], ["composer", "composer", "verifier", "composer", "verifier"])

    def test_llm_compose_falls_back_when_draft_uses_unknown_citation(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text="This draft invents an unsupported citation [P9 p9].",
                    )
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("# Evidence-Bound Results Draft", files["draft.md"])
            self.assertNotIn("P9 p9", files["draft.md"])
            audit = json.loads(files["composer_audit.json"])
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["reason"], "unknown_citation")
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "skipped")
            self.assertEqual(verifier_audit["reason"], "composer_not_trusted")
            self.assertEqual([role for role, _request in router.calls], ["composer"])

    def test_llm_compose_falls_back_when_revision_still_fails_verifier(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The draft overstates the evidence as clinically validated [P1 p2; P2 p2]."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The revision still says clinically validated [P1 p2; P2 p2]."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The second revision still says clinically validated [P1 p2; P2 p2]."
                            ),
                        ),
                        LLMResponse(
                            provider="claude_cli",
                            model="sonnet",
                            success=True,
                            text=(
                                "# Results\n\n"
                                "The third revision still says clinically validated [P1 p2; P2 p2]."
                            ),
                        ),
                    ],
                    "verifier": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "The phrase clinically validated is unsupported.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "The revision still contains clinically validated.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "The second revision still contains clinically validated.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "The third revision still contains clinically validated.",
                                    "unsupported_claims": ["clinically validated"],
                                    "citation_errors": [],
                                }
                            ),
                        ),
                    ],
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("# Evidence-Bound Results Draft", files["draft.md"])
            self.assertNotIn("clinically validated", files["draft.md"])
            self.assertIn("clinically validated", files["llm_draft.md"])
            self.assertIn("clinically validated", files["revised_llm_draft.md"])
            self.assertIn("clinically validated", files["revised_llm_draft_2.md"])
            self.assertIn("clinically validated", files["revised_llm_draft_3.md"])
            verifier_audit = json.loads(files["verifier_audit.json"])
            self.assertEqual(verifier_audit["status"], "fallback")
            self.assertEqual(verifier_audit["reason"], "verifier_rejected")
            self.assertEqual(verifier_audit["verdict"], "fail")
            revision_audit = json.loads(files["revision_audit.json"])
            self.assertEqual(revision_audit["status"], "fallback")
            self.assertEqual(revision_audit["reason"], "revision_rejected")
            self.assertEqual(revision_audit["attempt"], 3)


class FakeRouter:
    def __init__(self, responses):
        self.responses = {
            role: list(response) if isinstance(response, list) else [response]
            for role, response in responses.items()
        }
        self.calls = []

    def generate(self, role, request):
        self.calls.append((role, request))
        return self.responses[role].pop(0)


def _write_fixture_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "source_report.json",
        {
            "batch_id": "batch_test",
            "query": "MALDI AMR",
            "screened_count": 1000,
            "blocked_count": 25,
            "deep_read_count": 50,
        },
    )
    _write_json(
        package_dir / "paper_references.json",
        [
            {
                "label": "P1",
                "title": "MALDI antimicrobial resistance prediction",
                "year": 2024,
                "journal": "Nature Medicine",
                "doi": "10.1038/example-a",
                "evidence_count": 2,
            },
            {
                "label": "P2",
                "title": "MALDI-TOF antimicrobial susceptibility testing",
                "year": 2023,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-b",
                "evidence_count": 2,
            },
        ],
    )
    _write_json(
        package_dir / "supported_paragraphs.json",
        [
            {
                "paragraph_id": "S1.1",
                "block_id": "S1",
                "section": "Method",
                "evidence_type": "method",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "Across 2 papers, method evidence includes spectra classifiers [P1 p1; P2 p1].",
                "citations": ["P1 p1", "P2 p1"],
                "evidence_count": 2,
            },
            {
                "paragraph_id": "S2.1",
                "block_id": "S2",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "Across 2 papers, result evidence includes AUROC 0.91; sensitivity 88 percent [P1 p2; P2 p2].",
                "citations": ["P1 p2", "P2 p2"],
                "evidence_count": 2,
            },
        ],
    )
    _write_json(
        package_dir / "blocked_paragraphs.json",
        [
            {
                "paragraph_id": "S99.1",
                "block_id": "S99",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "MATERIAL_GAP",
                "reason": "unknown_page_citation",
                "paragraph": "unsupported generated result [P9 p9]",
                "citations": ["P9 p9"],
                "evidence_count": 0,
            }
        ],
    )
    _write_json(
        package_dir / "material_gaps.json",
        [
            {
                "reason": "evidence_gap",
                "message": "No page-anchored limitation evidence is available in this batch.",
            }
        ],
    )


def _write_grouped_fixture_package(package_dir: Path) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        package_dir / "source_report.json",
        {
            "batch_id": "batch_grouped",
            "query": "MALDI AMR",
            "screened_count": 1000,
            "blocked_count": 25,
            "deep_read_count": 50,
        },
    )
    _write_json(
        package_dir / "paper_references.json",
        [
            {
                "label": "P1",
                "title": "Positive MALDI detection result",
                "year": 2024,
                "journal": "Nature Medicine",
                "doi": "10.1038/example-positive",
                "evidence_count": 1,
            },
            {
                "label": "P2",
                "title": "Negative MALDI detection result",
                "year": 2024,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-negative",
                "evidence_count": 1,
            },
            {
                "label": "P3",
                "title": "MALDI classifier performance",
                "year": 2023,
                "journal": "Clinical Microbiology",
                "doi": "10.1038/example-performance",
                "evidence_count": 1,
            },
        ],
    )
    _write_json(
        package_dir / "supported_paragraphs.json",
        [
            {
                "paragraph_id": "S1.1",
                "block_id": "S1",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "MALDI-TOF improved resistant-isolate detection [P1 p2].",
                "citations": ["P1 p2"],
                "evidence_count": 1,
            },
            {
                "paragraph_id": "S2.1",
                "block_id": "S2",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "MALDI-TOF showed no improvement for resistant-isolate detection [P2 p2].",
                "citations": ["P2 p2"],
                "evidence_count": 1,
            },
            {
                "paragraph_id": "S3.1",
                "block_id": "S3",
                "section": "Result",
                "evidence_type": "result",
                "support_status": "SUPPORTED",
                "reason": "page_anchored",
                "paragraph": "The classifier achieved AUROC 0.91 [P3 p4].",
                "citations": ["P3 p4"],
                "evidence_count": 1,
            },
        ],
    )
    _write_json(package_dir / "blocked_paragraphs.json", [])
    _write_json(package_dir / "material_gaps.json", [])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
