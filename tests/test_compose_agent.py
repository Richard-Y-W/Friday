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
from friday.report_composer import (
    build_full_report_package_files,
    build_report_discourse_plan,
    build_report_faithfulness_audit,
    build_report_plan_adherence_audit,
    build_report_prose_quality_audit,
    build_report_trust_score,
)


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

    def test_llm_compose_uses_evidence_plan_to_hide_excluded_rows_from_composer(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "schema_version": "1.0",
                    "artifact_type": "writing_evidence_tables",
                    "all_rows": [
                        {
                            "row_id": "R1",
                            "claim_id": "C1",
                            "support_status": "SUPPORTED",
                            "table": "results",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P2 p2",
                            "page_number": 2,
                            "text": "Formally, the learning objective is defined as: L = CE(z,y) + mu L_SNP.",
                            "trust_label": "trusted",
                        },
                        {
                            "row_id": "R2",
                            "claim_id": "C2",
                            "support_status": "SUPPORTED",
                            "table": "results",
                            "evidence_type": "result",
                            "paper": "P2",
                            "citation": "P2 p2",
                            "page_number": 2,
                            "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            "trust_label": "trusted",
                        },
                    ],
                    "tables": {
                        "results": [
                            {
                                "row_id": "R1",
                                "evidence_type": "result",
                                "paper": "P1",
                                "citation": "P2 p2",
                                "page_number": 2,
                                "text": "Formally, the learning objective is defined as: L = CE(z,y) + mu L_SNP.",
                                "trust_label": "trusted",
                            },
                            {
                                "row_id": "R2",
                                "evidence_type": "result",
                                "paper": "P2",
                                "citation": "P2 p2",
                                "page_number": 2,
                                "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                                "trust_label": "trusted",
                            },
                        ]
                    },
                },
            )
            router = FakeRouter(
                {
                    "planner": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text=json.dumps(
                            {
                                "rows": [
                                    {
                                        "row_id": "R1",
                                        "citation": "P2 p2",
                                        "role": "formula_detail",
                                        "action": "appendix",
                                        "reason": "formula detail belongs outside prose",
                                    },
                                    {
                                        "row_id": "R2",
                                        "citation": "P2 p2",
                                        "role": "result",
                                        "action": "include",
                                        "reason": "result evidence belongs in prose",
                                    },
                                ]
                            }
                        ),
                    ),
                    "composer": LLMResponse(
                        provider="claude_cli",
                        model="sonnet",
                        success=True,
                        text="# Results\n\nThe classifier detected resistant isolates with 88 percent sensitivity [P2 p2].",
                    ),
                    "verifier": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "pass",
                                "summary": "Supported.",
                                "unsupported_claims": [],
                                "citation_errors": [],
                                "missing_material_gaps": [],
                            }
                        ),
                    ),
                }
            )

            files = build_llm_compose_package_files(package_dir, section="results", router=router)

            self.assertIn("evidence_plan.json", files)
            evidence_plan = json.loads(files["evidence_plan.json"])
            discourse_plan = json.loads(files["discourse_plan.json"])
            composer_prompt = json.loads(json.loads(files["composer_prompt.json"])["prompt"])

            self.assertEqual(evidence_plan["appendix_row_ids"], ["R1"])
            self.assertEqual(evidence_plan["included_row_ids"], ["R2"])
            self.assertEqual(evidence_plan["appendix_citations"], ["P2 p2"])
            self.assertEqual(evidence_plan["included_citations"], ["P2 p2"])
            self.assertEqual(discourse_plan["required_citations"], ["P2 p2"])
            self.assertNotIn("learning objective", json.dumps(composer_prompt["atomic_evidence_rows"]))
            self.assertIn("88 percent sensitivity", json.dumps(composer_prompt["atomic_evidence_rows"]))
            self.assertEqual([role for role, _request in router.calls], ["planner", "composer", "verifier"])

    def test_full_report_package_assembles_sections_pdf_and_audits(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            self.assertIn("report.md", files)
            self.assertTrue(files["report.pdf"].startswith(b"%PDF-1.4"))
            self.assertIn("citation_audit.json", files)
            self.assertIn("report_manifest.json", files)
            self.assertIn("evidence_table.md", files)
            self.assertIn("literature_table.md", files)
            self.assertIn("sections/results/draft.md", files)
            report = files["report.md"]
            self.assertIn("# Friday Research Report", report)
            self.assertIn("## Executive Summary", report)
            self.assertIn("## Background", report)
            self.assertIn("## Methods", report)
            self.assertIn("## Results", report)
            self.assertIn("## Limitations", report)
            self.assertIn("## Evidence Table", report)
            self.assertIn("## Literature", report)
            self.assertIn("AUROC 0.91", report)
            self.assertIn("[1, p. 2; 2, p. 2]", report)
            self.assertIn("MATERIAL GAP: No dedicated background evidence was available in this writing package.", report)
            self.assertIn("MATERIAL GAP: No supported limitation evidence is available in this writing package.", report)
            self.assertIn("MATERIAL GAP: No page-anchored limitation evidence is available in this batch.", report)
            audit = json.loads(files["citation_audit.json"])
            self.assertEqual(audit["artifact_type"], "full_report_citation_audit")
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["sections"]["results"]["claim_audit_status"], "pass")
            self.assertIn("P1 p2", audit["used_citations"])

    def test_full_report_uses_reader_facing_citations_and_readable_synthesis(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            report = files["report.md"]
            self.assertNotIn("evidence includes", report.lower())
            self.assertNotIn("[P1 p2", report)
            self.assertIn("[1, p. 2; 2, p. 2]", report)
            self.assertIn(
                "- **Results:** Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].",
                report,
            )
            self.assertIn("\n---\n\n## Background\n", report)
            self.assertIn("\n---\n\n## Evidence Table\n", report)

            audit = json.loads(files["citation_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertIn("P1 p2", audit["used_citations"])
            self.assertIn("P2 p2", audit["used_citations"])

    def test_full_report_pdf_uses_styled_section_fonts_and_heading_color(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            pdf = files["report.pdf"]
            self.assertIn(b"/BaseFont /Helvetica-Bold", pdf)
            self.assertIn(b"0.10 0.35 0.52 rg", pdf)
            self.assertIn(b"/F2 14 Tf", pdf)

    def test_full_report_prefers_atomic_rows_for_readable_fallback_body(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P1 p2",
                            "text": "The model achieved AUROC 0.91.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                        {
                            "row_id": "E2",
                            "evidence_type": "result",
                            "paper": "P2",
                            "citation": "P2 p2",
                            "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                    ],
                    "tables": {
                        "results": [
                            {
                                "row_id": "E1",
                                "evidence_type": "result",
                                "paper": "P1",
                                "citation": "P1 p2",
                                "text": "The model achieved AUROC 0.91.",
                                "support_status": "SUPPORTED",
                                "quality_label": "clean",
                            },
                            {
                                "row_id": "E2",
                                "evidence_type": "result",
                                "paper": "P2",
                                "citation": "P2 p2",
                                "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                                "support_status": "SUPPORTED",
                                "quality_label": "clean",
                            },
                        ]
                    },
                },
            )

            files = build_full_report_package_files(package_dir)

            report = files["report.md"]
            self.assertIn("One paper reported that the model achieved AUROC 0.91 [1, p. 2].", report)
            self.assertIn(
                "A second paper reported that the classifier detected resistant isolates with 88 percent sensitivity [2, p. 2].",
                report,
            )
            self.assertNotIn("Two papers reported AUROC 0.91 and sensitivity 88 percent", report)

    def test_full_report_exports_discourse_plan_and_connected_atomic_paragraphs(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P1 p2",
                            "text": "The model achieved AUROC 0.91.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                        {
                            "row_id": "E2",
                            "evidence_type": "result",
                            "paper": "P2",
                            "citation": "P2 p2",
                            "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                    ]
                },
            )

            files = build_full_report_package_files(package_dir)

            self.assertIn("report_discourse_plan.json", files)
            plan = json.loads(files["report_discourse_plan.json"])
            self.assertEqual(plan["artifact_type"], "report_discourse_plan")
            self.assertEqual(plan["sections"]["results"]["moves"][0]["kind"], "evidence_cluster")
            self.assertEqual(plan["sections"]["results"]["moves"][0]["row_ids"], ["E1", "E2"])
            self.assertEqual(plan["sections"]["results"]["moves"][0]["citations"], ["P1 p2", "P2 p2"])

            report = files["report.md"]
            self.assertIn(
                "- **Results:** One paper reported that the model achieved AUROC 0.91 [1, p. 2].",
                report,
            )
            self.assertIn(
                "The results section draws on two page-anchored findings. "
                "One paper reported that the model achieved AUROC 0.91 [1, p. 2]. "
                "A second paper reported that the classifier detected resistant isolates with 88 percent sensitivity [2, p. 2].",
                report,
            )

    def test_report_plan_adherence_audit_flags_missing_planned_move_citations(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P1 p2",
                            "text": "The model achieved AUROC 0.91.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                        {
                            "row_id": "E2",
                            "evidence_type": "result",
                            "paper": "P2",
                            "citation": "P2 p2",
                            "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                    ]
                },
            )
            package = load_writing_package(package_dir)
            plan = build_report_discourse_plan(package)
            report = (
                "# Friday Research Report\n\n"
                "## Results\n\n"
                "One paper reported that the model achieved AUROC 0.91 [1, p. 2].\n\n"
                "---\n\n"
                "## Evidence Table\n\n"
                "| Section | Evidence | Citations |\n"
                "| --- | --- | --- |\n"
                "| result | AUROC 0.91 | 1, p. 2 |\n"
            )
            from friday.claim_decomposition import build_report_claim_units

            claim_units = build_report_claim_units(report, package)

            audit = build_report_plan_adherence_audit(report, plan, claim_units)

            self.assertEqual(audit["artifact_type"], "report_plan_adherence_audit")
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["checked_move_count"], 1)
            self.assertEqual(audit["issues"][0]["rule"], "partial_planned_move")
            self.assertEqual(audit["issues"][0]["missing_citations"], ["P2 p2"])

    def test_full_report_exports_plan_adherence_audit_and_manifest_status(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P1 p2",
                            "text": "The model achieved AUROC 0.91.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        }
                    ]
                },
            )

            files = build_full_report_package_files(package_dir)

            self.assertIn("report_plan_adherence_audit.json", files)
            audit = json.loads(files["report_plan_adherence_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["checked_move_count"], 1)
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["plan_adherence_status"], "pass")
            self.assertEqual(manifest["plan_adherence_issue_count"], 0)

    def test_full_report_atomic_row_rewrite_does_not_create_studys_study(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "limitation",
                            "paper": "P1",
                            "citation": "P1 p15",
                            "text": "Our study did have its limitations.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        }
                    ]
                },
            )

            files = build_full_report_package_files(package_dir)

            report = files["report.md"]
            self.assertIn("One paper noted that the study had limitations [1, p. 15].", report)
            self.assertNotIn("study's study", report)

    def test_report_prose_quality_audit_flags_internal_syntax_and_dump_phrases(self):
        report = (
            "# Friday Research Report\n\n"
            "## Results\n\n"
            "Across 3 papers, claim evidence includes in this work, we present a model [P1 p2; P2 p3].\n"
        )

        audit = build_report_prose_quality_audit(report)

        self.assertEqual(audit["artifact_type"], "report_prose_quality_audit")
        self.assertEqual(audit["status"], "fallback")
        self.assertEqual(
            [issue["rule"] for issue in audit["issues"]],
            [
                "missing_required_heading",
                "internal_citation_syntax",
                "raw_evidence_dump_phrase",
                "source_author_voice",
            ],
        )

    def test_report_prose_quality_audit_consumes_applied_feedback_blocked_phrases(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            _write_feedback_rule_store(
                data_dir,
                value="clinically definitive",
                reason="Human feedback said this phrasing was too broad.",
            )
            report = (
                "# Friday Research Report\n\n"
                "## Executive Summary\n\n"
                "- **Results:** One paper reported a clinically definitive signal [1, p. 2].\n\n"
                "## Background\n\n"
                "One paper described spectra classifiers [1, p. 1].\n\n"
                "## Methods\n\n"
                "One paper described spectra classifiers [1, p. 1].\n\n"
                "## Results\n\n"
                "One paper reported a clinically definitive signal [1, p. 2].\n\n"
                "## Limitations\n\n"
                "One paper noted a limitation [1, p. 3].\n"
            )

            audit = build_report_prose_quality_audit(report, feedback_data_dir=data_dir)

            self.assertEqual(audit["status"], "fallback")
            learned = [issue for issue in audit["issues"] if issue["rule"] == "feedback_blocked_phrase"]
            self.assertEqual(len(learned), 1)
            self.assertEqual(learned[0]["phrase"], "clinically definitive")
            self.assertIn("too broad", learned[0]["reason"])
            self.assertEqual(audit["feedback_rule_count"], 1)

    def test_full_report_llm_candidate_falls_back_on_feedback_prose_rule(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "package"
            data_dir = root / ".friday"
            _write_fixture_package(package_dir)
            _write_feedback_rule_store(
                data_dir,
                value="clinically definitive",
                reason="Human feedback said this phrasing was too broad.",
            )
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=(
                            "# Friday Research Report\n\n"
                            "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                            "## Executive Summary\n\n"
                            "- **Results:** Two papers reported clinically definitive AUROC and sensitivity signals [1, p. 2; 2, p. 2].\n\n"
                            "---\n\n"
                            "## Background\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Methods\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Results\n\n"
                            "Two papers reported clinically definitive AUROC and sensitivity signals [1, p. 2; 2, p. 2].\n\n"
                            "---\n\n"
                            "## Limitations\n\n"
                            "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                            "## Evidence Table\n\n"
                            "| Paper | Evidence |\n"
                            "|---|---|\n"
                            "| P1 | AUROC 0.91 |\n\n"
                            "## Literature\n\n"
                            "| Paper | Title |\n"
                            "|---|---|\n"
                            "| 1 | MALDI antimicrobial resistance prediction |\n\n"
                            "## Citation Audit\n\n"
                            "- Citations checked.\n"
                        ),
                    )
                }
            )

            files = build_full_report_package_files(
                package_dir,
                router=router,
                use_report_llm=True,
                feedback_data_dir=data_dir,
            )

            composer_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(composer_audit["reason"], "prose_quality_failed")
            self.assertEqual(composer_audit["candidate_prose_quality_status"], "fallback")
            self.assertNotIn("clinically definitive", files["report.md"])

    def test_full_report_exports_prose_quality_audit_and_manifest_status(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            self.assertIn("report_prose_quality.json", files)
            audit = json.loads(files["report_prose_quality.json"])
            self.assertEqual(audit["status"], "pass")
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["prose_quality_status"], "pass")
            self.assertEqual(manifest["report_source"], "deterministic")

    def test_full_report_llm_uses_discourse_plan_and_accepts_quality_checked_candidate(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=(
                            "# Friday Research Report\n\n"
                            "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                            "## Executive Summary\n\n"
                            "- **Results:** Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
                            "---\n\n"
                            "## Background\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Methods\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Results\n\n"
                            "Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
                            "---\n\n"
                            "## Limitations\n\n"
                            "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                            "---\n\n"
                            "## Evidence Table\n\n"
                            "| Section | Evidence | Citations |\n"
                            "| --- | --- | --- |\n"
                            "| result | AUROC 0.91 | 1, p. 2 |\n\n"
                            "---\n\n"
                            "## Literature\n\n"
                            "| Paper | Title | Year | Venue | DOI |\n"
                            "| --- | --- | --- | --- | --- |\n"
                            "| P1 | MALDI antimicrobial resistance prediction | 2024 | Nature Medicine | 10.1038/example-a |\n\n"
                            "---\n\n"
                            "## Citation Audit\n\n"
                            "- Status: pass\n"
                            "- Used citations: 4\n"
                            "- Unknown citations: 0\n"
                        ),
                    ),
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            self.assertIn("report_llm_draft.md", files)
            self.assertIn("report_composer_prompt.json", files)
            self.assertIn("report_composer_audit.json", files)
            self.assertIn("Two papers reported AUROC 0.91", files["report.md"])
            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "pass")
            self.assertEqual(report_audit["final_report_source"], "llm")
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["report_source"], "llm")
            prompt = json.loads(files["report_composer_prompt.json"])
            self.assertIn("report_discourse_plan", prompt["prompt"])
            self.assertIn("EVIDENCE-BOUND", prompt["system_prompt"])
            self.assertEqual(router.calls[0][0], "composer")

    def test_full_report_llm_falls_back_when_candidate_fails_prose_quality(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=(
                            "# Friday Research Report\n\n"
                            "## Results\n\n"
                            "Across 2 papers, result evidence includes AUROC 0.91 [P1 p2; P2 p2].\n"
                        ),
                    ),
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            self.assertIn("report_llm_draft.md", files)
            self.assertIn("AUROC 0.91", files["report.md"])
            self.assertNotIn("[P1 p2", files["report.md"])
            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "fallback")
            self.assertEqual(report_audit["reason"], "prose_quality_failed")
            self.assertEqual(report_audit["final_report_source"], "deterministic")
            self.assertEqual(json.loads(files["report_prose_quality.json"])["status"], "pass")
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["report_source"], "deterministic")

    def test_full_report_llm_falls_back_when_candidate_violates_discourse_plan(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            _write_json(
                package_dir / "evidence_tables.json",
                {
                    "all_rows": [
                        {
                            "row_id": "E1",
                            "evidence_type": "result",
                            "paper": "P1",
                            "citation": "P1 p2",
                            "text": "The model achieved AUROC 0.91.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                        {
                            "row_id": "E2",
                            "evidence_type": "result",
                            "paper": "P2",
                            "citation": "P2 p2",
                            "text": "The classifier detected resistant isolates with 88 percent sensitivity.",
                            "support_status": "SUPPORTED",
                            "quality_label": "clean",
                        },
                    ]
                },
            )
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=(
                            "# Friday Research Report\n\n"
                            "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                            "## Executive Summary\n\n"
                            "- **Results:** One paper reported that the model achieved AUROC 0.91 [1, p. 2].\n\n"
                            "---\n\n"
                            "## Background\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Methods\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Results\n\n"
                            "One paper reported that the model achieved AUROC 0.91 [1, p. 2].\n\n"
                            "---\n\n"
                            "## Limitations\n\n"
                            "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                            "---\n\n"
                            "## Evidence Table\n\n"
                            "| Section | Evidence | Citations |\n"
                            "| --- | --- | --- |\n"
                            "| result | AUROC 0.91 | 1, p. 2 |\n\n"
                            "---\n\n"
                            "## Literature\n\n"
                            "| Paper | Title | Year | Venue | DOI |\n"
                            "| --- | --- | --- | --- | --- |\n"
                            "| P1 | MALDI antimicrobial resistance prediction | 2024 | Nature Medicine | 10.1038/example-a |\n\n"
                            "---\n\n"
                            "## Citation Audit\n\n"
                            "- Status: pass\n"
                        ),
                    )
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "fallback")
            self.assertEqual(report_audit["reason"], "plan_adherence_failed")
            self.assertEqual(report_audit["candidate_plan_adherence_status"], "fallback")
            self.assertIn("88 percent sensitivity", files["report.md"])
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["report_source"], "deterministic")

    def test_report_faithfulness_audit_flags_uncited_main_report_claims(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            package = load_writing_package(package_dir)
            report = (
                "# Friday Research Report\n\n"
                "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                "## Executive Summary\n\n"
                "- **Results:** This uncited sentence says MALDI is clinically validated for AMR deployment.\n\n"
                "---\n\n"
                "## Background\n\n"
                "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                "---\n\n"
                "## Methods\n\n"
                "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                "---\n\n"
                "## Results\n\n"
                "Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
                "---\n\n"
                "## Limitations\n\n"
                "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                "---\n\n"
                "## Evidence Table\n\n"
                "| Section | Evidence | Citations |\n| --- | --- | --- |\n| result | AUROC 0.91 | 1, p. 2 |\n\n"
                "---\n\n"
                "## Literature\n\n"
                "| Paper | Title | Year | Venue | DOI |\n| --- | --- | --- | --- | --- |\n| P1 | Title | 2024 | Journal | DOI |\n\n"
                "---\n\n"
                "## Citation Audit\n\n"
                "- Status: pass\n"
            )

            audit = build_report_faithfulness_audit(report, package)

            self.assertEqual(audit["artifact_type"], "report_faithfulness_audit")
            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["tier_a_status"], "fallback")
            self.assertEqual(audit["tier_b_status"], "pass")
            self.assertEqual(audit["issues"][0]["rule"], "uncited_factual_sentence")

    def test_report_faithfulness_audit_flags_cited_sentence_unsupported_by_evidence(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            package = load_writing_package(package_dir)
            report = (
                "# Friday Research Report\n\n"
                "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                "## Executive Summary\n\n"
                "- **Results:** One paper proved global hospital deployment with mortality benefit [1, p. 2].\n\n"
                "---\n\n"
                "## Background\n\n"
                "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                "---\n\n"
                "## Methods\n\n"
                "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                "---\n\n"
                "## Results\n\n"
                "One paper proved global hospital deployment with mortality benefit [1, p. 2].\n\n"
                "---\n\n"
                "## Limitations\n\n"
                "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                "---\n\n"
                "## Evidence Table\n\n"
                "| Section | Evidence | Citations |\n| --- | --- | --- |\n| result | AUROC 0.91 | 1, p. 2 |\n\n"
                "---\n\n"
                "## Literature\n\n"
                "| Paper | Title | Year | Venue | DOI |\n| --- | --- | --- | --- | --- |\n| P1 | Title | 2024 | Journal | DOI |\n\n"
                "---\n\n"
                "## Citation Audit\n\n"
                "- Status: pass\n"
            )

            audit = build_report_faithfulness_audit(report, package)

            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["tier_a_status"], "pass")
            self.assertEqual(audit["tier_b_status"], "fallback")
            unsupported = [issue for issue in audit["issues"] if issue["rule"] == "weak_evidence_overlap"]
            self.assertTrue(unsupported)
            self.assertIn("mortality", unsupported[0]["missing_terms"])

    def test_report_faithfulness_audit_exports_per_claim_unit_verdicts(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            package = load_writing_package(package_dir)
            report = (
                "# Friday Research Report\n\n"
                "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                "## Executive Summary\n\n"
                "- **Results:** Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n\n"
                "---\n\n"
                "## Results\n\n"
                "Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].\n"
            )

            audit = build_report_faithfulness_audit(report, package)

            self.assertEqual(audit["status"], "pass")
            self.assertGreaterEqual(audit["checked_claim_unit_count"], 1)
            supported = [
                unit
                for unit in audit["claim_units"]
                if "AUROC 0.91" in unit["text"] and unit["section"] == "Results"
            ]
            self.assertTrue(supported)
            self.assertEqual(supported[0]["verdict"], "supported")
            self.assertEqual(supported[0]["claim_type"], "synthesis")
            self.assertEqual(supported[0]["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(supported[0]["risk_terms"], [])

    def test_report_faithfulness_audit_flags_overstated_claim_units(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            package = load_writing_package(package_dir)
            report = (
                "# Friday Research Report\n\n"
                "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                "## Executive Summary\n\n"
                "- **Results:** One paper proved clinically definitive mortality benefit [1, p. 2].\n\n"
                "---\n\n"
                "## Results\n\n"
                "One paper proved clinically definitive mortality benefit [1, p. 2].\n"
            )

            audit = build_report_faithfulness_audit(report, package)

            self.assertEqual(audit["status"], "fallback")
            self.assertEqual(audit["tier_b_status"], "fallback")
            overstated = [issue for issue in audit["issues"] if issue["rule"] == "overstated_claim"]
            self.assertTrue(overstated)
            self.assertIn("proved", overstated[0]["risk_terms"])
            self.assertIn("clinically definitive", overstated[0]["risk_terms"])
            claim_verdicts = [unit for unit in audit["claim_units"] if unit["verdict"] == "overstated"]
            self.assertTrue(claim_verdicts)
            self.assertIn("mortality benefit", claim_verdicts[0]["risk_terms"])

    def test_full_report_exports_faithfulness_audit_and_manifest_status(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            self.assertIn("report_faithfulness_audit.json", files)
            audit = json.loads(files["report_faithfulness_audit.json"])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["tier_a_status"], "pass")
            self.assertEqual(audit["tier_b_status"], "pass")
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["faithfulness_status"], "pass")

    def test_full_report_exports_typed_claim_units(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            self.assertIn("claim_units.json", files)
            artifact = json.loads(files["claim_units.json"])
            self.assertEqual(artifact["artifact_type"], "report_claim_units")
            self.assertEqual(artifact["source_report"]["batch_id"], "batch_test")
            units = artifact["claim_units"]
            self.assertTrue(any(unit["claim_type"] == "synthesis" for unit in units))
            self.assertTrue(any(unit["claim_type"] == "material_gap" for unit in units))
            result_unit = next(unit for unit in units if "AUROC 0.91" in unit["text"])
            self.assertEqual(result_unit["citations"], ["P1 p2", "P2 p2"])
            self.assertEqual(result_unit["support_status"], "supported")
            self.assertEqual(result_unit["evidence_types"], ["result"])
            manifest = json.loads(files["report_manifest.json"])
            self.assertGreaterEqual(manifest["claim_unit_count"], 1)

    def test_full_report_llm_falls_back_when_candidate_fails_faithfulness(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=(
                            "# Friday Research Report\n\n"
                            "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
                            "## Executive Summary\n\n"
                            "- **Results:** One paper proved global hospital deployment with mortality benefit [1, p. 2].\n\n"
                            "---\n\n"
                            "## Background\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Methods\n\n"
                            "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
                            "---\n\n"
                            "## Results\n\n"
                            "One paper proved global hospital deployment with mortality benefit [1, p. 2].\n\n"
                            "---\n\n"
                            "## Limitations\n\n"
                            "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
                            "---\n\n"
                            "## Evidence Table\n\n"
                            "| Section | Evidence | Citations |\n"
                            "| --- | --- | --- |\n"
                            "| result | AUROC 0.91 | 1, p. 2 |\n\n"
                            "---\n\n"
                            "## Literature\n\n"
                            "| Paper | Title | Year | Venue | DOI |\n"
                            "| --- | --- | --- | --- | --- |\n"
                            "| P1 | MALDI antimicrobial resistance prediction | 2024 | Nature Medicine | 10.1038/example-a |\n\n"
                            "---\n\n"
                            "## Citation Audit\n\n"
                            "- Status: pass\n"
                            "- Used citations: 4\n"
                            "- Unknown citations: 0\n"
                        ),
                    ),
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            self.assertIn("report_llm_draft.md", files)
            self.assertNotIn("mortality benefit", files["report.md"])
            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "fallback")
            self.assertEqual(report_audit["reason"], "faithfulness_failed")
            self.assertEqual(report_audit["final_report_source"], "deterministic")
            self.assertEqual(json.loads(files["report_faithfulness_audit.json"])["status"], "pass")
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["report_source"], "deterministic")

    def test_full_report_llm_critic_revises_candidate_before_acceptance(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=_valid_full_report_text(
                                result_sentence="Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2]."
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=_valid_full_report_text(
                                result_sentence="In plain terms, two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2]."
                            ),
                        ),
                    ],
                    "critic": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps(
                                {
                                    "verdict": "fail",
                                    "summary": "The result prose is too table-like.",
                                    "issues": [
                                        {
                                            "severity": "important",
                                            "rule": "prose_clarity",
                                            "sentence": "Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2].",
                                        }
                                    ],
                                }
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=json.dumps({"verdict": "pass", "summary": "Revision is readable and evidence-bound.", "issues": []}),
                        ),
                    ],
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            self.assertIn("report_critic_prompt.json", files)
            self.assertIn("report_critic_audit.json", files)
            self.assertIn("report_revision_prompt.json", files)
            self.assertIn("report_revised_draft.md", files)
            critic_prompt = json.loads(files["report_critic_prompt.json"])
            critic_payload = json.loads(critic_prompt["prompt"])
            self.assertIn("report_claim_units", critic_payload)
            self.assertTrue(any(unit["claim_type"] == "synthesis" for unit in critic_payload["report_claim_units"]["claim_units"]))
            self.assertIn("In plain terms, two papers reported AUROC 0.91", files["report.md"])
            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "pass")
            self.assertEqual(report_audit["reason"], "critic_revision_accepted")
            self.assertEqual(report_audit["final_report_source"], "llm_revised")
            revision_audit = json.loads(files["report_revision_audit.json"])
            self.assertEqual(revision_audit["status"], "pass")
            self.assertEqual([role for role, _request in router.calls], ["composer", "critic", "composer", "critic"])

    def test_full_report_llm_falls_back_when_critic_revision_fails_gates(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": [
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=_valid_full_report_text(
                                result_sentence="Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2]."
                            ),
                        ),
                        LLMResponse(
                            provider="codex_cli",
                            model="",
                            success=True,
                            text=_valid_full_report_text(
                                result_sentence="One paper proved global hospital deployment with mortality benefit [1, p. 2]."
                            ),
                        ),
                    ],
                    "critic": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps(
                            {
                                "verdict": "fail",
                                "summary": "The report needs revision.",
                                "issues": [{"severity": "important", "rule": "faithfulness"}],
                            }
                        ),
                    ),
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            self.assertIn("report_revision_audit.json", files)
            self.assertNotIn("mortality benefit", files["report.md"])
            report_audit = json.loads(files["report_composer_audit.json"])
            self.assertEqual(report_audit["status"], "fallback")
            self.assertEqual(report_audit["reason"], "critic_revision_failed")
            self.assertEqual(report_audit["final_report_source"], "deterministic")
            revision_audit = json.loads(files["report_revision_audit.json"])
            self.assertEqual(revision_audit["status"], "fallback")
            self.assertEqual(revision_audit["faithfulness_status"], "fallback")

    def test_full_report_exports_trust_score_for_reviewable_deterministic_report(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)

            files = build_full_report_package_files(package_dir)

            self.assertIn("report_trust_score.json", files)
            trust = json.loads(files["report_trust_score.json"])
            self.assertEqual(trust["artifact_type"], "report_trust_score")
            self.assertEqual(trust["verdict"], "needs_review")
            self.assertEqual(trust["action"], "human_review")
            self.assertGreaterEqual(trust["score"], 70)
            self.assertIn("critic_not_run", trust["reasons"])
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["trust_verdict"], "needs_review")
            self.assertEqual(manifest["trust_action"], "human_review")

    def test_full_report_trust_score_is_publishable_after_critic_pass(self):
        with TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "package"
            _write_fixture_package(package_dir)
            router = FakeRouter(
                {
                    "composer": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=_valid_full_report_text(
                            result_sentence="Two papers reported AUROC 0.91 and sensitivity 88 percent [1, p. 2; 2, p. 2]."
                        ),
                    ),
                    "critic": LLMResponse(
                        provider="codex_cli",
                        model="",
                        success=True,
                        text=json.dumps({"verdict": "pass", "summary": "Evidence-bound and readable.", "issues": []}),
                    ),
                }
            )

            files = build_full_report_package_files(package_dir, router=router, use_report_llm=True)

            trust = json.loads(files["report_trust_score.json"])
            self.assertEqual(trust["verdict"], "publishable")
            self.assertEqual(trust["action"], "publish")
            self.assertGreaterEqual(trust["score"], 90)
            self.assertIn("critic_passed", trust["reasons"])
            manifest = json.loads(files["report_manifest.json"])
            self.assertEqual(manifest["trust_verdict"], "publishable")

    def test_report_trust_score_blocks_failed_required_gate(self):
        trust = build_report_trust_score(
            {"status": "pass"},
            {"status": "pass"},
            {"status": "fallback", "tier_a_status": "pass", "tier_b_status": "fallback"},
            report_composer_audit=None,
        )

        self.assertEqual(trust["verdict"], "blocked")
        self.assertEqual(trust["action"], "block")
        self.assertLess(trust["score"], 50)
        self.assertIn("faithfulness_failed", trust["reasons"])


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


def _valid_full_report_text(*, result_sentence: str) -> str:
    return (
        "# Friday Research Report\n\n"
        "Source: Batch `batch_test`; query `MALDI AMR`; screened `1000`; deep-read `50`\n\n"
        "## Executive Summary\n\n"
        f"- **Results:** {result_sentence}\n\n"
        "---\n\n"
        "## Background\n\n"
        "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
        "---\n\n"
        "## Methods\n\n"
        "Two papers described spectra classifiers [1, p. 1; 2, p. 1].\n\n"
        "---\n\n"
        "## Results\n\n"
        f"{result_sentence}\n\n"
        "---\n\n"
        "## Limitations\n\n"
        "- MATERIAL GAP: No page-anchored limitation evidence is available in this batch.\n\n"
        "---\n\n"
        "## Evidence Table\n\n"
        "| Section | Evidence | Citations |\n"
        "| --- | --- | --- |\n"
        "| result | AUROC 0.91 | 1, p. 2 |\n\n"
        "---\n\n"
        "## Literature\n\n"
        "| Paper | Title | Year | Venue | DOI |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| P1 | MALDI antimicrobial resistance prediction | 2024 | Nature Medicine | 10.1038/example-a |\n\n"
        "---\n\n"
        "## Citation Audit\n\n"
        "- Status: pass\n"
        "- Used citations: 4\n"
        "- Unknown citations: 0\n"
    )


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


def _write_feedback_rule_store(data_dir: Path, *, value: str, reason: str) -> None:
    _write_json(
        data_dir / "feedback" / "rules" / "prose_quality.json",
        {
            "schema_version": "1.0",
            "artifact_type": "feedback_rule_store",
            "target": "report_prose_quality",
            "rules": [
                {
                    "source_package": "fixture",
                    "target": "report_prose_quality",
                    "action": "add_blocked_phrase",
                    "value": value,
                    "reason": reason,
                    "proposal_summary": "Fixture learned prose rule.",
                    "created_at": "2026-06-13T00:00:00+00:00",
                }
            ],
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
