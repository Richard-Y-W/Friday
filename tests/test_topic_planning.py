import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.topic_planning import (
    build_topic_audit,
    evaluate_topic_curation,
    load_seed_profiles,
    load_learned_profiles,
    mine_metadata_profile,
    plan_topic,
    update_topic_memory,
)


class TopicPlanningTests(unittest.TestCase):
    def test_builds_stochastic_calculus_topic_profile(self):
        profile = plan_topic("stochastic calculus")

        self.assertIn("math_probability.stochastic_calculus", profile.topic_ids)
        self.assertEqual(profile.domain, "math_probability")
        self.assertIn("stochastic calculus", profile.core_terms)
        self.assertIn("Ito calculus", profile.positive_terms)
        self.assertIn("Brownian motion", profile.positive_terms)
        self.assertIn("stochastic differential equations", profile.positive_terms)
        self.assertIn("generic calculus", profile.negative_terms)
        self.assertIn("stochastic differential equations", profile.search_queries)
        self.assertIn("martingale stochastic calculus", profile.search_queries)
        self.assertIn("arxiv", profile.source_preferences)
        self.assertEqual(profile.evidence_policy_hint, "math_theory")

    def test_loads_seed_profiles_from_json_registry(self):
        profiles = load_seed_profiles()
        profile_ids = {profile.profile_id for profile in profiles}

        self.assertIn("math_probability.stochastic_calculus", profile_ids)
        self.assertIn("biomedical_amr.core", profile_ids)
        self.assertIn("instrumentation.mass_spectrometry", profile_ids)
        self.assertIn("machine_learning.core", profile_ids)

    def test_composes_multiple_matching_seed_profiles(self):
        profile = plan_topic("MALDI AMR")

        self.assertIn("biomedical_amr.core", profile.topic_ids)
        self.assertIn("instrumentation.mass_spectrometry", profile.topic_ids)
        self.assertIn("antimicrobial resistance", profile.positive_terms)
        self.assertIn("MALDI-TOF antimicrobial susceptibility", profile.search_queries)
        self.assertIn("abstract meaning representation", profile.negative_terms)

    def test_composes_math_probability_with_quantitative_finance(self):
        profile = plan_topic("stochastic calculus in finance")

        self.assertIn("math_probability.stochastic_calculus", profile.topic_ids)
        self.assertIn("finance.quantitative_finance", profile.topic_ids)
        self.assertIn("option pricing", profile.positive_terms)
        self.assertIn("stochastic calculus finance", profile.search_queries)

    def test_composes_language_math_profiles(self):
        profile = plan_topic("language is math")

        self.assertIn("mathematical_linguistics.core", profile.topic_ids)
        self.assertIn("information_theory.core", profile.topic_ids)
        self.assertIn("formal language theory", profile.positive_terms)
        self.assertIn("information theory language", profile.search_queries)

    def test_unknown_topics_keep_single_clean_search_query(self):
        profile = plan_topic("protein folding")

        self.assertEqual(profile.domain, "unknown")
        self.assertEqual(profile.core_terms, ("protein folding",))
        self.assertEqual(profile.search_queries, ("protein folding",))

    def test_mines_session_profile_from_scholarly_metadata(self):
        candidates = [
            Candidate(
                provider="openalex",
                title="Protein folding dynamics with AlphaFold structural biology",
                source_for_gate="10.1000/folding-1",
                abstract="Protein folding and structural biology benchmarks.",
                concepts="Protein folding; Structural biology; Molecular dynamics",
            ),
            Candidate(
                provider="openalex",
                title="Protein folding pathways in molecular dynamics",
                source_for_gate="10.1000/folding-2",
                abstract="Molecular dynamics simulations model protein folding pathways.",
                concepts="Protein folding; Molecular dynamics",
            ),
            Candidate(
                provider="arxiv",
                title="Graph layout optimization",
                source_for_gate="https://arxiv.org/pdf/2401.11111",
                abstract="Layout algorithms for networks.",
                concepts="Graph drawing",
            ),
        ]

        profile = mine_metadata_profile("protein folding", candidates)

        self.assertEqual(profile.domain, "session")
        self.assertIn("Protein folding", profile.positive_terms)
        self.assertIn("Molecular dynamics", profile.positive_terms)
        self.assertIn("protein folding Molecular dynamics", profile.search_queries)

    def test_updates_and_reuses_learned_topic_memory(self):
        relevant = [
            Candidate(
                provider="openalex",
                title="Protein folding dynamics",
                source_for_gate="10.1000/folding-1",
                abstract="Protein folding and structural biology.",
                concepts="Protein folding; Structural biology; Molecular dynamics",
            )
        ]
        irrelevant = [
            Candidate(
                provider="arxiv",
                title="Paper folding geometry",
                source_for_gate="https://arxiv.org/pdf/2401.22222",
                abstract="Origami and paper folding geometry.",
                concepts="Geometry; Origami",
            )
        ]

        with TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            learned = update_topic_memory(
                memory_dir,
                "protein folding",
                relevant_records=relevant,
                irrelevant_records=irrelevant,
            )
            loaded = load_learned_profiles(memory_dir)
            planned = plan_topic("protein folding", learned_profile_dir=memory_dir)

        self.assertEqual(learned.profile_id, "learned.protein_folding")
        self.assertEqual([profile.profile_id for profile in loaded], ["learned.protein_folding"])
        self.assertIn("Protein folding", planned.positive_terms)
        self.assertIn("Origami", planned.negative_terms)
        self.assertIn("learned.protein_folding", planned.topic_ids)

    def test_topic_curation_blocks_broad_amr_without_maldi_focus(self):
        profile = plan_topic("MALDI AMR")
        focused = Candidate(
            provider="openalex",
            title="Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra",
            source_for_gate="10.1000/focused",
            abstract="MALDI-TOF mass spectra support antibiotic resistance prediction.",
        )
        broad = Candidate(
            provider="openalex",
            title="Antimicrobial resistance surveillance in Africa",
            source_for_gate="10.1000/broad",
            abstract="Antimicrobial resistance surveillance across clinical isolates.",
        )

        focused_decision = evaluate_topic_curation(focused, profile)
        broad_decision = evaluate_topic_curation(broad, profile)

        self.assertTrue(focused_decision.eligible_for_deep_read)
        self.assertIn("MALDI-TOF", focused_decision.matched_positive_terms)
        self.assertFalse(broad_decision.eligible_for_deep_read)
        self.assertEqual(broad_decision.status, "topic_mismatch")

    def test_topic_curation_ignores_query_variant_for_focus_terms(self):
        profile = plan_topic("MALDI AMR")
        broad = Candidate(
            provider="openalex",
            title="Antimicrobial resistance surveillance in Africa",
            source_for_gate="10.1000/broad-query-variant",
            abstract="Antimicrobial resistance surveillance across clinical isolates.",
            query_variant="MALDI antimicrobial resistance",
        )

        decision = evaluate_topic_curation(broad, profile)

        self.assertFalse(decision.eligible_for_deep_read)
        self.assertNotIn("MALDI", decision.matched_query_terms)

    def test_composite_topic_curation_requires_all_components(self):
        profile = plan_topic("MALDI AMR")
        broad_amr = Candidate(
            provider="openalex",
            title="Antimicrobial resistance surveillance in clinical isolates",
            source_for_gate="10.1000/broad-amr",
            abstract="Antibiotic resistance and clinical isolate surveillance.",
        )
        maldi_only = Candidate(
            provider="openalex",
            title="MALDI-TOF species identification by mass spectrometry",
            source_for_gate="10.1000/maldi-only",
            abstract="Mass spectra support bacterial identification.",
        )
        focused = Candidate(
            provider="openalex",
            title="MALDI-TOF antimicrobial resistance detection",
            source_for_gate="10.1000/maldi-amr",
            abstract="MALDI-TOF mass spectra support antibiotic resistance detection.",
        )

        broad_decision = evaluate_topic_curation(broad_amr, profile)
        maldi_decision = evaluate_topic_curation(maldi_only, profile)
        focused_decision = evaluate_topic_curation(focused, profile)

        self.assertFalse(broad_decision.eligible_for_deep_read)
        self.assertEqual(broad_decision.reason, "missing_topic_component")
        self.assertFalse(maldi_decision.eligible_for_deep_read)
        self.assertEqual(maldi_decision.reason, "missing_topic_component")
        self.assertTrue(focused_decision.eligible_for_deep_read)

    def test_metadata_mined_topic_curation_blocks_generic_protein_paper(self):
        profile = mine_metadata_profile(
            "protein folding",
            [
                Candidate(
                    provider="openalex",
                    title="Protein folding dynamics with AlphaFold structural biology",
                    source_for_gate="10.1000/folding-1",
                    abstract="Protein folding and structural biology benchmarks.",
                    concepts="Protein folding; Structural biology; Molecular dynamics; Biochemistry",
                ),
                Candidate(
                    provider="openalex",
                    title="Protein folding pathways in molecular dynamics",
                    source_for_gate="10.1000/folding-2",
                    abstract="Molecular dynamics simulations model protein folding pathways.",
                    concepts="Protein folding; Molecular dynamics; Biochemistry",
                ),
            ],
        )
        generic = Candidate(
            provider="openalex",
            title="Protein measurement with the Folin phenol reagent",
            source_for_gate="10.1000/protein-assay",
            abstract="A biochemical assay measures protein concentration in samples.",
            concepts="Protein; Biochemistry; Chemistry",
        )
        focused = Candidate(
            provider="openalex",
            title="Protein folding dynamics in molecular simulations",
            source_for_gate="10.1000/focused-folding",
            abstract="Molecular dynamics simulations characterize protein folding pathways.",
            concepts="Protein folding; Molecular dynamics; Structural biology",
        )

        generic_decision = evaluate_topic_curation(generic, profile)
        focused_decision = evaluate_topic_curation(focused, profile)

        self.assertFalse(generic_decision.eligible_for_deep_read)
        self.assertNotIn("Biochemistry", profile.positive_terms)
        self.assertTrue(focused_decision.eligible_for_deep_read)

    def test_topic_curation_blocks_clinical_noise_without_query_focus(self):
        profile = plan_topic("sepsis procalcitonin antibiotic stewardship")
        focused = Candidate(
            provider="pubmed",
            title="Procalcitonin-guided antibiotic stewardship in sepsis",
            source_for_gate="10.1000/sepsis",
            abstract="Patients with sepsis received procalcitonin-guided antibiotic stewardship.",
        )
        noise = Candidate(
            provider="pubmed",
            title="Clinical diagnostic accuracy of Parkinson's disease",
            source_for_gate="10.1000/parkinsons",
            abstract="Diagnostic accuracy in patients with Parkinson's disease.",
        )

        self.assertTrue(evaluate_topic_curation(focused, profile).eligible_for_deep_read)
        noise_decision = evaluate_topic_curation(noise, profile)
        self.assertFalse(noise_decision.eligible_for_deep_read)
        self.assertEqual(noise_decision.reason, "missing_query_focus")

    def test_build_topic_audit_reports_profile_and_curation_counts(self):
        profile = plan_topic("MALDI AMR")
        items = [
            Candidate(
                provider="openalex",
                title="MALDI-TOF antibiotic resistance detection",
                source_for_gate="10.1000/focused",
                abstract="MALDI-TOF spectra for antibiotic resistance detection.",
            ),
            Candidate(
                provider="openalex",
                title="Antimicrobial resistance surveillance",
                source_for_gate="10.1000/broad",
                abstract="Antimicrobial resistance in clinical isolates.",
            ),
        ]

        audit = build_topic_audit("MALDI AMR", items, topic_profile=profile)

        self.assertIn("biomedical_amr.core", audit["profile"]["topic_ids"])
        self.assertEqual(audit["curation"]["eligible_for_deep_read_count"], 1)
        self.assertEqual(audit["curation"]["blocked_by_topic_count"], 1)
        self.assertEqual(audit["items"][1]["status"], "topic_mismatch")


if __name__ == "__main__":
    unittest.main()
