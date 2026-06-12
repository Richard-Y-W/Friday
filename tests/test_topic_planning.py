import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.topic_planning import (
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


if __name__ == "__main__":
    unittest.main()
