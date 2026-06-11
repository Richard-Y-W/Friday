import unittest

from jarvis_research.discovery import Candidate
from jarvis_research.relevance import rank_candidates, score_candidate


class RelevanceTests(unittest.TestCase):
    def test_scores_biomedical_maldi_amr_candidate_highly(self):
        candidate = Candidate(
            provider="openalex",
            title="Rapid AMR prediction in Pseudomonas aeruginosa combining MALDI-TOF MS with DNN model",
            source_for_gate="10.1093/jambio/lxad248",
            abstract=(
                "Antimicrobial resistance prediction from MALDI-TOF mass spectrometry "
                "for clinical bacterial isolates."
            ),
        )

        scored = score_candidate("MALDI AMR", candidate)

        self.assertGreaterEqual(scored.relevance_score, 70)
        self.assertIn("biomedical_terms", scored.relevance_reason)
        self.assertIn("amr_context", scored.relevance_reason)

    def test_scores_abstract_meaning_representation_candidate_low(self):
        candidate = Candidate(
            provider="arxiv",
            title="Pushing the Limits of AMR Parsing with Self-Learning",
            source_for_gate="https://arxiv.org/pdf/2010.10673v1",
            abstract=(
                "Abstract Meaning Representation parsing maps natural language "
                "sentences into semantic graphs."
            ),
        )

        scored = score_candidate("MALDI AMR", candidate)

        self.assertLessEqual(scored.relevance_score, 20)
        self.assertIn("nlp_amr_penalty", scored.relevance_reason)

    def test_rank_candidates_puts_biomedical_result_before_nlp_amr_collision(self):
        nlp_candidate = Candidate(
            provider="arxiv",
            title="A Survey: Neural Networks for AMR-to-Text",
            source_for_gate="https://arxiv.org/pdf/2206.07328v2",
            abstract="Natural language generation from Abstract Meaning Representation graphs.",
        )
        biomedical_candidate = Candidate(
            provider="pubmed",
            title="Early antifungal resistance prediction based on MALDI-TOF mass spectrometry",
            source_for_gate="10.1038/s41598-026-53519-y",
            abstract="Machine learning predicts antifungal resistance from MALDI-TOF spectra.",
        )

        ranked = rank_candidates("MALDI AMR", [nlp_candidate, biomedical_candidate])

        self.assertEqual(ranked[0].title, biomedical_candidate.title)
        self.assertGreater(ranked[0].relevance_score, ranked[1].relevance_score)

    def test_rank_candidates_respects_nlp_amr_query_intent(self):
        biomedical_candidate = Candidate(
            provider="pubmed",
            title="MALDI-TOF antimicrobial resistance prediction",
            source_for_gate="10.1000/biomedical-amr",
            doi="10.1000/biomedical-amr",
            abstract="Antimicrobial resistance and antibiotic susceptibility from MALDI spectra.",
            mesh_terms="Drug Resistance, Microbial; Mass Spectrometry",
            concepts="antimicrobial resistance; microbiology",
            year=2025,
        )
        nlp_candidate = Candidate(
            provider="arxiv",
            title="AMR parsing with semantic graph generation",
            source_for_gate="https://arxiv.org/pdf/2401.54321",
            arxiv_id="2401.54321",
            abstract="Abstract meaning representation parsing for natural language text generation and semantic graphs.",
            concepts="natural language processing; semantic graph parsing",
            year=2025,
        )

        ranked = rank_candidates("AMR parsing", [biomedical_candidate, nlp_candidate])

        self.assertEqual(ranked[0].source_for_gate, nlp_candidate.source_for_gate)
        self.assertIn("nlp_amr_context", ranked[0].relevance_reason)

    def test_ranking_uses_mesh_and_openalex_concepts_as_biomedical_signals(self):
        metadata_rich = Candidate(
            provider="pubmed",
            title="Cross-site model generalization",
            source_for_gate="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            abstract="Clinical spectra model validation.",
            mesh_terms="Drug Resistance, Microbial; Mass Spectrometry",
            concepts="Antimicrobial resistance; Mass spectrometry",
            journal="Journal of Clinical Microbiology",
        )
        metadata_poor = Candidate(
            provider="pubmed",
            title="Cross-site model generalization",
            source_for_gate="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            abstract="Clinical spectra model validation.",
        )

        ranked = rank_candidates("MALDI AMR", [metadata_poor, metadata_rich])

        self.assertEqual(ranked[0].source_for_gate, metadata_rich.source_for_gate)
        self.assertIn("metadata_terms", ranked[0].relevance_reason)

    def test_rank_candidates_respects_mathematical_linguistics_query_intent(self):
        formal_language = Candidate(
            provider="openalex",
            title="Algebraic automata in formal language theory",
            source_for_gate="10.1000/automata-language",
            abstract="Formal grammars, automata, syntax, and algebra model language structure.",
            concepts="formal language theory; mathematical linguistics",
            year=2023,
        )
        clinical_language = Candidate(
            provider="pubmed",
            title="Language recovery after stroke",
            source_for_gate="10.1000/stroke-language",
            abstract="Clinical patient study of aphasia therapy and language recovery.",
            concepts="clinical neuroscience",
            year=2023,
        )

        ranked = rank_candidates("formal language automata", [clinical_language, formal_language])

        self.assertEqual(ranked[0].source_for_gate, formal_language.source_for_gate)
        self.assertIn("math_language_context", ranked[0].relevance_reason)
        self.assertIn("math_language_wrong_domain_penalty", ranked[1].relevance_reason)


if __name__ == "__main__":
    unittest.main()
