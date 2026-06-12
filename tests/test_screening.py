import unittest

from friday.discovery import Candidate
from friday.llm_labeling import LlmLabelResult
from friday.relevance import score_candidate
from friday.screening import (
    auto_label_batch_items,
    build_llm_review_queue,
    rank_deep_read_items,
    recommend_unlabeled_items,
)
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class ScreeningTests(unittest.TestCase):
    def _label_for(self, query, candidate):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query=query, limit=1, mode="screening_test")
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            result = auto_label_batch_items(store, batch.batch_id, query=query)

        return result.decisions[0].label

    def test_recommendations_prefer_terms_from_relevant_labels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=4, mode="query")
            relevant_seed = Candidate(
                provider="pubmed",
                title="MALDI-TOF antimicrobial resistance spectra",
                source_for_gate="10.1128/example.relevant",
                abstract="Antimicrobial resistance prediction from MALDI spectra.",
                relevance_score=40,
            )
            irrelevant_seed = Candidate(
                provider="arxiv",
                title="Abstract Meaning Representation graph parsing",
                source_for_gate="https://arxiv.org/pdf/2201.11111",
                abstract="Semantic graph parsing for natural language.",
                relevance_score=35,
            )
            biomedical_unlabeled = Candidate(
                provider="arxiv",
                title="MALDI-TOF antibiotic resistance detection",
                source_for_gate="https://arxiv.org/pdf/2401.22222",
                abstract="Antibiotic resistance detection with mass spectrometry.",
                relevance_score=20,
            )
            nlp_unlabeled = Candidate(
                provider="arxiv",
                title="AMR semantic graph parser",
                source_for_gate="https://arxiv.org/pdf/2401.33333",
                abstract="Natural language semantic graph parsing.",
                relevance_score=30,
            )
            for candidate in [relevant_seed, irrelevant_seed, biomedical_unlabeled, nlp_unlabeled]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, relevant_seed.source_for_gate, "relevant")
            store.set_screening_label(batch.batch_id, irrelevant_seed.source_for_gate, "irrelevant")

            recommendations = recommend_unlabeled_items(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=2,
            )

            self.assertEqual(recommendations[0].item.source, biomedical_unlabeled.source_for_gate)
            self.assertIn("resistance", recommendations[0].relevant_overlap)
            self.assertEqual(recommendations[1].item.source, nlp_unlabeled.source_for_gate)

    def test_recommendations_weight_human_feedback_more_than_agent_feedback(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="feedback weighting", limit=4, mode="query")
            human_seed = Candidate(
                provider="openalex",
                title="Glycopeptide assay",
                source_for_gate="10.1000/human-seed",
                abstract="Glycopeptide evidence marked by a human reviewer.",
                relevance_score=10,
            )
            agent_seed = Candidate(
                provider="openalex",
                title="Riboswitch analysis",
                source_for_gate="10.1000/agent-seed",
                abstract="Riboswitch evidence marked by an agent.",
                relevance_score=10,
            )
            human_match = Candidate(
                provider="openalex",
                title="Glycopeptide validation",
                source_for_gate="10.1000/human-match",
                abstract="Follow-up glycopeptide evidence.",
                relevance_score=5,
            )
            agent_match = Candidate(
                provider="openalex",
                title="Riboswitch validation",
                source_for_gate="10.1000/agent-match",
                abstract="Follow-up riboswitch evidence.",
                relevance_score=25,
            )
            for candidate in [human_seed, agent_seed, human_match, agent_match]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, human_seed.source_for_gate, "relevant")
            store.set_screening_label(
                batch.batch_id,
                agent_seed.source_for_gate,
                "relevant",
                label_source="agent",
                confidence=0.88,
            )

            recommendations = recommend_unlabeled_items(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=2,
            )

            self.assertEqual(recommendations[0].item.source, human_match.source_for_gate)
            self.assertIn("glycopeptide", recommendations[0].relevant_overlap)

    def test_deep_read_order_prefers_human_maybe_over_agent_maybe(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=2, mode="query")
            human_maybe = Candidate(
                provider="pubmed",
                title="Human maybe MALDI paper",
                source_for_gate="10.1000/human-maybe",
                abstract="Human reviewer kept this as maybe.",
                relevance_score=12,
            )
            agent_maybe = Candidate(
                provider="pubmed",
                title="Agent maybe MALDI paper",
                source_for_gate="10.1000/agent-maybe",
                abstract="Agent kept this as maybe.",
                relevance_score=91,
            )
            for candidate in [human_maybe, agent_maybe]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, human_maybe.source_for_gate, "maybe")
            store.set_screening_label(
                batch.batch_id,
                agent_maybe.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.61,
            )

            ranked = rank_deep_read_items(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                min_relevance=0,
            )

            self.assertEqual(ranked[0].source, human_maybe.source_for_gate)

    def test_auto_label_batch_items_classifies_metadata_without_overwriting_human_labels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="language math", limit=3, mode="query")
            relevant = Candidate(
                provider="openalex",
                title="Mathematical structures in formal language theory",
                source_for_gate="10.1038/language-math",
                abstract="Language syntax can be modeled with algebra and formal grammars.",
                relevance_score=18,
            )
            maybe = Candidate(
                provider="openalex",
                title="Language and cognition",
                source_for_gate="10.1038/language-cognition",
                abstract="This study discusses language learning and symbolic reasoning.",
                relevance_score=18,
            )
            human_labeled = Candidate(
                provider="arxiv",
                title="Mathematical language models",
                source_for_gate="https://arxiv.org/pdf/2401.11111",
                abstract="Mathematics and language modeling.",
                relevance_score=60,
            )
            for candidate in [relevant, maybe, human_labeled]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, human_labeled.source_for_gate, "irrelevant")

            decisions = auto_label_batch_items(
                store,
                batch.batch_id,
                query="language math",
                limit=10,
                apply=True,
            )
            labels = {label.source: label for label in store.list_screening_labels(batch.batch_id)}

            self.assertEqual(decisions.applied_count, 2)
            self.assertEqual(decisions.skipped_human_count, 1)
            self.assertEqual(labels[relevant.source_for_gate].label, "relevant")
            self.assertEqual(labels[relevant.source_for_gate].label_source, "agent")
            self.assertGreaterEqual(labels[relevant.source_for_gate].confidence, 0.65)
            self.assertIn("query_overlap", labels[relevant.source_for_gate].signals)
            self.assertEqual(labels[maybe.source_for_gate].label, "maybe")
            self.assertEqual(labels[human_labeled.source_for_gate].label, "irrelevant")
            self.assertEqual(labels[human_labeled.source_for_gate].label_source, "human")

    def test_auto_label_uses_topic_profile_for_stochastic_calculus(self):
        relevant = score_candidate(
            "tell me about stochastic calculus",
            Candidate(
                provider="openalex",
                title="Stochastic calculus with anticipating integrands",
                source_for_gate="10.1007/bf00353876",
                doi="10.1007/bf00353876",
                abstract="Stochastic integration for Brownian motion, martingales, and semimartingales.",
            ),
        )
        generic_collision = score_candidate(
            "tell me about stochastic calculus",
            Candidate(
                provider="arxiv",
                title="On a Non-Newtonian Calculus of Variations",
                source_for_gate="https://arxiv.org/pdf/2107.14152v1",
                abstract="Non-Newtonian calculus of variations and Euler-Lagrange equations.",
            ),
        )

        self.assertEqual(self._label_for("tell me about stochastic calculus", relevant), "relevant")
        self.assertEqual(self._label_for("tell me about stochastic calculus", generic_collision), "irrelevant")

    def test_auto_label_uses_scholarly_query_terms_not_conversational_overlap(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="what is the importance of math in language", limit=4, mode="query")
            relevant = Candidate(
                provider="openalex",
                title="Mathematical linguistics and formal language theory",
                source_for_gate="10.1000/formal-language",
                abstract="Syntax, grammars, algebra, and automata provide mathematical models of natural language.",
                relevance_score=22,
                query_variant="mathematical linguistics",
            )
            weak_language_match = Candidate(
                provider="openalex",
                title="Large language models encode clinical knowledge",
                source_for_gate="10.1038/s41586-023-06291-2",
                abstract="Clinical question answering benchmarks evaluate factuality and possible harm in medicine.",
                relevance_score=25,
                query_variant="what is the importance of math in language",
            )
            generic_llm_education = Candidate(
                provider="openalex",
                title="ChatGPT and large language models for mathematics education",
                source_for_gate="10.1000/chatgpt-education",
                abstract="Students use language models for mathematics homework and classroom instruction.",
                relevance_score=17,
                query_variant="mathematical models of language acquisition",
            )
            maybe = Candidate(
                provider="arxiv",
                title="Information theory and language processing",
                source_for_gate="https://arxiv.org/pdf/2401.12345",
                abstract="Entropy and coding theory are used to analyze linguistic communication.",
                relevance_score=15,
                query_variant="information theory language",
            )
            for candidate in [relevant, weak_language_match, generic_llm_education, maybe]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )

            result = auto_label_batch_items(store, batch.batch_id, apply=True)
            labels = {label.source: label for label in store.list_screening_labels(batch.batch_id)}

            self.assertEqual(result.applied_count, 4)
            self.assertEqual(labels[relevant.source_for_gate].label, "relevant")
            self.assertEqual(labels[weak_language_match.source_for_gate].label, "irrelevant")
            self.assertIn("math_overlap=-", labels[weak_language_match.source_for_gate].signals)
            self.assertEqual(labels[generic_llm_education.source_for_gate].label, "maybe")
            self.assertEqual(labels[maybe.source_for_gate].label, "maybe")

    def test_real_smoke_tuning_handles_maldi_amr_noise(self):
        prescribing_ml = Candidate(
            provider="arxiv",
            title="Battling Antibiotic Resistance: Can Machine Learning Improve Prescribing?",
            source_for_gate="https://arxiv.org/pdf/1906.03044v1",
            abstract="Machine learning predicts urinary tract infection test outcomes and policies to improve antibiotic prescribing.",
            relevance_score=63,
        )
        resistance_proteins = Candidate(
            provider="arxiv",
            title="Small antimicrobial resistance proteins (SARPs): Small proteins conferring antimicrobial resistance",
            source_for_gate="https://arxiv.org/pdf/2310.17905v1",
            abstract="Small proteins can confer resistance to antimicrobial compounds and antibiotics.",
            relevance_score=57,
        )
        maldi_chemistry = Candidate(
            provider="arxiv",
            title="MALDI-TOF and Quantum Chemical Study of Non-stoichiometric Tantalum Oxychloride Clusters",
            source_for_gate="https://arxiv.org/pdf/1912.09801v1",
            abstract="MALDI-TOF spectroscopy and quantum chemical calculations study tantalum oxychloride clusters.",
            relevance_score=57,
        )

        self.assertEqual(self._label_for("MALDI AMR", prescribing_ml), "irrelevant")
        self.assertEqual(self._label_for("MALDI AMR", resistance_proteins), "relevant")
        self.assertEqual(self._label_for("MALDI AMR", maldi_chemistry), "irrelevant")

    def test_real_smoke_tuning_handles_esbl_cre_surveillance(self):
        clinical_surveillance = Candidate(
            provider="pubmed",
            title="Prevalence and Resistance Patterns of Uropathogens in Critically Ill Patients",
            source_for_gate="10.1177/10766294251384089",
            abstract="Prevalence and resistance patterns of uropathogens support antimicrobial stewardship and clinical surveillance.",
            relevance_score=57,
        )
        basic_beta_lactamase = Candidate(
            provider="arxiv",
            title="Coevolutionary landscape inference and the context-dependence of mutations in beta-lactamase TEM-1",
            source_for_gate="https://arxiv.org/pdf/1510.03224v1",
            abstract="Statistical analysis estimates mutational landscapes of beta-lactamase TEM-1 and antibiotic resistance effects.",
            relevance_score=39,
        )

        self.assertEqual(self._label_for("ESBL CRE surveillance", clinical_surveillance), "relevant")
        self.assertEqual(self._label_for("ESBL CRE surveillance", basic_beta_lactamase), "irrelevant")

    def test_real_smoke_tuning_promotes_math_language_methods(self):
        question_generation = Candidate(
            provider="arxiv",
            title="An Automated Multiple-Choice Question Generation Using Natural Language Processing Techniques",
            source_for_gate="https://arxiv.org/pdf/2103.14757v1",
            abstract="Natural language processing techniques generate questions from text using computational linguistic methods.",
            relevance_score=19,
        )
        statistics_for_linguistics = Candidate(
            provider="arxiv",
            title="Statistical methods for linguistic research: Foundational Ideas",
            source_for_gate="https://arxiv.org/pdf/1602.00245v1",
            abstract="Bayesian data analysis and statistical methods for linguistics and psycholinguistics.",
            relevance_score=9,
        )
        broad_language_processing = Candidate(
            provider="openalex",
            title="FREQUENCY EFFECTS IN LANGUAGE PROCESSING",
            source_for_gate="10.1017/s0272263102002024",
            abstract="Frequency effects in syntax, language comprehension, language acquisition, and usage-based linguistics.",
            concepts="Linguistics; Language acquisition; Syntax; Grammar; Natural language processing",
            relevance_score=71,
        )

        self.assertEqual(self._label_for("importance of math in language", question_generation), "relevant")
        self.assertEqual(self._label_for("importance of math in language", statistics_for_linguistics), "relevant")
        self.assertEqual(self._label_for("importance of math in language", broad_language_processing), "maybe")

    def test_real_smoke_tuning_handles_clinical_stewardship_noise(self):
        stewardship = Candidate(
            provider="arxiv",
            title="Benchmarking Machine Learning Architectures for Antimicrobial Stewardship in Pediatric ICUs",
            source_for_gate="https://arxiv.org/pdf/2605.22611v1",
            abstract="Antimicrobial stewardship in pediatric intensive care units, antibiotic de-escalation, and clinical decision support.",
            relevance_score=61,
        )
        simulator = Candidate(
            provider="arxiv",
            title="abx_amr_simulator: A simulation environment for antibiotic prescribing policy optimization",
            source_for_gate="https://arxiv.org/pdf/2603.11369v1",
            abstract="Python simulation package, reinforcement learning environment, Gymnasium API, and antibiotic resistance dynamics.",
            relevance_score=61,
        )
        biomarker = Candidate(
            provider="pubmed",
            title="Biomarkers in sepsis: Less is more? Yes, it is.",
            source_for_gate="10.1016/j.iccn.2025.104243",
            abstract="Biomarkers in sepsis and critical care nursing.",
            relevance_score=23,
        )
        generic_prediction = Candidate(
            provider="arxiv",
            title="Predicting sepsis in multi-site, multi-national intensive care cohorts using deep learning",
            source_for_gate="https://arxiv.org/pdf/2107.05230v1",
            abstract="A deep self-attention model predicts sepsis in ICU cohorts using harmonized clinical databases.",
            relevance_score=53,
        )

        query = "sepsis procalcitonin antibiotic stewardship"
        self.assertEqual(self._label_for(query, stewardship), "relevant")
        self.assertEqual(self._label_for(query, simulator), "irrelevant")
        self.assertEqual(self._label_for(query, biomarker), "relevant")
        self.assertEqual(self._label_for(query, generic_prediction), "irrelevant")

    def test_auto_label_batch_items_uses_llm_provider_without_overwriting_human_labels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FakeLlmClient:
            def __init__(self):
                self.calls = []

            def label(self, *, query, item, model):
                self.calls.append((query, item.source, model))
                return LlmLabelResult(
                    label="relevant",
                    confidence=0.88,
                    rationale="LLM judged title and abstract as relevant.",
                    evidence_terms=("mathematical linguistics", "formal grammar"),
                    exclusion_reason=None,
                )

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="math in language", limit=2, mode="query")
            llm_candidate = Candidate(
                provider="openalex",
                title="Mathematical Linguistics",
                source_for_gate="10.1000/llm-relevant",
                abstract="Formal grammar and mathematical models of language.",
                relevance_score=5,
            )
            human_candidate = Candidate(
                provider="arxiv",
                title="Human override",
                source_for_gate="https://arxiv.org/pdf/2401.22222",
                abstract="Formal language theory.",
                relevance_score=5,
            )
            for candidate in [llm_candidate, human_candidate]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, human_candidate.source_for_gate, "irrelevant")
            fake_client = FakeLlmClient()

            result = auto_label_batch_items(
                store,
                batch.batch_id,
                provider="llm",
                model="gpt-test",
                llm_client=fake_client,
                apply=True,
            )
            labels = {label.source: label for label in store.list_screening_labels(batch.batch_id)}

            self.assertEqual(result.applied_count, 1)
            self.assertEqual(result.skipped_human_count, 1)
            self.assertEqual(result.skipped_error_count, 0)
            self.assertEqual(fake_client.calls, [("math in language", "10.1000/llm-relevant", "gpt-test")])
            self.assertEqual(labels[llm_candidate.source_for_gate].label, "relevant")
            self.assertEqual(labels[llm_candidate.source_for_gate].label_source, "agent")
            self.assertEqual(labels[llm_candidate.source_for_gate].confidence, 0.88)
            self.assertIn("label_provider=llm", labels[llm_candidate.source_for_gate].signals)
            self.assertIn("model=gpt-test", labels[llm_candidate.source_for_gate].signals)
            self.assertIn("evidence_terms=mathematical linguistics,formal grammar", labels[llm_candidate.source_for_gate].signals)
            self.assertEqual(labels[human_candidate.source_for_gate].label_source, "human")

    def test_auto_label_batch_items_skips_llm_client_errors(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FailingLlmClient:
            def label(self, *, query, item, model):
                raise RuntimeError("network unavailable")

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="math in language", limit=1, mode="query")
            candidate = Candidate(
                provider="openalex",
                title="Mathematical Linguistics",
                source_for_gate="10.1000/llm-error",
                abstract="Formal grammar and mathematical models of language.",
                relevance_score=5,
            )
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            result = auto_label_batch_items(
                store,
                batch.batch_id,
                provider="llm",
                model="gpt-test",
                llm_client=FailingLlmClient(),
                apply=True,
            )

            self.assertEqual(result.decisions, [])
            self.assertEqual(result.applied_count, 0)
            self.assertEqual(result.skipped_error_count, 1)
            self.assertEqual(store.list_screening_labels(batch.batch_id), [])

    def test_llm_review_queue_prioritizes_borderline_conflicts_and_diversity(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="MALDI AMR", limit=7, mode="query")
            blocked = Candidate(
                provider="openalex",
                title="Blocked repository paper",
                source_for_gate="https://github.com/example/repo/blob/main/paper.pdf",
                relevance_score=99,
            )
            human = Candidate(
                provider="pubmed",
                title="Human reviewed MALDI resistance",
                source_for_gate="10.1000/human",
                abstract="MALDI antimicrobial resistance.",
                relevance_score=80,
            )
            maybe_high = Candidate(
                provider="pubmed",
                title="MALDI antimicrobial resistance maybe",
                source_for_gate="10.1000/maybe-high",
                abstract="MALDI antimicrobial resistance prediction.",
                relevance_score=78,
            )
            irrelevant_conflict = Candidate(
                provider="openalex",
                title="MALDI antimicrobial resistance conflict",
                source_for_gate="10.1000/conflict",
                abstract="MALDI antimicrobial resistance appears in metadata.",
                relevance_score=74,
            )
            unlabeled = Candidate(
                provider="arxiv",
                title="MALDI-TOF resistance unlabeled",
                source_for_gate="https://arxiv.org/pdf/2401.11111",
                abstract="Antibiotic resistance with MALDI spectra.",
                relevance_score=62,
            )
            duplicate_provider = Candidate(
                provider="pubmed",
                title="Another PubMed maybe",
                source_for_gate="10.1000/maybe-duplicate",
                abstract="MALDI resistance duplicate provider.",
                relevance_score=76,
            )
            for candidate in [blocked, human, maybe_high, irrelevant_conflict, unlabeled, duplicate_provider]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            store.set_screening_label(batch.batch_id, human.source_for_gate, "relevant")
            store.set_screening_label(
                batch.batch_id,
                maybe_high.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.61,
                rationale="metadata partially matches query",
                signals="provider=heuristic",
            )
            store.set_screening_label(
                batch.batch_id,
                irrelevant_conflict.source_for_gate,
                "irrelevant",
                label_source="agent",
                confidence=0.89,
                rationale="metadata has little query overlap",
                signals="provider=heuristic",
            )
            store.set_screening_label(
                batch.batch_id,
                duplicate_provider.source_for_gate,
                "maybe",
                label_source="agent",
                confidence=0.64,
                rationale="metadata partially matches query",
                signals="provider=heuristic",
            )

            queue = build_llm_review_queue(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=3,
            )

            self.assertEqual(
                [entry.item.source for entry in queue],
                [maybe_high.source_for_gate, irrelevant_conflict.source_for_gate, unlabeled.source_for_gate],
            )
            self.assertIn("heuristic_maybe_high_relevance", queue[0].reason)
            self.assertIn("heuristic_irrelevant_high_relevance", queue[1].reason)
            self.assertIn("unlabeled_high_relevance", queue[2].reason)
            self.assertGreater(queue[0].score, queue[2].score)
            self.assertNotIn(blocked.source_for_gate, [entry.item.source for entry in queue])
            self.assertNotIn(human.source_for_gate, [entry.item.source for entry in queue])

    def test_llm_auto_label_uses_review_queue_order(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        class FakeLlmClient:
            def __init__(self):
                self.calls = []

            def label(self, *, query, item, model):
                self.calls.append(item.source)
                return LlmLabelResult(
                    label="maybe",
                    confidence=0.77,
                    rationale="LLM reviewed queued item.",
                    evidence_terms=("queued",),
                    exclusion_reason=None,
                )

        with TemporaryDirectory() as tmp:
            store = FridayStore(Path(tmp) / "friday.db")
            batch = store.create_batch(query="language math", limit=3, mode="query")
            first = Candidate(
                provider="openalex",
                title="Mathematical structures in formal language theory",
                source_for_gate="10.1000/first",
                abstract="Formal grammars and algebra.",
                relevance_score=60,
            )
            queued = Candidate(
                provider="openalex",
                title="Language and cognition",
                source_for_gate="10.1000/queued",
                abstract="Language learning and symbolic reasoning.",
                relevance_score=70,
            )
            third = Candidate(
                provider="arxiv",
                title="Information theory and language",
                source_for_gate="https://arxiv.org/pdf/2401.22222",
                abstract="Entropy and language processing.",
                relevance_score=20,
            )
            for candidate in [first, queued, third]:
                store.add_batch_item(
                    batch.batch_id,
                    candidate.source_for_gate,
                    evaluate_source(candidate.source_for_gate),
                    candidate=candidate,
                )
            auto_label_batch_items(store, batch.batch_id, query="language math", apply=True)
            queue = build_llm_review_queue(
                store.list_batch_items(batch.batch_id),
                store.list_screening_labels(batch.batch_id),
                limit=1,
            )
            fake_client = FakeLlmClient()

            result = auto_label_batch_items(
                store,
                batch.batch_id,
                query="language math",
                provider="llm",
                model="gpt-test",
                llm_client=fake_client,
                review_queue=queue,
                apply=True,
            )
            labels = {label.source: label for label in store.list_screening_labels(batch.batch_id)}

            self.assertEqual(fake_client.calls, [queued.source_for_gate])
            self.assertEqual(result.applied_count, 1)
            self.assertIn("review_queue_score=", labels[queued.source_for_gate].signals)
            self.assertIn("review_queue_reason=heuristic_maybe_high_relevance", labels[queued.source_for_gate].signals)


if __name__ == "__main__":
    unittest.main()
