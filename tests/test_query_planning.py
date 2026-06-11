import unittest

from friday.query_planning import plan_query


class QueryPlanningTests(unittest.TestCase):
    def test_resolves_maldi_amr_to_antimicrobial_resistance(self):
        plan = plan_query("MALDI AMR")

        self.assertEqual(plan.intent, "biomedical")
        self.assertEqual(plan.expanded_queries[0], "MALDI antimicrobial resistance")
        self.assertIn("MALDI-TOF antibiotic resistance", plan.expanded_queries)
        self.assertIn("MALDI-TOF antimicrobial susceptibility", plan.expanded_queries)
        self.assertNotIn("MALDI AMR", plan.expanded_queries)
        self.assertEqual(plan.resolved_acronyms[0].acronym, "AMR")
        self.assertEqual(plan.resolved_acronyms[0].meaning, "antimicrobial resistance")
        self.assertIn("abstract meaning representation", plan.resolved_acronyms[0].rejected_meanings)
        self.assertIn("adaptive mesh refinement", plan.resolved_acronyms[0].rejected_meanings)

    def test_resolves_amr_parsing_to_abstract_meaning_representation(self):
        plan = plan_query("AMR parsing")

        self.assertEqual(plan.intent, "nlp")
        self.assertEqual(plan.resolved_acronyms[0].meaning, "abstract meaning representation")
        self.assertIn("abstract meaning representation parsing", plan.expanded_queries)

    def test_resolves_amr_mesh_to_adaptive_mesh_refinement(self):
        plan = plan_query("AMR mesh refinement")

        self.assertEqual(plan.intent, "computational")
        self.assertEqual(plan.resolved_acronyms[0].meaning, "adaptive mesh refinement")
        self.assertIn("adaptive mesh refinement", plan.expanded_queries[0])

    def test_expands_biomedical_mdr_and_mic_acronyms(self):
        plan = plan_query("MALDI MDR MIC")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(meanings["MDR"], "multidrug resistance")
        self.assertEqual(meanings["MIC"], "minimum inhibitory concentration")
        self.assertIn("MALDI multidrug resistance minimum inhibitory concentration", plan.expanded_queries)

    def test_expands_biomedical_ast_acronym_with_maldi_context(self):
        plan = plan_query("MALDI AST")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(plan.intent, "biomedical")
        self.assertEqual(meanings["AST"], "antimicrobial susceptibility testing")
        self.assertIn("MALDI antimicrobial susceptibility testing", plan.expanded_queries)

    def test_expands_esbl_and_cre_as_biomedical_acronyms(self):
        plan = plan_query("ESBL CRE surveillance")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(plan.intent, "biomedical")
        self.assertEqual(meanings["ESBL"], "extended-spectrum beta-lactamase")
        self.assertEqual(meanings["CRE"], "carbapenem-resistant Enterobacteriaceae")
        self.assertIn(
            "extended-spectrum beta-lactamase carbapenem-resistant Enterobacteriaceae surveillance",
            plan.expanded_queries,
        )

    def test_expands_pcr_as_biomedical_acronym(self):
        plan = plan_query("PCR assay diagnostic sensitivity")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(plan.intent, "biomedical")
        self.assertEqual(meanings["PCR"], "polymerase chain reaction")
        self.assertIn("polymerase chain reaction assay diagnostic sensitivity", plan.expanded_queries)

    def test_expands_cnn_as_ml_acronym(self):
        plan = plan_query("CNN image classification")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(plan.intent, "ml")
        self.assertEqual(meanings["CNN"], "convolutional neural network")
        self.assertIn("convolutional neural network image classification", plan.expanded_queries)

    def test_expands_svm_as_ml_acronym(self):
        plan = plan_query("SVM classifier feature selection")

        meanings = {resolved.acronym: resolved.meaning for resolved in plan.resolved_acronyms}

        self.assertEqual(plan.intent, "ml")
        self.assertEqual(meanings["SVM"], "support vector machine")
        self.assertIn("support vector machine classifier feature selection", plan.expanded_queries)

    def test_preserves_unknown_acronym_without_guessing(self):
        plan = plan_query("XYZ biomarker discovery")

        unresolved = [resolved for resolved in plan.resolved_acronyms if resolved.reason == "unresolved_acronym"]

        self.assertEqual(plan.intent, "unknown")
        self.assertEqual(plan.expanded_queries, ("XYZ biomarker discovery",))
        self.assertEqual([resolved.acronym for resolved in unresolved], ["XYZ"])
        self.assertEqual(unresolved[0].meaning, "XYZ")

    def test_keeps_unambiguous_query_when_no_acronym_is_resolved(self):
        plan = plan_query("Pseudomonas MALDI spectra")

        self.assertEqual(plan.intent, "unknown")
        self.assertEqual(plan.expanded_queries, ("Pseudomonas MALDI spectra",))
        self.assertEqual(plan.resolved_acronyms, ())

    def test_rewrites_casual_math_language_prompt_to_scholarly_queries(self):
        plan = plan_query("what is the importance of math in language")

        self.assertEqual(plan.intent, "mathematical_linguistics")
        self.assertEqual(plan.expanded_queries[0], "mathematical linguistics")
        self.assertIn("formal language theory natural language", plan.expanded_queries)
        self.assertIn("information theory language", plan.expanded_queries)
        self.assertIn("mathematical models of language acquisition", plan.expanded_queries)
        self.assertNotIn("what is the importance of math in language", plan.expanded_queries)

    def test_rewrites_language_computation_prompt_to_scholarly_queries(self):
        plan = plan_query("friday tell me how language is computation")

        self.assertEqual(plan.intent, "mathematical_linguistics")
        self.assertIn("formal language theory natural language", plan.expanded_queries)


if __name__ == "__main__":
    unittest.main()
