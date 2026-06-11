import unittest

from jarvis_research.acronyms import detect_acronyms, resolve_acronyms


class AcronymResolverTests(unittest.TestCase):
    def test_detects_uppercase_acronym_tokens_without_lowercase_words(self):
        self.assertEqual(detect_acronyms("PCR CNN xyz qPCR"), ("PCR", "CNN", "qPCR"))

    def test_resolves_unambiguous_biomedical_acronym(self):
        resolved = resolve_acronyms("PCR assay diagnostic sensitivity")

        self.assertEqual(resolved[0].acronym, "PCR")
        self.assertEqual(resolved[0].meaning, "polymerase chain reaction")
        self.assertEqual(resolved[0].intent, "biomedical")
        self.assertEqual(resolved[0].reason, "registry_single_sense")

    def test_resolves_unambiguous_ml_acronym(self):
        resolved = resolve_acronyms("CNN image classification")

        self.assertEqual(resolved[0].acronym, "CNN")
        self.assertEqual(resolved[0].meaning, "convolutional neural network")
        self.assertEqual(resolved[0].intent, "ml")

    def test_preserves_unknown_acronym_as_unresolved(self):
        resolved = resolve_acronyms("XYZ biomarker discovery")

        self.assertEqual(resolved[0].acronym, "XYZ")
        self.assertEqual(resolved[0].meaning, "XYZ")
        self.assertEqual(resolved[0].intent, "unknown")
        self.assertEqual(resolved[0].reason, "unresolved_acronym")

    def test_resolves_ambiguous_amr_from_context(self):
        parsing = resolve_acronyms("AMR parsing")
        maldi = resolve_acronyms("MALDI AMR")

        self.assertEqual(parsing[0].meaning, "abstract meaning representation")
        self.assertEqual(parsing[0].intent, "nlp")
        self.assertIn("antimicrobial resistance", parsing[0].rejected_meanings)
        self.assertEqual(maldi[0].meaning, "antimicrobial resistance")
        self.assertEqual(maldi[0].intent, "biomedical")
        self.assertIn("abstract meaning representation", maldi[0].rejected_meanings)


if __name__ == "__main__":
    unittest.main()
