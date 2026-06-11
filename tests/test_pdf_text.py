import unittest

from friday.pdf_text import clean_pdf_pages


class PdfTextCleanupTests(unittest.TestCase):
    def test_removes_repeated_headers_page_numbers_and_joins_hyphenation(self):
        pages = [
            (
                "MALDI-TOF for microbial identification and diagnosis\n"
                "1\n"
                "Methods\n"
                "We used MALDI-\n"
                "TOF spectra to train a classifier.\n"
                "Frontiers in Microbiology | www.frontiersin.org"
            ),
            (
                "MALDI-TOF for microbial identification and diagnosis\n"
                "2\n"
                "Results\n"
                "The model achieved an AUROC of 0.91.\n"
                "Frontiers in Microbiology | www.frontiersin.org"
            ),
        ]

        cleaned = clean_pdf_pages(pages)

        self.assertNotIn("MALDI-TOF for microbial identification and diagnosis", cleaned[0])
        self.assertNotIn("Frontiers in Microbiology", cleaned[1])
        self.assertNotIn("\n1\n", f"\n{cleaned[0]}\n")
        self.assertIn("MALDI-TOF spectra", cleaned[0])
        self.assertIn("Results\nThe model achieved an AUROC of 0.91.", cleaned[1])


if __name__ == "__main__":
    unittest.main()
