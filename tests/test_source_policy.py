import unittest

from jarvis_research.source_policy import evaluate_source


class SourcePolicyTests(unittest.TestCase):
    def test_allows_arxiv_pdf_url(self):
        decision = evaluate_source("https://arxiv.org/pdf/2401.12345")
        self.assertIs(decision.allowed, True)
        self.assertEqual(decision.kind, "url")
        self.assertEqual(decision.normalized, "https://arxiv.org/pdf/2401.12345")

    def test_allows_frontiers_pdf_url(self):
        decision = evaluate_source("https://www.frontiersin.org/articles/10.3389/fmicb.2015.00791/pdf")
        self.assertIs(decision.allowed, True)
        self.assertEqual(decision.kind, "url")
        self.assertEqual(
            decision.normalized,
            "https://www.frontiersin.org/articles/10.3389/fmicb.2015.00791/pdf",
        )

    def test_allows_clinical_microbiology_infection_pdf_url(self):
        decision = evaluate_source("https://www.clinicalmicrobiologyandinfection.com/article/S1198743X20301580/pdf")
        self.assertIs(decision.allowed, True)
        self.assertEqual(decision.kind, "url")
        self.assertEqual(
            decision.normalized,
            "https://www.clinicalmicrobiologyandinfection.com/article/S1198743X20301580/pdf",
        )

    def test_allows_common_open_access_publisher_pdf_urls(self):
        mdpi = evaluate_source("https://www.mdpi.com/2079-6382/10/1/123/pdf")
        sage = evaluate_source("https://journals.sagepub.com/doi/pdf/10.1177/10766294251384089")

        self.assertIs(mdpi.allowed, True)
        self.assertEqual(mdpi.kind, "url")
        self.assertIs(sage.allowed, True)
        self.assertEqual(sage.kind, "url")

    def test_allows_ncbi_ftp_pmc_pdf_url(self):
        decision = evaluate_source(
            "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/d8/72/IDR-19-614240.PMC13242821.pdf"
        )
        self.assertIs(decision.allowed, True)
        self.assertEqual(decision.kind, "url")
        self.assertEqual(
            decision.normalized,
            "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/d8/72/IDR-19-614240.PMC13242821.pdf",
        )

    def test_allows_bare_doi(self):
        decision = evaluate_source("10.1038/s41586-020-2649-2")
        self.assertIs(decision.allowed, True)
        self.assertEqual(decision.kind, "doi")
        self.assertEqual(decision.normalized, "10.1038/s41586-020-2649-2")

    def test_blocks_github_even_when_file_looks_scholarly(self):
        decision = evaluate_source("https://github.com/example/repo/blob/main/paper.pdf")
        self.assertIs(decision.allowed, False)
        self.assertEqual(decision.reason, "blocked_domain")

    def test_blocks_archives_and_code_artifacts(self):
        decision = evaluate_source("https://arxiv.org/e-print/2401.12345")
        self.assertIs(decision.allowed, False)
        self.assertEqual(decision.reason, "blocked_extension_or_artifact")


if __name__ == "__main__":
    unittest.main()
