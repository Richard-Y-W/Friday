import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday.discovery import Candidate
from friday.pdf_ingestion import DownloadedPdf, deep_read_source, resolve_candidate_pdf_url
from friday.source_policy import evaluate_source
from friday.storage import FridayStore


class PdfIngestionTests(unittest.TestCase):
    def test_resolves_arxiv_candidate_to_pdf_url(self):
        candidate = Candidate(
            provider="arxiv",
            title="Safe arXiv paper",
            source_for_gate="https://arxiv.org/pdf/2401.12345v1",
            arxiv_id="2401.12345v1",
            url="https://arxiv.org/abs/2401.12345v1",
        )

        resolution = resolve_candidate_pdf_url(candidate)

        self.assertEqual(resolution.pdf_url, "https://arxiv.org/pdf/2401.12345v1")
        self.assertEqual(resolution.reason, "resolved_arxiv_pdf")

    def test_does_not_resolve_doi_only_candidate_without_safe_pdf_url(self):
        candidate = Candidate(
            provider="pubmed",
            title="DOI-only paper",
            source_for_gate="10.1128/jcm.00123-23",
            doi="10.1128/jcm.00123-23",
            pmid="12345678",
        )

        resolution = resolve_candidate_pdf_url(candidate)

        self.assertIsNone(resolution.pdf_url)
        self.assertEqual(resolution.reason, "no_safe_pdf_url")

    def test_resolves_openalex_open_access_pdf_url_before_doi(self):
        candidate = Candidate(
            provider="openalex",
            title="Open access DOI paper",
            source_for_gate="10.1038/s41591-021-01619-9",
            doi="10.1038/s41591-021-01619-9",
            url="https://www.nature.com/articles/s41591-021-01619-9",
            open_access_pdf_url="https://www.nature.com/articles/s41591-021-01619-9.pdf",
        )

        resolution = resolve_candidate_pdf_url(candidate)

        self.assertEqual(
            resolution.pdf_url,
            "https://www.nature.com/articles/s41591-021-01619-9.pdf",
        )
        self.assertEqual(resolution.reason, "resolved_open_access_pdf")

    def test_explicit_safe_pdf_does_not_fetch_landing_page(self):
        candidate = Candidate(
            provider="openalex",
            title="Open access DOI paper",
            source_for_gate="10.1038/s41591-021-01619-9",
            doi="10.1038/s41591-021-01619-9",
            url="https://www.nature.com/articles/s41591-021-01619-9",
            open_access_pdf_url="https://www.nature.com/articles/s41591-021-01619-9.pdf",
        )
        fetched_urls = []

        def fake_text_fetcher(url):
            fetched_urls.append(url)
            return "<html></html>"

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(fetched_urls, [])
        self.assertEqual(
            resolution.pdf_url,
            "https://www.nature.com/articles/s41591-021-01619-9.pdf",
        )
        self.assertEqual(resolution.reason, "resolved_open_access_pdf")

    def test_resolves_publisher_landing_page_citation_pdf_url(self):
        candidate = Candidate(
            provider="openalex",
            title="Publisher landing page paper",
            source_for_gate="10.1038/s41591-021-01619-9",
            doi="10.1038/s41591-021-01619-9",
            url="https://www.nature.com/articles/s41591-021-01619-9",
        )
        fetched_urls = []

        def fake_text_fetcher(url):
            fetched_urls.append(url)
            return (
                '<html><head><meta name="citation_pdf_url" '
                'content="https://www.nature.com/articles/s41591-021-01619-9.pdf">'
                "</head></html>"
            )

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(fetched_urls, ["https://www.nature.com/articles/s41591-021-01619-9"])
        self.assertEqual(
            resolution.pdf_url,
            "https://www.nature.com/articles/s41591-021-01619-9.pdf",
        )
        self.assertEqual(resolution.reason, "resolved_landing_page_pdf")

    def test_resolves_relative_publisher_pdf_link_from_landing_page(self):
        candidate = Candidate(
            provider="openalex",
            title="Frontiers landing page paper",
            source_for_gate="10.3389/fmicb.2015.00791",
            doi="10.3389/fmicb.2015.00791",
            url="https://www.frontiersin.org/articles/10.3389/fmicb.2015.00791/full",
        )
        fetched_urls = []

        def fake_text_fetcher(url):
            fetched_urls.append(url)
            return '<html><body><a href="pdf">Download PDF</a></body></html>'

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(
            fetched_urls,
            ["https://www.frontiersin.org/articles/10.3389/fmicb.2015.00791/full"],
        )
        self.assertEqual(
            resolution.pdf_url,
            "https://www.frontiersin.org/articles/10.3389/fmicb.2015.00791/pdf",
        )
        self.assertEqual(resolution.reason, "resolved_landing_page_pdf")

    def test_resolves_sciencedirect_pdfft_link_from_landing_page(self):
        candidate = Candidate(
            provider="openalex",
            title="Clinical Microbiology landing page paper",
            source_for_gate="10.1016/j.cmi.2020.03.020",
            doi="10.1016/j.cmi.2020.03.020",
            url="https://www.sciencedirect.com/science/article/pii/S1198743X20301580",
        )

        def fake_text_fetcher(url):
            return (
                '<html><body><a href="/science/article/pii/S1198743X20301580/pdfft'
                '?isDTMRedir=true&download=true">PDF</a></body></html>'
            )

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(
            resolution.pdf_url,
            "https://www.sciencedirect.com/science/article/pii/S1198743X20301580/pdfft?isDTMRedir=true&download=true",
        )
        self.assertEqual(resolution.reason, "resolved_landing_page_pdf")

    def test_ignores_unsafe_pdf_links_from_safe_landing_page(self):
        candidate = Candidate(
            provider="openalex",
            title="Safe page with unsafe artifact link",
            source_for_gate="10.1038/example",
            doi="10.1038/example",
            url="https://www.nature.com/articles/example",
        )

        def fake_text_fetcher(url):
            return '<html><body><a href="https://github.com/example/repo/raw/main/paper.pdf">PDF</a></body></html>'

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertIsNone(resolution.pdf_url)
        self.assertEqual(resolution.reason, "no_safe_pdf_url")

    def test_resolves_pubmed_pmcid_to_safe_pmc_pdf_url(self):
        candidate = Candidate(
            provider="pubmed",
            title="PMC full text paper",
            source_for_gate="10.1128/jcm.00123-23",
            doi="10.1128/jcm.00123-23",
            pmid="12345678",
            pmcid="PMC7654321",
        )

        resolution = resolve_candidate_pdf_url(candidate)

        self.assertEqual(
            resolution.pdf_url,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/pdf/",
        )
        self.assertEqual(resolution.reason, "resolved_pmc_pdf")

    def test_resolves_pubmed_pmcid_to_pmc_oa_pdf_link_when_fetcher_available(self):
        candidate = Candidate(
            provider="pubmed",
            title="PMC full text paper",
            source_for_gate="10.2147/IDR.S614240",
            doi="10.2147/IDR.S614240",
            pmid="12345678",
            pmcid="PMC13242821",
        )
        fetched_urls = []

        def fake_text_fetcher(url):
            fetched_urls.append(url)
            return """<OA><records><record id="PMC13242821">
                <link format="pdf" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/d8/72/IDR-19-614240.PMC13242821.pdf" />
            </record></records></OA>"""

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(fetched_urls, ["https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC13242821"])
        self.assertEqual(
            resolution.pdf_url,
            "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/d8/72/IDR-19-614240.PMC13242821.pdf",
        )
        self.assertEqual(resolution.reason, "resolved_pmc_pdf")

    def test_resolves_pubmed_pmcid_to_article_pdf_link_when_oa_has_no_pdf(self):
        candidate = Candidate(
            provider="pubmed",
            title="PMC full text paper",
            source_for_gate="10.2147/IDR.S614240",
            doi="10.2147/IDR.S614240",
            pmid="12345678",
            pmcid="PMC13242821",
        )
        fetched_urls = []

        def fake_text_fetcher(url):
            fetched_urls.append(url)
            if "oa.fcgi" in url:
                return "<OA><records></records></OA>"
            return '<html><body><a href="pdf/IDR-19-614240.pdf">PDF</a></body></html>'

        resolution = resolve_candidate_pdf_url(candidate, text_fetcher=fake_text_fetcher)

        self.assertEqual(
            fetched_urls,
            [
                "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC13242821",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC13242821/",
            ],
        )
        self.assertEqual(
            resolution.pdf_url,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC13242821/pdf/IDR-19-614240.pdf",
        )
        self.assertEqual(resolution.reason, "resolved_pmc_pdf")

    def test_deep_read_source_uses_landing_page_pdf_resolution(self):
        candidate = Candidate(
            provider="openalex",
            title="Publisher landing page paper",
            source_for_gate="10.1038/s41591-021-01619-9",
            doi="10.1038/s41591-021-01619-9",
            url="https://www.nature.com/articles/s41591-021-01619-9",
        )

        def fake_text_fetcher(url):
            return (
                '<html><head><meta name="citation_pdf_url" '
                'content="https://www.nature.com/articles/s41591-021-01619-9.pdf">'
                "</head></html>"
            )

        def fake_downloader(url):
            self.assertEqual(url, "https://www.nature.com/articles/s41591-021-01619-9.pdf")
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="application/pdf",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                candidate.source_for_gate,
                candidate=candidate,
                downloader=fake_downloader,
                extractor=lambda path: ["Evidence-bearing page text."],
                text_fetcher=fake_text_fetcher,
            )

            self.assertEqual(result.status, "stored")
            self.assertEqual(result.pdf_url, "https://www.nature.com/articles/s41591-021-01619-9.pdf")
            self.assertEqual(result.reason, "pdf_text_extracted")

    def test_deep_read_source_stores_pdf_artifact_and_page_text(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="application/pdf",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        def fake_extractor(path):
            self.assertTrue(path.exists())
            return ["First page text", "Second page text"]

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=fake_extractor,
            )

            loaded = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)
            pages = store.list_pdf_pages(artifacts[0].artifact_id)
            evidence = store.list_evidence_records(artifacts[0].artifact_id)

            self.assertEqual(result.status, "stored")
            self.assertEqual(result.page_count, 2)
            self.assertEqual(loaded.deep_read_count, 1)
            self.assertEqual(artifacts[0].status, "stored")
            self.assertEqual(artifacts[0].reason, "pdf_text_extracted")
            self.assertEqual(artifacts[0].byte_count, 28)
            self.assertEqual(pages[0].page_number, 1)
            self.assertEqual(pages[0].text, "First page text")
            self.assertEqual(pages[1].page_number, 2)
            self.assertEqual(evidence, [])

    def test_deep_read_source_extracts_structured_evidence_from_pages(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="application/pdf",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        def fake_extractor(path):
            return [
                (
                    "We used MALDI-TOF spectra to train a classifier. "
                    "The dataset included 120 clinical isolates. "
                    "The model achieved an AUROC of 0.91."
                )
            ]

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=fake_extractor,
            )

            artifacts = store.list_pdf_artifacts(batch.batch_id)
            evidence = store.list_evidence_records(artifacts[0].artifact_id)

            self.assertEqual(result.status, "stored")
            self.assertEqual([item.evidence_type for item in evidence], ["method", "dataset_population", "result"])
            self.assertEqual(evidence[0].page_number, 1)
            self.assertIn("MALDI-TOF spectra", evidence[0].text)

    def test_deep_read_source_stores_cleaned_pages_before_extracting_evidence(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="application/pdf",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        def fake_extractor(path):
            return [
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

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=fake_extractor,
            )

            artifacts = store.list_pdf_artifacts(batch.batch_id)
            pages = store.list_pdf_pages(artifacts[0].artifact_id)
            evidence = store.list_evidence_records(artifacts[0].artifact_id)

            self.assertEqual(result.status, "stored")
            self.assertNotIn("MALDI-TOF for microbial identification and diagnosis", pages[0].text)
            self.assertNotIn("Frontiers in Microbiology", pages[1].text)
            self.assertIn("MALDI-TOF spectra", pages[0].text)
            self.assertTrue(any("AUROC of 0.91" in item.text for item in evidence))

    def test_deep_read_source_blocks_redirect_to_non_scholarly_domain(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url="https://github.com/example/repo/blob/main/paper.pdf",
                content_type="application/pdf",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=lambda path: ["should not run"],
            )

            loaded = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.reason, "final_url_blocked_domain")
            self.assertEqual(loaded.deep_read_count, 0)
            self.assertEqual(artifacts[0].status, "blocked")
            self.assertEqual(artifacts[0].local_path, None)

    def test_deep_read_source_rejects_non_pdf_bytes(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="application/pdf",
                content=b"<html>not a pdf</html>",
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=lambda path: ["should not run"],
            )

            loaded = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.reason, "not_pdf_bytes")
            self.assertEqual(loaded.deep_read_count, 0)
            self.assertEqual(artifacts[0].status, "blocked")

    def test_deep_read_source_rejects_html_content_type_even_with_pdf_bytes(self):
        def fake_downloader(url):
            return DownloadedPdf(
                requested_url=url,
                final_url=url,
                content_type="text/html; charset=utf-8",
                content=b"%PDF-1.7\nfake test pdf\n%%EOF",
            )

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            source = "https://arxiv.org/pdf/2401.12345v1"
            store.add_batch_item(batch.batch_id, source, evaluate_source(source))

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                source,
                downloader=fake_downloader,
                extractor=lambda path: ["should not run"],
            )

            loaded = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.reason, "content_type_not_pdf")
            self.assertEqual(loaded.deep_read_count, 0)
            self.assertEqual(artifacts[0].status, "blocked")

    def test_deep_read_source_records_no_safe_pdf_without_attempting_doi_download(self):
        candidate = Candidate(
            provider="pubmed",
            title="DOI-only paper",
            source_for_gate="10.1128/jcm.00123-23",
            doi="10.1128/jcm.00123-23",
            pmid="12345678",
        )

        def fake_downloader(url):
            raise AssertionError("DOI-only candidate should not be downloaded")

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / ".friday"
            store = FridayStore(data_dir / "friday.db")
            batch = store.create_batch(query="test query", limit=1, mode="query")
            store.add_batch_item(
                batch.batch_id,
                candidate.source_for_gate,
                evaluate_source(candidate.source_for_gate),
                candidate=candidate,
            )

            result = deep_read_source(
                store,
                data_dir,
                batch.batch_id,
                candidate.source_for_gate,
                candidate=candidate,
                downloader=fake_downloader,
                extractor=lambda path: ["should not run"],
            )

            loaded = store.get_batch(batch.batch_id)
            artifacts = store.list_pdf_artifacts(batch.batch_id)

            self.assertEqual(result.status, "blocked")
            self.assertEqual(result.reason, "no_safe_pdf_url")
            self.assertEqual(loaded.deep_read_count, 0)
            self.assertEqual(artifacts[0].status, "blocked")


if __name__ == "__main__":
    unittest.main()
