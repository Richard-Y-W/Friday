import unittest
from urllib.parse import parse_qs, urlparse

from friday.discovery import (
    discover_candidates,
    parse_arxiv,
    parse_pubmed_abstracts,
    parse_openalex,
    parse_pubmed_summary,
)


class DiscoveryParserTests(unittest.TestCase):
    def test_parse_openalex_candidate(self):
        payload = {
            "results": [
                {
                    "display_name": "MALDI-TOF antimicrobial resistance transfer",
                    "publication_year": 2024,
                    "ids": {
                        "doi": "https://doi.org/10.1038/s41586-020-2649-2",
                        "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                    },
                    "primary_location": {
                        "landing_page_url": "https://www.nature.com/articles/s41586-020-2649-2",
                        "source": {"display_name": "Nature"},
                    },
                    "best_oa_location": {
                        "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/",
                    },
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "hybrid",
                        "oa_url": "https://www.nature.com/articles/s41586-020-2649-2",
                    },
                    "locations": [
                        {
                            "landing_page_url": "https://www.nature.com/articles/s41586-020-2649-2",
                            "pdf_url": "https://www.nature.com/articles/s41586-020-2649-2.pdf",
                        }
                    ],
                    "concepts": [
                        {"display_name": "Antimicrobial resistance", "score": 0.93},
                        {"display_name": "Mass spectrometry", "score": 0.82},
                    ],
                    "topics": [
                        {"display_name": "Clinical microbiology"},
                    ],
                    "cited_by_count": 42,
                }
            ]
        }

        candidate = parse_openalex(payload)[0]

        self.assertEqual(candidate.provider, "openalex")
        self.assertEqual(candidate.title, "MALDI-TOF antimicrobial resistance transfer")
        self.assertEqual(candidate.doi, "10.1038/s41586-020-2649-2")
        self.assertEqual(candidate.pmid, "12345678")
        self.assertEqual(candidate.year, 2024)
        self.assertEqual(candidate.url, "https://www.nature.com/articles/s41586-020-2649-2")
        self.assertEqual(candidate.source_for_gate, "10.1038/s41586-020-2649-2")
        self.assertEqual(candidate.journal, "Nature")
        self.assertEqual(
            candidate.concepts,
            "Antimicrobial resistance; Mass spectrometry; Clinical microbiology",
        )
        self.assertEqual(candidate.oa_status, "hybrid")
        self.assertEqual(candidate.open_access_pdf_url, "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567/pdf/")

    def test_parse_openalex_strips_html_from_title(self):
        payload = {
            "results": [
                {
                    "display_name": "Rapid AMR in <i>Pseudomonas aeruginosa</i>",
                    "publication_year": 2024,
                    "ids": {"doi": "https://doi.org/10.1093/jambio/lxad248"},
                    "primary_location": {"landing_page_url": "https://academic.oup.com/example"},
                }
            ]
        }

        candidate = parse_openalex(payload)[0]

        self.assertEqual(candidate.title, "Rapid AMR in Pseudomonas aeruginosa")

    def test_parse_openalex_reconstructs_abstract_from_inverted_index(self):
        payload = {
            "results": [
                {
                    "display_name": "Rapid AMR prediction",
                    "publication_year": 2024,
                    "ids": {"doi": "https://doi.org/10.1093/jambio/lxad248"},
                    "primary_location": {"landing_page_url": "https://academic.oup.com/example"},
                    "abstract_inverted_index": {
                        "Antimicrobial": [0],
                        "resistance": [1],
                        "prediction": [2],
                        "from": [3],
                        "MALDI-TOF": [4],
                    },
                }
            ]
        }

        candidate = parse_openalex(payload)[0]

        self.assertEqual(
            candidate.abstract,
            "Antimicrobial resistance prediction from MALDI-TOF",
        )

    def test_parse_arxiv_candidate(self):
        atom = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2401.12345v2</id>
                <title>Low SNR drone RF fingerprinting</title>
                <summary>Signal classification with wireless measurements.</summary>
                <published>2024-01-01T00:00:00Z</published>
                <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" rel="related" type="application/pdf"/>
                <arxiv:doi>10.48550/arXiv.2401.12345</arxiv:doi>
              </entry>
        </feed>
        """

        candidate = parse_arxiv(atom)[0]

        self.assertEqual(candidate.provider, "arxiv")
        self.assertEqual(candidate.title, "Low SNR drone RF fingerprinting")
        self.assertEqual(candidate.doi, "10.48550/arXiv.2401.12345")
        self.assertEqual(candidate.arxiv_id, "2401.12345v2")
        self.assertEqual(candidate.year, 2024)
        self.assertEqual(candidate.url, "http://arxiv.org/abs/2401.12345v2")
        self.assertEqual(candidate.source_for_gate, "https://arxiv.org/pdf/2401.12345v2")
        self.assertEqual(candidate.abstract, "Signal classification with wireless measurements.")

    def test_parse_pubmed_summary_candidate(self):
        payload = {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "uid": "12345678",
                    "title": "Cross-site MALDI AMR generalization",
                    "pubdate": "2023 Oct",
                    "fulljournalname": "Journal of Clinical Microbiology",
                    "articleids": [
                        {"idtype": "doi", "value": "10.1128/jcm.00123-23"},
                        {"idtype": "pmc", "value": "PMC1234567"},
                        {"idtype": "pubmed", "value": "12345678"},
                    ],
                },
            }
        }

        candidate = parse_pubmed_summary(payload)[0]

        self.assertEqual(candidate.provider, "pubmed")
        self.assertEqual(candidate.title, "Cross-site MALDI AMR generalization")
        self.assertEqual(candidate.doi, "10.1128/jcm.00123-23")
        self.assertEqual(candidate.pmid, "12345678")
        self.assertEqual(candidate.pmcid, "PMC1234567")
        self.assertEqual(candidate.year, 2023)
        self.assertEqual(candidate.url, "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        self.assertEqual(candidate.source_for_gate, "10.1128/jcm.00123-23")
        self.assertEqual(candidate.journal, "Journal of Clinical Microbiology")

    def test_parse_pubmed_abstracts_adds_abstract_mesh_and_journal(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>12345678</PMID>
              <Article>
                <Journal>
                  <Title>Journal of Clinical Microbiology</Title>
                </Journal>
                <Abstract>
                  <AbstractText>We evaluated MALDI-TOF antimicrobial resistance prediction.</AbstractText>
                  <AbstractText Label="Results">Resistance detection improved in clinical isolates.</AbstractText>
                </Abstract>
              </Article>
              <MeshHeadingList>
                <MeshHeading><DescriptorName>Drug Resistance, Microbial</DescriptorName></MeshHeading>
                <MeshHeading><DescriptorName>Mass Spectrometry</DescriptorName></MeshHeading>
              </MeshHeadingList>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>
        """

        metadata = parse_pubmed_abstracts(xml)

        self.assertEqual(
            metadata["12345678"].abstract,
            "We evaluated MALDI-TOF antimicrobial resistance prediction. Results: Resistance detection improved in clinical isolates.",
        )
        self.assertEqual(metadata["12345678"].mesh_terms, "Drug Resistance, Microbial; Mass Spectrometry")
        self.assertEqual(metadata["12345678"].journal, "Journal of Clinical Microbiology")

    def test_discover_candidates_queries_providers_and_deduplicates(self):
        calls = []

        def fake_json(url):
            calls.append(url)
            if "api.openalex.org/works" in url:
                return {
                    "results": [
                        {
                            "display_name": "Shared DOI paper",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/shared"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/shared"},
                        }
                    ]
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": ["111", "222"]}}
            if "esummary.fcgi" in url:
                return {
                    "result": {
                        "uids": ["111"],
                        "111": {
                            "uid": "111",
                            "title": "PubMed-only paper",
                            "pubdate": "2022",
                            "articleids": [{"idtype": "pubmed", "value": "111"}],
                        },
                    }
                }
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            calls.append(url)
            return """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2401.12345v1</id>
                <title>Arxiv-only paper</title>
                <published>2024-01-01T00:00:00Z</published>
                <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
              </entry>
            </feed>
            """

        candidates = discover_candidates("maldi amr", limit=3, fetch_json=fake_json, fetch_text=fake_text)

        self.assertEqual(len(candidates), 3)
        self.assertEqual([candidate.provider for candidate in candidates], ["openalex", "arxiv", "pubmed"])
        self.assertTrue(any("api.openalex.org/works" in call for call in calls))
        self.assertTrue(any("export.arxiv.org/api/query" in call for call in calls))
        self.assertTrue(any("esearch.fcgi" in call for call in calls))
        self.assertTrue(any("esummary.fcgi" in call for call in calls))

    def test_discover_candidates_queries_multiple_indexes_even_when_openalex_has_enough(self):
        calls = []

        def fake_json(url):
            calls.append(url)
            if "api.openalex.org/works" in url:
                return {
                    "results": [
                        {
                            "display_name": "OpenAlex first",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/first"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/first"},
                        },
                        {
                            "display_name": "OpenAlex second",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/second"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/second"},
                        },
                        {
                            "display_name": "OpenAlex third",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/third"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/third"},
                        },
                        {
                            "display_name": "OpenAlex fourth",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/fourth"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/fourth"},
                        },
                        {
                            "display_name": "OpenAlex fifth",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/fifth"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/fifth"},
                        },
                    ]
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": ["111"]}}
            if "esummary.fcgi" in url:
                return {
                    "result": {
                        "uids": ["111"],
                        "111": {
                            "uid": "111",
                            "title": "PubMed paper",
                            "pubdate": "2023",
                            "articleids": [{"idtype": "pubmed", "value": "111"}],
                        },
                    }
                }
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            calls.append(url)
            return """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2401.12345v1</id>
                <title>Arxiv paper</title>
                <published>2024-01-01T00:00:00Z</published>
                <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf"/>
              </entry>
            </feed>
            """

        candidates = discover_candidates("maldi amr", limit=5, fetch_json=fake_json, fetch_text=fake_text)

        self.assertEqual(len(candidates), 5)
        self.assertTrue(any("api.openalex.org/works" in call for call in calls))
        self.assertTrue(any("export.arxiv.org/api/query" in call for call in calls))
        self.assertTrue(any("esearch.fcgi" in call for call in calls))
        self.assertTrue(any("esummary.fcgi" in call for call in calls))

    def test_discover_candidates_uses_expanded_query_variants_for_ambiguous_acronym(self):
        calls = []

        def fake_json(url):
            calls.append(url)
            if "api.openalex.org/works" in url:
                return {
                    "results": [
                        {
                            "display_name": "Expanded query paper",
                            "publication_year": 2024,
                            "ids": {"doi": "https://doi.org/10.1000/expanded"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/expanded"},
                        }
                    ]
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": []}}
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            calls.append(url)
            return """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>
            """

        candidates = discover_candidates("MALDI AMR", limit=3, fetch_json=fake_json, fetch_text=fake_text)

        openalex_searches = [
            parse_qs(urlparse(call).query).get("search", [""])[0]
            for call in calls
            if "api.openalex.org/works" in call
        ]
        arxiv_searches = [
            parse_qs(urlparse(call).query).get("search_query", [""])[0]
            for call in calls
            if "export.arxiv.org/api/query" in call
        ]

        self.assertIn("MALDI antimicrobial resistance", openalex_searches)
        self.assertIn("MALDI-TOF antibiotic resistance", openalex_searches)
        self.assertNotIn("MALDI AMR", openalex_searches)
        self.assertIn("all:MALDI antimicrobial resistance", arxiv_searches)
        self.assertEqual(candidates[0].query_variant, "MALDI antimicrobial resistance")
        self.assertEqual(candidates[0].query_intent, "biomedical")
        self.assertIn("AMR=antimicrobial resistance", candidates[0].acronym_expansions)

    def test_discover_candidates_keeps_successful_results_when_one_expanded_call_times_out(self):
        def fake_json(url):
            search = parse_qs(urlparse(url).query).get("search", [""])[0]
            if "MALDI-TOF antibiotic resistance" in search:
                raise TimeoutError("simulated timeout")
            if "api.openalex.org/works" in url:
                return {
                    "results": [
                        {
                            "display_name": f"Paper for {search}",
                            "publication_year": 2024,
                            "ids": {"doi": f"https://doi.org/10.1000/{search.replace(' ', '-')}"},
                            "primary_location": {"landing_page_url": "https://www.nature.com/articles/example"},
                        }
                    ]
                }
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": []}}
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            return """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>
            """

        candidates = discover_candidates("MALDI AMR", limit=3, fetch_json=fake_json, fetch_text=fake_text)

        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(all(candidate.query_variant != "MALDI-TOF antibiotic resistance" for candidate in candidates))

    def test_discover_candidates_fetches_pubmed_abstracts_and_mesh_terms(self):
        calls = []

        def fake_json(url):
            calls.append(url)
            if "api.openalex.org/works" in url:
                return {"results": []}
            if "esearch.fcgi" in url:
                return {"esearchresult": {"idlist": ["12345678"]}}
            if "esummary.fcgi" in url:
                return {
                    "result": {
                        "uids": ["12345678"],
                        "12345678": {
                            "uid": "12345678",
                            "title": "Cross-site MALDI AMR generalization",
                            "pubdate": "2023 Oct",
                            "fulljournalname": "Summary Journal",
                            "articleids": [{"idtype": "pubmed", "value": "12345678"}],
                        },
                    }
                }
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            calls.append(url)
            if "efetch.fcgi" in url:
                return """<?xml version="1.0" encoding="UTF-8"?>
                <PubmedArticleSet>
                  <PubmedArticle>
                    <MedlineCitation>
                      <PMID>12345678</PMID>
                      <Article>
                        <Journal><Title>Journal of Clinical Microbiology</Title></Journal>
                        <Abstract><AbstractText>MALDI antimicrobial resistance abstract.</AbstractText></Abstract>
                      </Article>
                      <MeshHeadingList>
                        <MeshHeading><DescriptorName>Drug Resistance, Microbial</DescriptorName></MeshHeading>
                      </MeshHeadingList>
                    </MedlineCitation>
                  </PubmedArticle>
                </PubmedArticleSet>
                """
            return """<?xml version="1.0" encoding="UTF-8"?>
            <feed xmlns="http://www.w3.org/2005/Atom"></feed>
            """

        candidates = discover_candidates("Pseudomonas MALDI spectra", limit=3, fetch_json=fake_json, fetch_text=fake_text)

        pubmed = next(candidate for candidate in candidates if candidate.provider == "pubmed")
        self.assertTrue(any("efetch.fcgi" in call for call in calls))
        self.assertEqual(pubmed.abstract, "MALDI antimicrobial resistance abstract.")
        self.assertEqual(pubmed.mesh_terms, "Drug Resistance, Microbial")
        self.assertEqual(pubmed.journal, "Journal of Clinical Microbiology")

    def test_discover_candidates_pages_each_provider_to_reach_large_limits(self):
        calls = []

        def fake_json(url):
            calls.append(url)
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            if "api.openalex.org/works" in url:
                page = int(query.get("page", ["1"])[0])
                if page == 1:
                    suffixes = ["oa1", "oa2"]
                elif page == 2:
                    suffixes = ["oa3"]
                else:
                    suffixes = []
                return {
                    "results": [
                        {
                            "display_name": f"OpenAlex {suffix}",
                            "publication_year": 2024,
                            "ids": {"doi": f"https://doi.org/10.1000/{suffix}"},
                            "primary_location": {"landing_page_url": f"https://www.nature.com/articles/{suffix}"},
                        }
                        for suffix in suffixes
                    ]
                }
            if "esearch.fcgi" in url:
                retstart = int(query.get("retstart", ["0"])[0])
                if retstart == 0:
                    ids = ["101", "102"]
                elif retstart == 2:
                    ids = ["103"]
                else:
                    ids = []
                return {"esearchresult": {"idlist": ids}}
            if "esummary.fcgi" in url:
                ids = query.get("id", [""])[0].split(",")
                return {
                    "result": {
                        "uids": ids,
                        **{
                            pmid: {
                                "uid": pmid,
                                "title": f"PubMed {pmid}",
                                "pubdate": "2024",
                                "articleids": [{"idtype": "pubmed", "value": pmid}],
                            }
                            for pmid in ids
                            if pmid
                        },
                    }
                }
            raise AssertionError(f"unexpected json url: {url}")

        def fake_text(url):
            calls.append(url)
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            if "export.arxiv.org/api/query" in url:
                start = int(query.get("start", ["0"])[0])
                if start == 0:
                    ids = ["2401.00001v1", "2401.00002v1"]
                elif start == 2:
                    ids = ["2401.00003v1"]
                else:
                    ids = []
                entries = "\n".join(
                    f"""
                    <entry>
                      <id>http://arxiv.org/abs/{arxiv_id}</id>
                      <title>Arxiv {arxiv_id}</title>
                      <published>2024-01-01T00:00:00Z</published>
                      <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}" rel="related" type="application/pdf"/>
                    </entry>
                    """
                    for arxiv_id in ids
                )
                return f"""<?xml version="1.0" encoding="UTF-8"?>
                <feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>
                """
            if "efetch.fcgi" in url:
                return """<?xml version="1.0" encoding="UTF-8"?><PubmedArticleSet></PubmedArticleSet>"""
            raise AssertionError(f"unexpected text url: {url}")

        candidates = discover_candidates(
            "proteomics",
            limit=9,
            page_size=2,
            fetch_json=fake_json,
            fetch_text=fake_text,
        )

        self.assertEqual(len(candidates), 9)
        openalex_pages = [
            parse_qs(urlparse(call).query).get("page", [""])[0]
            for call in calls
            if "api.openalex.org/works" in call
        ]
        arxiv_starts = [
            parse_qs(urlparse(call).query).get("start", [""])[0]
            for call in calls
            if "export.arxiv.org/api/query" in call
        ]
        pubmed_starts = [
            parse_qs(urlparse(call).query).get("retstart", [""])[0]
            for call in calls
            if "esearch.fcgi" in call
        ]

        self.assertEqual(openalex_pages, ["1", "2"])
        self.assertEqual(arxiv_starts, ["0", "2"])
        self.assertEqual(pubmed_starts, ["0", "2"])


if __name__ == "__main__":
    unittest.main()
