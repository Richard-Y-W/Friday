from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import ceil
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from friday.query_planning import QueryPlan, plan_query, render_acronym_expansions


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass(frozen=True)
class Candidate:
    provider: str
    title: str
    source_for_gate: str
    url: str | None = None
    pdf_url: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    abstract: str | None = None
    relevance_score: int | None = None
    relevance_reason: str | None = None
    query_variant: str | None = None
    query_intent: str | None = None
    acronym_expansions: str | None = None
    journal: str | None = None
    concepts: str | None = None
    mesh_terms: str | None = None
    oa_status: str | None = None
    open_access_pdf_url: str | None = None


@dataclass(frozen=True)
class PubMedFullTextMetadata:
    abstract: str | None = None
    mesh_terms: str | None = None
    journal: str | None = None


def discover_candidates(
    query: str,
    limit: int,
    fetch_json: Callable[[str], dict[str, Any]] | None = None,
    fetch_text: Callable[[str], str] | None = None,
    page_size: int = 200,
    request_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] | None = None,
    learned_profile_dir: Path | None = None,
) -> list[Candidate]:
    if limit <= 0:
        return []

    fetch_json = fetch_json or _fetch_json
    fetch_text = fetch_text or _fetch_text
    page_size = _bounded_page_size(page_size)
    throttle = _RequestThrottle(request_delay_seconds, sleep or time.sleep)
    candidates: list[Candidate] = []
    provider_limits = _provider_limits(limit)
    query_plan = plan_query(query, learned_profile_dir=learned_profile_dir)
    query_variants = query_plan.expanded_queries

    openalex_limit = _per_variant_limit(provider_limits["openalex"], query_variants)
    for query_variant in query_variants:
        for offset, page_limit in _page_offsets(openalex_limit, page_size):
            openalex_url = "https://api.openalex.org/works?" + urlencode(
                {
                    "search": query_variant,
                    "per-page": min(page_limit, 200),
                    "page": (offset // page_size) + 1,
                }
            )
            try:
                throttle.wait()
                page_candidates = parse_openalex(fetch_json(openalex_url))
            except Exception:
                break
            if not page_candidates:
                break
            candidates.extend(_tag_candidates(page_candidates, query_plan, query_variant))
            if len(page_candidates) < page_limit:
                break

    arxiv_limit = _per_variant_limit(provider_limits["arxiv"], query_variants)
    if arxiv_limit > 0:
        for query_variant in query_variants:
            for offset, page_limit in _page_offsets(arxiv_limit, page_size):
                arxiv_url = "https://export.arxiv.org/api/query?" + urlencode(
                    {"search_query": f"all:{query_variant}", "start": offset, "max_results": page_limit}
                )
                try:
                    throttle.wait()
                    page_candidates = parse_arxiv(fetch_text(arxiv_url))
                except Exception:
                    break
                if not page_candidates:
                    break
                candidates.extend(_tag_candidates(page_candidates, query_plan, query_variant))
                if len(page_candidates) < page_limit:
                    break

    pubmed_limit = _per_variant_limit(provider_limits["pubmed"], query_variants)
    if pubmed_limit > 0:
        for query_variant in query_variants:
            for offset, page_limit in _page_offsets(pubmed_limit, page_size):
                esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
                    {
                        "db": "pubmed",
                        "term": query_variant,
                        "retmode": "json",
                        "retmax": page_limit,
                        "retstart": offset,
                    }
                )
                try:
                    throttle.wait()
                    search_payload = fetch_json(esearch_url)
                except Exception:
                    break
                ids = (search_payload.get("esearchresult") or {}).get("idlist") or []
                if not ids:
                    break
                esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
                    {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
                )
                try:
                    throttle.wait()
                    pubmed_candidates = parse_pubmed_summary(fetch_json(esummary_url))
                    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(
                        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
                    )
                    try:
                        throttle.wait()
                        full_text_metadata = parse_pubmed_abstracts(fetch_text(efetch_url))
                    except Exception:
                        full_text_metadata = {}
                    candidates.extend(
                        _tag_candidates(
                            _merge_pubmed_metadata(pubmed_candidates, full_text_metadata),
                            query_plan,
                            query_variant,
                        )
                    )
                except Exception:
                    break
                if len(ids) < page_limit:
                    break

    return _dedupe(candidates)[:limit]


def parse_openalex(payload: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for work in payload.get("results", []):
        ids = work.get("ids") or {}
        doi = _normalize_doi(ids.get("doi"))
        pmid = _extract_trailing_id(ids.get("pmid"))
        pmcid = _extract_trailing_id(ids.get("pmcid"))
        location = work.get("primary_location") or {}
        landing_url = location.get("landing_page_url")
        pdf_url = location.get("pdf_url")
        source_metadata = location.get("source") or {}
        open_access = work.get("open_access") or {}
        url = landing_url or pdf_url
        source = doi or pdf_url or landing_url
        title = _clean_text(work.get("display_name"))
        if not title or not source:
            continue
        candidates.append(
            Candidate(
                provider="openalex",
                title=title,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                year=_safe_int(work.get("publication_year")),
                url=url,
                pdf_url=pdf_url,
                source_for_gate=source,
                abstract=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
                journal=_clean_text(source_metadata.get("display_name")) or None,
                concepts=_concepts_from_openalex(work),
                oa_status=_open_access_status(open_access),
                open_access_pdf_url=_open_access_pdf_url(work, open_access, pdf_url),
            )
        )
    return candidates


def parse_arxiv(atom_xml: str) -> list[Candidate]:
    root = ET.fromstring(atom_xml)
    candidates: list[Candidate] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _clean_text(_node_text(entry.find("atom:title", ATOM_NS)))
        abstract = _clean_text(_node_text(entry.find("atom:summary", ATOM_NS)))
        abs_url = _node_text(entry.find("atom:id", ATOM_NS))
        arxiv_id = _arxiv_id_from_abs_url(abs_url)
        published = _node_text(entry.find("atom:published", ATOM_NS))
        doi = _normalize_doi(_node_text(entry.find("arxiv:doi", ATOM_NS)))
        pdf_url = _arxiv_pdf_url(entry, arxiv_id)
        if not title or not abs_url or not arxiv_id:
            continue
        candidates.append(
            Candidate(
                provider="arxiv",
                title=title,
                doi=doi,
                arxiv_id=arxiv_id,
                year=_year_from_text(published),
                url=abs_url,
                pdf_url=pdf_url,
                source_for_gate=pdf_url,
                abstract=abstract or None,
            )
        )
    return candidates


def parse_pubmed_summary(payload: dict[str, Any]) -> list[Candidate]:
    result = payload.get("result") or {}
    candidates: list[Candidate] = []
    for uid in result.get("uids", []):
        item = result.get(str(uid)) or {}
        title = _clean_text(item.get("title"))
        pmid = str(item.get("uid") or uid)
        doi = _doi_from_article_ids(item.get("articleids") or [])
        pmcid = _pmcid_from_article_ids(item.get("articleids") or [])
        source = doi or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        if not title or not pmid:
            continue
        candidates.append(
            Candidate(
                provider="pubmed",
                title=title,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                year=_year_from_text(item.get("pubdate")),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source_for_gate=source,
                journal=_clean_text(item.get("fulljournalname")) or None,
            )
        )
    return candidates


def parse_pubmed_abstracts(xml_text: str) -> dict[str, PubMedFullTextMetadata]:
    root = ET.fromstring(xml_text)
    metadata: dict[str, PubMedFullTextMetadata] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean_text(_node_text(article.find(".//MedlineCitation/PMID")))
        if not pmid:
            continue
        abstract_parts = []
        for node in article.findall(".//Article/Abstract/AbstractText"):
            text = _clean_text(_node_text(node))
            if not text:
                continue
            label = _clean_text(node.attrib.get("Label"))
            abstract_parts.append(f"{label}: {text}" if label else text)
        mesh_terms = [
            _clean_text(_node_text(node))
            for node in article.findall(".//MeshHeading/DescriptorName")
            if _clean_text(_node_text(node))
        ]
        journal = _clean_text(_node_text(article.find(".//Article/Journal/Title")))
        metadata[pmid] = PubMedFullTextMetadata(
            abstract=" ".join(abstract_parts) or None,
            mesh_terms=_join_unique(mesh_terms),
            journal=journal or None,
        )
    return metadata


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    without_tags = re.sub(r"<[^>]+>", "", str(value))
    return " ".join(without_tags.split())


def _abstract_from_inverted_index(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    positioned_words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            try:
                positioned_words.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
    if not positioned_words:
        return None
    return _clean_text(" ".join(word for _, word in sorted(positioned_words)))


def _normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:", "", text, flags=re.IGNORECASE)
    return text or None


def _extract_trailing_id(value: Any) -> str | None:
    if not value:
        return None
    return str(value).rstrip("/").split("/")[-1]


def _node_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    return node.text


def _arxiv_id_from_abs_url(abs_url: str | None) -> str | None:
    if not abs_url:
        return None
    return abs_url.rstrip("/").split("/")[-1]


def _arxiv_pdf_url(entry: ET.Element, arxiv_id: str | None) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            href = link.attrib.get("href")
            if href:
                return href.replace("http://", "https://", 1)
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "https://arxiv.org/pdf/"


def _year_from_text(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group(0)) if match else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _doi_from_article_ids(article_ids: list[dict[str, Any]]) -> str | None:
    for article_id in article_ids:
        if article_id.get("idtype") == "doi":
            return _normalize_doi(article_id.get("value"))
    return None


def _pmcid_from_article_ids(article_ids: list[dict[str, Any]]) -> str | None:
    for article_id in article_ids:
        if article_id.get("idtype") == "pmc":
            value = _clean_text(article_id.get("value"))
            return value or None
    return None


def _merge_pubmed_metadata(
    candidates: list[Candidate],
    metadata: dict[str, PubMedFullTextMetadata],
) -> list[Candidate]:
    enriched = []
    for candidate in candidates:
        details = metadata.get(candidate.pmid or "")
        if details is None:
            enriched.append(candidate)
            continue
        enriched.append(
            replace(
                candidate,
                abstract=details.abstract or candidate.abstract,
                mesh_terms=details.mesh_terms,
                journal=details.journal or candidate.journal,
            )
        )
    return enriched


def _concepts_from_openalex(work: dict[str, Any]) -> str | None:
    names: list[str] = []
    for concept in work.get("concepts") or []:
        name = _clean_text((concept or {}).get("display_name"))
        if name:
            names.append(name)
    for topic in work.get("topics") or []:
        name = _clean_text((topic or {}).get("display_name"))
        if name:
            names.append(name)
    primary_topic = work.get("primary_topic") or {}
    primary_topic_name = _clean_text(primary_topic.get("display_name"))
    if primary_topic_name:
        names.append(primary_topic_name)
    return _join_unique(names)


def _open_access_status(open_access: dict[str, Any]) -> str | None:
    status = _clean_text(open_access.get("oa_status"))
    if status:
        return status
    if open_access.get("is_oa") is True:
        return "oa"
    if open_access.get("is_oa") is False:
        return "closed"
    return None


def _open_access_pdf_url(
    work: dict[str, Any],
    open_access: dict[str, Any],
    fallback_pdf_url: str | None,
) -> str | None:
    candidates = []
    for key in ("best_oa_location", "primary_location"):
        location = work.get(key) or {}
        if isinstance(location, dict):
            candidates.append(location.get("pdf_url"))
    for location in work.get("locations") or []:
        if isinstance(location, dict):
            candidates.append(location.get("pdf_url"))
    candidates.append(open_access.get("oa_url"))
    candidates.append(fallback_pdf_url)

    for value in candidates:
        url = _clean_text(value)
        if url and _looks_pdfish_url(url):
            return url
    return None


def _looks_pdfish_url(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path.lower()
    return path.endswith(".pdf") or path.endswith("/pdf") or "/pdf/" in path


def _join_unique(values: list[str]) -> str | None:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return "; ".join(unique) or None


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _tag_candidates(candidates: list[Candidate], query_plan: QueryPlan, query_variant: str) -> list[Candidate]:
    acronym_expansions = render_acronym_expansions(query_plan)
    return [
        replace(
            candidate,
            query_variant=query_variant,
            query_intent=query_plan.intent,
            acronym_expansions=acronym_expansions,
        )
        for candidate in candidates
    ]


def _candidate_key(candidate: Candidate) -> str:
    if candidate.doi:
        return f"doi:{candidate.doi.lower()}"
    if candidate.pmid:
        return f"pmid:{candidate.pmid}"
    if candidate.arxiv_id:
        return f"arxiv:{candidate.arxiv_id.lower()}"
    return f"source:{candidate.source_for_gate.lower()}"


def _provider_limits(limit: int) -> dict[str, int]:
    providers = ("openalex", "arxiv", "pubmed")
    base = limit // len(providers)
    remainder = limit % len(providers)
    return {
        provider: base + (1 if index < remainder else 0)
        for index, provider in enumerate(providers)
    }


def _per_variant_limit(provider_limit: int, query_variants: tuple[str, ...]) -> int:
    if provider_limit <= 0:
        return 0
    return max(1, ceil(provider_limit / max(1, len(query_variants))))


def _bounded_page_size(page_size: int) -> int:
    return max(1, min(page_size, 200))


def _page_offsets(total: int, page_size: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    return [
        (offset, min(page_size, total - offset))
        for offset in range(0, total, page_size)
    ]


class _RequestThrottle:
    def __init__(self, delay_seconds: float, sleep: Callable[[float], None]):
        self.delay_seconds = max(0.0, delay_seconds)
        self.sleep = sleep
        self._has_requested = False

    def wait(self) -> None:
        if self._has_requested and self.delay_seconds > 0:
            self.sleep(self.delay_seconds)
        self._has_requested = True
