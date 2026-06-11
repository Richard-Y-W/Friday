from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from jarvis_research.discovery import Candidate
from jarvis_research.evidence import extract_evidence_from_pages
from jarvis_research.pdf_text import clean_pdf_pages
from jarvis_research.source_policy import evaluate_source
from jarvis_research.storage import JarvisStore


MAX_PDF_BYTES = 50 * 1024 * 1024
PDF_TEXT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class PdfResolution:
    pdf_url: str | None
    reason: str


@dataclass(frozen=True)
class DownloadedPdf:
    requested_url: str
    final_url: str
    content_type: str | None
    content: bytes


@dataclass(frozen=True)
class PdfIngestionResult:
    status: str
    reason: str
    artifact_id: str | None
    pdf_url: str | None
    page_count: int


Downloader = Callable[[str], DownloadedPdf]
Extractor = Callable[[Path], list[str]]
TextFetcher = Callable[[str], str]


def resolve_candidate_pdf_url(
    candidate: Candidate | None,
    text_fetcher: TextFetcher | None = None,
) -> PdfResolution:
    if candidate is None:
        return PdfResolution(None, "no_candidate_metadata")

    if candidate.provider == "arxiv" and candidate.arxiv_id:
        return PdfResolution(f"https://arxiv.org/pdf/{candidate.arxiv_id}", "resolved_arxiv_pdf")

    for url, reason in _candidate_pdf_urls(candidate, text_fetcher):
        if not url or not _looks_like_pdf_url(url):
            continue
        decision = evaluate_source(url)
        if decision.allowed:
            return PdfResolution(decision.normalized, reason)

    return PdfResolution(None, "no_safe_pdf_url")


def deep_read_source(
    store: JarvisStore,
    data_dir: Path,
    batch_id: str,
    source: str,
    candidate: Candidate | None = None,
    downloader: Downloader | None = None,
    extractor: Extractor | None = None,
    text_fetcher: TextFetcher | None = None,
) -> PdfIngestionResult:
    resolution = resolve_candidate_pdf_url(candidate, text_fetcher=text_fetcher or fetch_url_text)
    pdf_url = resolution.pdf_url or source
    if resolution.pdf_url is None and not _looks_like_pdf_url(source):
        return _record_blocked(
            store,
            batch_id,
            source,
            None,
            None,
            None,
            None,
            None,
            resolution.reason if resolution.reason != "no_candidate_metadata" else "no_safe_pdf_url",
        )

    source_decision = evaluate_source(pdf_url)
    if not source_decision.allowed:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            None,
            None,
            None,
            None,
            f"source_{source_decision.reason}",
        )

    downloader = downloader or download_pdf
    extractor = extractor or extract_pdf_text_pages

    try:
        downloaded = downloader(source_decision.normalized)
    except Exception as exc:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            None,
            None,
            None,
            None,
            f"download_failed:{type(exc).__name__}",
        )

    final_decision = evaluate_source(downloaded.final_url)
    if not final_decision.allowed:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            None,
            f"final_url_{final_decision.reason}",
        )

    if len(downloaded.content) > MAX_PDF_BYTES:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            None,
            "pdf_too_large",
        )

    if not _content_type_allows_pdf(downloaded.content_type):
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            None,
            "content_type_not_pdf",
        )

    if not downloaded.content.startswith(b"%PDF-"):
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            None,
            "not_pdf_bytes",
        )

    content_hash = hashlib.sha256(downloaded.content).hexdigest()
    relative_path = Path("artifacts") / batch_id / f"{content_hash}.pdf"
    absolute_path = Path(data_dir) / relative_path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_path.write_bytes(downloaded.content)

    try:
        pages = clean_pdf_pages([page.strip() for page in extractor(absolute_path)])
    except Exception as exc:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            content_hash,
            f"text_extraction_failed:{type(exc).__name__}",
        )

    pages = [page for page in pages if page]
    if not pages:
        return _record_blocked(
            store,
            batch_id,
            source,
            pdf_url,
            downloaded.final_url,
            downloaded.content_type,
            len(downloaded.content),
            content_hash,
            "no_extractable_text",
        )

    artifact = store.add_pdf_artifact(
        batch_id,
        source=source,
        pdf_url=pdf_url,
        final_url=downloaded.final_url,
        sha256=content_hash,
        byte_count=len(downloaded.content),
        content_type=downloaded.content_type,
        local_path=relative_path.as_posix(),
        status="stored",
        reason="pdf_text_extracted",
    )
    store.add_pdf_pages(artifact.artifact_id, pages)
    store.add_evidence_records(artifact.artifact_id, extract_evidence_from_pages(pages))
    return PdfIngestionResult(
        status="stored",
        reason="pdf_text_extracted",
        artifact_id=artifact.artifact_id,
        pdf_url=pdf_url,
        page_count=len(pages),
    )


def download_pdf(url: str) -> DownloadedPdf:
    with urlopen(url, timeout=30) as response:
        content = response.read(MAX_PDF_BYTES + 1)
        return DownloadedPdf(
            requested_url=url,
            final_url=response.geturl(),
            content_type=response.headers.get("content-type"),
            content=content,
        )


def fetch_url_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read(2 * 1024 * 1024).decode("utf-8", errors="replace")


def extract_pdf_text_pages(path: Path) -> list[str]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext_not_found")

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "paper.txt"
        subprocess.run(
            [executable, "-f", "1", "-l", "1000", "-layout", str(path), str(output_path)],
            check=True,
            timeout=PDF_TEXT_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        text = output_path.read_text(encoding="utf-8", errors="replace")
    return text.split("\f")


def _record_blocked(
    store: JarvisStore,
    batch_id: str,
    source: str,
    pdf_url: str | None,
    final_url: str | None,
    content_type: str | None,
    byte_count: int | None,
    sha256: str | None,
    reason: str,
) -> PdfIngestionResult:
    artifact = store.add_pdf_artifact(
        batch_id,
        source=source,
        pdf_url=pdf_url,
        final_url=final_url,
        sha256=sha256,
        byte_count=byte_count,
        content_type=content_type,
        local_path=None,
        status="blocked",
        reason=reason,
    )
    return PdfIngestionResult(
        status="blocked",
        reason=reason,
        artifact_id=artifact.artifact_id,
        pdf_url=pdf_url,
        page_count=0,
    )


def _looks_like_pdf_url(value: str) -> bool:
    decision = evaluate_source(value)
    if not decision.allowed or decision.kind != "url":
        return False
    normalized = decision.normalized.lower()
    path = urlparse(normalized).path
    return (
        path.endswith(".pdf")
        or path.endswith("/pdf")
        or path.endswith("/pdfft")
        or "/pdf/" in path
    )


def _candidate_pdf_urls(candidate: Candidate, text_fetcher: TextFetcher | None) -> Iterable[tuple[str | None, str]]:
    yield candidate.open_access_pdf_url, "resolved_open_access_pdf"
    yield candidate.pdf_url, "resolved_safe_pdf_url"
    yield _pmc_pdf_url(candidate.pmcid, text_fetcher), "resolved_pmc_pdf"
    yield candidate.source_for_gate, "resolved_safe_pdf_url"
    yield candidate.url, "resolved_safe_pdf_url"

    seen_landing_urls: set[str] = set()
    for landing_url in (candidate.source_for_gate, candidate.url):
        if not landing_url or landing_url in seen_landing_urls:
            continue
        seen_landing_urls.add(landing_url)
        yield _landing_page_pdf_url(landing_url, text_fetcher), "resolved_landing_page_pdf"


def _pmc_pdf_url(pmcid: str | None, text_fetcher: TextFetcher | None = None) -> str | None:
    if not pmcid:
        return None
    value = pmcid.strip()
    if not value:
        return None
    if not value.upper().startswith("PMC"):
        value = f"PMC{value}"
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{value}/"
    if text_fetcher is not None:
        oa_pdf_url = _pmc_oa_pdf_url(value, text_fetcher)
        if oa_pdf_url:
            return oa_pdf_url
        try:
            html = text_fetcher(article_url)
        except Exception:
            html = ""
        for href in _extract_pdf_hrefs(html):
            pdf_url = urljoin(article_url, href)
            if _looks_like_pdf_url(pdf_url):
                return pdf_url
    return f"{article_url}pdf/"


def _pmc_oa_pdf_url(pmcid: str, text_fetcher: TextFetcher) -> str | None:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        xml_text = text_fetcher(url)
    except Exception:
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for link in root.findall(".//link"):
        if (link.attrib.get("format") or "").lower() != "pdf":
            continue
        href = link.attrib.get("href")
        if href and _looks_like_pdf_url(href):
            return href
    return None


def _landing_page_pdf_url(url: str | None, text_fetcher: TextFetcher | None) -> str | None:
    if not url or text_fetcher is None or _looks_like_pdf_url(url):
        return None
    decision = evaluate_source(url)
    if not decision.allowed or decision.kind != "url":
        return None
    try:
        html = text_fetcher(decision.normalized)
    except Exception:
        return None
    for href in _extract_pdf_hrefs(html):
        pdf_url = urljoin(decision.normalized, href)
        if _looks_like_pdf_url(pdf_url):
            return pdf_url
    return None


class _PdfHrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        attrs_by_name = {
            name.lower(): value
            for name, value in attrs
            if value is not None
        }
        if tag_name == "a" and attrs_by_name.get("href"):
            self.hrefs.append(attrs_by_name["href"])
            return
        if tag_name == "link" and attrs_by_name.get("href"):
            rel = attrs_by_name.get("rel", "").lower()
            media_type = attrs_by_name.get("type", "").lower()
            if "pdf" in rel or media_type == "application/pdf":
                self.hrefs.append(attrs_by_name["href"])
            return
        if tag_name != "meta" or not attrs_by_name.get("content"):
            return
        metadata_name = attrs_by_name.get("name") or attrs_by_name.get("property") or ""
        if metadata_name.lower() in {"citation_pdf_url", "dc.identifier", "og:pdf"}:
            self.hrefs.append(attrs_by_name["content"])


def _extract_pdf_hrefs(html: str) -> list[str]:
    parser = _PdfHrefParser()
    parser.feed(html)
    return [
        href
        for href in parser.hrefs
        if _pdf_href_looks_promising(href)
    ]


def _pdf_href_looks_promising(href: str) -> bool:
    value = href.lower()
    path = urlparse(value).path
    return (
        path in {"pdf", "pdfft"}
        or path.endswith(".pdf")
        or path.endswith("/pdf")
        or path.endswith("/pdfft")
        or "/pdf/" in path
    )


def _content_type_allows_pdf(value: str | None) -> bool:
    if not value:
        return True
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type in {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
        "binary/octet-stream",
    }
