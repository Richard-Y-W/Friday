from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse, urlunparse


DOI_PATTERN = re.compile(r"^(?:doi:)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)

ALLOWED_DOMAINS = {
    "arxiv.org",
    "export.arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "doi.org",
    "dx.doi.org",
    "crossref.org",
    "api.crossref.org",
    "openalex.org",
    "api.openalex.org",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "nature.com",
    "www.nature.com",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "www.sciencedirect.com",
    "ieee.org",
    "ieeexplore.ieee.org",
    "acm.org",
    "dl.acm.org",
    "plos.org",
    "journals.plos.org",
    "cell.com",
    "www.cell.com",
    "science.org",
    "www.science.org",
    "academic.oup.com",
    "cambridge.org",
    "www.cambridge.org",
    "bmj.com",
    "www.bmj.com",
    "jamanetwork.com",
    "www.jamanetwork.com",
    "nejm.org",
    "www.nejm.org",
    "asm.org",
    "journals.asm.org",
    "frontiersin.org",
    "www.frontiersin.org",
    "clinicalmicrobiologyandinfection.com",
    "www.clinicalmicrobiologyandinfection.com",
    "mdpi.com",
    "www.mdpi.com",
    "journals.sagepub.com",
    "ftp.ncbi.nlm.nih.gov",
}

BLOCKED_DOMAINS = {
    "github.com",
    "gist.github.com",
    "gitlab.com",
    "bitbucket.org",
    "drive.google.com",
    "docs.google.com",
    "dropbox.com",
    "www.dropbox.com",
}

BLOCKED_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".rar",
    ".7z",
    ".dmg",
    ".exe",
    ".bin",
    ".sh",
    ".py",
    ".ipynb",
    ".js",
)

BLOCKED_PATH_PREFIXES = ("/e-print/",)


@dataclass(frozen=True)
class SourceDecision:
    original: str
    normalized: str
    kind: str
    allowed: bool
    reason: str
    domain: str | None = None


def evaluate_source(source: str) -> SourceDecision:
    original = source.strip()
    doi_match = DOI_PATTERN.match(original)
    if doi_match:
        doi = doi_match.group(1)
        return SourceDecision(original, doi, "doi", True, "allowed_doi")

    parsed = urlparse(original)
    if parsed.scheme not in {"http", "https", "ftp"} or not parsed.netloc:
        return SourceDecision(original, original, "unknown", False, "unsupported_identifier")

    domain = parsed.netloc.lower()
    normalized = _normalize_url(parsed)
    path = parsed.path.lower()

    if domain in BLOCKED_DOMAINS:
        return SourceDecision(original, normalized, "url", False, "blocked_domain", domain)

    if _is_blocked_artifact(path):
        return SourceDecision(
            original,
            normalized,
            "url",
            False,
            "blocked_extension_or_artifact",
            domain,
        )

    if domain not in ALLOWED_DOMAINS:
        return SourceDecision(original, normalized, "url", False, "domain_not_allowlisted", domain)

    return SourceDecision(original, normalized, "url", True, "allowed_domain", domain)


def _normalize_url(parsed) -> str:
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", parsed.query, ""))


def _is_blocked_artifact(path: str) -> bool:
    if path.startswith(BLOCKED_PATH_PREFIXES):
        return True
    return any(path.endswith(suffix) for suffix in BLOCKED_SUFFIXES)
