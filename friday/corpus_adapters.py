from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def import_folder_corpus(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    entries = []
    rejected = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() != ".pdf":
            rejected.append(_rejection(path, "unsupported_file_type", "Only PDF files are imported by the folder adapter."))
            continue
        title = _title_from_stem(path.stem)
        entries.append(
            _entry(
                citation_key=_slug(path.stem),
                title=title,
                source_pointer=str(path.resolve()),
                source_type="pdf",
                obtained_via="folder",
            )
        )
    return _corpus(entries, "folder"), _rejection_log(rejected, "folder")


def import_zotero_corpus(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("items", [])
    entries = []
    rejected = []
    for index, item in enumerate(records, start=1):
        title = _clean_text(item.get("title"))
        if not title:
            rejected.append(_rejection(f"item_{index}", "missing_required_field", "Missing title."))
            continue
        citation_key = _clean_text(item.get("id") or item.get("citation-key") or item.get("citationKey")) or _slug(title)
        entries.append(
            _entry(
                citation_key=citation_key,
                title=title,
                source_pointer=_clean_text(item.get("URL")) or f"zotero:{citation_key}",
                source_type="zotero",
                obtained_via="zotero_json",
                authors=_csl_authors(item.get("author") or []),
                year=_csl_year(item.get("issued")),
                doi=_clean_text(item.get("DOI") or item.get("doi")),
                abstract=_clean_text(item.get("abstract")),
                venue=_clean_text(item.get("container-title") or item.get("publisher")),
            )
        )
    return _corpus(entries, "zotero_json"), _rejection_log(rejected, "zotero_json")


def import_obsidian_corpus(vault: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    vault = Path(vault)
    entries = []
    rejected = []
    for path in sorted(vault.rglob("*.md")):
        frontmatter = _frontmatter(path.read_text(encoding="utf-8"))
        title = _clean_text(frontmatter.get("title"))
        if not title:
            rejected.append(_rejection(path, "missing_required_field", "Missing frontmatter title."))
            continue
        entries.append(
            _entry(
                citation_key=_clean_text(frontmatter.get("citation_key")) or _slug(path.stem),
                title=title,
                source_pointer=str(path.resolve()),
                source_type="obsidian",
                obtained_via="obsidian",
                authors=_name_list(frontmatter.get("authors")),
                year=_safe_int(frontmatter.get("year")),
                doi=_clean_text(frontmatter.get("doi")),
                abstract=_clean_text(frontmatter.get("abstract")),
                venue=_clean_text(frontmatter.get("venue") or frontmatter.get("journal")),
                tags=_split_list(frontmatter.get("tags")),
            )
        )
    return _corpus(entries, "obsidian"), _rejection_log(rejected, "obsidian")


def write_corpus_outputs(
    output_path: Path,
    rejection_log_path: Path,
    corpus: dict[str, Any],
    rejection_log: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rejection_log_path.write_text(json.dumps(rejection_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _corpus(entries: list[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "literature_corpus",
        "adapter_name": adapter_name,
        "generated_at": _now(),
        "literature_corpus": sorted(entries, key=lambda item: item["citation_key"]),
    }


def _rejection_log(rejected: list[dict[str, Any]], adapter_name: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "artifact_type": "corpus_rejection_log",
        "adapter_name": adapter_name,
        "generated_at": _now(),
        "rejected": sorted(rejected, key=lambda item: str(item["source"])),
    }


def _entry(
    *,
    citation_key: str,
    title: str,
    source_pointer: str,
    source_type: str,
    obtained_via: str,
    authors: list[dict[str, str]] | None = None,
    year: int | None = None,
    doi: str | None = None,
    abstract: str | None = None,
    venue: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "citation_key": citation_key,
        "title": title,
        "authors": authors or [],
        "year": year,
        "doi": doi,
        "abstract": abstract,
        "venue": venue,
        "tags": tags or [],
        "source_pointer": source_pointer,
        "source_type": source_type,
        "obtained_via": obtained_via,
        "obtained_at": _now(),
    }


def _rejection(source: object, reason: str, detail: str) -> dict[str, Any]:
    return {
        "source": str(source),
        "reason": reason,
        "detail": detail,
    }


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    values: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip().strip("\"'")
    return values


def _csl_authors(authors: list[dict[str, Any]]) -> list[dict[str, str]]:
    parsed = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        if author.get("literal"):
            parsed.append({"literal": str(author["literal"])})
        elif author.get("family"):
            record = {"family": str(author["family"])}
            if author.get("given"):
                record["given"] = str(author["given"])
            parsed.append(record)
    return parsed


def _csl_year(issued: Any) -> int | None:
    try:
        return _safe_int(issued["date-parts"][0][0])
    except Exception:
        return None


def _name_list(value: str | None) -> list[dict[str, str]]:
    names = _split_list(value)
    return [{"family": name} for name in names]


def _split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]


def _title_from_stem(stem: str) -> str:
    return " ".join(stem.replace("_", " ").replace("-", " ").split())


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
