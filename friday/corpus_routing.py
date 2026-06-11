from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence


@dataclass(frozen=True)
class CorpusRouteMatch:
    entry: dict[str, Any]
    score: int
    matched_terms: tuple[str, ...]
    corpus_path: str

    @property
    def title(self) -> str:
        return _clean_text(self.entry.get("title")) or _clean_text(self.entry.get("citation_key")) or "Untitled"

    @property
    def source_pointer(self) -> str:
        return _clean_text(self.entry.get("source_pointer")) or _clean_text(self.entry.get("doi")) or "-"

    @property
    def citation_key(self) -> str:
        return _clean_text(self.entry.get("citation_key")) or "-"


@dataclass(frozen=True)
class CorpusRouteResult:
    query: str
    should_use_corpus: bool
    matches: list[CorpusRouteMatch]
    loaded_count: int
    rejected_paths: list[dict[str, str]]
    corpus_paths: list[str]


def route_corpus_query(
    query: str,
    corpus_paths: Sequence[str | Path] | str | None,
    *,
    min_score: int = 12,
    min_matches: int = 1,
    limit: int = 20,
) -> CorpusRouteResult:
    paths = parse_corpus_paths(corpus_paths)
    loaded_count = 0
    rejected_paths: list[dict[str, str]] = []
    scored: list[CorpusRouteMatch] = []

    for path in paths:
        entries, rejection = _load_corpus_entries(path)
        if rejection:
            rejected_paths.append(rejection)
            continue
        loaded_count += len(entries)
        for entry in entries:
            match = _score_entry(query, entry, path)
            if match.score >= min_score:
                scored.append(match)

    scored.sort(key=lambda item: (-item.score, item.title.lower(), item.source_pointer))
    matches = scored[: max(0, limit)]
    should_use = len(matches) >= max(1, min_matches)
    if not should_use:
        matches = []
    return CorpusRouteResult(
        query=query,
        should_use_corpus=should_use,
        matches=matches,
        loaded_count=loaded_count,
        rejected_paths=rejected_paths,
        corpus_paths=[str(path) for path in paths],
    )


def parse_corpus_paths(value: Sequence[str | Path] | str | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        raw_values = [str(value)]
    else:
        raw_values = [str(item) for item in value]
    paths: list[Path] = []
    for raw in raw_values:
        for part in re.split(rf"[,\n{re.escape(os.pathsep)}]", raw):
            cleaned = part.strip()
            if cleaned:
                paths.append(Path(cleaned).expanduser())
    return paths


def _load_corpus_entries(path: Path) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], _path_rejection(path, "missing_file", "Corpus path does not exist.")
    except json.JSONDecodeError as exc:
        return [], _path_rejection(path, "invalid_json", f"Corpus JSON could not be parsed: {exc.msg}.")
    except OSError as exc:
        return [], _path_rejection(path, "read_error", str(exc))

    entries = payload.get("literature_corpus") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return [], _path_rejection(path, "invalid_corpus", "Expected a literature_corpus list.")
    return [entry for entry in entries if isinstance(entry, dict)], None


def _score_entry(query: str, entry: dict[str, Any], corpus_path: Path) -> CorpusRouteMatch:
    query_terms = _significant_terms(query)
    fields = {
        "title": (_tokens(_clean_text(entry.get("title"))), 8),
        "abstract": (_tokens(_clean_text(entry.get("abstract"))), 4),
        "tags": (_tokens(_join_list(entry.get("tags"))), 6),
        "venue": (_tokens(_clean_text(entry.get("venue"))), 2),
        "authors": (_tokens(_join_authors(entry.get("authors"))), 1),
        "identifiers": (
            _tokens(
                " ".join(
                    value
                    for value in [
                        _clean_text(entry.get("citation_key")),
                        _clean_text(entry.get("doi")),
                        _clean_text(entry.get("source_pointer")),
                    ]
                    if value
                )
            ),
            1,
        ),
    }

    score = 0
    matched_terms: set[str] = set()
    for term in query_terms:
        for field_tokens, weight in fields.values():
            if term in field_tokens:
                score += weight
                matched_terms.add(term)
    return CorpusRouteMatch(
        entry=entry,
        score=score,
        matched_terms=tuple(sorted(matched_terms)),
        corpus_path=str(corpus_path),
    )


def _significant_terms(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in _STOPWORDS and len(token) > 1}


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {_normalize_token(token) for token in re.findall(r"[A-Za-z0-9]+", text.lower())}


def _normalize_token(token: str) -> str:
    if token in {"mathematics", "mathematical", "mathematic"}:
        return "math"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return _clean_text(value) or ""


def _join_authors(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    names: list[str] = []
    for author in value:
        if isinstance(author, dict):
            names.extend(str(part) for part in author.values() if part)
        elif author:
            names.append(str(author))
    return " ".join(names)


def _path_rejection(path: Path, reason: str, detail: str) -> dict[str, str]:
    return {"path": str(path), "reason": reason, "detail": detail}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "friday",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "using",
    "what",
    "with",
}
