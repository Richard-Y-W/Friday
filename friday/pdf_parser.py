from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows fallback
    _resource = None


PDF_TEXT_TIMEOUT_SECONDS = 30
PDF_PARSE_MAX_PAGES = 1000
PDF_PARSE_MAX_OUTPUT_BYTES = 20 * 1024 * 1024
PDF_PARSE_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.6
_PARSER_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)


@dataclass(frozen=True)
class ParsedPdfPage:
    page_number: int
    text: str
    confidence: float = 1.0
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PdfParseResult:
    parser_name: str
    parser_version: str
    pages: list[ParsedPdfPage]
    confidence: float
    flags: tuple[str, ...] = ()


class PdfParserFailure(RuntimeError):
    pass


PdfParser = Callable[[Path], PdfParseResult]


def parse_pdf_with_fallback(
    path: Path,
    *,
    parsers: list[PdfParser] | None = None,
    min_confidence: float = LOW_CONFIDENCE_THRESHOLD,
) -> PdfParseResult:
    candidates = parsers or [parse_with_pdftotext_layout, parse_with_pdftotext_default, parse_with_pdftotext_raw]
    failures = []
    best: PdfParseResult | None = None
    for parser in candidates:
        try:
            result = parser(path)
        except Exception as exc:
            failures.append(f"{_parser_name(parser)}:{type(exc).__name__}")
            continue
        if not result.pages:
            failures.append(f"{result.parser_name}:empty")
            continue
        best = _better_result(best, result)
    if best is not None and best.confidence >= min_confidence:
        return best
    if best is not None:
        return best
    detail = ", ".join(failures) if failures else "no_parsers"
    raise PdfParserFailure(f"pdf_parse_failed:{detail}")


def parse_with_pdftotext_layout(path: Path) -> PdfParseResult:
    pages = _run_pdftotext(path, mode="layout")
    parsed_pages = _parsed_pages_from_text_pages(pages)
    confidence, flags = _score_pages(parsed_pages)
    return PdfParseResult(
        parser_name="pdftotext-layout",
        parser_version="poppler",
        pages=parsed_pages,
        confidence=confidence,
        flags=flags,
    )


def parse_with_pdftotext_default(path: Path) -> PdfParseResult:
    pages = _run_pdftotext(path, mode="default")
    parsed_pages = _parsed_pages_from_text_pages(pages)
    confidence, flags = _score_pages(parsed_pages)
    return PdfParseResult(
        parser_name="pdftotext-default",
        parser_version="poppler",
        pages=parsed_pages,
        confidence=confidence,
        flags=flags,
    )


def parse_with_pdftotext_raw(path: Path) -> PdfParseResult:
    pages = _run_pdftotext(path, mode="raw")
    parsed_pages = _parsed_pages_from_text_pages(pages)
    confidence, flags = _score_pages(parsed_pages)
    return PdfParseResult(
        parser_name="pdftotext-raw",
        parser_version="poppler",
        pages=parsed_pages,
        confidence=confidence,
        flags=flags,
    )


def parser_from_extractor(extractor: Callable[[Path], list[str]]) -> PdfParser:
    def parse(path: Path) -> PdfParseResult:
        parsed_pages = _parsed_pages_from_text_pages(extractor(path))
        confidence, flags = _score_pages(parsed_pages)
        return PdfParseResult(
            parser_name="injected-extractor",
            parser_version="test",
            pages=parsed_pages,
            confidence=confidence,
            flags=flags,
        )

    return parse


def _run_pdftotext(path: Path, *, mode: str) -> list[str]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("pdftotext_not_found")

    command = [executable, "-q", "-f", "1", "-l", str(PDF_PARSE_MAX_PAGES)]
    if mode == "layout":
        command.append("-layout")
    elif mode == "raw":
        command.append("-raw")
    run_kwargs: dict[str, object] = {
        "check": True,
        "timeout": PDF_TEXT_TIMEOUT_SECONDS,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": _scrubbed_parser_env(),
    }
    preexec = _parser_preexec(PDF_PARSE_MEMORY_LIMIT_BYTES, PDF_TEXT_TIMEOUT_SECONDS + 5)
    if preexec is not None:
        run_kwargs["preexec_fn"] = preexec

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "paper.txt"
        subprocess.run(
            [*command, str(path), str(output_path)],
            **run_kwargs,
        )
        text = _read_capped(output_path, PDF_PARSE_MAX_OUTPUT_BYTES)
    return text.split("\f")


def _scrubbed_parser_env() -> dict[str, str]:
    return {key: os.environ[key] for key in _PARSER_ENV_ALLOWLIST if key in os.environ}


def _parser_preexec(memory_limit_bytes: int, cpu_seconds: int):
    if _resource is None:
        return None

    def _apply() -> None:  # pragma: no cover - runs only in the POSIX child
        for limit_name, limit_value in (
            (_resource.RLIMIT_AS, memory_limit_bytes),
            (_resource.RLIMIT_CPU, cpu_seconds),
        ):
            try:
                _resource.setrlimit(limit_name, (limit_value, limit_value))
            except (OSError, ValueError):
                # Some platforms reject specific rlimits in preexec_fn. Keep the
                # parser sandbox best-effort instead of aborting every parse.
                continue

    return _apply


def _read_capped(path: Path, max_bytes: int) -> str:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def _parsed_pages_from_text_pages(pages: list[str]) -> list[ParsedPdfPage]:
    parsed = []
    for index, page in enumerate(pages, start=1):
        text = page.strip()
        if not text:
            continue
        confidence, flags = _score_text(text)
        parsed.append(
            ParsedPdfPage(
                page_number=index,
                text=text,
                confidence=confidence,
                flags=flags,
            )
        )
    return parsed


def _score_pages(pages: list[ParsedPdfPage]) -> tuple[float, tuple[str, ...]]:
    if not pages:
        return 0.0, ("no_extractable_text",)
    average = sum(page.confidence for page in pages) / len(pages)
    flags = []
    for page in pages:
        flags.extend(page.flags)
    return round(average, 3), tuple(_ordered_unique(flags))


def _score_text(text: str) -> tuple[float, tuple[str, ...]]:
    flags = []
    confidence = 1.0
    word_count = len(text.split())
    if word_count < 20:
        flags.append("short_page")
        confidence -= 0.25
    if "Articles seTo" in text or "reSearch strategy" in text:
        flags.append("column_stitching")
        confidence -= 0.45
    if "  " * 8 in text:
        flags.append("wide_spacing")
        confidence -= 0.1
    confidence = max(0.0, min(1.0, confidence))
    return round(confidence, 3), tuple(_ordered_unique(flags))


def _better_result(current: PdfParseResult | None, candidate: PdfParseResult) -> PdfParseResult:
    if current is None:
        return candidate
    if candidate.confidence > current.confidence:
        return candidate
    if candidate.confidence == current.confidence and len(candidate.pages) > len(current.pages):
        return candidate
    return current


def _parser_name(parser: PdfParser) -> str:
    return getattr(parser, "__name__", parser.__class__.__name__)


def _ordered_unique(values: list[str] | tuple[str, ...]) -> list[str]:
    unique = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
