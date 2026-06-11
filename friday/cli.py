from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import sys
from typing import Callable, Sequence

from friday.compose_agent import (
    ComposePackageError,
    SECTION_CHOICES,
    build_compose_package_files,
)
from friday.corpus_adapters import (
    import_folder_corpus,
    import_obsidian_corpus,
    import_zotero_corpus,
    write_corpus_outputs,
)
from friday.corpus_routing import CorpusRouteResult, route_corpus_query
from friday.discovery import Candidate, discover_candidates
from friday.eval_suite import (
    available_eval_suites,
    render_eval_report_text,
    run_eval_suite,
)
from friday.label_export import (
    build_label_export_rows,
    render_label_export_csv,
    render_label_export_jsonl,
)
from friday.label_eval import build_label_evaluation
from friday.label_review import LABEL_REVIEW_FILTERS, build_label_review_rows
from friday.pdf_ingestion import PdfIngestionResult, deep_read_source
from friday.relevance import rank_candidates
from friday.reporting import (
    render_batch_report,
    render_batch_report_json,
    render_batch_report_markdown,
    render_scan_report,
    render_scan_report_json,
    render_scan_report_markdown,
)
from friday.research_artifacts import (
    build_batch_passport,
    build_rejection_log,
    build_research_run_summary,
    write_json_artifact,
)
from friday.run_summary import (
    RunSummaryTargetError,
    build_run_summary_dashboard,
    render_run_summary_text,
)
from friday.screening import (
    auto_label_batch_items,
    build_llm_review_queue,
    rank_deep_read_items,
    recommend_unlabeled_items,
)
from friday.settings import flatten_settings, load_settings, set_setting
from friday.source_policy import evaluate_source
from friday.storage import BatchItemRecord, FridayStore, SCREENING_LABEL_CHOICES
from friday.writing_copilot import (
    MODE_CHOICES,
    build_writing_package_files,
    build_writing_payload,
    render_writing_markdown,
)


COMMAND_NAMES = {
    "scan",
    "research",
    "smoke-run",
    "research-run",
    "research-runs",
    "eval-suite",
    "run-summary",
    "review-queue",
    "batches",
    "scans",
    "label",
    "labels",
    "auto-label",
    "settings",
    "/settings",
    "/setting",
    "report",
    "import-corpus",
    "write",
    "compose",
}

WRITING_FORMAT_CHOICES = ("markdown", "json", "package")
FRIDAY_VERSION = "1.0.0"
FRIDAY_ICON_ANSI = "\x1b[38;2;130;200;229m"
ANSI_RESET = "\x1b[0m"


def main(
    argv: Sequence[str] | None = None,
    discoverer: Callable[..., list[Candidate]] = discover_candidates,
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ] = deep_read_source,
    llm_label_client: object | None = None,
    input_stream: object | None = None,
    force_interactive: bool | None = None,
) -> int:
    parser = _build_parser()
    normalized_argv = _normalize_data_dir_args(argv)
    if _should_open_interactive_shell(normalized_argv, input_stream=input_stream, force_interactive=force_interactive):
        return _handle_interactive_shell(normalized_argv, discoverer, pdf_ingestor, llm_label_client, input_stream)
    if _is_natural_language_invocation(normalized_argv):
        return _handle_natural_language_query(normalized_argv, discoverer, pdf_ingestor, llm_label_client)
    args = parser.parse_args(normalized_argv)
    data_dir = Path(args.data_dir)

    if args.command == "import-corpus":
        return _handle_import_corpus(args)
    if args.command == "compose":
        return _handle_compose(args)
    if not args.command:
        parser.print_help()
        return 2
    if args.command in {"settings", "/settings", "/setting"}:
        return _handle_settings(args, data_dir)
    if args.command == "eval-suite":
        return _handle_eval_suite(args)

    store = FridayStore(data_dir / "friday.db")
    if args.command == "scan":
        return _handle_scan(args, store, data_dir, discoverer, pdf_ingestor)
    if args.command == "research":
        return _handle_research(args, store, data_dir, discoverer, pdf_ingestor)
    if args.command == "smoke-run":
        return _handle_smoke_run(args, store, data_dir, discoverer, pdf_ingestor, llm_label_client)
    if args.command == "research-run":
        return _handle_research_run(args, store, data_dir, discoverer, pdf_ingestor, llm_label_client)
    if args.command == "research-runs":
        return _handle_research_runs(store)
    if args.command == "run-summary":
        return _handle_run_summary(args, store)
    if args.command == "review-queue":
        return _handle_review_queue(args, store)
    if args.command == "batches":
        return _handle_batches(store)
    if args.command == "scans":
        return _handle_scans(store)
    if args.command == "label":
        return _handle_label(args, store)
    if args.command == "labels":
        return _handle_labels(args, store)
    if args.command == "auto-label":
        return _handle_auto_label(args, store, data_dir, llm_label_client)
    if args.command == "report":
        return _handle_report(args, store)
    if args.command == "write":
        return _handle_write(args, store)

    parser.print_help()
    return 2


def _should_open_interactive_shell(
    argv: Sequence[str],
    *,
    input_stream: object | None,
    force_interactive: bool | None,
) -> bool:
    _data_dir, rest = _split_data_dir_args(argv)
    if rest:
        return False
    if force_interactive is not None:
        return force_interactive
    stream = input_stream or sys.stdin
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _handle_interactive_shell(
    argv: Sequence[str],
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
    llm_label_client: object | None,
    input_stream: object | None,
) -> int:
    data_dir, _rest = _split_data_dir_args(argv)
    stream = input_stream or sys.stdin
    _print_interactive_splash()
    print("Scholarly-only evidence assistant. Type /help or /exit.")
    while True:
        print("friday> ", end="", flush=True)
        raw_line = stream.readline()
        if raw_line == "":
            print("")
            return 0
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() in {"/exit", "/quit", "exit", "quit"}:
            return 0
        code = _handle_interactive_line(
            line,
            data_dir=data_dir,
            discoverer=discoverer,
            pdf_ingestor=pdf_ingestor,
            llm_label_client=llm_label_client,
        )
        if code != 0:
            print(f"Command exited with status {code}.")


def _handle_interactive_line(
    line: str,
    *,
    data_dir: str,
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
    llm_label_client: object | None,
) -> int:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        print(f"Could not parse command: {exc}")
        return 2
    if not tokens:
        return 0
    if tokens[0] == "friday":
        tokens = tokens[1:]
        if not tokens:
            return 0
    if tokens[0] in {"/help", "help"}:
        _print_interactive_help()
        return 0
    if tokens[0].startswith("/") or tokens[0] in COMMAND_NAMES:
        return main(
            ["--data-dir", data_dir, *tokens],
            discoverer=discoverer,
            pdf_ingestor=pdf_ingestor,
            llm_label_client=llm_label_client,
            force_interactive=False,
        )

    package_dir = _interactive_report_package_dir(Path(data_dir), " ".join(tokens))
    package_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "--data-dir",
        data_dir,
        *tokens,
        "--format",
        "json",
        "--output",
        str(package_dir / "batch-report.json"),
        "--passport",
        str(package_dir / "passport.json"),
        "--rejection-log",
        str(package_dir / "rejection-log.json"),
        "--write",
        "--write-format",
        "package",
        "--write-output",
        str(package_dir),
    ]
    code = main(
        command,
        discoverer=discoverer,
        pdf_ingestor=pdf_ingestor,
        llm_label_client=llm_label_client,
        force_interactive=False,
    )
    if code == 0:
        desktop_pdf = _copy_report_pdf_to_desktop(package_dir)
        if desktop_pdf:
            print(f"Desktop PDF: {desktop_pdf}")
        print(f"Report package: {package_dir}")
    return code


def _print_interactive_splash() -> None:
    lines = [
        _splash_line("     ▄▄▄"),
        _splash_line("  ▄███████▄", f"        friday research {FRIDAY_VERSION}"),
        _splash_line(" ██  ▄ ▄  ██", "       Paper scanner - cited PDF reports"),
        _splash_line(" ██   ▀   ██", f"       {Path.cwd()}"),
        _splash_line("  ▀███████▀"),
    ]
    print("\n".join(lines))


def _splash_line(icon: str, text: str = "") -> str:
    return f"{FRIDAY_ICON_ANSI}{icon}{ANSI_RESET}{text}"


def _copy_report_pdf_to_desktop(package_dir: Path) -> Path | None:
    source = package_dir / "report.pdf"
    if not source.exists():
        return None
    output_dir = Path(os.environ.get("FRIDAY_DESKTOP_REPORT_DIR", Path.home() / "Desktop" / "FridayReports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{package_dir.name}.pdf"
    shutil.copyfile(source, destination)
    return destination


def _print_interactive_help() -> None:
    print("Interactive commands:")
    print("  tell me about <topic>        Run scholarly research and write a report package.")
    print("  friday tell me about <topic> Same as above; the prefix is optional.")
    print("  /settings                   Show saved limits and defaults.")
    print("  /settings set key value      Update a saved setting.")
    print("  /exit                       Leave Friday.")


def _interactive_report_package_dir(data_dir: Path, query: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return data_dir / "reports" / f"{_slugify_query(query)}-{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="friday")
    parser.add_argument("--data-dir", default=".friday", help="Directory for Friday local state.")
    subparsers = parser.add_subparsers(dest="command")

    scan = subparsers.add_parser("scan", help="Create a single-paper scan or batch scan record.")
    scan.add_argument("source", nargs="?", help="DOI, arXiv/PubMed/publisher URL, or allowlisted PDF URL.")
    scan.add_argument("--query", help="Create a metadata-discovery batch for a scholarly query.")
    scan.add_argument("--limit", type=int, help="Maximum candidates to discover for query mode.")
    scan.add_argument("--page-size", type=int, default=200, help="Maximum records to request per scholarly index page.")
    scan.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between scholarly index requests.",
    )
    scan.add_argument("--resume-batch", help="Resume an existing query batch by batch_id.")
    scan.add_argument(
        "--deep-read-limit",
        type=int,
        default=0,
        help="Safely download and parse at most N allowed scholarly PDFs from query results.",
    )
    scan.add_argument(
        "--min-relevance",
        type=int,
        default=25,
        help="Minimum relevance score required for query-mode deep reads.",
    )
    scan.add_argument(
        "--deep-read-workers",
        type=int,
        default=1,
        help="Maximum concurrent safe PDF deep reads for query batches.",
    )
    scan.add_argument("--manifest", help="Path to a newline or CSV manifest of sources.")
    scan.add_argument("--all-deep", action="store_true", help="Deep-scan every safe source in a manifest.")

    research = subparsers.add_parser("research", help="Run a query scan and render/export research artifacts.")
    research.add_argument("--query", required=True, help="Scholarly query to screen.")
    research.add_argument("--limit", type=int, default=100, help="Maximum candidates to discover.")
    research.add_argument("--page-size", type=int, default=200, help="Maximum records to request per scholarly index page.")
    research.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between scholarly index requests.",
    )
    research.add_argument("--resume-batch", help="Resume an existing query batch by batch_id.")
    research.add_argument(
        "--deep-read-limit",
        type=int,
        default=0,
        help="Safely download and parse at most N allowed scholarly PDFs from query results.",
    )
    research.add_argument(
        "--min-relevance",
        type=int,
        default=25,
        help="Minimum relevance score required for query-mode deep reads.",
    )
    research.add_argument(
        "--deep-read-workers",
        type=int,
        default=1,
        help="Maximum concurrent safe PDF deep reads for query batches.",
    )
    research.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="markdown",
        help="Report output format.",
    )
    research.add_argument("--output", help="Path to write the rendered report.")
    research.add_argument("--passport", help="Path to write the batch passport JSON.")
    research.add_argument("--rejection-log", help="Path to write the rejection log JSON.")

    smoke_run = subparsers.add_parser(
        "smoke-run",
        help="Run one dogfood research query and write a complete smoke artifact pack.",
    )
    smoke_run.add_argument("query", nargs="+", help="Scholarly query to screen.")
    smoke_run.add_argument("--limit", type=int, default=100, help="Maximum candidates to discover.")
    smoke_run.add_argument("--page-size", type=int, default=200, help="Maximum records to request per scholarly index page.")
    smoke_run.add_argument("--request-delay", type=float, default=0.0, help="Seconds to wait between scholarly index requests.")
    smoke_run.add_argument("--deep-read-limit", type=int, default=5, help="Safely parse at most N selected scholarly PDFs.")
    smoke_run.add_argument("--min-relevance", type=int, default=25, help="Minimum relevance score for unlabeled deep reads.")
    smoke_run.add_argument("--deep-read-workers", type=int, default=1, help="Maximum concurrent safe PDF deep reads.")
    smoke_run.add_argument(
        "--auto-label-provider",
        choices=("heuristic", "llm"),
        help="Use heuristic labels for all candidates, with optional LLM review when set to llm.",
    )
    smoke_run.add_argument("--auto-label-model", help="Model for optional provider=llm review.")
    smoke_run.add_argument("--auto-label-min-confidence", type=float, help="Minimum confidence required to apply labels.")
    smoke_run.add_argument("--auto-label-api-base-url", help="OpenAI-compatible API base URL for provider=llm.")
    smoke_run.add_argument("--auto-label-api-key-env", help="Environment variable containing the LLM API key.")
    smoke_run.add_argument("--llm-review-limit", type=int, help="Maximum top candidates to review with the LLM labeler.")
    smoke_run.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="markdown",
        help="Report output format.",
    )
    smoke_run.add_argument("--output-dir", help="Directory to write smoke-run artifacts.")

    research_run = subparsers.add_parser("research-run", help="Run or resume a scalable literature research workflow.")
    research_run.add_argument("query", nargs="*", help="Scholarly query to screen for a new run.")
    research_run.add_argument("--resume-run", help="Resume an existing research run by run_id.")
    research_run.add_argument("--latest", action="store_true", help="Resume the latest research run.")
    research_run.add_argument("--limit", type=int, help="Maximum candidates to discover.")
    research_run.add_argument("--page-size", type=int, help="Maximum records to request per scholarly index page.")
    research_run.add_argument("--request-delay", type=float, help="Seconds to wait between scholarly index requests.")
    research_run.add_argument("--deep-read-limit", type=int, help="Safely parse at most N selected scholarly PDFs.")
    research_run.add_argument("--min-relevance", type=int, help="Minimum relevance score for unlabeled deep reads.")
    research_run.add_argument("--deep-read-workers", type=int, help="Maximum concurrent safe PDF deep reads.")
    research_run.add_argument(
        "--auto-label-provider",
        choices=("heuristic", "llm"),
        help="Use heuristic labels for all candidates, with optional LLM review when set to llm.",
    )
    research_run.add_argument("--auto-label-model", help="Model for optional provider=llm review.")
    research_run.add_argument("--auto-label-min-confidence", type=float, help="Minimum confidence required to apply labels.")
    research_run.add_argument("--auto-label-api-base-url", help="OpenAI-compatible API base URL for provider=llm.")
    research_run.add_argument("--auto-label-api-key-env", help="Environment variable containing the LLM API key.")
    research_run.add_argument("--llm-review-limit", type=int, help="Maximum top candidates to review with the LLM labeler.")
    research_run.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        help="Report output format.",
    )
    research_run.add_argument("--output", help="Path to write the rendered report.")
    research_run.add_argument("--passport", help="Path to write the batch passport JSON.")
    research_run.add_argument("--rejection-log", help="Path to write the rejection log JSON.")
    research_run.add_argument("--run-summary", help="Path to write the research run summary JSON.")

    subparsers.add_parser("research-runs", help="List recent research run ledger entries.")

    eval_suite = subparsers.add_parser("eval-suite", help="Run offline Friday quality and safety evaluations.")
    eval_suite.add_argument("action", choices=("list", "run"), help="List suites or run an eval suite.")
    eval_suite.add_argument(
        "--suite",
        default="core",
        help="Suite to run: core, biomedical, natural-language, safety, or gold.",
    )
    eval_suite.add_argument("--format", choices=("text", "json"), default="text", help="Eval report output format.")

    run_summary = subparsers.add_parser("run-summary", help="Summarize the latest research run or batch.")
    run_summary_target = run_summary.add_mutually_exclusive_group(required=True)
    run_summary_target.add_argument("--latest", action="store_true", help="Summarize the latest run, or latest batch.")
    run_summary_target.add_argument("--run-id", help="Research run ID to summarize.")
    run_summary_target.add_argument("--batch-id", help="Batch ID to summarize.")
    run_summary.add_argument("--format", choices=("text", "json"), default="text", help="Summary output format.")
    run_summary.add_argument("--limit", type=int, default=5, help="Maximum attention rows per section.")

    review_queue = subparsers.add_parser("review-queue", help="Preview the smart LLM review queue for a batch.")
    review_queue_batch = review_queue.add_mutually_exclusive_group(required=True)
    review_queue_batch.add_argument("--batch-id", help="Batch ID to inspect.")
    review_queue_batch.add_argument("--latest", action="store_true", help="Inspect the latest batch.")
    review_queue.add_argument("--limit", type=int, default=20, help="Maximum queue rows to print.")

    batches = subparsers.add_parser("batches", help="List prior batch scans.")

    scans = subparsers.add_parser("scans", help="List prior single-paper scans.")

    label = subparsers.add_parser("label", help="Apply a human screening label to a batch item.")
    label_batch = label.add_mutually_exclusive_group(required=True)
    label_batch.add_argument("--batch-id", help="Batch ID containing the source to label.")
    label_batch.add_argument("--latest", action="store_true", help="Label an item in the latest batch.")
    label.add_argument("--source", required=True, help="Source URL, DOI, or normalized source to label.")
    label.add_argument("--label", required=True, choices=SCREENING_LABEL_CHOICES, help="Screening label.")
    label.add_argument("--note", help="Optional short note explaining the label.")

    labels = subparsers.add_parser("labels", help="List or export screening labels.")
    labels.add_argument(
        "action",
        nargs="?",
        choices=("eval", "export", "review", "set"),
        help="Use 'eval', 'review', 'set', or 'export' for label workflows.",
    )
    labels_batch = labels.add_mutually_exclusive_group(required=True)
    labels_batch.add_argument("--batch-id", help="Batch ID to inspect.")
    labels_batch.add_argument("--latest", action="store_true", help="Inspect the latest batch.")
    labels_batch.add_argument("--all", action="store_true", help="Export labels from all batches.")
    labels.add_argument("--recommend", action="store_true", help="Recommend unlabeled batch items from label feedback.")
    labels.add_argument("--limit", type=int, default=10, help="Maximum recommendations to print.")
    labels.add_argument("--format", choices=("jsonl", "csv", "json"), default="jsonl", help="Output format.")
    labels.add_argument("--output", help="Path to write exported labels.")
    labels.add_argument("--only", choices=LABEL_REVIEW_FILTERS, help="Filter review rows.")
    labels.add_argument("--min-relevance", type=int, default=0, help="Minimum relevance score for review rows.")
    labels.add_argument("--source", help="Source URL, DOI, or normalized source to label with labels set.")
    labels.add_argument("--label", choices=SCREENING_LABEL_CHOICES, help="Human label to apply with labels set.")
    labels.add_argument("--note", help="Optional note for labels set.")

    auto_label = subparsers.add_parser("auto-label", help="Apply metadata-only agent screening labels to a batch.")
    auto_label_batch = auto_label.add_mutually_exclusive_group(required=True)
    auto_label_batch.add_argument("--batch-id", help="Batch ID to auto-label.")
    auto_label_batch.add_argument("--latest", action="store_true", help="Auto-label the latest batch.")
    auto_label.add_argument("--limit", type=int, default=1000, help="Maximum allowed batch items to inspect.")
    auto_label.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence required to apply.")
    auto_label.add_argument("--provider", choices=("heuristic", "llm"), help="Auto-label provider.")
    auto_label.add_argument("--model", help="Model for provider=llm.")
    auto_label.add_argument("--api-base-url", help="OpenAI-compatible API base URL for provider=llm.")
    auto_label.add_argument("--api-key-env", help="Environment variable containing the LLM API key.")
    auto_label_mode = auto_label.add_mutually_exclusive_group()
    auto_label_mode.add_argument("--dry-run", action="store_true", help="Preview labels without writing them.")
    auto_label_mode.add_argument("--apply", action="store_true", help="Write agent labels to the batch.")

    settings = subparsers.add_parser(
        "settings",
        aliases=["/settings", "/setting"],
        help="Show or update Friday defaults.",
    )
    settings.add_argument("action", nargs="?", choices=("set",), help="Use 'set' to update one setting.")
    settings.add_argument("key", nargs="?", help="Setting key, such as research.limit.")
    settings.add_argument("value", nargs="?", help="New setting value.")

    report = subparsers.add_parser("report", help="Render a scan or batch report.")
    report.add_argument("target_id", nargs="?", help="scan_* or batch_* identifier.")
    report.add_argument("--latest", action="store_true", help="Report the latest batch.")
    report.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Report output format.",
    )
    report.add_argument("--output", help="Path to write the rendered report.")
    report.add_argument("--passport", help="Path to write a batch passport JSON.")
    report.add_argument("--rejection-log", help="Path to write a batch rejection log JSON.")

    import_corpus = subparsers.add_parser("import-corpus", help="Convert a user-owned corpus into Friday JSON.")
    corpus_source = import_corpus.add_mutually_exclusive_group(required=True)
    corpus_source.add_argument("--folder", help="Folder of PDF files to import.")
    corpus_source.add_argument("--zotero-json", help="Zotero CSL JSON export to import.")
    corpus_source.add_argument("--obsidian-vault", help="Obsidian vault folder to import.")
    import_corpus.add_argument("--output", required=True, help="Path to write literature_corpus JSON.")
    import_corpus.add_argument("--rejection-log", required=True, help="Path to write corpus rejection log JSON.")

    write = subparsers.add_parser("write", help="Draft evidence-bound writing from scanner artifacts.")
    write_source = write.add_mutually_exclusive_group(required=True)
    write_source.add_argument("--batch-id", help="Batch ID to draft from.")
    write_source.add_argument("--latest", action="store_true", help="Draft from the latest batch.")
    write_source.add_argument("--report", help="Path to a batch report JSON file.")
    write.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="claim-table",
        help="Writing mode.",
    )
    write.add_argument(
        "--format",
        choices=("markdown", "json", "package"),
        default="markdown",
        help="Writing output format.",
    )
    write.add_argument("--output", help="Path to write the writing draft or package directory.")

    compose = subparsers.add_parser("compose", help="Compose evidence-bound drafts from a writing package.")
    compose.add_argument("--package", dest="package_dir", required=True, help="Writing package directory to compose from.")
    compose.add_argument(
        "--section",
        choices=SECTION_CHOICES,
        default="results",
        help="Draft section to compose.",
    )
    compose.add_argument("--output", required=True, help="Directory to write compose artifacts.")

    return parser


def _normalize_data_dir_args(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    data_dir: str | None = None
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "--data-dir" and index + 1 < len(raw):
            data_dir = raw[index + 1]
            index += 2
            continue
        normalized.append(token)
        index += 1
    if data_dir is not None:
        return ["--data-dir", data_dir, *normalized]
    return normalized


def _is_natural_language_invocation(argv: Sequence[str]) -> bool:
    _data_dir, rest = _split_data_dir_args(argv)
    if not rest:
        return False
    command = rest[0]
    if command.startswith("-"):
        return False
    return command not in COMMAND_NAMES


def _handle_natural_language_query(
    argv: Sequence[str],
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
    llm_label_client: object | None,
) -> int:
    args = _parse_natural_language_args(argv)
    data_dir = Path(args.data_dir)
    if not args.query:
        print("Natural research query is empty.")
        return 2
    writing_error = _natural_writing_validation_error(args)
    if writing_error:
        print(writing_error)
        return 2

    corpus_route = route_corpus_query(
        args.query,
        args.corpus_paths,
        min_score=args.corpus_min_score,
        min_matches=args.corpus_min_matches,
        limit=args.corpus_limit,
    )
    if corpus_route.should_use_corpus and not args.write:
        print(f"Natural query: {args.query}")
        print("Natural query route: corpus")
        content = _format_corpus_report(args.format, corpus_route)
        _emit_output(content, args.output, label="corpus report")
        return 0

    store = FridayStore(data_dir / "friday.db")
    batch = store.create_batch(query=args.query, limit=args.limit, mode="query")
    discovery_error = None
    try:
        candidates = _call_discoverer(
            discoverer,
            args.query,
            args.limit,
            page_size=args.page_size,
            request_delay_seconds=args.request_delay,
        )
    except Exception as exc:
        candidates = []
        discovery_error = f"{type(exc).__name__}: {exc}"

    candidates = rank_candidates(args.query, candidates)
    for candidate in candidates:
        decision = evaluate_source(candidate.source_for_gate)
        store.add_batch_item_if_new(
            batch.batch_id,
            candidate.source_for_gate,
            decision,
            candidate=candidate,
        )

    if args.auto_label_enabled:
        auto_label_batch_items(
            store,
            batch.batch_id,
            query=args.query,
            limit=args.auto_label_limit,
            apply=args.auto_label_apply,
            min_confidence=args.auto_label_min_confidence,
            provider=args.auto_label_provider,
            model=args.auto_label_model,
            llm_client=llm_label_client,
            api_base_url=args.auto_label_api_base_url,
            api_key_env=args.auto_label_api_key_env,
        )

    _deep_read_ranked_batch_items(
        store,
        data_dir,
        batch.batch_id,
        deep_read_limit=args.deep_read_limit,
        min_relevance=args.min_relevance,
        deep_read_workers=args.deep_read_workers,
        pdf_ingestor=pdf_ingestor,
    )

    loaded = store.get_batch(batch.batch_id)
    report_data = None

    def report_json_data() -> dict:
        nonlocal report_data
        if report_data is None:
            report_data = render_batch_report_json(store, loaded.batch_id)
        return report_data

    print(f"Natural query: {args.query}")
    print("Natural query route: scholarly")
    print(f"Batch ID: {loaded.batch_id}")
    if discovery_error:
        print(f"Discovery error: {discovery_error}")
    content = _format_report(
        args.format,
        text=lambda: render_batch_report(store, loaded.batch_id),
        markdown=lambda: render_batch_report_markdown(store, loaded.batch_id),
        json_data=report_json_data,
    )
    _emit_output(content, args.output, label="report")
    if args.passport:
        write_json_artifact(Path(args.passport), build_batch_passport(store, loaded.batch_id, data_dir=data_dir))
        print(f"Wrote passport: {args.passport}")
    if args.rejection_log:
        write_json_artifact(Path(args.rejection_log), build_rejection_log(store, loaded.batch_id))
        print(f"Wrote rejection log: {args.rejection_log}")
    if args.write:
        payload = build_writing_payload(report_json_data(), mode=args.write_mode)
        print(f"Writing mode: {args.write_mode}")
        _emit_writing_output(payload, args.write_format, args.write_output)
    return 0


def _handle_scan(
    args: argparse.Namespace,
    store: FridayStore,
    data_dir: Path,
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
) -> int:
    if args.query or args.resume_batch:
        if args.resume_batch:
            try:
                batch = store.get_batch(args.resume_batch)
            except KeyError:
                print(f"Unknown batch id: {args.resume_batch}")
                return 1
            if batch.mode != "query":
                print(f"Batch is not a query batch: {args.resume_batch}")
                return 1
            query = args.query or batch.query
            if not query:
                print("--resume-batch requires a stored query batch or a new --query")
                return 2
            effective_limit = args.limit or batch.limit or 100
            store.update_batch_limit(batch.batch_id, effective_limit)
            resumed = True
        else:
            query = args.query
            effective_limit = args.limit or 100
            batch = store.create_batch(query=query, limit=effective_limit, mode="query")
            resumed = False

        discovery_error = None
        try:
            candidates = _call_discoverer(
                discoverer,
                query,
                effective_limit,
                page_size=args.page_size,
                request_delay_seconds=args.request_delay,
            )
        except Exception as exc:
            candidates = []
            discovery_error = f"{type(exc).__name__}: {exc}"
        candidates = rank_candidates(query, candidates)
        for candidate in candidates:
            decision = evaluate_source(candidate.source_for_gate)
            store.add_batch_item_if_new(
                batch.batch_id,
                candidate.source_for_gate,
                decision,
                candidate=candidate,
            )

        _deep_read_ranked_batch_items(
            store,
            data_dir,
            batch.batch_id,
            deep_read_limit=args.deep_read_limit,
            min_relevance=args.min_relevance,
            deep_read_workers=args.deep_read_workers,
            pdf_ingestor=pdf_ingestor,
        )
        loaded = store.get_batch(batch.batch_id)
        print(f"Batch ID: {batch.batch_id}")
        if resumed:
            print(f"Resumed: {batch.batch_id}")
        print(f"Query: {query}")
        print(f"Limit: {effective_limit}")
        print(f"Page size: {max(1, min(args.page_size, 200))}")
        print(f"Screened: {loaded.screened_count}")
        print(f"Blocked: {loaded.blocked_count}")
        print(f"Allowed: {loaded.screened_count - loaded.blocked_count}")
        print(f"Deep-scanned: {loaded.deep_read_count}")
        if discovery_error:
            print(f"Discovery error: {discovery_error}")
        print(f"Report: friday report {batch.batch_id}")
        return 0

    if args.manifest:
        batch = store.create_batch(
            query=None,
            limit=None,
            mode="manifest_all_deep" if args.all_deep else "manifest",
            manifest_path=args.manifest,
        )
        for source in _read_manifest(Path(args.manifest)):
            decision = evaluate_source(source)
            store.add_batch_item(
                batch.batch_id,
                source,
                decision,
            )
            if args.all_deep and decision.allowed:
                pdf_ingestor(store, data_dir, batch.batch_id, source, None)
        loaded = store.get_batch(batch.batch_id)
        print(f"Batch ID: {loaded.batch_id}")
        print(f"Manifest: {args.manifest}")
        print(f"Screened: {loaded.screened_count}")
        print(f"Blocked: {loaded.blocked_count}")
        print(f"Allowed: {loaded.screened_count - loaded.blocked_count}")
        print(f"Deep-scanned: {loaded.deep_read_count}")
        print(f"Report: friday report {loaded.batch_id}")
        return 0

    if args.source:
        decision = evaluate_source(args.source)
        scan = store.create_scan(args.source, decision)
        print(f"Scan ID: {scan.scan_id}")
        print(f"Status: {'allowed' if scan.allowed else 'blocked'}")
        print(f"Reason: {scan.reason}")
        print(f"Report: friday report {scan.scan_id}")
        return 0

    print("scan requires a source, --query, or --manifest")
    return 2


def _handle_research(
    args: argparse.Namespace,
    store: FridayStore,
    data_dir: Path,
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
) -> int:
    if args.resume_batch:
        try:
            batch = store.get_batch(args.resume_batch)
        except KeyError:
            print(f"Unknown batch id: {args.resume_batch}")
            return 1
        if batch.mode != "query":
            print(f"Batch is not a query batch: {args.resume_batch}")
            return 1
        query = args.query or batch.query
        effective_limit = args.limit or batch.limit or 100
        store.update_batch_limit(batch.batch_id, effective_limit)
        resumed = True
    else:
        query = args.query
        effective_limit = args.limit or 100
        batch = store.create_batch(query=query, limit=effective_limit, mode="query")
        resumed = False

    discovery_error = None
    try:
        candidates = _call_discoverer(
            discoverer,
            query,
            effective_limit,
            page_size=args.page_size,
            request_delay_seconds=args.request_delay,
        )
    except Exception as exc:
        candidates = []
        discovery_error = f"{type(exc).__name__}: {exc}"

    candidates = rank_candidates(query, candidates)
    for candidate in candidates:
        decision = evaluate_source(candidate.source_for_gate)
        store.add_batch_item_if_new(
            batch.batch_id,
            candidate.source_for_gate,
            decision,
            candidate=candidate,
        )

    _deep_read_ranked_batch_items(
        store,
        data_dir,
        batch.batch_id,
        deep_read_limit=args.deep_read_limit,
        min_relevance=args.min_relevance,
        deep_read_workers=args.deep_read_workers,
        pdf_ingestor=pdf_ingestor,
    )

    loaded = store.get_batch(batch.batch_id)
    print(f"Batch ID: {loaded.batch_id}")
    if resumed:
        print(f"Resumed: {loaded.batch_id}")
    print(f"Query: {query}")
    print(f"Limit: {effective_limit}")
    print(f"Screened: {loaded.screened_count}")
    print(f"Blocked: {loaded.blocked_count}")
    print(f"Allowed: {loaded.screened_count - loaded.blocked_count}")
    print(f"Deep-scanned: {loaded.deep_read_count}")
    if discovery_error:
        print(f"Discovery error: {discovery_error}")

    report_content = _format_report(
        args.format,
        text=lambda: render_batch_report(store, loaded.batch_id),
        markdown=lambda: render_batch_report_markdown(store, loaded.batch_id),
        json_data=lambda: render_batch_report_json(store, loaded.batch_id),
    )
    _emit_output(report_content, args.output, label="report")

    if args.passport:
        write_json_artifact(Path(args.passport), build_batch_passport(store, loaded.batch_id, data_dir=data_dir))
        print(f"Wrote passport: {args.passport}")
    if args.rejection_log:
        write_json_artifact(Path(args.rejection_log), build_rejection_log(store, loaded.batch_id))
        print(f"Wrote rejection log: {args.rejection_log}")
    return 0


def _handle_smoke_run(
    args: argparse.Namespace,
    store: FridayStore,
    data_dir: Path,
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
    llm_label_client: object | None,
) -> int:
    query = " ".join(args.query).strip()
    if not query:
        print("smoke-run requires a query.")
        return 2

    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "smoke-runs" / _slugify_query(query)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _smoke_run_artifact_paths(output_dir, args.format)

    research_args = argparse.Namespace(
        query=[query],
        resume_run=None,
        latest=False,
        limit=args.limit,
        page_size=args.page_size,
        request_delay=args.request_delay,
        deep_read_limit=args.deep_read_limit,
        min_relevance=args.min_relevance,
        deep_read_workers=args.deep_read_workers,
        auto_label_provider=args.auto_label_provider,
        auto_label_model=args.auto_label_model,
        auto_label_min_confidence=args.auto_label_min_confidence,
        auto_label_api_base_url=args.auto_label_api_base_url,
        auto_label_api_key_env=args.auto_label_api_key_env,
        llm_review_limit=args.llm_review_limit,
        format=args.format,
        output=str(paths["report"]),
        passport=str(paths["passport"]),
        rejection_log=str(paths["rejection_log"]),
        run_summary=str(paths["run_summary"]),
    )
    code = _handle_research_run(
        research_args,
        store,
        data_dir,
        discoverer,
        pdf_ingestor,
        llm_label_client,
    )
    if code != 0:
        return code

    run = store.latest_research_run()
    if run is None or not run.batch_id:
        print("smoke-run could not find the completed research run.")
        return 1
    batch = store.get_batch(run.batch_id)
    items = store.list_batch_items(batch.batch_id)
    labels = store.list_screening_labels(batch.batch_id)

    review_rows = build_label_review_rows(
        items,
        labels,
        limit=max(0, args.limit),
    )
    write_json_artifact(
        paths["labels_review"],
        {
            "schema_version": "1.0",
            "artifact_type": "labels_review",
            "run_id": run.run_id,
            "batch_id": batch.batch_id,
            "query": batch.query,
            "row_count": len(review_rows),
            "rows": review_rows,
        },
    )
    print(f"Wrote labels review: {paths['labels_review']}")

    export_rows = build_label_export_rows(store, batch_ids=[batch.batch_id])
    export_content = render_label_export_jsonl(export_rows)
    paths["labels_export"].write_text((export_content.rstrip() + "\n") if export_content else "", encoding="utf-8")
    print(f"Wrote labels export: {paths['labels_export']}")

    label_eval = {
        "batch_id": batch.batch_id,
        "query": batch.query,
        **build_label_evaluation(items, labels),
    }
    write_json_artifact(paths["label_eval"], label_eval)
    print(f"Wrote label eval: {paths['label_eval']}")

    next_commands = [
        f"friday run-summary --run-id {run.run_id}",
        f"friday labels review --batch-id {batch.batch_id}",
        f"friday labels export --batch-id {batch.batch_id} --output {paths['labels_export']}",
        f"friday labels eval --batch-id {batch.batch_id}",
    ]
    manifest_artifacts = {key: str(path) for key, path in paths.items() if key != "manifest"}
    write_json_artifact(
        paths["manifest"],
        {
            "schema_version": "1.0",
            "artifact_type": "smoke_run_manifest",
            "query": query,
            "limit": args.limit,
            "deep_read_limit": args.deep_read_limit,
            "min_relevance": args.min_relevance,
            "run_id": run.run_id,
            "batch_id": batch.batch_id,
            "output_dir": str(output_dir),
            "artifacts": manifest_artifacts,
            "next_commands": next_commands,
        },
    )
    print(f"Wrote smoke manifest: {paths['manifest']}")
    print(f"Smoke run directory: {output_dir}")
    print("Next commands:")
    for command in next_commands:
        print(f"  {command}")
    return 0


def _handle_research_run(
    args: argparse.Namespace,
    store: FridayStore,
    data_dir: Path,
    discoverer: Callable[..., list[Candidate]],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
    llm_label_client: object | None,
) -> int:
    settings = load_settings(data_dir)
    resumed = False
    run = None
    discovery_error = None
    auto_label_applied = 0
    llm_reviewed = 0
    llm_review_queue = []
    query_arg = " ".join(args.query).strip() if isinstance(args.query, list) else (args.query or "")

    if args.latest and args.resume_run:
        print("research-run accepts either --latest or --resume-run, not both.")
        return 2

    if args.latest:
        run = store.latest_research_run()
        if run is None:
            print("No research runs found.")
            return 1
        if query_arg and query_arg != run.query:
            print("research-run --latest cannot change the stored query.")
            return 2
        query = run.query
        resumed = True
    elif args.resume_run:
        try:
            run = store.get_research_run(args.resume_run)
        except KeyError:
            print(f"Unknown research run id: {args.resume_run}")
            return 1
        if query_arg and query_arg != run.query:
            print("research-run --resume-run cannot change the stored query.")
            return 2
        query = run.query
        resumed = True
    else:
        query = query_arg
        if not query:
            print("research-run requires a query or --resume-run")
            return 2

    limit = args.limit if args.limit is not None else (run.limit if run else int(settings["research"]["limit"]))
    page_size = args.page_size if args.page_size is not None else int(settings["research"]["page_size"])
    request_delay = (
        args.request_delay if args.request_delay is not None else float(settings["research"]["request_delay"])
    )
    deep_read_limit = (
        args.deep_read_limit
        if args.deep_read_limit is not None
        else (run.deep_read_limit if run else int(settings["research"]["deep_read_limit"]))
    )
    min_relevance = (
        args.min_relevance
        if args.min_relevance is not None
        else (run.min_relevance if run else int(settings["research"]["min_relevance"]))
    )
    deep_read_workers = (
        args.deep_read_workers
        if args.deep_read_workers is not None
        else int(settings["research"]["deep_read_workers"])
    )
    auto_label_provider = args.auto_label_provider or (
        run.auto_label_provider if run else str(settings["auto_label"]["provider"])
    )
    auto_label_model = args.auto_label_model or str(settings["auto_label"]["model"])
    auto_label_min_confidence = (
        args.auto_label_min_confidence
        if args.auto_label_min_confidence is not None
        else float(settings["auto_label"]["min_confidence"])
    )
    auto_label_api_base_url = args.auto_label_api_base_url or str(settings["auto_label"]["api_base_url"])
    auto_label_api_key_env = args.auto_label_api_key_env or str(settings["auto_label"]["api_key_env"])
    llm_review_limit = (
        args.llm_review_limit if args.llm_review_limit is not None else (run.llm_review_limit if run else 0)
    )
    report_format = args.format or str(settings["report"]["format"])

    if auto_label_provider not in {"heuristic", "llm"}:
        print(f"unsupported auto-label provider: {auto_label_provider}")
        return 2

    try:
        if run is None:
            run = store.create_research_run(
                query=query,
                limit=limit,
                deep_read_limit=deep_read_limit,
                min_relevance=min_relevance,
                auto_label_provider=auto_label_provider,
                llm_review_limit=max(0, llm_review_limit),
            )
            batch = store.create_batch(query=query, limit=limit, mode="research_run")
            run = store.update_research_run(run.run_id, batch_id=batch.batch_id)
        else:
            if not run.batch_id:
                batch = store.create_batch(query=query, limit=limit, mode="research_run")
                run = store.update_research_run(run.run_id, batch_id=batch.batch_id)
            else:
                batch = store.get_batch(run.batch_id)
            store.update_batch_limit(batch.batch_id, limit)
            run = store.update_research_run(
                run.run_id,
                limit=limit,
                deep_read_limit=deep_read_limit,
                min_relevance=min_relevance,
                auto_label_provider=auto_label_provider,
                llm_review_limit=max(0, llm_review_limit),
                error=None,
            )

        loaded_batch = store.get_batch(run.batch_id)
        if loaded_batch.screened_count == 0:
            run = store.update_research_run(run.run_id, status="discovering", error=None)
            try:
                candidates = _call_discoverer(
                    discoverer,
                    query,
                    limit,
                    page_size=page_size,
                    request_delay_seconds=request_delay,
                )
            except Exception as exc:
                candidates = []
                discovery_error = f"{type(exc).__name__}: {exc}"
                store.update_research_run(run.run_id, error=discovery_error)
            candidates = rank_candidates(query, candidates)
            for candidate in candidates:
                decision = evaluate_source(candidate.source_for_gate)
                store.add_batch_item_if_new(
                    loaded_batch.batch_id,
                    candidate.source_for_gate,
                    decision,
                    candidate=candidate,
                )
            store.sync_research_run_counts(run.run_id)

        run = store.update_research_run(run.run_id, status="labeling")
        if bool(settings["auto_label"]["enabled"]):
            heuristic_result = auto_label_batch_items(
                store,
                loaded_batch.batch_id,
                query=query,
                limit=limit,
                apply=True,
                min_confidence=auto_label_min_confidence,
                provider="heuristic",
            )
            auto_label_applied += heuristic_result.applied_count
            if auto_label_provider == "llm" and llm_review_limit > 0:
                llm_review_queue = build_llm_review_queue(
                    store.list_batch_items(loaded_batch.batch_id),
                    store.list_screening_labels(loaded_batch.batch_id),
                    limit=llm_review_limit,
                )
                llm_result = auto_label_batch_items(
                    store,
                    loaded_batch.batch_id,
                    query=query,
                    limit=llm_review_limit,
                    apply=True,
                    min_confidence=auto_label_min_confidence,
                    provider="llm",
                    model=auto_label_model,
                    llm_client=llm_label_client,
                    api_base_url=auto_label_api_base_url,
                    api_key_env=auto_label_api_key_env,
                    review_queue=llm_review_queue,
                )
                auto_label_applied += llm_result.applied_count
                llm_reviewed = len(llm_result.decisions)

        run = store.update_research_run(run.run_id, status="deep_reading")
        _deep_read_ranked_batch_items(
            store,
            data_dir,
            loaded_batch.batch_id,
            deep_read_limit=deep_read_limit,
            min_relevance=min_relevance,
            deep_read_workers=deep_read_workers,
            pdf_ingestor=pdf_ingestor,
        )
        store.sync_research_run_counts(run.run_id)

        run = store.update_research_run(run.run_id, status="reporting")
        loaded_batch = store.get_batch(loaded_batch.batch_id)
        report_content = _format_report(
            report_format,
            text=lambda: render_batch_report(store, loaded_batch.batch_id),
            markdown=lambda: render_batch_report_markdown(store, loaded_batch.batch_id),
            json_data=lambda: render_batch_report_json(store, loaded_batch.batch_id),
        )
        run = store.update_research_run(run.run_id, status="complete")
        run = store.sync_research_run_counts(run.run_id)
    except Exception as exc:
        if run is not None:
            store.update_research_run(run.run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"Research run failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"Run ID: {run.run_id}")
    if resumed:
        print(f"Resumed run: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Batch ID: {loaded_batch.batch_id}")
    print(f"Query: {query}")
    print(f"Limit: {limit}")
    print(f"Screened: {run.screened_count}")
    print(f"Blocked: {run.blocked_count}")
    print(f"Allowed: {run.allowed_count}")
    print(f"Deep-scanned: {run.deep_read_count}")
    print(f"Auto-labeled: {auto_label_applied}")
    if llm_reviewed:
        print(f"LLM-reviewed: {llm_reviewed}")
    if discovery_error:
        print(f"Discovery error: {discovery_error}")

    default_paths = _default_research_run_artifact_paths(data_dir, run.run_id, report_format)
    report_path = args.output or str(default_paths["report"])
    passport_path = args.passport or str(default_paths["passport"])
    rejection_log_path = args.rejection_log or str(default_paths["rejection_log"])
    run_summary_path = args.run_summary or str(default_paths["run_summary"])

    _emit_output(report_content, report_path, label="report")
    write_json_artifact(
        Path(passport_path),
        build_batch_passport(
            store,
            loaded_batch.batch_id,
            data_dir=data_dir,
            llm_review_queue=llm_review_queue,
        ),
    )
    print(f"Wrote passport: {passport_path}")
    write_json_artifact(Path(rejection_log_path), build_rejection_log(store, loaded_batch.batch_id))
    print(f"Wrote rejection log: {rejection_log_path}")
    write_json_artifact(
        Path(run_summary_path),
        build_research_run_summary(
            store,
            run.run_id,
            data_dir=data_dir,
            llm_review_queue=llm_review_queue,
        ),
    )
    print(f"Wrote run summary: {run_summary_path}")
    return 0


def _handle_review_queue(args: argparse.Namespace, store: FridayStore) -> int:
    batch_id = _batch_id_from_args(args, store, "review-queue")
    if batch_id is None:
        return 1
    try:
        batch = store.get_batch(batch_id)
        queue = build_llm_review_queue(
            store.list_batch_items(batch_id),
            store.list_screening_labels(batch_id),
            limit=args.limit,
        )
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1

    print(f"LLM review queue for batch: {batch.batch_id}")
    print(f"Query: {batch.query or '-'}")
    if not queue:
        print("No LLM review candidates found.")
        return 0
    for rank, entry in enumerate(queue, start=1):
        confidence = "-" if entry.confidence is None else f"{entry.confidence:.3f}"
        title = entry.item.title or entry.item.source
        print(
            f"rank={rank}\tscore={entry.score}\treason={entry.reason}\t"
            f"label={entry.label or '-'}\tconfidence={confidence}\t"
            f"relevance={entry.item.relevance_score or 0}\t"
            f"source={entry.item.source}\ttitle={title}"
        )
    return 0


def _handle_research_runs(store: FridayStore) -> int:
    runs = store.list_research_runs()
    if not runs:
        print("No research runs found.")
        return 0
    for run in runs:
        print(
            f"{run.run_id}\tstatus={run.status}\tquery={run.query}\t"
            f"screened={run.screened_count}\tdeep={run.deep_read_count}\t"
            f"blocked={run.blocked_count}\tbatch={run.batch_id or '-'}\tcreated={run.created_at}"
        )
    return 0


def _handle_eval_suite(args: argparse.Namespace) -> int:
    if args.action == "list":
        for suite in available_eval_suites():
            print(suite)
        return 0

    try:
        report = run_eval_suite(args.suite)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.format == "json":
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_eval_report_text(report))
    return 0 if report["status"] == "pass" else 1


def _handle_run_summary(args: argparse.Namespace, store: FridayStore) -> int:
    try:
        summary = build_run_summary_dashboard(
            store,
            latest=bool(args.latest),
            run_id=args.run_id,
            batch_id=args.batch_id,
            limit=args.limit,
        )
    except RunSummaryTargetError as exc:
        print(str(exc))
        return 1
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    if args.format == "json":
        print(json.dumps(summary, sort_keys=True))
    else:
        print(render_run_summary_text(summary))
    return 0


def _handle_batches(store: FridayStore) -> int:
    batches = store.list_batches()
    if not batches:
        print("No batches found.")
        return 0
    for batch in batches:
        label = batch.query or batch.manifest_path or batch.mode
        print(
            f"{batch.batch_id}\t{label}\tscreened={batch.screened_count}\t"
            f"deep={batch.deep_read_count}\tblocked={batch.blocked_count}"
        )
    return 0


def _handle_scans(store: FridayStore) -> int:
    scans = store.list_scans()
    if not scans:
        print("No scans found.")
        return 0
    for scan in scans:
        status = "allowed" if scan.allowed else "blocked"
        print(f"{scan.scan_id}\t{status}\t{scan.source}\t{scan.reason}")
    return 0


def _handle_label(args: argparse.Namespace, store: FridayStore) -> int:
    batch_id = _batch_id_from_args(args, store, "label")
    if batch_id is None:
        return 1
    try:
        label = store.set_screening_label(batch_id, args.source, args.label, note=args.note)
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"Labeled: {label.label}")
    print(f"Batch ID: {label.batch_id}")
    print(f"Source: {label.source}")
    if label.note:
        print(f"Note: {label.note}")
    return 0


def _handle_labels(args: argparse.Namespace, store: FridayStore) -> int:
    if args.action == "eval":
        return _handle_labels_eval(args, store)
    if args.action == "export":
        return _handle_labels_export(args, store)
    if args.action == "review":
        return _handle_labels_review(args, store)
    if args.action == "set":
        return _handle_labels_set(args, store)
    if getattr(args, "all", False):
        print("labels --all is only valid with labels export.")
        return 2
    batch_id = _batch_id_from_args(args, store, "labels")
    if batch_id is None:
        return 1
    try:
        labels = store.list_screening_labels(batch_id)
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    counts = {choice: 0 for choice in SCREENING_LABEL_CHOICES}
    for label in labels:
        counts[label.label] = counts.get(label.label, 0) + 1

    print(f"Labels for batch: {batch_id}")
    print(" ".join(f"{choice}={counts.get(choice, 0)}" for choice in SCREENING_LABEL_CHOICES))
    if labels:
        for label in labels:
            note = f"\tnote={label.note}" if label.note else ""
            print(f"{label.label}\t{label.source}{note}")
    else:
        print("No labels found.")

    if args.recommend:
        items = store.list_batch_items(batch_id)
        recommendations = recommend_unlabeled_items(items, labels, limit=args.limit)
        print("Recommendations:")
        if not recommendations:
            print("No unlabeled recommendations found.")
        for recommendation in recommendations:
            relevant_terms = ",".join(recommendation.relevant_overlap[:5]) or "-"
            irrelevant_terms = ",".join(recommendation.irrelevant_overlap[:5]) or "-"
            title = recommendation.item.title or recommendation.item.source
            print(
                f"score={recommendation.score}\tbase={recommendation.base_score}\t"
                f"{recommendation.item.source}\t{title}\t"
                f"relevant_terms={relevant_terms}\tirrelevant_terms={irrelevant_terms}"
            )
    return 0


def _handle_labels_export(args: argparse.Namespace, store: FridayStore) -> int:
    if args.format not in {"jsonl", "csv"}:
        print("labels export requires --format jsonl or --format csv.")
        return 2
    try:
        batch_ids = None if args.all else [_batch_id_from_args(args, store, "labels export")]
        if batch_ids is not None and batch_ids[0] is None:
            return 1
        rows = build_label_export_rows(store, batch_ids=batch_ids)
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    content = render_label_export_csv(rows) if args.format == "csv" else render_label_export_jsonl(rows)
    _emit_output(content, args.output, label="label export")
    print(f"Rows: {len(rows)}")
    return 0


def _handle_labels_eval(args: argparse.Namespace, store: FridayStore) -> int:
    if getattr(args, "all", False):
        print("labels eval requires --batch-id or --latest, not --all.")
        return 2
    batch_id = _batch_id_from_args(args, store, "labels eval")
    if batch_id is None:
        return 1
    try:
        batch = store.get_batch(batch_id)
        report = build_label_evaluation(
            store.list_batch_items(batch_id),
            store.list_screening_labels(batch_id),
        )
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    report = {
        "batch_id": batch.batch_id,
        "query": batch.query,
        **report,
    }
    if args.format == "json":
        print(json.dumps(report, sort_keys=True))
        return 0

    counts = report["human_label_counts"]
    print(f"Label evaluation for batch: {batch.batch_id}")
    print(f"Query: {batch.query or '-'}")
    print(
        "Human labels: "
        + " ".join(f"{choice}={counts.get(choice, 0)}" for choice in SCREENING_LABEL_CHOICES)
    )
    print(f"Comparable overrides: {report['comparable_count']}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    precision = report["precision"]
    recall = report["recall"]
    print("Precision: " + " ".join(f"{choice}={precision.get(choice, 0.0):.3f}" for choice in SCREENING_LABEL_CHOICES))
    print("Recall: " + " ".join(f"{choice}={recall.get(choice, 0.0):.3f}" for choice in SCREENING_LABEL_CHOICES))
    for recommendation in report["recommendations"]:
        confidence = recommendation["confidence"]
        confidence_text = "-" if confidence is None else f"{confidence:.3f}"
        print(f"Recommendation: {recommendation['type']} confidence={confidence_text} {recommendation['message']}")
    if report["disagreements"]:
        print("Disagreements:")
        for row in report["disagreements"][: args.limit]:
            confidence = row["agent_confidence"]
            confidence_text = "-" if confidence is None else f"{confidence:.3f}"
            title = row["title"] or row["source"]
            print(
                f"human={row['human_label']}\tagent={row['agent_label']}\t"
                f"confidence={confidence_text}\trelevance={row['relevance_score']}\t"
                f"title={title}\tsource={row['source']}"
            )
    else:
        print("No disagreements found.")
    return 0


def _handle_labels_review(args: argparse.Namespace, store: FridayStore) -> int:
    if getattr(args, "all", False):
        print("labels review requires --batch-id or --latest, not --all.")
        return 2
    batch_id = _batch_id_from_args(args, store, "labels review")
    if batch_id is None:
        return 1
    try:
        batch = store.get_batch(batch_id)
        rows = build_label_review_rows(
            store.list_batch_items(batch_id),
            store.list_screening_labels(batch_id),
            only=args.only,
            min_relevance=args.min_relevance,
            limit=args.limit,
        )
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    print(f"Label review for batch: {batch.batch_id}")
    print(f"Query: {batch.query or '-'}")
    print(f"Filters: only={args.only or '-'} min_relevance={args.min_relevance} limit={args.limit}")
    if not rows:
        print("No label review rows found.")
        return 0
    for rank, row in enumerate(rows, start=1):
        confidence = "-" if row["confidence"] is None else f"{row['confidence']:.3f}"
        label = row["label"] or "-"
        queue_reason = row["review_queue_reason"] or "-"
        queue_score = "-" if row["review_queue_score"] is None else row["review_queue_score"]
        print(
            f"rank={rank}\tlabel={label}\tsource_label={row['label_source']}\t"
            f"confidence={confidence}\trelevance={row['relevance_score']}\t"
            f"queue_reason={queue_reason}\tqueue_score={queue_score}\t"
            f"title={row['title']}\tsource={row['source']}"
        )
    return 0


def _handle_labels_set(args: argparse.Namespace, store: FridayStore) -> int:
    if getattr(args, "all", False):
        print("labels set requires --batch-id or --latest, not --all.")
        return 2
    if not args.source or not args.label:
        print("labels set requires --source and --label")
        return 2
    batch_id = _batch_id_from_args(args, store, "labels set")
    if batch_id is None:
        return 1
    try:
        existing_label = _existing_screening_label_for_source(store, batch_id, args.source)
        signals = _previous_agent_signals(existing_label)
        label = store.set_screening_label(
            batch_id,
            args.source,
            args.label,
            note=args.note,
            label_source="human",
            signals=signals,
        )
    except KeyError as exc:
        print(_clean_exception_message(exc))
        return 1
    except ValueError as exc:
        print(str(exc))
        return 2
    print(f"Labeled: {label.label}")
    print(f"Batch ID: {label.batch_id}")
    print(f"Source: {label.source}")
    if label.note:
        print(f"Note: {label.note}")
    return 0


def _existing_screening_label_for_source(store: FridayStore, batch_id: str, source: str):
    lookup = source.strip().lower()
    for label in store.list_screening_labels(batch_id):
        if label.source == source or label.normalized == lookup:
            return label
    return None


def _previous_agent_signals(label) -> str | None:
    if label is None or label.label_source != "agent":
        return None
    parts = [f"previous_agent_label={label.label}"]
    if label.confidence is not None:
        parts.append(f"previous_agent_confidence={label.confidence:.3f}")
    if label.rationale:
        parts.append(f"previous_agent_rationale={_compact_signal_value(label.rationale)}")
    if label.signals:
        parts.append(f"previous_agent_signals={_compact_signal_value(label.signals)}")
    return ";".join(parts)


def _compact_signal_value(value: str) -> str:
    return value.replace(";", ",").replace("\t", " ").replace("\n", " ").strip()


def _handle_auto_label(
    args: argparse.Namespace,
    store: FridayStore,
    data_dir: Path,
    llm_label_client: object | None,
) -> int:
    batch_id = _batch_id_from_args(args, store, "auto-label")
    if batch_id is None:
        return 1
    settings = load_settings(data_dir)
    provider = args.provider or str(settings["auto_label"]["provider"])
    model = args.model or str(settings["auto_label"]["model"])
    api_base_url = args.api_base_url or str(settings["auto_label"]["api_base_url"])
    api_key_env = args.api_key_env or str(settings["auto_label"]["api_key_env"])
    try:
        batch = store.get_batch(batch_id)
        result = auto_label_batch_items(
            store,
            batch_id,
            query=batch.query,
            limit=args.limit,
            apply=bool(args.apply),
            min_confidence=args.min_confidence,
            provider=provider,
            model=model,
            llm_client=llm_label_client,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
        )
    except (KeyError, ValueError) as exc:
        print(_clean_exception_message(exc))
        return 1
    mode = "apply" if args.apply else "dry-run"
    print(f"Auto-label mode: {mode}")
    print(f"Provider: {provider}")
    if provider == "llm":
        print(f"Model: {model}")
    print(f"Batch ID: {batch_id}")
    print(f"Decisions: {len(result.decisions)}")
    print(f"Applied: {result.applied_count}")
    print(f"Skipped human labels: {result.skipped_human_count}")
    print(f"Skipped low confidence: {result.skipped_low_confidence_count}")
    print(f"Skipped errors: {result.skipped_error_count}")
    for error in result.errors[:3]:
        print(f"Auto-label error: {error}")
    for decision in result.decisions[:20]:
        print(
            f"{decision.label}\tconfidence={decision.confidence:.3f}\t"
            f"{decision.item.source}\t{decision.rationale}\t{decision.signals}"
        )
    return 0


def _handle_settings(args: argparse.Namespace, data_dir: Path) -> int:
    if args.action == "set":
        if not args.key or args.value is None:
            print("settings set requires a key and value")
            return 2
        try:
            settings = set_setting(data_dir, args.key, args.value)
        except (KeyError, ValueError) as exc:
            print(str(exc).strip("'"))
            return 2
    else:
        settings = load_settings(data_dir)
    print("Friday settings")
    print("")
    for key, value in flatten_settings(settings):
        print(f"{key}: {_setting_text(value)}")
    return 0


def _handle_report(args: argparse.Namespace, store: FridayStore) -> int:
    target_id = args.target_id
    if args.latest:
        latest = store.latest_batch()
        if latest is None:
            print("No batches found.")
            return 1
        target_id = latest.batch_id

    if not target_id:
        print("report requires a scan id, batch id, or --latest")
        return 2

    if target_id.startswith("scan_"):
        content = _format_report(
            args.format,
            text=lambda: render_scan_report(store, target_id),
            markdown=lambda: render_scan_report_markdown(store, target_id),
            json_data=lambda: render_scan_report_json(store, target_id),
        )
        _emit_output(content, args.output, label="report")
        return 0
    if target_id.startswith("batch_"):
        content = _format_report(
            args.format,
            text=lambda: render_batch_report(store, target_id),
            markdown=lambda: render_batch_report_markdown(store, target_id),
            json_data=lambda: render_batch_report_json(store, target_id),
        )
        _emit_output(content, args.output, label="report")
        if args.passport:
            write_json_artifact(Path(args.passport), build_batch_passport(store, target_id))
            print(f"Wrote passport: {args.passport}")
        if args.rejection_log:
            write_json_artifact(Path(args.rejection_log), build_rejection_log(store, target_id))
            print(f"Wrote rejection log: {args.rejection_log}")
        return 0

    print(f"Unknown report target: {target_id}")
    return 2


def _handle_import_corpus(args: argparse.Namespace) -> int:
    if args.folder:
        corpus, rejection_log = import_folder_corpus(Path(args.folder))
    elif args.zotero_json:
        corpus, rejection_log = import_zotero_corpus(Path(args.zotero_json))
    elif args.obsidian_vault:
        corpus, rejection_log = import_obsidian_corpus(Path(args.obsidian_vault))
    else:
        print("import-corpus requires --folder, --zotero-json, or --obsidian-vault")
        return 2
    write_corpus_outputs(Path(args.output), Path(args.rejection_log), corpus, rejection_log)
    print(f"Wrote corpus: {args.output}")
    print(f"Wrote rejection log: {args.rejection_log}")
    return 0


def _handle_write(args: argparse.Namespace, store: FridayStore) -> int:
    if args.report:
        report_data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    else:
        batch_id = args.batch_id
        if args.latest:
            latest = store.latest_batch()
            if latest is None:
                print("No batches found.")
                return 1
            batch_id = latest.batch_id
        if not batch_id:
            print("write requires --batch-id, --latest, or --report")
            return 2
        report_data = render_batch_report_json(store, batch_id)

    if report_data.get("report_type") != "batch":
        print("write requires a batch report JSON or batch id.")
        return 2

    payload = build_writing_payload(report_data, mode=args.mode)
    if args.format == "package":
        if not args.output:
            print("write --format package requires --output directory.")
            return 2
    _emit_writing_output(payload, args.format, args.output)
    return 0


def _handle_compose(args: argparse.Namespace) -> int:
    try:
        files = build_compose_package_files(Path(args.package_dir), section=args.section)
    except ComposePackageError as exc:
        print(f"Compose package error: {exc}")
        return 2
    _emit_package_output(files, Path(args.output))
    print(f"Wrote compose package: {args.output}")
    return 0


def _format_report(
    output_format: str,
    *,
    text: Callable[[], str],
    markdown: Callable[[], str],
    json_data: Callable[[], dict],
) -> str:
    if output_format == "json":
        return json.dumps(json_data(), indent=2, sort_keys=True)
    if output_format == "markdown":
        return markdown()
    return text()


def _format_corpus_report(output_format: str, route: CorpusRouteResult) -> str:
    if output_format == "json":
        return json.dumps(_corpus_report_json(route), indent=2, sort_keys=True)
    if output_format == "markdown":
        return _corpus_report_markdown(route)
    return _corpus_report_text(route)


def _corpus_report_json(route: CorpusRouteResult) -> dict:
    return {
        "report_type": "corpus",
        "route": "corpus",
        "query": route.query,
        "loaded_count": route.loaded_count,
        "matched_count": len(route.matches),
        "corpus_paths": route.corpus_paths,
        "rejected_paths": route.rejected_paths,
        "matches": [
            {
                "rank": index,
                "title": match.title,
                "score": match.score,
                "matched_terms": list(match.matched_terms),
                "citation_key": match.citation_key,
                "source_pointer": match.source_pointer,
                "source_type": match.entry.get("source_type"),
                "venue": match.entry.get("venue"),
                "year": match.entry.get("year"),
                "doi": match.entry.get("doi"),
                "abstract": match.entry.get("abstract"),
                "corpus_path": match.corpus_path,
            }
            for index, match in enumerate(route.matches, start=1)
        ],
    }


def _corpus_report_markdown(route: CorpusRouteResult) -> str:
    lines = [
        "# Friday Corpus Report",
        "",
        f"Query: {route.query}",
        f"Route: corpus",
        f"Loaded corpus entries: {route.loaded_count}",
        f"Matched entries: {len(route.matches)}",
        "",
        "## Matches",
        "",
    ]
    for index, match in enumerate(route.matches, start=1):
        terms = ", ".join(match.matched_terms) or "-"
        lines.extend(
            [
                f"### {index}. {match.title}",
                "",
                f"- Score: {match.score}",
                f"- Matched terms: {terms}",
                f"- Citation key: {match.citation_key}",
                f"- Source: {match.source_pointer}",
            ]
        )
        venue = match.entry.get("venue")
        year = match.entry.get("year")
        if venue:
            lines.append(f"- Venue: {venue}")
        if year:
            lines.append(f"- Year: {year}")
        abstract = match.entry.get("abstract")
        if abstract:
            lines.extend(["", str(abstract)])
        lines.append("")
    if route.rejected_paths:
        lines.extend(["## Corpus Path Warnings", ""])
        for rejection in route.rejected_paths:
            lines.append(f"- {rejection['path']}: {rejection['reason']} ({rejection['detail']})")
    return "\n".join(lines).rstrip()


def _corpus_report_text(route: CorpusRouteResult) -> str:
    lines = [
        "Friday Corpus Report",
        f"Query: {route.query}",
        "Route: corpus",
        f"Loaded corpus entries: {route.loaded_count}",
        f"Matched entries: {len(route.matches)}",
    ]
    for index, match in enumerate(route.matches, start=1):
        terms = ", ".join(match.matched_terms) or "-"
        lines.append(
            f"{index}. score={match.score} terms={terms} title={match.title} source={match.source_pointer}"
        )
    return "\n".join(lines)


def _emit_output(content: str, output_path: str | None, *, label: str) -> None:
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote {label}: {output_path}")
        return
    print(content)


def _emit_writing_output(payload: dict, output_format: str, output_path: str | None) -> None:
    if output_format == "package":
        if not output_path:
            raise ValueError("writing package output requires a directory")
        _emit_package_output(build_writing_package_files(payload), Path(output_path))
        print(f"Wrote writing package: {output_path}")
        return
    if output_format == "json":
        content = json.dumps(payload, indent=2, sort_keys=True)
    else:
        content = render_writing_markdown(payload)
    _emit_output(content, output_path, label="writing draft")


def _emit_package_output(files: dict[str, str | bytes], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        path = output_dir / filename
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _smoke_run_artifact_paths(output_dir: Path, report_format: str) -> dict[str, Path]:
    report_extension = _report_extension(report_format)
    return {
        "report": output_dir / f"report.{report_extension}",
        "passport": output_dir / "passport.json",
        "rejection_log": output_dir / "rejection-log.json",
        "run_summary": output_dir / "run-summary.json",
        "labels_review": output_dir / "labels-review.json",
        "labels_export": output_dir / "labels-export.jsonl",
        "label_eval": output_dir / "label-eval.json",
        "manifest": output_dir / "smoke-manifest.json",
    }


def _default_research_run_artifact_paths(data_dir: Path, run_id: str, report_format: str) -> dict[str, Path]:
    run_dir = data_dir / "runs" / run_id
    report_extension = _report_extension(report_format)
    return {
        "report": run_dir / f"report.{report_extension}",
        "passport": run_dir / "passport.json",
        "rejection_log": run_dir / "rejection-log.json",
        "run_summary": run_dir / "run-summary.json",
    }


def _report_extension(report_format: str) -> str:
    return {"json": "json", "markdown": "md", "text": "txt"}.get(report_format, "md")


def _slugify_query(query: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return slug[:80] or "query"


def _call_discoverer(
    discoverer: Callable[..., list[Candidate]],
    query: str,
    limit: int,
    *,
    page_size: int,
    request_delay_seconds: float,
) -> list[Candidate]:
    try:
        parameters = inspect.signature(discoverer).parameters
    except (TypeError, ValueError):
        return discoverer(query, limit)
    kwargs = {}
    if "page_size" in parameters:
        kwargs["page_size"] = page_size
    if "request_delay_seconds" in parameters:
        kwargs["request_delay_seconds"] = request_delay_seconds
    return discoverer(query, limit, **kwargs)


def _deep_read_ranked_batch_items(
    store: FridayStore,
    data_dir: Path,
    batch_id: str,
    *,
    deep_read_limit: int,
    min_relevance: int,
    deep_read_workers: int,
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
) -> None:
    if deep_read_limit <= 0:
        return

    attempted_sources = {artifact.source for artifact in store.list_pdf_artifacts(batch_id)}
    stored_count = store.get_batch(batch_id).deep_read_count
    candidates = [
        item
        for item in _ranked_deep_read_items(store, batch_id, min_relevance)
        if item.source not in attempted_sources
    ]
    workers = max(1, deep_read_workers)
    index = 0
    while stored_count < deep_read_limit and index < len(candidates):
        remaining_target = deep_read_limit - stored_count
        window_size = min(workers, remaining_target, len(candidates) - index)
        window = candidates[index : index + window_size]
        index += window_size
        stored_count += _run_deep_read_window(
            store,
            data_dir,
            batch_id,
            window,
            attempted_sources,
            pdf_ingestor,
        )


def _run_deep_read_window(
    store: FridayStore,
    data_dir: Path,
    batch_id: str,
    items: list[BatchItemRecord],
    attempted_sources: set[str],
    pdf_ingestor: Callable[
        [FridayStore, Path, str, str, Candidate | None],
        PdfIngestionResult,
    ],
) -> int:
    if len(items) == 1:
        item = items[0]
        result = pdf_ingestor(
            store,
            data_dir,
            batch_id,
            item.source,
            _candidate_from_batch_item(item),
        )
        attempted_sources.add(item.source)
        return 1 if result.status == "stored" else 0

    stored_count = 0
    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        futures = {
            executor.submit(
                pdf_ingestor,
                store,
                data_dir,
                batch_id,
                item.source,
                _candidate_from_batch_item(item),
            ): item
            for item in items
        }
        for future in as_completed(futures):
            item = futures[future]
            attempted_sources.add(item.source)
            result = future.result()
            if result.status == "stored":
                stored_count += 1
    return stored_count


def _ranked_deep_read_items(store: FridayStore, batch_id: str, min_relevance: int) -> list[BatchItemRecord]:
    return rank_deep_read_items(
        store.list_batch_items(batch_id),
        store.list_screening_labels(batch_id),
        min_relevance=min_relevance,
    )


def _candidate_from_batch_item(item: BatchItemRecord) -> Candidate:
    return Candidate(
        provider=item.provider or "stored",
        title=item.title or item.source,
        source_for_gate=item.source,
        url=item.url,
        pdf_url=None,
        doi=item.doi,
        pmid=item.pmid,
        pmcid=item.pmcid,
        arxiv_id=item.arxiv_id,
        year=item.year,
        abstract=item.abstract,
        relevance_score=item.relevance_score,
        relevance_reason=item.relevance_reason,
        query_variant=item.query_variant,
        query_intent=item.query_intent,
        acronym_expansions=item.acronym_expansions,
        journal=item.journal,
        concepts=item.concepts,
        mesh_terms=item.mesh_terms,
        oa_status=item.oa_status,
        open_access_pdf_url=item.open_access_pdf_url,
    )


def _read_manifest(path: Path) -> list[str]:
    sources: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," in sample:
            reader = csv.reader(handle)
            for row in reader:
                if row and row[0].strip() and not row[0].lower().startswith("source"):
                    sources.append(row[0].strip())
        else:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    sources.append(value)
    return sources


def _batch_id_from_args(args: argparse.Namespace, store: FridayStore, command_name: str) -> str | None:
    if getattr(args, "latest", False):
        latest = store.latest_batch()
        if latest is None:
            print("No batches found.")
            return None
        return latest.batch_id
    batch_id = getattr(args, "batch_id", None)
    if not batch_id:
        print(f"{command_name} requires --batch-id or --latest")
        return None
    return batch_id


def _clean_exception_message(exc: Exception) -> str:
    return str(exc).strip("'")


def _natural_writing_validation_error(args: argparse.Namespace) -> str | None:
    if not args.write:
        return None
    if args.write_mode not in MODE_CHOICES:
        return f"unsupported writing mode: {args.write_mode}"
    if args.write_format not in WRITING_FORMAT_CHOICES:
        return "natural --write-format requires markdown, json, or package."
    if args.write_format == "package" and not args.write_output:
        return "natural --write-format package requires --write-output directory."
    return None


def _split_data_dir_args(argv: Sequence[str]) -> tuple[str, list[str]]:
    args = list(argv)
    if len(args) >= 2 and args[0] == "--data-dir":
        return args[1], args[2:]
    return ".friday", args


def _parse_natural_language_args(argv: Sequence[str]) -> argparse.Namespace:
    data_dir, rest = _split_data_dir_args(argv)
    settings = load_settings(Path(data_dir))
    options = {
        "limit": int(settings["research"]["limit"]),
        "page_size": int(settings["research"]["page_size"]),
        "request_delay": float(settings["research"]["request_delay"]),
        "deep_read_limit": int(settings["research"]["deep_read_limit"]),
        "min_relevance": int(settings["research"]["min_relevance"]),
        "deep_read_workers": int(settings["research"]["deep_read_workers"]),
        "format": str(settings["report"]["format"]),
        "output": None,
        "passport": None,
        "rejection_log": None,
        "write": False,
        "write_mode": "literature-review",
        "write_format": "markdown",
        "write_output": None,
        "auto_label_enabled": bool(settings["auto_label"]["enabled"]),
        "auto_label_apply": bool(settings["auto_label"]["apply"]),
        "auto_label_limit": int(settings["auto_label"]["limit"]),
        "auto_label_min_confidence": float(settings["auto_label"]["min_confidence"]),
        "auto_label_provider": str(settings["auto_label"]["provider"]),
        "auto_label_model": str(settings["auto_label"]["model"]),
        "auto_label_api_base_url": str(settings["auto_label"]["api_base_url"]),
        "auto_label_api_key_env": str(settings["auto_label"]["api_key_env"]),
        "corpus_paths": str(settings["corpus"]["paths"]),
        "corpus_min_score": int(settings["corpus"]["min_score"]),
        "corpus_min_matches": int(settings["corpus"]["min_matches"]),
        "corpus_limit": int(settings["corpus"]["limit"]),
    }
    value_options = {
        "--limit": ("limit", int),
        "--page-size": ("page_size", int),
        "--request-delay": ("request_delay", float),
        "--deep-read-limit": ("deep_read_limit", int),
        "--min-relevance": ("min_relevance", int),
        "--deep-read-workers": ("deep_read_workers", int),
        "--format": ("format", str),
        "--output": ("output", str),
        "--passport": ("passport", str),
        "--rejection-log": ("rejection_log", str),
        "--write-mode": ("write_mode", str),
        "--write-format": ("write_format", str),
        "--write-output": ("write_output", str),
        "--auto-label-limit": ("auto_label_limit", int),
        "--auto-label-min-confidence": ("auto_label_min_confidence", float),
        "--auto-label-provider": ("auto_label_provider", str),
        "--auto-label-model": ("auto_label_model", str),
        "--auto-label-api-base-url": ("auto_label_api_base_url", str),
        "--auto-label-api-key-env": ("auto_label_api_key_env", str),
        "--corpus-paths": ("corpus_paths", str),
        "--corpus-min-score": ("corpus_min_score", int),
        "--corpus-min-matches": ("corpus_min_matches", int),
        "--corpus-limit": ("corpus_limit", int),
    }
    query_parts: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "--no-auto-label":
            options["auto_label_enabled"] = False
            index += 1
            continue
        if token == "--auto-label":
            options["auto_label_enabled"] = True
            index += 1
            continue
        if token in {"--write", "--draft"}:
            options["write"] = True
            index += 1
            continue
        if token in value_options and index + 1 < len(rest):
            key, converter = value_options[token]
            options[key] = converter(rest[index + 1])
            if key.startswith("write_"):
                options["write"] = True
            index += 2
            continue
        query_parts.append(token)
        index += 1
    return argparse.Namespace(
        data_dir=data_dir,
        query=" ".join(query_parts).strip(),
        **options,
    )


def _setting_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
