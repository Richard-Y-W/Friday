# Friday — Implementation Progress

Tracks the build plan in [PLAN.md](PLAN.md). Phase numbers match §11 of the plan.

| Phase | Description | Status |
|---|---|---|
| — | Migrate JarvisResearch → Friday, rebrand, green baseline | ✅ done |
| — | Close leaked SQLite connections (`_connect` context manager) | ✅ done |
| **0** | LLM provider abstraction (`friday/llm/`) | ✅ done |
| **0.1** | Subscription-backed providers (`claude_cli`/`codex_cli`) + role config + `friday llm` | ✅ done |
| 1 | Sandbox the PDF parser | ◐ in progress |
| 2 | Evidence store + decompose/type | ◐ in progress |
| 3 | Discourse planner + claim/connective split | ◐ in progress |
| 4 | Style-aware composer + style packs (biomed first) | ◐ in progress |
| 4.1 | Executive-summary synthesis + quality gate | ◐ in progress |
| 4.2 | Reader-facing report polish: typography, sections, citations, plain-English synthesis | ◐ in progress |
| 5 | Tier A + Tier B faithfulness gate | ◐ in progress |
| 6 | Tier C critic panel (faithfulness + prose + structure) + revise loop | ◐ in progress |
| 7 | Trust score + verdict→action | ◐ in progress |
| 8 | Human feedback capture (`friday review-draft`) | ◐ in progress |
| 9 | Feedback flywheel (eval-gated) | ◐ in progress |
| 10 | Refactor `cli.py` monolith into a pipeline layer | ▢ ongoing |

## Phase 0 notes (done)

`friday/llm/` — a stdlib-only, dependency-free port of Tactician's provider/router design:

- `types.py` — `LLMRequest`, `LLMResponse`, `ProviderStatus`, `ModelConfig`, `Provider` protocol, `Role`.
- `parse.py` — `strip_markdown_fences` / `extract_json` (robust JSON from fenced/prose-wrapped model output).
- `providers/` — `OllamaProvider` (local, zero-token), `OpenAIProvider`, `AnthropicProvider`. HTTP transport is injectable (`opener`) so tests never touch the network.
- `router.py` — `ModelRouter`: role → provider+model, availability checks, and **graceful failure** (`success=False`, never raises) when a role is unconfigured or its provider is down. This is what lets later phases keep the "runs without an LLM" fallback.

**Default posture:** no roles configured → every `generate()` returns a structured failure → the deterministic spine is unaffected and zero-token. Local Ollama keeps style/critic work free; different providers per role serve the "independent model family" invariant (PLAN §6).

Tests: `tests/test_llm_parse.py`, `tests/test_llm_router.py`, `tests/test_llm_providers.py` (26 cases, no network, no SQLite).

## Phase 0.1 notes (done) — subscription providers, no API tokens

Friday's LLM leaves run against the user's **Claude / ChatGPT subscriptions** (the rolling usage window), never per-token API credits. New since Phase 0:

- `friday/llm/providers/claude_cli.py` — `ClaudeCliProvider`: shells out to the `claude` CLI in print mode (`claude -p --output-format json`), parses the `result` field. Prompt goes over **stdin** (no argv length/quoting limits). Built-in tools denied (`--disallowedTools`) per PLAN §12.
- `friday/llm/providers/codex_cli.py` — `CodexCliProvider`: `codex exec --sandbox read-only --output-last-message <file>`, the independent model family for verify/critique (§6).
- `friday/llm/_subprocess.py` — injectable `CommandRunner` (tests never spawn a real process) and `run_command`, which **strips `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` from the child env** — the hard guarantee that Friday can never silently fall back to token billing. Handles Windows `.cmd` shims via `cmd /c`.
- `friday/llm/config.py` — per-role wiring from settings. Default: composer→`claude_cli` (sonnet), verifier/critic→`codex_cli`, screener/extractor→`none` (high-volume work stays deterministic, §1).
- `friday/settings.py` — new `llm` section (`<role>_provider`/`<role>_model`).
- `friday llm status` / `friday llm test --role <role>` — inspect wiring + run one live generation to confirm subscription auth.

Tests: `tests/test_llm_cli_providers.py`, `tests/test_llm_config.py`, `LlmCommandTests` in `tests/test_cli.py`. Verified live: `friday llm test --role composer` generated on the Claude subscription with no API key set.

**Login:** `claude` is already authenticated (subscription). For the verifier/critic run `codex login` (ChatGPT subscription) when ready; until then those roles return a structured failure and the deterministic spine is unaffected.

## Phase 1 notes (in progress) — sandboxed PDF parser

`extract_pdf_text_pages` (`friday/pdf_ingestion.py`) now treats the PDF as
attacker-controlled input and parses it out-of-process under hard limits:

- **Scrubbed environment** — the parser inherits only a library/font allowlist
  (`_PARSER_ENV_ALLOWLIST`); Friday's API keys/tokens/network config never reach
  it. (`_scrubbed_parser_env`)
- **Wall-clock timeout** + **bounded page count** (`PDF_PARSE_MAX_PAGES`) +
  **capped output size** (`PDF_PARSE_MAX_OUTPUT_BYTES`, read via `_read_capped`)
  so a PDF that explodes into a huge text stream can't exhaust memory downstream.
- **POSIX memory/CPU rlimits** via `preexec_fn` (`RLIMIT_AS` + `RLIMIT_CPU`).
- **stdin closed** (`DEVNULL`) and crash → `RuntimeError` → caller records a
  blocked artifact.

Verified live: 13 pages extracted from `PLAN.pdf` through the hardened path.
Tests: `tests/test_pdf_sandbox.py`.

**Remaining for Phase 1:** hard memory caps on **Windows** (POSIX rlimits no-op
there) need a Job Object wrapper — the one platform-specific piece still open.
Network isolation is currently posture-only (pdftotext doesn't network, and the
env scrub removes proxy/credential vars); a namespace/Job-Object network block is
the stronger follow-up. Then Phase 2 (evidence store + `parse_confidence` +
decompose/type) can build on a trusted parse.

## Phase 2 note (in progress) — evidence store + claim decomposition/type

Initial implementation:

- PDF artifacts, PDF pages, and evidence records already persist parser quality,
  page parse confidence, evidence quality, trust labels, and trust scores.
- Full report packages now export `claim_units.json`.
- `claim_units.json` decomposes the final report body into typed units:
  `synthesis`, `result`, `method`, `limitation`, `factual`, and
  `material_gap`.
- Each claim unit carries its source section, source sentence, normalized text,
  citations, support status, evidence row IDs, evidence types, and minimum
  quality/parse/trust scores when those are available.
- Support statuses are explicit: `supported`, `weak_support`, `uncited`,
  `unknown_citation`, or `material_gap`.
- `report_manifest.json` now records claim-unit status, count, and issue count.
- `friday compose --section report` persists report claim units into SQLite
  (`report_claim_units`) keyed by the generated report package path.
- Tier C critic prompts and revision prompts now receive structured report claim
  units alongside the raw report and evidence rows.

Still open:

- Use claim units as the primary input for the independent semantic verifier,
  not just the critic/revision prompts.
- Add richer claim type taxonomy once the verifier needs it.

## Phase 3 note (in progress) — report-level discourse planning

Initial implementation:

- Full report packages now export `report_discourse_plan.json`.
- The plan groups clean, page-anchored atomic evidence rows by report section
  and rhetorical cluster (`Claims`, `Dataset and population`, `Methods`,
  `Findings`, `Limitations`).
- Report rendering consumes the plan to compose connected section paragraphs
  instead of dumping one raw evidence row per line.
- The full-report LLM composer receives the report discourse plan and
  deterministic report as its only rewrite context.
- Full report packages now export `report_plan_adherence_audit.json`.
- The plan-adherence audit checks typed claim units against planned evidence
  moves and flags missing, partial, unplanned, or out-of-order planned moves.
- Full-report LLM candidates and critic revisions are now rejected if they pass
  citation/prose/faithfulness checks but drop planned evidence moves.
- `report_manifest.json` records plan-adherence status, checked move count, and
  issue count; `report_trust_score.json` includes plan adherence as a blocking
  component.

Still open:

- Richer move ordering and synthesis beyond citation-level planned moves.
- More explicit connective classification inside claim units.
- Independent semantic verifier checks that consume the plan-adherence audit.

## Phase 4 note (in progress) — connected fallback composer

Initial implementation:

- Deterministic fallback reports now write short connected paragraphs from
  atomic evidence rows.
- Paragraphs use reader-facing transitions such as `The same paper also...` and
  `A second paper...` while keeping page anchors.
- The executive summary now selects the first cited sentence from a planned
  paragraph instead of trying to summarize the entire paragraph.
- Full-report LLM rewriting is gated: Friday accepts a model-polished report
  only when citation audit and prose-quality audit pass; otherwise it falls back
  to the deterministic report.

Still open:

- Discipline-specific style packs.
- Section-level style packs and a richer revise loop for failed prose.

## Phase 4.1 note (todo) — executive-summary synthesis + quality gate

The live `tell me about cancer` run showed that even when section-level LLM
drafts pass citation and verifier checks, the full-report executive summary can
still pull raw stitched evidence too directly. That creates long, awkward,
source-fragment summaries instead of a clean synthesis.

Fix:

- Generate the executive summary from verified section drafts and evidence
  clusters, not directly from raw evidence table rows.
- Add a summary-specific quality gate for overlong citation bundles, repeated
  citations, parse artifacts, stitched fragments, and table-like prose.
- If the summary fails, revise it or fall back to a short material-gap-aware
  summary rather than emitting raw evidence text.
- Keep the citation audit hard requirement: every factual summary sentence still
  needs supported paper/page anchors.

Initial implementation:

- Executive summaries now label sections explicitly (`Background`, `Methods`,
  `Results`, `Limitations`).
- Summary citations are rendered in reader-facing form while the audit still
  maps them back to internal `P1 p2` tokens.
- Long fallback summary sentences are trimmed rather than dumping entire raw
  evidence clusters.
- The full-report prose-quality audit blocks missing required report headings,
  oversized citation bundles, internal citation syntax, raw evidence-dump
  phrases, first-person source-paper voice, and known awkward generated phrases.

Still open:

- A real summary-specific quality gate that scores prose quality and triggers a
  revise/fallback path.
- Better synthesis across multiple papers instead of selecting the first
  supported row.

## Phase 4.2 note (todo) — reader-facing report polish

The `tell me about cancer` PDF also exposed a presentation and syntax problem:
the evidence can be technically cited while still being hard to read. The report
currently exposes internal citation syntax (`P1 p2`) too directly, uses plain
Markdown/PDF styling, and sometimes phrases synthesis as "Across N papers, X
evidence includes..." followed by raw source fragments. That is correct for an
audit table, but not acceptable for the main report body.

Fix:

- Improve PDF typography: readable font, cleaner spacing, stronger section
  hierarchy, and optional muted colors for headings/citation markers.
- Separate sections visually and structurally: Background, Dataset/Population,
  Methods, Results, Limitations, Evidence Table, Literature, and Citation Audit
  should scan as distinct parts of the report.
- Replace raw internal citation tokens in prose with reader-facing markers such
  as `[1, p. 2]`, footnotes, or short paper labels, while preserving the internal
  `P1 p2` audit mapping in machine-readable files.
- Rewrite synthesis phrasing into natural explanatory sentences, for example
  "One paper showed..." or "A breast- and lung-cancer imaging study found..."
  instead of "claim evidence includes...".
- Keep raw evidence snippets in evidence tables/appendices, not in the main
  narrative unless they are explicitly quoted and useful to the reader.
- Add tests that fail on report-body phrases like `evidence includes`, long
  citation bundles, duplicated page markers, and unstyled/ambiguous section
  boundaries.

Initial implementation:

- Full reports now use reader-facing citation syntax like `[1, p. 2]` in prose,
  while `citation_audit.json` preserves the internal `P1 p2` mapping.
- Main report sections are separated with visible dividers.
- The lightweight PDF renderer now uses a serif body font, bold heading font,
  and muted colored headings.
- Deterministic fallback reports prefer atomic evidence rows when available
  instead of raw aggregated `evidence includes` paragraphs.
- Regression tests cover citation syntax, section boundaries, PDF heading style,
  atomic-row fallback prose, and the `study's study` rewrite bug.
- Full report packages now export `report_prose_quality.json`; the manifest
  records `report_source`, citation status, and prose-quality status.
- Model-polished full reports export `report_llm_draft.md`,
  `report_composer_prompt.json`, and `report_composer_audit.json` so accepted
  and rejected rewrites are inspectable.

Still open:

- Better field-specific phrasing through style packs.

## Phase 5 note (in progress) — Tier A + Tier B faithfulness gate

Initial implementation:

- Full report packages now export `report_faithfulness_audit.json`.
- Tier A checks the final report body for uncited factual sentences, unknown
  citations, and material-gap lines that are not exact package gaps or known
  structural no-evidence gaps.
- Tier B checks cited final-report sentences against the package evidence text
  using citation-scoped term overlap. This catches cases where a model keeps a
  valid citation marker but attaches it to unsupported claims such as deployment
  or mortality benefit.
- The faithfulness audit now consumes typed report claim units and exports
  per-claim verdicts (`supported`, `weak`, `overstated`, `unsupported`, or
  `material_gap`) with citation, evidence-type, row-id, parse-confidence, and
  support-detail metadata.
- Tier B now has an explicit overstatement detector for high-risk phrasing such
  as `proved`, `clinically definitive`, `standard of care`, `mortality benefit`,
  causal claims, deployment claims, and practice-changing language when those
  terms are not present in the cited evidence text.
- Full report packages now export `report_semantic_faithfulness_audit.json`.
  When a verifier role is configured, Friday sends typed claim units, citation
  evidence, evidence rows, and the lexical faithfulness audit to the verifier
  and records per-claim semantic verdicts.
- Full-report LLM rewrites and critic revisions are rejected when the semantic
  verifier returns unsupported, overstated, causal-overreach, or citation-
  mismatch issues. If the verifier is unavailable, Friday does not block the
  deterministic spine, but trust stays at human-review rather than publishable.
- Full-report LLM rewrites are now accepted only when citation audit,
  prose-quality audit, lexical faithfulness audit, plan-adherence audit, and
  available semantic verifier all pass. If a required gate fails, Friday keeps
  the deterministic report and records the gate-specific reason in
  `report_composer_audit.json`.
- `report_manifest.json` records `faithfulness_status`,
  `faithfulness_tier_a_status`, `faithfulness_tier_b_status`, and
  `semantic_faithfulness_status`.
- Draft review queues now surface semantic verifier issues directly from
  `report_semantic_faithfulness_audit.json`.

Still open:

- Semantic verification depends on the configured verifier role, so skipped
  verifier runs remain human-review rather than publishable.
- The overstatement detector is rule-based and should be calibrated with real
  reports; it intentionally favors review over accepting broad clinical or
  causal claims.
- The gate needs more real-report calibration once we run additional live
  writing packages across biomedical, math, and general science topics.

## Phase 6 note (mostly done) — Tier C critic panel + revise loop

Implemented:

- Full-report LLM composition can now call the configured `critic` role after
  the candidate passes citation, prose-quality, and faithfulness gates.
- The single generic critic is now split into faithfulness, prose, and
  structure critics. Each critic gets a scoped prompt and writes its own audit:
  `report_faithfulness_critic_audit.json`,
  `report_prose_critic_audit.json`, and
  `report_structure_critic_audit.json`.
- The combined panel verdict is exported as `report_critic_panel_audit.json`.
  The legacy `report_critic_prompt.json` and `report_critic_audit.json` files
  remain available for older review/feedback commands.
- If the panel rejects a candidate, Friday asks the `composer` role for an
  evidence-bound revision. A second revision attempt is allowed when the first
  revision passes deterministic gates but is still rejected by the critic panel.
- Attempt-specific files are exported as
  `report_revision_1_critic_panel_audit.json`,
  `report_revision_2_critic_panel_audit.json`, etc., while
  `report_revision_audit.json` records the final attempt count and status.
- Draft-review queues now flatten nested critic-panel issues so humans can see
  whether faithfulness, prose, or structure caused the rejection.

Still open:

- More real-run calibration of critic prompts and thresholds.
- Feeding critic outcomes into durable style/evidence feedback proposals for
  future runs.

## Phase 7 note (in progress) — trust score + verdict action

Initial implementation:

- Full report packages now export `report_trust_score.json`.
- The trust score aggregates citation, prose-quality, faithfulness, Tier A,
  Tier B, composer, and critic status into a single score, verdict, and action.
- Verdicts are:
  - `publishable` / `publish` when required gates pass and the critic passes.
  - `needs_review` / `human_review` when required gates pass but no critic has
    approved the report.
  - `blocked` / `block` when required citation or faithfulness gates fail.
- `report_manifest.json` now records `trust_score`, `trust_verdict`, and
  `trust_action`.

Still open:

- Calibrate score thresholds against real report runs.
- Surface verdict/action in the CLI output and Jarvis interactive shell.
- Add Phase 8 human review capture so `needs_review` has a first-class workflow.

## Phase 8 note (in progress) — human draft feedback capture

Initial implementation:

- `friday review-draft --package <report-package>` now exports
  `draft_feedback.json` and `review_queue.md`.
- The review queue is auto-prefilled from `report_trust_score.json`,
  `report_manifest.json`, prose-quality, faithfulness, critic, revision, and
  citation audit artifacts.
- Human decisions can be captured directly with `--decision`, `--note`, and
  `--reviewer`, so package-level feedback becomes a durable artifact instead of
  a loose chat note.
- The command is package-local and does not create or require the Friday SQLite
  store.

Still open:

- Add an interactive shortcut such as `friday review latest`.
- Feed captured decisions into the Phase 9 feedback flywheel for thresholds,
  topic memory, and style-profile tuning.

## Phase 9 note (in progress) — feedback interpreter + tuning proposal

Initial implementation:

- Added a first-class `feedback` LLM role. The Codex profile uses
  `codex_cli`; the Claude profile uses `claude_cli`/`sonnet`, so the user can
  choose which subscription-backed interpreter processes feedback.
- Interactive `friday` report runs now ask a short post-report review flow:
  review now, decision, and note.
- Captured answers write `draft_feedback.json` and `review_queue.md`, then the
  selected `feedback` role converts that packet plus trust/audit artifacts into
  `tuning_proposal.json`, `tuning_proposal.md`, and
  `feedback_interpreter_prompt.json`.
- If the selected interpreter is unavailable, Friday keeps the raw feedback and
  clearly reports that no tuning proposal was generated.
- `friday feedback review --package <report-package>` prints a readable
  proposal summary.
- `friday feedback approve|reject --package <report-package>` writes
  `tuning_decision.json` and `tuning_decision.md`. Approval records
  `apply_status=not_applied`; rejection records `apply_status=not_applicable`.
- Interactive report review now asks whether to apply the tuning proposal if
  evals pass; answering yes writes an approved decision, runs the eval gates,
  and writes `tuning_apply.json` / `tuning_apply.md`.
- `friday feedback apply --package <report-package>` runs approved proposals
  through their required eval gates and applies only bounded local rule-store
  updates under `.friday/feedback/rules/`.
- Full report composition now consumes applied local prose-quality rules from
  `.friday/feedback/rules/prose_quality.json`; `add_blocked_phrase` rules
  become `feedback_blocked_phrase` prose-quality issues in future reports and
  can reject LLM report rewrites.

Still open:

- Durable benchmark export from approved/failing report packages.
- Feeding accepted proposals into live faithfulness, topic/ranking, and
  evidence-quality rules.
