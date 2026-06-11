# Jarvis Research

Jarvis Research is a scholarly-only literature scanner and evidence report generator. It searches academic indexes, screens results, reads safe open-access PDFs, extracts page-anchored evidence, and writes cited report packages.

It is not a general web crawler. By default, Jarvis blocks GitHub, code archives, supplementary artifacts, and arbitrary web downloads. The scanner is allowed to read papers, not execute or trust content from the papers.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/byungkim113/JarvisResearch.git
cd JarvisResearch
```

Run Jarvis directly from the repo:

```bash
python3 -m jarvis_research --help
python3 -m jarvis_research /settings
```

Start the interactive shell:

```bash
python3 -m jarvis_research
```

Inside the shell:

```text
jarvis> /settings
jarvis> tell me about MALDI AMR
jarvis> jarvis tell me about the importance of math in language
jarvis> /exit
```

## Install The `jarvis` Command

For day-to-day use, create a small launcher in `~/.local/bin`:

```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/jarvis <<EOF
#!/usr/bin/env bash
set -euo pipefail

REPO="$PWD"
export PYTHONPATH="\${REPO}\${PYTHONPATH:+:\${PYTHONPATH}}"

exec /usr/bin/env python3 -m jarvis_research "\$@"
EOF
chmod +x ~/.local/bin/jarvis
```

Make sure `~/.local/bin` is on your `PATH`. For zsh:

```bash
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
```

Open a new terminal, then run:

```bash
jarvis
```

## Reports

Interactive natural-language queries write report packages under:

```text
.jarvis/reports/<query-slug>-<timestamp>/
```

They also copy the reader-facing PDF to:

```text
~/Desktop/JarvisReports/<query-slug>-<timestamp>.pdf
```

Each package includes:

- `report.md`
- `report.pdf`
- `literature_table.csv`
- `evidence_table.csv`
- `citation_audit.json`
- `writing.json`
- source, screening, supported-paragraph, blocked-paragraph, and material-gap audit files

## Settings

Show current defaults:

```bash
jarvis /settings
```

Update a default:

```bash
jarvis /settings set research.limit 100
jarvis /settings set research.deep_read_limit 5
```

The same commands work inside the interactive shell:

```text
jarvis> /settings
jarvis> /settings set research.limit 100
```

## Token Use

The default auto-label provider is heuristic, so it does not use LLM tokens:

```text
auto_label.provider: heuristic
```

If you switch to an LLM provider, Jarvis will use the API key configured in `auto_label.api_key_env`, which defaults to `OPENAI_API_KEY`.

## Safety Model

Jarvis enforces a scholarly source policy:

- allowed: DOI, arXiv PDFs, PubMed/PMC, OpenAlex open-access locations, and known academic PDF hosts
- blocked by default: GitHub, code repositories, archives, drive links, arbitrary websites, and supplementary/code artifacts
- PDFs are treated as untrusted input; extracted text is evidence, not instructions

## Current Limit

The report writer is evidence-bound and citation-audited, but the quality of the prose still depends on PDF text extraction. Some publisher PDFs produce noisy text, so evidence cleanup remains the next major improvement area.
