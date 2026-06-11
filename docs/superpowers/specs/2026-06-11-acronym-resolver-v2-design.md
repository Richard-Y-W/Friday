# Acronym Resolver V2 Design

Date: 2026-06-11
Status: Approved for implementation

## Summary

Acronym handling should not be a chain of one-off conditionals in `query_planning.py`. Jarvis should detect acronym-like tokens generically, resolve known acronyms through an extensible local registry, and preserve unknown acronyms with explicit audit data instead of inventing meanings.

## Goals

- Detect acronym-like tokens in scholarly queries across domains.
- Resolve known acronyms from a structured registry with multiple possible senses.
- Disambiguate ambiguous acronyms from nearby context terms.
- Preserve unresolved acronyms and include them in `QueryPlan.resolved_acronyms` with `intent="unknown"` and `reason="unresolved_acronym"`.
- Keep existing AMR, MDR, MIC, AST, ESBL, CRE behavior compatible.
- Add non-biomedical coverage for ML/NLP acronyms such as CNN, SVM, LLM, NLP, and PCR.
- Keep the resolver offline and deterministic.

## Non-Goals

- No web lookup for acronyms.
- No LLM-based acronym expansion.
- No claim that every acronym is understood.
- No broad rewrite of discovery, PDF ingestion, or writing copilot behavior in this slice.

## Architecture

Create `jarvis_research/acronyms.py` as the registry and resolution boundary. `query_planning.py` will call the resolver and then build query expansions from the resolved records. Domain-specific query rewrites such as mathematical-linguistics natural prompt handling stay in `query_planning.py`, but acronym-specific meaning data moves to the registry.

Known acronym senses will include:

- Biomedical and clinical: AMR, AST, CRE, ESBL, MDR, MIC, PCR.
- NLP/AI/ML: AMR, CNN, LLM, NLP, SVM.
- Computational science: AMR.

Each sense defines an acronym, meaning, intent, context terms, and optional expansion templates. If a token is acronym-like but not in the registry, the resolver returns it as unresolved.

## Data Flow

1. Normalize the query.
2. Detect acronym tokens with a conservative pattern: uppercase tokens of length 2-8 containing at least two letters.
3. Resolve each token:
   - If the registry has one sense, resolve it directly.
   - If multiple senses exist, choose the first sense whose context terms appear in the query.
   - If no sense context matches, use the registry default sense.
   - If no registry entry exists, return an unresolved record.
4. Query planning determines dominant intent from resolved non-unknown records.
5. Query expansion replaces only known resolved acronyms. Unknown acronyms remain unchanged.
6. Gold eval and reports can inspect unresolved acronyms through the same `resolved_acronyms` field.

## Error Handling

The resolver must never fail a query because an acronym is unknown. Unknown acronyms are recorded with `meaning=<original acronym>`, `intent="unknown"`, `reason="unresolved_acronym"`, and no rejected meanings.

## Testing

- Unit tests for acronym detection and resolution.
- Query planning tests for ML, biomedical, ambiguous, and unknown acronyms.
- Gold eval cases for PCR, CNN, SVM, LLM, and unresolved acronyms.
- Full offline verification with `python3 -m unittest discover -v` and `python3 -m jarvis_research eval-suite run --suite gold`.
