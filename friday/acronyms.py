from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AcronymSense:
    acronym: str
    meaning: str
    intent: str
    context_terms: tuple[str, ...] = ()
    requires_context: bool = False
    emit: bool = True


@dataclass(frozen=True)
class AcronymResolution:
    acronym: str
    meaning: str
    intent: str
    reason: str
    rejected_meanings: tuple[str, ...] = ()


BIOMEDICAL_CONTEXT = (
    "assay",
    "antibiotic",
    "antifungal",
    "antimicrobial",
    "bacteria",
    "bacterial",
    "beta lactamase",
    "biomarker",
    "carbapenem",
    "clinical",
    "diagnostic",
    "drug resistance",
    "extended spectrum",
    "infection",
    "isolate",
    "isolates",
    "maldi",
    "maldi tof",
    "microbial",
    "pathogen",
    "pseudomonas",
    "resistance",
    "resistant",
    "sensitivity",
    "spectra",
    "spectrometry",
    "susceptibility",
    "surveillance",
)

NLP_CONTEXT = (
    "generation",
    "graph",
    "graphs",
    "language",
    "meaning representation",
    "natural language",
    "nlp",
    "parsing",
    "parser",
    "semantic",
    "semantics",
    "text",
)

COMPUTATIONAL_CONTEXT = (
    "adaptive",
    "finite",
    "grid",
    "mesh",
    "refinement",
    "simulation",
    "solver",
    "timestepping",
)

ML_CONTEXT = (
    "classification",
    "classifier",
    "computer vision",
    "deep learning",
    "feature",
    "features",
    "image",
    "learning",
    "machine learning",
    "model",
    "neural",
    "network",
    "prediction",
    "selection",
    "supervised",
)


ACRONYM_REGISTRY: dict[str, tuple[AcronymSense, ...]] = {
    "AMR": (
        AcronymSense("AMR", "antimicrobial resistance", "biomedical", BIOMEDICAL_CONTEXT),
        AcronymSense("AMR", "abstract meaning representation", "nlp", NLP_CONTEXT),
        AcronymSense("AMR", "adaptive mesh refinement", "computational", COMPUTATIONAL_CONTEXT),
    ),
    "AST": (
        AcronymSense(
            "AST",
            "antimicrobial susceptibility testing",
            "biomedical",
            BIOMEDICAL_CONTEXT,
            requires_context=True,
        ),
    ),
    "CNN": (
        AcronymSense("CNN", "convolutional neural network", "ml", ML_CONTEXT),
    ),
    "CRE": (
        AcronymSense("CRE", "carbapenem-resistant Enterobacteriaceae", "biomedical", BIOMEDICAL_CONTEXT),
    ),
    "ESBL": (
        AcronymSense("ESBL", "extended-spectrum beta-lactamase", "biomedical", BIOMEDICAL_CONTEXT),
    ),
    "LLM": (
        AcronymSense("LLM", "large language model", "nlp", NLP_CONTEXT + ML_CONTEXT),
    ),
    "MALDI": (
        AcronymSense(
            "MALDI",
            "matrix-assisted laser desorption/ionization",
            "biomedical",
            BIOMEDICAL_CONTEXT,
            emit=False,
        ),
    ),
    "MDR": (
        AcronymSense("MDR", "multidrug resistance", "biomedical", BIOMEDICAL_CONTEXT, requires_context=True),
    ),
    "MIC": (
        AcronymSense(
            "MIC",
            "minimum inhibitory concentration",
            "biomedical",
            BIOMEDICAL_CONTEXT,
            requires_context=True,
        ),
    ),
    "NLP": (
        AcronymSense("NLP", "natural language processing", "nlp", NLP_CONTEXT + ML_CONTEXT),
    ),
    "PCR": (
        AcronymSense("PCR", "polymerase chain reaction", "biomedical", BIOMEDICAL_CONTEXT),
    ),
    "SVM": (
        AcronymSense("SVM", "support vector machine", "ml", ML_CONTEXT),
    ),
    "TOF": (
        AcronymSense("TOF", "time of flight", "biomedical", BIOMEDICAL_CONTEXT, emit=False),
    ),
}


def detect_acronyms(query: str) -> tuple[str, ...]:
    acronyms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{1,7}\b", query):
        if sum(1 for char in token if char.isupper()) < 2:
            continue
        key = token.upper()
        if key in seen:
            continue
        seen.add(key)
        acronyms.append(token)
    return tuple(acronyms)


def resolve_acronyms(query: str) -> tuple[AcronymResolution, ...]:
    normalized = _normalize(query)
    resolutions: list[AcronymResolution] = []
    for acronym in detect_acronyms(query):
        registry_key = acronym.upper()
        senses = ACRONYM_REGISTRY.get(registry_key)
        if not senses:
            resolutions.append(_unresolved(acronym))
            continue
        emitted_senses = tuple(sense for sense in senses if sense.emit)
        if not emitted_senses:
            continue
        selected, reason = _select_sense(normalized, emitted_senses)
        if selected is None:
            resolutions.append(_unresolved(acronym))
            continue
        rejected = tuple(sense.meaning for sense in emitted_senses if sense.meaning != selected.meaning)
        resolutions.append(
            AcronymResolution(
                acronym=selected.acronym,
                meaning=selected.meaning,
                intent=selected.intent,
                reason=reason,
                rejected_meanings=rejected,
            )
        )
    return tuple(resolutions)


def _select_sense(normalized_query: str, senses: tuple[AcronymSense, ...]) -> tuple[AcronymSense | None, str]:
    if len(senses) == 1:
        sense = senses[0]
        if sense.requires_context and not _context_score(normalized_query, sense):
            return None, "unresolved_acronym"
        reason = "context_match" if sense.requires_context else "registry_single_sense"
        return sense, reason

    scored = [(sense, _context_score(normalized_query, sense)) for sense in senses]
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored[0][1] > 0:
        return scored[0][0], "context_match"
    return senses[0], "registry_default"


def _context_score(normalized_query: str, sense: AcronymSense) -> int:
    return sum(1 for term in sense.context_terms if _contains_normalized(normalized_query, term))


def _unresolved(acronym: str) -> AcronymResolution:
    return AcronymResolution(
        acronym=acronym,
        meaning=acronym,
        intent="unknown",
        reason="unresolved_acronym",
    )


def _contains_normalized(normalized_query: str, term: str) -> bool:
    normalized_term = _normalize(term)
    return f" {normalized_term} " in f" {normalized_query} "


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()
