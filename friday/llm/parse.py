from __future__ import annotations

import json
import re
from typing import Any, Optional


_FENCED = re.compile(r"^[\s\S]*?```[^\n]*\n([\s\S]*?)\n```\s*$")
_OPEN_FENCE = re.compile(r"^```[^\n]*\n")
_CLOSE_FENCE = re.compile(r"\n```\s*$")
_LEADING_NON_JSON = re.compile(r"^[^\[{]*")


def strip_markdown_fences(text: str) -> str:
    """Remove a surrounding ```code fence``` from model output.

    Handles leading prose before the fence, an optional language tag, and CRLF
    line endings. Returns the text unchanged when no fence is present.
    """
    normalized = text.replace("\r\n", "\n")
    match = _FENCED.match(normalized)
    if match:
        return match.group(1)
    return _CLOSE_FENCE.sub("", _OPEN_FENCE.sub("", normalized)).strip()


def extract_json(text: str) -> Optional[Any]:
    """Best-effort parse of a JSON value from messy model output.

    Strips fences, then tries both the stripped text and the text with any
    leading prose removed. Returns ``None`` on failure rather than raising, so
    callers can treat an unparseable response as a soft failure.
    """
    stripped = strip_markdown_fences(text)
    candidates = [stripped, _LEADING_NON_JSON.sub("", stripped, count=1)]
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None
