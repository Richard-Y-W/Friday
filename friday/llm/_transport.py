from __future__ import annotations

import json
from typing import Any, Callable
from urllib.request import Request, urlopen


# An ``opener`` matches the urllib.request.urlopen signature and is injectable so
# tests can supply a fake transport instead of touching the network.
Opener = Callable[..., Any]


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    opener: Opener = urlopen,
    timeout: float = 60.0,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with opener(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    opener: Opener = urlopen,
    timeout: float = 30.0,
) -> str:
    request = Request(url, headers=headers or {}, method="GET")
    with opener(request, timeout=timeout) as response:
        return response.read().decode("utf-8")
