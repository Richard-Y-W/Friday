from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, dict[str, object]] = {
    "research": {
        "limit": 100,
        "page_size": 200,
        "request_delay": 0.0,
        "deep_read_limit": 10,
        "min_relevance": 25,
        "deep_read_workers": 1,
    },
    "auto_label": {
        "enabled": True,
        "apply": True,
        "limit": 1000,
        "min_confidence": 0.0,
        "provider": "heuristic",
        "model": "gpt-5.5",
        "api_base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "corpus": {
        "paths": "",
        "min_score": 12,
        "min_matches": 1,
        "limit": 20,
    },
    "report": {
        "format": "markdown",
    },
}


def load_settings(data_dir: Path) -> dict[str, dict[str, object]]:
    path = _settings_path(data_dir)
    settings = _copy_defaults()
    if not path.exists():
        return settings
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return settings
    for section, values in loaded.items():
        if section not in settings or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in settings[section]:
                settings[section][key] = value
    return settings


def save_settings(data_dir: Path, settings: dict[str, dict[str, object]]) -> None:
    path = _settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_setting(data_dir: Path, key: str, value: str) -> dict[str, dict[str, object]]:
    section, name = _split_key(key)
    settings = load_settings(data_dir)
    if section not in DEFAULT_SETTINGS or name not in DEFAULT_SETTINGS[section]:
        raise KeyError(f"unknown setting: {key}")
    settings[section][name] = _coerce_value(DEFAULT_SETTINGS[section][name], value)
    save_settings(data_dir, settings)
    return settings


def flatten_settings(settings: dict[str, dict[str, object]]) -> list[tuple[str, object]]:
    rows = []
    for section in sorted(settings):
        for key in sorted(settings[section]):
            rows.append((f"{section}.{key}", settings[section][key]))
    return rows


def _settings_path(data_dir: Path) -> Path:
    return Path(data_dir) / "settings.json"


def _copy_defaults() -> dict[str, dict[str, object]]:
    return {section: dict(values) for section, values in DEFAULT_SETTINGS.items()}


def _split_key(key: str) -> tuple[str, str]:
    if "." not in key:
        raise KeyError(f"unknown setting: {key}")
    section, name = key.split(".", 1)
    return section, name


def _coerce_value(default: object, value: str) -> object:
    if isinstance(default, bool):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected boolean value, got: {value}")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
