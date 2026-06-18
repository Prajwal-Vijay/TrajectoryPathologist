"""Configuration loading for trajectory analysis runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")
BUNDLE_CONFIG_FILENAMES = ("pathologist_config.json",)


def load_pipeline_config(
    config_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load built-in defaults, then optional bundle and explicit overrides."""

    config = _load_json_object(DEFAULT_CONFIG_PATH)

    if bundle_path is not None:
        bundle_dir = Path(bundle_path).expanduser().resolve()
        for filename in BUNDLE_CONFIG_FILENAMES:
            candidate = bundle_dir / filename
            if candidate.exists():
                config = deep_merge(config, _load_json_object(candidate))

    if config_path is not None:
        config = deep_merge(config, _load_json_object(Path(config_path).expanduser().resolve()))

    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_section(config: dict[str, Any], *path: str) -> dict[str, Any]:
    """Return a nested config section, or an empty dict when absent."""

    current: Any = config
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def config_value(config: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    """Return a nested config value, or a default when absent."""

    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must contain an object: {path}")
    return payload
