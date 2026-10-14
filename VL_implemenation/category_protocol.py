"""Frozen configuration helpers for the seven-category ABC experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_CONFIG = PACKAGE_DIRECTORY / "abc_category_experiment.json"
DEFAULT_TAXONOMY_CONFIG = PACKAGE_DIRECTORY / "abc_segment_category_mapping.json"
DEFAULT_ASSIGNMENTS_PATH = PACKAGE_DIRECTORY / "abc_segment_assignments.jsonl"


def load_experiment_config(
    path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
) -> dict[str, Any]:
    """Load and validate the frozen category experiment configuration."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a JSON object: {resolved}")
    categories = payload.get("categories")
    if not isinstance(categories, dict) or len(categories) != 7:
        raise ValueError("Experiment config must contain exactly seven categories")
    template = payload.get("prompt_template")
    if not isinstance(template, str) or "{type}" not in template or "{examples}" not in template:
        raise ValueError("prompt_template must contain {type} and {examples}")
    _validate_against_taxonomy(categories)
    for category, details in categories.items():
        if not isinstance(details, dict):
            raise ValueError(f"Invalid category config: {category}")
        metrics = details.get("telemetry_metrics")
        if not isinstance(metrics, list) or not metrics:
            raise ValueError(f"{category} must define telemetry_metrics")
        if len(metrics) != len(set(metrics)):
            raise ValueError(f"{category} contains duplicate telemetry metrics")
        prompt = render_category_prompt(payload, category)
        if any(
            forbidden in prompt.casefold()
            for forbidden in ("not reaching", "not carrying", "task name")
        ):
            raise ValueError(f"{category} prompt contains a forbidden confounder")
    payload["_config_path"] = str(resolved)
    payload["_config_sha256"] = _sha256(resolved)
    payload["_taxonomy_sha256"] = _sha256(DEFAULT_TAXONOMY_CONFIG)
    return payload


def render_category_prompt(config: dict[str, Any], category: str) -> str:
    """Render the exact fixed prompt for one category."""

    categories = config.get("categories")
    if not isinstance(categories, dict) or category not in categories:
        raise KeyError(f"Unknown category: {category}")
    details = categories[category]
    if not isinstance(details, dict):
        raise ValueError(f"Invalid category config: {category}")
    prompt_type = str(details.get("type", "")).strip()
    examples = details.get("examples")
    if not prompt_type or not isinstance(examples, list) or not examples:
        raise ValueError(f"{category} requires type and examples")
    example_text = ", ".join(str(example) for example in examples)
    return str(config["prompt_template"]).format(
        type=prompt_type,
        examples=example_text,
    )


def selected_categories(
    config: dict[str, Any],
    requested: Iterable[str] | None = None,
) -> list[str]:
    """Return categories in frozen config order, optionally filtered."""

    categories = list(config["categories"])
    if requested is None:
        return categories
    requested_list = list(requested)
    unknown = sorted(set(requested_list) - set(categories))
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}")
    requested_set = set(requested_list)
    return [category for category in categories if category in requested_set]


def load_segment_assignments(
    path: str | Path = DEFAULT_ASSIGNMENTS_PATH,
) -> list[dict[str, Any]]:
    """Load the fixed per-segment assignment manifest."""

    resolved = Path(path).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 8045:
        raise ValueError(f"Expected 8045 segment assignments, found {len(rows)}")
    keys = {
        (str(row["bundle_id"]), int(row["segment_index"]))
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("Segment assignment manifest contains duplicate keys")
    return rows


def assignments_by_bundle(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Index assignments by bundle while preserving segment order."""

    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(str(row["bundle_id"]), []).append(row)
    for assignments in output.values():
        assignments.sort(key=lambda row: int(row["segment_index"]))
    return output


def _validate_against_taxonomy(categories: dict[str, Any]) -> None:
    taxonomy = json.loads(DEFAULT_TAXONOMY_CONFIG.read_text(encoding="utf-8"))
    taxonomy_categories = taxonomy.get("categories")
    if not isinstance(taxonomy_categories, dict):
        raise ValueError("Invalid ABC category taxonomy")
    if set(categories) != set(taxonomy_categories):
        raise ValueError("Experiment categories do not match the taxonomy")
    for category, taxonomy_details in taxonomy_categories.items():
        configured = categories[category]
        expected = taxonomy_details.get("labels")
        actual = configured.get("examples") if isinstance(configured, dict) else None
        if actual != expected:
            raise ValueError(
                f"{category} examples must exactly match the fixed taxonomy labels"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
