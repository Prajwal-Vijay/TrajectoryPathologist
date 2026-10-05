"""Midpoint evaluation for ABC insertion/removal localization."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


VALID_PARSE_STATUSES = {"valid_event", "valid_no_event"}


def evaluate_localizations(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = [score_row(row) for row in rows]
    summary = _aggregate(scored)
    summary["clustered_bootstrap_95ci"] = _bootstrap(
        scored,
        samples=bootstrap_samples,
        seed=seed,
    )
    return summary, scored


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    ground = list(row.get("ground_intervals", []))
    parse_status = str(row.get("parse_status", "missing"))
    event = row.get("event")
    valid = parse_status in VALID_PARSE_STATUSES
    has_ground = bool(ground)
    predicted = parse_status == "valid_event" and isinstance(event, dict)

    midpoint = None
    event_type = None
    type_aware_matches: list[int] = []
    type_agnostic_matches: list[int] = []
    if predicted:
        midpoint = float(event["trajectory_midpoint_sec"])
        event_type = str(event["type"])
        for index, interval in enumerate(ground):
            if _contains(interval, midpoint):
                type_agnostic_matches.append(index)
                if str(interval["event_type"]) == event_type:
                    type_aware_matches.append(index)

    type_aware_hit = bool(type_aware_matches)
    type_agnostic_hit = bool(type_agnostic_matches)
    true_no_event = (
        parse_status == "valid_no_event" and not has_ground
    )
    decision_correct = type_aware_hit or true_no_event

    return {
        "task_name": row.get("task_name"),
        "bundle_id": row.get("bundle_id"),
        "run_status": row.get("status"),
        "parse_status": parse_status,
        "valid_response": valid,
        "has_ground_event": has_ground,
        "ground_interval_count": len(ground),
        "ground_insertion_count": sum(
            interval.get("event_type") == "insertion" for interval in ground
        ),
        "ground_removal_count": sum(
            interval.get("event_type") == "removal" for interval in ground
        ),
        "predicted_event": predicted,
        "predicted_type": event_type,
        "predicted_midpoint_sec": midpoint,
        "type_aware_hit": type_aware_hit,
        "type_agnostic_hit": type_agnostic_hit,
        "matched_ground_indices": type_aware_matches,
        "type_agnostic_matched_ground_indices": type_agnostic_matches,
        "matched_ground_segment_count": len(type_aware_matches),
        "true_no_event": true_no_event,
        "decision_correct": decision_correct,
        "parse_failure": not valid,
    }


def write_evaluation_outputs(
    output_directory: Path,
    summary: dict[str, Any],
    scored: list[dict[str, Any]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "evaluation.json", summary)
    csv_path = output_directory / "per_episode_evaluation.csv"
    fieldnames = [
        "task_name",
        "bundle_id",
        "run_status",
        "parse_status",
        "valid_response",
        "has_ground_event",
        "ground_interval_count",
        "ground_insertion_count",
        "ground_removal_count",
        "predicted_event",
        "predicted_type",
        "predicted_midpoint_sec",
        "type_aware_hit",
        "type_agnostic_hit",
        "matched_ground_segment_count",
        "true_no_event",
        "decision_correct",
        "parse_failure",
    ]
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    temporary.replace(csv_path)


def _contains(interval: dict[str, Any], point_sec: float) -> bool:
    return float(interval["start_sec"]) <= point_sec < float(interval["end_sec"])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory_count = len(rows)
    event_rows = [row for row in rows if row["has_ground_event"]]
    no_event_rows = [row for row in rows if not row["has_ground_event"]]
    predicted_rows = [row for row in rows if row["predicted_event"]]
    valid_rows = [row for row in rows if row["valid_response"]]
    ground_segment_count = sum(row["ground_interval_count"] for row in rows)
    matched_segment_count = sum(row["matched_ground_segment_count"] for row in rows)

    output = {
        "trajectory_count": trajectory_count,
        "event_trajectory_count": len(event_rows),
        "no_event_trajectory_count": len(no_event_rows),
        "ground_segment_count": ground_segment_count,
        "prediction_count": len(predicted_rows),
        "valid_response_count": len(valid_rows),
        "parse_failure_count": sum(row["parse_failure"] for row in rows),
        "type_aware_hit_count": sum(row["type_aware_hit"] for row in rows),
        "type_agnostic_hit_count": sum(row["type_agnostic_hit"] for row in rows),
        "true_no_event_count": sum(row["true_no_event"] for row in rows),
        "point_precision": _ratio(
            sum(row["type_aware_hit"] for row in predicted_rows),
            len(predicted_rows),
        ),
        "event_episode_hit_recall": _ratio(
            sum(row["type_aware_hit"] for row in event_rows),
            len(event_rows),
        ),
        "type_agnostic_event_episode_hit_recall": _ratio(
            sum(row["type_agnostic_hit"] for row in event_rows),
            len(event_rows),
        ),
        "ground_segment_recall": _ratio(
            matched_segment_count,
            ground_segment_count,
        ),
        "no_event_specificity": _ratio(
            sum(row["true_no_event"] for row in no_event_rows),
            len(no_event_rows),
        ),
        "all_episode_decision_accuracy": _ratio(
            sum(row["decision_correct"] for row in rows),
            trajectory_count,
        ),
        "parse_failure_rate": _ratio(
            sum(row["parse_failure"] for row in rows),
            trajectory_count,
        ),
    }
    output["by_event_type"] = {
        event_type: _aggregate_type(rows, event_type)
        for event_type in ("insertion", "removal")
    }
    return output


def _aggregate_type(rows: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    relevant = [
        row
        for row in rows
        if row[f"ground_{event_type}_count"] > 0
    ]
    predictions = [
        row for row in rows if row["predicted_type"] == event_type
    ]
    return {
        "event_trajectory_count": len(relevant),
        "prediction_count": len(predictions),
        "hit_count": sum(row["type_aware_hit"] for row in predictions),
        "episode_hit_recall": _ratio(
            sum(row["type_aware_hit"] for row in relevant),
            len(relevant),
        ),
        "point_precision": _ratio(
            sum(row["type_aware_hit"] for row in predictions),
            len(predictions),
        ),
    }


def _bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    if samples <= 0 or not rows:
        return {}
    rng = random.Random(seed)
    keys = (
        "point_precision",
        "event_episode_hit_recall",
        "type_agnostic_event_episode_hit_recall",
        "ground_segment_recall",
        "no_event_specificity",
        "all_episode_decision_accuracy",
        "parse_failure_rate",
    )
    distributions = {key: [] for key in keys}
    for _ in range(samples):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        aggregate = _aggregate(sampled)
        for key in keys:
            value = aggregate[key]
            if value is not None:
                distributions[key].append(float(value))
    return {key: _percentile(values) for key, values in distributions.items()}


def _percentile(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)
    last = len(ordered) - 1
    return [ordered[round(0.025 * last)], ordered[round(0.975 * last)]]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
