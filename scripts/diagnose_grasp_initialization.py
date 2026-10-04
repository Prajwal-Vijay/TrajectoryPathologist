"""Export SAM 3 held-object initialization QA for reviewed ABC trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathologist.adapters.abc130k_bundle import ABC130KBundleAdapter
from pathologist.config import load_pipeline_config
from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.grasp_initialization import GraspObjectInitializer
from pathologist.metrics.grasp_integrity import GraspIntegrityMetric, HoldWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        default="ABC130K_CVPR_all_methods_k5/abc_failure_review_100.csv",
    )
    parser.add_argument(
        "--data-root",
        default="/media/helen/28TB_VOL4/ABC-130k_subset/data/train",
    )
    parser.add_argument(
        "--config",
        default="pathologist/configs/abc130k_grasp_integrity.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/helen/28TB_VOL4/ABC130K_grasp_integrity/initialization_qa_30",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default="/media/helen/28TB_VOL4/ABC130K_grasp_integrity/native_frame_cache",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--run-signals", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_pipeline_config(config_path=args.config)
    metric_config = config["metrics"]["grasp_integrity"]
    metric_config["tracking_enabled"] = False
    trajectories = _reviewed_trajectories(
        Path(args.review_csv),
        limit=args.limit,
        episode_ids=set(args.episode_id),
    )
    rows_path = output / "initialization_results.jsonl"
    completed = (
        {
            str(row["episode_id"])
            for row in _read_jsonl(rows_path)
            if row.get("status") == "complete"
        }
        if args.resume
        else set()
    )
    if not args.resume:
        rows_path.write_text("", encoding="utf-8")

    adapter = ABC130KBundleAdapter()
    initializer = GraspObjectInitializer(
        config=metric_config,
        output_dir=output / "per_trajectory",
    )
    for index, record in enumerate(trajectories, start=1):
        episode_id = record["episode_id"]
        if episode_id in completed:
            print(f"[{index}/{len(trajectories)}] resume skip {episode_id}")
            continue
        task_name = record["task_name"]
        print(f"[{index}/{len(trajectories)}] {task_name}/{episode_id}", flush=True)
        episode_dir = Path(args.data_root) / task_name / episode_id
        try:
            bundle = adapter.load(
                episode_dir / "episode.mcap",
                episode_dir / "annotation.mcap",
                frame_cache_dir=(
                    Path(args.frame_cache_dir) / task_name / episode_id
                ),
                camera_name=None,
                camera_names=["left_wrist", "right_wrist"],
                camera_sample_hz=None,
                robot_downsample_hz=50.0,
                extract_frames=True,
            )
            metric = GraspIntegrityMetric().compute(
                MetricComputationContext(bundle=bundle, config=config)
            )
            windows = {
                arm: [_hold_from_dict(item) for item in values]
                for arm, values in metric.metadata.get("hold_windows", {}).items()
            }
            results = initializer.initialize_bundle(bundle, windows)
            row = {
                "status": "complete",
                "task_name": task_name,
                "episode_id": episode_id,
                "hold_window_count": len(results),
                "results": [result.to_dict() for result in results],
            }
        except Exception as exc:
            row = {
                "status": "error",
                "task_name": task_name,
                "episode_id": episode_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rows = _read_jsonl(rows_path)
    results = [
        result
        for row in rows
        if row.get("status") == "complete"
        for result in row.get("results", [])
    ]
    errors = [row for row in rows if row.get("status") == "error"]
    no_object = sum(bool(result.get("no_object_found")) for result in results)
    summary = {
        "trajectory_count": len(trajectories),
        "complete_trajectory_count": len(rows) - len(errors),
        "error_count": len(errors),
        "hold_window_count": len(results),
        "no_object_found_count": no_object,
        "no_object_found_fraction": no_object / len(results) if results else None,
        "qa_review_required": True,
        "acceptance_threshold": 0.8,
    }
    (output / "initialization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_review_csv(output / "initialization_review.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.run_signals:
        metric_config["tracking_enabled"] = True
        _run_signal_diagnostics(
            args=args,
            config=config,
            trajectories=trajectories,
            output=output,
        )


def _reviewed_trajectories(
    path: Path,
    *,
    limit: int,
    episode_ids: set[str],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trajectory = str(row.get("trajectory") or "").strip()
            episode_id = str(
                row.get("episode_id")
                or row.get("trajectory_id")
                or row.get("bundle_id")
                or (trajectory.rsplit("/", 1)[-1] if "/" in trajectory else "")
                or ""
            ).strip()
            task_name = str(
                row.get("task_name")
                or row.get("task")
                or (trajectory.rsplit("/", 1)[0] if "/" in trajectory else "")
            ).strip()
            if (
                not episode_id
                or not task_name
                or episode_id in seen
                or (episode_ids and episode_id not in episode_ids)
            ):
                continue
            seen.add(episode_id)
            output.append(
                {
                    "task_name": task_name,
                    "episode_id": episode_id,
                    "failure_segments": str(row.get("failure_segments") or ""),
                }
            )
            if len(output) >= limit:
                break
    return output


def _hold_from_dict(payload: dict[str, Any]) -> HoldWindow:
    return HoldWindow(
        arm=str(payload["arm"]),
        close_intent_sec=float(payload["close_intent_sec"]),
        settle_sec=float(payload["settle_sec"]),
        open_sec=float(payload["open_sec"]),
        state_plateau=float(payload["state_plateau"]),
        command_plateau=float(payload["command_plateau"]),
        empty_closed=float(payload["empty_closed"]),
        grasp_width=float(payload["grasp_width"]),
        capacity=float(payload["capacity"]),
        telemetry_empty=bool(payload["telemetry_empty"]),
        telemetry_blind=bool(payload["telemetry_blind"]),
    )


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_name",
                "episode_id",
                "arm",
                "window_index",
                "settle_sec",
                "init_confidence",
                "n_candidates",
                "qa_overlay_path",
                "review_label",
            ],
        )
        writer.writeheader()
        for row in rows:
            if row.get("status") != "complete":
                continue
            for result in row.get("results", []):
                writer.writerow(
                    {
                        "task_name": row["task_name"],
                        "episode_id": row["episode_id"],
                        "arm": result["arm"],
                        "window_index": result["window_index"],
                        "settle_sec": result["settle_sec"],
                        "init_confidence": result["init_confidence"],
                        "n_candidates": result["n_candidates"],
                        "qa_overlay_path": result["qa_overlay_path"],
                        "review_label": "",
                    }
                )


_INTERVAL_RE = re.compile(
    r"(?P<label>[^;\[]*?)\[\s*(?P<start>\d+:\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<end>\d+:\d+(?:\.\d+)?)\s*\)"
)


def _run_signal_diagnostics(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    trajectories: list[dict[str, str]],
    output: Path,
) -> None:
    rows_path = output / "signal_diagnostics.jsonl"
    completed = (
        {
            str(row["episode_id"])
            for row in _read_jsonl(rows_path)
            if row.get("status") == "complete"
        }
        if args.resume
        else set()
    )
    if not args.resume:
        rows_path.write_text("", encoding="utf-8")
    adapter = ABC130KBundleAdapter()
    for index, record in enumerate(trajectories, start=1):
        episode_id = record["episode_id"]
        if episode_id in completed:
            print(f"[signals {index}/{len(trajectories)}] resume skip {episode_id}")
            continue
        print(
            f"[signals {index}/{len(trajectories)}] "
            f"{record['task_name']}/{episode_id}",
            flush=True,
        )
        episode_dir = Path(args.data_root) / record["task_name"] / episode_id
        try:
            bundle = adapter.load(
                episode_dir / "episode.mcap",
                episode_dir / "annotation.mcap",
                frame_cache_dir=(
                    Path(args.frame_cache_dir) / record["task_name"] / episode_id
                ),
                camera_name=None,
                camera_names=["left_wrist", "right_wrist"],
                camera_sample_hz=None,
                robot_downsample_hz=50.0,
                extract_frames=True,
            )
            metric = GraspIntegrityMetric().compute(
                MetricComputationContext(bundle=bundle, config=config)
            )
            if metric.metadata.get("available") is not True:
                raise RuntimeError(
                    str(metric.metadata.get("unavailable_reason") or "metric unavailable")
                )
            if metric.metadata.get("vision_available") is not True:
                raise RuntimeError(str(metric.metadata.get("vision_warning")))
            intervals = _parse_reviewed_intervals(record["failure_segments"])
            timestamps = [
                float(sample.timestamp_sec) for sample in metric.samples
            ]
            labels = [_inside_any(value, intervals) for value in timestamps]
            component_values = metric.metadata["component_values"]
            per_arm_values = metric.metadata["per_arm_values"]
            aucs = {
                name: _binary_auc(labels, list(map(float, values)))
                for name, values in component_values.items()
            }
            arm_aucs = {
                arm: {
                    name: _binary_auc(labels, list(map(float, values)))
                    for name, values in signals.items()
                }
                for arm, signals in per_arm_values.items()
            }
            events = [
                {
                    **window["event"],
                    "arm": window["arm"],
                    "window_index": window["window_index"],
                }
                for window in metric.metadata.get("tracking_windows", [])
                if isinstance(window.get("event"), dict)
            ]
            event_matches = [
                _match_event(event, intervals) for event in events
            ]
            row = {
                "status": "complete",
                "task_name": record["task_name"],
                "episode_id": episode_id,
                "task_category": _task_category(record["task_name"]),
                "intervals": intervals,
                "auroc": aucs,
                "per_arm_auroc": arm_aucs,
                "events": events,
                "event_matches": event_matches,
                "false_positive_event_count": sum(
                    match["failure_type"] is None for match in event_matches
                ),
                "telemetry_blind_window_count": sum(
                    bool(window["telemetry_blind"])
                    for windows in metric.metadata.get("hold_windows", {}).values()
                    for window in windows
                ),
                "tracking_windows": metric.metadata.get("tracking_windows", []),
            }
        except Exception as exc:
            row = {
                "status": "error",
                "task_name": record["task_name"],
                "episode_id": episode_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    rows = _read_jsonl(rows_path)
    complete = [row for row in rows if row.get("status") == "complete"]
    rng = np.random.default_rng(args.seed)
    signal_names = (
        "grasp_attach_loss",
        "grasp_rel_motion",
        "grasp_aperture_collapse",
    )
    summary = {
        "trajectory_count": len(complete),
        "error_count": sum(row.get("status") == "error" for row in rows),
        "per_signal_auroc": {
            name: _bootstrap_mean(
                [
                    float(row["auroc"][name])
                    for row in complete
                    if row["auroc"].get(name) is not None
                ],
                samples=args.bootstrap_samples,
                rng=rng,
            )
            for name in signal_names
        },
        "event_count": sum(len(row["events"]) for row in complete),
        "false_positive_event_count": sum(
            int(row["false_positive_event_count"]) for row in complete
        ),
        "telemetry_blind_window_count": sum(
            int(row["telemetry_blind_window_count"]) for row in complete
        ),
        "timing_lag_sec": _bootstrap_mean(
            [
                float(match["lag_sec"])
                for row in complete
                for match in row["event_matches"]
                if match.get("lag_sec") is not None
            ],
            samples=args.bootstrap_samples,
            rng=rng,
        ),
        "by_task_category": {
            category: {
                name: _bootstrap_mean(
                    [
                        float(row["auroc"][name])
                        for row in complete
                        if row["task_category"] == category
                        and row["auroc"].get(name) is not None
                    ],
                    samples=args.bootstrap_samples,
                    rng=rng,
                )
                for name in signal_names
            }
            for category in ("rigid", "deformable")
        },
    }
    (output / "signal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_reviewed_intervals(text: str) -> list[dict[str, Any]]:
    return [
        {
            "label": match.group("label").strip(),
            "failure_type": _failure_type(match.group("label")),
            "start_sec": _parse_timestamp(match.group("start")),
            "end_sec": _parse_timestamp(match.group("end")),
        }
        for match in _INTERVAL_RE.finditer(text)
    ]


def _failure_type(label: str) -> str:
    lowered = label.lower()
    if "drop" in lowered or "fall" in lowered:
        return "drop"
    if "slip" in lowered:
        return "slip"
    if "miss" in lowered:
        return "miss"
    return "other"


def _match_event(
    event: dict[str, Any],
    intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    timestamp = float(event["timestamp_sec"])
    matched = next(
        (
            interval
            for interval in intervals
            if interval["start_sec"] - 0.5 <= timestamp < interval["end_sec"] + 0.5
        ),
        None,
    )
    return {
        "event_label": event["label"],
        "event_timestamp_sec": timestamp,
        "failure_type": matched["failure_type"] if matched else None,
        "lag_sec": timestamp - float(matched["start_sec"]) if matched else None,
    }


def _inside_any(
    timestamp: float,
    intervals: list[dict[str, Any]],
) -> bool:
    return any(
        float(interval["start_sec"]) <= timestamp < float(interval["end_sec"])
        for interval in intervals
    )


def _binary_auc(labels: list[bool], values: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _bootstrap_mean(
    values: list[float],
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not values:
        return {"mean": None, "bootstrap_95ci": None, "count": 0}
    array = np.asarray(values, dtype=float)
    draws = np.asarray(
        [
            rng.choice(array, size=len(array), replace=True).mean()
            for _ in range(max(1, samples))
        ]
    )
    return {
        "mean": float(array.mean()),
        "bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "count": len(values),
    }


def _task_category(task_name: str) -> str:
    deformable_tokens = ("fold", "shirt", "napkin", "paper_plane", "surgical_kit")
    return (
        "deformable"
        if any(token in task_name.lower() for token in deformable_tokens)
        else "rigid"
    )


def _parse_timestamp(value: str) -> float:
    minutes, seconds = value.split(":", 1)
    return 60.0 * int(minutes) + float(seconds)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            output.append(payload)
    return output


if __name__ == "__main__":
    main()
