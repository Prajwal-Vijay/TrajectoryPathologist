"""Run and evaluate cached-VLM PRISM ablations on ABC insert/remove events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pathologist.adapters.abc130k_bundle import ABC130KBundleAdapter
from pathologist.config import load_pipeline_config
from pathologist.events.metric_selector import MetricEventSelector
from pathologist.metrics import (
    ActionDiscontinuityMetric,
    CachedVLMYesNoMetric,
    EefAngularJerkMetric,
    EefRotationalStallingMetric,
    EefTranslationalJerkMetric,
    EefTranslationalStallingMetric,
    ECODSurpriseScorer,
    MetricEngine,
    MotionSmoothnessMetric,
)
from pathologist.schemas.metrics import MetricProfile

from VL_implemenation.abc_data import Episode, discover_episodes
from VL_implemenation.yesno_scan import cache_paths


METHOD_CONFIGS = {
    "telemetry": Path("pathologist/configs/abc_ir_telemetry.json"),
    "vlm_only": Path("pathologist/configs/abc_ir_vlm_only.json"),
    "sum": Path("pathologist/configs/abc_ir_sum.json"),
    "grouped": Path("pathologist/configs/abc_ir_grouped.json"),
}
BASE_METRICS = [
    ActionDiscontinuityMetric(),
    MotionSmoothnessMetric(),
    EefTranslationalStallingMetric(),
    EefRotationalStallingMetric(),
    EefTranslationalJerkMetric(),
    EefAngularJerkMetric(),
    CachedVLMYesNoMetric(),
]


def run_ablation(
    episodes: Sequence[Episode],
    *,
    cache_root: Path,
    output_root: Path,
    robot_downsample_hz: float,
    top_k: int,
    resume: bool,
) -> list[dict[str, Any]]:
    """Compute base telemetry once, then run all four cheap ECOD variants."""

    output_root.mkdir(parents=True, exist_ok=True)
    peaks_path = output_root / "ranked_peaks.jsonl"
    existing = _read_jsonl(peaks_path) if resume else []
    completed = {str(row["bundle_id"]) for row in existing}
    if not resume and peaks_path.exists():
        peaks_path.unlink()
    configs = {
        name: load_pipeline_config(path)
        for name, path in METHOD_CONFIGS.items()
    }
    base_config = {
        "metrics": {"action_discontinuity": {"ignored_dimensions": []}},
        "surprise": {"enabled": False},
        "vlm_yesno": {"cache_root": str(cache_root)},
    }
    engine = MetricEngine(metrics=list(BASE_METRICS), config=base_config)
    adapter = ABC130KBundleAdapter()
    rows = list(existing)

    for index, episode in enumerate(episodes, start=1):
        if episode.bundle_id in completed:
            continue
        bundle = adapter.load(
            episode.episode_path,
            episode.annotation_path,
            extract_frames=False,
            robot_downsample_hz=robot_downsample_hz,
        )
        base_profile = engine.compute(bundle)
        metric_lookup = {metric.name: metric for metric in base_profile.metrics}
        vlm_metric = metric_lookup["vlm_insert_remove_margin"]
        method_peaks: dict[str, list[dict[str, float]]] = {}
        all_peaks: dict[str, list[dict[str, float]]] = {}
        method_sources: dict[str, list[str]] = {}
        for method_name, config in configs.items():
            profile = ECODSurpriseScorer(config=config).append_to_profile(
                MetricProfile(
                    bundle_id=base_profile.bundle_id,
                    metrics=list(base_profile.metrics),
                    global_events=list(base_profile.global_events),
                    metadata=dict(base_profile.metadata),
                )
            )
            # Preserve every local maximum so alternative post-selection
            # strategies (persistence, NMS, or gating) can run before top-K.
            proposals = _selector(config, 0).select(
                profile,
                trajectory_bundle=None,
            )
            serialized_proposals = [
                {
                    "rank": rank,
                    "time_sec": float(proposal.metadata["peak_sec"]),
                    "persistence": float(proposal.metadata["persistence"]),
                    "peak_score": float(proposal.metadata["peak_score"]),
                }
                for rank, proposal in enumerate(proposals, start=1)
            ]
            method_peaks[method_name] = serialized_proposals[:top_k]
            all_peaks[method_name] = serialized_proposals
            aggregate = next(
                (
                    metric
                    for metric in profile.metrics
                    if metric.name == "ecod_aggregate_score"
                ),
                None,
            )
            method_sources[method_name] = (
                list(aggregate.metadata.get("source_metric_names", []))
                if aggregate is not None
                else []
            )
        row = {
            "task_name": episode.task_name,
            "bundle_id": episode.bundle_id,
            "duration_sec": episode.duration_sec,
            "ground_intervals": [
                interval.to_dict() for interval in episode.ground_intervals
            ],
            "vlm_available": bool(vlm_metric.metadata.get("available")),
            "vlm_unavailable_reason": vlm_metric.metadata.get(
                "unavailable_reason"
            ),
            "method_source_metrics": method_sources,
            "peaks": method_peaks,
            "all_peaks": all_peaks,
        }
        _append_jsonl(peaks_path, row)
        rows.append(row)
        print(
            f"[{index}/{len(episodes)}] {episode.task_name}/{episode.bundle_id}",
            flush=True,
        )
    _write_json_atomic(
        output_root / "run_summary.json",
        {
            "status": "complete",
            "episode_count": len(rows),
            "vlm_available_count": sum(row["vlm_available"] for row in rows),
            "vlm_missing_count": sum(not row["vlm_available"] for row in rows),
            "robot_downsample_hz": robot_downsample_hz,
            "top_k": top_k,
            "all_method_peaks_saved": True,
            "method_configs": {
                name: str(path) for name, path in METHOD_CONFIGS.items()
            },
            "ranked_peaks_path": str(peaks_path),
        },
    )
    return rows


def evaluate_ablation(
    rows: Sequence[dict[str, Any]],
    *,
    output_root: Path,
    bootstrap_samples: int,
    null_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate merged insert/remove localization and paired differences."""

    event_rows = [row for row in rows if row["ground_intervals"]]
    no_event_rows = [row for row in rows if not row["ground_intervals"]]
    method_names = list(METHOD_CONFIGS)
    table: dict[str, dict[str, Any]] = {}
    for method_name in method_names:
        metrics = _score_rows(event_rows, method_name=method_name)
        metrics["bootstrap_95ci"] = _bootstrap_metrics(
            event_rows,
            method_name=method_name,
            samples=bootstrap_samples,
            seed=seed,
        )
        table[method_name] = metrics

    null_metrics, null_intervals = _chance_null(
        event_rows,
        draws=null_draws,
        seed=seed,
    )
    null_metrics["simulation_95ci"] = null_intervals
    table["null"] = null_metrics
    paired = {
        comparison: _paired_bootstrap(
            event_rows,
            left=left,
            right=right,
            samples=bootstrap_samples,
            seed=seed,
        )
        for comparison, left, right in (
            ("grouped_minus_telemetry", "grouped", "telemetry"),
            ("grouped_minus_vlm_only", "grouped", "vlm_only"),
            ("sum_minus_telemetry", "sum", "telemetry"),
        )
    }
    no_event_max_p_yes = _no_event_max_probabilities(no_event_rows)
    summary = {
        "protocol": {
            "event_class": "insert_remove",
            "interval_semantics": "half-open [start_sec, end_sec)",
            "population": "episodes with at least one semantic insert/remove interval",
            "top_k": 5,
            "chance_draws_per_episode": null_draws,
            "bootstrap_episode_resamples": bootstrap_samples,
            "seed": seed,
        },
        "event_episode_count": len(event_rows),
        "ground_segment_count": sum(
            len(row["ground_intervals"]) for row in event_rows
        ),
        "no_event_episode_count": len(no_event_rows),
        "table": table,
        "paired_differences": paired,
        "no_event_max_p_yes": no_event_max_p_yes,
    }
    _write_json_atomic(output_root / "evaluation.json", summary)
    _write_table_csv(output_root / "evaluation_table.csv", table)
    _write_jsonl_atomic(
        output_root / "no_event_max_p_yes.jsonl",
        no_event_max_p_yes["episodes"],
    )
    return summary


def _selector(config: dict[str, Any], top_k: int) -> MetricEventSelector:
    selector_config = config.get("selector", {})
    flood = dict(selector_config.get("surprise_flood", {}))
    flood["max_proposals"] = top_k
    flood["ranking"] = "persistence"
    return MetricEventSelector(
        frame_stride_within_window=1,
        visual_context_sec=0.0,
        max_frames_per_camera=max(1, top_k),
        metric_packet_context_steps=0,
        surprise_flood_config=flood,
    )


def _score_rows(
    rows: Sequence[dict[str, Any]],
    *,
    method_name: str,
) -> dict[str, Any]:
    hit_by_k = {1: 0, 3: 0, 5: 0}
    segment_hits = 0
    point_hits = 0
    prediction_count = 0
    observed_event_types = {
        str(interval["event_type"])
        for row in rows
        for interval in row["ground_intervals"]
    }
    event_types = (
        {"insertion", "removal"}
        if observed_event_types <= {"insertion", "removal"}
        else observed_event_types
    )
    per_type = {
        event_type: {"segment_count": 0, "hit_segment_count": 0}
        for event_type in sorted(event_types)
    }
    for row in rows:
        intervals = row["ground_intervals"]
        peaks = [
            float(peak["time_sec"])
            for peak in row["peaks"].get(method_name, [])
        ]
        for k in hit_by_k:
            hit_by_k[k] += any(
                _contains_any(intervals, point)
                for point in peaks[:k]
            )
        for interval in intervals:
            event_type = str(interval["event_type"])
            hit = any(_contains(interval, point) for point in peaks[:5])
            segment_hits += int(hit)
            per_type[event_type]["segment_count"] += 1
            per_type[event_type]["hit_segment_count"] += int(hit)
        prediction_count += len(peaks[:5])
        point_hits += sum(
            _contains_any(intervals, point)
            for point in peaks[:5]
        )
    episode_count = len(rows)
    segment_count = sum(len(row["ground_intervals"]) for row in rows)
    return {
        "trajectory_hit_at_1": hit_by_k[1] / episode_count if episode_count else None,
        "trajectory_hit_at_3": hit_by_k[3] / episode_count if episode_count else None,
        "trajectory_hit_at_5": hit_by_k[5] / episode_count if episode_count else None,
        "segment_recall_at_5": segment_hits / segment_count if segment_count else None,
        "precision_at_5": point_hits / prediction_count if prediction_count else None,
        "trajectory_count": episode_count,
        "segment_count": segment_count,
        "prediction_count_at_5": prediction_count,
        "per_type": {
            event_type: {
                **counts,
                "segment_recall_at_5": (
                    counts["hit_segment_count"] / counts["segment_count"]
                    if counts["segment_count"]
                    else None
                ),
            }
            for event_type, counts in per_type.items()
        },
    }


def _bootstrap_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    method_name: str,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0 or not rows:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        name: []
        for name in (
            "trajectory_hit_at_1",
            "trajectory_hit_at_3",
            "trajectory_hit_at_5",
            "segment_recall_at_5",
            "precision_at_5",
        )
    }
    for _ in range(samples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        metrics = _score_rows(sampled, method_name=method_name)
        for name in values:
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))
    return {
        name: _percentile_interval(metric_values)
        for name, metric_values in values.items()
    }


def _paired_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    metric_names = (
        "trajectory_hit_at_1",
        "trajectory_hit_at_3",
        "trajectory_hit_at_5",
        "segment_recall_at_5",
        "precision_at_5",
    )
    observed_left = _score_rows(rows, method_name=left)
    observed_right = _score_rows(rows, method_name=right)
    output = {
        name: {
            "difference": (
                observed_left[name] - observed_right[name]
                if observed_left[name] is not None
                and observed_right[name] is not None
                else None
            ),
            "bootstrap_95ci": [],
        }
        for name in metric_names
    }
    if samples <= 0 or not rows:
        return output
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in metric_names}
    for _ in range(samples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        left_metrics = _score_rows(sampled, method_name=left)
        right_metrics = _score_rows(sampled, method_name=right)
        for name in metric_names:
            differences[name].append(
                left_metrics[name] - right_metrics[name]
            )
    for name in metric_names:
        output[name]["bootstrap_95ci"] = _percentile_interval(
            differences[name]
        )
    return output


def _chance_null(
    rows: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    if not rows:
        empty = {
            name: None
            for name in (
                "trajectory_hit_at_1",
                "trajectory_hit_at_3",
                "trajectory_hit_at_5",
                "segment_recall_at_5",
                "precision_at_5",
            )
        }
        empty.update(
            {
                "trajectory_count": 0,
                "segment_count": 0,
                "prediction_count_at_5": 0,
                "per_type": {
                    event_type: {
                        "segment_count": 0,
                        "segment_recall_at_5": None,
                    }
                    for event_type in ("insertion", "removal")
                },
            }
        )
        return empty, {}
    rng = np.random.default_rng(seed)
    metrics_by_draw: list[dict[str, Any]] = []
    aggregate_rows = []
    for row in rows:
        aggregate_rows.append({**row, "peaks": {**row["peaks"], "null": []}})
    for _ in range(draws):
        sampled_rows = []
        for row in aggregate_rows:
            points = rng.uniform(0.0, float(row["duration_sec"]), size=5)
            sampled_rows.append(
                {
                    **row,
                    "peaks": {
                        **row["peaks"],
                        "null": [
                            {"time_sec": float(point)} for point in points
                        ],
                    },
                }
            )
        metrics_by_draw.append(_score_rows(sampled_rows, method_name="null"))
    metric_names = (
        "trajectory_hit_at_1",
        "trajectory_hit_at_3",
        "trajectory_hit_at_5",
        "segment_recall_at_5",
        "precision_at_5",
    )
    mean_metrics = {
        name: float(np.mean([metrics[name] for metrics in metrics_by_draw]))
        for name in metric_names
    }
    mean_metrics.update(
        {
            "trajectory_count": len(rows),
            "segment_count": sum(len(row["ground_intervals"]) for row in rows),
            "prediction_count_at_5": len(rows) * 5,
            "per_type": {
                event_type: {
                    "segment_count": sum(
                        interval["event_type"] == event_type
                        for row in rows
                        for interval in row["ground_intervals"]
                    ),
                    "segment_recall_at_5": float(
                        np.mean(
                            [
                                metrics["per_type"][event_type][
                                    "segment_recall_at_5"
                                ]
                                for metrics in metrics_by_draw
                            ]
                        )
                    ),
                }
                for event_type in ("insertion", "removal")
            },
        }
    )
    intervals = {
        name: _percentile_interval(
            [metrics[name] for metrics in metrics_by_draw]
        )
        for name in metric_names
    }
    return mean_metrics, intervals


def _no_event_max_probabilities(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    episodes = []
    for row in rows:
        path = Path(row["vlm_cache_path"]) if row.get("vlm_cache_path") else None
        maximum = None
        if path is not None and path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                margins = np.asarray(cached["margin"], dtype=np.float64)
            if margins.size:
                maximum = float(
                    1.0 / (1.0 + np.exp(-np.clip(np.max(margins), -80.0, 80.0)))
                )
        episodes.append(
            {
                "task_name": row["task_name"],
                "bundle_id": row["bundle_id"],
                "max_p_yes": maximum,
            }
        )
    values = [
        float(row["max_p_yes"])
        for row in episodes
        if row["max_p_yes"] is not None
    ]
    return {
        "episode_count": len(episodes),
        "available_count": len(values),
        "median": float(np.median(values)) if values else None,
        "q1": float(np.percentile(values, 25)) if values else None,
        "q3": float(np.percentile(values, 75)) if values else None,
        "episodes": episodes,
    }


def _contains(interval: dict[str, Any], point: float) -> bool:
    return float(interval["start_sec"]) <= point < float(interval["end_sec"])


def _contains_any(intervals: Sequence[dict[str, Any]], point: float) -> bool:
    return any(_contains(interval, point) for interval in intervals)


def _percentile_interval(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _write_table_csv(path: Path, table: dict[str, dict[str, Any]]) -> None:
    columns = [
        "method",
        "trajectory_hit_at_1",
        "trajectory_hit_at_3",
        "trajectory_hit_at_5",
        "segment_recall_at_5",
        "precision_at_5",
        "insertion_segment_recall_at_5",
        "removal_segment_recall_at_5",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method, metrics in table.items():
            writer.writerow(
                {
                    "method": method,
                    **{name: metrics[name] for name in columns[1:6]},
                    "insertion_segment_recall_at_5": metrics["per_type"][
                        "insertion"
                    ]["segment_recall_at_5"],
                    "removal_segment_recall_at_5": metrics["per_type"][
                        "removal"
                    ]["segment_recall_at_5"],
                }
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["run", "evaluate", "full"],
        default="full",
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("VL_implemenation/config.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--robot-downsample-hz", type=float, default=50.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    external_output = Path(config["output_root"]).expanduser().resolve()
    cache_root = external_output / "vlm_scores"
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else external_output / "prism_ablation"
    )
    episodes = discover_episodes(config["data_root"])
    if args.limit is not None:
        episodes = episodes[: args.limit]
    missing = [
        episode
        for episode in episodes
        if not cache_paths(cache_root, episode)[0].is_file()
    ]
    if args.command in {"run", "full"} and missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} VLM caches; finish Stage 1 before Stage 2"
        )
    if args.command in {"run", "full"}:
        rows = run_ablation(
            episodes,
            cache_root=cache_root,
            output_root=output_root,
            robot_downsample_hz=args.robot_downsample_hz,
            top_k=args.top_k,
            resume=not args.no_resume,
        )
    else:
        rows = _read_jsonl(output_root / "ranked_peaks.jsonl")
    for row in rows:
        episode = next(
            item for item in episodes if item.bundle_id == row["bundle_id"]
        )
        row["vlm_cache_path"] = str(cache_paths(cache_root, episode)[0])
    if args.command in {"evaluate", "full"}:
        summary = evaluate_ablation(
            rows,
            output_root=output_root,
            bootstrap_samples=args.bootstrap_samples,
            null_draws=args.null_draws,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
