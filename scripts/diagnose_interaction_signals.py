"""Diagnose slip/drop interaction signals on manually reviewed ABC-130K runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathologist.adapters.abc130k_bundle import ABC130KBundleAdapter
from pathologist.config import load_pipeline_config
from pathologist.events.metric_selector import MetricEventSelector
from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.engine import MetricEngine
from pathologist.metrics.helpers import extract_vector_series
from pathologist.metrics.object_interaction import (
    CoTrackerRunner,
    GroundedSAM2InteractionSegmenter,
    _dilate_mask,
    _erode_mask,
    _gripper_signal_series,
    _sample_mask_points,
    grasp_closed_mask,
)
from pathologist.metrics.surprise import ECODSurpriseScorer
from pathologist.schemas import MetricProfile, MetricResult, MetricSample


SIGNALS = (
    "object_residual_speed",
    "gripper_object_relative_speed",
    "gripper_object_separation_speed",
    "unexplained_object_motion",
    "unexplained_object_motion_grasped",
    "object_visibility_drop",
    "object_visibility_drop_grasped",
    "visual_flow",
)
CONFIG_PATHS = {
    "gated": Path("pathologist/configs/abc130k_interaction_gated.json"),
    "ungated": Path("pathologist/configs/abc130k_interaction_ungated.json"),
    "legacy": Path("pathologist/configs/abc130k_interaction_legacy.json"),
}
INTERVAL_RE = re.compile(
    r"(?P<label>[^;\[]*?)\[\s*(?P<start>\d+:\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<end>\d+:\d+(?:\.\d+)?)\s*\)"
)


@dataclass(frozen=True)
class ReviewedInterval:
    label: str
    start_sec: float
    end_sec: float

    @property
    def failure_type(self) -> str:
        lowered = self.label.lower()
        if "drop" in lowered:
            return "drop"
        if "slip" in lowered:
            return "slip"
        return "other"


@dataclass(frozen=True)
class ReviewedTrajectory:
    task_name: str
    episode_id: str
    intervals: tuple[ReviewedInterval, ...]


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
        "--output-dir",
        default="/media/helen/28TB_VOL4/ABC130K_interaction_signals/diagnostics_30",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default="/media/helen/28TB_VOL4/ABC130K_interaction_signals/frame_cache",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--camera-sample-hz", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = read_reviewed_trajectories(
        Path(args.review_csv),
        limit=args.limit,
    )
    configs = {
        name: load_pipeline_config(config_path=path)
        for name, path in CONFIG_PATHS.items()
    }
    for config in configs.values():
        config.setdefault("metrics", {}).setdefault(
            "visual_flow",
            {},
        )["camera_names"] = ["top"]
        flood = config.setdefault("selector", {}).setdefault(
            "surprise_flood",
            {},
        )
        flood["max_proposals"] = args.top_k

    per_trajectory_path = output_dir / "per_trajectory_diagnostics.jsonl"
    completed = (
        {
            str(row["episode_id"])
            for row in _read_jsonl(per_trajectory_path)
            if row.get("status") == "complete"
        }
        if args.resume
        else set()
    )
    if not args.resume:
        per_trajectory_path.write_text("", encoding="utf-8")

    adapter = ABC130KBundleAdapter()
    all_rows = _read_jsonl(per_trajectory_path) if args.resume else []
    for index, reviewed in enumerate(trajectories, start=1):
        if reviewed.episode_id in completed:
            print(f"[{index}/{len(trajectories)}] resume skip {reviewed.episode_id}")
            continue
        print(f"[{index}/{len(trajectories)}] {reviewed.task_name}/{reviewed.episode_id}")
        episode_dir = (
            Path(args.data_root)
            / reviewed.task_name
            / reviewed.episode_id
        )
        try:
            bundle = adapter.load(
                episode_dir / "episode.mcap",
                episode_dir / "annotation.mcap",
                frame_cache_dir=(
                    Path(args.frame_cache_dir)
                    / reviewed.task_name
                    / reviewed.episode_id
                ),
                camera_names=["top"],
                camera_sample_hz=args.camera_sample_hz,
                robot_downsample_hz=50.0,
                extract_frames=True,
            )
            profile = MetricEngine(config=configs["gated"]).compute(bundle)
            raw_profile = _raw_profile(profile)
            signal_rows, invalid_in_failure = _signal_diagnostics(
                raw_profile,
                reviewed.intervals,
                bundle,
                configs["gated"],
                args.top_k,
            )
            method_rows = {
                name: _evaluate_config(
                    raw_profile,
                    bundle,
                    reviewed.intervals,
                    config,
                    args.top_k,
                )
                for name, config in configs.items()
            }
            if not args.skip_qa:
                qa_dir = output_dir / "interaction_qa" / reviewed.episode_id
                _export_failure_qa(
                    bundle,
                    reviewed.intervals,
                    configs["gated"],
                    qa_dir,
                )
                if index <= 3:
                    _export_gating_orientation(
                        bundle,
                        reviewed.intervals,
                        qa_dir / "gating_orientation.png",
                    )
            row = {
                "status": "complete",
                "task_name": reviewed.task_name,
                "episode_id": reviewed.episode_id,
                "duration_sec": float(bundle.duration_sec or 0.0),
                "intervals": [
                    {
                        "label": interval.label,
                        "failure_type": interval.failure_type,
                        "start_sec": interval.start_sec,
                        "end_sec": interval.end_sec,
                    }
                    for interval in reviewed.intervals
                ],
                "signals": signal_rows,
                "methods": method_rows,
                "invalid_object_track_inside_failure": invalid_in_failure,
            }
        except Exception as exc:
            row = {
                "status": "error",
                "task_name": reviewed.task_name,
                "episode_id": reviewed.episode_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append_jsonl(per_trajectory_path, row)
        all_rows.append(row)

    complete_rows = [
        row for row in _read_jsonl(per_trajectory_path)
        if row.get("status") == "complete"
    ]
    summary = summarize_diagnostics(
        complete_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        top_k=args.top_k,
    )
    summary["requested_trajectory_count"] = len(trajectories)
    summary["complete_trajectory_count"] = len(complete_rows)
    summary["error_count"] = len(trajectories) - len(complete_rows)
    summary["per_trajectory_path"] = str(per_trajectory_path)
    (output_dir / "interaction_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def read_reviewed_trajectories(
    path: Path,
    *,
    limit: int,
) -> list[ReviewedTrajectory]:
    output: list[ReviewedTrajectory] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            trajectory = row[0].strip()
            if "/" not in trajectory:
                continue
            task_name, episode_id = trajectory.split("/", 1)
            annotation = ",".join(row[1:])
            intervals = tuple(
                ReviewedInterval(
                    label=match.group("label").strip(" \",'\""),
                    start_sec=_parse_timestamp(match.group("start")),
                    end_sec=_parse_timestamp(match.group("end")),
                )
                for match in INTERVAL_RE.finditer(annotation)
            )
            if intervals:
                output.append(
                    ReviewedTrajectory(task_name, episode_id, intervals)
                )
            if len(output) >= limit:
                break
    return output


def _raw_profile(profile: MetricProfile) -> MetricProfile:
    metrics = [
        metric
        for metric in profile.metrics
        if not metric.name.endswith("_ecod_score")
        and metric.name != "ecod_aggregate_score"
    ]
    return MetricProfile(
        bundle_id=profile.bundle_id,
        metrics=metrics,
        global_events=profile.global_events,
        metadata=profile.metadata,
    )


def _signal_diagnostics(
    profile: MetricProfile,
    intervals: tuple[ReviewedInterval, ...],
    bundle: Any,
    config: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, Any], list[float]]:
    by_name = {metric.name: metric for metric in profile.metrics}
    context = MetricComputationContext(bundle=bundle, config=config)
    stream, gripper_ts, gripper_values = _gripper_signal_series(
        context,
        "auto",
    )
    del stream
    signal_rows: dict[str, Any] = {}
    invalid_in_failure: list[float] = []
    parent = by_name.get("object_interaction_motion")
    if parent is not None:
        validity = parent.metadata.get("object_track_valid", [])
        for sample, valid in zip(parent.samples, validity):
            if not bool(valid) and _inside_any(
                float(sample.timestamp_sec),
                intervals,
            ):
                invalid_in_failure.append(float(sample.timestamp_sec))

    flood_config = copy.deepcopy(
        config.get("selector", {}).get("surprise_flood", {})
    )
    flood_config["max_proposals"] = top_k
    for name in SIGNALS:
        metric = by_name.get(name)
        if metric is None or not metric.samples:
            signal_rows[name] = {"available": False}
            continue
        timestamps = [float(sample.timestamp_sec) for sample in metric.samples]
        values = [float(sample.value) for sample in metric.samples]
        labels = [
            _inside_any(timestamp, intervals) for timestamp in timestamps
        ]
        auc = _binary_auc(labels, values)
        scored = ECODSurpriseScorer(
            config={"surprise": {"source_metrics": [name]}}
        ).score_metric(metric)
        peak_times = (
            _persistent_peak_times(scored, flood_config, top_k)
            if scored is not None and scored.samples
            else []
        )
        phase_counts: Counter[str] = Counter()
        if timestamps and gripper_ts:
            closed = grasp_closed_mask(
                timestamps,
                gripper_ts,
                gripper_values,
                settle_s=0.0,
            )
            if closed is not None:
                transitions = [
                    timestamps[index]
                    for index in range(1, len(timestamps))
                    if bool(closed[index]) != bool(closed[index - 1])
                ]
                for peak_time in peak_times:
                    if any(abs(peak_time - time) <= 0.3 for time in transitions):
                        phase_counts["transition"] += 1
                    else:
                        nearest = min(
                            range(len(timestamps)),
                            key=lambda i: abs(timestamps[i] - peak_time),
                        )
                        phase_counts[
                            "closed" if bool(closed[nearest]) else "open"
                        ] += 1
        signal_rows[name] = {
            "available": True,
            "auroc": auc,
            "sample_count": len(values),
            "inside_failure_sample_count": sum(labels),
            "top_peak_times_sec": peak_times,
            "peak_phase_counts": dict(phase_counts),
        }
    return signal_rows, invalid_in_failure


def _evaluate_config(
    raw_profile: MetricProfile,
    bundle: Any,
    intervals: tuple[ReviewedInterval, ...],
    config: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    scored = ECODSurpriseScorer(config=config).append_to_profile(raw_profile)
    flood = copy.deepcopy(
        config.get("selector", {}).get("surprise_flood", {})
    )
    flood["max_proposals"] = top_k
    selector = MetricEventSelector(
        frame_stride_within_window=1,
        visual_context_sec=0.5,
        max_frames_per_camera=top_k,
        metric_packet_context_steps=0,
        surprise_flood_config=flood,
    )
    proposals = selector.select(scored, trajectory_bundle=bundle)[:top_k]
    peaks = [
        float(proposal.metadata.get("peak_sec", proposal.time_window.start_sec))
        for proposal in proposals
    ]
    hit_indices = {
        index
        for index, interval in enumerate(intervals)
        if any(interval.start_sec <= peak < interval.end_sec for peak in peaks)
    }
    peak_hits = sum(_inside_any(peak, intervals) for peak in peaks)
    return {
        "peaks_sec": peaks,
        "selected_peak_count": len(peaks),
        "peak_hit_count": peak_hits,
        "hit_interval_indices": sorted(hit_indices),
    }


def _persistent_peak_times(
    metric: MetricResult,
    flood_config: dict[str, Any],
    top_k: int,
) -> list[float]:
    aggregate = MetricResult(
        name="ecod_aggregate_score",
        description=metric.description,
        samples=[
            MetricSample(
                timestamp_sec=float(sample.timestamp_sec),
                value=float(sample.value),
            )
            for sample in metric.samples
        ],
    )
    profile = MetricProfile(bundle_id="signal", metrics=[aggregate])
    config = copy.deepcopy(flood_config)
    config["aggregate_metric_name"] = aggregate.name
    config["max_proposals"] = top_k
    selector = MetricEventSelector(
        metric_packet_context_steps=0,
        surprise_flood_config=config,
    )
    return [
        float(proposal.metadata["peak_sec"])
        for proposal in selector.select(profile)[:top_k]
    ]


def summarize_diagnostics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    auc_summary: dict[str, Any] = {}
    for signal in SIGNALS:
        values = [
            float(row["signals"][signal]["auroc"])
            for row in rows
            if row["signals"].get(signal, {}).get("auroc") is not None
        ]
        auc_summary[signal] = _bootstrap_mean(
            values,
            bootstrap_samples,
            rng,
        )

    paired: dict[str, Any] = {}
    target = "unexplained_object_motion_grasped"
    for legacy in (
        "object_residual_speed",
        "gripper_object_relative_speed",
        "gripper_object_separation_speed",
    ):
        differences = [
            float(row["signals"][target]["auroc"])
            - float(row["signals"][legacy]["auroc"])
            for row in rows
            if row["signals"].get(target, {}).get("auroc") is not None
            and row["signals"].get(legacy, {}).get("auroc") is not None
        ]
        paired[legacy] = _bootstrap_mean(
            differences,
            bootstrap_samples,
            rng,
        )

    phase_summary: dict[str, Any] = {}
    for signal in SIGNALS:
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(
                row["signals"].get(signal, {}).get("peak_phase_counts", {})
            )
        total = sum(counts.values())
        phase_summary[signal] = {
            phase: counts[phase] / total if total else None
            for phase in ("open", "closed", "transition")
        }

    methods = {
        method: _method_summary(rows, method)
        for method in CONFIG_PATHS
    }
    random_null = _randomized_interior_null(
        rows,
        top_k=top_k,
        repetitions=max(1, bootstrap_samples),
        rng=rng,
    )
    return {
        "per_signal_auroc": auc_summary,
        "paired_auroc_difference_vs_gated_unexplained_motion": paired,
        "peak_phase_fraction": phase_summary,
        "end_to_end": methods,
        "randomized_interior_null": random_null,
        "invalid_track_trajectories": [
            {
                "episode_id": row["episode_id"],
                "invalid_timestamps_sec": row[
                    "invalid_object_track_inside_failure"
                ],
            }
            for row in rows
            if row["invalid_object_track_inside_failure"]
        ],
    }


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = hits = segment_hits = segment_count = trajectory_hits = 0
    type_hits: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for row in rows:
        result = row["methods"][method]
        selected += int(result["selected_peak_count"])
        hits += int(result["peak_hit_count"])
        hit_indices = set(result["hit_interval_indices"])
        segment_hits += len(hit_indices)
        segment_count += len(row["intervals"])
        trajectory_hits += bool(hit_indices)
        for index, interval in enumerate(row["intervals"]):
            failure_type = interval["failure_type"]
            type_counts[failure_type] += 1
            type_hits[failure_type] += index in hit_indices
    return {
        "peak_precision": hits / selected if selected else 0.0,
        "segment_recall": (
            segment_hits / segment_count if segment_count else 0.0
        ),
        "trajectory_recall": (
            trajectory_hits / len(rows) if rows else 0.0
        ),
        "selected_peak_count": selected,
        "segment_recall_by_review_label": {
            label: type_hits[label] / count
            for label, count in sorted(type_counts.items())
        },
        "segment_counts_by_review_label": dict(type_counts),
    }


def _randomized_interior_null(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    precisions = []
    recalls = []
    trajectory_recalls = []
    for _ in range(repetitions):
        selected = hits = segment_hits = segment_count = trajectory_hits = 0
        for row in rows:
            duration = float(row["duration_sec"])
            peaks = rng.uniform(0.0, duration, top_k)
            intervals = [
                ReviewedInterval(
                    interval["label"],
                    float(interval["start_sec"]),
                    float(interval["end_sec"]),
                )
                for interval in row["intervals"]
            ]
            peak_hits = sum(_inside_any(float(peak), intervals) for peak in peaks)
            hit_segments = sum(
                any(interval.start_sec <= peak < interval.end_sec for peak in peaks)
                for interval in intervals
            )
            selected += top_k
            hits += peak_hits
            segment_hits += hit_segments
            segment_count += len(intervals)
            trajectory_hits += hit_segments > 0
        precisions.append(hits / selected if selected else 0.0)
        recalls.append(segment_hits / segment_count if segment_count else 0.0)
        trajectory_recalls.append(
            trajectory_hits / len(rows) if rows else 0.0
        )
    return {
        "peak_precision": float(np.mean(precisions)),
        "segment_recall": float(np.mean(recalls)),
        "trajectory_recall": float(np.mean(trajectory_recalls)),
    }


def _export_gating_orientation(
    bundle: Any,
    intervals: tuple[ReviewedInterval, ...],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    stream = next(
        (item for item in bundle.streams if item.name == "gripper_action"),
        None,
    )
    timestamps, vectors = extract_vector_series(stream, ("gripper_action",))
    if not timestamps or not vectors:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_times = [
        intervals[0].start_sec + offset
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ]
    fig = plt.figure(figsize=(18, 7))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.0, 1.5))
    axis = fig.add_subplot(grid[0, :])
    width = min(len(vector) for vector in vectors)
    for dimension in range(width):
        axis.plot(
            timestamps,
            [vector[dimension] for vector in vectors],
            label=f"commanded gripper {dimension}",
        )
    for interval in intervals:
        axis.axvspan(interval.start_sec, interval.end_sec, alpha=0.15)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("time (s)")
    axis.legend()
    frames = [
        frame for frame in bundle.camera_frames if frame.camera_name == "top"
    ]
    for column, timestamp in enumerate(frame_times):
        image_axis = fig.add_subplot(grid[1, column])
        nearest = min(
            frames,
            key=lambda frame: abs(float(frame.timestamp_sec) - timestamp),
        )
        with Image.open(nearest.image_path) as image:
            image_axis.imshow(image.convert("RGB"))
        image_axis.set_title(f"{nearest.timestamp_sec:.2f}s")
        image_axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _export_failure_qa(
    bundle: Any,
    intervals: tuple[ReviewedInterval, ...],
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    import cv2

    metric_config = config["metrics"]["object_interaction_motion"]
    segmentation_config = metric_config.get("segmentation", {})
    tracking_config = metric_config.get("tracking", {})
    segmenter = GroundedSAM2InteractionSegmenter(segmentation_config)
    tracker = CoTrackerRunner(tracking_config)
    frames = sorted(
        (
            frame
            for frame in bundle.camera_frames
            if frame.camera_name == "top"
        ),
        key=lambda frame: frame.timestamp_sec,
    )
    if not frames:
        return
    resize = tuple(metric_config.get("tracker_resize", [384, 384]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for interval_index, interval in enumerate(intervals, start=1):
        target_times = [
            interval.start_sec - 0.5,
            interval.start_sec,
            interval.start_sec + 0.5,
        ]
        selected = [
            min(
                frames,
                key=lambda frame: abs(frame.timestamp_sec - target),
            )
            for target in target_times
        ]
        images = [
            cv2.resize(cv2.imread(str(frame.image_path)), resize)
            for frame in selected
        ]
        if any(image is None for image in images):
            continue
        segmenter.reset_sequence()
        masks = segmenter.masks_for_paths(
            [frame.image_path for frame in selected]
        )
        object_masks = [
            cv2.resize(
                np.asarray(mask, dtype=np.uint8),
                resize,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks.object_masks
        ]
        gripper_masks = [
            cv2.resize(
                np.asarray(mask, dtype=np.uint8),
                resize,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks.gripper_masks
        ]
        if not object_masks or not gripper_masks:
            continue
        object_points = _sample_mask_points(
            _erode_mask(object_masks[0], 2),
            int(metric_config.get("object_point_count", 48)),
        )
        gripper_points = _sample_mask_points(
            _erode_mask(gripper_masks[0], 2),
            int(metric_config.get("gripper_point_count", 24)),
        )
        foreground = _dilate_mask(
            object_masks[0] | gripper_masks[0],
            6,
        )
        background_points = _sample_mask_points(
            ~foreground,
            int(metric_config.get("background_point_count", 96)),
            border_fraction=0.05,
        )
        if not len(object_points) or not len(gripper_points):
            continue
        queries = np.concatenate(
            [object_points, gripper_points, background_points]
        )
        video_rgb = np.asarray(
            [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        )
        tracks, visibility = tracker.track(video_rgb, queries)
        object_end = len(object_points)
        gripper_end = object_end + len(gripper_points)
        for frame_index, (frame, image) in enumerate(zip(selected, images)):
            overlay = image.copy()
            overlay[object_masks[frame_index]] = (
                0.45 * overlay[object_masks[frame_index]]
                + 0.55 * np.asarray([0, 255, 255])
            ).astype(np.uint8)
            overlay[gripper_masks[frame_index]] = (
                0.45 * overlay[gripper_masks[frame_index]]
                + 0.55 * np.asarray([255, 255, 0])
            ).astype(np.uint8)
            for point_index, point in enumerate(tracks[frame_index]):
                if not visibility[frame_index, point_index]:
                    continue
                color = (
                    (0, 255, 255)
                    if point_index < object_end
                    else (
                        (255, 255, 0)
                        if point_index < gripper_end
                        else (0, 255, 0)
                    )
                )
                cv2.circle(
                    overlay,
                    tuple(np.rint(point).astype(int)),
                    2,
                    color,
                    -1,
                )
            visible_fraction = float(
                visibility[frame_index, :object_end].mean()
            )
            cv2.putText(
                overlay,
                f"t={frame.timestamp_sec:.3f}s object_vis={visible_fraction:.2f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                str(
                    output_dir
                    / f"failure_{interval_index:02d}_{frame_index - 1:+d}.jpg"
                ),
                overlay,
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


def _inside_any(
    timestamp: float,
    intervals: Iterable[ReviewedInterval],
) -> bool:
    return any(
        interval.start_sec <= timestamp < interval.end_sec
        for interval in intervals
    )


def _parse_timestamp(value: str) -> float:
    minutes, seconds = value.split(":", 1)
    return 60.0 * int(minutes) + float(seconds)


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


if __name__ == "__main__":
    main()
