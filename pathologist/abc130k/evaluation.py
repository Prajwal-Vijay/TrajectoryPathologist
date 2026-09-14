"""Failure-localization evaluation for annotated ABC-130K trajectories."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.adapters.abc130k_bundle import (
    ABC130KBundleAdapter,
    ABC130KSegment,
    load_abc130k_segments,
)
from pathologist.config import deep_merge, load_pipeline_config
from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.metrics.surprise import DEFAULT_SURPRISE_SOURCE_METRICS
from pathologist.qa.robofac_selectors import (
    select_kite_flow_keyframes,
    select_uniform_keyframes,
)
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


DEFAULT_SELECTORS = ("uniform", "kite_flow", "metric")
ABC_CAMERA_NAMES = ("top", "left_wrist", "right_wrist")
ABC_PRISM_SOURCE_METRICS = tuple(DEFAULT_SURPRISE_SOURCE_METRICS)
ABC_VISUAL_FLOW_RESIZE = (128, 128)
ABC_VISUAL_FLOW_SPATIAL_AGGREGATION = "mean"
ABC_ROBOT_MASK_CHUNK_SIZE_FRAMES = 20
ABC_ROBOT_MASK_DUPLICATE_IOU_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class ABCEpisodeRecord:
    """Paths and labels needed to decide whether an episode is evaluable."""

    episode_path: Path
    annotation_path: Path
    task_name: str
    bundle_id: str
    origin_ns: int
    end_ns: int
    segments: tuple[ABC130KSegment, ...]

    @property
    def failed_segments(self) -> list[ABC130KSegment]:
        return [segment for segment in self.segments if segment.failure]


def run_abc130k_eval_pipeline(
    *,
    data_root: str | Path = "ABC-130k_subset/data/train",
    output_dir: str | Path = "ABC130K_failure_eval",
    top_k: int = 5,
    selectors: list[str] | tuple[str, ...] | None = None,
    pad_selectors: list[str] | tuple[str, ...] | None = None,
    camera_names: list[str] | tuple[str, ...] | None = None,
    camera_sample_hz: float | None = None,
    robot_downsample_hz: float | None = 50.0,
    frame_cache_dir: str | Path | None = None,
    metric_ranking: str = "persistence",
    smoothing_window: int = 5,
    visual_flow_frame_stride: int = 2,
    kite_frame_stride: int = 2,
    robot_mask_visual_flow: bool = False,
    robot_mask_prompt: str = "robot arm. robot gripper.",
    robot_mask_box_threshold: float = 0.35,
    robot_mask_dilation_px: int = 8,
    robot_mask_cache_dir: str | Path | None = None,
    one_per_task: bool = True,
    max_episodes: int | None = None,
    config_path: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    resume: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compare PRISM, Uniform, and KITE at K on exact ``mistake`` intervals."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1.")
    if visual_flow_frame_stride < 1:
        raise ValueError("visual_flow_frame_stride must be at least 1.")
    if kite_frame_stride < 1:
        raise ValueError("kite_frame_stride must be at least 1.")
    if not 0.0 <= robot_mask_box_threshold <= 1.0:
        raise ValueError("robot_mask_box_threshold must be in [0, 1].")
    if robot_mask_dilation_px < 0:
        raise ValueError("robot_mask_dilation_px cannot be negative.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples cannot be negative.")
    resolved_selectors = _resolve_selectors(selectors)
    resolved_pad_selectors = _resolve_pad_selectors(pad_selectors)
    resolved_camera_names = _resolve_camera_names(camera_names)
    inactive_pad_selectors = set(resolved_pad_selectors) - set(resolved_selectors)
    if inactive_pad_selectors:
        names = ", ".join(sorted(inactive_pad_selectors))
        raise ValueError(f"Cannot pad selectors that are not enabled: {names}")
    resolved_ranking = metric_ranking.strip().lower()
    if resolved_ranking not in {"area", "persistence"}:
        raise ValueError("metric_ranking must be area or persistence.")

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Could not find ABC-130K data root: {root}")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_robot_mask_cache_dir = (
        Path(robot_mask_cache_dir).expanduser().resolve()
        if robot_mask_cache_dir is not None
        else output / "robot_mask_cache"
    )
    frame_cache = (
        Path(frame_cache_dir).expanduser().resolve()
        if frame_cache_dir is not None
        else output / "frame_cache"
    )
    frame_cache.mkdir(parents=True, exist_ok=True)
    eval_path = output / "abc130k_eval.jsonl"
    manifest_path = output / "abc130k_manifest.json"
    stats_path = output / "abc130k_stats.json"
    summary_path = output / "run_summary.json"

    records = discover_abc130k_episodes(root)
    failure_records = [record for record in records if record.failed_segments]
    if one_per_task:
        failure_records = _one_failure_episode_per_task(failure_records)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive when provided.")
        failure_records = failure_records[:max_episodes]

    config = deep_merge(
        load_pipeline_config(config_path=config_path),
        _abc_config_override(
            camera_names=resolved_camera_names,
            top_k=top_k,
            metric_ranking=resolved_ranking,
            smoothing_window=smoothing_window,
            visual_flow_frame_stride=visual_flow_frame_stride,
            kite_frame_stride=kite_frame_stride,
            robot_mask_visual_flow=robot_mask_visual_flow,
            robot_mask_prompt=robot_mask_prompt,
            robot_mask_box_threshold=robot_mask_box_threshold,
            robot_mask_dilation_px=robot_mask_dilation_px,
            robot_mask_cache_dir=resolved_robot_mask_cache_dir,
        ),
    )
    if resume and summary_path.is_file():
        _validate_resume_settings(
            json.loads(summary_path.read_text(encoding="utf-8")),
            {
                "top_k": top_k,
                "selectors": resolved_selectors,
                "pad_selectors": resolved_pad_selectors,
                "camera_names": resolved_camera_names,
                "camera_sample_hz": camera_sample_hz,
                "robot_downsample_hz": robot_downsample_hz,
                "frame_cache_dir": str(frame_cache),
                "metric_ranking": resolved_ranking,
                "smoothing_window": smoothing_window,
                "visual_flow_frame_stride": visual_flow_frame_stride,
                "kite_frame_stride": kite_frame_stride,
                "robot_mask_visual_flow": robot_mask_visual_flow,
                "robot_mask_prompt": robot_mask_prompt,
                "robot_mask_box_threshold": robot_mask_box_threshold,
                "robot_mask_dilation_px": robot_mask_dilation_px,
                "robot_mask_cache_dir": str(resolved_robot_mask_cache_dir),
                "visual_flow_resize": list(ABC_VISUAL_FLOW_RESIZE),
                "visual_flow_spatial_aggregation": (
                    ABC_VISUAL_FLOW_SPATIAL_AGGREGATION
                ),
                "robot_mask_chunk_size_frames": ABC_ROBOT_MASK_CHUNK_SIZE_FRAMES,
                "robot_mask_duplicate_iou_threshold": (
                    ABC_ROBOT_MASK_DUPLICATE_IOU_THRESHOLD
                ),
                "one_per_task": one_per_task,
                "prism_source_metrics": list(ABC_PRISM_SOURCE_METRICS),
                "force_telemetry_used": False,
            },
        )
    existing_rows = _read_jsonl(eval_path) if resume else []
    requested = set(resolved_selectors)
    completed_by_bundle: dict[str, set[str]] = defaultdict(set)
    for row in existing_rows:
        completed_by_bundle[str(row.get("bundle_id", ""))].add(
            str(row.get("selector_name", ""))
        )
    completed = {
        bundle_id
        for bundle_id, names in completed_by_bundle.items()
        if requested.issubset(names)
    }
    rows = [
        row
        for row in existing_rows
        if str(row.get("bundle_id", "")) in completed
        and str(row.get("selector_name", "")) in requested
    ]
    if not resume:
        eval_path.write_text("", encoding="utf-8")

    manifest: list[dict[str, Any]] = []
    if resume and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            manifest = [
                entry
                for entry in payload
                if isinstance(entry, dict) and str(entry.get("bundle_id", "")) in completed
            ]

    summary_base = {
        "status": "running",
        "data_root": str(root),
        "output_dir": str(output),
        "top_k": top_k,
        "selectors": resolved_selectors,
        "pad_selectors": resolved_pad_selectors,
        "camera_names": resolved_camera_names,
        "camera_sample_hz": camera_sample_hz,
        "robot_downsample_hz": robot_downsample_hz,
        "frame_cache_dir": str(frame_cache),
        "metric_ranking": resolved_ranking,
        "smoothing_window": smoothing_window,
        "visual_flow_frame_stride": visual_flow_frame_stride,
        "kite_frame_stride": kite_frame_stride,
        "robot_mask_visual_flow": robot_mask_visual_flow,
        "robot_mask_prompt": robot_mask_prompt,
        "robot_mask_box_threshold": robot_mask_box_threshold,
        "robot_mask_dilation_px": robot_mask_dilation_px,
        "robot_mask_cache_dir": str(resolved_robot_mask_cache_dir),
        "visual_flow_resize": list(ABC_VISUAL_FLOW_RESIZE),
        "visual_flow_spatial_aggregation": ABC_VISUAL_FLOW_SPATIAL_AGGREGATION,
        "robot_mask_chunk_size_frames": ABC_ROBOT_MASK_CHUNK_SIZE_FRAMES,
        "robot_mask_duplicate_iou_threshold": (
            ABC_ROBOT_MASK_DUPLICATE_IOU_THRESHOLD
        ),
        "prism_source_metrics": list(ABC_PRISM_SOURCE_METRICS),
        "force_telemetry_used": False,
        "one_per_task": one_per_task,
        "annotated_episode_count": len(records),
        "failure_episode_count_before_task_deduplication": sum(
            1 for record in records if record.failed_segments
        ),
        "selected_failure_episode_count": len(failure_records),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "eval_path": str(eval_path),
        "stats_path": str(stats_path),
    }
    _write_json(summary_path, summary_base)

    adapter = ABC130KBundleAdapter()
    for index, record in enumerate(failure_records, start=1):
        if record.bundle_id in completed:
            _log(verbose, f"[{index}/{len(failure_records)}] resume skip {record.bundle_id}")
            continue
        _log(
            verbose,
            f"[{index}/{len(failure_records)}] {record.task_name}/{record.bundle_id}",
        )
        try:
            bundle = adapter.load(
                record.episode_path,
                record.annotation_path,
                frame_cache_dir=frame_cache / record.task_name / record.bundle_id,
                camera_name=None,
                camera_names=resolved_camera_names,
                camera_sample_hz=camera_sample_hz,
                robot_downsample_hz=robot_downsample_hz,
                extract_frames=True,
            )
            metric_profile = (
                MetricEngine(config=config).compute(bundle)
                if "metric" in resolved_selectors
                else None
            )
            proposals = (
                _metric_selector(config, top_k).select(
                    metric_profile,
                    trajectory_bundle=bundle,
                    ecod_aggregate_path=None,
                )
                if metric_profile is not None
                else []
            )
            episode_rows = [
                _evaluate_selector(
                    selector_name=name,
                    bundle=bundle,
                    proposals=proposals,
                    top_k=top_k,
                    config=config,
                    pad_selectors=set(resolved_pad_selectors),
                )
                for name in resolved_selectors
            ]
            _append_jsonl(eval_path, episode_rows)
            rows.extend(episode_rows)
            manifest.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "task_name": record.task_name,
                    "episode_path": str(record.episode_path),
                    "annotation_path": str(record.annotation_path),
                    "duration_sec": bundle.duration_sec,
                    "failed_segment_count": len(record.failed_segments),
                    "decoded_frame_count": len(bundle.camera_frames),
                    "stream_counts": {
                        stream.name: len(stream.timestamps_sec) for stream in bundle.streams
                    },
                    "metric_availability": (
                        _metric_availability(metric_profile) if metric_profile is not None else {}
                    ),
                    "selected_peak_count_by_selector": {
                        row["selector_name"]: row["selected_peak_count"]
                        for row in episode_rows
                    },
                    "status": "complete",
                }
            )
        except Exception as exc:
            manifest.append(
                {
                    "bundle_id": record.bundle_id,
                    "task_name": record.task_name,
                    "episode_path": str(record.episode_path),
                    "annotation_path": str(record.annotation_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(verbose, f"  ERROR {type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        interim = aggregate_abc130k_stats(rows, bootstrap_samples=0, seed=seed)
        _write_json(stats_path, interim)
        completed_bundle_count = len(
            {
                str(row.get("bundle_id", ""))
                for row in rows
                if row.get("bundle_id")
            }
        )
        _log(
            verbose,
            _running_metrics_message(
                interim,
                selectors=resolved_selectors,
                completed_count=completed_bundle_count,
                total_count=len(failure_records),
            ),
        )
        _write_json(
            summary_path,
            {
                **summary_base,
                "processed_episode_count": sum(
                    entry.get("status") == "complete" for entry in manifest
                ),
                "error_count": sum(entry.get("status") == "error" for entry in manifest),
            },
        )

    stats = aggregate_abc130k_stats(
        rows,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    _write_json(manifest_path, manifest)
    _write_json(stats_path, stats)
    summary = {
        **summary_base,
        "status": "complete",
        "processed_episode_count": sum(entry.get("status") == "complete" for entry in manifest),
        "error_count": sum(entry.get("status") == "error" for entry in manifest),
        "evaluation_row_count": len(rows),
        "stats": stats,
    }
    _write_json(summary_path, summary)
    return summary


def discover_abc130k_episodes(data_root: str | Path) -> list[ABCEpisodeRecord]:
    """Discover complete episode/annotation pairs and decode their small label files."""

    root = Path(data_root).expanduser().resolve()
    make_reader = _mcap_reader()
    output: list[ABCEpisodeRecord] = []
    for episode_path in sorted(root.glob("*/*/episode.mcap")):
        annotation_path = episode_path.with_name("annotation.mcap")
        if not annotation_path.is_file():
            continue
        with episode_path.open("rb") as handle:
            summary = make_reader(handle).get_summary()
        if summary is None or summary.statistics is None:
            continue
        origin_ns = int(summary.statistics.message_start_time)
        end_ns = int(summary.statistics.message_end_time)
        segments = load_abc130k_segments(
            annotation_path,
            episode_origin_ns=origin_ns,
            episode_end_ns=end_ns,
        )
        output.append(
            ABCEpisodeRecord(
                episode_path=episode_path,
                annotation_path=annotation_path,
                task_name=episode_path.parent.parent.name,
                bundle_id=episode_path.parent.name,
                origin_ns=origin_ns,
                end_ns=end_ns,
                segments=tuple(segments),
            )
        )
    return output


def aggregate_abc130k_stats(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate pooled and macro localization metrics by selector."""

    by_selector: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_selector[str(row["selector_name"])].append(row)
    output: dict[str, Any] = {}
    for selector_name, selector_rows in sorted(by_selector.items()):
        point = _aggregate_rows(selector_rows)
        ci = (
            _bootstrap_rows(selector_rows, bootstrap_samples=bootstrap_samples, seed=seed)
            if bootstrap_samples
            else {}
        )
        output[selector_name] = {**point, "clustered_bootstrap_95ci": ci}
    output["failure_coverage"] = _failure_coverage(rows)
    if "metric" in by_selector:
        output["paired_differences"] = {
            f"metric_minus_{baseline}": _paired_bootstrap_difference(
                by_selector,
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            for baseline in ("uniform", "kite_flow")
            if baseline in by_selector
        }
    return output


def _evaluate_selector(
    *,
    selector_name: str,
    bundle: TrajectoryBundle,
    proposals: list[KeyEventProposal],
    top_k: int,
    config: dict[str, Any],
    pad_selectors: set[str] | None = None,
) -> dict[str, Any]:
    padded_selectors = pad_selectors or set()
    if selector_name == "metric":
        selected = [
            _proposal_peak(proposal, rank)
            for rank, proposal in enumerate(proposals[:top_k], start=1)
        ]
        method_selected_count = len(selected)
        if selector_name in padded_selectors:
            selected = _pad_selected_peaks_uniformly(
                selected,
                bundle.camera_frames,
                top_k,
            )
    else:
        frames: list[CameraFrameRef]
        kite_selection_stats: dict[str, Any] = {}
        if selector_name == "uniform":
            frames = select_uniform_keyframes(bundle.camera_frames, top_k)
            method_selected_count = len(frames)
            method_frame_keys = {
                (frame.camera_name, frame.frame_index, frame.timestamp_sec)
                for frame in frames
            }
        elif selector_name == "kite_flow":
            frames = select_kite_flow_keyframes(
                bundle.camera_frames,
                top_k,
                config=config,
                selection_stats=kite_selection_stats,
                pad_to_budget=selector_name in padded_selectors,
            )
            method_selected_count = int(
                kite_selection_stats.get("method_frame_count", len(frames))
            )
            method_frame_keys = {
                tuple(key)
                for key in kite_selection_stats.get("method_frame_keys", [])
            }
        else:
            raise ValueError(f"Unsupported ABC selector: {selector_name}")
        selected = [
            {
                "rank": rank,
                "peak_sec": frame.timestamp_sec,
                "camera_name": frame.camera_name,
                "frame_index": frame.frame_index,
                "image_path": str(frame.image_path),
                "selection_source": (
                    selector_name
                    if (frame.camera_name, frame.frame_index, frame.timestamp_sec)
                    in method_frame_keys
                    else "uniform_padding"
                ),
            }
            for rank, frame in enumerate(frames[:top_k], start=1)
        ]

    failures = [
        segment
        for segment in bundle.raw_metadata.get("abc130k_segments", [])
        if isinstance(segment, dict) and segment.get("failure") is True
    ]
    peak_times = [float(item["peak_sec"]) for item in selected]
    matched_peak_times = [
        peak for peak in peak_times if any(_contains(segment, peak) for segment in failures)
    ]
    matched_failures = [
        {
            **segment,
            "matched_peak_secs": [peak for peak in peak_times if _contains(segment, peak)],
        }
        for segment in failures
        if any(_contains(segment, peak) for peak in peak_times)
    ]
    selected_count = len(peak_times)
    failure_count = len(failures)
    return {
        "bundle_id": bundle.bundle_id,
        "task_name": bundle.raw_metadata.get("task_name"),
        "episode_path": str(bundle.root_path),
        "selector_name": selector_name,
        "top_k": top_k,
        "duration_sec": bundle.duration_sec,
        "selected_peaks": selected,
        "selected_peak_count": selected_count,
        "method_selected_peak_count": method_selected_count,
        "uniform_padded_peak_count": max(0, selected_count - method_selected_count),
        "selected_peak_hit_count": len(matched_peak_times),
        "point_precision_at_k": (
            len(matched_peak_times) / selected_count if selected_count else None
        ),
        "failed_segments": failures,
        "failed_segment_count": failure_count,
        "hit_failed_segment_count": len(matched_failures),
        "failed_segment_recall_at_k": (
            len(matched_failures) / failure_count if failure_count else None
        ),
        "episode_hit_at_k": bool(matched_failures) if failure_count else None,
        "matched_peak_secs": matched_peak_times,
        "matched_failed_segments": matched_failures,
    }


def _proposal_peak(proposal: KeyEventProposal, rank: int) -> dict[str, Any]:
    peak_sec = None
    metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
    if isinstance(metadata.get("peak_sec"), (int, float)):
        peak_sec = float(metadata["peak_sec"])
    if peak_sec is None:
        for event in proposal.metric_events:
            if isinstance(event.value, dict) and isinstance(event.value.get("peak_sec"), (int, float)):
                peak_sec = float(event.value["peak_sec"])
                break
    if peak_sec is None:
        peak_sec = float(proposal.time_window.center_sec)
    return {
        "rank": rank,
        "peak_sec": peak_sec,
        "start_sec": proposal.time_window.start_sec,
        "end_sec": proposal.time_window.end_sec,
        "rank_score": metadata.get("rank_score"),
        "rank_score_type": metadata.get("rank_score_type"),
        "selection_source": "metric",
    }


def _pad_selected_peaks_uniformly(
    selected: list[dict[str, Any]],
    frames: list[CameraFrameRef],
    top_k: int,
) -> list[dict[str, Any]]:
    """Pad proposal timestamps uniformly over all remaining trajectory frames."""

    chosen = [dict(item) for item in selected[:top_k]]
    if len(chosen) >= top_k:
        return chosen
    selected_frame_keys = {
        (
            str(item.get("camera_name")),
            item.get("frame_index"),
            float(item["peak_sec"]),
        )
        for item in chosen
        if item.get("camera_name") is not None
        and item.get("frame_index") is not None
        and isinstance(item.get("peak_sec"), (int, float))
    }
    selected_times = {
        float(item["peak_sec"])
        for item in chosen
        if isinstance(item.get("peak_sec"), (int, float))
    }
    remaining_frames = [
        frame
        for frame in frames
        if (
            frame.camera_name,
            frame.frame_index,
            frame.timestamp_sec,
        )
        not in selected_frame_keys
        and frame.timestamp_sec not in selected_times
    ]
    remaining_budget = top_k - len(chosen)
    padding = select_uniform_keyframes(remaining_frames, remaining_budget)
    for frame in padding:
        chosen.append(
            {
                "rank": len(chosen) + 1,
                "peak_sec": frame.timestamp_sec,
                "camera_name": frame.camera_name,
                "frame_index": frame.frame_index,
                "image_path": str(frame.image_path),
                "selection_source": "uniform_padding",
            }
        )
    return chosen


def _contains(segment: dict[str, Any], peak_sec: float) -> bool:
    return float(segment["start_sec"]) <= peak_sec < float(segment["end_sec"])


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory_count = len(rows)
    episode_hits = sum(bool(row.get("episode_hit_at_k")) for row in rows)
    failure_count = sum(int(row.get("failed_segment_count", 0)) for row in rows)
    failure_hits = sum(int(row.get("hit_failed_segment_count", 0)) for row in rows)
    peak_count = sum(int(row.get("selected_peak_count", 0)) for row in rows)
    peak_hits = sum(int(row.get("selected_peak_hit_count", 0)) for row in rows)
    macro_precision_values = [
        float(row["point_precision_at_k"])
        for row in rows
        if row.get("point_precision_at_k") is not None
    ]
    macro_recall_values = [
        float(row["failed_segment_recall_at_k"])
        for row in rows
        if row.get("failed_segment_recall_at_k") is not None
    ]
    return {
        "trajectory_count": trajectory_count,
        "episode_hits": episode_hits,
        "episode_recall_at_k": episode_hits / trajectory_count if trajectory_count else None,
        "failed_segment_count": failure_count,
        "failed_segment_hits": failure_hits,
        "failed_segment_recall_at_k": failure_hits / failure_count if failure_count else None,
        "selected_peak_count": peak_count,
        "selected_peak_hits": peak_hits,
        "point_precision_at_k": peak_hits / peak_count if peak_count else None,
        "macro_point_precision_at_k": (
            sum(macro_precision_values) / len(macro_precision_values)
            if macro_precision_values
            else None
        ),
        "macro_failed_segment_recall_at_k": (
            sum(macro_recall_values) / len(macro_recall_values)
            if macro_recall_values
            else None
        ),
    }


def _bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    if not rows:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    keys = (
        "episode_recall_at_k",
        "failed_segment_recall_at_k",
        "point_precision_at_k",
        "macro_point_precision_at_k",
        "macro_failed_segment_recall_at_k",
    )
    for _ in range(bootstrap_samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        aggregate = _aggregate_rows(sample)
        for key in keys:
            value = aggregate.get(key)
            if value is not None:
                values[key].append(float(value))
    return {key: _percentile_interval(values.get(key, [])) for key in keys}


def _paired_bootstrap_difference(
    by_selector: dict[str, list[dict[str, Any]]],
    *,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metric_by_id = {str(row["bundle_id"]): row for row in by_selector["metric"]}
    baseline_by_id = {str(row["bundle_id"]): row for row in by_selector[baseline]}
    bundle_ids = sorted(set(metric_by_id) & set(baseline_by_id))
    paired = {
        bundle_id: {
            "metric": metric_by_id[bundle_id],
            "baseline": baseline_by_id[bundle_id],
        }
        for bundle_id in bundle_ids
    }

    def differences(sample_ids: list[str]) -> dict[str, float]:
        metric_rows = [paired[bundle_id]["metric"] for bundle_id in sample_ids]
        baseline_rows = [paired[bundle_id]["baseline"] for bundle_id in sample_ids]
        metric_stats = _aggregate_rows(metric_rows)
        baseline_stats = _aggregate_rows(baseline_rows)
        return {
            "point_precision_at_k": (
                float(metric_stats["point_precision_at_k"])
                - float(baseline_stats["point_precision_at_k"])
            ),
            "failed_segment_recall_at_k": (
                float(metric_stats["failed_segment_recall_at_k"])
                - float(baseline_stats["failed_segment_recall_at_k"])
            ),
            "episode_recall_at_k": (
                float(metric_stats["episode_recall_at_k"])
                - float(baseline_stats["episode_recall_at_k"])
            ),
        }

    if not bundle_ids:
        return {"trajectory_count": 0, "differences": {}}
    observed = differences(bundle_ids)
    samples: dict[str, list[float]] = defaultdict(list)
    if bootstrap_samples:
        rng = random.Random(f"{seed}:{baseline}")
        for _ in range(bootstrap_samples):
            sampled_ids = [
                bundle_ids[rng.randrange(len(bundle_ids))] for _ in bundle_ids
            ]
            for key, value in differences(sampled_ids).items():
                samples[key].append(value)
    return {
        "trajectory_count": len(bundle_ids),
        "differences": {
            key: {
                "estimate": value,
                "clustered_bootstrap_95ci": _percentile_interval(samples.get(key, [])),
            }
            for key, value in observed.items()
        },
    }


def _failure_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique_rows.setdefault(str(row["bundle_id"]), row)
    valid = [
        row
        for row in unique_rows.values()
        if isinstance(row.get("duration_sec"), (int, float))
        and float(row["duration_sec"]) > 0
    ]
    failed_durations = [
        sum(
            float(segment["end_sec"]) - float(segment["start_sec"])
            for segment in row.get("failed_segments", [])
        )
        for row in valid
    ]
    total_duration = sum(float(row["duration_sec"]) for row in valid)
    total_failed_duration = sum(failed_durations)
    fractions = [
        failed_duration / float(row["duration_sec"])
        for row, failed_duration in zip(valid, failed_durations)
    ]
    return {
        "trajectory_count": len(valid),
        "total_duration_sec": total_duration,
        "total_failed_duration_sec": total_failed_duration,
        "duration_weighted_failure_fraction": (
            total_failed_duration / total_duration if total_duration else None
        ),
        "mean_trajectory_failure_fraction": (
            sum(fractions) / len(fractions) if fractions else None
        ),
    }


def _percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    ordered = sorted(values)
    low = ordered[round(0.025 * (len(ordered) - 1))]
    high = ordered[round(0.975 * (len(ordered) - 1))]
    return [low, high]


def _one_failure_episode_per_task(records: list[ABCEpisodeRecord]) -> list[ABCEpisodeRecord]:
    by_task: dict[str, list[ABCEpisodeRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_name].append(record)
    return [
        min(task_records, key=lambda record: (record.episode_path.stat().st_size, record.bundle_id))
        for _, task_records in sorted(by_task.items())
    ]


def _metric_selector(config: dict[str, Any], top_k: int) -> MetricEventSelector:
    selector_config = config.get("selector", {})
    flood = selector_config.get("surprise_flood", {})
    return MetricEventSelector(
        frame_stride_within_window=int(selector_config.get("frame_stride_within_window", 1)),
        visual_context_sec=float(selector_config.get("visual_context_sec", 0.5)),
        max_frames_per_camera=max(
            top_k,
            int(selector_config.get("max_frames_per_camera", top_k)),
        ),
        metric_packet_context_steps=0,
        surprise_flood_config=flood,
    )


def _abc_config_override(
    *,
    camera_names: list[str] | tuple[str, ...],
    top_k: int,
    metric_ranking: str,
    smoothing_window: int,
    visual_flow_frame_stride: int,
    kite_frame_stride: int,
    robot_mask_visual_flow: bool,
    robot_mask_prompt: str,
    robot_mask_box_threshold: float,
    robot_mask_dilation_px: int,
    robot_mask_cache_dir: str | Path,
) -> dict[str, Any]:
    return {
        "metrics": {
            "action_discontinuity": {"ignored_dimensions": []},
            "visual_flow": {
                "camera_names": list(camera_names),
                "frame_stride": visual_flow_frame_stride,
                "resize": list(ABC_VISUAL_FLOW_RESIZE),
                "spatial_aggregation": ABC_VISUAL_FLOW_SPATIAL_AGGREGATION,
                "robot_masking": {
                    "enabled": robot_mask_visual_flow,
                    "prompt": robot_mask_prompt,
                    "box_threshold": robot_mask_box_threshold,
                    "dilation_px": robot_mask_dilation_px,
                    "cache_dir": str(robot_mask_cache_dir),
                    "chunk_size_frames": ABC_ROBOT_MASK_CHUNK_SIZE_FRAMES,
                    "duplicate_iou_threshold": (
                        ABC_ROBOT_MASK_DUPLICATE_IOU_THRESHOLD
                    ),
                },
            },
        },
        "robofac_qa": {
            "kite_flow": {
                "frame_stride": kite_frame_stride,
            }
        },
        "selector": {
            "surprise_flood": {
                "max_proposals": top_k,
                "ranking": metric_ranking,
                "smoothing_window": smoothing_window,
            }
        },
        "surprise": {
            "source_metrics": list(ABC_PRISM_SOURCE_METRICS)
        },
    }


def _resolve_camera_names(
    camera_names: list[str] | tuple[str, ...] | None,
) -> list[str]:
    requested = list(camera_names) if camera_names else list(ABC_CAMERA_NAMES)
    output: list[str] = []
    for name in requested:
        if name not in ABC_CAMERA_NAMES:
            choices = ", ".join(ABC_CAMERA_NAMES)
            raise ValueError(
                f"Unsupported ABC-130K camera {name!r}; choose one of: {choices}"
            )
        if name not in output:
            output.append(name)
    return output


def _resolve_selectors(
    selectors: list[str] | tuple[str, ...] | None,
) -> list[str]:
    names = [str(name).lower() for name in (selectors or DEFAULT_SELECTORS)]
    aliases = {"prism": "metric", "kite": "kite_flow", "flow": "kite_flow"}
    output: list[str] = []
    for name in names:
        resolved = aliases.get(name, name)
        if resolved not in DEFAULT_SELECTORS:
            raise ValueError(f"Unsupported selector: {name}")
        if resolved not in output:
            output.append(resolved)
    return output


def _resolve_pad_selectors(
    selectors: list[str] | tuple[str, ...] | None,
) -> list[str]:
    names = [str(name).lower() for name in (selectors or ())]
    aliases = {"prism": "metric", "kite": "kite_flow", "flow": "kite_flow"}
    output: list[str] = []
    for name in names:
        resolved = aliases.get(name, name)
        if resolved not in {"metric", "kite_flow"}:
            raise ValueError(
                f"Unsupported padding selector: {name}; use prism and/or kite."
            )
        if resolved not in output:
            output.append(resolved)
    return output


def _validate_resume_settings(
    existing_summary: Any,
    requested: dict[str, Any],
) -> None:
    if not isinstance(existing_summary, dict):
        raise ValueError("Cannot resume: existing run_summary.json is not an object.")
    for key, value in requested.items():
        if key in existing_summary and existing_summary[key] != value:
            raise ValueError(
                f"Resume settings mismatch for {key}: "
                f"existing={existing_summary[key]!r} requested={value!r}"
            )


def _metric_availability(profile: Any) -> dict[str, bool]:
    return {
        metric.name: bool(metric.metadata.get("available", bool(metric.samples)))
        for metric in profile.metrics
    }


def _mcap_reader() -> Any:
    try:
        from mcap.reader import make_reader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ABC evaluation requires the mcap package.") from exc
    return make_reader


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            output.append(value)
    return output


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _running_metrics_message(
    stats: dict[str, Any],
    *,
    selectors: list[str],
    completed_count: int,
    total_count: int,
) -> str:
    """Format running precision and recall without bootstrap intervals."""

    selector_summaries = []
    for selector_name in selectors:
        selector_stats = stats.get(selector_name)
        if not isinstance(selector_stats, dict):
            continue
        point_precision = 100.0 * float(selector_stats.get("point_precision_at_k", 0.0))
        segment_recall = 100.0 * float(
            selector_stats.get("failed_segment_recall_at_k", 0.0)
        )
        trajectory_recall = 100.0 * float(
            selector_stats.get("episode_recall_at_k", 0.0)
        )
        selector_summaries.append(
            f"{selector_name}: P={point_precision:.1f}% "
            f"SegR={segment_recall:.1f}% TrajR={trajectory_recall:.1f}%"
        )
    joined = " | ".join(selector_summaries) or "no completed selector rows"
    return f"  running [{completed_count}/{total_count}] {joined}"
