"""Deterministic RoboFail top-K localization evaluation."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from pathologist.adapters.robofail_bundle import (
    RoboFailRealBundleAdapter,
    RoboFailSimBundleAdapter,
    load_real_world_task_map,
)
from pathologist.config import config_section, load_pipeline_config
from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac_selectors import select_kite_flow_keyframes, select_uniform_keyframes
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


DEFAULT_SELECTORS = ["uniform", "kite_flow", "metric"]


def run_robofail_eval_pipeline(
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path = "robofail_outputs",
    top_k: int = 5,
    selectors: list[str] | None = None,
    config_path: str | Path | None = None,
    max_episodes: int | None = None,
    extract_frames: bool = True,
    plot_ecod_aggregates: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run top-K timestamp localization evaluation on RoboFail episodes."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    resolved_split = split.lower().strip()
    if resolved_split not in {"sim", "real"}:
        raise ValueError("--split must be one of: sim, real")
    resolved_selectors = _resolve_selectors(selectors)

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Could not find RoboFail dataset root: {root}")
    if resolved_split == "real":
        _ensure_real_dependencies()

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "robofail_manifest.json"
    eval_path = output_path / "robofail_eval.jsonl"
    stats_path = output_path / "robofail_stats.json"
    summary_path = output_path / "run_summary.json"

    config = load_pipeline_config(config_path=config_path)
    if "real" == resolved_split:
        _set_visual_flow_camera_names(config, ["color"])
    elif "sim" == resolved_split:
        _set_visual_flow_camera_names(config, ["ego"])

    episode_paths = _discover_episode_paths(root, resolved_split)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    manifest: list[dict[str, Any]] = []
    stats_accumulator: dict[str, dict[str, Any]] = defaultdict(_empty_stats_bucket)
    eval_path.write_text("", encoding="utf-8")
    _write_checkpoint(
        manifest_path=manifest_path,
        stats_path=stats_path,
        summary_path=summary_path,
        manifest=manifest,
        stats_accumulator=stats_accumulator,
        episode_paths=episode_paths,
        split=resolved_split,
        top_k=top_k,
        selectors=resolved_selectors,
        plot_ecod_aggregates=plot_ecod_aggregates,
        output_path=output_path,
        eval_path=eval_path,
        status="running",
    )

    sim_adapter = RoboFailSimBundleAdapter()
    real_adapter = RoboFailRealBundleAdapter()
    tasks_metadata_path = root / "tasks_real_world.json"

    for index, episode_path in enumerate(episode_paths, start=1):
        _log(verbose, f"[{index}/{len(episode_paths)}] Loading {episode_path}")
        try:
            if resolved_split == "sim":
                bundle = sim_adapter.load(episode_path)
            else:
                bundle = real_adapter.load(
                    episode_path,
                    tasks_metadata_path=tasks_metadata_path,
                    frame_cache_dir=output_path / "frame_cache" / episode_path.name,
                    extract_frames=extract_frames,
                )
            gt_failure_steps_sec = parse_failure_steps(bundle.raw_metadata.get("gt_failure_step"))
            if not gt_failure_steps_sec:
                manifest.append(
                    _manifest_entry(
                        episode_path=episode_path,
                        bundle=bundle,
                        included=False,
                        skip_reason="missing_gt_failure_step",
                    )
                )
                _write_checkpoint(
                    manifest_path=manifest_path,
                    stats_path=stats_path,
                    summary_path=summary_path,
                    manifest=manifest,
                    stats_accumulator=stats_accumulator,
                    episode_paths=episode_paths,
                    split=resolved_split,
                    top_k=top_k,
                    selectors=resolved_selectors,
                    plot_ecod_aggregates=plot_ecod_aggregates,
                    output_path=output_path,
                    eval_path=eval_path,
                    status="running",
                )
                continue

            metric_profile = None
            proposals: list[KeyEventProposal] = []
            metric_error: str | None = None
            if "metric" in resolved_selectors:
                try:
                    metric_profile = MetricEngine(config=config).compute(bundle)
                    proposals = _selector_from_config(config).select(
                        metric_profile,
                        trajectory_bundle=bundle,
                        debug_timeline_path=output_path / "timeline_debug" / f"{bundle.bundle_id}.json",
                        ecod_aggregate_path=(
                            output_path / "ecod_aggregates" / bundle.bundle_id
                            if plot_ecod_aggregates
                            else None
                        ),
                    )
                except Exception as exc:
                    metric_error = f"{type(exc).__name__}: {exc}"

            rows = [
                _selector_evaluation_row(
                    selector_name=selector_name,
                    bundle=bundle,
                    gt_failure_steps_sec=gt_failure_steps_sec,
                    proposals=proposals,
                    top_k=top_k,
                    config=config,
                    metric_error=metric_error if selector_name == "metric" else None,
                )
                for selector_name in resolved_selectors
            ]
            _append_eval_rows(eval_path, rows)
            for row in rows:
                _update_stats(stats_accumulator, row)

            manifest_entry = _manifest_entry(
                episode_path=episode_path,
                bundle=bundle,
                included=True,
                skip_reason=None,
            )
            manifest_entry.update(
                {
                    "selectors": resolved_selectors,
                    "gt_failure_steps_sec": gt_failure_steps_sec,
                    "gt_failure_timestamp_count": len(gt_failure_steps_sec),
                    "metric_proposal_count": len(proposals),
                    "metric_count": len(metric_profile.metrics) if metric_profile is not None else 0,
                    "selected_peak_count_by_selector": {
                        row["selector_name"]: len(row["selected_peaks"])
                        for row in rows
                    },
                }
            )
            if metric_error is not None:
                manifest_entry["metric_error"] = metric_error
            manifest.append(manifest_entry)
            _write_checkpoint(
                manifest_path=manifest_path,
                stats_path=stats_path,
                summary_path=summary_path,
                manifest=manifest,
                stats_accumulator=stats_accumulator,
                episode_paths=episode_paths,
                split=resolved_split,
                top_k=top_k,
                selectors=resolved_selectors,
                plot_ecod_aggregates=plot_ecod_aggregates,
                output_path=output_path,
                eval_path=eval_path,
                status="running",
            )
            _log(
                verbose,
                _progress_message(
                    bundle_id=bundle.bundle_id,
                    stats=_finalize_stats(stats_accumulator),
                    selectors=resolved_selectors,
                ),
            )
        except Exception as exc:
            manifest.append(
                {
                    "episode_path": str(episode_path),
                    "bundle_id": episode_path.name,
                    "split": resolved_split,
                    "included": False,
                    "skip_reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_checkpoint(
                manifest_path=manifest_path,
                stats_path=stats_path,
                summary_path=summary_path,
                manifest=manifest,
                stats_accumulator=stats_accumulator,
                episode_paths=episode_paths,
                split=resolved_split,
                top_k=top_k,
                selectors=resolved_selectors,
                plot_ecod_aggregates=plot_ecod_aggregates,
                output_path=output_path,
                eval_path=eval_path,
                status="running",
            )
            _log(verbose, f"[{episode_path.name}] error: {type(exc).__name__}: {exc}")

    return _write_checkpoint(
        manifest_path=manifest_path,
        stats_path=stats_path,
        summary_path=summary_path,
        manifest=manifest,
        stats_accumulator=stats_accumulator,
        episode_paths=episode_paths,
        split=resolved_split,
        top_k=top_k,
        selectors=resolved_selectors,
        plot_ecod_aggregates=plot_ecod_aggregates,
        output_path=output_path,
        eval_path=eval_path,
        status="complete",
    )


def parse_failure_steps(value: Any) -> list[float]:
    """Parse RoboFail gt_failure_step values into seconds."""

    if value is None:
        return []
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if "," in stripped:
            parsed: list[float] = []
            for part in stripped.split(","):
                parsed.extend(parse_failure_steps(part))
            return parsed
        if ":" in stripped:
            parts = stripped.split(":")
            if not 2 <= len(parts) <= 3 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"Invalid gt_failure_step timestamp: {value!r}")
            numbers = [int(part) for part in parts]
            if len(numbers) == 2:
                minutes, seconds = numbers
                return [float(minutes * 60 + seconds)]
            hours, minutes, seconds = numbers
            return [float(hours * 3600 + minutes * 60 + seconds)]
        try:
            return [float(stripped)]
        except ValueError as exc:
            raise ValueError(f"Invalid gt_failure_step timestamp: {value!r}") from exc
    if isinstance(value, (list, tuple)):
        parsed = []
        for item in value:
            parsed.extend(parse_failure_steps(item))
        return parsed
    raise ValueError(f"Unsupported gt_failure_step value: {value!r}")


def _ensure_real_dependencies() -> None:
    try:
        import numcodecs  # noqa: F401
        import zarr  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboFail real evaluation requires zarr and numcodecs. "
            "Install them in the active environment, for example: conda install -c conda-forge zarr numcodecs"
        ) from exc


def _discover_episode_paths(dataset_root: Path, split: str) -> list[Path]:
    if split == "sim":
        sim_root = dataset_root / "sim_data"
        if not sim_root.is_dir():
            raise FileNotFoundError(f"Could not find RoboFail sim_data directory: {sim_root}")
        return sorted(
            [path.parent for path in sim_root.glob("*/*/task.json")],
            key=lambda path: _natural_path_key(path.relative_to(sim_root)),
        )

    real_root = dataset_root / "real_data" / "reflect_dataset" / "real_data"
    if not real_root.is_dir():
        raise FileNotFoundError(f"Could not find RoboFail real_data directory: {real_root}")
    task_map = load_real_world_task_map(dataset_root / "tasks_real_world.json")
    available = {
        path.name: path
        for path in real_root.iterdir()
        if path.is_dir() and (path / "replay_buffer.zarr").exists()
    }
    return [
        available[folder_name]
        for folder_name in sorted(task_map, key=_natural_string_key)
        if folder_name in available
    ]


def _selector_from_config(config: dict[str, Any]) -> MetricEventSelector:
    selector_config = config_section(config, "selector")
    surprise_flood_config = selector_config.get("surprise_flood", {})
    if not isinstance(surprise_flood_config, dict):
        surprise_flood_config = {}
    return MetricEventSelector(
        frame_stride_within_window=int(selector_config.get("frame_stride_within_window", 1)),
        visual_context_sec=float(selector_config.get("visual_context_sec", 0.5)),
        max_frames_per_camera=int(selector_config.get("max_frames_per_camera", 10)),
        metric_packet_context_steps=int(selector_config.get("metric_packet_context_steps", 10)),
        surprise_flood_config=surprise_flood_config,
    )


def _set_visual_flow_camera_names(config: dict[str, Any], camera_names: list[str]) -> None:
    metrics_config = config.setdefault("metrics", {})
    if not isinstance(metrics_config, dict):
        return
    visual_flow_config = metrics_config.setdefault("visual_flow", {})
    if not isinstance(visual_flow_config, dict):
        return
    visual_flow_config["camera_names"] = list(camera_names)


def _selector_evaluation_row(
    selector_name: str,
    bundle: TrajectoryBundle,
    gt_failure_steps_sec: list[float],
    proposals: list[KeyEventProposal],
    top_k: int,
    config: dict[str, Any],
    metric_error: str | None = None,
) -> dict[str, Any]:
    selected_peaks: list[dict[str, Any]]
    selector_error = metric_error
    if selector_name == "metric":
        selected_peaks = [_proposal_to_peak_dict(proposal) for proposal in proposals[:top_k]]
    else:
        try:
            if selector_name == "uniform":
                frames = select_uniform_keyframes(bundle.camera_frames, top_k)
            elif selector_name == "kite_flow":
                frames = select_kite_flow_keyframes(bundle.camera_frames, top_k, config=config)
            else:
                raise ValueError(f"Unsupported RoboFail selector: {selector_name}")
            selected_peaks = [
                _frame_to_peak_dict(frame, selector_name=selector_name, rank=rank)
                for rank, frame in enumerate(frames[:top_k], start=1)
            ]
        except Exception as exc:
            selected_peaks = []
            selector_error = f"{type(exc).__name__}: {exc}"

    row = _evaluation_row_from_selected_peaks(
        bundle=bundle,
        selector_name=selector_name,
        selected_peaks=selected_peaks,
        gt_failure_steps_sec=gt_failure_steps_sec,
        top_k=top_k,
    )
    if selector_error is not None:
        row["selector_error"] = selector_error
    return row


def _evaluation_row_from_selected_peaks(
    bundle: TrajectoryBundle,
    selector_name: str,
    selected_peaks: list[dict[str, Any]],
    gt_failure_steps_sec: list[float],
    top_k: int,
) -> dict[str, Any]:
    selected_peak_times = [_as_float_or_none(peak.get("peak_sec")) for peak in selected_peaks]
    selected_peak_times = [peak for peak in selected_peak_times if peak is not None]
    per_gt_errors = _closest_peak_errors(selected_peak_times, gt_failure_steps_sec)
    per_selected_peak_errors = _selected_peak_errors(selected_peak_times, gt_failure_steps_sec)
    scored_errors = [
        float(item["error_sec"])
        for item in per_gt_errors
        if isinstance(item.get("error_sec"), (int, float, bool))
    ]
    topk_errors = [
        float(item["error_sec"])
        for item in per_selected_peak_errors
        if isinstance(item.get("error_sec"), (int, float, bool))
    ]
    sum_abs_error_sec = sum(scored_errors) if scored_errors else None
    topk_sum_abs_error_sec = sum(topk_errors) if topk_errors else None
    duration_sec = _as_float_or_none(bundle.duration_sec)
    topk_within_duration_thresholds = _topk_within_duration_thresholds(topk_errors, duration_sec)
    return {
        "bundle_id": bundle.bundle_id,
        "episode_path": str(bundle.root_path) if bundle.root_path is not None else None,
        "split": bundle.raw_metadata.get("split"),
        "source_format": bundle.source_format,
        "duration_sec": duration_sec,
        "task_metadata": _task_metadata_for_row(bundle),
        "gt_failure_step": bundle.raw_metadata.get("gt_failure_step"),
        "gt_failure_steps_sec": gt_failure_steps_sec,
        "selector_name": selector_name,
        "top_k": top_k,
        "selected_peaks": selected_peaks,
        "per_gt_closest_peak": per_gt_errors,
        "per_selected_peak_nearest_gt": per_selected_peak_errors,
        "gt_failure_timestamp_count": len(gt_failure_steps_sec),
        "scored_gt_failure_timestamp_count": len(scored_errors),
        "missing_prediction_count": len(gt_failure_steps_sec) - len(scored_errors),
        "sum_abs_error_sec": sum_abs_error_sec,
        "mae_sec": (
            sum_abs_error_sec / len(scored_errors)
            if sum_abs_error_sec is not None and scored_errors
            else None
        ),
        "topk_selected_peak_count": len(topk_errors),
        "topk_expected_peak_count": top_k,
        "topk_missing_peak_count": max(top_k - len(topk_errors), 0),
        "topk_sum_abs_error_sec": topk_sum_abs_error_sec,
        "topk_mean_abs_error_sec": (
            topk_sum_abs_error_sec / len(topk_errors)
            if topk_sum_abs_error_sec is not None and topk_errors
            else None
        ),
        **topk_within_duration_thresholds,
    }


def _closest_peak_errors(
    selected_peak_times: list[float],
    gt_failure_steps_sec: list[float],
) -> list[dict[str, Any]]:
    output = []
    for gt_sec in gt_failure_steps_sec:
        if not selected_peak_times:
            output.append({"gt_sec": gt_sec, "closest_peak_sec": None, "error_sec": None})
            continue
        closest_peak_sec = min(selected_peak_times, key=lambda peak_sec: abs(peak_sec - gt_sec))
        output.append(
            {
                "gt_sec": gt_sec,
                "closest_peak_sec": closest_peak_sec,
                "error_sec": abs(closest_peak_sec - gt_sec),
            }
        )
    return output


def _selected_peak_errors(
    selected_peak_times: list[float],
    gt_failure_steps_sec: list[float],
) -> list[dict[str, Any]]:
    output = []
    for peak_sec in selected_peak_times:
        if not gt_failure_steps_sec:
            output.append({"peak_sec": peak_sec, "nearest_gt_sec": None, "error_sec": None})
            continue
        nearest_gt_sec = min(gt_failure_steps_sec, key=lambda gt_sec: abs(peak_sec - gt_sec))
        output.append(
            {
                "peak_sec": peak_sec,
                "nearest_gt_sec": nearest_gt_sec,
                "error_sec": abs(peak_sec - nearest_gt_sec),
            }
        )
    return output


def _topk_within_duration_thresholds(
    topk_errors: list[float],
    duration_sec: float | None,
) -> dict[str, Any]:
    if duration_sec is None or duration_sec <= 0:
        return {
            "topk_duration_threshold_peak_count": 0,
            "topk_within_5pct_duration_threshold_sec": None,
            "topk_within_5pct_duration_count": 0,
            "topk_within_5pct_duration_fraction": None,
            "topk_within_10pct_duration_threshold_sec": None,
            "topk_within_10pct_duration_count": 0,
            "topk_within_10pct_duration_fraction": None,
        }
    threshold_5pct = 0.05 * duration_sec
    threshold_10pct = 0.10 * duration_sec
    count = len(topk_errors)
    within_5pct = sum(1 for error in topk_errors if error <= threshold_5pct)
    within_10pct = sum(1 for error in topk_errors if error <= threshold_10pct)
    return {
        "topk_duration_threshold_peak_count": count,
        "topk_within_5pct_duration_threshold_sec": threshold_5pct,
        "topk_within_5pct_duration_count": within_5pct,
        "topk_within_5pct_duration_fraction": within_5pct / count if count else None,
        "topk_within_10pct_duration_threshold_sec": threshold_10pct,
        "topk_within_10pct_duration_count": within_10pct,
        "topk_within_10pct_duration_fraction": within_10pct / count if count else None,
    }


def _proposal_to_peak_dict(proposal: KeyEventProposal) -> dict[str, Any]:
    return {
        "event_id": proposal.event_id,
        "timeline_rank": proposal.metadata.get("timeline_rank"),
        "peak_sec": _proposal_peak_sec(proposal),
        "start_sec": proposal.time_window.start_sec,
        "end_sec": proposal.time_window.end_sec,
        "proposal_score": proposal.metadata.get("proposal_score"),
        "rank_score_type": proposal.metadata.get("rank_score_type"),
        "persistence": proposal.metadata.get("persistence"),
        "area_above_base": proposal.metadata.get("area_above_base"),
        "width_sec": proposal.metadata.get("width_sec"),
        "candidate_frame_count": len(proposal.candidate_frames),
    }


def _frame_to_peak_dict(frame: CameraFrameRef, selector_name: str, rank: int) -> dict[str, Any]:
    return {
        "event_id": f"{selector_name}_frame_{rank:03d}",
        "timeline_rank": rank,
        "peak_sec": frame.timestamp_sec,
        "start_sec": frame.timestamp_sec,
        "end_sec": frame.timestamp_sec,
        "proposal_score": None,
        "persistence": None,
        "area_above_base": None,
        "width_sec": 0.0,
        "candidate_frame_count": 1,
        "camera_name": frame.camera_name,
        "frame_index": frame.frame_index,
        "image_path": str(frame.image_path),
    }


def _proposal_peak_sec(proposal: KeyEventProposal) -> float | None:
    value = proposal.metadata.get("peak_sec")
    if isinstance(value, (int, float, bool)):
        return float(value)
    end_sec = proposal.time_window.end_sec
    if end_sec is not None:
        return (proposal.time_window.start_sec + end_sec) / 2.0
    return proposal.time_window.start_sec


def _task_metadata_for_row(bundle: TrajectoryBundle) -> dict[str, Any]:
    if bundle.raw_metadata.get("split") == "real":
        metadata = bundle.raw_metadata.get("task_metadata")
        return metadata if isinstance(metadata, dict) else {}
    metadata = bundle.raw_metadata.get("task")
    return metadata if isinstance(metadata, dict) else {}


def _empty_stats_bucket() -> dict[str, Any]:
    return {
        "trajectory_count": 0,
        "gt_failure_timestamp_count": 0,
        "scored_gt_failure_timestamp_count": 0,
        "sum_abs_error_sec": 0.0,
        "abs_errors_sec": [],
        "missing_prediction_count": 0,
        "topk_expected_peak_count": 0,
        "topk_selected_peak_count": 0,
        "topk_missing_peak_count": 0,
        "topk_sum_abs_error_sec": 0.0,
        "topk_abs_errors_sec": [],
        "topk_duration_threshold_peak_count": 0,
        "topk_within_5pct_duration_count": 0,
        "topk_within_10pct_duration_count": 0,
    }


def _update_stats(stats_accumulator: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    selector_name = str(row.get("selector_name", "metric"))
    bucket = stats_accumulator[selector_name]
    bucket["trajectory_count"] += 1
    bucket["gt_failure_timestamp_count"] += int(row.get("gt_failure_timestamp_count", 0))
    bucket["scored_gt_failure_timestamp_count"] += int(row.get("scored_gt_failure_timestamp_count", 0))
    bucket["missing_prediction_count"] += int(row.get("missing_prediction_count", 0))
    errors = [
        float(item["error_sec"])
        for item in row.get("per_gt_closest_peak", [])
        if isinstance(item, dict) and isinstance(item.get("error_sec"), (int, float, bool))
    ]
    bucket["sum_abs_error_sec"] += sum(errors)
    bucket["abs_errors_sec"].extend(errors)
    topk_errors = [
        float(item["error_sec"])
        for item in row.get("per_selected_peak_nearest_gt", [])
        if isinstance(item, dict) and isinstance(item.get("error_sec"), (int, float, bool))
    ]
    bucket["topk_expected_peak_count"] += int(row.get("topk_expected_peak_count", 0))
    bucket["topk_selected_peak_count"] += int(row.get("topk_selected_peak_count", len(topk_errors)))
    bucket["topk_missing_peak_count"] += int(row.get("topk_missing_peak_count", 0))
    bucket["topk_sum_abs_error_sec"] += sum(topk_errors)
    bucket["topk_abs_errors_sec"].extend(topk_errors)
    bucket["topk_duration_threshold_peak_count"] += int(
        row.get("topk_duration_threshold_peak_count", 0)
    )
    bucket["topk_within_5pct_duration_count"] += int(
        row.get("topk_within_5pct_duration_count", 0)
    )
    bucket["topk_within_10pct_duration_count"] += int(
        row.get("topk_within_10pct_duration_count", 0)
    )


def _finalize_stats(stats_accumulator: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    selector_names = list(DEFAULT_SELECTORS)
    selector_names.extend(sorted(set(stats_accumulator) - set(DEFAULT_SELECTORS)))
    for selector_name in selector_names:
        output[selector_name] = _finalize_stats_bucket(stats_accumulator.get(selector_name, {}))
    return output


def _finalize_stats_bucket(values: dict[str, Any]) -> dict[str, Any]:
    trajectory_count = int(values.get("trajectory_count", 0))
    gt_count = int(values.get("gt_failure_timestamp_count", 0))
    scored_gt_count = int(values.get("scored_gt_failure_timestamp_count", 0))
    sum_abs_error_sec = float(values.get("sum_abs_error_sec", 0.0))
    errors = [float(error) for error in values.get("abs_errors_sec", [])]
    topk_selected_peak_count = int(values.get("topk_selected_peak_count", 0))
    topk_expected_peak_count = int(values.get("topk_expected_peak_count", 0))
    topk_sum_abs_error_sec = float(values.get("topk_sum_abs_error_sec", 0.0))
    topk_errors = [float(error) for error in values.get("topk_abs_errors_sec", [])]
    topk_duration_threshold_peak_count = int(values.get("topk_duration_threshold_peak_count", 0))
    topk_within_5pct_duration_count = int(values.get("topk_within_5pct_duration_count", 0))
    topk_within_10pct_duration_count = int(values.get("topk_within_10pct_duration_count", 0))
    return {
        "trajectory_count": trajectory_count,
        "gt_failure_timestamp_count": gt_count,
        "scored_gt_failure_timestamp_count": scored_gt_count,
        "sum_abs_error_sec": sum_abs_error_sec,
        "mae_sec": sum_abs_error_sec / scored_gt_count if scored_gt_count else None,
        "median_abs_error_sec": median(errors) if errors else None,
        "missing_prediction_count": int(values.get("missing_prediction_count", 0)),
        "topk_expected_peak_count": topk_expected_peak_count,
        "topk_selected_peak_count": topk_selected_peak_count,
        "topk_missing_peak_count": int(values.get("topk_missing_peak_count", 0)),
        "topk_sum_abs_error_sec": topk_sum_abs_error_sec,
        "topk_mean_abs_error_sec": (
            topk_sum_abs_error_sec / topk_selected_peak_count
            if topk_selected_peak_count
            else None
        ),
        "topk_median_abs_error_sec": median(topk_errors) if topk_errors else None,
        "topk_duration_threshold_peak_count": topk_duration_threshold_peak_count,
        "topk_within_5pct_duration_count": topk_within_5pct_duration_count,
        "topk_within_5pct_duration_fraction": (
            topk_within_5pct_duration_count / topk_duration_threshold_peak_count
            if topk_duration_threshold_peak_count
            else None
        ),
        "topk_within_10pct_duration_count": topk_within_10pct_duration_count,
        "topk_within_10pct_duration_fraction": (
            topk_within_10pct_duration_count / topk_duration_threshold_peak_count
            if topk_duration_threshold_peak_count
            else None
        ),
    }


def _write_checkpoint(
    manifest_path: Path,
    stats_path: Path,
    summary_path: Path,
    manifest: list[dict[str, Any]],
    stats_accumulator: dict[str, dict[str, Any]],
    episode_paths: list[Path],
    split: str,
    top_k: int,
    selectors: list[str],
    output_path: Path,
    eval_path: Path,
    status: str,
    plot_ecod_aggregates: bool = False,
) -> dict[str, Any]:
    stats = _finalize_stats(stats_accumulator)
    summary = {
        "status": status,
        "dataset": "RoboFail",
        "split": split,
        "evaluation_unit": "episode",
        "input_count": len(episode_paths),
        "processed_count": len(manifest),
        "remaining_count": max(len(episode_paths) - len(manifest), 0),
        "included_count": sum(1 for entry in manifest if entry.get("included")),
        "skipped_count": sum(1 for entry in manifest if not entry.get("included")),
        "top_k": top_k,
        "selectors": selectors,
        "plot_ecod_aggregates": plot_ecod_aggregates,
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "eval_path": str(eval_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(stats_path, stats)
    _write_json_atomic(summary_path, summary)
    return summary


def _append_eval_rows(eval_path: Path, rows: list[dict[str, Any]]) -> None:
    with eval_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _manifest_entry(
    episode_path: Path,
    bundle: TrajectoryBundle,
    included: bool,
    skip_reason: str | None,
) -> dict[str, Any]:
    return {
        "episode_path": str(episode_path),
        "bundle_id": bundle.bundle_id,
        "split": bundle.raw_metadata.get("split"),
        "source_format": bundle.source_format,
        "duration_sec": bundle.duration_sec,
        "included": included,
        "skip_reason": skip_reason,
        "stream_names": [stream.name for stream in bundle.streams],
        "camera_frame_count": len(bundle.camera_frames),
        "gt_failure_step": bundle.raw_metadata.get("gt_failure_step"),
    }


def _progress_message(bundle_id: str, stats: dict[str, Any], selectors: list[str]) -> str:
    pieces = [f"[{bundle_id}]"]
    for selector_name in selectors:
        selector_stats = stats.get(selector_name, {})
        mae = selector_stats.get("mae_sec")
        count = selector_stats.get("scored_gt_failure_timestamp_count", 0)
        if mae is None:
            pieces.append(f"{selector_name}: mae=n/a n={count}")
        else:
            pieces.append(f"{selector_name}: mae={float(mae):.3f}s n={count}")
    return " ".join(pieces)


def _resolve_selectors(selectors: list[str] | None) -> list[str]:
    raw_selectors = selectors or DEFAULT_SELECTORS
    resolved = []
    for selector in raw_selectors:
        normalized = selector.lower().strip()
        if normalized == "flow":
            normalized = "kite_flow"
        if normalized not in DEFAULT_SELECTORS:
            raise ValueError(f"Unsupported RoboFail selector: {selector}")
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _natural_path_key(path: Path) -> list[int | str]:
    return _natural_string_key(path.as_posix())


def _natural_string_key(value: str) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", value):
        if part.isdigit():
            parts.append(int(part))
        elif part:
            parts.append(part)
    return parts


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)
