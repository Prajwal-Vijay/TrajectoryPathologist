"""Deterministic REASSEMBLE failure-segment recall evaluation."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.adapters.reassemble_bundle import ReassembleBundleAdapter
from pathologist.config import config_section, deep_merge, load_pipeline_config
from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac_selectors import select_kite_flow_keyframes, select_uniform_keyframes
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


DEFAULT_SELECTORS = ["uniform", "kite_flow", "metric"]


REASSEMBLE_CONFIG_OVERRIDE = {
    "metrics": {
        "visual_flow": {
            "camera_names": ["hand"],
            "frame_stride": 2,
            "resize": [128, 128],
        }
    }
}


def run_reassemble_eval_pipeline(
    h5_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path = "reassemble_outputs",
    top_k: int = 5,
    segment_level: str = "both",
    config_path: str | Path | None = None,
    robot_downsample_hz: float | None = None,
    camera_names: list[str] | None = None,
    include_event_camera: bool | None = None,
    extract_frames: bool | None = None,
    max_files: int | None = None,
    selectors: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run deterministic recall@K evaluation on REASSEMBLE HDF5 files."""

    if h5_path is None and data_root is None:
        raise ValueError("Pass either h5_path or data_root.")
    if h5_path is not None and data_root is not None:
        raise ValueError("Pass only one of h5_path or data_root, not both.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    resolved_segment_level = segment_level.lower()
    if resolved_segment_level not in {"high", "low", "both"}:
        raise ValueError("--segment-level must be one of: high, low, both")
    resolved_selectors = _resolve_selectors(selectors)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "reassemble_manifest.json"
    eval_path = output_path / "reassemble_eval.jsonl"
    stats_path = output_path / "reassemble_stats.json"
    alignment_path = output_path / "reassemble_alignment_debug.json"
    summary_path = output_path / "run_summary.json"
    selected_peaks_dir = output_path / "selected_peaks"
    selected_peaks_dir.mkdir(parents=True, exist_ok=True)

    config = deep_merge(load_pipeline_config(config_path=config_path), REASSEMBLE_CONFIG_OVERRIDE)
    reassemble_config = config_section(config, "reassemble")
    resolved_robot_downsample_hz = (
        float(robot_downsample_hz)
        if robot_downsample_hz is not None
        else float(reassemble_config.get("robot_downsample_hz", 50.0))
    )
    resolved_camera_names = camera_names or _string_list(reassemble_config.get("camera_names"), ["hand"])
    resolved_include_event_camera = (
        bool(include_event_camera)
        if include_event_camera is not None
        else bool(reassemble_config.get("include_event_camera", False))
    )
    resolved_extract_frames = (
        bool(extract_frames)
        if extract_frames is not None
        else bool(reassemble_config.get("extract_frames", True))
    )

    h5_paths = _resolve_h5_paths(h5_path=h5_path, data_root=data_root)
    if max_files is not None:
        h5_paths = h5_paths[:max_files]

    adapter = ReassembleBundleAdapter()
    manifest: list[dict[str, Any]] = []
    alignment_debug: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    stats_accumulator: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "trajectory_count": 0,
                "episode_hits": 0,
                "failed_segment_count": 0,
                "failed_segment_hits": 0,
            }
        )
    )

    eval_path.write_text("", encoding="utf-8")
    _write_checkpoint(
        manifest_path=manifest_path,
        stats_path=stats_path,
        alignment_path=alignment_path,
        summary_path=summary_path,
        manifest=manifest,
        alignment_debug=alignment_debug,
        stats_accumulator=stats_accumulator,
        h5_paths=h5_paths,
        top_k=top_k,
        segment_level=resolved_segment_level,
        selectors=resolved_selectors,
        output_path=output_path,
        eval_path=eval_path,
        status="running",
    )
    for index, current_h5_path in enumerate(h5_paths, start=1):
        _log(verbose, f"[{index}/{len(h5_paths)}] Loading {current_h5_path}")
        try:
            bundle = adapter.load(
                current_h5_path,
                frame_cache_dir=output_path / "frame_cache" / current_h5_path.stem,
                robot_downsample_hz=resolved_robot_downsample_hz,
                camera_names=resolved_camera_names,
                include_event_camera=resolved_include_event_camera,
                extract_frames=resolved_extract_frames,
            )
            manifest_entry = _manifest_entry(bundle, current_h5_path)
            if not bool(bundle.raw_metadata.get("has_failed_segment")):
                manifest_entry.update({"included": False, "skip_reason": "no_failed_segments"})
                manifest.append(manifest_entry)
                alignment_debug.append(_alignment_entry(bundle, status="skipped", reason="no_failed_segments"))
                _write_checkpoint(
                    manifest_path=manifest_path,
                    stats_path=stats_path,
                    alignment_path=alignment_path,
                    summary_path=summary_path,
                    manifest=manifest,
                    alignment_debug=alignment_debug,
                    stats_accumulator=stats_accumulator,
                    h5_paths=h5_paths,
                    top_k=top_k,
                    segment_level=resolved_segment_level,
                    selectors=resolved_selectors,
                    output_path=output_path,
                    eval_path=eval_path,
                    status="running",
                )
                _log(verbose, f"[{bundle.bundle_id}] skipped: no failed segments")
                continue

            manifest_entry.update({"included": True, "skip_reason": None})
            metric_profile = None
            proposals: list[KeyEventProposal] = []
            if "metric" in resolved_selectors:
                metric_profile = MetricEngine(config=config).compute(bundle)
                selector = _selector_from_config(config)
                proposals = selector.select(
                    metric_profile,
                    trajectory_bundle=bundle,
                    debug_timeline_path=output_path / "timeline_debug" / f"{bundle.bundle_id}.json",
                )
            rows = [
                _selector_evaluation_row(
                    selector_name=selector_name,
                    bundle=bundle,
                    proposals=proposals,
                    top_k=top_k,
                    requested_segment_level=resolved_segment_level,
                    config=config,
                )
                for selector_name in resolved_selectors
            ]
            _append_eval_rows(eval_path, rows)
            for row in rows:
                _update_stats(stats_accumulator, row)
                eval_rows.append(row)
            selected_peak_debug_error = None
            try:
                _write_selected_peak_debug(selected_peaks_dir, bundle, rows)
            except Exception as exc:
                selected_peak_debug_error = f"{type(exc).__name__}: {exc}"
            manifest_entry.update(
                {
                    "selectors": resolved_selectors,
                    "metric_proposal_count": len(proposals),
                    "selected_peak_count_by_selector": {
                        row["selector_name"]: len(row["selected_peaks"])
                        for row in rows
                    },
                    "metric_count": len(metric_profile.metrics) if metric_profile is not None else 0,
                }
            )
            if selected_peak_debug_error is not None:
                manifest_entry["selected_peak_debug_error"] = selected_peak_debug_error
            manifest.append(manifest_entry)
            alignment_debug.append(
                _alignment_entry(
                    bundle,
                    status="included",
                    metric_unavailable_reasons=(
                        _metric_unavailable_reasons(metric_profile)
                        if metric_profile is not None
                        else {}
                    ),
                )
            )
            _write_checkpoint(
                manifest_path=manifest_path,
                stats_path=stats_path,
                alignment_path=alignment_path,
                summary_path=summary_path,
                manifest=manifest,
                alignment_debug=alignment_debug,
                stats_accumulator=stats_accumulator,
                h5_paths=h5_paths,
                top_k=top_k,
                segment_level=resolved_segment_level,
                selectors=resolved_selectors,
                output_path=output_path,
                eval_path=eval_path,
                status="running",
            )
            _log(
                verbose,
                (
                    f"[{bundle.bundle_id}] metric_proposals={len(proposals)} "
                    f"selectors={resolved_selectors}"
                ),
            )
        except Exception as exc:
            manifest.append(
                {
                    "h5_path": str(current_h5_path),
                    "bundle_id": current_h5_path.stem,
                    "included": False,
                    "skip_reason": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            alignment_debug.append(
                {
                    "h5_path": str(current_h5_path),
                    "bundle_id": current_h5_path.stem,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_checkpoint(
                manifest_path=manifest_path,
                stats_path=stats_path,
                alignment_path=alignment_path,
                summary_path=summary_path,
                manifest=manifest,
                alignment_debug=alignment_debug,
                stats_accumulator=stats_accumulator,
                h5_paths=h5_paths,
                top_k=top_k,
                segment_level=resolved_segment_level,
                selectors=resolved_selectors,
                output_path=output_path,
                eval_path=eval_path,
                status="running",
            )
            _log(verbose, f"[{current_h5_path.stem}] error: {type(exc).__name__}: {exc}")

    summary = _write_checkpoint(
        manifest_path=manifest_path,
        stats_path=stats_path,
        alignment_path=alignment_path,
        summary_path=summary_path,
        manifest=manifest,
        alignment_debug=alignment_debug,
        stats_accumulator=stats_accumulator,
        h5_paths=h5_paths,
        top_k=top_k,
        segment_level=resolved_segment_level,
        selectors=resolved_selectors,
        output_path=output_path,
        eval_path=eval_path,
        status="complete",
    )
    return summary


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


def _evaluation_row(
    bundle: TrajectoryBundle,
    proposals: list[KeyEventProposal],
    top_k: int,
    requested_segment_level: str,
    selector_name: str = "metric",
) -> dict[str, Any]:
    return _evaluation_row_from_selected_peaks(
        bundle=bundle,
        selected_peaks=[_proposal_to_peak_dict(proposal) for proposal in proposals],
        top_k=top_k,
        requested_segment_level=requested_segment_level,
        selector_name=selector_name,
    )


def _evaluation_row_from_selected_peaks(
    bundle: TrajectoryBundle,
    selected_peaks: list[dict[str, Any]],
    top_k: int,
    requested_segment_level: str,
    selector_name: str,
) -> dict[str, Any]:
    segments = _segments(bundle)
    failed_segments_by_level = {
        "high": [segment for segment in segments if segment["level"] == "high" and not segment["success"]],
        "low": [segment for segment in segments if segment["level"] == "low" and not segment["success"]],
    }
    failed_segments_by_level["combined"] = [
        *failed_segments_by_level["high"],
        *failed_segments_by_level["low"],
    ]
    selected_peak_times = [_as_float_or_none(peak.get("peak_sec")) for peak in selected_peaks]
    selected_peak_times = [peak for peak in selected_peak_times if peak is not None]
    levels = _levels_for_request(requested_segment_level)
    recall_by_level = {}
    for level in levels:
        failed_segments = failed_segments_by_level[level]
        matched_segments = [
            {
                **segment,
                "matched_peak_secs": [
                    peak_sec
                    for peak_sec in selected_peak_times
                    if _segment_contains_peak(segment, peak_sec)
                ],
            }
            for segment in failed_segments
        ]
        matched_segments = [segment for segment in matched_segments if segment["matched_peak_secs"]]
        recall_by_level[level] = {
            "failed_segment_count": len(failed_segments),
            "hit_failed_segment_count": len(matched_segments),
            "failed_segment_recall_at_k": (
                len(matched_segments) / len(failed_segments)
                if failed_segments
                else None
            ),
            "episode_hit_at_k": bool(matched_segments) if failed_segments else None,
            "matched_failed_segments": matched_segments,
        }
    return {
        "bundle_id": bundle.bundle_id,
        "h5_path": str(bundle.root_path) if bundle.root_path is not None else None,
        "selector_name": selector_name,
        "top_k": top_k,
        "selected_peaks": selected_peaks,
        "failed_segments": failed_segments_by_level,
        "recall_by_level": recall_by_level,
    }


def _selector_evaluation_row(
    selector_name: str,
    bundle: TrajectoryBundle,
    proposals: list[KeyEventProposal],
    top_k: int,
    requested_segment_level: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if selector_name == "metric":
        return _evaluation_row(
            bundle=bundle,
            proposals=proposals[:top_k],
            top_k=top_k,
            requested_segment_level=requested_segment_level,
            selector_name=selector_name,
        )

    selected_peaks: list[dict[str, Any]]
    error: str | None = None
    try:
        if selector_name == "uniform":
            frames = select_uniform_keyframes(bundle.camera_frames, top_k)
        elif selector_name == "kite_flow":
            frames = select_kite_flow_keyframes(bundle.camera_frames, top_k, config=config)
        else:
            raise ValueError(f"Unsupported REASSEMBLE selector: {selector_name}")
        selected_peaks = [
            _frame_to_peak_dict(frame, selector_name=selector_name, rank=rank)
            for rank, frame in enumerate(frames[:top_k], start=1)
        ]
    except Exception as exc:
        selected_peaks = []
        error = f"{type(exc).__name__}: {exc}"

    row = _evaluation_row_from_selected_peaks(
        bundle=bundle,
        selected_peaks=selected_peaks,
        top_k=top_k,
        requested_segment_level=requested_segment_level,
        selector_name=selector_name,
    )
    if error is not None:
        row["selector_error"] = error
    return row


def _update_stats(stats_accumulator: dict[str, dict[str, dict[str, float]]], row: dict[str, Any]) -> None:
    selector_name = str(row.get("selector_name", "metric"))
    for level, recall in row["recall_by_level"].items():
        failed_count = recall["failed_segment_count"]
        if failed_count <= 0:
            continue
        stats_accumulator[selector_name][level]["trajectory_count"] += 1
        stats_accumulator[selector_name][level]["episode_hits"] += 1 if recall["episode_hit_at_k"] else 0
        stats_accumulator[selector_name][level]["failed_segment_count"] += failed_count
        stats_accumulator[selector_name][level]["failed_segment_hits"] += recall["hit_failed_segment_count"]


def _finalize_stats(stats_accumulator: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for selector_name in DEFAULT_SELECTORS:
        output[selector_name] = _finalize_selector_stats(stats_accumulator.get(selector_name, {}))
    for selector_name in sorted(set(stats_accumulator) - set(DEFAULT_SELECTORS)):
        output[selector_name] = _finalize_selector_stats(stats_accumulator.get(selector_name, {}))
    return output


def _finalize_selector_stats(level_values: dict[str, dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for level in ("high", "low", "combined"):
        values = level_values.get(level, {})
        trajectory_count = int(values.get("trajectory_count", 0))
        episode_hits = int(values.get("episode_hits", 0))
        failed_segment_count = int(values.get("failed_segment_count", 0))
        failed_segment_hits = int(values.get("failed_segment_hits", 0))
        output[level] = {
            "trajectory_count": trajectory_count,
            "episode_hits": episode_hits,
            "episode_recall_at_k": episode_hits / trajectory_count if trajectory_count else None,
            "failed_segment_count": failed_segment_count,
            "failed_segment_hits": failed_segment_hits,
            "failed_segment_recall_at_k": (
                failed_segment_hits / failed_segment_count
                if failed_segment_count
                else None
            ),
        }
    return output


def _write_checkpoint(
    manifest_path: Path,
    stats_path: Path,
    alignment_path: Path,
    summary_path: Path,
    manifest: list[dict[str, Any]],
    alignment_debug: list[dict[str, Any]],
    stats_accumulator: dict[str, dict[str, dict[str, float]]],
    h5_paths: list[Path],
    top_k: int,
    segment_level: str,
    selectors: list[str],
    output_path: Path,
    eval_path: Path,
    status: str,
) -> dict[str, Any]:
    """Persist current evaluation state so long runs survive interruption."""

    stats = _finalize_stats(stats_accumulator)
    summary = {
        "status": status,
        "input_count": len(h5_paths),
        "processed_count": len(manifest),
        "remaining_count": max(len(h5_paths) - len(manifest), 0),
        "included_count": sum(1 for entry in manifest if entry.get("included")),
        "skipped_count": sum(1 for entry in manifest if not entry.get("included")),
        "top_k": top_k,
        "segment_level": segment_level,
        "selectors": selectors,
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "eval_path": str(eval_path),
        "stats_path": str(stats_path),
        "alignment_debug_path": str(alignment_path),
        "stats": stats,
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(stats_path, stats)
    _write_json_atomic(alignment_path, alignment_debug)
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


def _manifest_entry(bundle: TrajectoryBundle, h5_path: Path) -> dict[str, Any]:
    return {
        "h5_path": str(h5_path),
        "bundle_id": bundle.bundle_id,
        "source_format": bundle.source_format,
        "duration_sec": bundle.duration_sec,
        "stream_names": [stream.name for stream in bundle.streams],
        "camera_frame_count": len(bundle.camera_frames),
        "has_failed_segment": bundle.raw_metadata.get("has_failed_segment"),
        "failed_high_level_count": bundle.raw_metadata.get("failed_high_level_count"),
        "failed_low_level_count": bundle.raw_metadata.get("failed_low_level_count"),
        "high_level_segment_count": bundle.raw_metadata.get("high_level_segment_count"),
        "low_level_segment_count": bundle.raw_metadata.get("low_level_segment_count"),
    }


def _alignment_entry(
    bundle: TrajectoryBundle,
    status: str,
    reason: str | None = None,
    metric_unavailable_reasons: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "h5_path": str(bundle.root_path) if bundle.root_path is not None else None,
        "status": status,
        "reason": reason,
        "timestamp_origin_sec": bundle.raw_metadata.get("timestamp_origin_sec"),
        "duration_sec": bundle.duration_sec,
        "stream_counts": bundle.raw_metadata.get("stream_counts"),
        "decoded_frame_count": bundle.raw_metadata.get("decoded_frame_count"),
        "camera_names_requested": bundle.raw_metadata.get("camera_names_requested"),
        "metric_unavailable_reasons": metric_unavailable_reasons or {},
    }


def _metric_unavailable_reasons(metric_profile: Any) -> dict[str, str]:
    reasons = {}
    for metric in metric_profile.metrics:
        metadata = metric.metadata if isinstance(metric.metadata, dict) else {}
        if metadata.get("available") is False:
            reasons[metric.name] = str(metadata.get("unavailable_reason", "unavailable"))
    return reasons


def _proposal_to_peak_dict(proposal: KeyEventProposal) -> dict[str, Any]:
    return {
        "event_id": proposal.event_id,
        "timeline_rank": proposal.metadata.get("timeline_rank"),
        "peak_sec": _proposal_peak_sec(proposal),
        "start_sec": proposal.time_window.start_sec,
        "end_sec": proposal.time_window.end_sec,
        "proposal_score": proposal.metadata.get("proposal_score"),
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


def _segments(bundle: TrajectoryBundle) -> list[dict[str, Any]]:
    value = bundle.raw_metadata.get("reassemble_segments", [])
    return [segment for segment in value if isinstance(segment, dict)]


def _segment_contains_peak(segment: dict[str, Any], peak_sec: float) -> bool:
    return float(segment["start_sec"]) <= peak_sec <= float(segment["end_sec"])


def _levels_for_request(segment_level: str) -> list[str]:
    if segment_level == "high":
        return ["high"]
    if segment_level == "low":
        return ["low"]
    return ["high", "low", "combined"]


def _write_selected_peak_debug(
    selected_peaks_dir: Path,
    bundle: TrajectoryBundle,
    rows: list[dict[str, Any]],
) -> None:
    bundle_dir = selected_peaks_dir / bundle.bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_id": bundle.bundle_id,
        "selected_peaks_by_selector": {
            row["selector_name"]: row["selected_peaks"]
            for row in rows
        },
    }
    (bundle_dir / "selected_peaks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        selector_dir = bundle_dir / str(row["selector_name"])
        selector_dir.mkdir(exist_ok=True)
        for peak in row["selected_peaks"]:
            raw_image_path = peak.get("image_path")
            if not isinstance(raw_image_path, str) or not raw_image_path:
                continue
            image_path = Path(raw_image_path)
            if image_path.is_file():
                target = selector_dir / f"{int(peak.get('timeline_rank') or 0):03d}_{peak.get('camera_name', 'frame')}_{peak.get('frame_index')}.jpg"
                if not target.exists():
                    shutil.copy2(image_path, target)


def _resolve_h5_paths(h5_path: str | Path | None, data_root: str | Path | None) -> list[Path]:
    if h5_path is not None:
        path = Path(h5_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not find REASSEMBLE HDF5 file: {path}")
        return [path]
    root = Path(data_root).expanduser().resolve() if data_root is not None else Path(".").resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Could not find REASSEMBLE data root: {root}")
    paths = sorted(path for path in root.glob("*.h5") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .h5 files found in REASSEMBLE data root: {root}")
    return paths


def _string_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        output = [str(item) for item in value if isinstance(item, str) and item]
        if output:
            return output
    return default


def _resolve_selectors(selectors: list[str] | None) -> list[str]:
    raw_selectors = selectors or DEFAULT_SELECTORS
    resolved: list[str] = []
    for selector in raw_selectors:
        normalized = str(selector).lower()
        if normalized == "flow":
            normalized = "kite_flow"
        if normalized not in {"uniform", "kite_flow", "metric"}:
            raise ValueError(f"Unsupported REASSEMBLE selector: {selector}")
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved or list(DEFAULT_SELECTORS)


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[reassemble] {message}", flush=True)
