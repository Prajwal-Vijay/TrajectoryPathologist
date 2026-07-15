"""Deterministic REASSEMBLE failure-segment recall evaluation."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.adapters.reassemble_bundle import ReassembleBundleAdapter
from pathologist.config import config_section, deep_merge, load_pipeline_config
from pathologist.events import MetricEventSelector
from pathologist.metrics import MetricEngine
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.trajectory_bundle import TrajectoryBundle


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
    stats_accumulator: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "trajectory_count": 0,
            "episode_hits": 0,
            "failed_segment_count": 0,
            "failed_segment_hits": 0,
        }
    )

    eval_path.write_text("", encoding="utf-8")
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
                _log(verbose, f"[{bundle.bundle_id}] skipped: no failed segments")
                continue

            manifest_entry.update({"included": True, "skip_reason": None})
            metric_profile = MetricEngine(config=config).compute(bundle)
            selector = _selector_from_config(config)
            proposals = selector.select(
                metric_profile,
                trajectory_bundle=bundle,
                debug_timeline_path=output_path / "timeline_debug" / f"{bundle.bundle_id}.json",
            )
            selected_proposals = proposals[:top_k]
            row = _evaluation_row(
                bundle=bundle,
                proposals=selected_proposals,
                top_k=top_k,
                requested_segment_level=resolved_segment_level,
            )
            _update_stats(stats_accumulator, row)
            _write_selected_peak_debug(selected_peaks_dir, bundle, selected_proposals)
            eval_rows.append(row)
            with eval_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            manifest_entry.update(
                {
                    "proposal_count": len(proposals),
                    "selected_peak_count": len(selected_proposals),
                    "metric_count": len(metric_profile.metrics),
                }
            )
            manifest.append(manifest_entry)
            alignment_debug.append(
                _alignment_entry(
                    bundle,
                    status="included",
                    metric_unavailable_reasons=_metric_unavailable_reasons(metric_profile),
                )
            )
            _log(verbose, f"[{bundle.bundle_id}] proposals={len(proposals)} selected={len(selected_proposals)}")
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
            _log(verbose, f"[{current_h5_path.stem}] error: {type(exc).__name__}: {exc}")

    stats = _finalize_stats(stats_accumulator)
    summary = {
        "input_count": len(h5_paths),
        "included_count": sum(1 for entry in manifest if entry.get("included")),
        "skipped_count": sum(1 for entry in manifest if not entry.get("included")),
        "top_k": top_k,
        "segment_level": resolved_segment_level,
        "output_dir": str(output_path),
        "manifest_path": str(manifest_path),
        "eval_path": str(eval_path),
        "stats_path": str(stats_path),
        "alignment_debug_path": str(alignment_path),
        "stats": stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    alignment_path.write_text(json.dumps(alignment_debug, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    selected_peak_times = [_proposal_peak_sec(proposal) for proposal in proposals]
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
        "top_k": top_k,
        "selected_peaks": [_proposal_to_peak_dict(proposal) for proposal in proposals],
        "failed_segments": failed_segments_by_level,
        "recall_by_level": recall_by_level,
    }


def _update_stats(stats_accumulator: dict[str, dict[str, float]], row: dict[str, Any]) -> None:
    for level, recall in row["recall_by_level"].items():
        failed_count = recall["failed_segment_count"]
        if failed_count <= 0:
            continue
        stats_accumulator[level]["trajectory_count"] += 1
        stats_accumulator[level]["episode_hits"] += 1 if recall["episode_hit_at_k"] else 0
        stats_accumulator[level]["failed_segment_count"] += failed_count
        stats_accumulator[level]["failed_segment_hits"] += recall["hit_failed_segment_count"]


def _finalize_stats(stats_accumulator: dict[str, dict[str, float]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for level in ("high", "low", "combined"):
        values = stats_accumulator.get(level, {})
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
    proposals: list[KeyEventProposal],
) -> None:
    bundle_dir = selected_peaks_dir / bundle.bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_id": bundle.bundle_id,
        "selected_peaks": [_proposal_to_peak_dict(proposal) for proposal in proposals],
        "candidate_frames": [
            {
                "event_id": proposal.event_id,
                "frames": [
                    {
                        "camera_name": frame.camera_name,
                        "timestamp_sec": frame.timestamp_sec,
                        "frame_index": frame.frame_index,
                        "image_path": str(frame.image_path),
                    }
                    for frame in proposal.candidate_frames
                ],
            }
            for proposal in proposals
        ],
    }
    (bundle_dir / "selected_peaks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for proposal in proposals:
        for frame in proposal.candidate_frames[:3]:
            if frame.image_path.exists():
                target = bundle_dir / f"{proposal.metadata.get('timeline_rank', 0):03d}_{frame.camera_name}_{frame.frame_index}.jpg"
                if not target.exists():
                    shutil.copy2(frame.image_path, target)


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


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[reassemble] {message}", flush=True)
