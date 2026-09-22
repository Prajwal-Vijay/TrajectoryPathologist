"""Metric-driven key-event proposal selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.schemas.common import EventProposalSource, TimeWindow
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.metrics import MetricEvent, MetricProfile, MetricResult
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

@dataclass(slots=True)
class SurpriseFloodEvent:
    """Peak basin extracted from the aggregate surprise landscape."""

    peak_index: int
    peak_sec: float
    start_sec: float
    end_sec: float
    peak_score: float
    base_score: float
    persistence: float
    contour_score: float
    width_sec: float
    area_above_base: float
    rank_score: float
    rank_score_type: str = "area_above_base"
    death_index: int | None = None
    merge_sec: float | None = None
    is_global_peak: bool = False


class MetricEventSelector:
    """Select key-event proposals from persistent peaks in ECDF surprise."""

    def __init__(
        self,
        frame_stride_within_window: int = 10,
        visual_context_sec: float = 1.0,
        max_frames_per_camera: int = 5,
        metric_packet_context_steps: int = 10,
        surprise_flood_config: dict[str, Any] | None = None,
    ) -> None:
        if frame_stride_within_window <= 0:
            raise ValueError("frame_stride_within_window must be positive.")
        if visual_context_sec < 0:
            raise ValueError("visual_context_sec must be non-negative.")
        if max_frames_per_camera <= 0:
            raise ValueError("max_frames_per_camera must be positive.")
        if metric_packet_context_steps < 0:
            raise ValueError("metric_packet_context_steps must be non-negative.")
        self.frame_stride_within_window = frame_stride_within_window
        self.visual_context_sec = visual_context_sec
        self.max_frames_per_camera = max_frames_per_camera
        self.metric_packet_context_steps = metric_packet_context_steps
        self.surprise_flood_config = surprise_flood_config or {}

    def select(
        self,
        metric_profile: MetricProfile,
        trajectory_bundle: TrajectoryBundle | None = None,
        debug_timeline_path: str | Path | None = None,
        ecod_aggregate_path: str | Path | None = None,
    ) -> list[KeyEventProposal]:
        """Build ranked key-event proposals from the aggregate surprise landscape."""

        proposals = self._select_surprise_flood_proposals(
            metric_profile=metric_profile,
            trajectory_bundle=trajectory_bundle,
            ecod_aggregate_path=Path(ecod_aggregate_path) if ecod_aggregate_path is not None else None,
        )
        if debug_timeline_path is not None:
            self._write_surprise_flood_debug_timeline(
                metric_profile=metric_profile,
                proposals=proposals,
                debug_timeline_path=Path(debug_timeline_path),
            )
        return proposals

    def _select_surprise_flood_proposals(
        self,
        metric_profile: MetricProfile,
        trajectory_bundle: TrajectoryBundle | None,
        ecod_aggregate_path: Path | None = None,
    ) -> list[KeyEventProposal]:
        """Build proposals from persistent peaks in the aggregate ECOD surprise curve."""

        aggregate_metric_name = str(
            self.surprise_flood_config.get("aggregate_metric_name", "ecod_aggregate_score")
        )
        metric_lookup = {metric.name: metric for metric in metric_profile.metrics}
        aggregate_metric = metric_lookup.get(aggregate_metric_name)
        if aggregate_metric is None or not aggregate_metric.samples:
            if ecod_aggregate_path is not None:
                self._write_ecod_aggregate_debug_unavailable(
                    metric_profile=metric_profile,
                    ecod_aggregate_path=ecod_aggregate_path,
                    aggregate_metric_name=aggregate_metric_name,
                    reason="aggregate metric missing or has no samples",
                )
            return []

        timestamps: list[float] = []
        raw_scores: list[float] = []
        for sample in aggregate_metric.samples:
            if not isinstance(sample.value, (int, float, bool)):
                continue
            timestamps.append(float(sample.timestamp_sec))
            raw_scores.append(float(sample.value))
        if len(timestamps) < 3:
            if ecod_aggregate_path is not None:
                self._write_ecod_aggregate_debug_unavailable(
                    metric_profile=metric_profile,
                    ecod_aggregate_path=ecod_aggregate_path,
                    aggregate_metric_name=aggregate_metric_name,
                    reason="requires at least three numeric aggregate samples",
                    timestamps=timestamps,
                    raw_scores=raw_scores,
                )
            return []

        smooth_window = int(self.surprise_flood_config.get("smoothing_window", 1))
        gamma = float(self.surprise_flood_config.get("gamma", 0.5))
        max_proposals = int(self.surprise_flood_config.get("max_proposals", 0))
        ranking = _resolve_surprise_flood_ranking(
            self.surprise_flood_config.get("ranking", "area")
        )
        gamma = min(1.0, max(0.0, gamma))
        # smoothed_scores = _centered_moving_average(raw_scores, max(1, smooth_window))
        # smoothed_scores = _centered_gaussian_filter(raw_scores, sigma=2.5)
        smoothed_scores = _centered_savgol_filter(raw_scores, window_length=smooth_window)

        flood_events = _extract_surprise_flood_events(
            timestamps=timestamps,
            scores=smoothed_scores,
            gamma=gamma,
        )
        flood_events = _rank_surprise_flood_events(flood_events, ranking=ranking)
        all_flood_events = flood_events
        selected_flood_events = (
            all_flood_events[:max_proposals]
            if max_proposals > 0
            else all_flood_events
        )
        if ecod_aggregate_path is not None:
            self._write_ecod_aggregate_debug(
                metric_profile=metric_profile,
                ecod_aggregate_path=ecod_aggregate_path,
                aggregate_metric_name=aggregate_metric_name,
                timestamps=timestamps,
                raw_scores=raw_scores,
                smoothed_scores=smoothed_scores,
                all_flood_events=all_flood_events,
                selected_flood_events=selected_flood_events,
                smooth_window=smooth_window,
                gamma=gamma,
                ranking=ranking,
                max_proposals=max_proposals,
            )
        flood_events = selected_flood_events

        proposals: list[KeyEventProposal] = []
        for rank, flood_event in enumerate(flood_events, start=1):
            proposal_time_window = TimeWindow(flood_event.start_sec, flood_event.end_sec)
            metric_event = MetricEvent(
                metric_name=aggregate_metric_name,
                time_window=proposal_time_window,
                value={
                    "peak_score": flood_event.peak_score,
                    "base_score": flood_event.base_score,
                    "persistence": flood_event.persistence,
                    "area_above_base": flood_event.area_above_base,
                    "width_sec": flood_event.width_sec,
                    "peak_sec": flood_event.peak_sec,
                },
                severity="surprise",
                reason=(
                    "Peak in aggregate ECOD surprise landscape "
                    f"at {flood_event.peak_sec:.3f}s"
                ),
            )
            metric_packet = [
                _local_metric_context(
                    aggregate_metric,
                    proposal_time_window,
                    context_steps=self.metric_packet_context_steps,
                )
            ]
            candidate_frames = self._select_candidate_frames(
                trajectory_bundle=trajectory_bundle,
                proposal_time_window=proposal_time_window,
            )
            proposals.append(
                KeyEventProposal(
                    event_id=f"{metric_profile.bundle_id}_surprise_flood_{rank:03d}",
                    time_window=proposal_time_window,
                    source=EventProposalSource.METRIC_TRIGGER,
                    trigger_reason=(
                        f"aggregate surprise flood peak at {flood_event.peak_sec:.3f}s "
                        f"with height {flood_event.peak_score:.3f}, "
                        f"area {flood_event.area_above_base:.3f}, and "
                        f"persistence {flood_event.persistence:.3f}"
                    ),
                    candidate_frames=candidate_frames,
                    metric_events=[metric_event],
                    metric_packet=metric_packet,
                    metadata={
                        "timeline_rank": rank,
                        "selector_mode": "surprise_flood",
                        "proposal_score": flood_event.rank_score,
                        "rank_score_type": flood_event.rank_score_type,
                        "aggregate_metric_name": aggregate_metric_name,
                        "peak_index": flood_event.peak_index,
                        "peak_sec": flood_event.peak_sec,
                        "peak_score": flood_event.peak_score,
                        "base_score": flood_event.base_score,
                        "persistence": flood_event.persistence,
                        "contour_score": flood_event.contour_score,
                        "width_sec": flood_event.width_sec,
                        "area_above_base": flood_event.area_above_base,
                        "death_index": flood_event.death_index,
                        "merge_sec": flood_event.merge_sec,
                        "is_global_peak": flood_event.is_global_peak,
                        "smoothing_window": max(1, smooth_window),
                        "gamma": gamma,
                        "candidate_frame_count": len(candidate_frames),
                        "visual_context_sec": self.visual_context_sec,
                    },
                )
            )
        return proposals

    def _select_candidate_frames(
        self,
        trajectory_bundle: TrajectoryBundle | None,
        proposal_time_window: TimeWindow,
    ) -> list[CameraFrameRef]:
        """Attach sampled camera frames around the proposal window."""

        if trajectory_bundle is None or not trajectory_bundle.camera_frames:
            return []

        proposal_end_sec = proposal_time_window.end_sec or proposal_time_window.start_sec
        visual_start_sec = max(0.0, proposal_time_window.start_sec - self.visual_context_sec)
        visual_end_sec = proposal_end_sec + self.visual_context_sec
        frames_in_context = [
            frame
            for frame in trajectory_bundle.camera_frames
            if visual_start_sec <= frame.timestamp_sec <= visual_end_sec
        ]
        if not frames_in_context:
            return []

        sampled_frames: list[CameraFrameRef] = []
        camera_names = sorted({frame.camera_name for frame in frames_in_context})
        for camera_name in camera_names:
            camera_frames = [
                frame
                for frame in frames_in_context
                if frame.camera_name == camera_name
            ]
            camera_frames.sort(key=lambda frame: (frame.timestamp_sec, frame.frame_index or -1))
            strided_frames = camera_frames[:: self.frame_stride_within_window]
            sampled_frames.extend(_evenly_sample_frames(strided_frames, self.max_frames_per_camera))
        sampled_frames.sort(key=_camera_frame_sort_key)
        return sampled_frames

    def _write_surprise_flood_debug_timeline(
        self,
        metric_profile: MetricProfile,
        proposals: list[KeyEventProposal],
        debug_timeline_path: Path,
    ) -> None:
        """Write the surprise-flood proposal timeline to JSON."""

        debug_timeline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_id": metric_profile.bundle_id,
            "selector_mode": "surprise_flood",
            "surprise_flood_config": self.surprise_flood_config,
            "visual_context_sec": self.visual_context_sec,
            "max_frames_per_camera": self.max_frames_per_camera,
            "metric_packet_context_steps": self.metric_packet_context_steps,
            "surprise_flood_events": [
                {
                    "rank": rank,
                    "event_id": proposal.event_id,
                    "time_window": _time_window_to_dict(proposal.time_window),
                    "proposal_score": proposal.metadata.get("proposal_score"),
                    "rank_score_type": proposal.metadata.get("rank_score_type"),
                    "peak_sec": proposal.metadata.get("peak_sec"),
                    "peak_score": proposal.metadata.get("peak_score"),
                    "base_score": proposal.metadata.get("base_score"),
                    "persistence": proposal.metadata.get("persistence"),
                    "contour_score": proposal.metadata.get("contour_score"),
                    "width_sec": proposal.metadata.get("width_sec"),
                    "area_above_base": proposal.metadata.get("area_above_base"),
                    "merge_sec": proposal.metadata.get("merge_sec"),
                    "is_global_peak": proposal.metadata.get("is_global_peak"),
                    "candidate_frame_count": len(proposal.candidate_frames),
                }
                for rank, proposal in enumerate(proposals, start=1)
            ],
        }
        debug_timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_ecod_aggregate_debug_unavailable(
        self,
        metric_profile: MetricProfile,
        ecod_aggregate_path: Path,
        aggregate_metric_name: str,
        reason: str,
        timestamps: list[float] | None = None,
        raw_scores: list[float] | None = None,
    ) -> None:
        paths = ecod_aggregate_output_paths(ecod_aggregate_path)
        paths["data"].parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_id": metric_profile.bundle_id,
            "aggregate_metric_name": aggregate_metric_name,
            "available": False,
            "reason": reason,
            "timestamps_sec": timestamps or [],
            "raw_scores": raw_scores or [],
            "smoothed_scores": [],
            "source_metric_series": {},
            "local_maxima": [],
            "selected_local_maxima": [],
            "plot_paths": {
                "raw": str(paths["raw_plot"]),
                "smoothed": str(paths["smoothed_plot"]),
            },
        }
        paths["data"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_ecod_aggregate_debug(
        self,
        metric_profile: MetricProfile,
        ecod_aggregate_path: Path,
        aggregate_metric_name: str,
        timestamps: list[float],
        raw_scores: list[float],
        smoothed_scores: list[float],
        all_flood_events: list[SurpriseFloodEvent],
        selected_flood_events: list[SurpriseFloodEvent],
        smooth_window: int,
        gamma: float,
        ranking: str,
        max_proposals: int,
    ) -> None:
        paths = ecod_aggregate_output_paths(ecod_aggregate_path)
        paths["data"].parent.mkdir(parents=True, exist_ok=True)
        selected_indices = {event.peak_index for event in selected_flood_events}
        local_maxima = [
            {
                **_surprise_flood_event_to_dict(event),
                "selected": event.peak_index in selected_indices,
            }
            for event in all_flood_events
        ]
        payload = {
            "bundle_id": metric_profile.bundle_id,
            "aggregate_metric_name": aggregate_metric_name,
            "available": True,
            "smoothing": {
                "method": "centered_savgol_filter",
                "window_length": max(1, int(smooth_window)),
                "gamma": gamma,
            },
            "max_proposals": max_proposals,
            "ranking": ranking,
            "timestamps_sec": timestamps,
            "raw_scores": raw_scores,
            "smoothed_scores": smoothed_scores,
            "source_metric_series": _source_metric_series(
                metric_profile=metric_profile,
                aggregate_metric_name=aggregate_metric_name,
            ),
            "local_maxima": local_maxima,
            "selected_local_maxima": [
                _surprise_flood_event_to_dict(event)
                for event in selected_flood_events
            ],
            "local_maxima_count": len(all_flood_events),
            "selected_local_maxima_count": len(selected_flood_events),
            "plot_paths": {
                "raw": str(paths["raw_plot"]),
                "smoothed": str(paths["smoothed_plot"]),
            },
        }
        plot_error = _write_ecod_aggregate_plots(
            raw_plot_path=paths["raw_plot"],
            smoothed_plot_path=paths["smoothed_plot"],
            bundle_id=metric_profile.bundle_id,
            timestamps=timestamps,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            local_maxima=all_flood_events,
            selected_local_maxima=selected_flood_events,
        )
        if plot_error is not None:
            payload["plot_error"] = plot_error
        paths["data"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_metric_series(
    *,
    metric_profile: MetricProfile,
    aggregate_metric_name: str,
) -> dict[str, dict[str, Any]]:
    metric_lookup = {metric.name: metric for metric in metric_profile.metrics}
    aggregate_metric = metric_lookup.get(aggregate_metric_name)
    if aggregate_metric is None:
        return {}
    source_metric_names = aggregate_metric.metadata.get("source_metric_names")
    if not isinstance(source_metric_names, list):
        return {}

    series: dict[str, dict[str, Any]] = {}
    for metric_name in source_metric_names:
        if not isinstance(metric_name, str):
            continue
        metric = metric_lookup.get(metric_name)
        if metric is None:
            continue
        timestamps: list[float] = []
        values: list[float] = []
        for sample in metric.samples:
            if not isinstance(sample.value, (int, float, bool)):
                continue
            timestamps.append(float(sample.timestamp_sec))
            values.append(float(sample.value))
        if not timestamps:
            continue
        series[metric_name] = {
            "timestamps_sec": timestamps,
            "raw_values": values,
            "unit": metric.unit,
            "description": metric.description,
        }
    return series


def ecod_aggregate_output_paths(path: Path) -> dict[str, Path]:
    if path.suffix:
        base = path.with_suffix("")
        data_path = path if path.suffix.lower() == ".json" else base.with_suffix(".json")
    else:
        base = path
        data_path = base.with_suffix(".json")
    return {
        "data": data_path,
        "raw_plot": base.with_name(f"{base.name}_raw.png"),
        "smoothed_plot": base.with_name(f"{base.name}_smoothed.png"),
    }


def _surprise_flood_event_to_dict(event: SurpriseFloodEvent) -> dict[str, Any]:
    return {
        "peak_index": event.peak_index,
        "peak_sec": event.peak_sec,
        "start_sec": event.start_sec,
        "end_sec": event.end_sec,
        "peak_score": event.peak_score,
        "base_score": event.base_score,
        "persistence": event.persistence,
        "contour_score": event.contour_score,
        "width_sec": event.width_sec,
        "area_above_base": event.area_above_base,
        "rank_score": event.rank_score,
        "rank_score_type": event.rank_score_type,
        "death_index": event.death_index,
        "merge_sec": event.merge_sec,
        "is_global_peak": event.is_global_peak,
    }


def _write_ecod_aggregate_plots(
    raw_plot_path: Path,
    smoothed_plot_path: Path,
    bundle_id: str,
    timestamps: list[float],
    raw_scores: list[float],
    smoothed_scores: list[float],
    local_maxima: list[SurpriseFloodEvent],
    selected_local_maxima: list[SurpriseFloodEvent],
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathologist_matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pathologist_cache")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    raw_plot_path.parent.mkdir(parents=True, exist_ok=True)
    smoothed_plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _plot_ecod_raw_scores(
            plt=plt,
            plot_path=raw_plot_path,
            bundle_id=bundle_id,
            timestamps=timestamps,
            raw_scores=raw_scores,
        )
        _plot_ecod_smoothed_scores(
            plt=plt,
            plot_path=smoothed_plot_path,
            bundle_id=bundle_id,
            timestamps=timestamps,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            local_maxima=local_maxima,
            selected_local_maxima=selected_local_maxima,
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _plot_ecod_raw_scores(
    plt: Any,
    plot_path: Path,
    bundle_id: str,
    timestamps: list[float],
    raw_scores: list[float],
) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(timestamps, raw_scores, color="#111111", linewidth=1.2, label="raw_ecod_aggregate")
    axis.scatter(timestamps, raw_scores, color="#111111", s=9, alpha=0.6)
    axis.set_xlabel("timestamp_sec")
    axis.set_ylabel("Aggregate ECOD score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.suptitle(f"{bundle_id}: raw ECOD aggregate score")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _plot_ecod_smoothed_scores(
    plt: Any,
    plot_path: Path,
    bundle_id: str,
    timestamps: list[float],
    raw_scores: list[float],
    smoothed_scores: list[float],
    local_maxima: list[SurpriseFloodEvent],
    selected_local_maxima: list[SurpriseFloodEvent],
) -> None:
    selected_indices = {event.peak_index for event in selected_local_maxima}
    unselected_maxima = [event for event in local_maxima if event.peak_index not in selected_indices]

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(timestamps, raw_scores, color="#9a9a9a", linewidth=0.8, alpha=0.45, label="raw")
    axis.plot(timestamps, smoothed_scores, color="#111111", linewidth=1.4, label="smoothed")
    if unselected_maxima:
        axis.scatter(
            [event.peak_sec for event in unselected_maxima],
            [event.peak_score for event in unselected_maxima],
            color="#1f77b4",
            s=28,
            alpha=0.8,
            label="local maxima",
            zorder=4,
        )
    if selected_local_maxima:
        axis.scatter(
            [event.peak_sec for event in selected_local_maxima],
            [event.peak_score for event in selected_local_maxima],
            color="#d62728",
            marker="*",
            s=90,
            alpha=0.95,
            label="selected top-K",
            zorder=5,
        )
        for rank, event in enumerate(selected_local_maxima, start=1):
            axis.annotate(
                str(rank),
                xy=(event.peak_sec, event.peak_score),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=8,
                color="#d62728",
            )
    axis.set_xlabel("timestamp_sec")
    axis.set_ylabel("Aggregate ECOD score")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.suptitle(f"{bundle_id}: smoothed ECOD aggregate with local maxima")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def _local_metric_context(
    metric: MetricResult,
    proposal_time_window: TimeWindow,
    context_steps: int,
) -> MetricResult:
    """Return a metric packet trimmed to the proposal neighborhood."""

    timestep_sec = _metric_sample_timestep(metric)
    context_radius_sec = timestep_sec * context_steps
    context_start_sec = max(0.0, proposal_time_window.start_sec - context_radius_sec)
    proposal_end_sec = proposal_time_window.end_sec or proposal_time_window.start_sec
    context_end_sec = proposal_end_sec + context_radius_sec

    local_samples = [
        sample
        for sample in metric.samples
        if _sample_overlaps_window(sample, context_start_sec, context_end_sec)
    ]
    local_events = [
        event
        for event in metric.events
        if _time_windows_overlap(event.time_window, context_start_sec, context_end_sec)
    ]

    metadata = _local_metric_metadata(metric.metadata, context_start_sec, context_end_sec)
    metadata.update(
        {
            "local_context_window": {
                "start_sec": context_start_sec,
                "end_sec": context_end_sec,
            },
            "local_context_steps": context_steps,
            "local_context_timestep_sec": timestep_sec,
            "original_sample_count": len(metric.samples),
            "local_sample_count": len(local_samples),
            "original_event_count": len(metric.events),
            "local_event_count": len(local_events),
        }
    )

    return MetricResult(
        name=metric.name,
        description=metric.description,
        source_modalities=metric.source_modalities,
        unit=metric.unit,
        samples=local_samples,
        summary=metric.summary,
        thresholds=metric.thresholds,
        events=local_events,
        display_hint=metric.display_hint,
        metadata=metadata,
    )


def _evenly_sample_frames(
    frames: list[CameraFrameRef],
    max_frames: int,
) -> list[CameraFrameRef]:
    """Return up to max_frames spread across the candidate frame sequence."""

    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]

    last_index = len(frames) - 1
    sampled_indices = {
        round(index * last_index / (max_frames - 1))
        for index in range(max_frames)
    }
    return [frames[index] for index in sorted(sampled_indices)]


def _camera_frame_sort_key(frame: CameraFrameRef) -> tuple[int, str, float, int]:
    """Prefer object-level camera sequences before wrist-camera closeups."""

    camera_name = frame.camera_name.lower()
    if "head" in camera_name or "front" in camera_name or "overhead" in camera_name:
        camera_priority = 0
    elif "wrist" in camera_name:
        camera_priority = 1
    else:
        camera_priority = 2
    return (camera_priority, frame.camera_name, frame.timestamp_sec, frame.frame_index or -1)


def _local_metric_metadata(
    metadata: dict[str, Any],
    context_start_sec: float,
    context_end_sec: float,
) -> dict[str, Any]:
    """Copy metric metadata, trimming timestamped/windowed debug lists locally."""

    local_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        trimmed_value = _trim_windowed_metadata_list(value, context_start_sec, context_end_sec)
        if trimmed_value is None:
            local_metadata[key] = value
            continue

        local_metadata[key] = trimmed_value
        local_metadata[f"original_{key}_count"] = len(value)
        local_metadata[f"local_{key}_count"] = len(trimmed_value)
    return local_metadata


def _trim_windowed_metadata_list(
    value: Any,
    context_start_sec: float,
    context_end_sec: float,
) -> list[dict[str, Any]] | None:
    """Trim metadata lists whose items have start/end or timestamp fields."""

    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, dict) for item in value):
        return None

    local_items: list[dict[str, Any]] = []
    for item in value:
        item_start_sec = _metadata_item_start_sec(item)
        if item_start_sec is None:
            return None
        item_end_sec = _metadata_item_end_sec(item, item_start_sec)
        if item_start_sec <= context_end_sec and item_end_sec >= context_start_sec:
            local_items.append(item)
    return local_items


def _metadata_item_start_sec(item: dict[str, Any]) -> float | None:
    value = item.get("start_sec", item.get("timestamp_sec"))
    if value is None:
        return None
    return float(value)


def _metadata_item_end_sec(item: dict[str, Any], fallback_start_sec: float) -> float:
    value = item.get("end_sec")
    if value is None:
        return fallback_start_sec
    return float(value)


def _metric_sample_timestep(metric: MetricResult) -> float:
    """Estimate a metric's sample timestep from its timestamped samples."""

    timestamps = sorted({sample.timestamp_sec for sample in metric.samples})
    dts = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if current > previous
    ]
    if not dts:
        return 0.0
    midpoint = len(dts) // 2
    sorted_dts = sorted(dts)
    if len(sorted_dts) % 2 == 1:
        return sorted_dts[midpoint]
    return (sorted_dts[midpoint - 1] + sorted_dts[midpoint]) / 2.0


def _sample_overlaps_window(sample, start_sec: float, end_sec: float) -> bool:
    if sample.window is not None:
        return _time_windows_overlap(sample.window, start_sec, end_sec)
    return start_sec <= sample.timestamp_sec <= end_sec


def _time_windows_overlap(time_window: TimeWindow, start_sec: float, end_sec: float) -> bool:
    window_end_sec = time_window.end_sec or time_window.start_sec
    return time_window.start_sec <= end_sec and window_end_sec >= start_sec


def _time_window_to_dict(time_window: TimeWindow) -> dict[str, float | None]:
    """Serialize a TimeWindow for debug output."""

    return {
        "start_sec": time_window.start_sec,
        "end_sec": time_window.end_sec,
    }


def _extract_surprise_flood_events(
    timestamps: list[float],
    scores: list[float],
    gamma: float,
) -> list[SurpriseFloodEvent]:
    """Extract persistent peak basins from a 1-D surprise landscape."""

    if len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must have the same length.")
    if len(scores) < 3:
        return []

    topology_peak_indices = _local_maxima_indices(
        scores,
        include_boundary=True,
    )
    peak_indices = _local_maxima_indices(scores)
    if not peak_indices:
        return []

    topology_peak_set = set(topology_peak_indices)
    parent = list(range(len(scores)))
    component_peak: dict[int, int] = {}
    active = [False] * len(scores)
    dead_events: dict[int, dict[str, Any]] = {}

    for index in sorted(range(len(scores)), key=lambda item: (-scores[item], timestamps[item])):
        active[index] = True
        parent[index] = index
        if index in topology_peak_set:
            component_peak[index] = index

        for neighbor in (index - 1, index + 1):
            if neighbor < 0 or neighbor >= len(scores) or not active[neighbor]:
                continue
            root_index = _find_union_parent(parent, index)
            root_neighbor = _find_union_parent(parent, neighbor)
            if root_index == root_neighbor:
                continue
            survivor_root, loser_peak = _merge_surprise_components(
                parent=parent,
                component_peak=component_peak,
                root_a=root_index,
                root_b=root_neighbor,
                scores=scores,
                timestamps=timestamps,
            )
            if loser_peak is not None and loser_peak not in dead_events:
                dead_events[loser_peak] = {
                    "death_index": index,
                    "death_level": scores[index],
                    "merge_sec": timestamps[index],
                }
            parent[_find_union_parent(parent, index)] = survivor_root

    global_peak_index = max(
        topology_peak_indices,
        key=lambda item: (scores[item], -timestamps[item]),
    )
    global_base = min(scores)
    events: list[SurpriseFloodEvent] = []
    for peak_index in peak_indices:
        death = dead_events.get(peak_index)
        is_global_peak = peak_index == global_peak_index and death is None
        base_score = float(death["death_level"]) if death is not None else float(global_base)
        peak_score = float(scores[peak_index])
        persistence = max(0.0, peak_score - base_score)
        contour_score = base_score + gamma * persistence
        start_sec, end_sec = _contour_window(
            timestamps=timestamps,
            scores=scores,
            peak_index=peak_index,
            contour_score=contour_score,
        )
        area = _area_above_base(
            timestamps=timestamps,
            scores=scores,
            start_sec=start_sec,
            end_sec=end_sec,
            base_score=base_score,
        )
        width_sec = max(0.0, end_sec - start_sec)
        events.append(
            SurpriseFloodEvent(
                peak_index=peak_index,
                peak_sec=timestamps[peak_index],
                start_sec=start_sec,
                end_sec=end_sec,
                peak_score=peak_score,
                base_score=base_score,
                persistence=persistence,
                contour_score=contour_score,
                width_sec=width_sec,
                area_above_base=area,
                rank_score=area,
                death_index=int(death["death_index"]) if death is not None else None,
                merge_sec=float(death["merge_sec"]) if death is not None else None,
                is_global_peak=is_global_peak,
            )
        )

    return sorted(
        events,
        key=lambda event: (
            -event.rank_score,
            -event.persistence,
            -event.peak_score,
            event.peak_sec,
        ),
    )


def _resolve_surprise_flood_ranking(value: Any) -> str:
    aliases = {
        "area": "area",
        "area_above_base": "area",
        "persistence": "persistence",
    }
    ranking = aliases.get(str(value).strip().lower())
    if ranking is None:
        raise ValueError(
            "selector.surprise_flood.ranking must be 'area' or 'persistence'."
        )
    return ranking


def _rank_surprise_flood_events(
    events: list[SurpriseFloodEvent],
    *,
    ranking: str,
) -> list[SurpriseFloodEvent]:
    resolved_ranking = _resolve_surprise_flood_ranking(ranking)
    for event in events:
        if resolved_ranking == "persistence":
            event.rank_score = event.persistence
            event.rank_score_type = "persistence"
        else:
            event.rank_score = event.area_above_base
            event.rank_score_type = "area_above_base"
    if resolved_ranking == "persistence":
        return sorted(
            events,
            key=lambda event: (
                -event.persistence,
                -event.area_above_base,
                -event.peak_score,
                event.peak_sec,
            ),
        )
    return sorted(
        events,
        key=lambda event: (
            -event.area_above_base,
            -event.persistence,
            -event.peak_score,
            event.peak_sec,
        ),
    )


def _local_maxima_indices(
    scores: list[float],
    *,
    include_boundary: bool = False,
) -> list[int]:
    """Return representative local maxima, optionally including boundary plateaus."""

    peaks: list[int] = []
    index = 0
    last = len(scores) - 1
    while index <= last:
        plateau_start = index
        while index < last and scores[index + 1] == scores[plateau_start]:
            index += 1
        plateau_end = index
        left_value = scores[plateau_start - 1] if plateau_start > 0 else None
        right_value = scores[plateau_end + 1] if plateau_end < last else None
        current_value = scores[plateau_start]
        left_ok = left_value is None or current_value > left_value
        right_ok = right_value is None or current_value > right_value
        touches_boundary = plateau_start == 0 or plateau_end == last
        is_constant_series = plateau_start == 0 and plateau_end == last
        if (
            left_ok
            and right_ok
            and (include_boundary or not touches_boundary or is_constant_series)
        ):
            peaks.append((plateau_start + plateau_end) // 2)
        index += 1
    return peaks


def _find_union_parent(parent: list[int], index: int) -> int:
    """Find the active component representative with path compression."""

    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _merge_surprise_components(
    parent: list[int],
    component_peak: dict[int, int],
    root_a: int,
    root_b: int,
    scores: list[float],
    timestamps: list[float],
) -> tuple[int, int | None]:
    """Merge two active components and return the surviving root and dying peak."""

    peak_a = component_peak.get(root_a)
    peak_b = component_peak.get(root_b)
    if peak_a is None and peak_b is None:
        parent[root_b] = root_a
        return root_a, None
    if peak_a is None:
        parent[root_a] = root_b
        return root_b, None
    if peak_b is None:
        parent[root_b] = root_a
        return root_a, None

    key_a = (scores[peak_a], -timestamps[peak_a])
    key_b = (scores[peak_b], -timestamps[peak_b])
    if key_a >= key_b:
        survivor_root, loser_root = root_a, root_b
        survivor_peak, loser_peak = peak_a, peak_b
    else:
        survivor_root, loser_root = root_b, root_a
        survivor_peak, loser_peak = peak_b, peak_a
    parent[loser_root] = survivor_root
    component_peak[survivor_root] = survivor_peak
    component_peak.pop(loser_root, None)
    return survivor_root, loser_peak


def _contour_window(
    timestamps: list[float],
    scores: list[float],
    peak_index: int,
    contour_score: float,
) -> tuple[float, float]:
    """Find left/right contour crossings around a peak."""

    start_sec = timestamps[0]
    for index in range(peak_index, 0, -1):
        left_score = scores[index - 1]
        right_score = scores[index]
        if left_score <= contour_score <= right_score:
            start_sec = _interpolate_crossing(
                timestamps[index - 1],
                timestamps[index],
                left_score,
                right_score,
                contour_score,
            )
            break

    end_sec = timestamps[-1]
    for index in range(peak_index, len(scores) - 1):
        left_score = scores[index]
        right_score = scores[index + 1]
        if left_score >= contour_score >= right_score:
            end_sec = _interpolate_crossing(
                timestamps[index],
                timestamps[index + 1],
                left_score,
                right_score,
                contour_score,
            )
            break
    return start_sec, end_sec


def _interpolate_crossing(
    left_time: float,
    right_time: float,
    left_score: float,
    right_score: float,
    target_score: float,
) -> float:
    """Linearly interpolate the timestamp where a segment crosses a score level."""

    score_delta = right_score - left_score
    if score_delta == 0:
        return (left_time + right_time) / 2.0
    fraction = (target_score - left_score) / score_delta
    fraction = min(1.0, max(0.0, fraction))
    return left_time + fraction * (right_time - left_time)


def _area_above_base(
    timestamps: list[float],
    scores: list[float],
    start_sec: float,
    end_sec: float,
    base_score: float,
) -> float:
    """Integrate positive surprise above local base within an event window."""

    if end_sec <= start_sec:
        return 0.0
    points = [(start_sec, _score_at_time(timestamps, scores, start_sec))]
    points.extend(
        (timestamp, score)
        for timestamp, score in zip(timestamps, scores)
        if start_sec < timestamp < end_sec
    )
    points.append((end_sec, _score_at_time(timestamps, scores, end_sec)))
    area = 0.0
    for (left_time, left_score), (right_time, right_score) in zip(points, points[1:]):
        left_height = max(0.0, left_score - base_score)
        right_height = max(0.0, right_score - base_score)
        area += (right_time - left_time) * (left_height + right_height) / 2.0
    return area


def _score_at_time(timestamps: list[float], scores: list[float], timestamp_sec: float) -> float:
    """Linearly interpolate the score at a timestamp."""

    if timestamp_sec <= timestamps[0]:
        return scores[0]
    if timestamp_sec >= timestamps[-1]:
        return scores[-1]
    for index in range(len(timestamps) - 1):
        left_time = timestamps[index]
        right_time = timestamps[index + 1]
        if left_time <= timestamp_sec <= right_time:
            return _interpolate_value(
                left_time,
                right_time,
                scores[index],
                scores[index + 1],
                timestamp_sec,
            )
    return scores[-1]


def _interpolate_value(
    left_time: float,
    right_time: float,
    left_score: float,
    right_score: float,
    target_time: float,
) -> float:
    """Linearly interpolate a score at a target timestamp."""

    time_delta = right_time - left_time
    if time_delta == 0:
        return (left_score + right_score) / 2.0
    fraction = (target_time - left_time) / time_delta
    fraction = min(1.0, max(0.0, fraction))
    return left_score + fraction * (right_score - left_score)


def _centered_moving_average(values: list[float], window: int) -> list[float]:
    """Smooth values using a centered moving average."""

    if window <= 1 or not values:
        return list(values)
    radius_left = (window - 1) // 2
    radius_right = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius_left)
        end = min(len(values), index + radius_right + 1)
        window_values = values[start:end]
        smoothed.append(sum(window_values) / len(window_values))
    return smoothed

def _centered_gaussian_filter(values: list[float], sigma: float) -> list[float]:
    """
    Smooth values using a centered, zero-phase Gaussian filter.
    
    Parameters:
    - values: The 1D list of ECOD quality scores.
    - sigma: The standard deviation of the Gaussian kernel. 
             (Rule of thumb: window_size ≈ 6 * sigma)
    """
    if not values or sigma <= 0:
        return list(values)
    
    # gaussian_filter1d is inherently zero-phase (centered).
    # mode='nearest' or 'reflect' fixes the boundary volatility issue 
    # by mirroring the data at the edges rather than shrinking the window.
    smoothed = gaussian_filter1d(values, sigma=sigma, mode='nearest')
    
    return smoothed.tolist()

def _centered_savgol_filter(values: list[float], window_length: int, polyorder: int = 3) -> list[float]:
    """
    Smooth values using a centered, zero-phase Savitzky-Golay filter.
    
    Parameters:
    - values: The 1D list of ECOD quality scores.
    - window_length: The size of the smoothing window (MUST be an odd integer).
    - polyorder: The degree of the polynomial used to fit the data. 
                 Typically 2 (quadratic) or 3 (cubic).
    """
    if not values or window_length <= 1:
        return list(values)
    
    # Mathematical Constraint 1: Window length MUST be odd to be perfectly centered.
    # If an even number is passed, we force it to the next odd integer to maintain zero phase.
    if window_length % 2 == 0:
        window_length += 1
        
    # Boundary Safety: Window cannot be larger than the dataset.
    if window_length > len(values):
        window_length = len(values) if len(values) % 2 != 0 else len(values) - 1
        
    # Mathematical Constraint 2: Polynomial order must be strictly less than the window length.
    if polyorder >= window_length:
        polyorder = window_length - 1

    # mode='nearest' prevents the shrinking-window variance explosion at t=0 and t=T.
    smoothed = savgol_filter(values, window_length, polyorder, mode='nearest')
    
    return smoothed.tolist()
