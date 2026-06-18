"""Metric-driven key-event proposal selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.schemas.common import EventProposalSource, TimeWindow
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.metrics import MetricEvent, MetricProfile, MetricResult
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle

# NOTE:
# I Have not yet implemented the semantic triggers and regular checkpoints, this is only metric calculated events.

@dataclass(slots=True)
class NormalizedMetricEvent:
    """Metric event after point timestamps are expanded into windows."""

    event_id: str
    metric_name: str
    metric_event: MetricEvent
    original_time_window: TimeWindow
    normalized_time_window: TimeWindow


@dataclass(slots=True)
class TimelineSegment:
    """Intersection interval with a stable set of active metric events."""

    start_sec: float
    end_sec: float
    metric_names: list[str]
    member_event_ids: list[str]

    @property
    def overlap_count(self) -> int:
        """Number of distinct metrics supporting the segment."""

        return len(self.metric_names)


@dataclass(slots=True)
class MetricSupportInterval:
    """Unioned active interval for one metric."""

    metric_name: str
    start_sec: float
    end_sec: float
    member_event_ids: list[str]


class MetricEventSelector:
    """Fuse metric-event windows into ranked key-event proposals."""

    def __init__(
        self,
        point_event_delta_sec: float,
        ignored_metric_names: set[str] | None = None,
        frame_stride_within_window: int = 10,
        metric_packet_context_steps: int = 10,
    ) -> None:
        if point_event_delta_sec < 0:
            raise ValueError("point_event_delta_sec must be non-negative.")
        if frame_stride_within_window <= 0:
            raise ValueError("frame_stride_within_window must be positive.")
        if metric_packet_context_steps < 0:
            raise ValueError("metric_packet_context_steps must be non-negative.")
        self.point_event_delta_sec = point_event_delta_sec
        self.ignored_metric_names = ignored_metric_names or {"timestamp_regularity"}
        self.frame_stride_within_window = frame_stride_within_window
        self.metric_packet_context_steps = metric_packet_context_steps

    def select(
        self,
        metric_profile: MetricProfile,
        trajectory_bundle: TrajectoryBundle | None = None,
        debug_timeline_path: str | Path | None = None,
    ) -> list[KeyEventProposal]:
        """Build ranked key-event proposals from metric overlaps."""

        metric_lookup = {
            metric.name: metric
            for metric in metric_profile.metrics
            if metric.name not in self.ignored_metric_names
        }
        normalized_events = self._normalize_metric_events(metric_lookup) # Expands the point metrics into windows
        metric_support_intervals = _merge_events_within_each_metric(normalized_events)
        timeline_segments = _build_timeline_segments(metric_support_intervals)
        ranked_segments = sorted(
            timeline_segments,
            key=lambda segment: (-segment.overlap_count, segment.start_sec),
        )

        event_lookup = {event.event_id: event for event in normalized_events}
        proposals = []
        for rank, segment in enumerate(ranked_segments, start=1):
            member_events = [event_lookup[event_id].metric_event for event_id in segment.member_event_ids]
            proposal_time_window = TimeWindow(segment.start_sec, segment.end_sec)
            metric_packet = [
                _local_metric_context(
                    metric_lookup[name],
                    proposal_time_window,
                    context_steps=self.metric_packet_context_steps,
                )
                for name in segment.metric_names
                if name in metric_lookup
            ]
            candidate_frames = self._select_candidate_frames(
                trajectory_bundle=trajectory_bundle,
                proposal_time_window=proposal_time_window,
            )
            proposals.append(
                KeyEventProposal(
                    event_id=f"{metric_profile.bundle_id}_metric_overlap_{rank:03d}",
                    time_window=proposal_time_window,
                    source=EventProposalSource.METRIC_TRIGGER,
                    trigger_reason=(
                        f"{segment.overlap_count} metric(s) overlap on "
                        f"[{segment.start_sec:.3f}, {segment.end_sec:.3f}]"
                    ),
                    candidate_frames=candidate_frames,
                    metric_events=member_events,
                    metric_packet=metric_packet,
                    metadata={
                        "timeline_rank": rank,
                        "overlap_metric_count": segment.overlap_count,
                        "metric_names": segment.metric_names,
                        "member_event_ids": segment.member_event_ids,
                        "candidate_frame_count": len(candidate_frames),
                    },
                )
            )

        if debug_timeline_path is not None:
            self._write_debug_timeline(
                metric_profile=metric_profile,
                normalized_events=normalized_events,
                metric_support_intervals=metric_support_intervals,
                timeline_segments=ranked_segments,
                debug_timeline_path=Path(debug_timeline_path),
            )

        return proposals

    def _select_candidate_frames(
        self,
        trajectory_bundle: TrajectoryBundle | None,
        proposal_time_window: TimeWindow,
    ) -> list[CameraFrameRef]:
        """Attach sampled camera frames inside the proposal window."""

        if trajectory_bundle is None or not trajectory_bundle.camera_frames:
            return []

        frames_in_window = [
            frame
            for frame in trajectory_bundle.camera_frames
            if proposal_time_window.start_sec <= frame.timestamp_sec <= (proposal_time_window.end_sec or proposal_time_window.start_sec)
        ]
        if not frames_in_window:
            return []

        sampled_frames: list[CameraFrameRef] = []
        camera_names = sorted({frame.camera_name for frame in frames_in_window})
        for camera_name in camera_names:
            camera_frames = [
                frame
                for frame in frames_in_window
                if frame.camera_name == camera_name
            ]
            camera_frames.sort(key=lambda frame: (frame.timestamp_sec, frame.frame_index or -1))
            sampled_frames.extend(camera_frames[:: self.frame_stride_within_window])
        sampled_frames.sort(key=lambda frame: (frame.timestamp_sec, frame.camera_name, frame.frame_index or -1))
        return sampled_frames

    def _normalize_metric_events(
        self,
        metric_lookup: dict[str, MetricResult],
    ) -> list[NormalizedMetricEvent]:
        """Expand point metric events into windows."""

        normalized_events: list[NormalizedMetricEvent] = []
        for metric in metric_lookup.values():
            for index, event in enumerate(metric.events):
                normalized_window = _expand_time_window(
                    event.time_window,
                    point_event_delta_sec=self.point_event_delta_sec,
                )
                normalized_events.append(
                    NormalizedMetricEvent(
                        event_id=f"{metric.name}_event_{index:03d}",
                        metric_name=metric.name,
                        metric_event=event,
                        original_time_window=event.time_window,
                        normalized_time_window=normalized_window,
                    )
                )
        return normalized_events

    def _write_debug_timeline(
        self,
        metric_profile: MetricProfile,
        normalized_events: list[NormalizedMetricEvent],
        metric_support_intervals: list[MetricSupportInterval],
        timeline_segments: list[TimelineSegment],
        debug_timeline_path: Path,
    ) -> None:
        """Write the normalized metric timeline and fused intersections to JSON."""

        debug_timeline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_id": metric_profile.bundle_id,
            "point_event_delta_sec": self.point_event_delta_sec,
            "metric_packet_context_steps": self.metric_packet_context_steps,
            "ignored_metric_names": sorted(self.ignored_metric_names),
            "normalized_metric_events": [
                {
                    "event_id": event.event_id,
                    "metric_name": event.metric_name,
                    "original_time_window": _time_window_to_dict(event.original_time_window),
                    "normalized_time_window": _time_window_to_dict(event.normalized_time_window),
                    "reason": event.metric_event.reason,
                }
                for event in normalized_events
            ],
            "metric_support_intervals": [
                {
                    "metric_name": interval.metric_name,
                    "start_sec": interval.start_sec,
                    "end_sec": interval.end_sec,
                    "member_event_ids": interval.member_event_ids,
                }
                for interval in metric_support_intervals
            ],
            "fused_timeline_segments": [
                {
                    "rank": rank,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "metric_names": segment.metric_names,
                    "overlap_count": segment.overlap_count,
                    "member_event_ids": segment.member_event_ids,
                }
                for rank, segment in enumerate(timeline_segments, start=1)
            ],
        }
        debug_timeline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _expand_time_window(
    time_window: TimeWindow,
    point_event_delta_sec: float,
) -> TimeWindow:
    """Expand point timestamps into windows while leaving interval events unchanged."""

    if time_window.end_sec is not None and time_window.end_sec > time_window.start_sec:
        return TimeWindow(time_window.start_sec, time_window.end_sec)
    start_sec = max(0.0, time_window.start_sec - point_event_delta_sec)
    end_sec = time_window.start_sec + point_event_delta_sec
    return TimeWindow(start_sec, end_sec)


def _merge_events_within_each_metric(
    normalized_events: list[NormalizedMetricEvent],
) -> list[MetricSupportInterval]:
    """Union overlapping event windows within each metric before cross-metric fusion."""

    events_by_metric: dict[str, list[NormalizedMetricEvent]] = {}
    for event in normalized_events:
        events_by_metric.setdefault(event.metric_name, []).append(event)

    support_intervals: list[MetricSupportInterval] = []
    for metric_name, metric_events in events_by_metric.items():
        sorted_events = sorted(
            metric_events,
            key=lambda event: (
                event.normalized_time_window.start_sec,
                event.normalized_time_window.end_sec or event.normalized_time_window.start_sec,
                event.event_id,
            ),
        )
        current_start: float | None = None
        current_end: float | None = None
        current_event_ids: list[str] = []

        for event in sorted_events:
            start_sec = event.normalized_time_window.start_sec
            end_sec = event.normalized_time_window.end_sec or event.normalized_time_window.start_sec
            if current_start is None or current_end is None:
                current_start = start_sec
                current_end = end_sec
                current_event_ids = [event.event_id]
                continue

            if start_sec <= current_end:
                current_end = max(current_end, end_sec)
                current_event_ids.append(event.event_id)
                continue

            support_intervals.append(
                MetricSupportInterval(
                    metric_name=metric_name,
                    start_sec=current_start,
                    end_sec=current_end,
                    member_event_ids=sorted(current_event_ids),
                )
            )
            current_start = start_sec
            current_end = end_sec
            current_event_ids = [event.event_id]

        if current_start is not None and current_end is not None:
            support_intervals.append(
                MetricSupportInterval(
                    metric_name=metric_name,
                    start_sec=current_start,
                    end_sec=current_end,
                    member_event_ids=sorted(current_event_ids),
                )
            )

    return sorted(
        support_intervals,
        key=lambda interval: (interval.start_sec, interval.end_sec, interval.metric_name),
    )


def _build_timeline_segments(
    metric_support_intervals: list[MetricSupportInterval],
) -> list[TimelineSegment]:
    """Build exact intersection intervals from metric-level support intervals."""

    if not metric_support_intervals:
        return []

    boundaries = sorted(
        {
            interval.start_sec
            for interval in metric_support_intervals
        }
    |   {
            interval.end_sec
            for interval in metric_support_intervals
        }
    )

    segments: list[TimelineSegment] = []
    for start_sec, end_sec in zip(boundaries, boundaries[1:]):
        if end_sec <= start_sec:
            continue
        active_intervals = [
            interval
            for interval in metric_support_intervals
            if interval.start_sec <= start_sec
            and interval.end_sec >= end_sec
        ]
        if not active_intervals:
            continue
        metric_names = sorted(interval.metric_name for interval in active_intervals)
        member_event_ids = sorted(
            {
                event_id
                for interval in active_intervals
                for event_id in interval.member_event_ids
            }
        )
        segment = TimelineSegment(
            start_sec=start_sec,
            end_sec=end_sec,
            metric_names=metric_names,
            member_event_ids=member_event_ids,
        )
        if (
            segments
            # If the previous event was just a part of this event then merge!
            and segments[-1].metric_names == segment.metric_names 
            and segments[-1].member_event_ids == segment.member_event_ids
        ):
            segments[-1] = TimelineSegment(
                start_sec=segments[-1].start_sec,
                end_sec=segment.end_sec,
                metric_names=segment.metric_names,
                member_event_ids=segment.member_event_ids,
            )
        else:
            segments.append(segment)
    return segments


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
