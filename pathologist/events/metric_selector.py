"""Metric-driven key-event proposal selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


class MetricEventSelector:
    """Fuse metric-event windows into ranked key-event proposals."""

    def __init__(
        self,
        point_event_delta_sec: float,
        ignored_metric_names: set[str] | None = None,
        frame_stride_within_window: int = 10,
    ) -> None:
        if point_event_delta_sec < 0:
            raise ValueError("point_event_delta_sec must be non-negative.")
        if frame_stride_within_window <= 0:
            raise ValueError("frame_stride_within_window must be positive.")
        self.point_event_delta_sec = point_event_delta_sec
        self.ignored_metric_names = ignored_metric_names or {"timestamp_regularity"}
        self.frame_stride_within_window = frame_stride_within_window

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
        timeline_segments = _build_timeline_segments(normalized_events)
        ranked_segments = sorted(
            timeline_segments,
            key=lambda segment: (-segment.overlap_count, -len(segment.member_event_ids), segment.start_sec),
        )

        event_lookup = {event.event_id: event for event in normalized_events}
        proposals = []
        for rank, segment in enumerate(ranked_segments, start=1):
            member_events = [event_lookup[event_id].metric_event for event_id in segment.member_event_ids]
            metric_packet = [metric_lookup[name] for name in segment.metric_names if name in metric_lookup]
            proposal_time_window = TimeWindow(segment.start_sec, segment.end_sec)
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
        timeline_segments: list[TimelineSegment],
        debug_timeline_path: Path,
    ) -> None:
        """Write the normalized metric timeline and fused intersections to JSON."""

        debug_timeline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_id": metric_profile.bundle_id,
            "point_event_delta_sec": self.point_event_delta_sec,
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


def _build_timeline_segments(
    normalized_events: list[NormalizedMetricEvent],
) -> list[TimelineSegment]:
    """Build exact intersection intervals from overlapping normalized events."""

    if not normalized_events:
        return []

    boundaries = sorted(
        {
            event.normalized_time_window.start_sec
            for event in normalized_events
        }
    |   {
            event.normalized_time_window.end_sec or event.normalized_time_window.start_sec
            for event in normalized_events
        }
    )

    segments: list[TimelineSegment] = []
    for start_sec, end_sec in zip(boundaries, boundaries[1:]):
        if end_sec <= start_sec:
            continue
        active_events = [
            event
            for event in normalized_events
            # This basically checks if an event covers this tiny slice of the timeline. Jerks will appear in tiny slices on their own.
            if event.normalized_time_window.start_sec <= start_sec
            and (event.normalized_time_window.end_sec or event.normalized_time_window.start_sec) >= end_sec
        ]
        if not active_events:
            continue
        metric_names = sorted({event.metric_name for event in active_events})
        member_event_ids = sorted(event.event_id for event in active_events)
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


def _time_window_to_dict(time_window: TimeWindow) -> dict[str, float | None]:
    """Serialize a TimeWindow for debug output."""

    return {
        "start_sec": time_window.start_sec,
        "end_sec": time_window.end_sec,
    }
