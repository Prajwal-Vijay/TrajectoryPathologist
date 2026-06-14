"""Smoke tests for metric-event selection."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.events import MetricEventSelector
from pathologist.schemas import (
    CameraFrameRef,
    BundleFormat,
    MetricEvent,
    MetricProfile,
    MetricResult,
    TrajectoryBundle,
    Severity,
    TimeWindow,
)


def test_metric_selector_fuses_overlapping_metric_events(tmp_path: Path | None = None) -> None:
    output_dir = tmp_path or Path("/tmp/pathologist_metric_selector_test")
    debug_path = output_dir / "timeline_debug.json"
    profile = MetricProfile(
        bundle_id="demo",
        metrics=[
            MetricResult(
                name="motion_smoothness",
                description="jerk spikes",
                events=[
                    MetricEvent(
                        metric_name="motion_smoothness",
                        time_window=TimeWindow(10.0),
                        value=12.0,
                        severity=Severity.HIGH,
                        reason="jerk spike",
                    )
                ],
            ),
            MetricResult(
                name="gripper_chatter",
                description="gripper chatter",
                events=[
                    MetricEvent(
                        metric_name="gripper_chatter",
                        time_window=TimeWindow(9.8, 10.4),
                        value=3.0,
                        severity=Severity.MEDIUM,
                        reason="local chatter burst",
                    )
                ],
            ),
            MetricResult(
                name="action_range_saturation",
                description="action saturation",
                events=[
                    MetricEvent(
                        metric_name="action_range_saturation",
                        time_window=TimeWindow(10.1, 10.3),
                        value={"dims": [0]},
                        severity=Severity.MEDIUM,
                        reason="action hit bound",
                    )
                ],
            ),
            MetricResult(
                name="timestamp_regularity",
                description="timing anomalies",
                events=[
                    MetricEvent(
                        metric_name="timestamp_regularity",
                        time_window=TimeWindow(10.1, 10.5),
                        value=0.2,
                        severity=Severity.LOW,
                        reason="gap",
                    )
                ],
            ),
        ],
    )

    bundle = TrajectoryBundle(
        bundle_id="demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=[
            CameraFrameRef(camera_name="head", timestamp_sec=9.85, image_path=Path("/tmp/head_000.jpg"), frame_index=0),
            CameraFrameRef(camera_name="head", timestamp_sec=10.10, image_path=Path("/tmp/head_001.jpg"), frame_index=1),
            CameraFrameRef(camera_name="head", timestamp_sec=10.15, image_path=Path("/tmp/head_002.jpg"), frame_index=2),
            CameraFrameRef(camera_name="left", timestamp_sec=10.12, image_path=Path("/tmp/left_001.jpg"), frame_index=1),
            CameraFrameRef(camera_name="left", timestamp_sec=10.18, image_path=Path("/tmp/left_002.jpg"), frame_index=2),
            CameraFrameRef(camera_name="right", timestamp_sec=10.19, image_path=Path("/tmp/right_001.jpg"), frame_index=1),
        ],
    )

    selector = MetricEventSelector(point_event_delta_sec=0.2, frame_stride_within_window=10)
    proposals = selector.select(profile, trajectory_bundle=bundle, debug_timeline_path=debug_path)

    assert proposals
    assert proposals[0].time_window is not None
    assert proposals[0].time_window.start_sec == 10.1
    assert proposals[0].time_window.end_sec == 10.2
    assert proposals[0].metadata["overlap_metric_count"] == 3
    assert "timestamp_regularity" not in proposals[0].metadata["metric_names"]
    assert len(proposals[0].candidate_frames) == 3
    assert {frame.camera_name for frame in proposals[0].candidate_frames} == {"head", "left", "right"}

    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "demo"
    assert payload["point_event_delta_sec"] == 0.2
    assert payload["fused_timeline_segments"][0]["overlap_count"] == 3
