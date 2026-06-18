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
    MetricSample,
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
                samples=[
                    MetricSample(timestamp_sec=9.0, value=1.0),
                    MetricSample(timestamp_sec=9.5, value=2.0),
                    MetricSample(timestamp_sec=10.0, value=12.0),
                    MetricSample(timestamp_sec=10.5, value=3.0),
                    MetricSample(timestamp_sec=11.0, value=1.0),
                ],
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
                metadata={
                    "chatter_bins": [
                        {"start_sec": 9.0, "end_sec": 9.5, "transition_count": 1},
                        {"start_sec": 10.05, "end_sec": 10.15, "transition_count": 3},
                        {"start_sec": 10.25, "end_sec": 10.30, "transition_count": 2},
                    ],
                },
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

    selector = MetricEventSelector(
        point_event_delta_sec=0.2,
        frame_stride_within_window=10,
        metric_packet_context_steps=1,
    )
    proposals = selector.select(profile, trajectory_bundle=bundle, debug_timeline_path=debug_path)

    assert proposals
    assert proposals[0].time_window is not None
    assert proposals[0].time_window.start_sec == 10.1
    assert proposals[0].time_window.end_sec == 10.2
    assert proposals[0].metadata["overlap_metric_count"] == 3
    assert "timestamp_regularity" not in proposals[0].metadata["metric_names"]
    assert len(proposals[0].candidate_frames) == 3
    assert {frame.camera_name for frame in proposals[0].candidate_frames} == {"head", "left", "right"}
    motion_packet = next(packet for packet in proposals[0].metric_packet if packet.name == "motion_smoothness")
    assert [sample.timestamp_sec for sample in motion_packet.samples] == [10.0, 10.5]
    assert motion_packet.metadata["original_sample_count"] == 5
    assert motion_packet.metadata["local_sample_count"] == 2
    gripper_packet = next(packet for packet in proposals[0].metric_packet if packet.name == "gripper_chatter")
    assert gripper_packet.metadata["chatter_bins"] == [
        {"start_sec": 10.05, "end_sec": 10.15, "transition_count": 3}
    ]
    assert gripper_packet.metadata["original_chatter_bins_count"] == 3
    assert gripper_packet.metadata["local_chatter_bins_count"] == 1

    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["bundle_id"] == "demo"
    assert payload["point_event_delta_sec"] == 0.2
    assert payload["metric_packet_context_steps"] == 1
    assert payload["fused_timeline_segments"][0]["overlap_count"] == 3


def test_metric_selector_unions_same_metric_events_before_intersection(tmp_path: Path | None = None) -> None:
    output_dir = tmp_path or Path("/tmp/pathologist_metric_selector_union_test")
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
                        value=10.0,
                        severity=Severity.HIGH,
                        reason="first jerk spike",
                    ),
                    MetricEvent(
                        metric_name="motion_smoothness",
                        time_window=TimeWindow(10.1),
                        value=11.0,
                        severity=Severity.HIGH,
                        reason="second jerk spike",
                    ),
                ],
            ),
            MetricResult(
                name="gripper_chatter",
                description="gripper chatter",
                events=[
                    MetricEvent(
                        metric_name="gripper_chatter",
                        time_window=TimeWindow(10.05, 10.15),
                        value=2,
                        severity=Severity.MEDIUM,
                        reason="chatter burst",
                    )
                ],
            ),
        ],
    )

    selector = MetricEventSelector(point_event_delta_sec=0.2)
    proposals = selector.select(profile, debug_timeline_path=debug_path)

    assert proposals[0].time_window is not None
    assert proposals[0].time_window.start_sec == 10.05
    assert proposals[0].time_window.end_sec == 10.15
    assert proposals[0].metadata["overlap_metric_count"] == 2
    assert proposals[0].metadata["metric_names"] == ["gripper_chatter", "motion_smoothness"]
    assert proposals[0].metadata["member_event_ids"] == [
        "gripper_chatter_event_000",
        "motion_smoothness_event_000",
        "motion_smoothness_event_001",
    ]

    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    motion_support = [
        interval
        for interval in payload["metric_support_intervals"]
        if interval["metric_name"] == "motion_smoothness"
    ]
    assert len(motion_support) == 1
    assert motion_support[0]["start_sec"] == 9.8
    assert round(motion_support[0]["end_sec"], 3) == 10.3
    assert motion_support[0]["member_event_ids"] == [
        "motion_smoothness_event_000",
        "motion_smoothness_event_001",
    ]


def test_metric_selector_attaches_visual_context_frames() -> None:
    profile = MetricProfile(
        bundle_id="demo",
        metrics=[
            MetricResult(
                name="motion_smoothness",
                description="jerk spikes",
                events=[
                    MetricEvent(
                        metric_name="motion_smoothness",
                        time_window=TimeWindow(229.6),
                        value=100.0,
                        severity=Severity.HIGH,
                        reason="jerk spike",
                    )
                ],
            )
        ],
    )
    bundle = TrajectoryBundle(
        bundle_id="demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=[
            CameraFrameRef(camera_name="head", timestamp_sec=229.5, image_path=Path("/tmp/head_459.jpg"), frame_index=459),
            CameraFrameRef(camera_name="head", timestamp_sec=230.0, image_path=Path("/tmp/head_460.jpg"), frame_index=460),
            CameraFrameRef(camera_name="head", timestamp_sec=230.5, image_path=Path("/tmp/head_461.jpg"), frame_index=461),
            CameraFrameRef(camera_name="head", timestamp_sec=232.0, image_path=Path("/tmp/head_464.jpg"), frame_index=464),
        ],
    )

    selector = MetricEventSelector(
        point_event_delta_sec=0.2,
        frame_stride_within_window=1,
        visual_context_sec=1.0,
        max_frames_per_camera=5,
    )
    proposals = selector.select(profile, trajectory_bundle=bundle)

    assert proposals[0].time_window.start_sec == 229.4
    assert proposals[0].time_window.end_sec == 229.79999999999998
    assert [frame.frame_index for frame in proposals[0].candidate_frames] == [459, 460, 461]
