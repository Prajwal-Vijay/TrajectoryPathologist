"""Smoke tests for ECDF-surprise flood key-event selection."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.events import MetricEventSelector
from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    MetricProfile,
    MetricResult,
    MetricSample,
    TrajectoryBundle,
)


def test_metric_selector_extracts_surprise_flood_event_windows(
    tmp_path: Path | None = None,
) -> None:
    output_dir = tmp_path or Path("/tmp/pathologist_metric_selector_surprise_flood_test")
    debug_path = output_dir / "timeline_debug.json"
    profile = _aggregate_profile()
    selector = MetricEventSelector(
        metric_packet_context_steps=1,
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "max_proposals": 2,
        },
    )

    proposals = selector.select(profile, debug_timeline_path=debug_path)

    assert len(proposals) == 2
    assert proposals[0].event_id == "demo_surprise_flood_001"
    assert proposals[0].metadata["selector_mode"] == "surprise_flood"
    assert proposals[0].metadata["peak_sec"] == 6.0
    assert proposals[0].metadata["base_score"] == 0.0
    assert proposals[0].metadata["persistence"] == 8.0
    assert round(proposals[0].time_window.start_sec, 3) == 5.333
    assert round(proposals[0].time_window.end_sec, 3) == 6.667
    assert round(proposals[0].metadata["area_above_base"], 3) == 8.0
    assert proposals[0].metric_events[0].metric_name == "ecod_aggregate_score"
    assert proposals[0].metric_packet[0].name == "ecod_aggregate_score"
    assert [sample.timestamp_sec for sample in proposals[0].metric_packet[0].samples] == [5.0, 6.0, 7.0]

    assert proposals[1].metadata["peak_sec"] == 2.0
    assert round(proposals[1].time_window.start_sec, 3) == 1.375
    assert round(proposals[1].time_window.end_sec, 3) == 2.625

    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["selector_mode"] == "surprise_flood"
    assert len(payload["surprise_flood_events"]) == 2
    assert payload["surprise_flood_events"][0]["peak_sec"] == 6.0


def test_metric_selector_attaches_visual_context_frames() -> None:
    bundle = TrajectoryBundle(
        bundle_id="demo",
        source_format=BundleFormat.CUSTOM,
        camera_frames=[
            CameraFrameRef(camera_name="head", timestamp_sec=5.0, image_path=Path("/tmp/head_050.jpg"), frame_index=50),
            CameraFrameRef(camera_name="head", timestamp_sec=5.5, image_path=Path("/tmp/head_055.jpg"), frame_index=55),
            CameraFrameRef(camera_name="head", timestamp_sec=6.0, image_path=Path("/tmp/head_060.jpg"), frame_index=60),
            CameraFrameRef(camera_name="head", timestamp_sec=6.5, image_path=Path("/tmp/head_065.jpg"), frame_index=65),
            CameraFrameRef(camera_name="head", timestamp_sec=7.0, image_path=Path("/tmp/head_070.jpg"), frame_index=70),
            CameraFrameRef(camera_name="wrist", timestamp_sec=6.0, image_path=Path("/tmp/wrist_060.jpg"), frame_index=60),
        ],
    )
    selector = MetricEventSelector(
        frame_stride_within_window=1,
        visual_context_sec=0.2,
        max_frames_per_camera=3,
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "max_proposals": 1,
        },
    )

    proposals = selector.select(_aggregate_profile(), trajectory_bundle=bundle)

    assert proposals
    assert [frame.frame_index for frame in proposals[0].candidate_frames] == [55, 60, 65, 60]
    assert {frame.camera_name for frame in proposals[0].candidate_frames} == {"head", "wrist"}


def test_metric_selector_writes_ecod_aggregate_plots_and_local_maxima(tmp_path: Path) -> None:
    output_base = tmp_path / "ecod_aggregates" / "demo"
    selector = MetricEventSelector(
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "max_proposals": 2,
        },
    )

    selector.select(_aggregate_profile(), ecod_aggregate_path=output_base)

    assert output_base.with_suffix(".json").exists()
    assert output_base.with_name("demo_raw.png").exists()
    assert output_base.with_name("demo_smoothed.png").exists()
    payload = json.loads(output_base.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["available"] is True
    assert payload["local_maxima_count"] >= 2
    assert payload["selected_local_maxima_count"] == 2
    assert payload["raw_scores"]
    assert payload["smoothed_scores"]
    assert payload["source_metric_series"]["velocity_spike"]["raw_values"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
    ]


def test_metric_selector_returns_no_proposals_without_aggregate_score() -> None:
    profile = MetricProfile(
        bundle_id="demo",
        metrics=[
            MetricResult(
                name="motion_smoothness",
                description="raw jerk",
                samples=[
                    MetricSample(timestamp_sec=0.0, value=1.0),
                    MetricSample(timestamp_sec=1.0, value=2.0),
                    MetricSample(timestamp_sec=2.0, value=1.0),
                ],
            )
        ],
    )

    assert MetricEventSelector().select(profile) == []


def test_metric_selector_includes_zero_persistence_peak() -> None:
    profile = MetricProfile(
        bundle_id="flat",
        metrics=[
            MetricResult(
                name="ecod_aggregate_score",
                description="flat aggregate surprise",
                samples=[
                    MetricSample(timestamp_sec=float(index), value=1.0)
                    for index in range(5)
                ],
            )
        ],
    )
    selector = MetricEventSelector(
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "max_proposals": 5,
        },
    )

    proposals = selector.select(profile)

    assert len(proposals) == 1
    assert proposals[0].metadata["persistence"] == 0.0
    assert proposals[0].metadata["rank_score_type"] == "area_above_base"


def test_metric_selector_ranks_proposals_by_area_persistence_then_peak_height() -> None:
    profile = _area_vs_persistence_profile()
    selector = MetricEventSelector(
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "max_proposals": 2,
        },
    )

    proposals = selector.select(profile)

    assert [proposal.metadata["peak_sec"] for proposal in proposals] == [6.0, 2.0]
    assert [proposal.metadata["rank_score_type"] for proposal in proposals] == [
        "area_above_base",
        "area_above_base",
    ]


def test_metric_selector_can_rank_by_persistence_then_area() -> None:
    selector = MetricEventSelector(
        surprise_flood_config={
            "aggregate_metric_name": "ecod_aggregate_score",
            "smoothing_window": 1,
            "gamma": 0.5,
            "ranking": "persistence",
            "max_proposals": 2,
        },
    )

    proposals = selector.select(_area_vs_persistence_profile())

    assert [proposal.metadata["peak_sec"] for proposal in proposals] == [2.0, 6.0]
    assert [proposal.metadata["rank_score_type"] for proposal in proposals] == [
        "persistence",
        "persistence",
    ]
    assert [proposal.metadata["proposal_score"] for proposal in proposals] == [
        proposal.metadata["persistence"] for proposal in proposals
    ]


def _area_vs_persistence_profile() -> MetricProfile:
    return MetricProfile(
        bundle_id="height_rank",
        metrics=[
            MetricResult(
                name="ecod_aggregate_score",
                description="aggregate surprise",
                samples=[
                    MetricSample(timestamp_sec=0.0, value=0.0),
                    MetricSample(timestamp_sec=1.0, value=0.0),
                    MetricSample(timestamp_sec=2.0, value=10.0),
                    MetricSample(timestamp_sec=3.0, value=0.0),
                    MetricSample(timestamp_sec=4.0, value=0.0),
                    MetricSample(timestamp_sec=5.0, value=6.0),
                    MetricSample(timestamp_sec=6.0, value=6.0),
                    MetricSample(timestamp_sec=7.0, value=6.0),
                    MetricSample(timestamp_sec=8.0, value=6.0),
                    MetricSample(timestamp_sec=9.0, value=0.0),
                    MetricSample(timestamp_sec=10.0, value=0.0),
                ],
            )
        ],
    )


def _aggregate_profile() -> MetricProfile:
    return MetricProfile(
        bundle_id="demo",
        metrics=[
            MetricResult(
                name="ecod_aggregate_score",
                description="aggregate surprise",
                unit="surprise",
                metadata={"source_metric_names": ["velocity_spike"]},
                samples=[
                    MetricSample(timestamp_sec=0.0, value=0.0),
                    MetricSample(timestamp_sec=1.0, value=1.0),
                    MetricSample(timestamp_sec=2.0, value=5.0),
                    MetricSample(timestamp_sec=3.0, value=1.0),
                    MetricSample(timestamp_sec=4.0, value=0.0),
                    MetricSample(timestamp_sec=5.0, value=2.0),
                    MetricSample(timestamp_sec=6.0, value=8.0),
                    MetricSample(timestamp_sec=7.0, value=2.0),
                    MetricSample(timestamp_sec=8.0, value=0.0),
                ],
            ),
            MetricResult(
                name="velocity_spike",
                description="raw velocity spike score",
                unit="m/s",
                samples=[
                    MetricSample(timestamp_sec=0.0, value=0.0),
                    MetricSample(timestamp_sec=1.0, value=0.1),
                    MetricSample(timestamp_sec=2.0, value=0.2),
                    MetricSample(timestamp_sec=3.0, value=0.3),
                    MetricSample(timestamp_sec=4.0, value=0.4),
                    MetricSample(timestamp_sec=5.0, value=0.5),
                    MetricSample(timestamp_sec=6.0, value=0.6),
                    MetricSample(timestamp_sec=7.0, value=0.7),
                    MetricSample(timestamp_sec=8.0, value=0.8),
                ],
            )
        ],
    )
