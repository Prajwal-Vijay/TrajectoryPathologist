"""Tests for REASSEMBLE per-trajectory labeling exports."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.reassemble.artifacts import export_per_trajectory_results
from pathologist.schemas.common import TimeWindow
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.schemas.metrics import MetricProfile, MetricResult, MetricSample
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


def test_export_writes_ecod_series_three_camera_peaks_and_plot(tmp_path: Path) -> None:
    frames = []
    for camera_name in ("hama1", "hama2", "hand"):
        for frame_index, timestamp_sec in enumerate((0.0, 1.0, 2.0)):
            image_path = tmp_path / f"{camera_name}_{frame_index}.jpg"
            image_path.write_bytes(b"frame")
            frames.append(
                CameraFrameRef(
                    camera_name=camera_name,
                    timestamp_sec=timestamp_sec,
                    image_path=image_path,
                    frame_index=frame_index,
                )
            )
    bundle = TrajectoryBundle(bundle_id="demo_1", duration_sec=2.0, camera_frames=frames)
    profile = MetricProfile(
        bundle_id=bundle.bundle_id,
        metrics=[
            MetricResult(
                name="velocity_spike_ecod_score",
                description="ECOD velocity score.",
                samples=[
                    MetricSample(timestamp_sec=0.0, value=1.0),
                    MetricSample(timestamp_sec=1.0, value=4.0),
                    MetricSample(timestamp_sec=2.0, value=2.0),
                ],
                metadata={"source_metric_name": "velocity_spike"},
            ),
            MetricResult(
                name="ecod_aggregate_score",
                description="Aggregate ECOD score.",
                samples=[
                    MetricSample(timestamp_sec=0.0, value=1.0),
                    MetricSample(timestamp_sec=1.0, value=4.0),
                    MetricSample(timestamp_sec=2.0, value=2.0),
                ],
                display_hint="ecod_aggregate_surprise_timeseries",
                metadata={
                    "source_metric_names": ["velocity_spike"],
                    "aggregation_reduction": "sum",
                },
            ),
        ],
    )
    proposal = KeyEventProposal(
        event_id="peak_1",
        time_window=TimeWindow(start_sec=0.8, end_sec=1.2),
        source="metric",
        trigger_reason="persistent ECOD maximum",
        metadata={"peak_sec": 1.0},
    )
    aggregate_debug_path = tmp_path / "aggregate.json"
    aggregate_debug_path.write_text(
        json.dumps(
            {
                "available": True,
                "timestamps_sec": [0.0, 1.0, 2.0],
                "raw_scores": [1.0, 4.0, 2.0],
                "smoothed_scores": [1.0, 3.5, 2.0],
                "smoothing": {"method": "test"},
            }
        ),
        encoding="utf-8",
    )

    summary = export_per_trajectory_results(
        results_root=tmp_path / "per trajectory results",
        bundle=bundle,
        metric_profile=profile,
        proposals=[proposal],
        aggregate_debug_path=aggregate_debug_path,
        ground_truth_events=[
            {
                "label": "first contact",
                "start_sec": 1.0,
                "end_sec": 1.0,
                "reference_sec": 1.0,
            }
        ],
        annotation_labels=["first contact"],
    )

    trajectory_dir = Path(summary["trajectory_dir"])
    assert (trajectory_dir / "ECOD_velocity_spike.json").is_file()
    assert (trajectory_dir / "ECOD_aggregate_score.json").is_file()
    assert (trajectory_dir / "aggregate_ecod_peaks_ground_truth.png").is_file()
    assert len(list((trajectory_dir / "peak frames").glob("*/*.jpg"))) == 3
    assert summary["ecod_metric_count"] == 1
    assert summary["ground_truth_event_count"] == 1
