"""Tests for centered windowed excess-path retry signals."""

from __future__ import annotations

import pytest

from pathologist.metrics import MetricEngine, windowed_excess_path
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_windowed_excess_path_is_zero_for_monotonic_motion_and_positive_for_retry() -> None:
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]

    monotonic = [
        windowed_excess_path(
            timestamps=timestamps,
            values=[[value] for value in [0.0, 1.0, 2.0, 3.0, 4.0]],
            center_sec=center,
            half_width_sec=1.0,
        )
        for center in [1.0, 2.0, 3.0]
    ]
    retry = [
        windowed_excess_path(
            timestamps=timestamps,
            values=[[value] for value in [0.0, 1.0, 0.0, 1.0, 2.0]],
            center_sec=center,
            half_width_sec=1.0,
        )
        for center in [1.0, 2.0, 3.0]
    ]

    assert monotonic == [0.0, 0.0, 0.0]
    assert retry == pytest.approx([2.0, 2.0, 0.0])


def test_engine_adds_exactly_three_enabled_excess_path_sources_before_aggregation() -> None:
    timestamps = [float(index) for index in range(7)]
    left = [float(index) for index in range(7)]
    right_retry = [0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    bundle = TrajectoryBundle(
        bundle_id="excess-path",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {"proprioception": [left_value, right_value]}
                    for left_value, right_value in zip(left, right_retry)
                ],
                metadata={"vector_groups": {"left": [0], "right": [1]}},
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {"gripper": [left_value, right_value]}
                    for left_value, right_value in zip(left, right_retry)
                ],
                metadata={"vector_groups": {"left": [0], "right": [1]}},
            ),
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=[
                    {
                        "wrist_poses": {
                            "left": {"position": [left_value, 0.0, 0.0]},
                            "right": {"position": [right_value, 0.0, 0.0]},
                        }
                    }
                    for left_value, right_value in zip(left, right_retry)
                ],
            ),
        ],
    )
    source_names = [
        "joint_excess_path",
        "gripper_excess_path",
        "eef_excess_path",
    ]
    config = {
        "metrics": {
            name: {"enabled": True, "half_width_sec": 1.0}
            for name in source_names
        },
        "surprise": {"source_metrics": source_names},
    }

    profile = MetricEngine(config=config).compute(bundle)
    by_name = {metric.name: metric for metric in profile.metrics}

    for source_name in source_names:
        result = by_name[source_name]
        assert [sample.timestamp_sec for sample in result.samples] == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
        assert [sample.value for sample in result.samples[:2]] == pytest.approx([2.0, 2.0])
        assert result.metadata["group_reduction"] == "max"
        assert by_name[f"{source_name}_ecod_score"].metadata["score_tail_mode"] == "right"

    aggregate = by_name["ecod_aggregate_score"]
    assert set(aggregate.metadata["source_metric_names"]) == set(source_names)
