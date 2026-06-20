"""Tests for action and velocity event-trigger metrics."""

from __future__ import annotations

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.kinematics import VelocitySpikeMetric
from pathologist.metrics.signals import ActionDiscontinuityMetric
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_action_discontinuity_flags_abrupt_command_jump_without_bounds() -> None:
    timestamps = [index * 0.05 for index in range(6)]
    records = [
        {"action": [0.0, 0.0]},
        {"action": [0.1, 0.0]},
        {"action": [0.2, 0.0]},
        {"action": [5.0, 0.0]},
        {"action": [5.1, 0.0]},
        {"action": [5.2, 0.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="action-jump",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = ActionDiscontinuityMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["available"] is True
    assert result.events
    assert result.events[0].time_window.start_sec == 0.15000000000000002
    assert result.events[0].value["max_changed_dimension"] == 0
    assert result.events[0].value["action_delta_norm"] > 4.0


def test_action_discontinuity_can_ignore_gripper_dimensions() -> None:
    timestamps = [index * 0.05 for index in range(6)]
    records = [
        {"action": [0.0, 0.0, 0.0]},
        {"action": [0.1, 0.0, 0.0]},
        {"action": [0.2, 0.0, 1.0]},
        {"action": [5.0, 0.0, 0.0]},
        {"action": [5.1, 0.0, 1.0]},
        {"action": [5.2, 0.0, 0.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="ignore-gripper-action",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = ActionDiscontinuityMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={"metrics": {"action_discontinuity": {"ignored_dimensions": [2]}}},
        )
    )

    assert result.metadata["ignored_dimensions"] == [2]
    assert result.events
    assert result.events[0].time_window.start_sec == 0.15000000000000002
    assert result.events[0].value["max_changed_dimension"] == 0
    assert all(event.value["max_changed_dimension"] != 2 for event in result.events)


def test_velocity_spike_groups_contiguous_high_velocity_region() -> None:
    timestamps = [index * 0.05 for index in range(8)]
    records = [
        {"qvel": [0.1, 0.0]},
        {"qvel": [0.1, 0.0]},
        {"qvel": [5.0, 0.0]},
        {"qvel": [6.0, 0.0]},
        {"qvel": [5.5, 0.0]},
        {"qvel": [0.1, 0.0]},
        {"qvel": [0.1, 0.0]},
        {"qvel": [0.1, 0.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="velocity-spike",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="qvel",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = VelocitySpikeMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={"metrics": {"velocity_spike": {"medium_percentile": 75, "high_percentile": 99}}},
        )
    )

    assert result.metadata["available"] is True
    assert result.events
    assert result.events[0].time_window.start_sec == 0.1
    assert result.events[0].time_window.end_sec == 0.2
    assert result.events[0].value["sample_count"] == 3
    assert result.events[0].value["peak_velocity_norm"] == 6.0


def test_velocity_spike_unavailable_without_velocity_stream() -> None:
    bundle = TrajectoryBundle(
        bundle_id="no-velocity",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=[0.0, 0.05],
                records=[
                    {"proprioception": [0.0]},
                    {"proprioception": [1.0]},
                ],
            )
        ],
    )

    result = VelocitySpikeMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["available"] is False
