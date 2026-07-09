"""Tests for scalar action and velocity source metrics."""

from __future__ import annotations

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.kinematics import VelocitySpikeMetric
from pathologist.metrics.signals import ActionDiscontinuityMetric
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_action_discontinuity_records_command_delta_samples_without_bounds() -> None:
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
    assert not result.events
    assert not result.thresholds
    assert [sample.timestamp_sec for sample in result.samples] == timestamps[1:]
    assert max(sample.value for sample in result.samples) > 4.0


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
    assert not result.events
    assert not result.thresholds
    assert max(sample.value for sample in result.samples) > 4.0


def test_action_discontinuity_reports_out_of_range_configured_ignored_dimensions() -> None:
    timestamps = [index * 0.05 for index in range(6)]
    records = [
        {"action": [0.0, 0.0, 0.0]},
        {"action": [0.1, 0.0, 0.0]},
        {"action": [0.2, 0.0, 0.0]},
        {"action": [5.0, 0.0, 0.0]},
        {"action": [5.1, 0.0, 0.0]},
        {"action": [5.2, 0.0, 0.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="adapter-removed-grippers",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    "raw_action_width": 14,
                    "metric_action_width": 12,
                    "source_gripper_indices_removed": [6, 13],
                },
            )
        ],
    )

    result = ActionDiscontinuityMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={"metrics": {"action_discontinuity": {"ignored_dimensions": [14, 15]}}},
        )
    )

    assert result.metadata["ignored_dimensions"] == []
    assert result.metadata["configured_ignored_dimensions"] == [14, 15]
    assert result.metadata["ignored_dimensions_out_of_range"] == [14, 15]
    assert result.metadata["source_gripper_indices_removed"] == [6, 13]
    assert result.metadata["metric_action_width"] == 3
    assert result.metadata["raw_action_width"] == 14
    assert not result.events
    assert not result.thresholds


def test_velocity_spike_records_velocity_norm_samples() -> None:
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

    result = VelocitySpikeMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["available"] is True
    assert not result.events
    assert not result.thresholds
    assert [sample.timestamp_sec for sample in result.samples] == timestamps
    assert max(sample.value for sample in result.samples) == 6.0


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
