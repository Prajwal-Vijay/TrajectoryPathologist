"""Tests for kinematic metric semantics."""

from __future__ import annotations

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.kinematics import StallingMetric
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_stalling_metric_counts_gripper_transition_as_motion() -> None:
    timestamps = [0.0, 0.1, 0.2, 0.3]
    bundle = TrajectoryBundle(
        bundle_id="stall-with-gripper-transition",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {"proprioception": [0.0, 0.0]},
                    {"proprioception": [0.0, 0.0]},
                    {"proprioception": [0.0, 0.0]},
                    {"proprioception": [0.0, 0.0]},
                ],
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {"gripper": [0.0]},
                    {"gripper": [0.0]},
                    {"gripper": [1.0]},
                    {"gripper": [1.0]},
                ],
            ),
        ],
    )

    result = StallingMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["includes_gripper_state_change"] is True
    assert result.metadata["gripper_dims_used"] == 1
    assert [sample.value for sample in result.samples] == [0.0, 10.0, 0.0]
