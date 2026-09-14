"""Tests for kinematic metric semantics."""

from __future__ import annotations

import pytest

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.kinematics import (
    EefAngularJerkMetric,
    EefAngularVelocitySpikeMetric,
    EefLinearVelocitySpikeMetric,
    EefRotationalStallingMetric,
    EefTranslationalJerkMetric,
    EefTranslationalStallingMetric,
    StallingMetric,
)
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_eef_velocity_metrics_use_separate_linear_and_angular_norms() -> None:
    bundle = _eef_velocity_bundle()
    context = MetricComputationContext(bundle=bundle)

    linear_spike = EefLinearVelocitySpikeMetric().compute(context)
    angular_spike = EefAngularVelocitySpikeMetric().compute(context)
    translational_stall = EefTranslationalStallingMetric().compute(context)
    rotational_stall = EefRotationalStallingMetric().compute(context)

    assert [sample.value for sample in linear_spike.samples] == pytest.approx(
        [0.0, 1.0, 3.0, 6.0]
    )
    assert [sample.value for sample in angular_spike.samples] == pytest.approx(
        [0.0, 2.0, 6.0, 12.0]
    )
    assert [sample.value for sample in translational_stall.samples] == pytest.approx(
        [sample.value for sample in linear_spike.samples]
    )
    assert [sample.value for sample in rotational_stall.samples] == pytest.approx(
        [sample.value for sample in angular_spike.samples]
    )
    assert linear_spike.metadata["surprise_tail_interpretation"] == "auto"
    assert angular_spike.metadata["surprise_tail_interpretation"] == "auto"
    assert translational_stall.metadata["surprise_tail_interpretation"] == "left"
    assert rotational_stall.metadata["surprise_tail_interpretation"] == "left"


def test_eef_jerk_metrics_derive_from_measured_cartesian_velocity() -> None:
    context = MetricComputationContext(bundle=_eef_velocity_bundle())

    translational = EefTranslationalJerkMetric().compute(context)
    angular = EefAngularJerkMetric().compute(context)

    assert [sample.timestamp_sec for sample in translational.samples] == [2.0, 3.0]
    assert [sample.value for sample in translational.samples] == pytest.approx(
        [1.0, 1.0]
    )
    assert [sample.value for sample in angular.samples] == pytest.approx([2.0, 2.0])
    assert translational.metadata["surprise_tail_interpretation"] == "auto"
    assert angular.metadata["surprise_tail_interpretation"] == "auto"


def test_eef_metrics_keep_bimanual_components_separate_for_ecod() -> None:
    timestamps = [0.0, 1.0, 2.0, 3.0]
    bundle = TrajectoryBundle(
        bundle_id="bimanual-eef",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="eef_velocity",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=[
                    {
                        "eef_linear_velocity": [left, 0.0, 0.0, right, 0.0, 0.0],
                        "eef_angular_velocity": [0.0] * 6,
                    }
                    for left, right in zip(
                        [0.0, 1.0, 3.0, 6.0],
                        [0.0, 2.0, 6.0, 12.0],
                    )
                ],
                metadata={
                    "vector_groups": {
                        "left": [0, 1, 2],
                        "right": [3, 4, 5],
                    }
                },
            )
        ],
    )
    context = MetricComputationContext(bundle=bundle)

    velocity = EefLinearVelocitySpikeMetric().compute(context)
    jerk = EefTranslationalJerkMetric().compute(context)

    assert velocity.metadata["ecod_component_values"] == {
        "left": [0.0, 1.0, 3.0, 6.0],
        "right": [0.0, 2.0, 6.0, 12.0],
    }
    assert jerk.metadata["ecod_component_values"] == {
        "left": [1.0, 1.0],
        "right": [2.0, 2.0],
    }


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
                metadata={"vector_groups": {"left": [0], "right": [1]}},
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
                metadata={"vector_groups": {"left": [0]}},
            ),
        ],
    )

    result = StallingMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["includes_gripper_state_change"] is True
    assert result.metadata["gripper_dims_used"] == 1
    assert [sample.value for sample in result.samples] == [0.0, 10.0, 0.0]


def test_stalling_tracks_each_bimanual_group_before_ecod() -> None:
    timestamps = [0.0, 0.1, 0.2]
    bundle = TrajectoryBundle(
        bundle_id="bimanual-stalling",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {"proprioception": [0.0, 0.0]},
                    {"proprioception": [0.0, 1.0]},
                    {"proprioception": [2.0, 1.0]},
                ],
                metadata={"vector_groups": {"left": [0], "right": [1]}},
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {"gripper": [0.0, 0.0]},
                    {"gripper": [0.0, 1.0]},
                    {"gripper": [1.0, 1.0]},
                ],
                metadata={"vector_groups": {"left": [0], "right": [1]}},
            ),
        ],
    )

    result = StallingMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.metadata["ecod_component_values"] == {
        "left": [0.0, 10.0 * (5.0**0.5)],
        "right": [10.0 * (2.0**0.5), 0.0],
    }


def _eef_velocity_bundle() -> TrajectoryBundle:
    timestamps = [0.0, 1.0, 2.0, 3.0]
    linear_x = [0.0, 1.0, 3.0, 6.0]
    angular_x = [0.0, 2.0, 6.0, 12.0]
    return TrajectoryBundle(
        bundle_id="eef-velocity",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="eef_velocity",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=[
                    {
                        "eef_linear_velocity": [linear, 0.0, 0.0],
                        "eef_angular_velocity": [angular, 0.0, 0.0],
                    }
                    for linear, angular in zip(linear_x, angular_x)
                ],
            )
        ],
    )
