"""Tests for gripper-specific metric semantics."""

from __future__ import annotations

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.gripper import GripperChatterMetric, GripperTransitionMetric
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_single_gripper_transition_is_key_event() -> None:
    timestamps = [index * 0.05 for index in range(4)]
    records = [
        {"gripper": [0.0]},
        {"gripper": [0.0]},
        {"gripper": [1.0]},
        {"gripper": [1.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="transition",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = GripperTransitionMetric().compute(MetricComputationContext(bundle=bundle))

    assert len(result.events) == 1
    assert result.events[0].time_window.start_sec == 0.1
    assert result.events[0].value["transition_direction"] == "open_to_closed"
    assert result.events[0].value["changed_dimensions"] == [0]


def test_gripper_transition_reports_multiple_changed_dimensions() -> None:
    timestamps = [0.0, 0.05]
    records = [
        {"gripper": [0.0, 1.0]},
        {"gripper": [1.0, 0.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="multi-dim-transition",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = GripperTransitionMetric().compute(MetricComputationContext(bundle=bundle))

    assert len(result.events) == 1
    assert result.events[0].value["transition_direction"] == "mixed"
    assert result.events[0].value["transition_by_dimension"] == {
        "0": "open_to_closed",
        "1": "closed_to_open",
    }


def test_single_gripper_transition_is_not_chatter() -> None:
    timestamps = [index * 0.05 for index in range(6)]
    records = [
        {"gripper": [0.0]},
        {"gripper": [0.0]},
        {"gripper": [0.0]},
        {"gripper": [1.0]},
        {"gripper": [1.0]},
        {"gripper": [1.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="single-transition",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = GripperChatterMetric().compute(MetricComputationContext(bundle=bundle))

    assert [sample.value for sample in result.samples] == [0, 0, 1, 0, 0]
    assert result.events == []


def test_multiple_gripper_transitions_can_be_chatter() -> None:
    timestamps = [index * 0.05 for index in range(8)]
    records = [
        {"gripper": [0.0]},
        {"gripper": [1.0]},
        {"gripper": [0.0]},
        {"gripper": [1.0]},
        {"gripper": [0.0]},
        {"gripper": [1.0]},
        {"gripper": [1.0]},
        {"gripper": [1.0]},
    ]
    bundle = TrajectoryBundle(
        bundle_id="multi-transition",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
            )
        ],
    )

    result = GripperChatterMetric().compute(MetricComputationContext(bundle=bundle))

    assert result.events
    assert all(event.value["transition_count"] >= 2 for event in result.events)
