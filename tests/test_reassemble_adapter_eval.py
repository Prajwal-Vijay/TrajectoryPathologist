"""Tests for REASSEMBLE ingestion and deterministic recall evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologist.adapters.reassemble_bundle import ReassembleBundleAdapter
from pathologist.reassemble.evaluation import _evaluation_row, _failed_high_level_demo_bundles
from pathologist.schemas import BundleFormat, EventProposalSource, TimeWindow
from pathologist.schemas.key_event import KeyEventProposal


def test_reassemble_adapter_parses_failed_high_and_low_segments(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)

    bundle = ReassembleBundleAdapter().load(
        h5_path,
        extract_frames=False,
        robot_downsample_hz=10.0,
    )

    assert bundle.source_format == BundleFormat.REASSEMBLE_HDF5
    assert bundle.raw_metadata["has_failed_segment"] is True
    assert bundle.raw_metadata["failed_high_level_count"] == 1
    assert bundle.raw_metadata["failed_low_level_count"] == 1
    segments = bundle.raw_metadata["reassemble_segments"]
    failed_segments = [segment for segment in segments if not segment["success"]]
    assert [(segment["level"], segment["text"]) for segment in failed_segments] == [
        ("high", "Remove gear"),
        ("low", "Pull"),
    ]
    assert min(stream.timestamps_sec[0] for stream in bundle.streams) == 0.0
    assert {stream.name for stream in bundle.streams} == {
        "proprioception",
        "velocity",
        "gripper",
        "telemetry",
    }


def test_reassemble_adapter_marks_successful_trajectories_as_skippable(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "success_episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=False)

    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)

    assert bundle.raw_metadata["has_failed_segment"] is False
    assert bundle.raw_metadata["failed_high_level_count"] == 0
    assert bundle.raw_metadata["failed_low_level_count"] == 0


def test_reassemble_eval_splits_failed_high_level_actions_into_demo_bundles(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)
    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)

    demo_bundles = _failed_high_level_demo_bundles(bundle)

    assert [demo.bundle_id for demo in demo_bundles] == ["episode_high_1"]
    demo = demo_bundles[0]
    assert demo.duration_sec == 2.0
    assert demo.raw_metadata["evaluation_unit"] == "failed_high_level_action"
    failed_low_segments = [
        segment
        for segment in demo.raw_metadata["reassemble_segments"]
        if segment["level"] == "low" and not segment["success"]
    ]
    assert failed_low_segments == [
        {
            "segment_id": "1/1",
            "level": "low",
            "start_sec": 1.0,
            "end_sec": 2.0,
            "success": False,
            "text": "Pull",
            "index": None,
            "parent_segment_id": "1",
        }
    ]


def test_reassemble_recall_uses_peak_time_inside_failed_segment(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)
    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)
    proposals = [
        _proposal("demo_surprise_flood_001", peak_sec=1.2),
        _proposal("demo_surprise_flood_002", peak_sec=2.1),
    ]

    row = _evaluation_row(bundle, proposals, top_k=2, requested_segment_level="both")

    assert row["recall_by_level"]["high"]["episode_hit_at_k"] is True
    assert row["recall_by_level"]["high"]["failed_segment_recall_at_k"] == 1.0
    assert row["recall_by_level"]["high"]["point_precision_at_k"] == 1.0
    assert row["recall_by_level"]["low"]["episode_hit_at_k"] is True
    assert row["recall_by_level"]["low"]["failed_segment_recall_at_k"] == 1.0
    assert row["recall_by_level"]["low"]["point_precision_at_k"] == 0.5
    assert row["recall_by_level"]["combined"]["failed_segment_recall_at_k"] == 1.0
    assert row["recall_by_level"]["combined"]["selected_peak_hit_count"] == 2


def test_reassemble_recall_misses_when_peak_is_outside_failed_segment(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)
    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)

    row = _evaluation_row(
        bundle,
        [_proposal("demo_surprise_flood_001", peak_sec=4.0)],
        top_k=1,
        requested_segment_level="both",
    )

    assert row["recall_by_level"]["high"]["episode_hit_at_k"] is False
    assert row["recall_by_level"]["high"]["failed_segment_recall_at_k"] == 0.0
    assert row["recall_by_level"]["high"]["point_precision_at_k"] == 0.0
    assert row["recall_by_level"]["low"]["episode_hit_at_k"] is False
    assert row["recall_by_level"]["low"]["point_precision_at_k"] == 0.0


def _proposal(event_id: str, peak_sec: float) -> KeyEventProposal:
    return KeyEventProposal(
        event_id=event_id,
        time_window=TimeWindow(max(0.0, peak_sec - 0.1), peak_sec + 0.1),
        source=EventProposalSource.METRIC_TRIGGER,
        trigger_reason="test peak",
        metadata={"peak_sec": peak_sec, "timeline_rank": 1},
    )


def _write_reassemble_fixture(object_h5py: object, h5_path: Path, include_failure: bool) -> None:
    import numpy as np

    with object_h5py.File(h5_path, "w") as handle:
        robot_state = handle.create_group("robot_state")
        timestamps = handle.create_group("timestamps")
        base_times = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=float)
        robot_state.create_dataset("joint_positions", data=np.stack([np.arange(7) + i for i in range(5)]))
        robot_state.create_dataset("joint_velocities", data=np.ones((5, 7)))
        robot_state.create_dataset("gripper_positions", data=np.zeros((5, 2)))
        robot_state.create_dataset("pose", data=np.stack([[i, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for i in range(5)]))
        robot_state.create_dataset("measured_force", data=np.zeros((5, 3)))
        robot_state.create_dataset("measured_torque", data=np.zeros((5, 3)))
        for name in (
            "joint_positions",
            "joint_velocities",
            "gripper_positions",
            "pose",
            "measured_force",
            "measured_torque",
        ):
            timestamps.create_dataset(name, data=base_times)

        segments = handle.create_group("segments_info")
        first = segments.create_group("0")
        first.create_dataset("index", data=0)
        first.create_dataset("start", data=100.0)
        first.create_dataset("end", data=101.0)
        first.create_dataset("success", data=True)
        first.create_dataset("text", data=b"No action")

        second = segments.create_group("1")
        second.create_dataset("index", data=1)
        second.create_dataset("start", data=101.0)
        second.create_dataset("end", data=103.0)
        second.create_dataset("success", data=not include_failure)
        second.create_dataset("text", data=b"Remove gear")
        low_level = second.create_group("low_level")
        grasp = low_level.create_group("0")
        grasp.create_dataset("start", data=101.0)
        grasp.create_dataset("end", data=102.0)
        grasp.create_dataset("success", data=True)
        grasp.create_dataset("text", data=b"Grasp")
        pull = low_level.create_group("1")
        pull.create_dataset("start", data=102.0)
        pull.create_dataset("end", data=103.0)
        pull.create_dataset("success", data=not include_failure)
        pull.create_dataset("text", data=b"Pull")
