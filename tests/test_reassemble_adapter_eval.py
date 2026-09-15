"""Tests for REASSEMBLE ingestion and deterministic recall evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from pathologist.adapters.reassemble_bundle import ReassembleBundleAdapter
from pathologist.reassemble.evaluation import (
    _configured_metric_ranking,
    _empty_stats_bucket,
    _evaluation_row,
    _failed_high_level_demo_bundles,
    _finalize_stats,
    _high_level_task_kind,
    _prepare_reassemble_ronar_calibration,
    _set_metric_ranking,
    _selector_evaluation_row,
    _set_visual_flow_camera_names,
    _update_stats,
)
from pathologist.run_reassemble_eval_pipeline import build_arg_parser
from pathologist.schemas import BundleFormat, EventProposalSource, TimeWindow
from pathologist.schemas.key_event import KeyEventProposal


def test_reassemble_cli_accepts_persistence_metric_ranking() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data-root",
            "REASSEMBLE",
            "--metric-ranking",
            "persistence",
            "--frame-cache-dir",
            "shared_frames",
        ]
    )

    assert args.metric_ranking == "persistence"
    assert args.frame_cache_dir == "shared_frames"


def test_reassemble_cli_accepts_both_ronar_protocols() -> None:
    args = build_arg_parser().parse_args(
        [
            "--data-root",
            "REASSEMBLE",
            "--selectors",
            "ronar",
            "ronar_k",
            "--ronar-sample-period-sec",
            "0.25",
            "--ronar-threshold",
            "60",
            "--ronar-calibration-cache",
            "ronar.json",
        ]
    )

    assert args.selectors == ["ronar", "ronar_k"]
    assert args.ronar_sample_period_sec == pytest.approx(0.25)
    assert args.ronar_threshold == pytest.approx(60.0)
    assert args.ronar_calibration_cache == "ronar.json"


def test_reassemble_metric_ranking_updates_selector_config() -> None:
    config = {"selector": {"surprise_flood": {"ranking": "area"}}}

    _set_metric_ranking(config, "persistence")

    assert _configured_metric_ranking(config) == "persistence"


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
        "eef_velocity",
        "gripper",
        "compensated_base_force",
        "telemetry",
        "hama1_audio",
    }
    force_stream = next(stream for stream in bundle.streams if stream.name == "compensated_base_force")
    assert force_stream.metadata["coordinate_frame"] == "robot_base"
    assert force_stream.records is not None
    assert force_stream.records[1]["compensated_base_force"] == [0.0, 3.0, 4.0]
    audio_stream = next(stream for stream in bundle.streams if stream.name == "hama1_audio")
    assert audio_stream.data_path is not None
    assert audio_stream.data_path.read_bytes() == b"\xff\xf3\xc8\x64"
    eef_velocity_stream = next(
        stream for stream in bundle.streams if stream.name == "eef_velocity"
    )
    assert eef_velocity_stream.records is not None
    assert eef_velocity_stream.records[0]["eef_linear_velocity"] == [1.0, 2.0, 3.0]
    assert eef_velocity_stream.records[0]["eef_angular_velocity"] == [4.0, 5.0, 6.0]


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
    assert _high_level_task_kind(demo) == "remove"
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


def test_reassemble_stats_only_report_low_level_and_task_breakdown(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)
    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)
    demo = _failed_high_level_demo_bundles(bundle)[0]
    row = _evaluation_row(
        demo,
        [_proposal("demo_surprise_flood_001", peak_sec=1.2)],
        top_k=1,
        requested_segment_level="both",
    )
    stats_accumulator = defaultdict(lambda: defaultdict(_empty_stats_bucket))
    task_stats_accumulator = defaultdict(lambda: defaultdict(_empty_stats_bucket))

    _update_stats(stats_accumulator, task_stats_accumulator, row)
    stats = _finalize_stats(stats_accumulator, task_stats_accumulator)

    assert set(stats["metric"]) == {"low", "by_high_level_task"}
    assert "high" not in stats["metric"]
    assert "combined" not in stats["metric"]
    assert stats["metric"]["low"]["trajectory_count"] == 1
    assert stats["metric"]["by_high_level_task"]["remove"]["trajectory_count"] == 1
    assert stats["metric"]["low"]["mean_selected_events_per_trajectory"] == 1.0


def test_reassemble_ronar_k_scores_exact_budget_and_records_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "episode.h5"
    _write_reassemble_fixture(h5py, h5_path, include_failure=True)
    bundle = ReassembleBundleAdapter().load(h5_path, extract_frames=False)
    demo = _failed_high_level_demo_bundles(bundle)[0]

    class FakeEvent:
        def __init__(self, timestamp: float) -> None:
            self.timestamp = timestamp

        def to_dict(self, rank: int) -> dict[str, object]:
            return {
                "rank": rank,
                "peak_sec": self.timestamp,
                "selection_source": "ronar_k_crossing",
            }

    protocol = {
        "protocol": "per_trajectory_exact_k",
        "threshold": 3.5,
        "genuine_selected_event_count": 2,
        "uniform_padded_event_count": 0,
    }
    monkeypatch.setattr(
        "pathologist.reassemble.evaluation.select_ronar_keyevents",
        lambda *args, **kwargs: SimpleNamespace(
            events=[FakeEvent(0.25), FakeEvent(1.25)],
            metadata=protocol,
        ),
    )

    row = _selector_evaluation_row(
        selector_name="ronar_k",
        bundle=demo,
        proposals=[],
        top_k=2,
        requested_segment_level="low",
        config={},
        ronar_statistics_by_task={
            "remove gear": {
                "arm_motion": {"mean": 0.0, "std": 1.0, "count": 2}
            }
        },
    )

    assert len(row["selected_peaks"]) == 2
    assert row["recall_by_level"]["low"]["selected_peak_hit_count"] == 1
    assert row["selection_protocol"] == protocol


def test_reassemble_ronar_calibration_reuses_compatible_cache(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "episode.h5"
    h5_path.write_bytes(b"fixture")
    cache_path = tmp_path / "ronar_calibration.json"
    settings = {
        "version": 1,
        "h5_paths": [
            {
                "path": str(h5_path.resolve()),
                "size": h5_path.stat().st_size,
                "mtime_ns": h5_path.stat().st_mtime_ns,
            }
        ],
        "camera_names": ["hand"],
        "include_event_camera": False,
        "robot_downsample_hz": 50.0,
        "sample_period_sec": 0.2,
        "resize": [128, 128],
        "calibration_cohort": "successful_high_level_demonstrations",
    }
    statistics = {
        "remove gear": {
            "arm_motion": {"mean": 0.5, "std": 0.25, "count": 10}
        }
    }
    cache_path.write_text(
        json.dumps(
            {
                "settings": settings,
                "statistics_by_task": statistics,
                "summary": {"calibration_demo_count": 3},
            }
        ),
        encoding="utf-8",
    )

    result = _prepare_reassemble_ronar_calibration(
        h5_paths=[h5_path],
        adapter=SimpleNamespace(),
        frame_cache_dir=tmp_path / "frames",
        camera_names=["hand"],
        include_event_camera=False,
        robot_downsample_hz=50.0,
        sample_period_sec=0.2,
        cache_path=cache_path,
        verbose=False,
    )

    assert result["statistics_by_task"] == statistics
    assert result["summary"]["status"] == "loaded_from_cache"


def test_reassemble_camera_choice_updates_visual_flow_config() -> None:
    config = {"metrics": {"visual_flow": {"camera_names": ["hand"], "resize": [128, 128]}}}

    _set_visual_flow_camera_names(config, ["hama1"])

    assert config["metrics"]["visual_flow"]["camera_names"] == ["hama1"]
    assert config["metrics"]["visual_flow"]["resize"] == [128, 128]


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
        handle.create_dataset(
            "hama1_audio",
            data=np.array([255, 243, 200, 100], dtype=np.int64),
        )
        robot_state = handle.create_group("robot_state")
        timestamps = handle.create_group("timestamps")
        base_times = np.array([100.0, 100.1, 100.2, 100.3, 100.4], dtype=float)
        robot_state.create_dataset("joint_positions", data=np.stack([np.arange(7) + i for i in range(5)]))
        robot_state.create_dataset("joint_velocities", data=np.ones((5, 7)))
        robot_state.create_dataset(
            "velocity",
            data=np.tile(np.arange(1.0, 7.0), (5, 1)),
        )
        robot_state.create_dataset("gripper_positions", data=np.zeros((5, 2)))
        robot_state.create_dataset("pose", data=np.stack([[i, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for i in range(5)]))
        robot_state.create_dataset(
            "compensated_base_force",
            data=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [3.0, 4.0, 0.0],
                    [0.0, 3.0, 4.0],
                    [1.0, 2.0, 2.0],
                    [0.0, 0.0, 6.0],
                ]
            ),
        )
        robot_state.create_dataset("measured_force", data=np.zeros((5, 3)))
        robot_state.create_dataset("measured_torque", data=np.zeros((5, 3)))
        for name in (
            "joint_positions",
            "joint_velocities",
            "velocity",
            "gripper_positions",
            "pose",
            "compensated_base_force",
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
