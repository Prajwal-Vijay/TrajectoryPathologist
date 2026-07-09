"""Tests for RoboFAC trajectory adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathologist.adapters.robofac_bundle import RoboFACBundleAdapter, resolve_episode_match


def test_resolve_episode_match_uses_unique_id(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "traj.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("traj_15")
        group.create_dataset("actions", data=[[0.0] * 8 for _ in range(3)])
        group.create_dataset("env_states/articulations/panda_wristcam", data=[[0.0] * 31 for _ in range(4)])
    metadata = {
        "episodes": [
            {"episode_id": 2, "unique_id": "other", "elapsed_steps": 9},
            {"episode_id": 15, "unique_id": "feefd58b-6145-4a96-93ad-9fe01c596b38", "elapsed_steps": 3},
        ]
    }

    match = resolve_episode_match(
        video_id="feefd58b-6145-4a96-93ad-9fe01c596b38",
        trajectory_metadata=metadata,
        h5_path=h5_path,
        frame_count=3,
        fps=30.0,
    )

    assert match.episode_id == 15
    assert match.traj_key == "traj_15"
    assert match.action_count == 3
    assert match.state_count == 4
    assert match.validation_notes == []


def test_robofac_adapter_exposes_robot_only_streams(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    h5py = pytest.importorskip("h5py")
    video_path = tmp_path / "video-uuid.mp4"
    h5_path = tmp_path / "traj.h5"
    json_path = tmp_path / "traj.json"
    _write_test_video(cv2, video_path, frame_count=4)
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("traj_1")
        group.create_dataset("actions", data=[[float(i)] * 8 for i in range(4)])
        panda_rows = []
        for i in range(5):
            row = [0.0] * 31
            row[13:20] = [float(i + dim) for dim in range(7)]
            row[20:22] = [0.0, 0.04]
            row[22:29] = [float(dim) for dim in range(7)]
            row[29:31] = [0.1, 0.2]
            panda_rows.append(row)
        group.create_dataset("env_states/articulations/panda_wristcam", data=panda_rows)
        group.create_dataset("rewards", data=[1.0, 1.0, 1.0, 1.0])
        group.create_dataset("success", data=[False, False, False, False])
        actor_group = group.create_group("env_states/actors")
        actor_group.create_dataset("cup", data=[[0.0] * 13 for _ in range(5)])
    json_path.write_text(
        json.dumps(
            {
                "env_info": {"env_id": "MicrowaveTask-mug"},
                "episodes": [
                    {
                        "episode_id": 1,
                        "unique_id": "video-uuid",
                        "elapsed_steps": 4,
                        "success": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = RoboFACBundleAdapter().load(
        video_path=video_path,
        h5_path=h5_path,
        traj_json_path=json_path,
        frame_cache_dir=tmp_path / "frames",
    )

    assert bundle.bundle_id == "video-uuid"
    assert len(bundle.camera_frames) == 4
    assert {stream.name for stream in bundle.streams} == {
        "telemetry",
        "action",
        "proprioception",
        "velocity",
        "gripper",
    }
    action_stream = next(stream for stream in bundle.streams if stream.name == "action")
    assert action_stream.records is not None
    assert len(action_stream.records[0]["action"]) == 7
    assert len(action_stream.records[0]["gripper"]) == 2
    assert "success" not in bundle.raw_metadata
    assert "rewards" not in bundle.raw_metadata
    assert bundle.raw_metadata["oracle_fields_excluded"]


def _write_test_video(cv2: object, path: Path, frame_count: int) -> None:
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("Could not create test video.")
    for index in range(frame_count):
        frame = np.full((32, 32, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

