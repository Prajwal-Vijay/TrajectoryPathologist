"""Tests for ViFailBack episode ingestion helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pathologist.adapters.vifailback_bundle import (
    ViFailBackBundleAdapter,
    _build_records,
    _build_streams,
    _eef_velocity_records,
    _frames_from_image_dir,
    _gripper_normalization,
    _load_episode_annotation,
    _resolve_gripper_indices,
    _uniform_timestamps,
    _validate_native_sample_rate,
)


def test_vifailback_records_split_gripper_dims_from_action_and_qpos() -> None:
    episode = {
        "action": [
            [float(index) for index in range(14)],
            [float(index + 1) for index in range(14)],
        ],
        "observations/qpos": [
            [float(index) / 10.0 for index in range(14)],
            [float(index + 1) / 10.0 for index in range(14)],
        ],
        "observations/qvel": [
            [float(index) / 100.0 for index in range(14)],
            [float(index + 1) / 100.0 for index in range(14)],
        ],
        "observations/effort": [
            [float(index) / 1000.0 for index in range(14)],
            [float(index + 1) / 1000.0 for index in range(14)],
        ],
    }
    gripper_indices = _resolve_gripper_indices(
        gripper_indices=None,
        action=episode["action"],
        qpos=episode["observations/qpos"],
        default_by_width=ViFailBackBundleAdapter.DEFAULT_GRIPPER_INDICES_BY_WIDTH,
    )

    gripper_normalization = _gripper_normalization(episode, gripper_indices)
    records = _build_records(episode, [0.0, 0.1], gripper_indices, gripper_normalization)
    streams = _build_streams(
        records,
        [0.0, 0.1],
        Path("/tmp/episode_0.hdf5"),
        gripper_indices,
        gripper_normalization,
    )

    assert gripper_indices == [6, 13]
    assert len(records[0]["action"]) == 12
    assert records[0]["action"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert len(records[0]["proprioception"]) == 12
    assert records[0]["raw_gripper"] == [0.6, 1.3]
    assert records[0]["gripper"] == [0.0, 0.0]
    assert records[1]["gripper"] == [1.0, 1.0]
    gripper_stream = next(stream for stream in streams if stream.name == "gripper")
    assert gripper_stream.metadata["gripper_value_representation"] == "normalized_0_to_1"
    assert {stream.name for stream in streams} == {
        "telemetry",
        "action",
        "proprioception",
        "velocity",
        "effort",
        "gripper",
    }


def test_vifailback_adapter_accepts_preextracted_frame_directory(tmp_path: Path) -> None:
    frame_dir = tmp_path / "annotated_frames"
    frame_dir.mkdir(parents=True)
    for index in range(3):
        (frame_dir / f"frame_{index:06d}.jpg").write_bytes(b"jpeg")

    frames = ViFailBackBundleAdapter()._load_video_frames(
        video_path=frame_dir,
        frame_cache_dir=tmp_path / "unused_cache",
        video_sample_fps=2.0,
    )

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_sec for frame in frames] == [0.0, 0.5, 1.0]
    assert all(frame.camera_name == "annotated_frames" for frame in frames)


def test_vifailback_adapter_writes_embedded_hdf5_rgb_frames(tmp_path: Path) -> None:
    episode = {
        "observations/images/cam_high": [
            b"\xff\xd8\xfffirst",
            b"\xff\xd8\xffsecond",
        ],
        "observations/images/cam_left_wrist": [
            b"\xff\xd8\xffleft_first",
        ],
    }

    frames = ViFailBackBundleAdapter()._write_hdf5_rgb_frames(
        episode=episode,
        frame_cache_dir=tmp_path / "hdf5_rgb",
        hdf5_fps=25.0,
    )

    assert len(frames) == 3
    high_frames = [frame for frame in frames if frame.camera_name == "cam_high"]
    assert [frame.timestamp_sec for frame in high_frames] == [0.0, 0.04]
    assert high_frames[0].image_path.exists()
    assert high_frames[0].image_path.suffix == ".jpg"
    assert high_frames[0].image_path.read_bytes() == b"\xff\xd8\xfffirst"


def test_vifailback_adapter_reuses_existing_hdf5_rgb_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hdf5_rgb"
    camera_dir = cache_dir / "cam_high"
    camera_dir.mkdir(parents=True)
    cached_frame = camera_dir / "000000_t0000.000.jpg"
    cached_frame.write_bytes(b"cached")
    episode = {
        "observations/images/cam_high": [
            b"\xff\xd8\xffnew",
        ],
    }

    frames = ViFailBackBundleAdapter()._write_hdf5_rgb_frames(
        episode=episode,
        frame_cache_dir=cache_dir,
        hdf5_fps=25.0,
    )

    assert len(frames) == 1
    assert frames[0].image_path == cached_frame
    assert cached_frame.read_bytes() == b"cached"


def test_vifailback_native_timestamps_are_exactly_25_hz() -> None:
    timestamps = _uniform_timestamps(4, 25.0)

    assert timestamps == [0.0, 0.04, 0.08, 0.12]
    _validate_native_sample_rate(25.0)
    with pytest.raises(ValueError, match="exactly 25 Hz"):
        _validate_native_sample_rate(50.0)


def test_vifailback_eef_velocity_uses_bimanual_target_poses() -> None:
    identity = [0.0, 0.0, 0.0, 1.0]
    quarter_turn_z = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    poses = [
        [0.0, 0.0, 0.0, *identity, 0.0, 1.0, 0.0, 0.0, *identity, 0.0],
        [0.04, 0.0, 0.0, *quarter_turn_z, 0.0, 1.0, 0.08, 0.0, *identity, 0.0],
    ]

    records = _eef_velocity_records(poses, [0.0, 0.04])

    assert len(records) == 1
    assert records[0]["sample_index"] == 1
    assert records[0]["eef_linear_velocity"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 2.0, 0.0]
    )
    assert records[0]["eef_angular_velocity"] == pytest.approx(
        [0.0, 0.0, (math.pi / 2.0) / 0.04, 0.0, 0.0, 0.0]
    )


def test_cached_frame_indices_restart_for_each_camera(tmp_path: Path) -> None:
    cache_dir = tmp_path / "hdf5_rgb"
    for camera_name in ("cam_high", "cam_left_wrist"):
        camera_dir = cache_dir / camera_name
        camera_dir.mkdir(parents=True)
        for index in range(2):
            (camera_dir / f"{index:06d}_t{index / 25.0:08.3f}.jpg").write_bytes(b"jpeg")

    frames = _frames_from_image_dir(cache_dir, 25.0)

    for camera_name in ("cam_high", "cam_left_wrist"):
        camera_frames = [frame for frame in frames if frame.camera_name == camera_name]
        assert [frame.frame_index for frame in camera_frames] == [0, 1]
        assert [frame.timestamp_sec for frame in camera_frames] == [0.0, 0.04]


def test_vifailback_annotation_preserves_exact_source_frame(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ViFailback-Dataset"
    h5_path = dataset_root / "raw_data" / "example_task" / "episode_7.hdf5"
    h5_path.parent.mkdir(parents=True)
    annotation_dir = dataset_root / "annotated_data" / "example_task"
    annotation_dir.mkdir(parents=True)
    annotation_path = annotation_dir / "example_task_annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "annotated_data/example_task/videos/episode_7_cam_high.mp4",
                    "task": "Example task",
                    "failure_detection": "yes",
                    "failure_type": "gripper 6d-pose",
                    "keyframe": [
                        {
                            "avoid_keyframe": (
                                "annotated_data/example_task/frames/"
                                "episode_7_cam_high/172_avoid_keyframe.jpg"
                            ),
                            "correct_keyframe": (
                                "annotated_data/example_task/frames/"
                                "episode_7_cam_high/269_correct_keyframe.jpg"
                            ),
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = _load_episode_annotation(h5_path, 25.0)

    assert metadata["annotation_available"] is True
    assert metadata["avoid_keyframe_indices"] == [172]
    assert metadata["avoid_keyframe_timestamps_sec"] == pytest.approx([6.88])
    assert metadata["correct_keyframe_indices"] == [269]
    assert metadata["correct_keyframe_timestamps_sec"] == pytest.approx([10.76])
