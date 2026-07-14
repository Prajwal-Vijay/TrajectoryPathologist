"""Adapter for RoboFAC video + ManiSkill trajectory files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TaskContract,
    TrajectoryBundle,
)


@dataclass(frozen=True, slots=True)
class RoboFACEpisodeMatch:
    """Resolved mapping between a RoboFAC video id and HDF5 trajectory."""

    video_id: str
    episode_id: int
    unique_id: str
    traj_key: str
    robot_state_path: str
    elapsed_steps: int | None
    action_count: int
    state_count: int
    frame_count: int
    fps: float
    validation_notes: list[str]


class RoboFACBundleAdapter:
    """Load one RoboFAC video and matching robot telemetry into TrajectoryBundle.

    The adapter intentionally exposes only robot-side signals. Object actor
    states, rewards, success flags, and termination labels stay out of the
    bundle to avoid leaking simulator oracle information into metrics or QA.
    """

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

    def load(
        self,
        video_path: str | Path,
        h5_path: str | Path,
        traj_json_path: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        video_id: str | None = None,
    ) -> TrajectoryBundle:
        """Load a RoboFAC sample from video, HDF5 trajectory, and metadata JSON."""

        resolved_video_path = Path(video_path).expanduser().resolve()
        resolved_h5_path = Path(h5_path).expanduser().resolve()
        resolved_traj_json_path = Path(traj_json_path).expanduser().resolve()
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"Could not find RoboFAC video: {resolved_video_path}")
        if not resolved_h5_path.exists():
            raise FileNotFoundError(f"Could not find RoboFAC HDF5 trajectory file: {resolved_h5_path}")
        if not resolved_traj_json_path.exists():
            raise FileNotFoundError(f"Could not find RoboFAC trajectory metadata JSON: {resolved_traj_json_path}")

        resolved_video_id = video_id or resolved_video_path.stem
        trajectory_metadata = _load_json_object(resolved_traj_json_path)
        video_metadata = _video_metadata(resolved_video_path)
        episode_match = resolve_episode_match(
            video_id=resolved_video_id,
            trajectory_metadata=trajectory_metadata,
            h5_path=resolved_h5_path,
            frame_count=video_metadata["frame_count"],
            fps=video_metadata["fps"],
        )

        raw_episode = _read_robofac_episode(
            resolved_h5_path,
            traj_key=episode_match.traj_key,
        )
        records = _build_robot_records(
            actions=raw_episode["actions"],
            panda_states=raw_episode["panda_states"],
            fps=episode_match.fps,
        )
        timestamps = [record["timestamp_sec"] for record in records]
        frame_cache = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else resolved_video_path.parent / "pathologist_frame_cache" / resolved_video_path.stem
        )
        camera_frames = materialize_video_frames(
            video_path=resolved_video_path,
            frame_cache_dir=frame_cache,
            fps=episode_match.fps,
        )

        streams = _build_streams(
            records=records,
            timestamps=timestamps,
            video_path=resolved_video_path,
            h5_path=resolved_h5_path,
            traj_json_path=resolved_traj_json_path,
            episode_match=episode_match,
        )
        duration_sec = episode_match.frame_count / episode_match.fps if episode_match.fps > 0 else None
        env_info = trajectory_metadata.get("env_info") if isinstance(trajectory_metadata.get("env_info"), dict) else {}
        raw_metadata = {
            "dataset": "RoboFAC",
            "video_id": resolved_video_id,
            "video_path": str(resolved_video_path),
            "h5_path": str(resolved_h5_path),
            "traj_json_path": str(resolved_traj_json_path),
            "task_name": _task_name_from_env_info(env_info),
            "episode_id": episode_match.episode_id,
            "unique_id": episode_match.unique_id,
            "traj_key": episode_match.traj_key,
            "robot_state_path": episode_match.robot_state_path,
            "frame_count": episode_match.frame_count,
            "fps": episode_match.fps,
            "action_count": episode_match.action_count,
            "state_count": episode_match.state_count,
            "duration_sec": duration_sec,
            "frame_cache_dir": str(frame_cache),
            "alignment_validation_notes": episode_match.validation_notes,
            "oracle_fields_excluded": [
                "env_states/actors",
                "rewards",
                "success",
                "terminated",
                "truncated",
            ],
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_video_id,
            source_format=BundleFormat.ROBOFAC_HDF5_VIDEO,
            root_path=resolved_video_path,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=camera_frames,
            raw_metadata=raw_metadata,
        )

    def load_video_only(
        self,
        video_path: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        video_id: str | None = None,
        task_name: str | None = None,
    ) -> TrajectoryBundle:
        """Load a RoboFAC sample when only the video demonstration is available."""

        resolved_video_path = Path(video_path).expanduser().resolve()
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"Could not find RoboFAC video: {resolved_video_path}")

        resolved_video_id = video_id or resolved_video_path.stem
        video_metadata = _video_metadata(resolved_video_path)
        frame_cache = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else resolved_video_path.parent / "pathologist_frame_cache" / resolved_video_path.stem
        )
        camera_frames = materialize_video_frames(
            video_path=resolved_video_path,
            frame_cache_dir=frame_cache,
            fps=video_metadata["fps"],
        )
        duration_sec = (
            video_metadata["frame_count"] / video_metadata["fps"]
            if video_metadata["fps"] > 0
            else None
        )
        raw_metadata = {
            "dataset": "RoboFAC",
            "video_id": resolved_video_id,
            "video_path": str(resolved_video_path),
            "task_name": task_name,
            "frame_count": video_metadata["frame_count"],
            "fps": video_metadata["fps"],
            "duration_sec": duration_sec,
            "frame_cache_dir": str(frame_cache),
            "video_only": True,
            "alignment_validation_notes": ["trajectory telemetry unavailable; loaded video frames only"],
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_video_id,
            source_format=BundleFormat.CUSTOM,
            root_path=resolved_video_path,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=[],
            camera_frames=camera_frames,
            raw_metadata=raw_metadata,
        )


def resolve_episode_match(
    video_id: str,
    trajectory_metadata: dict[str, Any],
    h5_path: str | Path,
    frame_count: int,
    fps: float,
) -> RoboFACEpisodeMatch:
    """Resolve a RoboFAC video id to an HDF5 trajectory using episode unique_id."""

    episodes = trajectory_metadata.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("RoboFAC trajectory metadata must contain an episodes list.")
    matched_episode = None
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        if str(episode.get("unique_id")) == video_id:
            matched_episode = episode
            break
    if matched_episode is None:
        raise ValueError(f"Could not map RoboFAC video id to episode unique_id: {video_id}")
    episode_id = int(matched_episode["episode_id"])
    traj_key = f"traj_{episode_id}"
    action_count, state_count, robot_state_path = _h5_episode_counts(Path(h5_path), traj_key)
    elapsed_steps = matched_episode.get("elapsed_steps")
    elapsed_steps = int(elapsed_steps) if isinstance(elapsed_steps, (int, float)) else None
    notes = []
    if elapsed_steps is not None and elapsed_steps != action_count:
        notes.append(f"elapsed_steps={elapsed_steps} differs from action_count={action_count}")
    if abs(frame_count - action_count) > 1:
        notes.append(f"frame_count={frame_count} differs from action_count={action_count}")
    if state_count not in {action_count, action_count + 1}:
        notes.append(f"state_count={state_count} is not action_count or action_count+1")
    if fps <= 0:
        notes.append("video fps was unavailable or non-positive")
    return RoboFACEpisodeMatch(
        video_id=video_id,
        episode_id=episode_id,
        unique_id=str(matched_episode["unique_id"]),
        traj_key=traj_key,
        robot_state_path=robot_state_path,
        elapsed_steps=elapsed_steps,
        action_count=action_count,
        state_count=state_count,
        frame_count=frame_count,
        fps=fps,
        validation_notes=notes,
    )


def materialize_video_frames(
    video_path: Path,
    frame_cache_dir: Path,
    fps: float,
) -> list[CameraFrameRef]:
    """Extract all video frames to a cache and return timestamped references."""

    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _frames_from_image_dir(frame_cache_dir, fps=fps)
    metadata = _video_metadata(video_path)
    if len(existing) >= metadata["frame_count"]:
        return existing[: metadata["frame_count"]]
    _clear_cached_images(frame_cache_dir)
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RoboFAC video: {video_path}")
    refs: list[CameraFrameRef] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = index / fps if fps > 0 else 0.0
        image_path = frame_cache_dir / f"{index:06d}_t{timestamp:08.3f}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            capture.release()
            raise RuntimeError(f"Could not write extracted frame: {image_path}")
        refs.append(
            CameraFrameRef(
                camera_name="rgb",
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=index,
            )
        )
        index += 1
    capture.release()
    return refs


def _build_streams(
    records: list[dict[str, Any]],
    timestamps: list[float],
    video_path: Path,
    h5_path: Path,
    traj_json_path: Path,
    episode_match: RoboFACEpisodeMatch,
) -> list[ModalityStream]:
    base_metadata = {
        "dataset": "RoboFAC",
        "video_path": str(video_path),
        "h5_path": str(h5_path),
        "traj_json_path": str(traj_json_path),
        "video_id": episode_match.video_id,
        "episode_id": episode_match.episode_id,
        "traj_key": episode_match.traj_key,
        "fps": episode_match.fps,
        "robot_state_path": episode_match.robot_state_path,
        "object_and_oracle_state_excluded": True,
    }
    return [
        ModalityStream(
            name="telemetry",
            modality=Modality.TELEMETRY,
            timestamps_sec=timestamps,
            records=records,
            metadata=base_metadata,
        ),
        ModalityStream(
            name="action",
            modality=Modality.ACTION,
            timestamps_sec=timestamps,
            records=records,
            metadata={
                **base_metadata,
                "raw_action_width": 8,
                "metric_action_width": 7,
                "source_gripper_indices_removed": [7],
                "vector_representation": "actions_0_to_6_arm_only",
            },
        ),
        ModalityStream(
            name="proprioception",
            modality=Modality.PROPRIOCEPTION,
            timestamps_sec=timestamps,
            records=records,
            metadata={
                **base_metadata,
                "raw_robot_state_width": 31,
                "metric_proprioception_width": 7,
                "source_robot_state_path": episode_match.robot_state_path,
                "source_qpos_slice": [13, 20],
                "source_gripper_qpos_slice": [20, 22],
                "vector_representation": "robot_articulation_qpos_13_to_19_arm_only",
            },
        ),
        ModalityStream(
            name="velocity",
            modality=Modality.PROPRIOCEPTION,
            timestamps_sec=timestamps,
            records=records,
            metadata={
                **base_metadata,
                "raw_robot_state_width": 31,
                "metric_velocity_width": 7,
                "source_robot_state_path": episode_match.robot_state_path,
                "source_qvel_slice": [22, 29],
                "source_gripper_qvel_slice": [29, 31],
                "vector_representation": "robot_articulation_qvel_22_to_28_arm_only",
            },
        ),
        ModalityStream(
            name="gripper",
            modality=Modality.GRIPPER,
            timestamps_sec=timestamps,
            records=records,
            metadata={
                **base_metadata,
                "source_gripper_action_index": 7,
                "source_gripper_qpos_slice": [20, 22],
                "source_gripper_qvel_slice": [29, 31],
                "low_state_label": "closed",
                "high_state_label": "open",
            },
        ),
    ]


def _build_robot_records(actions: list[list[float]], panda_states: list[list[float]], fps: float) -> list[dict[str, Any]]:
    sample_count = len(actions)
    if sample_count <= 0:
        raise ValueError("RoboFAC trajectory contains no actions.")
    if len(panda_states) < sample_count:
        raise ValueError("RoboFAC robot articulation state count must be at least action count.")
    records = []
    for index in range(sample_count):
        action = actions[index]
        state = panda_states[index]
        if len(action) < 8:
            raise ValueError(f"RoboFAC action at index {index} has width {len(action)}, expected at least 8.")
        if len(state) < 31:
            raise ValueError(
                f"RoboFAC robot articulation state at index {index} has width {len(state)}, expected at least 31."
            )
        timestamp = index / fps if fps > 0 else float(index)
        arm_action = action[:7]
        gripper_action = [action[7]]
        arm_qpos = state[13:20]
        gripper_qpos = state[20:22]
        arm_qvel = state[22:29]
        gripper_qvel = state[29:31]
        records.append(
            {
                "timestamp_sec": timestamp,
                "sample_index": index,
                "action": arm_action,
                "raw_action": action,
                "gripper_action": gripper_action,
                "proprioception": arm_qpos,
                "joint_positions": arm_qpos,
                "state": arm_qpos,
                "qvel": arm_qvel,
                "velocity": arm_qvel,
                "joint_velocity": arm_qvel,
                "gripper": gripper_qpos,
                "gripper_state": gripper_qpos,
                "gripper_velocity": gripper_qvel,
            }
        )
    return records


def _read_robofac_episode(h5_path: Path, traj_key: str) -> dict[str, list[list[float]]]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RoboFACBundleAdapter requires h5py to read .h5 files.") from exc

    with h5py.File(h5_path, "r") as handle:
        if traj_key not in handle:
            raise ValueError(f"Could not find {traj_key} in RoboFAC HDF5 file: {h5_path}")
        group = handle[traj_key]
        if "actions" not in group:
            raise ValueError(f"RoboFAC {traj_key} is missing actions.")
        robot_state_path = _robot_state_dataset_path(group)
        return {
            "actions": _dataset_to_float_rows(group["actions"]),
            "panda_states": _dataset_to_float_rows(group[robot_state_path]),
        }


def _h5_episode_counts(h5_path: Path, traj_key: str) -> tuple[int, int, str]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RoboFACBundleAdapter requires h5py to read .h5 files.") from exc

    with h5py.File(h5_path, "r") as handle:
        if traj_key not in handle:
            raise ValueError(f"Could not find {traj_key} in RoboFAC HDF5 file: {h5_path}")
        group = handle[traj_key]
        action_count = int(group["actions"].shape[0])
        robot_state_path = _robot_state_dataset_path(group)
        state_count = int(group[robot_state_path].shape[0])
        return action_count, state_count, robot_state_path


def _robot_state_dataset_path(traj_group: Any) -> str:
    articulations_path = "env_states/articulations"
    if articulations_path not in traj_group:
        raise ValueError(f"RoboFAC trajectory is missing {articulations_path}.")
    articulations = traj_group[articulations_path]
    candidates: list[tuple[int, str]] = []
    for name, dataset in articulations.items():
        shape = getattr(dataset, "shape", None)
        if not shape or len(shape) != 2 or int(shape[1]) < 31:
            continue
        lowered = str(name).lower()
        if lowered == "panda_wristcam":
            priority = 0
        elif "panda" in lowered:
            priority = 1
        elif "wrist" in lowered:
            priority = 2
        else:
            priority = 3
        candidates.append((priority, f"{articulations_path}/{name}"))
    if not candidates:
        available = ", ".join(str(name) for name in articulations.keys())
        raise ValueError(
            "RoboFAC trajectory has no usable robot articulation state dataset "
            f"under {articulations_path}. Available: {available or '<none>'}"
        )
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][1]


def _dataset_to_float_rows(dataset: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in dataset[:].tolist()]


def _video_metadata(video_path: Path) -> dict[str, Any]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RoboFAC video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if frame_count <= 0:
        raise ValueError(f"Could not read frame count from RoboFAC video: {video_path}")
    if fps <= 0:
        fps = 30.0
    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _frames_from_image_dir(frame_dir: Path, fps: float) -> list[CameraFrameRef]:
    image_paths = sorted(
        [
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in RoboFACBundleAdapter.IMAGE_SUFFIXES
        ]
    ) if frame_dir.exists() else []
    return [
        CameraFrameRef(
            camera_name="rgb",
            timestamp_sec=index / fps if fps > 0 else float(index),
            image_path=image_path,
            frame_index=index,
        )
        for index, image_path in enumerate(image_paths)
    ]


def _clear_cached_images(frame_cache_dir: Path) -> None:
    for path in frame_cache_dir.iterdir():
        if path.is_file() and path.suffix.lower() in RoboFACBundleAdapter.IMAGE_SUFFIXES:
            path.unlink()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RoboFAC JSON must contain an object: {path}")
    return payload


def _task_name_from_env_info(env_info: dict[str, Any]) -> str | None:
    env_id = env_info.get("env_id")
    if isinstance(env_id, str):
        return env_id.split("-", 1)[0]
    return None


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboFAC video loading requires opencv-python/cv2. "
            "Install opencv-python or use an environment that already provides cv2."
        ) from exc
    return cv2
