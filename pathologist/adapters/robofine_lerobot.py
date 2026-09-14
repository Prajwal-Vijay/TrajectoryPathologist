"""Shared LeRobot parquet + video loader for RoboFine-Bench telemetry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pathologist.adapters.robofac_bundle import _clear_cached_images, _import_cv2, _video_metadata
from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TaskContract,
    TrajectoryBundle,
)

OrientationKind = Literal["quaternion", "euler"]
BENCH_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "RoboFine-Bench"
EVAL_DATA_DEFAULT = BENCH_DIR_DEFAULT / "EvalData"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SURPRISE_SOURCE_METRICS = (
    "action_discontinuity",
    "motion_smoothness",
    "velocity_spike",
    "stalling",
    "visual_flow",
)


@dataclass(frozen=True, slots=True)
class RoboFineDatasetSpec:
    """Per-dataset layout of the shared LeRobot 8-d state / 7-d action vectors."""

    dataset: str
    fps: float
    orientation: OrientationKind
    default_view: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


class RoboFineLeRobotAdapter:
    """Load one RoboFine-Bench video and its matching LeRobot parquet episode."""

    spec: RoboFineDatasetSpec

    def load(
        self,
        *,
        sample_id: str,
        video_path: str | Path,
        parquet_path: str | Path,
        camera_name: str,
        frame_cache_dir: str | Path,
        manifest_entry: dict[str, Any],
        verification_entry: dict[str, Any] | None = None,
        eval_sample: dict[str, Any] | None = None,
        task_contract: TaskContract | None = None,
    ) -> TrajectoryBundle:
        """Load a verified RoboFine sample. Frame i is parquet row i."""

        resolved_video = Path(video_path).expanduser().resolve()
        resolved_parquet = Path(parquet_path).expanduser().resolve()
        if not resolved_video.exists():
            raise FileNotFoundError(f"Could not find RoboFine video: {resolved_video}")
        if not resolved_parquet.exists():
            raise FileNotFoundError(f"Could not find RoboFine telemetry parquet: {resolved_parquet}")
        _require_manifest_alignment(sample_id, manifest_entry)

        table = _read_parquet(resolved_parquet)
        timestamps = [float(value) for value in table["timestamp"]]
        states = [_as_float_list(row, 8, "observation.state") for row in table["observation.state"]]
        actions = [_as_float_list(row, 7, "action") for row in table["action"]]
        if not (len(timestamps) == len(states) == len(actions)):
            raise ValueError(
                f"{sample_id}: parquet timestamp/state/action lengths disagree "
                f"({len(timestamps)}, {len(states)}, {len(actions)})."
            )
        video_metadata = _video_metadata(resolved_video)
        frame_count = int(video_metadata["frame_count"])
        if frame_count != len(timestamps):
            raise ValueError(
                f"{sample_id}: video frame count {frame_count} does not match "
                f"parquet rows {len(timestamps)}."
            )
        manifest_rows = manifest_entry.get("rows")
        manifest_frames = manifest_entry.get("video_frames")
        if manifest_rows is not None and int(manifest_rows) != len(timestamps):
            raise ValueError(
                f"{sample_id}: manifest rows={manifest_rows} disagrees with parquet rows={len(timestamps)}."
            )
        if manifest_frames is not None and int(manifest_frames) != frame_count:
            raise ValueError(
                f"{sample_id}: manifest video_frames={manifest_frames} disagrees with video frames={frame_count}."
            )

        records = self._build_records(timestamps=timestamps, states=states, actions=actions)
        camera_frames = materialize_video_frames(
            video_path=resolved_video,
            frame_cache_dir=Path(frame_cache_dir).expanduser().resolve(),
            timestamps_sec=timestamps,
            camera_name=camera_name,
            expected_frame_count=frame_count,
        )
        if len(camera_frames) != len(timestamps):
            raise ValueError(
                f"{sample_id}: extracted {len(camera_frames)} frames, expected {len(timestamps)}."
            )
        for index, frame in enumerate(camera_frames):
            if frame.frame_index != index:
                raise ValueError(f"{sample_id}: extracted frames are not in index order.")
            frame.timestamp_sec = timestamps[index]

        duration_sec = _duration_sec(timestamps, self.spec.fps)
        streams = self._build_streams(
            records=records,
            timestamps=timestamps,
            sample_id=sample_id,
            video_path=resolved_video,
            parquet_path=resolved_parquet,
            camera_name=camera_name,
            manifest_entry=manifest_entry,
        )
        eval_sample = eval_sample or {}
        notes = [
            f"frame_index i maps to parquet row i ({len(timestamps)} steps)",
            f"timestamps taken from parquet (fps={self.spec.fps:g})",
            f"orientation={self.spec.orientation}",
        ]
        pixel_corr = None if verification_entry is None else verification_entry.get("pixel_corr")
        if pixel_corr is not None:
            notes.append(f"pixel_corr={pixel_corr}")
        return TrajectoryBundle(
            bundle_id=sample_id,
            source_format=BundleFormat.ROBOFINE_LEROBOT,
            root_path=resolved_video,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=camera_frames,
            raw_metadata={
                "dataset": self.spec.dataset,
                "sample_id": sample_id,
                "video_path": str(resolved_video),
                "telemetry_path": str(resolved_parquet),
                "camera_name": camera_name,
                "view_names": [camera_name],
                "fps": self.spec.fps,
                "frame_count": frame_count,
                "telemetry_rows": len(timestamps),
                "duration_sec": duration_sec,
                "orientation": self.spec.orientation,
                "state_names": list(self.spec.state_names),
                "action_names": list(self.spec.action_names),
                "episode_index": manifest_entry.get("episode_index"),
                "repo_id": manifest_entry.get("repo_id"),
                "instruction_raw": eval_sample.get("instruction_raw"),
                "robot_type": eval_sample.get("robot_type"),
                "pixel_corr": pixel_corr,
                "alignment_validation_notes": notes,
                "manifest_status": manifest_entry.get("status"),
                "length_matches_video": True,
            },
        )

    def _build_records(
        self,
        timestamps: list[float],
        states: list[list[float]],
        actions: list[list[float]],
    ) -> list[dict[str, Any]]:
        poses = self._unwrapped_poses(states)
        velocities = _finite_difference_velocity(poses, timestamps)
        records: list[dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            state = states[index]
            action = actions[index]
            pose = poses[index]
            velocity = velocities[index]
            xyz = pose[:3]
            gripper_state = [float(state[7])]
            gripper_action = [float(action[6])]
            arm_action = [float(value) for value in action[:6]]
            records.append(
                {
                    "timestamp_sec": timestamp,
                    "sample_index": index,
                    "raw_state": state,
                    "raw_action": action,
                    "action": arm_action,
                    "gripper_action": gripper_action,
                    "proprioception": pose,
                    "joint_positions": pose,
                    "state": pose,
                    "qvel": velocity,
                    "velocity": velocity,
                    "joint_velocity": velocity,
                    "gripper": gripper_state,
                    "gripper_state": gripper_state,
                    "wrist_poses": {"ee": {"position": xyz}},
                }
            )
        return records

    def _unwrapped_poses(self, states: list[list[float]]) -> list[list[float]]:
        if self.spec.orientation == "quaternion":
            quaternions = unwrap_quaternion_sign([state[3:7] for state in states])
            return [
                [float(state[0]), float(state[1]), float(state[2]), *quaternion_to_rotvec(quaternion)]
                for state, quaternion in zip(states, quaternions)
            ]
        poses = [
            [float(state[0]), float(state[1]), float(state[2]), float(state[3]), float(state[4]), float(state[5])]
            for state in states
        ]
        return _unwrap_euclidean_tail(poses, start_index=3)

    def _build_streams(
        self,
        records: list[dict[str, Any]],
        timestamps: list[float],
        sample_id: str,
        video_path: Path,
        parquet_path: Path,
        camera_name: str,
        manifest_entry: dict[str, Any],
    ) -> list[ModalityStream]:
        base_metadata = {
            "dataset": self.spec.dataset,
            "sample_id": sample_id,
            "video_path": str(video_path),
            "telemetry_path": str(parquet_path),
            "camera_name": camera_name,
            "fps": self.spec.fps,
            "orientation": self.spec.orientation,
            "episode_index": manifest_entry.get("episode_index"),
            "repo_id": manifest_entry.get("repo_id"),
        }
        return [
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "vector_representation": "ee_xyz_plus_unwrapped_orientation_plus_gripper",
                },
            ),
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "raw_action_width": 7,
                    "metric_action_width": 6,
                    "source_gripper_indices_removed": [6],
                    "vector_representation": "xyz_rpy_arm_only",
                },
            ),
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "raw_state_width": 8,
                    "metric_proprioception_width": 6,
                    "vector_representation": (
                        "xyz_plus_unwrapped_rotvec"
                        if self.spec.orientation == "quaternion"
                        else "xyz_plus_unwrapped_rpy"
                    ),
                },
            ),
            ModalityStream(
                name="velocity",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "metric_velocity_width": 6,
                    "vector_representation": "finite_difference_of_unwrapped_pose",
                },
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "source_gripper_state_index": 7,
                    "source_gripper_action_index": 6,
                    "low_state_label": "closed",
                    "high_state_label": "open",
                },
            ),
        ]


def adapter_for_dataset(dataset: str) -> RoboFineLeRobotAdapter:
    """Return the dataset-specific RoboFine adapter."""

    from pathologist.adapters.robofine_bcz import RoboFineBCZAdapter
    from pathologist.adapters.robofine_bridge import RoboFineBridgeAdapter
    from pathologist.adapters.robofine_rt1 import RoboFineRT1Adapter

    adapters: dict[str, Callable[[], RoboFineLeRobotAdapter]] = {
        "RT-1": RoboFineRT1Adapter,
        "BridgeDataV2": RoboFineBridgeAdapter,
        "BC-Z": RoboFineBCZAdapter,
    }
    factory = adapters.get(dataset)
    if factory is None:
        raise ValueError(f"No RoboFine LeRobot adapter for dataset {dataset!r}.")
    return factory()


def load_jsonl_by_sample_id(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a JSONL file keyed by sample_id."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("sample_id"):
            entries[str(record["sample_id"])] = record
    return entries


def resolve_sample_paths(
    sample: dict[str, Any],
    manifest_entry: dict[str, Any],
    eval_data: str | Path,
    bench_dir: str | Path | None = None,
) -> tuple[Path, Path, str]:
    """Return (video_path, parquet_path, camera_name) for one EvalSets sample."""

    eval_data_path = Path(eval_data).expanduser().resolve()
    bench_path = (
        Path(bench_dir).expanduser().resolve()
        if bench_dir is not None
        else eval_data_path.parent
    )
    sample_id = str(sample["sample_id"])
    dataset = str(sample["dataset"])
    view_names = sample.get("meta", {}).get("view_names") or []
    if not view_names:
        raise ValueError(f"{sample_id}: EvalSets meta.view_names is empty.")
    camera_name = str(view_names[0])
    video_path = eval_data_path / "videos" / dataset / sample_id / f"{camera_name}.mp4"
    telemetry_path = manifest_entry.get("telemetry_path")
    candidates: list[Path] = [eval_data_path / "telemetry" / dataset / f"{sample_id}.parquet"]
    if telemetry_path:
        relative = Path(str(telemetry_path))
        if relative.is_absolute():
            candidates.insert(0, relative)
        else:
            candidates.extend(
                [
                    bench_path / relative,
                    eval_data_path / relative,
                    eval_data_path.parent / relative,
                ]
            )
    parquet_path = next((path for path in candidates if path.exists()), candidates[0])
    return video_path, parquet_path.expanduser().resolve(), camera_name


def quaternion_to_rotvec(quaternion: list[float]) -> list[float]:
    """Convert a possibly unnormalized xyzw quaternion to a rotation vector."""

    x, y, z, w = (float(component) for component in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return [0.0, 0.0, 0.0]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xyz_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(xyz_norm, w)
    if xyz_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    scale = angle / xyz_norm
    return [x * scale, y * scale, z * scale]


def unwrap_quaternion_sign(quaternions: list[list[float]]) -> list[list[float]]:
    """Flip sequential quaternion signs so consecutive dots stay non-negative."""

    unwrapped: list[list[float]] = []
    previous: list[float] | None = None
    for raw in quaternions:
        quaternion = [float(component) for component in raw]
        if previous is not None:
            dot = sum(left * right for left, right in zip(quaternion, previous))
            if dot < 0.0:
                quaternion = [-component for component in quaternion]
        unwrapped.append(quaternion)
        previous = quaternion
    return unwrapped


def _unwrap_euclidean_tail(poses: list[list[float]], start_index: int) -> list[list[float]]:
    """Unwrap orientation components by removing 2π jumps relative to the previous step."""

    if not poses:
        return []
    unwrapped = [list(poses[0])]
    for pose in poses[1:]:
        previous = unwrapped[-1]
        next_pose = list(pose)
        for index in range(start_index, len(pose)):
            delta = float(pose[index]) - float(previous[index])
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta <= -math.pi:
                delta += 2.0 * math.pi
            next_pose[index] = float(previous[index]) + delta
        unwrapped.append(next_pose)
    return unwrapped


def _finite_difference_velocity(poses: list[list[float]], timestamps: list[float]) -> list[list[float]]:
    if not poses:
        return []
    velocities: list[list[float]] = []
    for index, pose in enumerate(poses):
        if index == 0:
            if len(poses) == 1:
                velocities.append([0.0] * len(pose))
                continue
            dt = timestamps[1] - timestamps[0]
            previous = poses[0]
            current = poses[1]
        else:
            dt = timestamps[index] - timestamps[index - 1]
            previous = poses[index - 1]
            current = pose
        if dt <= 0:
            dt = 1.0
        velocities.append([(left - right) / dt for left, right in zip(current, previous)])
    return velocities


def materialize_video_frames(
    video_path: Path,
    frame_cache_dir: Path,
    timestamps_sec: list[float],
    camera_name: str,
    expected_frame_count: int,
) -> list[CameraFrameRef]:
    """Extract video frames and stamp them with parquet timestamps."""

    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _frames_from_cache(frame_cache_dir, camera_name, timestamps_sec)
    if len(existing) == expected_frame_count:
        return existing
    _clear_cached_images(frame_cache_dir)
    decoded = _decode_video_frames(video_path)
    if len(decoded) != expected_frame_count:
        raise ValueError(
            f"Decoded {len(decoded)} frames from {video_path}, expected {expected_frame_count}."
        )
    if len(decoded) != len(timestamps_sec):
        raise ValueError(
            f"Video {video_path} has {len(decoded)} frames, parquet has {len(timestamps_sec)} timestamps."
        )
    cv2 = _import_cv2()
    refs: list[CameraFrameRef] = []
    for index, frame in enumerate(decoded):
        timestamp = timestamps_sec[index]
        image_path = frame_cache_dir / f"{index:06d}_t{timestamp:08.3f}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Could not write extracted frame: {image_path}")
        refs.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=index,
            )
        )
    return refs


def _decode_video_frames(video_path: Path) -> list[Any]:
    """Decode every video frame as BGR arrays. Prefers PyAV so AV1 files work."""

    try:
        import av
    except ModuleNotFoundError:
        av = None
    if av is not None:
        av.logging.set_level(av.logging.PANIC)
        frames: list[Any] = []
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="bgr24"))
        if frames:
            return frames
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RoboFine video: {video_path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _frames_from_cache(
    frame_dir: Path,
    camera_name: str,
    timestamps_sec: list[float],
) -> list[CameraFrameRef]:
    if not frame_dir.exists():
        return []
    image_paths = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(image_paths) != len(timestamps_sec):
        return []
    return [
        CameraFrameRef(
            camera_name=camera_name,
            timestamp_sec=timestamps_sec[index],
            image_path=image_path,
            frame_index=index,
        )
        for index, image_path in enumerate(image_paths)
    ]


def _read_parquet(path: Path) -> dict[str, list[Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RoboFine LeRobot adapters require pyarrow to read parquet.") from exc
    table = pq.read_table(path)
    required = ("observation.state", "action", "timestamp")
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}.")
    return {name: table.column(name).to_pylist() for name in required}


def _as_float_list(row: Any, expected_width: int, column_name: str) -> list[float]:
    if not isinstance(row, (list, tuple)) or len(row) < expected_width:
        width = len(row) if isinstance(row, (list, tuple)) else type(row).__name__
        raise ValueError(f"{column_name} has width {width}, expected at least {expected_width}.")
    return [float(value) for value in row[:expected_width]]


def _duration_sec(timestamps: list[float], fps: float) -> float:
    if not timestamps:
        return 0.0
    if len(timestamps) == 1:
        return 1.0 / fps if fps > 0 else 0.0
    dt = timestamps[1] - timestamps[0]
    if dt <= 0:
        dt = 1.0 / fps if fps > 0 else 0.0
    return float(timestamps[-1] + dt)


def _require_manifest_alignment(sample_id: str, manifest_entry: dict[str, Any]) -> None:
    status = manifest_entry.get("status")
    if status != "ok":
        raise ValueError(f"{sample_id}: telemetry manifest status is {status!r}, expected 'ok'.")
    if manifest_entry.get("length_matches_video") is not True:
        raise ValueError(f"{sample_id}: telemetry length does not match the local video.")
