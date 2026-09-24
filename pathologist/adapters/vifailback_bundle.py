"""Adapter for ViFailBack HDF5 episodes."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from functools import lru_cache
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


class ViFailBackBundleAdapter:
    """Load one ViFailBack raw .h5 episode into TrajectoryBundle."""

    NATIVE_SAMPLE_RATE_HZ = 25.0
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    RGB_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    DEFAULT_GRIPPER_INDICES_BY_WIDTH = {
        14: [6, 13],
        16: [7, 15],
    }

    def load(
        self,
        h5_path: str | Path,
        video_path: str | Path | None = None,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        video_sample_fps: float = 2.0,
        hdf5_fps: float = NATIVE_SAMPLE_RATE_HZ,
        default_control_hz: float = NATIVE_SAMPLE_RATE_HZ,
        gripper_indices: list[int] | None = None,
    ) -> TrajectoryBundle:
        """Load one ViFailBack episode file.

        The HDF5 file supplies both telemetry and compressed RGB frames. The
        optional video_path remains a fallback for external annotated videos or
        pre-extracted frame folders.
        """

        h5_episode_path = Path(h5_path).expanduser().resolve()
        if not h5_episode_path.exists() or not h5_episode_path.is_file():
            raise FileNotFoundError(f"Could not find ViFailBack HDF5 episode: {h5_episode_path}")
        if video_sample_fps <= 0:
            raise ValueError("video_sample_fps must be positive.")
        _validate_native_sample_rate(hdf5_fps)
        _validate_native_sample_rate(default_control_hz, argument_name="default_control_hz")

        cache_dir = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else h5_episode_path.parent / "pathologist_frame_cache" / h5_episode_path.stem
        )
        hdf5_rgb_cache_dir = cache_dir / "hdf5_rgb"
        hdf5_rgb_frames = _frames_from_image_dir(hdf5_rgb_cache_dir, hdf5_fps) if hdf5_rgb_cache_dir.exists() else []
        episode = self._read_hdf5_episode(
            h5_episode_path,
            load_images=not hdf5_rgb_frames,
        )
        sample_count = _infer_sample_count(episode)
        if sample_count <= 0:
            raise ValueError(f"Could not infer sample count from ViFailBack HDF5 episode: {h5_episode_path}")
        _validate_loaded_series_lengths(episode, sample_count)

        if hdf5_rgb_frames and not _camera_frames_are_complete(
            hdf5_rgb_frames,
            sample_count=sample_count,
            expected_camera_names=self.RGB_CAMERA_NAMES,
        ):
            episode = self._read_hdf5_episode(h5_episode_path, load_images=True)
            _validate_loaded_series_lengths(episode, sample_count)
            hdf5_rgb_frames = []
        if not hdf5_rgb_frames:
            hdf5_rgb_frames = self._write_hdf5_rgb_frames(
                episode=episode,
                frame_cache_dir=hdf5_rgb_cache_dir,
                hdf5_fps=hdf5_fps,
            )
        decoded_frames = hdf5_rgb_frames
        resolved_video_path = None
        if not decoded_frames and video_path is not None:
            resolved_video_path = Path(video_path).expanduser().resolve()
            if not resolved_video_path.exists():
                raise FileNotFoundError(f"Could not find ViFailBack video/image source: {resolved_video_path}")
            decoded_frames = self._load_video_frames(
                video_path=resolved_video_path,
                frame_cache_dir=cache_dir / "video",
                video_sample_fps=video_sample_fps,
            )

        timestamps = _uniform_timestamps(sample_count, hdf5_fps)
        duration_sec = timestamps[-1] if timestamps else 0.0
        inferred_gripper_indices = _resolve_gripper_indices(
            gripper_indices=gripper_indices,
            action=episode.get("action"),
            qpos=episode.get("observations/qpos"),
            default_by_width=self.DEFAULT_GRIPPER_INDICES_BY_WIDTH,
        )
        gripper_normalization = _gripper_normalization(episode, inferred_gripper_indices)
        records = _build_records(
            episode,
            timestamps,
            inferred_gripper_indices,
            gripper_normalization,
        )
        eef_velocity_records = _eef_velocity_records(
            episode.get("action_eef"),
            timestamps,
        )
        streams = _build_streams(
            records,
            timestamps,
            h5_episode_path,
            inferred_gripper_indices,
            gripper_normalization,
            eef_velocity_records,
        )
        annotation_metadata = _load_episode_annotation(h5_episode_path, hdf5_fps)
        _validate_annotation_frame_indices(annotation_metadata, sample_count)

        raw_metadata = {
            "dataset": "ViFailBack",
            "h5_path": str(h5_episode_path),
            "video_path": str(resolved_video_path) if resolved_video_path is not None else None,
            "sample_count": sample_count,
            "duration_sec": duration_sec,
            "hdf5_fps": hdf5_fps,
            "hdf5_timestep_sec": 1.0 / hdf5_fps,
            "timestamp_source": "exact_frame_index_divided_by_native_25_hz",
            "native_sample_rate_hz": self.NATIVE_SAMPLE_RATE_HZ,
            "video_sample_fps": video_sample_fps,
            "default_control_hz": default_control_hz,
            "gripper_indices": inferred_gripper_indices,
            "gripper_normalization": gripper_normalization,
            "hdf5_datasets": sorted(episode),
            "hdf5_rgb_frame_count": len(hdf5_rgb_frames),
            "decoded_frame_count": len(decoded_frames),
            "frame_cache_dir": str(cache_dir),
            "eef_velocity_sample_count": len(eef_velocity_records),
            **annotation_metadata,
        }

        return TrajectoryBundle(
            bundle_id=bundle_id or f"{h5_episode_path.parent.name}_{h5_episode_path.stem}",
            source_format=BundleFormat.VIFAILBACK_HDF5_VIDEO,
            root_path=h5_episode_path,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=decoded_frames,
            raw_metadata=raw_metadata,
        )

    def _write_hdf5_rgb_frames(
        self,
        episode: dict[str, Any],
        frame_cache_dir: Path,
        hdf5_fps: float,
    ) -> list[CameraFrameRef]:
        """Materialize compressed HDF5 RGB image bytes to frame files."""

        frames_by_camera = {
            camera_name: episode.get(f"observations/images/{camera_name}")
            for camera_name in self.RGB_CAMERA_NAMES
        }
        expected_sample_count = max(
            (
                len(frames)
                for frames in frames_by_camera.values()
                if isinstance(frames, list)
            ),
            default=0,
        )
        if frame_cache_dir.exists():
            cached_frames = _frames_from_image_dir(frame_cache_dir, hdf5_fps)
            if expected_sample_count > 0 and _camera_frames_are_complete(
                cached_frames,
                sample_count=expected_sample_count,
                expected_camera_names=self.RGB_CAMERA_NAMES,
            ):
                return cached_frames

        if not any(isinstance(frames, list) and frames for frames in frames_by_camera.values()):
            return []

        camera_refs: list[CameraFrameRef] = []
        for camera_name, frames in frames_by_camera.items():
            if not isinstance(frames, list) or not frames:
                continue
            camera_dir = frame_cache_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            for index, image_bytes in enumerate(frames):
                if not isinstance(image_bytes, (bytes, bytearray)):
                    continue
                suffix = _image_suffix_from_bytes(bytes(image_bytes))
                image_path = camera_dir / f"{index:06d}_t{index / hdf5_fps:08.3f}{suffix}"
                if not image_path.exists():
                    image_path.write_bytes(bytes(image_bytes))
                camera_refs.append(
                    CameraFrameRef(
                        camera_name=camera_name,
                        timestamp_sec=index / hdf5_fps,
                        image_path=image_path,
                        frame_index=index,
                    )
                )
        camera_refs.sort(key=lambda frame: (frame.timestamp_sec, frame.camera_name, frame.frame_index or -1))
        return camera_refs

    def _read_hdf5_episode(self, h5_path: Path, load_images: bool = True) -> dict[str, Any]:
        try:
            import h5py  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ViFailBackBundleAdapter requires h5py to read .h5 episodes. "
                "Install h5py in this environment before loading ViFailBack data."
            ) from exc

        episode: dict[str, Any] = {}
        with h5py.File(h5_path, "r") as handle:
            for dataset_path in (
                "action",
                "action_eef",
                "action_leader",
                "base_action",
                "observations/effort",
                "observations/qpos",
                "observations/qvel",
            ):
                if dataset_path in handle:
                    episode[dataset_path] = _dataset_to_python(handle[dataset_path])
            if load_images:
                for camera_name in self.RGB_CAMERA_NAMES:
                    dataset_path = f"observations/images/{camera_name}"
                    if dataset_path in handle:
                        episode[dataset_path] = _image_dataset_to_bytes_list(handle[dataset_path])
        return episode

    def _load_video_frames(
        self,
        video_path: Path,
        frame_cache_dir: Path,
        video_sample_fps: float,
    ) -> list[CameraFrameRef]:
        if video_path.is_dir():
            return _frames_from_image_dir(video_path, video_sample_fps)
        if video_path.suffix.lower() in self.IMAGE_SUFFIXES:
            return [
                CameraFrameRef(
                    camera_name=video_path.stem,
                    timestamp_sec=0.0,
                    image_path=video_path,
                    frame_index=0,
                )
            ]
        if video_path.suffix.lower() not in self.VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported ViFailBack visual source: {video_path}")

        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = sorted(
            [
                path
                for path in frame_cache_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            ],
            key=_natural_sort_key,
        )
        if not existing_frames:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                raise FileNotFoundError(
                    "ffmpeg is required to extract frames from ViFailBack videos. "
                    "Install ffmpeg or pass a directory of pre-extracted frames as video_path."
                )
            output_pattern = frame_cache_dir / "frame_%06d.jpg"
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={video_sample_fps}",
                str(output_pattern),
            ]
            subprocess.run(command, check=True)
            existing_frames = sorted(
                [
                    path
                    for path in frame_cache_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
                ],
                key=_natural_sort_key,
            )

        return [
            CameraFrameRef(
                camera_name=video_path.stem,
                timestamp_sec=index / video_sample_fps,
                image_path=frame_path,
                frame_index=index,
            )
            for index, frame_path in enumerate(existing_frames)
        ]


def _dataset_to_python(dataset: Any) -> Any:
    value = dataset[()]
    return _to_builtin(value)


def _image_dataset_to_bytes_list(dataset: Any) -> list[bytes]:
    value = dataset[()]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    frames: list[bytes] = []
    for item in value:
        if isinstance(item, bytes):
            frames.append(item)
        elif isinstance(item, bytearray):
            frames.append(bytes(item))
        elif hasattr(item, "tobytes"):
            frames.append(item.tobytes())
        elif isinstance(item, list):
            frames.append(bytes(item))
    return frames


def _to_builtin(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def _infer_sample_count(episode: dict[str, Any]) -> int:
    for key in (
        "action",
        "observations/qpos",
        "observations/qvel",
        "observations/effort",
        "action_eef",
        "action_leader",
        "base_action",
    ):
        value = episode.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _validate_native_sample_rate(
    sample_rate_hz: float,
    argument_name: str = "hdf5_fps",
) -> None:
    native_rate = ViFailBackBundleAdapter.NATIVE_SAMPLE_RATE_HZ
    if not math.isclose(float(sample_rate_hz), native_rate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"ViFailBack is sampled at exactly {native_rate:g} Hz; "
            f"{argument_name}={sample_rate_hz!r} would misalign frame annotations."
        )


def _validate_loaded_series_lengths(
    episode: dict[str, Any],
    sample_count: int,
) -> None:
    mismatches = {
        key: len(value)
        for key, value in episode.items()
        if isinstance(value, list) and len(value) != sample_count
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}={length}"
            for key, length in sorted(mismatches.items())
        )
        raise ValueError(
            "ViFailBack episode streams must remain frame-synchronous at 25 Hz; "
            f"expected {sample_count} samples but found {formatted}."
        )


def _uniform_timestamps(sample_count: int, sample_rate_hz: float) -> list[float]:
    return [
        index / sample_rate_hz
        for index in range(sample_count)
    ]


def _resolve_gripper_indices(
    gripper_indices: list[int] | None,
    action: Any,
    qpos: Any,
    default_by_width: dict[int, list[int]],
) -> list[int]:
    if gripper_indices is not None:
        return sorted({int(index) for index in gripper_indices if int(index) >= 0})
    width = _series_width(qpos) or _series_width(action)
    return list(default_by_width.get(width, []))


def _series_width(series: Any) -> int | None:
    if isinstance(series, list) and series and isinstance(series[0], list):
        return len(series[0])
    return None


def _gripper_normalization(
    episode: dict[str, Any],
    gripper_indices: list[int],
) -> dict[str, Any]:
    raw_series = _gripper_raw_series(episode, gripper_indices)
    if not raw_series or not gripper_indices:
        return {
            "source": None,
            "indices": gripper_indices,
            "minimum": [],
            "maximum": [],
            "range": [],
            "normalized": False,
        }

    width = max(len(sample) for sample in raw_series)
    minimum = []
    maximum = []
    ranges = []
    for dim in range(width):
        values = [
            sample[dim]
            for sample in raw_series
            if dim < len(sample)
        ]
        dim_min = min(values) if values else 0.0
        dim_max = max(values) if values else dim_min
        minimum.append(dim_min)
        maximum.append(dim_max)
        ranges.append(dim_max - dim_min)
    return {
        "source": "observations/qpos" if episode.get("observations/qpos") is not None else "action",
        "indices": gripper_indices,
        "minimum": minimum,
        "maximum": maximum,
        "range": ranges,
        "normalized": any(value > 0 for value in ranges),
    }


def _gripper_raw_series(
    episode: dict[str, Any],
    gripper_indices: list[int],
) -> list[list[float]]:
    source = episode.get("observations/qpos") or episode.get("action")
    if not isinstance(source, list):
        return []
    return [
        _take_indices(_row(source, index), gripper_indices)
        for index in range(len(source))
    ]


def _build_records(
    episode: dict[str, Any],
    timestamps: list[float],
    gripper_indices: list[int],
    gripper_normalization: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sample_count = len(timestamps)
    records: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        full_action = _row(episode.get("action"), index)
        action_eef = _row(episode.get("action_eef"), index)
        qpos = _row(episode.get("observations/qpos"), index)
        qvel = _row(episode.get("observations/qvel"), index)
        effort = _row(episode.get("observations/effort"), index)
        action = _drop_indices(full_action, gripper_indices)
        proprioception = _drop_indices(qpos, gripper_indices)
        velocity = _drop_indices(qvel, gripper_indices)
        raw_gripper = _take_indices(qpos, gripper_indices) or _take_indices(full_action, gripper_indices)
        gripper = _normalize_gripper(raw_gripper, gripper_normalization)

        record: dict[str, Any] = {
            "timestamp_sec": timestamp,
            "sample_index": index,
        }
        if full_action:
            record["raw_action"] = full_action
            record["action"] = action
        if action_eef:
            record["action_eef"] = action_eef
            wrist_poses = _eef_wrist_poses(action_eef)
            if wrist_poses:
                record["wrist_poses"] = wrist_poses
        if _row(episode.get("action_leader"), index):
            record["action_leader"] = _row(episode.get("action_leader"), index)
        if _row(episode.get("base_action"), index):
            record["base_action"] = _row(episode.get("base_action"), index)
        if qpos:
            record["qpos"] = qpos
            record["proprioception"] = proprioception
            record["joint_positions"] = proprioception
            record["state"] = proprioception
        if qvel:
            record["raw_qvel"] = qvel
            record["qvel"] = velocity
            record["velocity"] = velocity
            record["joint_velocity"] = velocity
        if effort:
            record["effort"] = effort
        if raw_gripper:
            record["raw_gripper"] = raw_gripper
        if gripper:
            record["gripper"] = gripper
            record["gripper_state"] = gripper
        records.append(record)
    if len(records) != sample_count:
        raise ValueError("Internal ViFailBack adapter error: record count does not match timestamps.")
    return records


def _build_streams(
    records: list[dict[str, Any]],
    timestamps: list[float],
    h5_path: Path,
    gripper_indices: list[int],
    gripper_normalization: dict[str, Any] | None = None,
    eef_velocity_records: list[dict[str, Any]] | None = None,
) -> list[ModalityStream]:
    raw_action_width = _first_vector_width(records, "raw_action")
    action_width = _first_vector_width(records, "action")
    raw_qpos_width = _first_vector_width(records, "qpos")
    proprioception_width = _first_vector_width(records, "proprioception")
    raw_qvel_width = _first_vector_width(records, "raw_qvel")
    velocity_width = _first_vector_width(records, "velocity")
    base_metadata = {
        "source_h5_path": str(h5_path),
        "source_gripper_indices_removed": gripper_indices,
        "gripper_indices_removed_from_state_action": gripper_indices,
    }
    streams = [
        ModalityStream(
            name="telemetry",
            modality=Modality.TELEMETRY,
            timestamps_sec=timestamps,
            records=records,
            metadata=base_metadata,
        )
    ]
    if eef_velocity_records:
        streams.append(
            ModalityStream(
                name="eef_velocity",
                modality=Modality.TELEMETRY,
                timestamps_sec=[
                    float(record["timestamp_sec"])
                    for record in eef_velocity_records
                ],
                records=eef_velocity_records,
                metadata={
                    **base_metadata,
                    "source_field": "action_eef",
                    "velocity_source": "finite_difference_of_target_eef_pose",
                    "sample_rate_hz": ViFailBackBundleAdapter.NATIVE_SAMPLE_RATE_HZ,
                    "vector_groups": {
                        "left": [0, 1, 2],
                        "right": [3, 4, 5],
                    },
                },
            )
        )
    if _records_have(records, "action"):
        streams.append(
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "raw_action_width": raw_action_width,
                    "metric_action_width": action_width,
                    "vector_representation": "raw_action_without_source_gripper_indices",
                },
            )
        )
    if _records_have(records, "proprioception"):
        streams.append(
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "raw_qpos_width": raw_qpos_width,
                    "metric_proprioception_width": proprioception_width,
                    "vector_representation": "raw_qpos_without_source_gripper_indices",
                },
            )
        )
    if _records_have(records, "qvel"):
        streams.append(
            ModalityStream(
                name="velocity",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "raw_qvel_width": raw_qvel_width,
                    "metric_velocity_width": velocity_width,
                    "vector_representation": "raw_qvel_without_source_gripper_indices",
                },
            )
        )
    if _records_have(records, "effort"):
        streams.append(
            ModalityStream(
                name="effort",
                modality=Modality.FORCE,
                timestamps_sec=timestamps,
                records=records,
                metadata=base_metadata,
            )
        )
    if _records_have(records, "gripper"):
        streams.append(
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "gripper_binary_threshold": 0.5,
                    "low_state_label": "closed",
                    "high_state_label": "open",
                    "gripper_value_representation": "normalized_0_to_1",
                    "raw_gripper_value_representation": "dataset_qpos_or_action_units",
                    "gripper_normalization": gripper_normalization or {},
                },
            )
        )
    return streams


def _row(series: Any, index: int) -> list[float]:
    if not isinstance(series, list) or index >= len(series):
        return []
    value = series[index]
    if not isinstance(value, list):
        return []
    return [
        float(item)
        for item in value
        if isinstance(item, (int, float, bool))
    ]


def _drop_indices(vector: list[float], indices: list[int]) -> list[float]:
    excluded = set(indices)
    return [
        value
        for index, value in enumerate(vector)
        if index not in excluded
    ]


def _take_indices(vector: list[float], indices: list[int]) -> list[float]:
    return [
        vector[index]
        for index in indices
        if 0 <= index < len(vector)
    ]


def _normalize_gripper(
    raw_gripper: list[float],
    gripper_normalization: dict[str, Any] | None,
) -> list[float]:
    if not raw_gripper:
        return []
    if not gripper_normalization or not gripper_normalization.get("normalized"):
        return raw_gripper
    minimum = gripper_normalization.get("minimum")
    ranges = gripper_normalization.get("range")
    if not isinstance(minimum, list) or not isinstance(ranges, list):
        return raw_gripper
    normalized = []
    for dim, value in enumerate(raw_gripper):
        dim_range = ranges[dim] if dim < len(ranges) else 0.0
        dim_min = minimum[dim] if dim < len(minimum) else 0.0
        if not isinstance(dim_range, (int, float)) or dim_range <= 0:
            normalized.append(0.0)
            continue
        normalized.append((value - float(dim_min)) / float(dim_range))
    return normalized


def _records_have(records: list[dict[str, Any]], key: str) -> bool:
    return any(key in record and record[key] for record in records)


def _first_vector_width(records: list[dict[str, Any]], key: str) -> int | None:
    for record in records:
        value = record.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _eef_wrist_poses(action_eef: list[float]) -> dict[str, dict[str, list[float]]]:
    if len(action_eef) < 16:
        return {}
    poses = {}
    for wrist_name, offset in (("left", 0), ("right", 8)):
        quaternion = _normalized_quaternion_xyzw(action_eef[offset + 3 : offset + 7])
        if quaternion is None:
            return {}
        poses[wrist_name] = {
            "position": [float(value) for value in action_eef[offset : offset + 3]],
            "quaternion_xyzw": quaternion,
        }
    return poses


def _eef_velocity_records(
    action_eef_series: Any,
    timestamps: list[float],
) -> list[dict[str, Any]]:
    if not isinstance(action_eef_series, list) or len(action_eef_series) != len(timestamps):
        return []
    output = []
    for index in range(1, len(timestamps)):
        previous_poses = _eef_wrist_poses(_row(action_eef_series, index - 1))
        current_poses = _eef_wrist_poses(_row(action_eef_series, index))
        dt = timestamps[index] - timestamps[index - 1]
        if dt <= 0 or set(previous_poses) != {"left", "right"} or set(current_poses) != {"left", "right"}:
            continue
        linear_velocity: list[float] = []
        angular_velocity: list[float] = []
        for wrist_name in ("left", "right"):
            previous_pose = previous_poses[wrist_name]
            current_pose = current_poses[wrist_name]
            linear_velocity.extend(
                [
                    (current - previous) / dt
                    for previous, current in zip(
                        previous_pose["position"],
                        current_pose["position"],
                    )
                ]
            )
            angular_velocity.extend(
                _angular_velocity_from_quaternions(
                    previous_pose["quaternion_xyzw"],
                    current_pose["quaternion_xyzw"],
                    dt,
                )
            )
        output.append(
            {
                "timestamp_sec": timestamps[index],
                "sample_index": index,
                "eef_linear_velocity": linear_velocity,
                "eef_angular_velocity": angular_velocity,
                "eef_velocity": [
                    *linear_velocity[:3],
                    *angular_velocity[:3],
                    *linear_velocity[3:],
                    *angular_velocity[3:],
                ],
            }
        )
    return output


def _normalized_quaternion_xyzw(values: list[float]) -> list[float] | None:
    if len(values) != 4 or not all(math.isfinite(float(value)) for value in values):
        return None
    norm = sum(float(value) ** 2 for value in values) ** 0.5
    if norm <= 1e-12:
        return None
    return [float(value) / norm for value in values]


def _angular_velocity_from_quaternions(
    previous_xyzw: list[float],
    current_xyzw: list[float],
    dt: float,
) -> list[float]:
    previous_inverse = [
        -previous_xyzw[0],
        -previous_xyzw[1],
        -previous_xyzw[2],
        previous_xyzw[3],
    ]
    relative = _quaternion_multiply_xyzw(current_xyzw, previous_inverse)
    if relative[3] < 0:
        relative = [-value for value in relative]
    vector_norm = sum(value * value for value in relative[:3]) ** 0.5
    if vector_norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, min(1.0, relative[3])))
    scale = angle / (vector_norm * dt)
    return [value * scale for value in relative[:3]]


def _quaternion_multiply_xyzw(
    left: list[float],
    right: list[float],
) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _load_episode_annotation(
    h5_path: Path,
    sample_rate_hz: float,
) -> dict[str, Any]:
    raw_data_dir = next(
        (parent for parent in h5_path.parents if parent.name == "raw_data"),
        None,
    )
    task_name = h5_path.parent.name
    base = {
        "source_task": task_name,
        "source_episode": h5_path.stem,
        "annotation_available": False,
        "annotation_timing": "exact_source_frame_index_at_25_hz",
        "avoid_keyframe_indices": [],
        "avoid_keyframe_timestamps_sec": [],
        "correct_keyframe_indices": [],
        "correct_keyframe_timestamps_sec": [],
    }
    if raw_data_dir is None:
        return base

    annotation_path = (
        raw_data_dir.parent
        / "annotated_data"
        / task_name
        / f"{task_name}_annotations.json"
    )
    if not annotation_path.is_file():
        return {**base, "annotation_path": str(annotation_path)}

    expected_video_stem = f"{h5_path.stem}_cam_high"
    annotation = next(
        (
            item
            for item in _read_annotation_file(str(annotation_path))
            if Path(str(item.get("video", ""))).stem == expected_video_stem
        ),
        None,
    )
    if annotation is None:
        return {**base, "annotation_path": str(annotation_path)}

    avoid_indices = _annotation_keyframe_indices(annotation, "avoid")
    correct_indices = _annotation_keyframe_indices(annotation, "correct")
    return {
        **base,
        "annotation_available": True,
        "annotation_path": str(annotation_path),
        "task_instruction": annotation.get("task"),
        "failure_detection": annotation.get("failure_detection"),
        "failure_subtask": annotation.get("failure_subtask"),
        "failure_type": annotation.get("failure_type"),
        "annotation_source_index": annotation.get("source_index"),
        "avoid_keyframe_indices": avoid_indices,
        "avoid_keyframe_timestamps_sec": [
            index / sample_rate_hz
            for index in avoid_indices
        ],
        "correct_keyframe_indices": correct_indices,
        "correct_keyframe_timestamps_sec": [
            index / sample_rate_hz
            for index in correct_indices
        ],
    }


def _validate_annotation_frame_indices(
    annotation_metadata: dict[str, Any],
    sample_count: int,
) -> None:
    invalid = {
        key: index
        for key in ("avoid_keyframe_indices", "correct_keyframe_indices")
        for index in annotation_metadata.get(key, [])
        if not isinstance(index, int) or index < 0 or index >= sample_count
    }
    if invalid:
        formatted = ", ".join(f"{key}={index}" for key, index in invalid.items())
        raise ValueError(
            "ViFailBack annotation frame lies outside its source trajectory: "
            f"{formatted}; sample_count={sample_count}."
        )


@lru_cache(maxsize=128)
def _read_annotation_file(annotation_path: str) -> tuple[dict[str, Any], ...]:
    with Path(annotation_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of ViFailBack annotations in {annotation_path}")
    return tuple(item for item in payload if isinstance(item, dict))


def _annotation_keyframe_indices(
    annotation: dict[str, Any],
    keyframe_kind: str,
) -> list[int]:
    indices = set()
    key_name = f"{keyframe_kind}_keyframe"
    keyframes = annotation.get("keyframe")
    if isinstance(keyframes, list):
        for entry in keyframes:
            if isinstance(entry, dict):
                index = _frame_index_from_path(entry.get(key_name), keyframe_kind)
                if index is not None:
                    indices.add(index)
    if not indices:
        images = annotation.get("images")
        if isinstance(images, list):
            for image_path in images:
                index = _frame_index_from_path(image_path, keyframe_kind)
                if index is not None:
                    indices.add(index)
    return sorted(indices)


def _frame_index_from_path(value: Any, keyframe_kind: str) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(
        rf"(?:^|/)(\d+)_{re.escape(keyframe_kind)}_keyframe(?:_annotations)?\.[^.]+$",
        value,
    )
    return int(match.group(1)) if match else None


def _camera_frames_are_complete(
    frames: list[CameraFrameRef],
    sample_count: int,
    expected_camera_names: tuple[str, ...],
) -> bool:
    frame_counts_by_camera = {
        camera_name: sum(
            frame.camera_name == camera_name
            for frame in frames
        )
        for camera_name in expected_camera_names
    }
    frame_indices_by_camera = {
        camera_name: {
            frame.frame_index
            for frame in frames
            if frame.camera_name == camera_name
        }
        for camera_name in expected_camera_names
    }
    expected_indices = set(range(sample_count))
    return all(
        frame_counts_by_camera[camera_name] == sample_count
        and frame_indices_by_camera[camera_name] == expected_indices
        for camera_name in expected_camera_names
    )


def _frames_from_image_dir(image_dir: Path, sample_fps: float) -> list[CameraFrameRef]:
    image_paths_by_camera: dict[str, list[Path]] = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ViFailBackBundleAdapter.IMAGE_SUFFIXES:
            image_paths_by_camera.setdefault(path.parent.name, []).append(path)

    frames = []
    for camera_name, image_paths in image_paths_by_camera.items():
        for frame_index, image_path in enumerate(sorted(image_paths, key=_natural_sort_key)):
            frames.append(
                CameraFrameRef(
                    camera_name=camera_name,
                    timestamp_sec=_timestamp_from_image_name(
                        image_path,
                        frame_index,
                        sample_fps,
                    ),
                    image_path=image_path,
                    frame_index=frame_index,
                )
            )
    frames.sort(
        key=lambda frame: (
            frame.timestamp_sec,
            frame.camera_name,
            frame.frame_index if frame.frame_index is not None else -1,
        )
    )
    return frames


def _timestamp_from_image_name(image_path: Path, index: int, sample_fps: float) -> float:
    stem_numbers = re.findall(r"\d+(?:\.\d+)?", image_path.stem)
    if stem_numbers and any("." in token for token in stem_numbers):
        try:
            return float(stem_numbers[-1])
        except ValueError:
            pass
    return index / sample_fps


def _image_suffix_from_bytes(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _natural_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    key.append(path.suffix.lower())
    return key
