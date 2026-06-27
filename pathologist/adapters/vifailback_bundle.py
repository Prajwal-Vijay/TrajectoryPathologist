"""Adapter for ViFailBack HDF5 episodes."""

from __future__ import annotations

import re
import shutil
import subprocess
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
        hdf5_fps: float = 25.0,
        default_control_hz: float = 50.0,
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
        if hdf5_fps <= 0:
            raise ValueError("hdf5_fps must be positive.")
        if default_control_hz <= 0:
            raise ValueError("default_control_hz must be positive.")

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

        duration_sec = _infer_duration_sec(
            sample_count=sample_count,
            decoded_frames=decoded_frames,
            hdf5_fps=hdf5_fps,
            default_control_hz=default_control_hz,
        )
        timestamps = _uniform_timestamps(sample_count, duration_sec)
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
        streams = _build_streams(
            records,
            timestamps,
            h5_episode_path,
            inferred_gripper_indices,
            gripper_normalization,
        )

        raw_metadata = {
            "dataset": "ViFailBack",
            "h5_path": str(h5_episode_path),
            "video_path": str(resolved_video_path) if resolved_video_path is not None else None,
            "sample_count": sample_count,
            "duration_sec": duration_sec,
            "hdf5_fps": hdf5_fps,
            "hdf5_timestep_sec": 1.0 / hdf5_fps,
            "video_sample_fps": video_sample_fps,
            "default_control_hz": default_control_hz,
            "gripper_indices": inferred_gripper_indices,
            "gripper_normalization": gripper_normalization,
            "hdf5_datasets": sorted(episode),
            "hdf5_rgb_frame_count": len(hdf5_rgb_frames),
            "decoded_frame_count": len(decoded_frames),
            "frame_cache_dir": str(cache_dir),
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

        if frame_cache_dir.exists():
            cached_frames = _frames_from_image_dir(frame_cache_dir, hdf5_fps)
            if cached_frames:
                return cached_frames

        frames_by_camera = {
            camera_name: episode.get(f"observations/images/{camera_name}")
            for camera_name in self.RGB_CAMERA_NAMES
        }
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


def _infer_duration_sec(
    sample_count: int,
    decoded_frames: list[CameraFrameRef],
    hdf5_fps: float,
    default_control_hz: float,
) -> float:
    if decoded_frames:
        return max(max(frame.timestamp_sec for frame in decoded_frames), (sample_count - 1) / hdf5_fps)
    if sample_count <= 1:
        return 0.0
    return (sample_count - 1) / (hdf5_fps or default_control_hz)


def _uniform_timestamps(sample_count: int, duration_sec: float) -> list[float]:
    if sample_count <= 1:
        return [0.0] if sample_count == 1 else []
    if duration_sec <= 0:
        return [float(index) for index in range(sample_count)]
    return [
        duration_sec * index / (sample_count - 1)
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
        if _row(episode.get("action_eef"), index):
            record["action_eef"] = _row(episode.get("action_eef"), index)
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


def _frames_from_image_dir(image_dir: Path, sample_fps: float) -> list[CameraFrameRef]:
    image_paths = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ViFailBackBundleAdapter.IMAGE_SUFFIXES
        ],
        key=_natural_sort_key,
    )
    return [
        CameraFrameRef(
            camera_name=image_path.parent.name,
            timestamp_sec=_timestamp_from_image_name(image_path, index, sample_fps),
            image_path=image_path,
            frame_index=index,
        )
        for index, image_path in enumerate(image_paths)
    ]


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
