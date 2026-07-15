"""Adapter for REASSEMBLE HDF5 trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TaskContract,
    TrajectoryBundle,
)


@dataclass(frozen=True, slots=True)
class ReassembleSegment:
    """One high-level or low-level REASSEMBLE segment annotation."""

    segment_id: str
    level: str
    start_sec: float
    end_sec: float
    success: bool
    text: str
    index: int | None = None
    parent_segment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "level": self.level,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "success": self.success,
            "text": self.text,
            "index": self.index,
            "parent_segment_id": self.parent_segment_id,
        }


class ReassembleBundleAdapter:
    """Load one REASSEMBLE .h5 trajectory into the internal trajectory schema."""

    RGB_CAMERA_DATASETS = ("hama1", "hama2", "hand")
    EVENT_CAMERA_DATASET = "capture_node-camera-image"
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

    def load(
        self,
        h5_path: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        robot_downsample_hz: float = 50.0,
        camera_names: Iterable[str] | None = None,
        include_event_camera: bool = False,
        extract_frames: bool = True,
    ) -> TrajectoryBundle:
        """Load a REASSEMBLE HDF5 file.

        The adapter intentionally does not synthesize an action stream because
        REASSEMBLE stores observations/robot states, not explicit action
        commands, in the observed file format.
        """

        resolved_h5_path = Path(h5_path).expanduser().resolve()
        if not resolved_h5_path.exists() or not resolved_h5_path.is_file():
            raise FileNotFoundError(f"Could not find REASSEMBLE HDF5 file: {resolved_h5_path}")
        if robot_downsample_hz <= 0:
            raise ValueError("robot_downsample_hz must be positive.")

        try:
            import h5py  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ReassembleBundleAdapter requires h5py to read .h5 files.") from exc

        cache_dir = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else resolved_h5_path.parent / "pathologist_frame_cache" / resolved_h5_path.stem
        )
        requested_cameras = _camera_names(camera_names)
        if include_event_camera and self.EVENT_CAMERA_DATASET not in requested_cameras:
            requested_cameras.append(self.EVENT_CAMERA_DATASET)

        with h5py.File(resolved_h5_path, "r") as handle:
            timestamp_origin = _global_timestamp_origin(handle)
            duration_sec = _duration_sec(handle, timestamp_origin)
            segments = _parse_segments(handle, timestamp_origin)
            streams = _build_streams(
                handle=handle,
                h5_path=resolved_h5_path,
                timestamp_origin=timestamp_origin,
                robot_downsample_hz=robot_downsample_hz,
            )
            camera_frames = (
                self._materialize_camera_frames(
                    handle=handle,
                    frame_cache_dir=cache_dir,
                    timestamp_origin=timestamp_origin,
                    camera_names=requested_cameras,
                )
                if extract_frames
                else []
            )
            available_datasets = sorted(_top_level_dataset_names(handle))
            stream_counts = _stream_counts(handle)

        high_segments = [segment for segment in segments if segment.level == "high"]
        low_segments = [segment for segment in segments if segment.level == "low"]
        failed_high_segments = [segment for segment in high_segments if not segment.success]
        failed_low_segments = [segment for segment in low_segments if not segment.success]
        raw_metadata = {
            "dataset": "REASSEMBLE",
            "h5_path": str(resolved_h5_path),
            "frame_cache_dir": str(cache_dir),
            "timestamp_origin_sec": timestamp_origin,
            "duration_sec": duration_sec,
            "robot_downsample_hz": robot_downsample_hz,
            "camera_names_requested": requested_cameras,
            "decoded_frame_count": len(camera_frames),
            "available_top_level_datasets": available_datasets,
            "stream_counts": stream_counts,
            "reassemble_segments": [segment.to_dict() for segment in segments],
            "high_level_segment_count": len(high_segments),
            "low_level_segment_count": len(low_segments),
            "failed_high_level_count": len(failed_high_segments),
            "failed_low_level_count": len(failed_low_segments),
            "has_failed_segment": bool(failed_high_segments or failed_low_segments),
            "action_stream_available": False,
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_h5_path.stem,
            source_format=BundleFormat.REASSEMBLE_HDF5,
            root_path=resolved_h5_path,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=camera_frames,
            raw_metadata=raw_metadata,
        )

    def _materialize_camera_frames(
        self,
        handle: Any,
        frame_cache_dir: Path,
        timestamp_origin: float,
        camera_names: list[str],
    ) -> list[CameraFrameRef]:
        frames: list[CameraFrameRef] = []
        for camera_name in camera_names:
            if camera_name not in handle or f"timestamps/{camera_name}" not in handle:
                continue
            camera_dir = frame_cache_dir / "images" / camera_name
            timestamps = _normalized_timestamps(handle[f"timestamps/{camera_name}"], timestamp_origin)
            cached_frames = _frames_from_image_dir(camera_dir, camera_name, timestamps)
            if len(cached_frames) >= len(timestamps):
                frames.extend(cached_frames[: len(timestamps)])
                continue

            video_cache_path = frame_cache_dir / "videos" / f"{camera_name}.mp4"
            video_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not video_cache_path.exists():
                video_cache_path.write_bytes(bytes(handle[camera_name][()]))
            frames.extend(
                _extract_video_frames(
                    video_path=video_cache_path,
                    frame_cache_dir=camera_dir,
                    camera_name=camera_name,
                    timestamps=timestamps,
                )
            )
        frames.sort(key=lambda frame: (frame.timestamp_sec, frame.camera_name, frame.frame_index or -1))
        return frames


def _build_streams(
    handle: Any,
    h5_path: Path,
    timestamp_origin: float,
    robot_downsample_hz: float,
) -> list[ModalityStream]:
    base_metadata = {
        "dataset": "REASSEMBLE",
        "h5_path": str(h5_path),
        "timestamp_origin_sec": timestamp_origin,
        "robot_downsample_hz": robot_downsample_hz,
    }
    streams: list[ModalityStream] = []

    joint_positions = _read_robot_series(handle, "joint_positions", timestamp_origin, robot_downsample_hz)
    if joint_positions is not None:
        timestamps, values = joint_positions
        records = [
            {
                "timestamp_sec": timestamp,
                "sample_index": index,
                "proprioception": value,
                "joint_positions": value,
                "state": value,
            }
            for index, (timestamp, value) in enumerate(zip(timestamps, values))
        ]
        streams.append(
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={**base_metadata, "source_dataset": "robot_state/joint_positions"},
            )
        )

    joint_velocities = _read_robot_series(handle, "joint_velocities", timestamp_origin, robot_downsample_hz)
    if joint_velocities is not None:
        timestamps, values = joint_velocities
        records = [
            {
                "timestamp_sec": timestamp,
                "sample_index": index,
                "qvel": value,
                "velocity": value,
                "joint_velocity": value,
                "joint_velocities": value,
            }
            for index, (timestamp, value) in enumerate(zip(timestamps, values))
        ]
        streams.append(
            ModalityStream(
                name="velocity",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=records,
                metadata={**base_metadata, "source_dataset": "robot_state/joint_velocities"},
            )
        )

    gripper_positions = _read_robot_series(handle, "gripper_positions", timestamp_origin, robot_downsample_hz)
    if gripper_positions is not None:
        timestamps, values = gripper_positions
        records = [
            {
                "timestamp_sec": timestamp,
                "sample_index": index,
                "gripper": value,
                "gripper_state": value,
            }
            for index, (timestamp, value) in enumerate(zip(timestamps, values))
        ]
        streams.append(
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=records,
                metadata={
                    **base_metadata,
                    "source_dataset": "robot_state/gripper_positions",
                    "gripper_binary_threshold": 0.5,
                },
            )
        )

    pose_series = _read_robot_series(handle, "pose", timestamp_origin, robot_downsample_hz)
    telemetry_records: list[dict[str, Any]] = []
    telemetry_timestamps: list[float] = []
    if pose_series is not None:
        timestamps, values = pose_series
        for index, (timestamp, pose) in enumerate(zip(timestamps, values)):
            position = pose[:3]
            quaternion = pose[3:7]
            telemetry_records.append(
                {
                    "timestamp_sec": timestamp,
                    "sample_index": index,
                    "pose": pose,
                    "wrist_poses": {
                        "eef": {
                            "position": position,
                            "orientation_xyzw_or_wxyz": quaternion,
                        }
                    },
                }
            )
        telemetry_timestamps = timestamps

    force_metadata = _force_torque_metadata(handle)
    if telemetry_records:
        streams.append(
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=telemetry_timestamps,
                records=telemetry_records,
                metadata={
                    **base_metadata,
                    "source_dataset": "robot_state/pose",
                    "force_torque_sources": force_metadata,
                },
            )
        )
    elif force_metadata:
        streams.append(
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=[0.0],
                records=[{"timestamp_sec": 0.0}],
                metadata={**base_metadata, "force_torque_sources": force_metadata},
            )
        )
    return streams


def _read_robot_series(
    handle: Any,
    dataset_name: str,
    timestamp_origin: float,
    target_hz: float,
) -> tuple[list[float], list[list[float]]] | None:
    dataset_path = f"robot_state/{dataset_name}"
    timestamp_path = f"timestamps/{dataset_name}"
    if dataset_path not in handle or timestamp_path not in handle:
        return None
    timestamps = _normalized_timestamps(handle[timestamp_path], timestamp_origin)
    values = handle[dataset_path][:]
    if len(timestamps) == 0 or len(values) == 0:
        return None
    count = min(len(timestamps), len(values))
    indices = _downsample_indices(timestamps[:count], target_hz)
    return (
        [float(timestamps[index]) for index in indices],
        [[float(value) for value in values[index].tolist()] for index in indices],
    )


def _parse_segments(handle: Any, timestamp_origin: float) -> list[ReassembleSegment]:
    if "segments_info" not in handle:
        return []
    output: list[ReassembleSegment] = []
    segments_group = handle["segments_info"]
    for segment_key in sorted(segments_group.keys(), key=_numeric_sort_key):
        group = segments_group[segment_key]
        if not all(key in group for key in ("start", "end", "success", "text")):
            continue
        index = int(_scalar(group["index"])) if "index" in group else _safe_int(segment_key)
        output.append(
            ReassembleSegment(
                segment_id=str(segment_key),
                level="high",
                start_sec=float(_scalar(group["start"])) - timestamp_origin,
                end_sec=float(_scalar(group["end"])) - timestamp_origin,
                success=bool(_scalar(group["success"])),
                text=str(_scalar(group["text"])),
                index=index,
            )
        )
        low_level = group.get("low_level")
        if low_level is None:
            continue
        for low_key in sorted(low_level.keys(), key=_numeric_sort_key):
            low_group = low_level[low_key]
            if not all(key in low_group for key in ("start", "end", "success", "text")):
                continue
            output.append(
                ReassembleSegment(
                    segment_id=f"{segment_key}/{low_key}",
                    level="low",
                    start_sec=float(_scalar(low_group["start"])) - timestamp_origin,
                    end_sec=float(_scalar(low_group["end"])) - timestamp_origin,
                    success=bool(_scalar(low_group["success"])),
                    text=str(_scalar(low_group["text"])),
                    parent_segment_id=str(segment_key),
                )
            )
    output.sort(key=lambda segment: (segment.start_sec, segment.level, segment.segment_id))
    return output


def _global_timestamp_origin(handle: Any) -> float:
    if "timestamps" not in handle:
        return 0.0
    minima: list[float] = []
    for dataset in handle["timestamps"].values():
        if getattr(dataset, "shape", ()) and int(dataset.shape[0]) > 0:
            values = dataset[:]
            minima.append(float(values.min()))
    return min(minima) if minima else 0.0


def _duration_sec(handle: Any, timestamp_origin: float) -> float | None:
    if "timestamps" not in handle:
        return None
    maxima: list[float] = []
    for dataset in handle["timestamps"].values():
        if getattr(dataset, "shape", ()) and int(dataset.shape[0]) > 0:
            values = dataset[:]
            maxima.append(float(values.max()) - timestamp_origin)
    return max(maxima) if maxima else None


def _normalized_timestamps(dataset: Any, timestamp_origin: float) -> list[float]:
    return [float(value) - timestamp_origin for value in dataset[:].tolist()]


def _downsample_indices(timestamps: list[float], target_hz: float) -> list[int]:
    if not timestamps:
        return []
    min_step = 1.0 / target_hz
    indices = [0]
    last_timestamp = float(timestamps[0])
    for index, timestamp in enumerate(timestamps[1:], start=1):
        if float(timestamp) - last_timestamp >= min_step:
            indices.append(index)
            last_timestamp = float(timestamp)
    if indices[-1] != len(timestamps) - 1:
        indices.append(len(timestamps) - 1)
    return indices


def _materialized_video_metadata(video_path: Path) -> dict[str, int | float]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open REASSEMBLE video blob: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def _extract_video_frames(
    video_path: Path,
    frame_cache_dir: Path,
    camera_name: str,
    timestamps: list[float],
) -> list[CameraFrameRef]:
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_frames = _frames_from_image_dir(frame_cache_dir, camera_name, timestamps)
    if len(cached_frames) >= len(timestamps):
        return cached_frames[: len(timestamps)]

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open REASSEMBLE video blob: {video_path}")
    refs: list[CameraFrameRef] = []
    index = 0
    while index < len(timestamps):
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = timestamps[index]
        image_path = frame_cache_dir / f"{index:06d}_t{timestamp:08.3f}.jpg"
        if not image_path.exists() and not cv2.imwrite(str(image_path), frame):
            capture.release()
            raise RuntimeError(f"Could not write REASSEMBLE frame: {image_path}")
        refs.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=index,
            )
        )
        index += 1
    capture.release()
    return refs


def _frames_from_image_dir(
    frame_dir: Path,
    camera_name: str,
    timestamps: list[float],
) -> list[CameraFrameRef]:
    if not frame_dir.exists():
        return []
    image_paths = sorted(
        [
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ReassembleBundleAdapter.IMAGE_SUFFIXES
        ],
        key=_natural_sort_key,
    )
    refs: list[CameraFrameRef] = []
    for index, image_path in enumerate(image_paths):
        timestamp = timestamps[index] if index < len(timestamps) else _timestamp_from_filename(image_path, index)
        refs.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=index,
            )
        )
    return refs


def _force_torque_metadata(handle: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in (
        "measured_force",
        "measured_torque",
        "compensated_base_force",
        "compensated_base_torque",
        "joint_efforts",
        "gripper_efforts",
        "gripper_velocities",
    ):
        path = f"robot_state/{name}"
        timestamp_path = f"timestamps/{name}"
        if path in handle:
            metadata[name] = {
                "dataset": path,
                "shape": [int(item) for item in handle[path].shape],
                "timestamp_count": int(handle[timestamp_path].shape[0]) if timestamp_path in handle else None,
            }
    return metadata


def _stream_counts(handle: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "robot_state" in handle:
        for name, dataset in handle["robot_state"].items():
            shape = getattr(dataset, "shape", ())
            counts[f"robot_state/{name}"] = int(shape[0]) if shape else 1
    if "timestamps" in handle:
        for name, dataset in handle["timestamps"].items():
            shape = getattr(dataset, "shape", ())
            counts[f"timestamps/{name}"] = int(shape[0]) if shape else 1
    return counts


def _top_level_dataset_names(handle: Any) -> list[str]:
    return [
        str(name)
        for name, value in handle.items()
        if not hasattr(value, "keys")
    ]


def _camera_names(camera_names: Iterable[str] | None) -> list[str]:
    names = [str(name) for name in camera_names or ("hand",) if str(name)]
    return names or ["hand"]


def _scalar(dataset: Any) -> Any:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _numeric_sort_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**9, value)


def _natural_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: list[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    key.append(path.suffix.lower())
    return key


def _timestamp_from_filename(path: Path, fallback_index: int) -> float:
    match = re.search(r"_t(\d+(?:\.\d+)?)", path.stem)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return float(fallback_index)


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "REASSEMBLE frame extraction requires opencv-python/cv2."
        ) from exc
    return cv2
