"""Adapters for RoboFail simulation and real-world trajectories."""

from __future__ import annotations

import json
import pickle
import re
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


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class RoboFailRealTaskMetadata:
    """Metadata row from tasks_real_world.json keyed by folder name."""

    folder_name: str
    task_id: str | None
    payload: dict[str, Any]


class RoboFailSimBundleAdapter:
    """Load one RoboFail sim episode into the internal trajectory schema."""

    def load(
        self,
        episode_dir: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
    ) -> TrajectoryBundle:
        resolved_episode_dir = Path(episode_dir).expanduser().resolve()
        if not resolved_episode_dir.exists() or not resolved_episode_dir.is_dir():
            raise FileNotFoundError(f"Could not find RoboFail sim episode directory: {resolved_episode_dir}")

        task_json_path = resolved_episode_dir / "task.json"
        if not task_json_path.is_file():
            raise FileNotFoundError(f"Could not find RoboFail sim task.json: {task_json_path}")
        task_payload = _read_json_object(task_json_path)

        ego_img_dir = resolved_episode_dir / "ego_img"
        frames = _sim_camera_frames(ego_img_dir)
        duration_sec = _sim_duration_sec(resolved_episode_dir / "original-video.mp4", frames)
        action_records, action_metadata = _sim_action_label_records(resolved_episode_dir)
        streams = []
        if action_records:
            numeric_action_records, action_stream_metadata = _numeric_action_records(action_records)
            streams.append(
                ModalityStream(
                    name="action",
                    modality=Modality.ACTION,
                    timestamps_sec=[
                        float(record["timestamp_sec"])
                        for record in numeric_action_records
                    ],
                    records=numeric_action_records,
                    metadata={
                        "dataset": "RoboFail",
                        "split": "sim",
                        "description": "Numeric action-label IDs derived from RoboFail action pickles.",
                        **action_stream_metadata,
                    },
                )
            )
            streams.append(
                ModalityStream(
                    name="action_labels",
                    modality=Modality.METADATA,
                    timestamps_sec=[float(record["timestamp_sec"]) for record in action_records],
                    records=action_records,
                    metadata={
                        "dataset": "RoboFail",
                        "split": "sim",
                        "description": "Human-readable action labels from RoboFail pickles.",
                        **action_metadata,
                    },
                )
            )

        raw_metadata = {
            "dataset": "RoboFail",
            "split": "sim",
            "episode_dir": str(resolved_episode_dir),
            "task_json_path": str(task_json_path),
            "task": task_payload,
            "task_name": task_payload.get("task_name") or task_payload.get("task"),
            "gt_failure_step": task_payload.get("gt_failure_step"),
            "frame_count": len(frames),
            "ego_img_dir": str(ego_img_dir),
            "original_video_path": str(resolved_episode_dir / "original-video.mp4"),
            "action_label_metadata": action_metadata,
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_episode_dir.name,
            source_format=BundleFormat.ROBOFAIL_SIM,
            root_path=resolved_episode_dir,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=frames,
            raw_metadata=raw_metadata,
        )


class RoboFailRealBundleAdapter:
    """Load one RoboFail real-world episode into the internal trajectory schema."""

    def load(
        self,
        episode_dir: str | Path,
        tasks_metadata_path: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        extract_frames: bool = True,
    ) -> TrajectoryBundle:
        resolved_episode_dir = Path(episode_dir).expanduser().resolve()
        if not resolved_episode_dir.exists() or not resolved_episode_dir.is_dir():
            raise FileNotFoundError(f"Could not find RoboFail real episode directory: {resolved_episode_dir}")

        task_map = load_real_world_task_map(tasks_metadata_path)
        metadata = task_map.get(resolved_episode_dir.name)
        if metadata is None:
            raise KeyError(
                "Could not map RoboFail real episode to tasks_real_world.json: "
                f"{resolved_episode_dir.name}"
            )

        zarr_root = resolved_episode_dir / "replay_buffer.zarr"
        if not zarr_root.exists():
            raise FileNotFoundError(f"Could not find RoboFail real replay buffer: {zarr_root}")
        if not (zarr_root / ".zgroup").is_file():
            raise FileNotFoundError(f"RoboFail real replay buffer is missing .zgroup: {zarr_root}")
        _import_zarr()
        data_group = _read_local_zarr_v2_data_group(zarr_root)

        timestamps = _read_zarr_timestamps(data_group)
        streams, array_shapes = _real_streams(data_group, timestamps)

        video_path = resolved_episode_dir / "videos" / "color.mp4"
        video_metadata = _video_metadata(video_path) if video_path.is_file() else {}
        cache_dir = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else resolved_episode_dir / "pathologist_frame_cache"
        )
        frames = (
            _extract_video_frames(video_path, cache_dir / "images" / "color", "color")
            if extract_frames and video_path.is_file()
            else []
        )
        duration_sec = _real_duration_sec(timestamps, video_metadata, frames)

        raw_metadata = {
            "dataset": "RoboFail",
            "split": "real",
            "episode_dir": str(resolved_episode_dir),
            "tasks_metadata_path": str(Path(tasks_metadata_path).expanduser().resolve()),
            "task_metadata": metadata.payload,
            "task_id": metadata.task_id,
            "general_folder_name": metadata.folder_name,
            "gt_failure_step": metadata.payload.get("gt_failure_step"),
            "zarr_path": str(zarr_root),
            "video_path": str(video_path),
            "video_metadata": video_metadata,
            "array_shapes": array_shapes,
            "frame_cache_dir": str(cache_dir),
            "decoded_frame_count": len(frames),
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_episode_dir.name,
            source_format=BundleFormat.ROBOFAIL_REAL,
            root_path=resolved_episode_dir,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=frames,
            raw_metadata=raw_metadata,
        )


def load_real_world_task_map(tasks_metadata_path: str | Path) -> dict[str, RoboFailRealTaskMetadata]:
    """Load tasks_real_world.json keyed by `general_folder_name`."""

    resolved_path = Path(tasks_metadata_path).expanduser().resolve()
    payload = _read_json_object_or_list(resolved_path)
    keyed_rows: list[tuple[str | None, dict[str, Any]]] = []
    if isinstance(payload, dict):
        rows = None
        for row_key in ("tasks", "data", "episodes"):
            if row_key in payload:
                rows = payload[row_key]
                break
        if rows is None:
            keyed_rows = [
                (str(key), value)
                for key, value in payload.items()
                if isinstance(value, dict)
            ]
        elif isinstance(rows, dict):
            keyed_rows = [
                (str(key), value)
                for key, value in rows.items()
                if isinstance(value, dict)
            ]
        elif isinstance(rows, list):
            keyed_rows = [
                (None, value)
                for value in rows
                if isinstance(value, dict)
            ]
    else:
        if isinstance(payload, list):
            keyed_rows = [
                (None, value)
                for value in payload
                if isinstance(value, dict)
            ]
    if not keyed_rows:
        raise ValueError(f"Expected a list of task entries in {resolved_path}")

    task_map: dict[str, RoboFailRealTaskMetadata] = {}
    for index, (metadata_key, row) in enumerate(keyed_rows):
        folder_name = row.get("general_folder_name") or row.get("folder_name") or row.get("episode")
        if not isinstance(folder_name, str) or not folder_name.strip():
            continue
        task_id = row.get("task_id", row.get("id", metadata_key or index))
        task_map[folder_name.strip()] = RoboFailRealTaskMetadata(
            folder_name=folder_name.strip(),
            task_id=str(task_id) if task_id is not None else None,
            payload=row,
        )
    return task_map


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json_object_or_list(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _read_json_object_or_list(path: Path) -> dict[str, Any] | list[Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"Expected JSON object or list at {path}")
    return payload


def _sim_camera_frames(ego_img_dir: Path) -> list[CameraFrameRef]:
    if not ego_img_dir.is_dir():
        return []
    frames: list[CameraFrameRef] = []
    for image_path in sorted(ego_img_dir.iterdir(), key=_natural_sort_key):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        step = _step_from_image_name(image_path)
        if step is None:
            continue
        frames.append(
            CameraFrameRef(
                camera_name="ego",
                timestamp_sec=float(step),
                image_path=image_path,
                frame_index=int(step),
            )
        )
    frames.sort(key=lambda frame: (frame.timestamp_sec, frame.frame_index or -1))
    return frames


def _sim_duration_sec(video_path: Path, frames: list[CameraFrameRef]) -> float | None:
    frame_duration = max((float(frame.timestamp_sec) for frame in frames), default=0.0)
    if not video_path.is_file():
        return frame_duration if frames else None
    try:
        metadata = _video_metadata(video_path)
    except Exception:
        return frame_duration if frames else None
    video_duration = _duration_from_video_metadata(metadata)
    if video_duration is None:
        return frame_duration if frames else None
    return max(video_duration, frame_duration)


def _sim_action_label_records(episode_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for filename, source in (
        ("interact_actions.pickle", "interact"),
        ("nav_actions.pickle", "nav"),
    ):
        path = episode_dir / filename
        loaded, error = _safe_pickle_load(path)
        metadata[f"{source}_actions_path"] = str(path)
        metadata[f"{source}_actions_loaded"] = error is None and loaded is not None
        if error is not None:
            metadata[f"{source}_actions_error"] = error
            continue
        source_records = _action_records_from_pickle_payload(loaded, source)
        metadata[f"{source}_action_label_count"] = len(source_records)
        records.extend(source_records)
    records.sort(key=lambda record: (float(record["timestamp_sec"]), str(record.get("action_source"))))
    for index, record in enumerate(records):
        record["sample_index"] = index
    return records, metadata


def _safe_pickle_load(path: Path) -> tuple[Any | None, str | None]:
    if not path.is_file():
        return None, "file_not_found"
    try:
        with path.open("rb") as handle:
            return pickle.load(handle), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _action_records_from_pickle_payload(payload: Any, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return records
    for raw_key, raw_label in payload.items():
        label = str(raw_label)
        if isinstance(raw_key, (int, float, bool)):
            start_step = int(raw_key)
            end_step = start_step
        elif isinstance(raw_key, (tuple, list)) and raw_key:
            numeric_parts = [part for part in raw_key if isinstance(part, (int, float, bool))]
            if not numeric_parts:
                continue
            start_step = int(numeric_parts[0])
            end_step = int(numeric_parts[-1])
        else:
            continue
        records.append(
            {
                "timestamp_sec": float(start_step),
                "start_step": start_step,
                "end_step": end_step,
                "action_label": label,
                "action_source": source,
            }
        )
    return records


def _numeric_action_records(action_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = sorted({str(record.get("action_label", "")) for record in action_records})
    label_to_id = {label: index for index, label in enumerate(labels)}
    records = []
    for index, record in enumerate(action_records):
        label = str(record.get("action_label", ""))
        action_id = float(label_to_id[label])
        numeric_record = {
            **record,
            "sample_index": index,
            "action": [action_id],
            "action_label_id": int(action_id),
        }
        records.append(numeric_record)
    high = max(float(len(label_to_id) - 1), 0.0)
    return records, {
        "action_label_to_id": label_to_id,
        "action_low": [0.0],
        "action_high": [high],
    }


def _import_zarr() -> Any:
    try:
        import numcodecs  # noqa: F401
        import zarr  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RoboFail real adapter requires zarr and numcodecs. "
            "Install them in the active environment, for example: conda install -c conda-forge zarr numcodecs"
        ) from exc
    return zarr


def _read_local_zarr_v2_data_group(zarr_root: Path) -> dict[str, Any]:
    """Read local one-group Zarr v2 arrays without opening the store through zarr.

    RoboFail real replay buffers are plain local Zarr v2 directories. Some Zarr 3
    versions are extremely slow to open these stores through ``zarr.open_group`` in
    this environment, while decoding the chunk files directly is fast and stable.
    """

    data_root = zarr_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"RoboFail real replay buffer is missing data group: {data_root}")
    arrays: dict[str, Any] = {}
    for array_dir in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not array_dir.is_dir() or not (array_dir / ".zarray").is_file():
            continue
        arrays[array_dir.name] = _read_local_zarr_v2_array(array_dir)
    return arrays


def _read_local_zarr_v2_array(array_dir: Path) -> Any:
    import numpy as np
    from numcodecs import get_codec

    metadata = json.loads((array_dir / ".zarray").read_text(encoding="utf-8"))
    if int(metadata.get("zarr_format", 2)) != 2:
        raise ValueError(f"Expected Zarr v2 array at {array_dir}")
    shape = tuple(int(value) for value in metadata.get("shape", []))
    chunks = tuple(int(value) for value in metadata.get("chunks", []))
    if not shape or not chunks or len(shape) != len(chunks):
        raise ValueError(f"Invalid Zarr array metadata at {array_dir}")
    dtype = np.dtype(metadata["dtype"])
    order = str(metadata.get("order", "C"))
    result = np.full(shape, metadata.get("fill_value", 0), dtype=dtype, order=order)
    compressor_config = metadata.get("compressor")
    compressor = get_codec(compressor_config) if compressor_config is not None else None
    chunk_counts = [
        (size + chunk_size - 1) // chunk_size
        for size, chunk_size in zip(shape, chunks)
    ]
    for chunk_index in _chunk_indices(chunk_counts):
        chunk_path = array_dir / ".".join(str(index) for index in chunk_index)
        if not chunk_path.is_file():
            continue
        raw = chunk_path.read_bytes()
        if compressor is not None:
            raw = compressor.decode(raw)
        nominal_chunk_shape = tuple(
            min(chunk_size, size - index * chunk_size)
            for index, chunk_size, size in zip(chunk_index, chunks, shape)
        )
        values = np.frombuffer(raw, dtype=dtype)
        full_chunk_size = 1
        for chunk_size in chunks:
            full_chunk_size *= chunk_size
        chunk_shape = chunks if values.size == full_chunk_size else nominal_chunk_shape
        chunk = values.reshape(chunk_shape, order=order)
        cropped_chunk = chunk[
            tuple(slice(0, size) for size in nominal_chunk_shape)
        ]
        slices = tuple(
            slice(index * chunk_size, index * chunk_size + cropped_chunk.shape[axis])
            for axis, (index, chunk_size) in enumerate(zip(chunk_index, chunks))
        )
        result[slices] = cropped_chunk
    return result


def _chunk_indices(chunk_counts: list[int]) -> list[tuple[int, ...]]:
    if not chunk_counts:
        return []
    indices: list[tuple[int, ...]] = [()]
    for count in chunk_counts:
        indices = [prefix + (index,) for prefix in indices for index in range(count)]
    return indices


def _read_zarr_timestamps(data_group: Any) -> list[float]:
    if "timestamp" not in data_group:
        raise KeyError("RoboFail real replay buffer is missing data/timestamp")
    raw = data_group["timestamp"][:]
    values = _array_to_float_lists(raw)
    flat = [float(item[0] if isinstance(item, list) else item) for item in values]
    if not flat:
        return []
    origin = flat[0]
    return [max(0.0, float(value) - origin) for value in flat]


def _real_streams(data_group: Any, timestamps: list[float]) -> tuple[list[ModalityStream], dict[str, Any]]:
    streams: list[ModalityStream] = []
    array_shapes: dict[str, Any] = {}

    def add_stream(
        *,
        array_name: str,
        stream_name: str,
        modality: Modality,
        record_builder: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if array_name not in data_group:
            return
        raw = data_group[array_name][:]
        array_shapes[array_name] = list(getattr(raw, "shape", ()))
        values = _array_to_float_lists(raw)
        count = min(len(timestamps), len(values))
        records = [record_builder(index, timestamps[index], values[index]) for index in range(count)]
        streams.append(
            ModalityStream(
                name=stream_name,
                modality=modality,
                timestamps_sec=[float(timestamps[index]) for index in range(count)],
                records=records,
                metadata={
                    "dataset": "RoboFail",
                    "split": "real",
                    "source_dataset": f"data/{array_name}",
                    **(metadata or {}),
                },
            )
        )

    add_stream(
        array_name="robot_joint",
        stream_name="proprioception",
        modality=Modality.PROPRIOCEPTION,
        record_builder=lambda index, timestamp, value: {
            "timestamp_sec": timestamp,
            "sample_index": index,
            "proprioception": _as_vector(value),
            "joint_positions": _as_vector(value),
            "state": _as_vector(value),
        },
    )
    add_stream(
        array_name="robot_joint_vel",
        stream_name="velocity",
        modality=Modality.PROPRIOCEPTION,
        record_builder=lambda index, timestamp, value: {
            "timestamp_sec": timestamp,
            "sample_index": index,
            "qvel": _as_vector(value),
            "velocity": _as_vector(value),
            "joint_velocity": _as_vector(value),
            "joint_velocities": _as_vector(value),
        },
    )
    add_stream(
        array_name="robot_eef_pose",
        stream_name="end_effector_pose",
        modality=Modality.TELEMETRY,
        record_builder=lambda index, timestamp, value: {
            "timestamp_sec": timestamp,
            "sample_index": index,
            "pose": _as_vector(value),
            "ee_pose": _as_vector(value),
            "wrist_poses": {"eef": _pose_to_wrist_record(_as_vector(value))},
        },
    )
    add_stream(
        array_name="robot_eef_pose_vel",
        stream_name="end_effector_velocity",
        modality=Modality.TELEMETRY,
        record_builder=lambda index, timestamp, value: {
            "timestamp_sec": timestamp,
            "sample_index": index,
            "velocity": _as_vector(value),
            "ee_velocity": _as_vector(value),
        },
    )
    add_stream(
        array_name="action",
        stream_name="action",
        modality=Modality.ACTION,
        record_builder=lambda index, timestamp, value: {
            "timestamp_sec": timestamp,
            "sample_index": index,
            "action": _as_vector(value),
        },
    )

    gripper_names = ["gripper_pos", "gripper_state", "gripper_force"]
    if any(name in data_group for name in gripper_names):
        gripper_values = {
            name: _array_to_float_lists(data_group[name][:])
            for name in gripper_names
            if name in data_group
        }
        for name, values in gripper_values.items():
            array_shapes[name] = [len(values)] if not hasattr(data_group[name], "shape") else list(data_group[name].shape)
        count = min([len(timestamps), *[len(values) for values in gripper_values.values()]])
        records = []
        for index in range(count):
            record: dict[str, Any] = {"timestamp_sec": timestamps[index], "sample_index": index}
            if "gripper_pos" in gripper_values:
                record["gripper"] = _as_vector(gripper_values["gripper_pos"][index])
                record["gripper_pos"] = _as_vector(gripper_values["gripper_pos"][index])
            if "gripper_state" in gripper_values:
                record["gripper_state"] = _as_vector(gripper_values["gripper_state"][index])
            if "gripper_force" in gripper_values:
                record["gripper_force"] = _as_vector(gripper_values["gripper_force"][index])
            records.append(record)
        streams.append(
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=[float(timestamps[index]) for index in range(count)],
                records=records,
                metadata={
                    "dataset": "RoboFail",
                    "split": "real",
                    "source_datasets": [name for name in gripper_names if name in data_group],
                },
            )
        )

    if "timestamp" in data_group:
        array_shapes["timestamp"] = list(getattr(data_group["timestamp"], "shape", ()))
    return streams, array_shapes


def _array_to_float_lists(array: Any) -> list[Any]:
    try:
        values = array.tolist()
    except AttributeError:
        values = list(array)
    if not isinstance(values, list):
        return [float(values)]
    return values


def _as_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def _pose_to_wrist_record(pose: list[float]) -> dict[str, Any]:
    return {
        "position": pose[:3],
        "orientation_xyzw_or_wxyz": pose[3:],
    }


def _real_duration_sec(
    timestamps: list[float],
    video_metadata: dict[str, Any],
    frames: list[CameraFrameRef],
) -> float | None:
    candidates = []
    if timestamps:
        candidates.append(max(timestamps))
    video_duration = _duration_from_video_metadata(video_metadata)
    if video_duration is not None:
        candidates.append(video_duration)
    if frames:
        candidates.append(max(float(frame.timestamp_sec) for frame in frames))
    return max(candidates) if candidates else None


def _video_metadata(video_path: Path) -> dict[str, int | float]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height}


def _duration_from_video_metadata(metadata: dict[str, Any]) -> float | None:
    frame_count = float(metadata.get("frame_count") or 0.0)
    fps = float(metadata.get("fps") or 0.0)
    if frame_count <= 0 or fps <= 0:
        return None
    return frame_count / fps


def _extract_video_frames(video_path: Path, frame_cache_dir: Path, camera_name: str) -> list[CameraFrameRef]:
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = _video_metadata(video_path)
    fps = float(metadata.get("fps") or 0.0)
    if fps <= 0:
        raise RuntimeError(f"Could not determine FPS for video: {video_path}")
    cached_frames = _frames_from_image_dir(frame_cache_dir, camera_name, fps)
    expected_count = int(metadata.get("frame_count") or 0)
    if expected_count > 0 and len(cached_frames) >= expected_count:
        return cached_frames[:expected_count]

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: list[CameraFrameRef] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        timestamp = float(index) / fps
        image_path = frame_cache_dir / f"{index:06d}_t{timestamp:08.3f}.jpg"
        if not image_path.exists() and not cv2.imwrite(str(image_path), image):
            capture.release()
            raise RuntimeError(f"Could not write RoboFail frame: {image_path}")
        frames.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=index,
            )
        )
        index += 1
    capture.release()
    return frames


def _frames_from_image_dir(frame_dir: Path, camera_name: str, fps: float) -> list[CameraFrameRef]:
    if not frame_dir.exists():
        return []
    frames = []
    for index, image_path in enumerate(sorted(frame_dir.iterdir(), key=_natural_sort_key)):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        timestamp = _timestamp_from_filename(image_path)
        if timestamp is None:
            timestamp = float(index) / fps
        frames.append(
            CameraFrameRef(
                camera_name=camera_name,
                timestamp_sec=timestamp,
                image_path=image_path,
                frame_index=_leading_int(image_path.stem) if _leading_int(image_path.stem) is not None else index,
            )
        )
    return frames


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RoboFail adapters require opencv-python/cv2 for video metadata or extraction.") from exc
    return cv2


def _step_from_image_name(path: Path) -> int | None:
    match = re.search(r"img_step_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _timestamp_from_filename(path: Path) -> float | None:
    match = re.search(r"_t([0-9]+(?:\.[0-9]+)?)", path.stem)
    return float(match.group(1)) if match else None


def _leading_int(value: str) -> int | None:
    match = re.match(r"(\d+)", value)
    return int(match.group(1)) if match else None


def _natural_sort_key(path: Path) -> list[int | str]:
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", path.name):
        if part.isdigit():
            parts.append(int(part))
        elif part:
            parts.append(part)
    return parts
