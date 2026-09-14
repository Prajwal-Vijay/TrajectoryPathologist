"""Read-only helpers for the REASSEMBLE results viewer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


SELECTOR_ORDER = ("uniform", "kite_flow", "metric")
CAMERA_ORDER = ("hand", "hama1", "hama2")


@dataclass(slots=True)
class CameraFrameImage:
    camera_name: str
    timestamp_sec: float
    image_path: Path
    frame_index: int | None = None


@dataclass(slots=True)
class NearestFrame:
    frame: CameraFrameImage
    target_timestamp_sec: float

    @property
    def delta_sec(self) -> float:
        return self.frame.timestamp_sec - self.target_timestamp_sec


@dataclass(slots=True)
class ViewerBundle:
    bundle_id: str
    h5_path: str | None
    source_file_bundle_id: str
    high_level_task_text: str | None
    high_level_task_kind: str | None
    duration_sec: float | None
    failed_low_level_coverage_fraction: float | None
    manifest_entry: dict[str, Any]
    rows_by_selector: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def failed_segments(self) -> dict[str, list[dict[str, Any]]]:
        for row in self.rows_by_selector.values():
            value = row.get("failed_segments")
            if isinstance(value, dict):
                return {
                    "high": _dict_list(value.get("high")),
                    "low": _dict_list(value.get("low")),
                    "combined": _dict_list(value.get("combined")),
                }
        return {"high": [], "low": [], "combined": []}

    @property
    def selectors(self) -> list[str]:
        known = [selector for selector in SELECTOR_ORDER if selector in self.rows_by_selector]
        extra = sorted(selector for selector in self.rows_by_selector if selector not in SELECTOR_ORDER)
        return known + extra

    @property
    def peaks_by_selector(self) -> dict[str, list[dict[str, Any]]]:
        return {
            selector: _dict_list(row.get("selected_peaks"))
            for selector, row in self.rows_by_selector.items()
        }


@dataclass(frozen=True, slots=True)
class GripperTransitionMark:
    timestamp_sec: float
    label: str
    state: str
    previous_state: str
    raw_value: float


@dataclass(frozen=True, slots=True)
class GripperTransitionOverlay:
    marks: list[GripperTransitionMark]
    sample_count: int
    threshold: float | None
    value_min: float | None
    value_max: float | None
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetrySeries:
    name: str
    timestamps_sec: list[float]
    values_by_component: dict[str, list[float]]
    sample_count: int
    source_dataset: str
    unavailable_reason: str | None = None


@dataclass(slots=True)
class ViewerDataset:
    output_dir: Path
    bundles: list[ViewerBundle]
    manifest: list[dict[str, Any]]


def load_viewer_dataset(output_dir: str | Path) -> ViewerDataset:
    """Load manifest/eval outputs into trajectory records for the viewer."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = resolved_output_dir / "reassemble_manifest.json"
    eval_path = resolved_output_dir / "reassemble_eval.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing REASSEMBLE manifest: {manifest_path}")
    if not eval_path.is_file():
        raise FileNotFoundError(f"Missing REASSEMBLE eval JSONL: {eval_path}")

    manifest = _read_json_list(manifest_path)
    rows_by_bundle: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(eval_path):
        bundle_id = row.get("bundle_id")
        selector_name = row.get("selector_name")
        if not isinstance(bundle_id, str) or not isinstance(selector_name, str):
            continue
        rows_by_bundle.setdefault(bundle_id, {})[selector_name] = row

    manifest_by_bundle = {
        entry["bundle_id"]: entry
        for entry in manifest
        if isinstance(entry, dict)
        and isinstance(entry.get("bundle_id"), str)
        and entry.get("included") is True
    }
    bundles = [
        _viewer_bundle_from_manifest(entry, rows_by_bundle.get(bundle_id, {}))
        for bundle_id, entry in manifest_by_bundle.items()
        if rows_by_bundle.get(bundle_id)
    ]
    bundles.sort(key=lambda bundle: (bundle.h5_path or "", bundle.bundle_id))
    return ViewerDataset(output_dir=resolved_output_dir, bundles=bundles, manifest=manifest)


def load_ecod_aggregate(output_dir: str | Path, bundle_id: str) -> dict[str, Any] | None:
    aggregate_path = Path(output_dir).expanduser().resolve() / "ecod_aggregates" / f"{bundle_id}.json"
    if not aggregate_path.is_file():
        return None
    value = json.loads(aggregate_path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def build_camera_frame_index(
    output_dir: str | Path,
    source_file_bundle_id: str,
) -> dict[str, list[CameraFrameImage]]:
    image_root = Path(output_dir).expanduser().resolve() / "frame_cache" / source_file_bundle_id / "images"
    if not image_root.is_dir():
        return {}

    result: dict[str, list[CameraFrameImage]] = {}
    for camera_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        frames: list[CameraFrameImage] = []
        for image_path in sorted(camera_dir.iterdir(), key=_natural_sort_key):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            timestamp = timestamp_from_frame_path(image_path)
            if timestamp is None:
                continue
            frames.append(
                CameraFrameImage(
                    camera_name=camera_dir.name,
                    timestamp_sec=timestamp,
                    image_path=image_path,
                    frame_index=_leading_int(image_path.stem),
                )
            )
        if frames:
            result[camera_dir.name] = frames
    return result


def infer_full_timeline_offset_sec(bundle: ViewerBundle) -> float | None:
    """Infer full-H5 timestamp offset from selected frame paths.

    Evaluation rows store peak seconds relative to the cropped high-level action,
    while cached frame filenames store the full source H5 timestamp. Metric peaks
    do not always carry image paths, so this offset lets the viewer locate frames
    for metric peaks across every decoded camera.
    """

    offsets: list[float] = []
    for peaks in bundle.peaks_by_selector.values():
        for peak in peaks:
            peak_sec = _as_float(peak.get("peak_sec"))
            image_path_value = peak.get("image_path")
            if peak_sec is None or not isinstance(image_path_value, str):
                continue
            frame_sec = timestamp_from_frame_path(Path(image_path_value))
            if frame_sec is not None:
                offsets.append(frame_sec - peak_sec)
    return float(median(offsets)) if offsets else None


def nearest_frame(
    frame_index: dict[str, list[CameraFrameImage]],
    camera_name: str,
    target_timestamp_sec: float,
) -> NearestFrame | None:
    frames = frame_index.get(camera_name)
    if not frames:
        return None
    closest = min(frames, key=lambda frame: abs(frame.timestamp_sec - target_timestamp_sec))
    return NearestFrame(frame=closest, target_timestamp_sec=target_timestamp_sec)


def frames_in_time_window(
    frame_index: dict[str, list[CameraFrameImage]],
    camera_name: str,
    start_sec: float,
    end_sec: float,
) -> list[CameraFrameImage]:
    frames = frame_index.get(camera_name, [])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return [
        frame
        for frame in frames
        if start_sec <= frame.timestamp_sec <= end_sec
    ]


def sample_frames_evenly(
    frames: list[CameraFrameImage],
    max_count: int,
) -> list[CameraFrameImage]:
    if max_count <= 0 or not frames:
        return []
    if len(frames) <= max_count:
        return list(frames)
    if max_count == 1:
        return [frames[len(frames) // 2]]
    indexes = [
        round(index * (len(frames) - 1) / (max_count - 1))
        for index in range(max_count)
    ]
    sampled: list[CameraFrameImage] = []
    seen: set[int] = set()
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        sampled.append(frames[index])
    return sampled


def flatten_peaks(
    bundle: ViewerBundle,
    visible_selectors: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    visible = set(visible_selectors) if visible_selectors is not None else set(bundle.selectors)
    selector_order = {selector: index for index, selector in enumerate(bundle.selectors)}
    peaks: list[dict[str, Any]] = []
    for selector in bundle.selectors:
        if selector not in visible:
            continue
        for peak in bundle.peaks_by_selector.get(selector, []):
            item = dict(peak)
            item["selector_name"] = selector
            peaks.append(item)
    peaks.sort(
        key=lambda peak: (
            selector_order.get(str(peak.get("selector_name")), len(selector_order)),
            int(peak.get("timeline_rank") or 0),
            float(peak.get("peak_sec") or 0.0),
        )
    )
    return peaks


def aggregate_score_at_time(aggregate: dict[str, Any] | None, timestamp_sec: float) -> float | None:
    if aggregate is None:
        return None
    timestamps = _float_list(aggregate.get("timestamps_sec"))
    scores = _float_list(aggregate.get("smoothed_scores")) or _float_list(aggregate.get("raw_scores"))
    if len(timestamps) != len(scores) or not timestamps:
        return None
    if timestamp_sec <= timestamps[0]:
        return scores[0]
    if timestamp_sec >= timestamps[-1]:
        return scores[-1]
    for index in range(1, len(timestamps)):
        left_time = timestamps[index - 1]
        right_time = timestamps[index]
        if left_time <= timestamp_sec <= right_time:
            left_score = scores[index - 1]
            right_score = scores[index]
            width = right_time - left_time
            if width <= 0:
                return right_score
            alpha = (timestamp_sec - left_time) / width
            return left_score + alpha * (right_score - left_score)
    return None


def load_gripper_transition_marks(
    bundle: ViewerBundle,
    *,
    offset_sec: float | None,
    threshold: float | None = None,
    min_range: float = 0.005,
) -> list[GripperTransitionMark]:
    return load_gripper_transition_overlay(
        bundle,
        offset_sec=offset_sec,
        threshold=threshold,
        min_range=min_range,
    ).marks


def load_gripper_transition_overlay(
    bundle: ViewerBundle,
    *,
    offset_sec: float | None,
    threshold: float | None = None,
    min_range: float = 0.005,
) -> GripperTransitionOverlay:
    """Load gripper open/close transitions for the selected high-level segment.

    This is intentionally a viewer-only helper. It reads only the timestamp and
    gripper-position arrays from the source REASSEMBLE H5 file and returns marks
    relative to the current high-level bundle.
    """

    if offset_sec is None or not bundle.h5_path or bundle.duration_sec is None:
        return GripperTransitionOverlay(
            marks=[],
            sample_count=0,
            threshold=threshold,
            value_min=None,
            value_max=None,
            unavailable_reason="missing high-level offset, H5 path, or duration",
        )
    h5_path = Path(bundle.h5_path).expanduser()
    if not h5_path.is_file():
        return GripperTransitionOverlay(
            marks=[],
            sample_count=0,
            threshold=threshold,
            value_min=None,
            value_max=None,
            unavailable_reason=f"source H5 not found: {h5_path}",
        )
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to show REASSEMBLE gripper marks.") from exc

    with h5py.File(h5_path, "r") as handle:
        value_path = "robot_state/gripper_positions"
        timestamp_path = "timestamps/gripper_positions"
        if value_path not in handle or timestamp_path not in handle:
            return GripperTransitionOverlay(
                marks=[],
                sample_count=0,
                threshold=threshold,
                value_min=None,
                value_max=None,
                unavailable_reason="robot_state/gripper_positions or timestamps/gripper_positions missing",
            )
        timestamp_origin = _h5_global_timestamp_origin(handle)
        timestamps = [
            float(value) - timestamp_origin
            for value in handle[timestamp_path][:].tolist()
        ]
        values = handle[value_path][:]
        scalar_values = [_row_mean(value) for value in values]

    count = min(len(timestamps), len(scalar_values))
    if count < 2:
        return GripperTransitionOverlay(
            marks=[],
            sample_count=count,
            threshold=threshold,
            value_min=None,
            value_max=None,
            unavailable_reason="requires at least two gripper samples",
        )
    start_sec = float(offset_sec)
    end_sec = start_sec + float(bundle.duration_sec)
    window_timestamps: list[float] = []
    window_values: list[float] = []
    for timestamp, value in zip(timestamps[:count], scalar_values[:count]):
        if start_sec <= timestamp <= end_sec:
            window_timestamps.append(timestamp - start_sec)
            window_values.append(value)
    if len(window_values) < 2:
        return GripperTransitionOverlay(
            marks=[],
            sample_count=len(window_values),
            threshold=threshold,
            value_min=None,
            value_max=None,
            unavailable_reason="no gripper samples inside high-level segment",
        )
    value_min = min(window_values)
    value_max = max(window_values)
    value_range = value_max - value_min
    if value_range < max(0.0, float(min_range)):
        return GripperTransitionOverlay(
            marks=[],
            sample_count=len(window_values),
            threshold=threshold,
            value_min=value_min,
            value_max=value_max,
            unavailable_reason=f"gripper range {value_range:.6g} is below min range {float(min_range):.6g}",
        )
    effective_threshold = (
        float(threshold)
        if threshold is not None
        else _adaptive_gripper_threshold(window_values)
    )
    marks = _gripper_transitions_from_series(
        window_timestamps,
        window_values,
        threshold=effective_threshold,
        hysteresis=0.15 * value_range,
    )
    return GripperTransitionOverlay(
        marks=marks,
        sample_count=len(window_values),
        threshold=effective_threshold,
        value_min=value_min,
        value_max=value_max,
        unavailable_reason=None,
    )


def load_reassemble_telemetry_series(
    bundle: ViewerBundle,
    *,
    offset_sec: float | None,
    max_samples: int = 3000,
) -> list[TelemetrySeries]:
    """Load raw REASSEMBLE robot-state streams for the current high-level segment."""

    if not bundle.h5_path or bundle.duration_sec is None:
        return [
            TelemetrySeries(
                name="unavailable",
                timestamps_sec=[],
                values_by_component={},
                sample_count=0,
                source_dataset="",
                unavailable_reason="missing H5 path or duration",
            )
        ]
    start_sec = _high_level_start_sec(bundle, offset_sec)
    if start_sec is None:
        return [
            TelemetrySeries(
                name="unavailable",
                timestamps_sec=[],
                values_by_component={},
                sample_count=0,
                source_dataset="",
                unavailable_reason="missing high-level start timestamp",
            )
        ]
    h5_path = Path(bundle.h5_path).expanduser()
    return load_reassemble_telemetry_series_from_h5(
        h5_path=h5_path,
        start_sec=start_sec,
        duration_sec=float(bundle.duration_sec),
        max_samples=max_samples,
    )


def load_reassemble_telemetry_series_from_h5(
    *,
    h5_path: str | Path,
    start_sec: float,
    duration_sec: float,
    max_samples: int = 3000,
) -> list[TelemetrySeries]:
    resolved_h5_path = Path(h5_path).expanduser()
    if not resolved_h5_path.is_file():
        return [
            TelemetrySeries(
                name="unavailable",
                timestamps_sec=[],
                values_by_component={},
                sample_count=0,
                source_dataset="",
                unavailable_reason=f"source H5 not found: {resolved_h5_path}",
            )
        ]
    try:
        import h5py  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to show REASSEMBLE telemetry streams.") from exc

    end_sec = float(start_sec) + float(duration_sec)
    max_samples = max(100, int(max_samples))
    with h5py.File(resolved_h5_path, "r") as handle:
        robot_state = handle.get("robot_state")
        timestamps_group = handle.get("timestamps")
        if robot_state is None or timestamps_group is None:
            return [
                TelemetrySeries(
                    name="unavailable",
                    timestamps_sec=[],
                    values_by_component={},
                    sample_count=0,
                    source_dataset="",
                    unavailable_reason="robot_state or timestamps group missing",
                )
            ]
        timestamp_origin = _h5_global_timestamp_origin(handle)
        output: list[TelemetrySeries] = []
        for name in sorted(robot_state.keys()):
            if name not in timestamps_group:
                continue
            dataset = robot_state[name]
            timestamp_dataset = timestamps_group[name]
            if not getattr(dataset, "shape", ()) or not getattr(timestamp_dataset, "shape", ()):
                continue
            timestamps = [
                float(value) - timestamp_origin
                for value in timestamp_dataset[:].tolist()
            ]
            values = dataset[:]
            count = min(len(timestamps), int(values.shape[0]) if getattr(values, "shape", ()) else len(values))
            indices = [
                index
                for index, timestamp in enumerate(timestamps[:count])
                if float(start_sec) <= timestamp <= end_sec
            ]
            if not indices:
                continue
            indices = _downsample_index_list(indices, max_samples)
            window_timestamps = [timestamps[index] - float(start_sec) for index in indices]
            values_by_component = _telemetry_components(name, values, indices)
            if values_by_component:
                output.append(
                    TelemetrySeries(
                        name=name,
                        timestamps_sec=window_timestamps,
                        values_by_component=values_by_component,
                        sample_count=len(indices),
                        source_dataset=f"robot_state/{name}",
                    )
                )
        return output


def _high_level_start_sec(bundle: ViewerBundle, offset_sec: float | None) -> float | None:
    if offset_sec is not None:
        return float(offset_sec)
    manifest_start = bundle.manifest_entry.get("high_level_segment_start_sec_in_file")
    return _as_float(manifest_start)


def _downsample_index_list(indices: list[int], max_samples: int) -> list[int]:
    if len(indices) <= max_samples:
        return indices
    if max_samples <= 1:
        return [indices[0]]
    step = (len(indices) - 1) / float(max_samples - 1)
    selected: list[int] = []
    previous: int | None = None
    for output_index in range(max_samples):
        index = indices[int(round(output_index * step))]
        if index != previous:
            selected.append(index)
            previous = index
    return selected


def _telemetry_components(name: str, values: Any, indices: list[int]) -> dict[str, list[float]]:
    component_names = _telemetry_component_names(name, values)
    rows: list[list[float]] = []
    for index in indices:
        row = values[index]
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, list):
            numeric_row = [float(item) for item in row if isinstance(item, (int, float))]
        else:
            numeric_row = [float(row)]
        if numeric_row:
            rows.append(numeric_row)
    if not rows:
        return {}

    width = min(len(component_names), min(len(row) for row in rows))
    components = {
        component_names[index]: [row[index] for row in rows]
        for index in range(width)
    }
    if width >= 2:
        components["magnitude"] = [
            sum(value * value for value in row[:width]) ** 0.5
            for row in rows
        ]
    return components


def _telemetry_component_names(name: str, values: Any) -> list[str]:
    width = 1
    shape = getattr(values, "shape", ())
    if len(shape) >= 2:
        width = int(shape[1])
    known = {
        "compensated_base_force": ["x", "y", "z"],
        "compensated_base_torque": ["x", "y", "z"],
        "measured_force": ["x", "y", "z"],
        "measured_torque": ["x", "y", "z"],
        "pose": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        "velocity": ["linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"],
        "gripper_efforts": ["left", "right"],
        "gripper_positions": ["left", "right"],
        "gripper_velocities": ["left", "right"],
    }
    if name in {"joint_efforts", "joint_positions", "joint_velocities"}:
        return [f"joint_{index}" for index in range(width)]
    base = known.get(name)
    if base is None:
        return [f"dim_{index}" for index in range(width)]
    if len(base) < width:
        base = [*base, *[f"dim_{index}" for index in range(len(base), width)]]
    return base[:width]


def _gripper_transitions_from_series(
    timestamps_sec: list[float],
    values: list[float],
    *,
    threshold: float,
    hysteresis: float = 0.0,
) -> list[GripperTransitionMark]:
    count = min(len(timestamps_sec), len(values))
    if count < 2:
        return []
    low_threshold = float(threshold) - max(0.0, float(hysteresis))
    high_threshold = float(threshold) + max(0.0, float(hysteresis))
    first_value = float(values[0])
    current_state = "open" if first_value >= threshold else "closed"
    marks: list[GripperTransitionMark] = []
    for index in range(1, count):
        value = float(values[index])
        previous_state = current_state
        if current_state == "closed" and value >= high_threshold:
            current_state = "open"
        elif current_state == "open" and value <= low_threshold:
            current_state = "closed"
        else:
            continue
        marks.append(
            GripperTransitionMark(
                timestamp_sec=float(timestamps_sec[index]),
                label="O" if current_state == "open" else "C",
                state=current_state,
                previous_state=previous_state,
                raw_value=value,
            )
        )
    return marks


def _adaptive_gripper_threshold(values: list[float]) -> float:
    value_min = min(values)
    value_max = max(values)
    if value_max <= value_min:
        return value_min
    return value_min + 0.5 * (value_max - value_min)


def _h5_global_timestamp_origin(handle: Any) -> float:
    timestamps_group = handle.get("timestamps")
    if timestamps_group is None:
        return 0.0
    minima: list[float] = []
    for dataset in timestamps_group.values():
        if getattr(dataset, "shape", ()) and int(dataset.shape[0]) > 0:
            values = dataset[:]
            minima.append(float(values.min()))
    return min(minima) if minima else 0.0


def _row_mean(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        numbers = [float(item) for item in value if isinstance(item, (int, float))]
        return sum(numbers) / len(numbers) if numbers else 0.0
    return float(value)


def timestamp_from_frame_path(image_path: Path) -> float | None:
    match = re.search(r"_t([0-9]+(?:\.[0-9]+)?)", image_path.stem)
    if not match:
        return None
    return float(match.group(1))


def _viewer_bundle_from_manifest(
    manifest_entry: dict[str, Any],
    rows_by_selector: dict[str, dict[str, Any]],
) -> ViewerBundle:
    coverage = manifest_entry.get("failed_low_level_coverage")
    high_level_segment = manifest_entry.get("high_level_segment")
    return ViewerBundle(
        bundle_id=str(manifest_entry["bundle_id"]),
        h5_path=str(manifest_entry.get("h5_path")) if manifest_entry.get("h5_path") else None,
        source_file_bundle_id=str(
            manifest_entry.get("source_file_bundle_id") or _source_id_from_h5(manifest_entry.get("h5_path"))
        ),
        high_level_task_text=(
            str(high_level_segment.get("text"))
            if isinstance(high_level_segment, dict) and high_level_segment.get("text") is not None
            else None
        ),
        high_level_task_kind=(
            str(manifest_entry.get("high_level_task_kind"))
            if manifest_entry.get("high_level_task_kind") is not None
            else None
        ),
        duration_sec=_as_float(manifest_entry.get("duration_sec")),
        failed_low_level_coverage_fraction=(
            _as_float(coverage.get("failed_low_level_coverage_fraction"))
            if isinstance(coverage, dict)
            else None
        ),
        manifest_entry=manifest_entry,
        rows_by_selector=rows_by_selector,
    )


def _source_id_from_h5(value: Any) -> str:
    if isinstance(value, str) and value:
        return Path(value).stem
    return ""


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [item for item in value if isinstance(item, dict)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _dict_list(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        number = _as_float(item)
        if number is not None:
            result.append(number)
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _leading_int(value: str) -> int | None:
    match = re.match(r"^(\d+)", value)
    return int(match.group(1)) if match else None


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)
