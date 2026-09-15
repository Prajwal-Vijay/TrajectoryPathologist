"""RONAR-style multimodal key-event selection.

This module implements the threshold rule described in Appendix B.1.1 of the
RONAR paper.  It intentionally does not implement planner-state triggers or the
adjacent-image clarity heuristic: neither is part of the fair E+I comparison
used by this project.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from pathologist.metrics.helpers import extract_vector_series, find_stream
from pathologist.schemas.common import Modality
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


RONAR_DEFAULT_SAMPLE_PERIOD_SEC = 0.2
RONAR_DEFAULT_THRESHOLD = 80.0
RONAR_DEFAULT_RESIZE = (128, 128)


@dataclass(slots=True)
class RonarFeatureSeries:
    """Aligned RONAR E+I feature values for one trajectory."""

    timestamps_sec: list[float]
    values_by_feature: dict[str, list[float]]
    camera_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RonarEvent:
    """One emitted threshold crossing or explicit fixed-budget fallback."""

    timestamp_sec: float
    frame: CameraFrameRef
    selection_source: str
    accumulated_score: float | None = None
    instantaneous_score: float | None = None

    def to_dict(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "peak_sec": self.timestamp_sec,
            "camera_name": self.frame.camera_name,
            "frame_index": self.frame.frame_index,
            "image_path": str(self.frame.image_path),
            "selection_source": self.selection_source,
            "accumulated_score": self.accumulated_score,
            "instantaneous_score": self.instantaneous_score,
        }


@dataclass(slots=True)
class RonarSelection:
    """Selected RONAR events and audit metadata."""

    events: list[RonarEvent]
    metadata: dict[str, Any]
    features: RonarFeatureSeries


@dataclass(slots=True)
class RonarStatisticsAccumulator:
    """Streaming per-feature statistics for task-level RONAR calibration."""

    _states: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    def update(self, features: RonarFeatureSeries) -> None:
        for name, values in features.values_by_feature.items():
            count, center, squared_deviation = self._states.get(
                name,
                (0, 0.0, 0.0),
            )
            for raw_value in values:
                value = float(raw_value)
                if not math.isfinite(value):
                    continue
                count += 1
                delta = value - center
                center += delta / count
                squared_deviation += delta * (value - center)
            self._states[name] = (count, center, squared_deviation)

    def finalize(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "mean": center,
                "std": (
                    math.sqrt(squared_deviation / count)
                    if count > 0
                    else 0.0
                ),
                "count": count,
            }
            for name, (count, center, squared_deviation) in sorted(
                self._states.items()
            )
            if count > 0
        }


def extract_ronar_features(
    bundle: TrajectoryBundle,
    *,
    camera_names: Iterable[str] | None = None,
    sample_period_sec: float = RONAR_DEFAULT_SAMPLE_PERIOD_SEC,
    resize: tuple[int, int] = RONAR_DEFAULT_RESIZE,
) -> RonarFeatureSeries:
    """Align camera and joint data and compute RONAR's E+I features.

    Each configured camera contributes one average dense-flow magnitude.  The
    internal-state feature is the L1 change of the full joint-position vector,
    matching the public RONAR code's sum of absolute arm-joint changes.
    """

    if sample_period_sec <= 0:
        raise ValueError("RONAR sample_period_sec must be positive.")
    if resize[0] <= 0 or resize[1] <= 0:
        raise ValueError("RONAR resize dimensions must be positive.")

    resolved_cameras = _resolve_camera_names(bundle.camera_frames, camera_names)
    duration_sec = _bundle_duration(bundle)
    timestamps = _sample_grid(duration_sec, sample_period_sec)
    values_by_feature: dict[str, list[float]] = {}

    frames_by_camera = {
        camera_name: sorted(
            [
                frame
                for frame in bundle.camera_frames
                if frame.camera_name == camera_name
            ],
            key=_frame_sort_key,
        )
        for camera_name in resolved_cameras
    }
    for camera_name, frames in frames_by_camera.items():
        values_by_feature[f"flow:{camera_name}"] = _aligned_flow_values(
            frames,
            timestamps,
            resize=resize,
        )

    proprioception = find_stream(
        bundle,
        modalities=(Modality.PROPRIOCEPTION,),
        preferred_names=("proprioception",),
    )
    joint_timestamps, joint_positions = extract_vector_series(
        proprioception,
        keys=("joint_positions", "proprioception", "state", "qpos", "position"),
    )
    if joint_positions:
        values_by_feature["arm_motion"] = _aligned_joint_motion(
            joint_timestamps,
            joint_positions,
            timestamps,
        )

    return RonarFeatureSeries(
        timestamps_sec=timestamps,
        values_by_feature=values_by_feature,
        camera_names=resolved_cameras,
    )


def feature_statistics(
    feature_series: Iterable[RonarFeatureSeries],
) -> dict[str, dict[str, float | int]]:
    """Pool feature values into serializable mean/std/count statistics."""

    pooled: dict[str, list[float]] = {}
    for series in feature_series:
        for name, values in series.values_by_feature.items():
            pooled.setdefault(name, []).extend(
                float(value) for value in values if math.isfinite(float(value))
            )
    return {
        name: {
            "mean": fmean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }
        for name, values in sorted(pooled.items())
        if values
    }


def select_ronar_keyevents(
    bundle: TrajectoryBundle,
    *,
    camera_names: Iterable[str] | None = None,
    sample_period_sec: float = RONAR_DEFAULT_SAMPLE_PERIOD_SEC,
    resize: tuple[int, int] = RONAR_DEFAULT_RESIZE,
    threshold: float = RONAR_DEFAULT_THRESHOLD,
    normalization_stats: dict[str, dict[str, Any]] | None = None,
    frame_budget: int | None = None,
) -> RonarSelection:
    """Select strict threshold events or the per-trajectory exact-K adaptation."""

    if threshold < 0:
        raise ValueError("RONAR threshold cannot be negative.")
    if frame_budget is not None and frame_budget <= 0:
        raise ValueError("RONAR frame_budget must be positive.")

    features = extract_ronar_features(
        bundle,
        camera_names=camera_names,
        sample_period_sec=sample_period_sec,
        resize=resize,
    )
    resolved_stats = (
        _validated_statistics(normalization_stats)
        if normalization_stats is not None
        else feature_statistics([features])
    )
    scores = positive_zscore_sum(features, resolved_stats)
    normalization_scope = (
        "provided_task_statistics"
        if normalization_stats is not None
        else "within_trajectory"
    )

    if frame_budget is None:
        crossing_indices, crossing_accumulators = threshold_crossings(
            scores,
            threshold,
        )
        events = _events_from_crossings(
            bundle.camera_frames,
            features.timestamps_sec,
            scores,
            crossing_indices,
            crossing_accumulators,
            selection_source="ronar_threshold",
        )
        metadata = {
            "protocol": "published_threshold",
            "normalization_scope": normalization_scope,
            "threshold": threshold,
            "sample_period_sec": sample_period_sec,
            "feature_names": sorted(features.values_by_feature),
            "camera_names": features.camera_names,
            "aligned_sample_count": len(features.timestamps_sec),
            "genuine_crossing_count": len(events),
            "downselected_crossing_count": 0,
            "uniform_padded_event_count": 0,
            "selected_event_count": len(events),
        }
        return RonarSelection(events=events, metadata=metadata, features=features)

    calibrated_threshold, crossing_indices, crossing_accumulators = (
        _threshold_for_at_least_k(scores, frame_budget)
    )
    all_crossing_count = len(crossing_indices)
    downselected_count = 0
    if len(crossing_indices) > frame_budget:
        keep = _uniform_indices(len(crossing_indices), frame_budget)
        crossing_indices = [crossing_indices[index] for index in keep]
        crossing_accumulators = [
            crossing_accumulators[index] for index in keep
        ]
        downselected_count = all_crossing_count - frame_budget
    events = _events_from_crossings(
        bundle.camera_frames,
        features.timestamps_sec,
        scores,
        crossing_indices,
        crossing_accumulators,
        selection_source="ronar_k_crossing",
    )
    genuine_selected_count = len(events)
    if len(events) < frame_budget:
        events = _pad_events_uniformly(
            events,
            bundle.camera_frames,
            frame_budget,
        )
    padding_count = max(0, len(events) - genuine_selected_count)
    metadata = {
        "protocol": "per_trajectory_exact_k",
        "normalization_scope": normalization_scope,
        "threshold": calibrated_threshold,
        "requested_frame_budget": frame_budget,
        "sample_period_sec": sample_period_sec,
        "feature_names": sorted(features.values_by_feature),
        "camera_names": features.camera_names,
        "aligned_sample_count": len(features.timestamps_sec),
        "available_crossing_count_at_min_threshold": sum(
            score > 0 for score in scores
        ),
        "genuine_crossing_count": all_crossing_count,
        "genuine_selected_event_count": genuine_selected_count,
        "downselected_crossing_count": downselected_count,
        "uniform_padded_event_count": padding_count,
        "selected_event_count": len(events),
    }
    return RonarSelection(
        events=sorted(events[:frame_budget], key=lambda event: event.timestamp_sec),
        metadata=metadata,
        features=features,
    )


def positive_zscore_sum(
    features: RonarFeatureSeries,
    statistics: dict[str, dict[str, Any]],
) -> list[float]:
    """Return the sum of positive normalized feature values at each sample."""

    sample_count = len(features.timestamps_sec)
    scores = [0.0] * sample_count
    for name, values in features.values_by_feature.items():
        stats = statistics.get(name)
        if not isinstance(stats, dict):
            continue
        center = _finite_float(stats.get("mean"))
        scale = _finite_float(stats.get("std"))
        if center is None or scale is None or scale <= 0:
            continue
        for index, value in enumerate(values[:sample_count]):
            zscore = (float(value) - center) / scale
            if math.isfinite(zscore) and zscore > 0:
                scores[index] += zscore
    return scores


def threshold_crossings(
    scores: list[float],
    threshold: float,
) -> tuple[list[int], list[float]]:
    """Apply RONAR's reset-on-crossing cumulative threshold rule."""

    if threshold < 0:
        raise ValueError("RONAR threshold cannot be negative.")
    indices: list[int] = []
    accumulators: list[float] = []
    accumulator = 0.0
    for index, raw_score in enumerate(scores):
        score = float(raw_score)
        if math.isfinite(score) and score > 0:
            accumulator += score
        if accumulator > threshold:
            indices.append(index)
            accumulators.append(accumulator)
            accumulator = 0.0
    return indices, accumulators


def _threshold_for_at_least_k(
    scores: list[float],
    frame_budget: int,
) -> tuple[float, list[int], list[float]]:
    """Find the highest per-trajectory threshold still yielding at least K events."""

    minimum_indices, minimum_accumulators = threshold_crossings(scores, 0.0)
    if len(minimum_indices) <= frame_budget:
        return 0.0, minimum_indices, minimum_accumulators

    low = 0.0
    high = sum(max(0.0, float(score)) for score in scores)
    best_threshold = low
    best_indices = minimum_indices
    best_accumulators = minimum_accumulators
    for _ in range(80):
        middle = (low + high) / 2.0
        indices, accumulators = threshold_crossings(scores, middle)
        if len(indices) >= frame_budget:
            low = middle
            if len(indices) < len(best_indices) or (
                len(indices) == len(best_indices) and middle > best_threshold
            ):
                best_threshold = middle
                best_indices = indices
                best_accumulators = accumulators
        else:
            high = middle
    return best_threshold, best_indices, best_accumulators


def _events_from_crossings(
    all_frames: list[CameraFrameRef],
    timestamps: list[float],
    scores: list[float],
    crossing_indices: list[int],
    crossing_accumulators: list[float],
    *,
    selection_source: str,
) -> list[RonarEvent]:
    events: list[RonarEvent] = []
    seen: set[tuple[str, int | None, float]] = set()
    sorted_frames = sorted(all_frames, key=_frame_sort_key)
    for index, accumulated_score in zip(
        crossing_indices,
        crossing_accumulators,
    ):
        if index >= len(timestamps):
            continue
        frame = _nearest_frame(sorted_frames, timestamps[index])
        if frame is None:
            continue
        identity = _frame_identity(frame)
        if identity in seen:
            continue
        seen.add(identity)
        events.append(
            RonarEvent(
                timestamp_sec=frame.timestamp_sec,
                frame=frame,
                selection_source=selection_source,
                accumulated_score=float(accumulated_score),
                instantaneous_score=(
                    float(scores[index]) if index < len(scores) else None
                ),
            )
        )
    return events


def _pad_events_uniformly(
    events: list[RonarEvent],
    frames: list[CameraFrameRef],
    frame_budget: int,
) -> list[RonarEvent]:
    chosen = list(events[:frame_budget])
    seen = {_frame_identity(event.frame) for event in chosen}
    remaining = [
        frame
        for frame in sorted(frames, key=_frame_sort_key)
        if _frame_identity(frame) not in seen
    ]
    needed = frame_budget - len(chosen)
    for index in _uniform_indices(len(remaining), min(needed, len(remaining))):
        frame = remaining[index]
        chosen.append(
            RonarEvent(
                timestamp_sec=frame.timestamp_sec,
                frame=frame,
                selection_source="uniform_padding",
            )
        )
    return chosen


def _aligned_flow_values(
    frames: list[CameraFrameRef],
    timestamps: list[float],
    *,
    resize: tuple[int, int],
) -> list[float]:
    if not frames:
        return []
    aligned = [_nearest_frame(frames, timestamp) for timestamp in timestamps]
    cv2 = _import_cv2()
    gray_cache: dict[Path, Any] = {}
    output = [0.0]
    for previous, current in zip(aligned, aligned[1:]):
        if previous is None or current is None:
            output.append(0.0)
            continue
        if _frame_identity(previous) == _frame_identity(current):
            output.append(0.0)
            continue
        previous_gray = _read_gray(previous.image_path, resize, cv2, gray_cache)
        current_gray = _read_gray(current.image_path, resize, cv2, gray_cache)
        if previous_gray is None or current_gray is None:
            output.append(0.0)
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        output.append(float(magnitude.mean()))
    return output


def _aligned_joint_motion(
    source_timestamps: list[float],
    source_values: list[list[float]],
    target_timestamps: list[float],
) -> list[float]:
    aligned = [
        _nearest_vector(source_timestamps, source_values, timestamp)
        for timestamp in target_timestamps
    ]
    output = [0.0]
    for previous, current in zip(aligned, aligned[1:]):
        if previous is None or current is None:
            output.append(0.0)
            continue
        width = min(len(previous), len(current))
        output.append(
            sum(abs(current[index] - previous[index]) for index in range(width))
        )
    return output


def _nearest_vector(
    timestamps: list[float],
    values: list[list[float]],
    target: float,
) -> list[float] | None:
    if not timestamps or not values:
        return None
    index = _nearest_index(timestamps, target)
    return values[index] if index < len(values) else None


def _nearest_frame(
    frames: list[CameraFrameRef],
    target: float,
) -> CameraFrameRef | None:
    if not frames:
        return None
    timestamps = [frame.timestamp_sec for frame in frames]
    return frames[_nearest_index(timestamps, target)]


def _nearest_index(values: list[float], target: float) -> int:
    insertion = bisect.bisect_left(values, target)
    if insertion <= 0:
        return 0
    if insertion >= len(values):
        return len(values) - 1
    before = insertion - 1
    after = insertion
    if abs(values[before] - target) <= abs(values[after] - target):
        return before
    return after


def _read_gray(
    image_path: Path,
    resize: tuple[int, int],
    cv2: Any,
    cache: dict[Path, Any],
) -> Any | None:
    if image_path in cache:
        return cache[image_path]
    image = cv2.imread(str(image_path))
    if image is None:
        cache[image_path] = None
        return None
    image = cv2.resize(image, resize)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cache[image_path] = gray
    return gray


def _resolve_camera_names(
    frames: list[CameraFrameRef],
    camera_names: Iterable[str] | None,
) -> list[str]:
    available = sorted({frame.camera_name for frame in frames})
    if camera_names is None:
        return available
    requested: list[str] = []
    for raw_name in camera_names:
        name = str(raw_name)
        if name in available and name not in requested:
            requested.append(name)
    return requested


def _sample_grid(duration_sec: float, period_sec: float) -> list[float]:
    if duration_sec <= 0:
        return [0.0]
    count = int(math.floor(duration_sec / period_sec))
    timestamps = [index * period_sec for index in range(count + 1)]
    if duration_sec - timestamps[-1] > period_sec * 0.5:
        timestamps.append(duration_sec)
    return timestamps


def _bundle_duration(bundle: TrajectoryBundle) -> float:
    if bundle.duration_sec is not None:
        return float(bundle.duration_sec)
    candidates = [
        *(frame.timestamp_sec for frame in bundle.camera_frames),
        *(
            timestamp
            for stream in bundle.streams
            for timestamp in stream.timestamps_sec
        ),
    ]
    return max(candidates, default=0.0)


def _uniform_indices(item_count: int, selection_count: int) -> list[int]:
    if item_count <= 0 or selection_count <= 0:
        return []
    if selection_count >= item_count:
        return list(range(item_count))
    if selection_count == 1:
        return [item_count // 2]
    return [
        round(index * (item_count - 1) / (selection_count - 1))
        for index in range(selection_count)
    ]


def _validated_statistics(
    statistics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for name, raw_stats in statistics.items():
        if not isinstance(name, str) or not isinstance(raw_stats, dict):
            continue
        center = _finite_float(raw_stats.get("mean"))
        scale = _finite_float(raw_stats.get("std"))
        if center is None or scale is None:
            continue
        output[name] = {
            "mean": center,
            "std": max(0.0, scale),
            "count": int(raw_stats.get("count", 0)),
        }
    return output


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _frame_sort_key(frame: CameraFrameRef) -> tuple[float, str, int]:
    return (
        frame.timestamp_sec,
        frame.camera_name,
        frame.frame_index if frame.frame_index is not None else -1,
    )


def _frame_identity(frame: CameraFrameRef) -> tuple[str, int | None, float]:
    return (frame.camera_name, frame.frame_index, frame.timestamp_sec)


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RONAR flow selection requires opencv-python/cv2."
        ) from exc
    return cv2
