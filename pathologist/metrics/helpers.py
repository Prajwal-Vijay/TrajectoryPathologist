"""Shared helpers for metric computation."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev
from typing import Any, Iterable

from pathologist.schemas.common import Modality, NumericSummary
from pathologist.schemas.trajectory_bundle import ModalityStream, TrajectoryBundle


def find_stream(
    bundle: TrajectoryBundle,
    modalities: Iterable[Modality],
    preferred_names: Iterable[str] = (),
) -> ModalityStream | None:
    """Return the first stream matching preferred names, then modalities."""

    modalities = set(modalities)
    preferred_names = list(preferred_names)
    for name in preferred_names:
        for stream in bundle.streams:
            if stream.name == name and stream.modality in modalities:
                return stream
    for stream in bundle.streams:
        if stream.modality in modalities:
            return stream
    return None


def stream_records(stream: ModalityStream | None) -> list[dict[str, Any]]:
    """Return in-memory records for a stream.

    Adapters should populate records before metric computation. File-backed
    stream loading can be added later without changing individual metrics.
    """

    if stream is None or stream.records is None:
        return []
    return stream.records


def extract_vector_series(
    stream: ModalityStream | None,
    keys: Iterable[str],
) -> tuple[list[float], list[list[float]]]:
    """Extract a timestamped vector series from stream records."""

    records = stream_records(stream)
    if not records:
        return [], []

    keys = list(keys)
    timestamps = stream.timestamps_sec if stream and stream.timestamps_sec else []
    values: list[list[float]] = []
    value_timestamps: list[float] = []

    for index, record in enumerate(records):
        raw_value = None
        for key in keys:
            if key in record:
                raw_value = record[key]
                break
        if raw_value is None:
            continue
        vector = as_float_vector(raw_value)
        if not vector:
            continue
        if index < len(timestamps):
            timestamp = timestamps[index]
        else:
            timestamp = float(record.get("timestamp_sec", index))
        values.append(vector)
        value_timestamps.append(float(timestamp))
    return value_timestamps, values


def as_float_vector(value: Any) -> list[float]:
    """Convert scalar/list-like values into a flat float vector."""

    if value is None:
        return []
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        vector: list[float] = []
        for item in value:
            if isinstance(item, (int, float, bool)):
                vector.append(float(item))
        return vector
    return []


def align_timestamps(timestamps: list[float], count: int) -> list[float]:
    """Return timestamps with a safe fallback to sample indices."""

    if len(timestamps) >= count:
        return [float(timestamp) for timestamp in timestamps[:count]]
    return [float(index) for index in range(count)]


def l2_norm(vector: list[float]) -> float:
    """Compute Euclidean norm."""

    return math.sqrt(sum(item * item for item in vector))


def vector_diff(left: list[float], right: list[float]) -> list[float]:
    """Subtract equal-length prefix of two vectors."""

    width = min(len(left), len(right))
    return [left[index] - right[index] for index in range(width)]


def safe_dt(previous: float, current: float, fallback: float = 1.0) -> float:
    """Return a positive timestep."""

    dt = current - previous
    if dt <= 0:
        return fallback
    return dt


def summarize(values: list[float]) -> NumericSummary:
    """Build a NumericSummary from scalar samples."""

    if not values:
        return NumericSummary(count=0)
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1)
    return NumericSummary(
        mean=mean(values),
        std=pstdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        median=median(values),
        p95=sorted_values[p95_index],
        count=len(values),
    )


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""

    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.ceil((q / 100.0) * len(sorted_values)) - 1)
    return sorted_values[index]


def median_absolute_deviation(values: list[float]) -> float:
    """Return median absolute deviation."""

    if not values:
        return 0.0
    center = median(values)
    return median([abs(value - center) for value in values])


def top_indices(values: list[float], minimum_value: float, limit: int) -> list[int]:
    """Return indices of the largest values above a minimum."""

    candidates = [
        (index, value)
        for index, value in enumerate(values)
        if value >= minimum_value
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [index for index, _ in candidates[:limit]]


def unavailable_result(name: str, description: str, reason: str) -> MetricResult:
    """Create a MetricResult explaining why a metric could not run."""

    return MetricResult(
        name=name,
        description=description,
        metadata={"available": False, "unavailable_reason": reason},
    )
