"""Centered-window excess path for retries and reversals."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import (
    extract_vector_series,
    find_stream,
    l2_norm,
    summarize,
    vector_diff,
    vector_group_indices,
)
from pathologist.metrics.kinematics import _extract_wrist_positions
from pathologist.schemas.common import Modality, TimeWindow
from pathologist.schemas.metrics import MetricResult, MetricSample


@dataclass(frozen=True, slots=True)
class _GroupedVectorSeries:
    timestamps: list[float]
    values_by_group: dict[str, list[list[float]]]


class _WindowedExcessPathMetric:
    """Shared centered-window TV-minus-net-displacement implementation."""

    name = ""
    description = ""
    unit = ""
    source_modalities: list[Modality] = []

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = metric_config(context, self.name)
        if not bool(config.get("enabled", False)):
            return self._unavailable(f"disabled by metrics.{self.name}.enabled")
        half_width_sec = float(config.get("half_width_sec", 1.0))
        if half_width_sec <= 0.0:
            raise ValueError(f"metrics.{self.name}.half_width_sec must be positive")

        series, stream_name = self._extract_series(context)
        if series is None or len(series.timestamps) < 2 or not series.values_by_group:
            return self._unavailable("required source stream is unavailable")

        centers = [
            timestamp
            for timestamp in series.timestamps
            if timestamp - half_width_sec >= series.timestamps[0]
            and timestamp + half_width_sec <= series.timestamps[-1]
        ]
        samples: list[MetricSample] = []
        group_values = {name: [] for name in series.values_by_group}
        for center in centers:
            per_group = {
                name: windowed_excess_path(
                    timestamps=series.timestamps,
                    values=values,
                    center_sec=center,
                    half_width_sec=half_width_sec,
                )
                for name, values in series.values_by_group.items()
            }
            if any(value is None for value in per_group.values()):
                continue
            complete = {name: float(value) for name, value in per_group.items()}
            for name, value in complete.items():
                group_values[name].append(value)
            samples.append(
                MetricSample(
                    timestamp_sec=center,
                    value=max(complete.values()),
                    window=TimeWindow(
                        center - half_width_sec,
                        center + half_width_sec,
                    ),
                )
            )
        if len(samples) < 2:
            return self._unavailable(
                "requires at least two complete centered windows"
            )

        values = [float(sample.value) for sample in samples]
        return MetricResult(
            name=self.name,
            description=self.description,
            source_modalities=list(self.source_modalities),
            unit=self.unit,
            samples=samples,
            summary=summarize(values),
            thresholds=[],
            events=[],
            display_hint=f"{self.name}_timeseries",
            metadata={
                "available": True,
                "stream_name": stream_name,
                "half_width_sec": half_width_sec,
                "full_window_sec": 2.0 * half_width_sec,
                "formula": "sum_i ||x_(i+1)-x_i|| - ||x_end-x_start||",
                "boundary_policy": "full_centered_windows_only",
                "group_names": list(group_values),
                "per_group_values": group_values,
                "group_reduction": "max",
                "surprise_tail_interpretation": "right",
            },
        )

    def _extract_series(
        self,
        context: MetricComputationContext,
    ) -> tuple[_GroupedVectorSeries | None, str | None]:
        raise NotImplementedError

    def _unavailable(self, reason: str) -> MetricResult:
        return MetricResult(
            name=self.name,
            description=self.description,
            source_modalities=list(self.source_modalities),
            unit=self.unit,
            samples=[],
            display_hint=f"{self.name}_timeseries",
            metadata={"available": False, "unavailable_reason": reason},
        )


class JointExcessPathMetric(_WindowedExcessPathMetric):
    """Joint-space motion undone within a centered window."""

    name = "joint_excess_path"
    description = "Centered-window joint-space excess path (TV minus displacement)."
    unit = "joint_space_l2"
    source_modalities = [Modality.PROPRIOCEPTION]

    def _extract_series(
        self,
        context: MetricComputationContext,
    ) -> tuple[_GroupedVectorSeries | None, str | None]:
        stream = find_stream(
            context.bundle,
            [Modality.PROPRIOCEPTION],
            preferred_names=["proprioception", "state"],
        )
        timestamps, values = extract_vector_series(
            stream,
            ["joint_positions", "proprioception", "state"],
        )
        return _grouped_vector_series(timestamps, values, stream), (
            stream.name if stream is not None else None
        )


class GripperExcessPathMetric(_WindowedExcessPathMetric):
    """Gripper-aperture motion undone within a centered window."""

    name = "gripper_excess_path"
    description = "Centered-window gripper excess path (TV minus displacement)."
    unit = "aperture"
    source_modalities = [Modality.GRIPPER]

    def _extract_series(
        self,
        context: MetricComputationContext,
    ) -> tuple[_GroupedVectorSeries | None, str | None]:
        stream = find_stream(
            context.bundle,
            [Modality.GRIPPER],
            preferred_names=["gripper"],
        )
        timestamps, values = extract_vector_series(
            stream,
            ["gripper_state", "gripper"],
        )
        return _grouped_vector_series(timestamps, values, stream), (
            stream.name if stream is not None else None
        )


class EefExcessPathMetric(_WindowedExcessPathMetric):
    """End-effector Cartesian motion undone within a centered window."""

    name = "eef_excess_path"
    description = "Centered-window EEF Cartesian excess path (TV minus displacement)."
    unit = "meters"
    source_modalities = [Modality.TELEMETRY]

    def _extract_series(
        self,
        context: MetricComputationContext,
    ) -> tuple[_GroupedVectorSeries | None, str | None]:
        stream = find_stream(
            context.bundle,
            [Modality.TELEMETRY],
            preferred_names=["telemetry"],
        )
        timestamps, positions = _extract_wrist_positions(stream)
        complete = {
            name: values
            for name, values in positions.items()
            if len(values) == len(timestamps)
        }
        series = (
            _GroupedVectorSeries(timestamps=timestamps, values_by_group=complete)
            if timestamps and complete
            else None
        )
        return series, stream.name if stream is not None else None


def windowed_excess_path(
    *,
    timestamps: list[float],
    values: list[list[float]],
    center_sec: float,
    half_width_sec: float,
) -> float | None:
    """Return vector TV minus endpoint displacement in one centered window."""

    start = bisect_left(timestamps, center_sec - half_width_sec)
    end = bisect_right(timestamps, center_sec + half_width_sec)
    window_values = values[start:end]
    if len(window_values) < 2:
        return None
    total_variation = sum(
        l2_norm(vector_diff(current, previous))
        for previous, current in zip(window_values, window_values[1:])
    )
    net_displacement = l2_norm(
        vector_diff(window_values[-1], window_values[0])
    )
    return max(0.0, total_variation - net_displacement)


def _grouped_vector_series(
    timestamps: list[float],
    values: list[list[float]],
    stream,
) -> _GroupedVectorSeries | None:
    width = min((len(value) for value in values), default=0)
    if width <= 0 or len(timestamps) != len(values):
        return None
    groups = vector_group_indices(stream, width)
    if not groups:
        groups = {"all": list(range(width))}
    values_by_group = {
        name: [
            [float(vector[index]) for index in indices]
            for vector in values
        ]
        for name, indices in groups.items()
    }
    return _GroupedVectorSeries(
        timestamps=[float(timestamp) for timestamp in timestamps],
        values_by_group=values_by_group,
    )
