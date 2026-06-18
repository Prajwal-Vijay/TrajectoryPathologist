"""Action-signal metrics."""

from __future__ import annotations

import math

from pathologist.metrics.base import MetricComputationContext, metric_config, metric_option
from pathologist.metrics.helpers import (
    align_timestamps,
    extract_vector_series,
    find_stream,
    percentile,
    summarize,
    top_indices,
)
from pathologist.schemas.common import (
    Modality,
    Severity,
    SeverityBand,
    Threshold,
    ThresholdBasis,
    ThresholdCondition,
    ThresholdOperator,
    TimeWindow,
)
from pathologist.schemas.metrics import MetricEvent, MetricResult, MetricSample


class ActionRangeSaturationMetric:
    """Percentage of action dimensions at or outside configured bounds."""

    name = "action_range_saturation"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(context.bundle, [Modality.ACTION], preferred_names=["action"])
        timestamps, actions = extract_vector_series(stream, ["action", "actions"])
        if stream is None or not actions:
            return MetricResult(
                name=self.name,
                description="Percentage of timesteps where action dimensions hit min/max bounds.",
                metadata={"available": False, "unavailable_reason": "requires an action stream with records"},
            )

        config = metric_config(context, self.name)
        low, high = _action_bounds(stream.metadata, context.bundle.raw_metadata, config)
        if low is None or high is None:
            return MetricResult(
                name=self.name,
                description="Percentage of timesteps where action dimensions hit min/max bounds.",
                source_modalities=[Modality.ACTION],
                metadata={
                    "available": False,
                    "unavailable_reason": "requires action_low/action_high metadata or action_space bounds",
                    "stream_name": stream.name,
                },
            )

        width = min(len(low), len(high), min(len(action) for action in actions))
        tolerance = float(metric_option(config, stream.metadata, "saturation_tolerance", 1e-6))
        low_fraction = float(metric_option(config, stream.metadata, "low_fraction", 0.01))
        medium_fraction = float(metric_option(config, stream.metadata, "medium_fraction", 0.05))
        high_fraction = float(metric_option(config, stream.metadata, "high_fraction", 0.15))
        sample_values = []
        saturated_dims_per_step: list[list[int]] = []
        saturated_steps = 0
        saturated_dimensions = 0
        total_dimensions_obs = width * len(actions)
        for action in actions:
            saturated_dims = []
            count = 0
            for dim in range(width):
                if action[dim] <= low[dim] + tolerance or action[dim] >= high[dim] - tolerance:
                    count += 1
                    saturated_dims.append(dim)
            saturated_dimensions += count
            if count > 0:
                saturated_steps += 1
            sample_values.append(count / width if width else 0.0)
            saturated_dims_per_step.append(saturated_dims)

        step_fraction = saturated_steps / len(actions) if actions else 0.0
        dimension_fraction = saturated_dimensions / total_dimensions_obs if total_dimensions_obs else 0.0
        threshold = Threshold(
            name="action_saturation_at_bounds",
            basis=ThresholdBasis.HARD_LIMIT,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=0.0,
                reference_statistic="action_space_bounds",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.LOW, lower=low_fraction, upper=medium_fraction),
                SeverityBand(severity=Severity.MEDIUM, lower=medium_fraction, upper=high_fraction),
                SeverityBand(severity=Severity.HIGH, lower=high_fraction),
            ],
            numeric_context={
                "tolerance": tolerance,
                "saturated_step_fraction": step_fraction,
                "saturated_dimension_fraction": dimension_fraction,
                "low_fraction": low_fraction,
                "medium_fraction": medium_fraction,
                "high_fraction": high_fraction,
            },
        )
        events = []
        saturation_runs = _saturation_runs(saturated_dims_per_step)
        saturation_runs.sort(
            key=lambda run: (
                run[2] - run[1] + 1,
                sum(len(saturated_dims_per_step[index]) for index in range(run[1], run[2] + 1)),
            ),
            reverse=True,
        )
        for dimensions, start_index, end_index in saturation_runs[: context.max_events_per_metric]:
            run_step_count = end_index - start_index + 1
            run_saturated_dims = sum(
                len(saturated_dims_per_step[index])
                for index in range(start_index, end_index + 1)
            )
            run_fraction = run_saturated_dims / (run_step_count * width) if width else 0.0
            severity = _action_saturation_severity(run_fraction, medium_fraction, high_fraction)
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(timestamps[start_index], timestamps[end_index]),
                    value={
                        "saturated_dimensions": sorted(dimensions),
                        "num_steps": run_step_count,
                        "saturated_dimension_fraction": run_fraction,
                    },
                    severity=severity,
                    reason=(
                        f"action dimensions {sorted(dimensions)} touched configured bounds "
                        f"for {run_step_count} consecutive step(s)"
                    ),
                    threshold=threshold,
                )
            )

        return MetricResult(
            name=self.name,
            description="Percentage of timesteps where action dimensions hit min/max bounds.",
            source_modalities=[Modality.ACTION],
            unit="fraction",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(align_timestamps(timestamps, len(sample_values)), sample_values)
            ],
            summary=summarize(sample_values),
            thresholds=[threshold],
            events=events,
            metadata={
                "available": True,
                "stream_name": stream.name,
                "saturated_step_fraction": step_fraction,
                "saturated_dimension_fraction": dimension_fraction,
                "num_saturation_windows": len(saturation_runs),
            },
        )


class ActionDistributionEntropyMetric:
    """Normalized Shannon entropy of each action dimension."""

    name = "action_distribution_entropy"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(context.bundle, [Modality.ACTION], preferred_names=["action"])
        timestamps, actions = extract_vector_series(stream, ["action", "actions"])
        if stream is None or len(actions) < 2:
            return MetricResult(
                name=self.name,
                description="Normalized entropy of action distribution per dimension.",
                metadata={"available": False, "unavailable_reason": "requires at least two action samples"},
            )

        width = min(len(action) for action in actions)
        config = metric_config(context, self.name)
        bins = int(metric_option(config, stream.metadata, "entropy_bins", 10))
        low_entropy_threshold = float(metric_option(config, stream.metadata, "low_entropy_threshold", 0.15))
        high_entropy_threshold = float(metric_option(config, stream.metadata, "high_entropy_threshold", 0.95))
        entropy_per_dim = []
        for dim in range(width):
            values = [action[dim] for action in actions]
            entropy_per_dim.append(_normalized_entropy(values, bins))
        mean_entropy = sum(entropy_per_dim) / len(entropy_per_dim) if entropy_per_dim else 0.0

        threshold = Threshold(
            name="action_entropy_extremes",
            basis=ThresholdBasis.HEURISTIC_DEFAULT,
            condition=ThresholdCondition(
                operator=ThresholdOperator.OUTSIDE_RANGE,
                lower=low_entropy_threshold,
                upper=high_entropy_threshold,
                inclusive_lower=True,
                inclusive_upper=True,
                reference_statistic="normalized_entropy",
            ),
            severity_bands=[SeverityBand(severity=Severity.LOW, lower=0.0, upper=low_entropy_threshold)],
            description="Flags very low diversity or nearly random-looking action distributions as review hints.",
        )
        events = []
        if mean_entropy < low_entropy_threshold or mean_entropy > high_entropy_threshold:
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(timestamps[0], timestamps[-1]),
                    value=mean_entropy,
                    severity=Severity.LOW,
                    reason=(
                        f"mean action entropy {mean_entropy:.3f} is outside heuristic review range "
                        f"[{low_entropy_threshold:.3f}, {high_entropy_threshold:.3f}]"
                    ),
                    threshold=threshold,
                )
            )

        return MetricResult(
            name=self.name,
            description="Shannon entropy of action distribution per dimension, normalized to 0-1.",
            source_modalities=[Modality.ACTION],
            unit="normalized_entropy",
            samples=[],
            summary=summarize(entropy_per_dim),
            thresholds=[threshold],
            events=events,
            display_hint="per_action_dimension",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "entropy_per_dim": entropy_per_dim,
                "mean_entropy": mean_entropy,
                "num_bins": bins,
                "low_entropy_threshold": low_entropy_threshold,
                "high_entropy_threshold": high_entropy_threshold,
            },
        )


def _action_bounds(
    stream_metadata: dict[str, object],
    bundle_metadata: dict[str, object],
    config: dict[str, object],
) -> tuple[list[float] | None, list[float] | None]:
    low = config.get("action_low") or config.get("low")
    high = config.get("action_high") or config.get("high")
    low = low or stream_metadata.get("action_low") or stream_metadata.get("low")
    high = high or stream_metadata.get("action_high") or stream_metadata.get("high")
    action_space = bundle_metadata.get("action_space")
    if isinstance(action_space, dict):
        low = low or action_space.get("low")
        high = high or action_space.get("high")
    if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)):
        return None, None
    return [float(value) for value in low], [float(value) for value in high]


def _action_saturation_severity(
    fraction: float,
    medium_fraction: float,
    high_fraction: float,
) -> Severity:
    if fraction >= high_fraction:
        return Severity.HIGH
    if fraction >= medium_fraction:
        return Severity.MEDIUM
    return Severity.LOW


def _saturation_runs(saturated_dims_per_step: list[list[int]]) -> list[tuple[set[int], int, int]]:
    """Group consecutive saturation steps into event windows."""

    runs: list[tuple[set[int], int, int]] = []
    start_index: int | None = None
    current_dimensions: set[int] = set()

    for index, dimensions in enumerate(saturated_dims_per_step):
        if dimensions:
            if start_index is None:
                start_index = index
                current_dimensions = set(dimensions)
            else:
                current_dimensions.update(dimensions)
        elif start_index is not None:
            runs.append((current_dimensions, start_index, index - 1))
            start_index = None
            current_dimensions = set()

    if start_index is not None:
        runs.append((current_dimensions, start_index, len(saturated_dims_per_step) - 1))
    return runs

def _normalized_entropy(values: list[float], bins: int) -> float:
    if not values or bins <= 1:
        return 0.0
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return 0.0
    counts = [0 for _ in range(bins)]
    for value in values:
        index = int((value - minimum) / (maximum - minimum) * bins)
        if index == bins:
            index -= 1
        counts[index] += 1
    total = len(values)
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(bins)
