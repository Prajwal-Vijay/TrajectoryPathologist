"""Timing and synchronization metrics."""

from __future__ import annotations

from statistics import mean, median, pstdev

from pathologist.metrics.base import MetricComputationContext, metric_config, metric_option
from pathologist.metrics.helpers import find_stream, median_absolute_deviation, summarize
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


class TimestampRegularityMetric:
    """Detect dropped frames, timing jitter, and effective sampling rate."""

    name = "timestamp_regularity"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.ACTION, Modality.PROPRIOCEPTION, Modality.TELEMETRY],
            preferred_names=["action", "proprioception", "telemetry"],
        )
        if stream is None or len(stream.timestamps_sec) < 2:
            return MetricResult(
                name=self.name,
                description="Detects dropped frames, timing jitter, and frequency drift.",
                metadata={"available": False, "unavailable_reason": "requires at least two timestamps"},
            )

        timestamps = [float(timestamp) for timestamp in stream.timestamps_sec]
        dts = [
            timestamps[index] - timestamps[index - 1]
            for index in range(1, len(timestamps))
            if timestamps[index] >= timestamps[index - 1]
        ]
        if not dts:
            return MetricResult(
                name=self.name,
                description="Detects dropped frames, timing jitter, and frequency drift.",
                metadata={"available": False, "unavailable_reason": "timestamps are not monotonic"},
            )

        config = metric_config(context, self.name)
        gap_mad_multiplier = float(metric_option(config, stream.metadata, "gap_mad_multiplier", 3.0))
        gap_median_multiplier = float(metric_option(config, stream.metadata, "gap_median_multiplier", 1.0))
        high_gap_multiplier = float(metric_option(config, stream.metadata, "high_gap_multiplier", 2.0))

        dt_mean = mean(dts)
        dt_std = pstdev(dts) if len(dts) > 1 else 0.0
        dt_median = median(dts)
        dt_mad = median_absolute_deviation(dts)
        gap_threshold = dt_median + max(gap_mad_multiplier * dt_mad, gap_median_multiplier * dt_median)
        jitter_ratio = dt_std / dt_mean if dt_mean > 0 else 0.0
        gap_locations = [
            {"timestamp_sec": timestamps[index + 1], "dt_sec": dt}
            for index, dt in enumerate(dts)
            if dt > gap_threshold
        ]

        threshold = Threshold(
            name="timestamp_gap_intra_episode",
            basis=ThresholdBasis.INTRA_EPISODE_ROBUST_STAT,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=gap_threshold,
                reference_statistic="median_dt_plus_max_3mad_or_1median",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.MEDIUM, lower=gap_threshold, upper=high_gap_multiplier * gap_threshold),
                SeverityBand(severity=Severity.HIGH, lower=high_gap_multiplier * gap_threshold),
            ],
            numeric_context={
                "dt_median": dt_median,
                "dt_mad": dt_mad,
                "gap_mad_multiplier": gap_mad_multiplier,
                "gap_median_multiplier": gap_median_multiplier,
                "high_gap_multiplier": high_gap_multiplier,
            },
        )
        events = []
        for gap in gap_locations[: context.max_events_per_metric]:
            severity = Severity.HIGH if gap["dt_sec"] >= high_gap_multiplier * gap_threshold else Severity.MEDIUM
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(gap["timestamp_sec"]),
                    value=gap["dt_sec"],
                    severity=severity,
                    reason=f"timestamp gap {gap['dt_sec']:.4f}s exceeded intra-episode gap threshold {gap_threshold:.4f}s",
                    threshold=threshold,
                )
            )

        return MetricResult(
            name=self.name,
            description="Detects dropped frames, timing jitter, and frequency drift.",
            source_modalities=[stream.modality],
            unit="seconds",
            samples=[
                MetricSample(timestamp_sec=timestamps[index + 1], value=dt)
                for index, dt in enumerate(dts)
            ],
            summary=summarize(dts),
            thresholds=[threshold],
            events=events,
            metadata={
                "available": True,
                "stream_name": stream.name,
                "dt_mean": dt_mean,
                "dt_std": dt_std,
                "jitter_ratio": jitter_ratio,
                "num_gaps": len(gap_locations),
                "gap_locations": gap_locations,
                "effective_fps": 1.0 / dt_mean if dt_mean > 0 else None,
            },
        )
