"""Gripper metrics."""

from __future__ import annotations

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


class GripperChatterMetric:
    """Rapid gripper open/close transitions indicating noisy teleoperation."""

    name = "gripper_chatter"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.GRIPPER, Modality.ACTION, Modality.PROPRIOCEPTION], 
            # First try to look for gripper modality, if that does not exist, then look for action, if not look for proprioception
            preferred_names=["gripper", "action", "proprioception"],
        )
        timestamps, grippers = extract_vector_series(
            stream,
            ["gripper", "grippers", "proprioception_grippers", "gripper_state"],
        )
        if stream is None or len(grippers) < 2:
            return MetricResult(
                name=self.name,
                description="Counts rapid gripper open/close transitions.",
                metadata={"available": False, "unavailable_reason": "requires at least two gripper samples"},
            )

        config = metric_config(context, self.name)
        threshold_value = float(metric_option(config, stream.metadata, "gripper_binary_threshold", 0.5))
        binary = [
            [1 if value >= threshold_value else 0 for value in sample]
            for sample in grippers # Sample means one sample timestep
        ]
        transitions = []
        for index in range(1, len(binary)):
            width = min(len(binary[index]), len(binary[index - 1]))
            changed = sum(
                1 for dim in range(width)
                if binary[index][dim] != binary[index - 1][dim]
            )
            transitions.append(changed)

        transition_timestamps = align_timestamps(timestamps[1:], len(transitions))
        transition_samples = [
            MetricSample(timestamp_sec=timestamp, value=value)
            for timestamp, value in zip(transition_timestamps, transitions)
        ]
        total_transitions = sum(transitions)
        duration = max(timestamps[-1] - timestamps[0], 0.0)
        transitions_per_sec = total_transitions / duration if duration > 0 else 0.0
        
        sample_timestamps = align_timestamps(timestamps, len(binary))
        bin_steps = int(metric_option(config, stream.metadata, "chatter_bin_steps", 10))
        chatter_bins = _transition_bins(transitions, sample_timestamps, bin_steps)
        bin_rate_values = [rate for _, _, _, rate in chatter_bins]
        
        # Keep an absolute floor so a mostly quiet trajectory does not flag
        # its weakest nonzero bin only because it is high relative to itself.
        min_chatter_rate = float(metric_option(config, stream.metadata, "min_chatter_rate", 2.0))
        min_chatter_transitions = int(metric_option(config, stream.metadata, "min_chatter_transitions", 2))
        
        medium_percentile = float(metric_option(config, stream.metadata, "medium_percentile", 95))
        high_percentile = float(metric_option(config, stream.metadata, "high_percentile", 99))
        p95 = percentile(bin_rate_values, medium_percentile) or 0.0
        p99 = percentile(bin_rate_values, high_percentile) or p95

        effective_p95 = max(p95, min_chatter_rate)
        effective_p99 = max(p99, min_chatter_rate * 1.5)

        threshold = Threshold(
            name="gripper_chatter_binned_spike",
            basis=ThresholdBasis.INTRA_EPISODE_PERCENTILE,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=effective_p95,
                reference_statistic="max(binned_chatter_rate_p95, min_chatter_rate)",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.MEDIUM, lower=effective_p95, upper=effective_p99),
                SeverityBand(severity=Severity.HIGH, lower=effective_p99),
            ],
            description="Flags gripper open/close bursts in non-overlapping telemetry bins.",
            numeric_context={
                "binned_chatter_rate_p95": p95,
                "binned_chatter_rate_p99": p99,
                "medium_percentile": medium_percentile,
                "high_percentile": high_percentile,
                "chatter_bin_steps": bin_steps,
                "min_chatter_transitions": min_chatter_transitions,
            },
        )
        events = []
        if effective_p95 > 0.0:
            eligible_rate_values = [
                bin_rate if transition_count >= min_chatter_transitions else 0.0
                for _, _, transition_count, bin_rate in chatter_bins
            ]
            for index in top_indices(eligible_rate_values, effective_p95, context.max_events_per_metric):
                start_time, end_time, transition_count, bin_rate = chatter_bins[index]
                severity = Severity.HIGH if bin_rate >= effective_p99 else Severity.MEDIUM
                events.append(
                    MetricEvent(
                        metric_name=self.name,
                        time_window=TimeWindow(start_time, end_time),
                        value={
                            "binned_transition_rate": bin_rate,
                            "transition_count": transition_count,
                            "chatter_bin_steps": bin_steps,
                            "min_chatter_transitions": min_chatter_transitions,
                        },
                        severity=severity,
                        reason=(
                            f"gripper transition rate {bin_rate:.3f}/s in telemetry bin with "
                            f"{transition_count} transition(s) exceeded threshold {effective_p95:.3f}/s "
                            f"and minimum chatter count {min_chatter_transitions}"
                        ),
                        threshold=threshold,
                    )
                )

        return MetricResult(
            name=self.name,
            description="Counts rapid open/close gripper transitions.",
            source_modalities=[stream.modality],
            unit="transitions",
            samples=transition_samples,
            summary=summarize([float(value) for value in transitions]),
            thresholds=[threshold],
            events=events,
            metadata={
                "available": True,
                "stream_name": stream.name,
                "total_transitions": total_transitions,
                "transitions_per_sec": transitions_per_sec,
                "binary_threshold": threshold_value,
                "chatter_bin_steps": bin_steps,
                "chatter_bins": [
                    {
                        "start_sec": start_time,
                        "end_sec": end_time,
                        "transition_count": transition_count,
                        "transition_rate": local_rate,
                    }
                    for start_time, end_time, transition_count, local_rate in chatter_bins
                ],
            },
        )


def _transition_bins(
    transitions: list[int],
    sample_timestamps: list[float],
    bin_steps: int,
) -> list[tuple[float, float, int, float]]:
    """Count gripper transitions in non-overlapping telemetry bins."""

    if not transitions or len(sample_timestamps) < 2:
        return []
    bin_steps = max(bin_steps, 1)
    rates: list[tuple[float, float, int, float]] = []
    for start_index in range(0, len(transitions), bin_steps):
        end_index = min(len(transitions), start_index + bin_steps)
        start_time = sample_timestamps[start_index]
        end_time = sample_timestamps[min(end_index, len(sample_timestamps) - 1)]
        duration = max(end_time - start_time, 1e-9)
        transition_count = sum(transitions[start_index:end_index])
        rates.append((start_time, end_time, transition_count, transition_count / duration))
    return rates
