"""Gripper metrics."""

from __future__ import annotations

from pathologist.metrics.base import MetricComputationContext
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

        threshold_value = float(stream.metadata.get("gripper_binary_threshold", 0.5))
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
        
        window_radius_steps = int(stream.metadata.get("chatter_window_radius_steps", 5))
        local_rates = _local_transition_rates(transitions, transition_timestamps, window_radius_steps)
        local_rate_values = [rate for _, _, _, rate in local_rates]
        
        """
        # Define an absolute minimum heuristic threshold to prevent false positives 
        # on completely normal trajectories where the "worst" part is still fine.
        NOTE
        Still maybe we need to take someother p
        """
        min_chatter_rate = float(stream.metadata.get("min_chatter_rate", 2.0))
        
        p95 = percentile(local_rate_values, 95) or 0.0
        p99 = percentile(local_rate_values, 99) or p95

        effective_p95 = max(p95, min_chatter_rate)
        effective_p99 = max(p99, min_chatter_rate * 1.5)

        threshold = Threshold(
            name="gripper_chatter_local_spike",
            basis=ThresholdBasis.INTRA_EPISODE_PERCENTILE,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=effective_p95,
                reference_statistic="max(local_chatter_rate_p95, min_chatter_rate)",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.MEDIUM, lower=effective_p95, upper=effective_p99),
                SeverityBand(severity=Severity.HIGH, lower=effective_p99),
            ],
            description="Flags local gripper open/close bursts relative to this episode.",
            numeric_context={
                "local_chatter_rate_p95": p95,
                "local_chatter_rate_p99": p99,
                "window_radius_steps": window_radius_steps,
            },
        )
        events = []
        if effective_p95 > 0.0:
            for index in top_indices(local_rate_values, effective_p95, context.max_events_per_metric):
                start_time, end_time, transition_count, local_rate = local_rates[index]
                severity = Severity.HIGH if local_rate >= effective_p99 else Severity.MEDIUM
                events.append(
                    MetricEvent(
                        metric_name=self.name,
                        time_window=TimeWindow(start_time, end_time),
                        value={
                            "local_transition_rate": local_rate,
                            "transition_count": transition_count,
                        },
                        severity=severity,
                        reason=f"local gripper transition rate {local_rate:.3f}/s exceeded threshold {effective_p95:.3f}/s",
                        threshold=threshold,
                    )
                )
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(timestamps[0], timestamps[-1]),
                    value=transitions_per_sec,
                    severity=Severity.LOW,
                    reason=f"whole-trajectory gripper transition rate was {transitions_per_sec:.3f}/s; retained as summary context",
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
                "local_chatter_rates": [
                    {
                        "start_sec": start_time,
                        "end_sec": end_time,
                        "transition_count": transition_count,
                        "transition_rate": local_rate,
                    }
                    for start_time, end_time, transition_count, local_rate in local_rates
                ],
            },
        )


def _local_transition_rates(
    transitions: list[int],
    timestamps: list[float],
    radius_steps: int,
) -> list[tuple[float, float, int, float]]:
    """Compute transition rate around each timestep using a +/- n window."""

    if not transitions or not timestamps:
        return []
    radius_steps = max(radius_steps, 1)
    rates: list[tuple[float, float, int, float]] = []
    for index in range(len(transitions)):
        start_index = max(0, index - radius_steps)
        end_index = min(len(transitions) - 1, index + radius_steps)
        start_time = timestamps[start_index]
        end_time = timestamps[end_index]
        duration = max(end_time - start_time, 1e-9)
        transition_count = sum(transitions[start_index : end_index + 1])
        rates.append((start_time, end_time, transition_count, transition_count / duration))
    return rates
