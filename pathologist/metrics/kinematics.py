"""Kinematic metrics from proprioception/action trajectories."""

from __future__ import annotations

from dataclasses import asdict

from pathologist.metrics.base import MetricComputationContext, metric_config, metric_option
from pathologist.metrics.helpers import (
    align_timestamps,
    extract_vector_series,
    find_stream,
    l2_norm,
    median_absolute_deviation,
    percentile,
    safe_dt,
    summarize,
    top_indices,
    vector_diff,
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


class JointPathLengthMetric:
    """Sum of step-to-step distances in joint/proprioception space."""

    name = "joint_path_length"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.PROPRIOCEPTION, Modality.ACTION],
            preferred_names=["proprioception", "state", "action"],
        )
        timestamps, values = extract_vector_series(stream, ["proprioception", "state", "joint_positions", "action"])
        if stream is None or len(values) < 2:
            return MetricResult(
                name=self.name,
                description="Sum of step-to-step distances in joint/proprioception space. Relative metric used to compare episodes of the same task.",
                metadata={"available": False, "unavailable_reason": "requires at least two vector samples"},
            )

        increments = [
            l2_norm(vector_diff(values[index], values[index - 1])) # We find the individual distances
            for index in range(1, len(values))
        ]
        cumulative = []
        total = 0.0
        for increment in increments:
            total += increment
            cumulative.append(total)
        sample_timestamps = align_timestamps(timestamps[1:], len(cumulative))

        return MetricResult(
            name=self.name,
            description="Sum of step-to-step distances in joint/proprioception space.",
            source_modalities=[stream.modality],
            unit="state_space_l2",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(sample_timestamps, cumulative)
            ],
            summary=summarize(increments),
            display_hint="cumulative_joint_path_length",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "joint_path_length": total,
                "num_steps": len(values),
            },
        )


class EndEffectorCartesianPathLengthMetric:
    """Sum of end-effector/wrist movement distances in Cartesian space."""

    name = "end_effector_cartesian_path_length"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.TELEMETRY, Modality.PROPRIOCEPTION],
            preferred_names=["telemetry", "proprioception", "state"],
        )
        timestamps, wrist_positions = _extract_wrist_positions(stream)
        if stream is None or len(timestamps) < 2 or not wrist_positions:
            return MetricResult(
                name=self.name,
                description="Sum of wrist/end-effector movement distances in Cartesian space.",
                metadata={
                    "available": False,
                    "unavailable_reason": "requires records with wrist_poses[*].position",
                },
            )

        wrist_names = sorted(wrist_positions)
        per_wrist_increments: dict[str, list[float]] = {}
        per_wrist_cumulative: dict[str, list[float]] = {}
        per_wrist_path_length: dict[str, float] = {}

        for wrist_name in wrist_names:
            positions = wrist_positions[wrist_name]
            increments = [
                l2_norm(vector_diff(positions[index], positions[index - 1]))
                for index in range(1, len(positions))
            ]
            cumulative = []
            total = 0.0
            for increment in increments:
                total += increment
                cumulative.append(total)
            per_wrist_increments[wrist_name] = increments
            per_wrist_cumulative[wrist_name] = cumulative
            per_wrist_path_length[wrist_name] = total

        sample_timestamps = align_timestamps(timestamps[1:], len(timestamps) - 1)
        samples = []
        for index, timestamp in enumerate(sample_timestamps):
            cumulative_values = [
                per_wrist_cumulative[wrist_name][index]
                for wrist_name in wrist_names
                if index < len(per_wrist_cumulative[wrist_name])
            ]
            if cumulative_values:
                samples.append(MetricSample(timestamp_sec=timestamp, value=cumulative_values))

        per_wrist_step_summary = {
            wrist_name: asdict(summarize(per_wrist_increments[wrist_name]))
            for wrist_name in wrist_names
        }
        total_path_length = sum(per_wrist_path_length.values())
        mean_path_length = total_path_length / len(per_wrist_path_length) if per_wrist_path_length else 0.0
        max_path_length = max(per_wrist_path_length.values()) if per_wrist_path_length else 0.0
        min_path_length = min(per_wrist_path_length.values()) if per_wrist_path_length else 0.0
        path_length_imbalance = (
            (max_path_length - min_path_length) / total_path_length
            if total_path_length > 0 and len(per_wrist_path_length) > 1
            else 0.0
        )

        return MetricResult(
            name=self.name,
            description="Sum of wrist/end-effector movement distances in Cartesian space.",
            source_modalities=[stream.modality],
            unit="meters",
            samples=samples,
            summary=summarize([]), # use per_wrist_step_summary
            display_hint="cumulative_cartesian_path_length_per_wrist",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "wrist_names": wrist_names,
                "per_wrist_path_length_m": per_wrist_path_length,
                "per_wrist_step_distance_summary_m": per_wrist_step_summary,
                "total_wrist_path_length_m": total_path_length,
                "mean_wrist_path_length_m": mean_path_length,
                "max_wrist_path_length_m": max_path_length,
                "min_wrist_path_length_m": min_path_length,
                "path_length_imbalance": path_length_imbalance,
                "num_steps": len(timestamps),
            },
        )


class MotionSmoothnessMetric:
    """Jerk-based smoothness profile from vector trajectories."""

    name = "motion_smoothness"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.PROPRIOCEPTION, Modality.ACTION],
            preferred_names=["proprioception", "state", "action"],
        )
        timestamps, positions = extract_vector_series(stream, ["proprioception", "state", "joint_positions", "action"])
        if stream is None or len(positions) < 4:
            return MetricResult(
                name=self.name,
                description="Jerk-based smoothness metric computed from vector trajectory.",
                metadata={"available": False, "unavailable_reason": "requires at least four vector samples"},
            )

        velocities = []
        velocity_times = []
        for index in range(1, len(positions)):
            dt = safe_dt(timestamps[index - 1], timestamps[index])
            velocities.append([item / dt for item in vector_diff(positions[index], positions[index - 1])])
            velocity_times.append(timestamps[index])

        accelerations = []
        acceleration_times = []
        for index in range(1, len(velocities)):
            dt = safe_dt(velocity_times[index - 1], velocity_times[index])
            accelerations.append([item / dt for item in vector_diff(velocities[index], velocities[index - 1])])
            acceleration_times.append(velocity_times[index])

        jerks = []
        jerk_times = []
        for index in range(1, len(accelerations)):
            dt = safe_dt(acceleration_times[index - 1], acceleration_times[index])
            jerk_vector = [item / dt for item in vector_diff(accelerations[index], accelerations[index - 1])]
            jerks.append(l2_norm(jerk_vector))
            jerk_times.append(acceleration_times[index])

        config = metric_config(context, self.name)
        baseline_median = percentile(jerks, 50) or 0.0
        baseline_mad = median_absolute_deviation(jerks)
        max_tolerable_jerk = float(metric_option(config, stream.metadata, "max_tolerable_jerk", 0.05))
        
        medium_percentile = float(metric_option(config, stream.metadata, "medium_percentile", 95))
        high_percentile = float(metric_option(config, stream.metadata, "high_percentile", 99))
        robust_medium_mad_multiplier = float(metric_option(config, stream.metadata, "robust_medium_mad_multiplier", 3.0))
        robust_high_mad_multiplier = float(metric_option(config, stream.metadata, "robust_high_mad_multiplier", 6.0))
        p95 = percentile(jerks, medium_percentile) or 0.0
        p99 = percentile(jerks, high_percentile) or p95
        robust_medium_threshold = baseline_median + robust_medium_mad_multiplier * baseline_mad
        robust_high_threshold = baseline_median + robust_high_mad_multiplier * baseline_mad
        numerical_zero_jerk_norm = float(metric_option(config, stream.metadata, "numerical_zero_jerk_norm", 1e-9))
        min_jerk_norm = float(metric_option(config, stream.metadata, "min_jerk_norm", 0.0))
        effective_p95 = max(p95, robust_medium_threshold, numerical_zero_jerk_norm, max_tolerable_jerk, min_jerk_norm)
        effective_p99 = max(p99, robust_high_threshold, effective_p95)

        threshold = Threshold(
            name="jerk_spike_intra_episode",
            basis=ThresholdBasis.INTRA_EPISODE_ROBUST_STAT,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=effective_p95,
                reference_statistic="max(episode_p95, median_plus_3mad, min_jerk_norm)",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.MEDIUM, lower=effective_p95, upper=effective_p99),
                SeverityBand(severity=Severity.HIGH, lower=effective_p99),
            ],
            numeric_context={
                "baseline_median": baseline_median,
                "baseline_mad": baseline_mad,
                "median_plus_3mad": robust_medium_threshold,
                "median_plus_6mad": robust_high_threshold,
                "episode_p95": p95,
                "episode_p99": p99,
                "medium_percentile": medium_percentile,
                "high_percentile": high_percentile,
                "robust_medium_mad_multiplier": robust_medium_mad_multiplier,
                "robust_high_mad_multiplier": robust_high_mad_multiplier,
                "max_tolerable_jerk": max_tolerable_jerk,
                "numerical_zero_jerk_norm": numerical_zero_jerk_norm,
                "min_jerk_norm": min_jerk_norm,
                "effective_medium_threshold": effective_p95,
                "effective_high_threshold": effective_p99,
            },
        )
        
        events = []
        if effective_p95 > 0.0:
            for index in top_indices(jerks, effective_p95, context.max_events_per_metric):
                severity = Severity.HIGH if jerks[index] >= effective_p99 else Severity.MEDIUM
                events.append(
                    MetricEvent(
                        metric_name=self.name,
                        time_window=TimeWindow(jerk_times[index]),
                        value=jerks[index],
                        severity=severity,
                        reason=(
                            f"jerk norm {jerks[index]:.4f} exceeded effective intra-episode "
                            f"jerk threshold {effective_p95:.4f} "
                            f"(p95={p95:.4f}, median+3MAD={robust_medium_threshold:.4f})"
                        ),
                        threshold=threshold,
                    )
                )

        duration = max(timestamps[-1] - timestamps[0], 0.0)
        peak_velocity = max((l2_norm(velocity) for velocity in velocities), default=0.0)
        mean_jerk = sum(jerks) / len(jerks) if jerks else 0.0
        normalized_jerk = None
        if duration > 0 and peak_velocity > 0:
            normalized_jerk = mean_jerk * (duration**3) / peak_velocity

        return MetricResult(
            name=self.name,
            description="Jerk norm over time; spikes are flagged relative to this trajectory.",
            source_modalities=[stream.modality],
            unit="state_units_per_second_cubed",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(jerk_times, jerks)
            ],
            summary=summarize(jerks),
            thresholds=[threshold],
            events=events,
            display_hint="jerk_norm_timeseries",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "mean_jerk_norm": mean_jerk,
                "normalized_jerk": normalized_jerk,
                "peak_velocity_norm": peak_velocity,
            },
        )


class VelocitySpikeMetric:
    """Measured joint/state velocity spikes from explicit velocity streams."""

    name = "velocity_spike"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.PROPRIOCEPTION, Modality.TELEMETRY],
            preferred_names=["velocity", "qvel", "joint_velocity", "proprioception"],
        )
        timestamps, velocities = extract_vector_series(
            stream,
            ["qvel", "velocity", "velocities", "joint_velocity", "joint_velocities"],
        )
        if stream is None or len(velocities) < 2:
            return MetricResult(
                name=self.name,
                description="Measured velocity norm spikes from explicit velocity observations.",
                metadata={"available": False, "unavailable_reason": "requires at least two velocity samples"},
            )

        velocity_norms = [l2_norm(velocity) for velocity in velocities]
        sample_timestamps = align_timestamps(timestamps, len(velocity_norms))
        config = metric_config(context, self.name)
        medium_percentile = float(metric_option(config, stream.metadata, "medium_percentile", 95))
        high_percentile = float(metric_option(config, stream.metadata, "high_percentile", 99))
        robust_medium_mad_multiplier = float(metric_option(config, stream.metadata, "robust_medium_mad_multiplier", 3.0))
        robust_high_mad_multiplier = float(metric_option(config, stream.metadata, "robust_high_mad_multiplier", 6.0))
        min_velocity_norm = float(metric_option(config, stream.metadata, "min_velocity_norm", 0.0))
        min_region_duration_sec = float(metric_option(config, stream.metadata, "min_region_duration_sec", 0.0))

        baseline_median = percentile(velocity_norms, 50) or 0.0
        baseline_mad = median_absolute_deviation(velocity_norms)
        p95 = percentile(velocity_norms, medium_percentile) or 0.0
        p99 = percentile(velocity_norms, high_percentile) or p95
        robust_medium_threshold = baseline_median + robust_medium_mad_multiplier * baseline_mad
        robust_high_threshold = baseline_median + robust_high_mad_multiplier * baseline_mad
        effective_p95 = max(p95, robust_medium_threshold, min_velocity_norm)
        effective_p99 = max(p99, robust_high_threshold, effective_p95)

        threshold = Threshold(
            name="velocity_spike_intra_episode",
            basis=ThresholdBasis.INTRA_EPISODE_ROBUST_STAT,
            condition=ThresholdCondition(
                operator=ThresholdOperator.GREATER_THAN,
                threshold_value=effective_p95,
                reference_statistic="max(velocity_norm_p95, median_plus_3mad, min_velocity_norm)",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.MEDIUM, lower=effective_p95, upper=effective_p99),
                SeverityBand(severity=Severity.HIGH, lower=effective_p99),
            ],
            numeric_context={
                "baseline_median": baseline_median,
                "baseline_mad": baseline_mad,
                "median_plus_3mad": robust_medium_threshold,
                "median_plus_6mad": robust_high_threshold,
                "episode_p95": p95,
                "episode_p99": p99,
                "medium_percentile": medium_percentile,
                "high_percentile": high_percentile,
                "min_velocity_norm": min_velocity_norm,
                "min_region_duration_sec": min_region_duration_sec,
                "effective_medium_threshold": effective_p95,
                "effective_high_threshold": effective_p99,
            },
        )

        regions = _above_threshold_regions(velocity_norms, sample_timestamps, effective_p95)
        regions.sort(key=lambda region: region["peak_value"], reverse=True)
        events = []
        for region in regions:
            if region["duration_sec"] < min_region_duration_sec:
                continue
            severity = Severity.HIGH if region["peak_value"] >= effective_p99 else Severity.MEDIUM
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(region["start_sec"], region["end_sec"]),
                    value={
                        "peak_velocity_norm": region["peak_value"],
                        "mean_velocity_norm": region["mean_value"],
                        "duration_sec": region["duration_sec"],
                        "sample_count": region["sample_count"],
                        "peak_timestamp_sec": region["peak_timestamp_sec"],
                    },
                    severity=severity,
                    reason=(
                        f"velocity norm peaked at {region['peak_value']:.4f}, exceeding "
                        f"effective intra-episode threshold {effective_p95:.4f}"
                    ),
                    threshold=threshold,
                )
            )
            if len(events) >= context.max_events_per_metric:
                break

        return MetricResult(
            name=self.name,
            description="Measured velocity norm over time; contiguous high-velocity regions are key-event candidates.",
            source_modalities=[stream.modality],
            unit="state_units_per_second",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(sample_timestamps, velocity_norms)
            ],
            summary=summarize(velocity_norms),
            thresholds=[threshold],
            events=events,
            display_hint="velocity_norm_timeseries",
            metadata={
                "available": True,
                "stream_name": stream.name,
            },
        )


class StallingMetric:
    """Fraction of the episode where robot motion is near zero."""

    name = "stalling"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        stream = find_stream(
            context.bundle,
            [Modality.PROPRIOCEPTION, Modality.ACTION],
            preferred_names=["proprioception", "state", "action"],
        )
        timestamps, positions = extract_vector_series(stream, ["proprioception", "state", "joint_positions", "action"])
        if stream is None or len(positions) < 2:
            return MetricResult(
                name=self.name,
                description="Fraction of episode with little robot motion.",
                metadata={"available": False, "unavailable_reason": "requires at least two vector samples"},
            )

        speeds = []
        speed_times = []
        for index in range(1, len(positions)):
            dt = safe_dt(timestamps[index - 1], timestamps[index])
            speeds.append(l2_norm(vector_diff(positions[index], positions[index - 1])) / dt)
            speed_times.append(timestamps[index])

        config = metric_config(context, self.name)
        positive_speeds = [speed for speed in speeds if speed > 0.0]
        speed_percentile = float(metric_option(config, stream.metadata, "speed_percentile", 10))
        speed_median_fraction = float(metric_option(config, stream.metadata, "speed_median_fraction", 0.05))
        low_fraction = float(metric_option(config, stream.metadata, "low_fraction", 0.10))
        medium_fraction = float(metric_option(config, stream.metadata, "medium_fraction", 0.20))
        high_fraction = float(metric_option(config, stream.metadata, "high_fraction", 0.40))
        p10 = percentile(positive_speeds, speed_percentile) or 0.0
        p50 = percentile(positive_speeds, 50) or 0.0
        stall_threshold = min(p10, speed_median_fraction * p50) if p50 > 0 else p10
        stalled = [speed <= stall_threshold for speed in speeds]
        stall_fraction = sum(1 for value in stalled if value) / len(stalled) if stalled else 0.0
        threshold = Threshold(
            name="stall_fraction_intra_episode",
            basis=ThresholdBasis.INTRA_EPISODE_PERCENTILE,
            condition=ThresholdCondition(
                operator=ThresholdOperator.LESS_THAN_OR_EQUAL,
                threshold_value=stall_threshold,
                reference_statistic="min(speed_percentile, speed_median_fraction * speed_p50)",
            ),
            severity_bands=[
                SeverityBand(severity=Severity.LOW, lower=low_fraction, upper=medium_fraction),
                SeverityBand(severity=Severity.MEDIUM, lower=medium_fraction, upper=high_fraction),
                SeverityBand(severity=Severity.HIGH, lower=high_fraction),
            ],
            numeric_context={
                "speed_percentile_value": p10,
                "speed_percentile": speed_percentile,
                "speed_p50": p50,
                "speed_median_fraction": speed_median_fraction,
                "stall_fraction": stall_fraction,
            },
        )
        events = []
        for start_index, end_index in _true_runs(stalled):
            start_time = speed_times[start_index]
            end_time = speed_times[end_index]
            duration = end_time - start_time
            if duration <= 0:
                continue
            events.append(
                MetricEvent(
                    metric_name=self.name,
                    time_window=TimeWindow(start_time, end_time),
                    value=duration,
                    severity=_stall_severity(stall_fraction, medium_fraction, high_fraction),
                    reason=f"low-motion segment lasted {duration:.3f}s using intra-episode speed threshold {stall_threshold:.4f}",
                    threshold=threshold,
                )
            )
            if len(events) >= context.max_events_per_metric:
                break

        return MetricResult(
            name=self.name,
            description="Fraction of episode where the robot is essentially not moving.",
            source_modalities=[stream.modality],
            unit="speed_norm",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=speed)
                for timestamp, speed in zip(speed_times, speeds)
            ],
            summary=summarize(speeds),
            thresholds=[threshold],
            events=events,
            display_hint="speed_norm_timeseries",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "stall_fraction": stall_fraction,
                "stall_speed_threshold": stall_threshold,
                "speed_percentile": speed_percentile,
                "speed_median_fraction": speed_median_fraction,
            },
        )


def _stall_severity(
    stall_fraction: float,
    medium_fraction: float,
    high_fraction: float,
) -> Severity:
    if stall_fraction >= high_fraction:
        return Severity.HIGH
    if stall_fraction >= medium_fraction:
        return Severity.MEDIUM
    return Severity.LOW


def _true_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _above_threshold_regions(
    values: list[float],
    timestamps: list[float],
    threshold: float,
) -> list[dict[str, float | int]]:
    regions: list[dict[str, float | int]] = []
    above_threshold = [value >= threshold for value in values]
    for start_index, end_index in _true_runs(above_threshold):
        region_values = values[start_index:end_index + 1]
        peak_value = max(region_values)
        peak_offset = region_values.index(peak_value)
        start_sec = timestamps[start_index]
        end_sec = timestamps[end_index]
        regions.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "sample_count": len(region_values),
                "peak_value": peak_value,
                "mean_value": sum(region_values) / len(region_values),
                "peak_timestamp_sec": timestamps[start_index + peak_offset],
            }
        )
    return regions


def _extract_wrist_positions(stream) -> tuple[list[float], dict[str, list[list[float]]]]:
    """Extract wrist position vectors from records containing wrist_poses."""

    if stream is None or stream.records is None:
        return [], {}

    timestamps: list[float] = []
    positions_by_wrist: dict[str, list[list[float]]] = {}

    for index, record in enumerate(stream.records):
        wrist_poses = record.get("wrist_poses")
        if not isinstance(wrist_poses, dict):
            continue
        if stream.timestamps_sec and index < len(stream.timestamps_sec):
            timestamp = float(stream.timestamps_sec[index])
        else:
            timestamp = float(record.get("timestamp_sec", index))

        found_position = False
        for wrist_name, pose in wrist_poses.items():
            if not isinstance(pose, dict):
                continue
            raw_position = pose.get("position")
            if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 3:
                continue
            position = [
                float(raw_position[0]),
                float(raw_position[1]),
                float(raw_position[2]),
            ]
            positions_by_wrist.setdefault(str(wrist_name), []).append(position)
            found_position = True
        if found_position:
            timestamps.append(timestamp)

    positions_by_wrist = {
        wrist_name: positions
        for wrist_name, positions in positions_by_wrist.items()
        if len(positions) == len(timestamps)
    }
    return timestamps, positions_by_wrist
