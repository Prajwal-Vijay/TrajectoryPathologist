"""Kinematic metrics from proprioception/action trajectories."""

from __future__ import annotations

from dataclasses import asdict

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.helpers import (
    align_timestamps,
    extract_vector_series,
    find_stream,
    l2_norm,
    safe_dt,
    summarize,
    vector_diff,
)
from pathologist.schemas.common import (
    Modality,
)
from pathologist.schemas.metrics import MetricResult, MetricSample


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

        duration = max(timestamps[-1] - timestamps[0], 0.0)
        peak_velocity = max((l2_norm(velocity) for velocity in velocities), default=0.0)
        mean_jerk = sum(jerks) / len(jerks) if jerks else 0.0
        normalized_jerk = None
        if duration > 0 and peak_velocity > 0:
            normalized_jerk = mean_jerk * (duration**3) / peak_velocity

        return MetricResult(
            name=self.name,
            description="Jerk norm over time used as a scalar source for trajectory-relative ECDF surprise scoring.",
            source_modalities=[stream.modality],
            unit="state_units_per_second_cubed",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(jerk_times, jerks)
            ],
            summary=summarize(jerks),
            thresholds=[],
            events=[],
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
        return MetricResult(
            name=self.name,
            description="Measured velocity norm over time used as a scalar source for trajectory-relative ECDF surprise scoring.",
            source_modalities=[stream.modality],
            unit="state_units_per_second",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(sample_timestamps, velocity_norms)
            ],
            summary=summarize(velocity_norms),
            thresholds=[],
            events=[],
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
        gripper_stream = find_stream(
            context.bundle,
            [Modality.GRIPPER],
            preferred_names=["gripper"],
        )
        _, grippers = extract_vector_series(
            gripper_stream,
            ["gripper", "grippers", "proprioception_grippers", "gripper_state"],
        )
        positions, gripper_dims_used = _append_gripper_state_if_aligned(positions, grippers)

        speeds = []
        speed_times = []
        for index in range(1, len(positions)):
            dt = safe_dt(timestamps[index - 1], timestamps[index])
            speeds.append(l2_norm(vector_diff(positions[index], positions[index - 1])) / dt)
            speed_times.append(timestamps[index])

        return MetricResult(
            name=self.name,
            description="Robot speed norm over time; low-tail rarity is handled by ECDF surprise scoring.",
            source_modalities=[stream.modality],
            unit="speed_norm",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=speed)
                for timestamp, speed in zip(speed_times, speeds)
            ],
            summary=summarize(speeds),
            thresholds=[],
            events=[],
            display_hint="speed_norm_timeseries",
            metadata={
                "available": True,
                "stream_name": stream.name,
                "gripper_stream_name": gripper_stream.name if gripper_stream is not None else None,
                "gripper_dims_used": gripper_dims_used,
                "includes_gripper_state_change": gripper_dims_used > 0,
                "surprise_tail_interpretation": "left",
            },
        )


def _append_gripper_state_if_aligned(
    positions: list[list[float]],
    grippers: list[list[float]],
) -> tuple[list[list[float]], int]:
    """Include gripper state changes in stalling only when sample-aligned."""

    if not positions or len(positions) != len(grippers):
        return positions, 0
    gripper_width = min((len(sample) for sample in grippers), default=0)
    if gripper_width <= 0:
        return positions, 0
    return [
        position + gripper[:gripper_width]
        for position, gripper in zip(positions, grippers)
    ], gripper_width


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
