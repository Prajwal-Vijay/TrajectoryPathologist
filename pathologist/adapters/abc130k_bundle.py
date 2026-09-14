"""Adapter for annotated XDOF ABC-130K MCAP trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TaskContract,
    TrajectoryBundle,
)


CAMERA_TOPICS = {
    "top": "/top-camera",
    "left_wrist": "/left-wrist-camera",
    "right_wrist": "/right-wrist-camera",
}
ROBOT_TOPICS = {
    "/left-arm-action",
    "/right-arm-action",
    "/left-ee-action",
    "/right-ee-action",
    "/left-arm-state",
    "/right-arm-state",
    "/left-ee-state",
    "/right-ee-state",
}


@dataclass(frozen=True, slots=True)
class ABC130KSegment:
    """One interval implied by an ABC-130K subtask annotation."""

    segment_id: str
    start_sec: float
    end_sec: float
    text: str
    failure: bool
    ignored: bool = False

    def to_dict(self) -> dict[str, Any]:
        success = None if self.ignored else not self.failure
        return {
            "segment_id": self.segment_id,
            "level": "subtask",
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "failure": self.failure,
            "ignored": self.ignored,
            "success": success,
            "status": "ignored" if self.ignored else ("failure" if self.failure else "success"),
        }


class ABC130KBundleAdapter:
    """Load one ABC-130K ``episode.mcap`` and its annotation MCAP."""

    def load(
        self,
        episode_path: str | Path,
        annotation_path: str | Path | None = None,
        *,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
        frame_cache_dir: str | Path | None = None,
        camera_name: str | None = "top",
        camera_names: list[str] | tuple[str, ...] | None = None,
        camera_sample_hz: float | None = None,
        robot_downsample_hz: float | None = None,
        extract_frames: bool = True,
    ) -> TrajectoryBundle:
        """Load telemetry, one camera, and interval-valued failure labels.

        ABC annotations are change points. Each annotation therefore covers the
        half-open interval from its timestamp until the next annotation. The exact
        case-insensitive label ``mistake`` is considered a failure.
        """

        resolved_episode = _resolve_episode_path(episode_path)
        resolved_annotation = _resolve_annotation_path(resolved_episode, annotation_path)
        resolved_camera_names = _resolve_camera_names(
            camera_name=camera_name,
            camera_names=camera_names,
        )
        if camera_sample_hz is not None and camera_sample_hz <= 0:
            raise ValueError("camera_sample_hz must be positive when provided.")
        if robot_downsample_hz is not None and robot_downsample_hz <= 0:
            raise ValueError("robot_downsample_hz must be positive when provided.")

        make_reader, decoder_factory = _mcap_imports()
        with resolved_episode.open("rb") as handle:
            summary = make_reader(handle).get_summary()
        if summary is None or summary.statistics is None:
            raise ValueError(f"ABC-130K episode has no MCAP summary statistics: {resolved_episode}")
        origin_ns = int(summary.statistics.message_start_time)
        end_ns = int(summary.statistics.message_end_time)
        duration_sec = max(0.0, (end_ns - origin_ns) / 1e9)

        segments = load_abc130k_segments(
            resolved_annotation,
            episode_origin_ns=origin_ns,
            episode_end_ns=end_ns,
        )
        cache_root = (
            Path(frame_cache_dir).expanduser().resolve()
            if frame_cache_dir is not None
            else resolved_episode.parent / "pathologist_frame_cache"
        )
        topics = ["/instruction", *sorted(ROBOT_TOPICS)]
        if extract_frames:
            topics.extend(CAMERA_TOPICS[name] for name in resolved_camera_names)

        samples: dict[str, dict[int, list[float]]] = {
            topic: {} for topic in ROBOT_TOPICS
        }
        instruction: str | None = None
        frames: list[CameraFrameRef] = []
        camera_by_topic = {
            CAMERA_TOPICS[name]: name for name in resolved_camera_names
        }
        codecs = (
            {name: _new_h264_decoder() for name in resolved_camera_names}
            if extract_frames
            else {}
        )
        pending_frame_times: dict[str, list[tuple[int, int]]] = {
            name: [] for name in resolved_camera_names
        }
        last_saved_sec: dict[str, float | None] = {
            name: None for name in resolved_camera_names
        }
        camera_message_indices = {name: 0 for name in resolved_camera_names}
        frame_dirs = {
            name: cache_root / _camera_cache_name(name, camera_sample_hz)
            for name in resolved_camera_names
        }

        with resolved_episode.open("rb") as handle:
            reader = make_reader(handle, decoder_factories=[decoder_factory()])
            for _, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
                topic = channel.topic
                timestamp_ns = _message_timestamp_ns(decoded, int(message.log_time))
                if topic == "/instruction":
                    instruction = str(getattr(decoded, "data", "")).strip() or instruction
                    continue
                if topic in ROBOT_TOPICS:
                    value = [float(item) for item in getattr(decoded, "position", [])]
                    if topic.endswith("-arm-state") and not value:
                        value = [float(item) for item in getattr(decoded, "pose", [])]
                    samples[topic][timestamp_ns] = value
                    if "-arm-" in topic:
                        pose = [float(item) for item in getattr(decoded, "pose", [])]
                        torque = [float(item) for item in getattr(decoded, "torque", [])]
                        samples.setdefault(f"{topic}:pose", {})[timestamp_ns] = pose
                        samples.setdefault(f"{topic}:torque", {})[timestamp_ns] = torque
                    if topic.endswith("-arm-state"):
                        velocity = [float(item) for item in getattr(decoded, "velocity", [])]
                        samples.setdefault(f"{topic}:velocity", {})[timestamp_ns] = velocity
                    continue
                active_camera_name = camera_by_topic.get(topic)
                if active_camera_name is None:
                    continue
                pending_frame_times[active_camera_name].append(
                    (
                        timestamp_ns,
                        camera_message_indices[active_camera_name],
                    )
                )
                camera_message_indices[active_camera_name] += 1
                for video_frame in codecs[active_camera_name].decode(
                    _av_packet(bytes(decoded.data))
                ):
                    frame_timestamp_ns, frame_index = pending_frame_times[
                        active_camera_name
                    ].pop(0)
                    timestamp_sec = max(0.0, (frame_timestamp_ns - origin_ns) / 1e9)
                    if _sample_timestamp(
                        timestamp_sec,
                        last_saved_sec[active_camera_name],
                        camera_sample_hz,
                    ):
                        frames.append(
                            _materialize_frame(
                                video_frame,
                                frame_dir=frame_dirs[active_camera_name],
                                camera_name=active_camera_name,
                                timestamp_sec=timestamp_sec,
                                frame_index=frame_index,
                            )
                        )
                        last_saved_sec[active_camera_name] = timestamp_sec
            for active_camera_name, codec in codecs.items():
                for video_frame in codec.decode(None):
                    if not pending_frame_times[active_camera_name]:
                        break
                    frame_timestamp_ns, frame_index = pending_frame_times[
                        active_camera_name
                    ].pop(0)
                    timestamp_sec = max(0.0, (frame_timestamp_ns - origin_ns) / 1e9)
                    if _sample_timestamp(
                        timestamp_sec,
                        last_saved_sec[active_camera_name],
                        camera_sample_hz,
                    ):
                        frames.append(
                            _materialize_frame(
                                video_frame,
                                frame_dir=frame_dirs[active_camera_name],
                                camera_name=active_camera_name,
                                timestamp_sec=timestamp_sec,
                                frame_index=frame_index,
                            )
                        )
                        last_saved_sec[active_camera_name] = timestamp_sec

        streams = _build_robot_streams(
            samples,
            origin_ns=origin_ns,
            target_hz=robot_downsample_hz,
            episode_path=resolved_episode,
        )
        task_name = resolved_episode.parent.parent.name
        failed_segments = [segment for segment in segments if segment.failure]
        ignored_segments = [segment for segment in segments if segment.ignored]
        raw_metadata = {
            "dataset": "ABC-130K",
            "task_name": task_name,
            "instruction": instruction or task_name.replace("_", " "),
            "episode_path": str(resolved_episode),
            "annotation_path": str(resolved_annotation),
            "timestamp_origin_ns": origin_ns,
            "duration_sec": duration_sec,
            "camera_name": (
                resolved_camera_names[0] if len(resolved_camera_names) == 1 else None
            ),
            "camera_names": list(resolved_camera_names),
            "camera_topics": [
                CAMERA_TOPICS[name] for name in resolved_camera_names
            ],
            "camera_sample_hz": camera_sample_hz,
            "decoded_frame_count": len(frames),
            "robot_downsample_hz": robot_downsample_hz,
            "abc130k_segments": [segment.to_dict() for segment in segments],
            "failed_segment_count": len(failed_segments),
            "ignored_segment_count": len(ignored_segments),
            "has_failed_segment": bool(failed_segments),
            "failure_label": "mistake",
            "annotation_interval_semantics": "[annotation timestamp, next annotation timestamp)",
        }
        return TrajectoryBundle(
            bundle_id=bundle_id or resolved_episode.parent.name,
            source_format=BundleFormat.ABC130K_MCAP,
            root_path=resolved_episode,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=sorted(frames, key=lambda frame: frame.timestamp_sec),
            raw_metadata=raw_metadata,
        )


def load_abc130k_segments(
    annotation_path: str | Path,
    *,
    episode_origin_ns: int,
    episode_end_ns: int,
) -> list[ABC130KSegment]:
    """Decode ABC subtask change points into contiguous intervals."""

    resolved = Path(annotation_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find ABC-130K annotation MCAP: {resolved}")
    make_reader, decoder_factory = _mcap_imports()
    annotations: list[tuple[int, str]] = []
    with resolved.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[decoder_factory()])
        for _, _, message, decoded in reader.iter_decoded_messages(
            topics=["/subtask-annotation"]
        ):
            timestamp_ns = _message_timestamp_ns(decoded, int(message.log_time))
            text = str(getattr(decoded, "data", "")).strip()
            if text:
                annotations.append((timestamp_ns, text))
    return _segments_from_annotations(
        annotations,
        episode_origin_ns=episode_origin_ns,
        episode_end_ns=episode_end_ns,
    )


def _segments_from_annotations(
    annotations: list[tuple[int, str]],
    *,
    episode_origin_ns: int,
    episode_end_ns: int,
) -> list[ABC130KSegment]:
    ordered = sorted(annotations, key=lambda item: item[0])
    output: list[ABC130KSegment] = []
    for index, (start_ns, text) in enumerate(ordered):
        end_ns = ordered[index + 1][0] if index + 1 < len(ordered) else episode_end_ns
        clipped_start = min(max(start_ns, episode_origin_ns), episode_end_ns)
        clipped_end = min(max(end_ns, clipped_start), episode_end_ns)
        output.append(
            ABC130KSegment(
                segment_id=f"subtask_{index:04d}",
                start_sec=(clipped_start - episode_origin_ns) / 1e9,
                end_sec=(clipped_end - episode_origin_ns) / 1e9,
                text=text,
                failure=text.casefold() == "mistake",
                ignored=text.casefold() == "reset both arms",
            )
        )
    return output


def _build_robot_streams(
    samples: dict[str, dict[int, list[float]]],
    *,
    origin_ns: int,
    target_hz: float | None,
    episode_path: Path,
) -> list[ModalityStream]:
    metadata = {
        "dataset": "ABC-130K",
        "episode_path": str(episode_path),
        "timestamp_origin_ns": origin_ns,
        "robot_downsample_hz": target_hz,
    }
    streams: list[ModalityStream] = []

    action = _paired_samples(samples, "/left-arm-action", "/right-arm-action", origin_ns)
    action = _downsample_records(action, target_hz)
    if action:
        timestamps = [item[0] for item in action]
        streams.append(
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=[
                    {"timestamp_sec": timestamp, "action": value, "actions": value}
                    for timestamp, value in action
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-arm-action", "/right-arm-action"],
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-arm-action",
                        "/right-arm-action",
                    ),
                },
            )
        )
        streams[-1].metadata.update(
            {
                "vector_widths": sorted({len(value) for _, value in action}),
                "gripper_action_topics_excluded": ["/left-ee-action", "/right-ee-action"],
            }
        )

    state = _paired_samples(samples, "/left-arm-state", "/right-arm-state", origin_ns)
    state = _downsample_records(state, target_hz)
    if state:
        timestamps = [item[0] for item in state]
        streams.append(
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {
                        "timestamp_sec": timestamp,
                        "proprioception": value,
                        "joint_positions": value,
                        "state": value,
                    }
                    for timestamp, value in state
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-arm-state", "/right-arm-state"],
                    "vector_widths": sorted({len(value) for _, value in state}),
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-arm-state",
                        "/right-arm-state",
                    ),
                },
            )
        )

    velocity = _paired_samples(
        samples,
        "/left-arm-state:velocity",
        "/right-arm-state:velocity",
        origin_ns,
    )
    velocity = _downsample_records(velocity, target_hz)
    if velocity:
        timestamps = [item[0] for item in velocity]
        streams.append(
            ModalityStream(
                name="velocity",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {
                        "timestamp_sec": timestamp,
                        "qvel": value,
                        "velocity": value,
                        "joint_velocities": value,
                    }
                    for timestamp, value in velocity
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-arm-state", "/right-arm-state"],
                    "source_field": "velocity",
                    "vector_widths": sorted({len(value) for _, value in velocity}),
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-arm-state:velocity",
                        "/right-arm-state:velocity",
                    ),
                },
            )
        )

    gripper_action = _paired_samples(
        samples,
        "/left-ee-action",
        "/right-ee-action",
        origin_ns,
    )
    gripper_action = _downsample_records(gripper_action, target_hz)
    if gripper_action:
        timestamps = [item[0] for item in gripper_action]
        streams.append(
            ModalityStream(
                name="gripper_action",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {"timestamp_sec": timestamp, "gripper_action": value}
                    for timestamp, value in gripper_action
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-ee-action", "/right-ee-action"],
                    "vector_widths": sorted({len(value) for _, value in gripper_action}),
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-ee-action",
                        "/right-ee-action",
                    ),
                },
            )
        )

    gripper = _paired_samples(samples, "/left-ee-state", "/right-ee-state", origin_ns)
    gripper = _downsample_records(gripper, target_hz)
    if gripper:
        timestamps = [item[0] for item in gripper]
        streams.append(
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {
                        "timestamp_sec": timestamp,
                        "gripper": value,
                        "gripper_state": value,
                    }
                    for timestamp, value in gripper
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-ee-state", "/right-ee-state"],
                    "gripper_binary_threshold": 0.5,
                    "vector_widths": sorted({len(value) for _, value in gripper}),
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-ee-state",
                        "/right-ee-state",
                    ),
                },
            )
        )

    torque = _paired_samples(
        samples,
        "/left-arm-state:torque",
        "/right-arm-state:torque",
        origin_ns,
    )
    torque = _downsample_records(torque, target_hz)
    if torque:
        timestamps = [item[0] for item in torque]
        streams.append(
            ModalityStream(
                name="joint_torque",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=[
                    {"timestamp_sec": timestamp, "joint_torque": value, "torque": value}
                    for timestamp, value in torque
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-arm-state", "/right-arm-state"],
                    "source_field": "torque",
                    "vector_widths": sorted({len(value) for _, value in torque}),
                    "used_by_prism_surprise": False,
                    "vector_groups": _paired_vector_groups(
                        samples,
                        "/left-arm-state:torque",
                        "/right-arm-state:torque",
                    ),
                },
            )
        )

    wrist_poses = _paired_wrist_pose_records(samples, origin_ns)
    wrist_poses = _downsample_pose_records(wrist_poses, target_hz)
    if wrist_poses:
        eef_velocity_records = _eef_velocity_records(wrist_poses)
        if eef_velocity_records:
            streams.append(
                ModalityStream(
                    name="eef_velocity",
                    modality=Modality.TELEMETRY,
                    timestamps_sec=[
                        record["timestamp_sec"] for record in eef_velocity_records
                    ],
                    records=eef_velocity_records,
                    metadata={
                        **metadata,
                        "source_topics": ["/left-arm-state", "/right-arm-state"],
                        "source_field": "pose",
                        "derivative_source": "wrist_pose_finite_difference",
                        "linear_components": ["vx", "vy", "vz"],
                        "angular_components": ["wx", "wy", "wz"],
                        "vector_groups": {
                            "left": [0, 1, 2],
                            "right": [3, 4, 5],
                        },
                    },
                )
            )
        streams.append(
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=[item[0] for item in wrist_poses],
                records=[
                    {"timestamp_sec": timestamp, "wrist_poses": poses}
                    for timestamp, poses in wrist_poses
                ],
                metadata={
                    **metadata,
                    "source_topics": ["/left-arm-state", "/right-arm-state"],
                    "source_field": "pose",
                    "pose_representation": "row_major_4x4_transform",
                    "used_by_prism_surprise": False,
                },
            )
        )
    return streams


def _paired_samples(
    samples: dict[str, dict[int, list[float]]],
    left_key: str,
    right_key: str,
    origin_ns: int,
) -> list[tuple[float, list[float]]]:
    left = samples.get(left_key, {})
    right = samples.get(right_key, {})
    timestamps = sorted(set(left) & set(right))
    return [
        (max(0.0, (timestamp_ns - origin_ns) / 1e9), left[timestamp_ns] + right[timestamp_ns])
        for timestamp_ns in timestamps
        if left[timestamp_ns] and right[timestamp_ns]
    ]


def _paired_vector_groups(
    samples: dict[str, dict[int, list[float]]],
    left_key: str,
    right_key: str,
) -> dict[str, list[int]]:
    """Describe the vector dimensions contributed by each synchronized source."""

    left = samples.get(left_key, {})
    right = samples.get(right_key, {})
    for timestamp_ns in sorted(set(left) & set(right)):
        left_width = len(left[timestamp_ns])
        right_width = len(right[timestamp_ns])
        if left_width > 0 and right_width > 0:
            return {
                "left": list(range(left_width)),
                "right": list(range(left_width, left_width + right_width)),
            }
    return {}


def _downsample_records(
    records: list[tuple[float, list[float]]],
    target_hz: float | None,
) -> list[tuple[float, list[float]]]:
    if target_hz is None or len(records) <= 1:
        return records
    minimum_step = 1.0 / target_hz
    selected = [records[0]]
    for record in records[1:]:
        if record[0] - selected[-1][0] >= minimum_step:
            selected.append(record)
    if selected[-1] != records[-1]:
        selected.append(records[-1])
    return selected


def _paired_wrist_pose_records(
    samples: dict[str, dict[int, list[float]]],
    origin_ns: int,
) -> list[tuple[float, dict[str, dict[str, list[float]]]]]:
    left = samples.get("/left-arm-state:pose", {})
    right = samples.get("/right-arm-state:pose", {})
    timestamps = sorted(set(left) & set(right))
    output: list[tuple[float, dict[str, dict[str, list[float]]]]] = []
    for timestamp_ns in timestamps:
        left_pose = left[timestamp_ns]
        right_pose = right[timestamp_ns]
        if len(left_pose) < 12 or len(right_pose) < 12:
            continue
        output.append(
            (
                max(0.0, (timestamp_ns - origin_ns) / 1e9),
                {
                    "left": {
                        "position": [left_pose[3], left_pose[7], left_pose[11]],
                        "transform": left_pose,
                    },
                    "right": {
                        "position": [right_pose[3], right_pose[7], right_pose[11]],
                        "transform": right_pose,
                    },
                },
            )
        )
    return output


def _downsample_pose_records(
    records: list[tuple[float, dict[str, dict[str, list[float]]]]],
    target_hz: float | None,
) -> list[tuple[float, dict[str, dict[str, list[float]]]]]:
    if target_hz is None or len(records) <= 1:
        return records
    minimum_step = 1.0 / target_hz
    selected = [records[0]]
    for record in records[1:]:
        if record[0] - selected[-1][0] >= minimum_step:
            selected.append(record)
    if selected[-1] != records[-1]:
        selected.append(records[-1])
    return selected


def _eef_velocity_records(
    records: list[tuple[float, dict[str, dict[str, list[float]]]]],
) -> list[dict[str, Any]]:
    """Derive synchronized left/right Cartesian twists from wrist transforms."""

    output: list[dict[str, Any]] = []
    for previous, current in zip(records, records[1:]):
        previous_time, previous_poses = previous
        current_time, current_poses = current
        dt = current_time - previous_time
        if dt <= 1e-12:
            continue
        linear_velocity: list[float] = []
        angular_velocity: list[float] = []
        complete = True
        for wrist_name in ("left", "right"):
            previous_pose = previous_poses.get(wrist_name, {})
            current_pose = current_poses.get(wrist_name, {})
            previous_position = previous_pose.get("position", [])
            current_position = current_pose.get("position", [])
            previous_transform = previous_pose.get("transform", [])
            current_transform = current_pose.get("transform", [])
            if (
                len(previous_position) < 3
                or len(current_position) < 3
                or len(previous_transform) < 12
                or len(current_transform) < 12
            ):
                complete = False
                break
            linear_velocity.extend(
                [
                    (float(current_position[index]) - float(previous_position[index]))
                    / dt
                    for index in range(3)
                ]
            )
            angular_velocity.extend(
                _angular_velocity_from_transforms(
                    previous_transform,
                    current_transform,
                    dt,
                )
            )
        if not complete:
            continue
        output.append(
            {
                "timestamp_sec": current_time,
                "eef_velocity": [
                    *linear_velocity[:3],
                    *angular_velocity[:3],
                    *linear_velocity[3:6],
                    *angular_velocity[3:6],
                ],
                "eef_linear_velocity": linear_velocity,
                "eef_angular_velocity": angular_velocity,
            }
        )
    return output


def _angular_velocity_from_transforms(
    previous: list[float],
    current: list[float],
    dt: float,
) -> list[float]:
    """Return the rotation-vector finite difference divided by elapsed time."""

    previous_rotation = _rotation_matrix(previous)
    current_rotation = _rotation_matrix(current)
    relative = [
        [
            sum(
                previous_rotation[k][row] * current_rotation[k][column]
                for k in range(3)
            )
            for column in range(3)
        ]
        for row in range(3)
    ]
    quaternion = _quaternion_from_rotation(relative)
    vector_norm = sum(value * value for value in quaternion[1:]) ** 0.5
    if vector_norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    import math

    angle = 2.0 * math.atan2(vector_norm, max(quaternion[0], 0.0))
    scale = angle / (vector_norm * dt)
    return [value * scale for value in quaternion[1:]]


def _rotation_matrix(transform: list[float]) -> list[list[float]]:
    return [
        [float(transform[0]), float(transform[1]), float(transform[2])],
        [float(transform[4]), float(transform[5]), float(transform[6])],
        [float(transform[8]), float(transform[9]), float(transform[10])],
    ]


def _quaternion_from_rotation(rotation: list[list[float]]) -> list[float]:
    """Convert a rotation matrix to a unit quaternion with non-negative scalar."""

    import math

    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
        ]
    else:
        diagonal = [rotation[index][index] for index in range(3)]
        index = max(range(3), key=diagonal.__getitem__)
        first = (index + 1) % 3
        second = (index + 2) % 3
        scale = math.sqrt(
            max(
                1.0
                + rotation[index][index]
                - rotation[first][first]
                - rotation[second][second],
                0.0,
            )
        ) * 2.0
        if scale <= 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        vector = [0.0, 0.0, 0.0]
        vector[index] = 0.25 * scale
        vector[first] = (
            rotation[first][index] + rotation[index][first]
        ) / scale
        vector[second] = (
            rotation[second][index] + rotation[index][second]
        ) / scale
        quaternion = [
            (rotation[second][first] - rotation[first][second]) / scale,
            *vector,
        ]
    norm = sum(value * value for value in quaternion) ** 0.5
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    normalized = [value / norm for value in quaternion]
    return [-value for value in normalized] if normalized[0] < 0.0 else normalized


def _resolve_camera_names(
    *,
    camera_name: str | None,
    camera_names: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    requested = list(camera_names) if camera_names is not None else [camera_name or "top"]
    output: list[str] = []
    for name in requested:
        if name not in CAMERA_TOPICS:
            choices = ", ".join(sorted(CAMERA_TOPICS))
            raise ValueError(
                f"Unsupported ABC-130K camera {name!r}; choose one of: {choices}"
            )
        if name not in output:
            output.append(name)
    if not output:
        raise ValueError("At least one ABC-130K camera must be selected.")
    return tuple(output)


def _resolve_episode_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "episode.mcap"
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find ABC-130K episode MCAP: {resolved}")
    return resolved


def _resolve_annotation_path(episode_path: Path, annotation_path: str | Path | None) -> Path:
    resolved = (
        Path(annotation_path).expanduser().resolve()
        if annotation_path is not None
        else episode_path.with_name("annotation.mcap")
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find ABC-130K annotation MCAP: {resolved}")
    return resolved


def _message_timestamp_ns(decoded: Any, fallback_ns: int) -> int:
    timestamp = getattr(decoded, "timestamp", None)
    if timestamp is None:
        return fallback_ns
    seconds = getattr(timestamp, "seconds", None)
    nanos = getattr(timestamp, "nanos", None)
    if seconds is None or nanos is None:
        return fallback_ns
    return int(seconds) * 1_000_000_000 + int(nanos)


def _sample_timestamp(
    timestamp_sec: float,
    previous_saved_sec: float | None,
    target_hz: float | None,
) -> bool:
    if previous_saved_sec is None or target_hz is None:
        return True
    return timestamp_sec - previous_saved_sec >= (1.0 / target_hz) - 1e-9


def _camera_cache_name(camera_name: str, sample_hz: float | None) -> str:
    suffix = "native" if sample_hz is None else f"{sample_hz:g}hz"
    return f"{camera_name}_{suffix}"


def _materialize_frame(
    video_frame: Any,
    *,
    frame_dir: Path,
    camera_name: str,
    timestamp_sec: float,
    frame_index: int,
) -> CameraFrameRef:
    cv2 = _import_cv2()
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_path = frame_dir / f"{frame_index:06d}_t{timestamp_sec:09.3f}.jpg"
    if not image_path.exists():
        image = video_frame.to_ndarray(format="bgr24")
        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError(f"Could not write decoded ABC-130K frame: {image_path}")
    return CameraFrameRef(
        camera_name=camera_name,
        timestamp_sec=timestamp_sec,
        image_path=image_path,
        frame_index=frame_index,
    )


def _new_h264_decoder() -> Any:
    try:
        import av  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ABC130KBundleAdapter requires PyAV to decode H264.") from exc
    return av.CodecContext.create("h264", "r")


def _av_packet(data: bytes) -> Any:
    import av  # type: ignore[import-not-found]

    return av.Packet(data)


def _mcap_imports() -> tuple[Any, Any]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ABC130KBundleAdapter requires mcap and mcap-protobuf-support."
        ) from exc
    return make_reader, DecoderFactory


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ABC130KBundleAdapter requires opencv-python/cv2 to materialize frames."
        ) from exc
    return cv2


def frame_index_from_path(path: Path) -> int | None:
    """Return the native camera message index encoded in a cached frame name."""

    match = re.match(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None
