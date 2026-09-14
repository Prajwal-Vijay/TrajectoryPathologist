"""Smoke tests for the deterministic metric engine."""

from __future__ import annotations

from pathologist.metrics import MetricEngine
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


def test_metric_engine_computes_default_profile() -> None:
    timestamps = [index * 0.05 for index in range(20)]
    actions = [
        {"action": [0.0, 0.0, 0.0, 0.0]},
        {"action": [0.1, 0.0, 0.0, 0.0]},
        {"action": [0.2, 0.0, 0.0, 1.0]},
        {"action": [0.3, 0.0, 0.0, 0.0]},
        {"action": [1.0, 0.0, 0.0, 1.0]},
        {"action": [0.4, 0.0, 0.0, 0.0]},
        {"action": [0.5, 0.0, 0.0, 1.0]},
        {"action": [0.5, 0.0, 0.0, 0.0]},
        {"action": [0.5, 0.0, 0.0, 1.0]},
        {"action": [0.6, 0.0, 0.0, 0.0]},
        {"action": [0.7, 0.0, 0.0, 1.0]},
        {"action": [0.7, 0.0, 0.0, 0.0]},
        {"action": [0.7, 0.0, 0.0, 1.0]},
        {"action": [0.8, 0.0, 0.0, 0.0]},
        {"action": [0.9, 0.0, 0.0, 1.0]},
        {"action": [1.0, 0.0, 0.0, 0.0]},
        {"action": [0.8, 0.0, 0.0, 1.0]},
        {"action": [0.6, 0.0, 0.0, 0.0]},
        {"action": [0.4, 0.0, 0.0, 1.0]},
        {"action": [0.2, 0.0, 0.0, 0.0]},
    ]
    proprioception = [
        {"proprioception": [index * 0.01, 0.0, 0.0]}
        for index in range(20)
    ]
    grippers = [
        {"gripper": [record["action"][-1]]}
        for record in actions
    ]
    velocities = [
        {"qvel": [0.1 + index * 0.01, 0.0, 0.0]}
        for index in range(20)
    ]
    telemetry = [
        {
            "wrist_poses": {
                "handside.left": {"position": [0.5 + index * 0.01, 0.2, 1.0]},
                "handside.right": {"position": [0.5 + index * 0.02, -0.2, 1.0]},
            }
        }
        for index in range(20)
    ]
    bundle = TrajectoryBundle(
        bundle_id="synthetic",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=actions,
                metadata={"action_low": [-1.0, -1.0, -1.0, 0.0], "action_high": [1.0, 1.0, 1.0, 1.0]},
            ),
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=proprioception,
            ),
            ModalityStream(
                name="qvel",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=velocities,
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=grippers,
            ),
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=telemetry,
            ),
        ],
    )

    profile = MetricEngine(max_events_per_metric=5).compute(bundle)

    metric_names = {metric.name for metric in profile.metrics}
    assert "timestamp_regularity" in metric_names
    assert "action_range_saturation" in metric_names
    assert "action_discontinuity" in metric_names
    assert "audio_spectral_flux" in metric_names
    assert "action_distribution_entropy" in metric_names
    assert "joint_path_length" in metric_names
    assert "end_effector_cartesian_path_length" in metric_names
    assert "motion_smoothness" in metric_names
    assert "velocity_spike" in metric_names
    assert "stalling" in metric_names
    assert "eef_linear_velocity_spike" in metric_names
    assert "eef_angular_velocity_spike" in metric_names
    assert "eef_translational_stalling" in metric_names
    assert "eef_rotational_stalling" in metric_names
    assert "eef_translational_jerk" in metric_names
    assert "eef_angular_jerk" in metric_names
    assert "force_norm" in metric_names
    assert "visual_flow" in metric_names
    assert "gripper_aperture_speed" in metric_names
    assert "gripper_transition" in metric_names
    assert "gripper_chatter" in metric_names
    assert profile.bundle_id == "synthetic"


def test_metric_engine_maxes_bimanual_components_after_ecod() -> None:
    timestamps = [index * 0.1 for index in range(8)]
    vector_groups = {"left": [0], "right": [1]}
    bundle = TrajectoryBundle(
        bundle_id="bimanual",
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="action",
                modality=Modality.ACTION,
                timestamps_sec=timestamps,
                records=[
                    {"action": [float(index), float(index**2)]}
                    for index in range(8)
                ],
                metadata={"vector_groups": vector_groups},
            ),
            ModalityStream(
                name="proprioception",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {"proprioception": [float(index**2), float(index**3)]}
                    for index in range(8)
                ],
                metadata={"vector_groups": vector_groups},
            ),
            ModalityStream(
                name="velocity",
                modality=Modality.PROPRIOCEPTION,
                timestamps_sec=timestamps,
                records=[
                    {"qvel": [float(index), float(7 - index)]}
                    for index in range(8)
                ],
                metadata={"vector_groups": vector_groups},
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=timestamps,
                records=[
                    {"gripper": [float(index % 2), float((index + 1) % 2)]}
                    for index in range(8)
                ],
                metadata={"vector_groups": vector_groups},
            ),
        ],
    )
    source_metrics = [
        "action_discontinuity",
        "motion_smoothness",
        "velocity_spike",
        "stalling",
    ]

    profile = MetricEngine(
        config={"surprise": {"source_metrics": source_metrics}}
    ).compute(bundle)

    score_by_name = {metric.name: metric for metric in profile.metrics}
    for source_name in source_metrics:
        assert score_by_name[f"{source_name}_ecod_score"].metadata[
            "component_names"
        ] == ["left", "right"]
    aggregate = score_by_name["ecod_aggregate_score"]
    assert set(aggregate.metadata["grouped_feature_names"]) == set(source_metrics)
    assert aggregate.metadata["grouped_feature_reduction"] == "per_tail_max_after_ecod"
