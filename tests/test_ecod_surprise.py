"""Tests for ECOD-style metric surprise scoring."""

from __future__ import annotations

import math

from pathologist.metrics.surprise import ECODSurpriseScorer, ecod_scores
from pathologist.schemas import MetricProfile, MetricResult, MetricSample


def test_ecod_scores_compute_expected_tail_scores() -> None:
    scores = ecod_scores([1.0, 2.0, 3.0], epsilon=1e-12)

    assert scores.left_tail_scores == [
        -math.log(1 / 3),
        -math.log(2 / 3),
        -math.log(1.0),
    ]
    assert scores.right_tail_scores == [
        -math.log(1.0),
        -math.log(2 / 3),
        -math.log(1 / 3),
    ]
    assert scores.skewness_sign == 0
    assert scores.outlier_scores[0] == scores.skew_tail_scores[0]
    assert scores.outlier_scores[2] == scores.skew_tail_scores[2]


def test_ecod_scores_use_skew_aware_tail_for_positive_skew() -> None:
    scores = ecod_scores([0.0, 0.0, 1.0, 10.0], epsilon=1e-12)

    assert scores.skewness_sign == 1
    assert scores.skew_tail_scores == scores.right_tail_scores
    assert scores.outlier_scores[-1] == scores.right_tail_scores[-1]


def test_ecod_surprise_scorer_appends_score_metrics_with_aligned_timestamps() -> None:
    profile = MetricProfile(
        bundle_id="demo",
        metrics=[
            _metric("action_discontinuity"),
            _metric("motion_smoothness"),
            _metric("velocity_spike"),
            _metric("stalling"),
        ],
    )

    scored_profile = ECODSurpriseScorer(config={"surprise": {"enabled": True}}).append_to_profile(profile)

    score_metrics = [
        metric
        for metric in scored_profile.metrics
        if metric.name.endswith("_ecod_score")
    ]
    assert [metric.name for metric in score_metrics] == [
        "action_discontinuity_ecod_score",
        "motion_smoothness_ecod_score",
        "velocity_spike_ecod_score",
        "stalling_ecod_score",
    ]
    for score_metric in score_metrics:
        assert [sample.timestamp_sec for sample in score_metric.samples] == [0.0, 0.1, 0.2]
        assert not score_metric.events
        assert score_metric.metadata["source_metric_name"] in {
            "action_discontinuity",
            "motion_smoothness",
            "velocity_spike",
            "stalling",
        }
        assert "left_tail_scores" in score_metric.metadata
        assert "right_tail_scores" in score_metric.metadata
        if score_metric.name == "stalling_ecod_score":
            assert score_metric.metadata["score_tail_mode"] == "left"
            assert score_metric.metadata["raw_skewness_sign"] == 1
            assert score_metric.metadata["skewness_sign"] == -1
            assert score_metric.metadata["skewness_sign_override"] == -1
            assert [sample.value for sample in score_metric.samples] == score_metric.metadata["left_tail_scores"]

    aggregate_metric = next(metric for metric in scored_profile.metrics if metric.name == "ecod_aggregate_score")
    assert [sample.timestamp_sec for sample in aggregate_metric.samples] == [0.0, 0.1, 0.2]
    assert aggregate_metric.metadata["score_formula"] == (
        "O_i=max(sum_j U_l_ij, sum_j U_r_ij, sum_j U_skew_ij)"
    )
    assert aggregate_metric.metadata["feature_raw_skewness_sign"]["stalling"] == 1
    assert aggregate_metric.metadata["feature_skewness_sign"]["stalling"] == -1
    assert aggregate_metric.metadata["feature_skewness_sign_overrides"]["stalling"] == -1
    assert aggregate_metric.metadata["feature_score_tail_modes"]["stalling"] == "left"
    stalling_left_scores = aggregate_metric.metadata["feature_left_tail_scores"]["stalling"]
    assert aggregate_metric.metadata["feature_right_tail_scores"]["stalling"] == stalling_left_scores
    assert aggregate_metric.metadata["feature_skew_tail_scores"]["stalling"] == stalling_left_scores
    expected_scores = [
        max(
            aggregate_metric.metadata["aggregate_left_tail_scores"][index],
            aggregate_metric.metadata["aggregate_right_tail_scores"][index],
            aggregate_metric.metadata["aggregate_skew_tail_scores"][index],
        )
        for index in range(3)
    ]
    assert [sample.value for sample in aggregate_metric.samples] == expected_scores


def test_ecod_surprise_maxes_group_scores_instead_of_combining_raw_vectors() -> None:
    timestamps = [0.0, 0.1, 0.2, 0.3]
    grouped_metric = MetricResult(
        name="action_discontinuity",
        description="Grouped source metric",
        samples=[
            MetricSample(timestamp_sec=timestamp, value=value)
            for timestamp, value in zip(timestamps, [0.0, 10.0, 10.0, 0.0])
        ],
        metadata={
            "ecod_component_values": {
                "left": [0.0, 10.0, 0.0, 0.0],
                "right": [0.0, 0.0, 10.0, 0.0],
            }
        },
    )
    visual_metric = MetricResult(
        name="visual_flow",
        description="Scalar source metric",
        samples=[
            MetricSample(timestamp_sec=timestamp, value=value)
            for timestamp, value in zip(timestamps, [1.0, 2.0, 3.0, 4.0])
        ],
    )
    scorer = ECODSurpriseScorer(
        config={
            "surprise": {
                "enabled": True,
                "source_metrics": ["action_discontinuity", "visual_flow"],
            }
        }
    )

    scored_profile = scorer.append_to_profile(
        MetricProfile(bundle_id="grouped", metrics=[grouped_metric, visual_metric])
    )

    left_scores = ecod_scores([0.0, 10.0, 0.0, 0.0])
    right_scores = ecod_scores([0.0, 0.0, 10.0, 0.0])
    expected = [
        max(left_scores.outlier_scores[index], right_scores.outlier_scores[index])
        for index in range(len(timestamps))
    ]
    action_score = next(
        metric
        for metric in scored_profile.metrics
        if metric.name == "action_discontinuity_ecod_score"
    )
    assert [sample.value for sample in action_score.samples] == expected
    assert action_score.metadata["component_names"] == ["left", "right"]
    assert action_score.metadata["component_reduction"] == "per_tail_max_after_ecod"

    aggregate = next(
        metric
        for metric in scored_profile.metrics
        if metric.name == "ecod_aggregate_score"
    )
    assert aggregate.metadata["grouped_feature_names"] == {
        "action_discontinuity": ["left", "right"]
    }
    assert aggregate.metadata["grouped_feature_reduction"] == "per_tail_max_after_ecod"
    expected_left_tail = [
        max(left_scores.left_tail_scores[index], right_scores.left_tail_scores[index])
        for index in range(len(timestamps))
    ]
    assert (
        aggregate.metadata["feature_left_tail_scores"]["action_discontinuity"]
        == expected_left_tail
    )


def test_eef_metrics_only_override_stalling_tail_direction() -> None:
    metric_names = [
        "eef_linear_velocity_spike",
        "eef_angular_velocity_spike",
        "eef_translational_stalling",
        "eef_rotational_stalling",
        "eef_translational_jerk",
        "eef_angular_jerk",
    ]
    scorer = ECODSurpriseScorer(
        config={"surprise": {"source_metrics": metric_names}}
    )

    scored = scorer.append_to_profile(
        MetricProfile(
            bundle_id="eef-tail-modes",
            metrics=[_metric(name) for name in metric_names],
        )
    )
    score_by_name = {metric.name: metric for metric in scored.metrics}

    for metric_name in (
        "eef_linear_velocity_spike",
        "eef_angular_velocity_spike",
        "eef_translational_jerk",
        "eef_angular_jerk",
    ):
        assert score_by_name[f"{metric_name}_ecod_score"].metadata[
            "score_tail_mode"
        ] == "auto"
    for metric_name in (
        "eef_translational_stalling",
        "eef_rotational_stalling",
    ):
        assert score_by_name[f"{metric_name}_ecod_score"].metadata[
            "score_tail_mode"
        ] == "left"


def _metric(name: str) -> MetricResult:
    return MetricResult(
        name=name,
        description=f"{name} source metric",
        unit="source_unit",
        samples=[
            MetricSample(timestamp_sec=0.0, value=1.0),
            MetricSample(timestamp_sec=0.1, value=2.0),
            MetricSample(timestamp_sec=0.2, value=4.0),
        ],
    )
