"""ECOD-style surprise scoring for scalar metric time series."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np

from pathologist.metrics.helpers import summarize
from pathologist.schemas.metrics import MetricProfile, MetricResult, MetricSample


DEFAULT_SURPRISE_SOURCE_METRICS = [
    "action_discontinuity",
    "motion_smoothness",
    "velocity_spike",
    "stalling",
    "eef_linear_velocity_spike",
    "eef_angular_velocity_spike",
    "eef_translational_stalling",
    "eef_rotational_stalling",
    "eef_translational_jerk",
    "eef_angular_jerk",
    "gripper_aperture_speed",
    "visual_flow",
    "audio_spectral_flux",
]

@dataclass(slots=True)
class ECODScoreResult:
    """Per-sample ECOD-style score components for one scalar series."""

    left_tail_scores: list[float]
    right_tail_scores: list[float]
    skew_tail_scores: list[float]
    outlier_scores: list[float]
    skewness: float
    skewness_sign: int


class ECODSurpriseScorer:
    """Append ECOD-style surprise metrics to a metric profile."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        surprise_config = self.config.get("surprise")
        self.surprise_config = surprise_config if isinstance(surprise_config, dict) else {}
        self.enabled = bool(self.surprise_config.get("enabled", True))
        self.epsilon = float(self.surprise_config.get("epsilon", 1e-12))
        self.aggregation_reduction = str(
            self.surprise_config.get("aggregation_reduction", "sum")
        ).strip().lower()
        if self.aggregation_reduction not in {"sum", "max"}:
            raise ValueError(
                "surprise.aggregation_reduction must be either 'sum' or 'max'"
            )
        raw_groups = self.surprise_config.get("aggregation_groups")
        self.aggregation_groups = (
            _parse_aggregation_groups(raw_groups)
            if isinstance(raw_groups, dict)
            else None
        )
        self.within_group_reduction = str(
            self.surprise_config.get("within_group_reduction", "mean")
        ).strip().lower()
        self.across_group_reduction = str(
            self.surprise_config.get("across_group_reduction", "sum")
        ).strip().lower()
        for name, reduction_name in (
            ("within_group_reduction", self.within_group_reduction),
            ("across_group_reduction", self.across_group_reduction),
        ):
            if reduction_name not in {"mean", "sum", "max"}:
                raise ValueError(
                    f"surprise.{name} must be one of: mean, sum, max"
                )
        source_metrics = self.surprise_config.get("source_metrics", DEFAULT_SURPRISE_SOURCE_METRICS)
        self.source_metrics = [
            str(metric_name)
            for metric_name in source_metrics
            if isinstance(metric_name, str)
        ] or list(DEFAULT_SURPRISE_SOURCE_METRICS)
        self.aggregate_metric_name = str(
            self.surprise_config.get("aggregate_metric_name", "ecod_aggregate_score")
        )
        tail_overrides = self.surprise_config.get("score_tail_overrides", {})
        self.score_tail_overrides = {
            "stalling": "left",
            "eef_translational_stalling": "left",
            "eef_rotational_stalling": "left",
            "gripper_aperture_speed": "right",
            "audio_spectral_flux": "right",
        }
        if isinstance(tail_overrides, dict):
            self.score_tail_overrides.update(tail_overrides)
        self.score_tail_overrides["stalling"] = "left"
        self.score_tail_overrides["eef_translational_stalling"] = "left"
        self.score_tail_overrides["eef_rotational_stalling"] = "left"
        self.score_tail_overrides["gripper_aperture_speed"] = "right"
        self.score_tail_overrides["audio_spectral_flux"] = "right"

        skewness_overrides = self.surprise_config.get("skewness_sign_overrides", {})
        self.skewness_sign_overrides = {"stalling": -1}
        if isinstance(skewness_overrides, dict):
            self.skewness_sign_overrides.update(skewness_overrides)
        self.skewness_sign_overrides["stalling"] = -1

    def append_to_profile(self, profile: MetricProfile) -> MetricProfile:
        """Return a profile with ECOD surprise metrics appended."""

        if not self.enabled:
            return profile
        existing_names = {metric.name for metric in profile.metrics}
        surprise_metrics: list[MetricResult] = []
        for source_metric_name in self.source_metrics:
            source_metric = next(
                (metric for metric in profile.metrics if metric.name == source_metric_name),
                None,
            )
            if source_metric is None:
                continue
            surprise_name = f"{source_metric_name}_ecod_score"
            if surprise_name in existing_names:
                continue
            surprise_metric = self.score_metric(source_metric, surprise_name)
            if surprise_metric is not None:
                surprise_metrics.append(surprise_metric)
                existing_names.add(surprise_name)
        aggregate_metric = self.score_aggregate_metric(profile)
        if aggregate_metric is not None and aggregate_metric.name not in existing_names:
            surprise_metrics.append(aggregate_metric)
        if not surprise_metrics:
            return profile
        return MetricProfile(
            bundle_id=profile.bundle_id,
            metrics=[*profile.metrics, *surprise_metrics],
            global_events=profile.global_events,
            metadata=profile.metadata,
        )

    def score_metric(self, source_metric: MetricResult, surprise_name: str | None = None) -> MetricResult | None:
        """Build one ECOD surprise MetricResult from scalar samples."""

        scalar_samples = [
            sample
            for sample in source_metric.samples
            if isinstance(sample.value, (int, float, bool))
        ]
        if len(scalar_samples) < 2:
            return MetricResult(
                name=surprise_name or f"{source_metric.name}_ecod_score",
                description=f"ECOD-style trajectory-relative surprise score for {source_metric.name}.",
                unit="ecod_surprise",
                samples=[],
                display_hint="ecod_surprise_timeseries",
                metadata={
                    "available": False,
                    "source_metric_name": source_metric.name,
                    "unavailable_reason": "requires at least two scalar metric samples",
                },
            )

        values = [float(sample.value) for sample in scalar_samples]
        skewness_sign_override = _skewness_sign_override(source_metric.name, self.skewness_sign_overrides)
        component_values = _metric_component_values(source_metric, len(values))
        component_scores = {
            name: ecod_scores(
                values,
                epsilon=self.epsilon,
                skewness_sign_override=skewness_sign_override,
            )
            for name, values in component_values.items()
        }
        scores = (
            _max_component_scores(component_scores)
            if len(component_scores) > 1
            else ecod_scores(
                values,
                epsilon=self.epsilon,
                skewness_sign_override=skewness_sign_override,
            )
        )
        score_tail_mode = _score_tail_mode(source_metric.name, self.score_tail_overrides)
        sample_scores = _scores_for_tail_mode(scores, score_tail_mode)
        return MetricResult(
            name=surprise_name or f"{source_metric.name}_ecod_score",
            description=(
                f"ECOD-style trajectory-relative rarity score for {source_metric.name}; "
                "higher values indicate rarer samples within this trajectory."
            ),
            source_modalities=source_metric.source_modalities,
            unit="ecod_surprise",
            samples=[
                MetricSample(timestamp_sec=sample.timestamp_sec, value=score)
                for sample, score in zip(scalar_samples, sample_scores)
            ],
            summary=summarize(sample_scores),
            thresholds=[],
            events=[],
            display_hint="ecod_surprise_timeseries",
            metadata={
                "available": True,
                "source_metric_name": source_metric.name,
                "source_unit": source_metric.unit,
                "source_display_hint": source_metric.display_hint,
                "score_formula": "O_i=max(-log(F(x_i)), -log(F(-x_i)), U_skew_i)",
                "score_tail_mode": score_tail_mode,
                "epsilon": self.epsilon,
                "skewness": scores.skewness,
                "raw_skewness_sign": _sign(scores.skewness),
                "skewness_sign": scores.skewness_sign,
                "skewness_sign_override": skewness_sign_override,
                "left_tail_scores": scores.left_tail_scores,
                "right_tail_scores": scores.right_tail_scores,
                "skew_tail_scores": scores.skew_tail_scores,
                "source_values": values,
                "component_names": list(component_scores),
                "component_reduction": (
                    "per_tail_max_after_ecod" if len(component_scores) > 1 else None
                ),
                "component_source_values": component_values,
                "component_left_tail_scores": {
                    name: score.left_tail_scores for name, score in component_scores.items()
                },
                "component_right_tail_scores": {
                    name: score.right_tail_scores for name, score in component_scores.items()
                },
                "component_skew_tail_scores": {
                    name: score.skew_tail_scores for name, score in component_scores.items()
                },
            },
        )

    def score_aggregate_metric(self, profile: MetricProfile) -> MetricResult | None:
        """Build an ECOD row score by reducing aligned feature scores."""

        source_metrics = [
            metric
            for metric_name in self.source_metrics
            for metric in profile.metrics
            if metric.name == metric_name and _scalar_samples(metric)
        ]
        if len(source_metrics) < 1:
            return MetricResult(
                name=self.aggregate_metric_name,
                description="Aggregate ECOD-style surprise score across configured source metrics.",
                unit="ecod_surprise",
                samples=[],
                display_hint="ecod_aggregate_surprise_timeseries",
                metadata={
                    "available": False,
                    "unavailable_reason": "requires at least one source metric with scalar samples",
                    "source_metric_names": [metric.name for metric in source_metrics],
                },
            )

        aligned = _align_scalar_metric_samples(source_metrics)
        if len(aligned["timestamps"]) < 2:
            return MetricResult(
                name=self.aggregate_metric_name,
                description="Aggregate ECOD-style surprise score across configured source metrics.",
                unit="ecod_surprise",
                samples=[],
                display_hint="ecod_aggregate_surprise_timeseries",
                metadata={
                    "available": False,
                    "unavailable_reason": "requires at least two aligned timestamps",
                    "source_metric_names": [metric.name for metric in source_metrics],
                },
            )

        per_feature_scores: list[ECODScoreResult] = []
        feature_score_tail_modes: dict[str, str] = {}
        grouped_feature_names: dict[str, list[str]] = {}
        for index, (metric_name, feature_values) in enumerate(
            zip(aligned["metric_names"], aligned["feature_values"])
        ):
            skewness_sign_override = _skewness_sign_override(
                metric_name,
                self.skewness_sign_overrides,
            )
            source_metric = source_metrics[index]
            component_values = _metric_component_values(
                source_metric,
                len(_scalar_samples(source_metric)),
            )
            nearest_indices = aligned["feature_sample_indices"][index]
            aligned_component_values = {
                name: [values[sample_index] for sample_index in nearest_indices]
                for name, values in component_values.items()
            }
            component_scores = {
                name: ecod_scores(
                    values,
                    epsilon=self.epsilon,
                    skewness_sign_override=skewness_sign_override,
                )
                for name, values in aligned_component_values.items()
            }
            if len(component_scores) > 1:
                grouped_feature_names[metric_name] = list(component_scores)
                scores = _max_component_scores(component_scores)
            else:
                scores = ecod_scores(
                    feature_values,
                    epsilon=self.epsilon,
                    skewness_sign_override=skewness_sign_override,
                )
            score_tail_mode = _score_tail_mode(metric_name, self.score_tail_overrides)
            feature_score_tail_modes[metric_name] = score_tail_mode
            per_feature_scores.append(
                _scores_for_aggregate_tail_mode(scores, score_tail_mode)
            )
        group_metadata: dict[str, Any] = {}
        if self.aggregation_groups is None:
            reduction = sum if self.aggregation_reduction == "sum" else max
            aggregate_left_scores = [
                reduction(
                    feature_score.left_tail_scores[index]
                    for feature_score in per_feature_scores
                )
                for index in range(len(aligned["timestamps"]))
            ]
            aggregate_right_scores = [
                reduction(
                    feature_score.right_tail_scores[index]
                    for feature_score in per_feature_scores
                )
                for index in range(len(aligned["timestamps"]))
            ]
            aggregate_skew_scores = [
                reduction(
                    feature_score.skew_tail_scores[index]
                    for feature_score in per_feature_scores
                )
                for index in range(len(aligned["timestamps"]))
            ]
            score_formula = (
                "O_i=max(max_j U_l_ij, max_j U_r_ij, max_j U_skew_ij)"
                if self.aggregation_reduction == "max"
                else "O_i=max(sum_j U_l_ij, sum_j U_r_ij, sum_j U_skew_ij)"
            )
        else:
            scores_by_metric = dict(
                zip(aligned["metric_names"], per_feature_scores)
            )
            active_groups = _active_aggregation_groups(
                aligned["metric_names"],
                self.aggregation_groups,
            )
            group_left_scores = _group_branch_scores(
                scores_by_metric,
                active_groups,
                branch="left_tail_scores",
                reduction=self.within_group_reduction,
            )
            group_right_scores = _group_branch_scores(
                scores_by_metric,
                active_groups,
                branch="right_tail_scores",
                reduction=self.within_group_reduction,
            )
            group_skew_scores = _group_branch_scores(
                scores_by_metric,
                active_groups,
                branch="skew_tail_scores",
                reduction=self.within_group_reduction,
            )
            aggregate_left_scores = _reduce_group_branches(
                group_left_scores,
                self.across_group_reduction,
            )
            aggregate_right_scores = _reduce_group_branches(
                group_right_scores,
                self.across_group_reduction,
            )
            aggregate_skew_scores = _reduce_group_branches(
                group_skew_scores,
                self.across_group_reduction,
            )
            score_formula = (
                "O_i=max(reduce_across_g(reduce_within_g(U_l_ij)), "
                "reduce_across_g(reduce_within_g(U_r_ij)), "
                "reduce_across_g(reduce_within_g(U_skew_ij)))"
            )
            group_metadata = {
                "aggregation_groups": active_groups,
                "configured_aggregation_groups": self.aggregation_groups,
                "within_group_reduction": self.within_group_reduction,
                "across_group_reduction": self.across_group_reduction,
                "group_left_tail_scores": group_left_scores,
                "group_right_tail_scores": group_right_scores,
                "group_skew_tail_scores": group_skew_scores,
            }
        aggregate_scores = [
            max(left_score, right_score, skew_score)
            for left_score, right_score, skew_score in zip(
                aggregate_left_scores,
                aggregate_right_scores,
                aggregate_skew_scores,
            )
        ]
        return MetricResult(
            name=self.aggregate_metric_name,
            description=(
                "ECOD-style aggregate trajectory-relative surprise score; each timestamp "
                f"row uses the {self.aggregation_reduction} of feature-wise ECOD scores."
            ),
            unit="ecod_surprise",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=score)
                for timestamp, score in zip(aligned["timestamps"], aggregate_scores)
            ],
            summary=summarize(aggregate_scores),
            thresholds=[],
            events=[],
            display_hint="ecod_aggregate_surprise_timeseries",
            metadata={
                "available": True,
                "source_metric_names": aligned["metric_names"],
                "score_formula": score_formula,
                "aggregation_reduction": self.aggregation_reduction,
                **group_metadata,
                "alignment": aligned["alignment"],
                "epsilon": self.epsilon,
                "aggregate_left_tail_scores": aggregate_left_scores,
                "aggregate_right_tail_scores": aggregate_right_scores,
                "aggregate_skew_tail_scores": aggregate_skew_scores,
                "feature_left_tail_scores": {
                    metric_name: score.left_tail_scores
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_right_tail_scores": {
                    metric_name: score.right_tail_scores
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_skew_tail_scores": {
                    metric_name: score.skew_tail_scores
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_skewness": {
                    metric_name: score.skewness
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_raw_skewness_sign": {
                    metric_name: _sign(score.skewness)
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_skewness_sign": {
                    metric_name: score.skewness_sign
                    for metric_name, score in zip(aligned["metric_names"], per_feature_scores)
                },
                "feature_skewness_sign_overrides": {
                    metric_name: _skewness_sign_override(metric_name, self.skewness_sign_overrides)
                    for metric_name in aligned["metric_names"]
                },
                "feature_score_tail_modes": feature_score_tail_modes,
                "grouped_feature_names": grouped_feature_names,
                "grouped_feature_reduction": (
                    "per_tail_max_after_ecod" if grouped_feature_names else None
                ),
            },
        )


def ecod_scores(
    values: list[float],
    epsilon: float = 1e-12,
    skewness_sign_override: int | None = None,
) -> ECODScoreResult:
    """Compute ECOD-style score components for one scalar series."""

    if not values:
        return ECODScoreResult([], [], [], [], 0.0, 0)
    clipped_epsilon = max(float(epsilon), 1e-300)
    left_probs = _empirical_cdf(values)
    right_probs = _empirical_cdf([-value for value in values])
    left_scores = [-math.log(max(probability, clipped_epsilon)) for probability in left_probs]
    right_scores = [-math.log(max(probability, clipped_epsilon)) for probability in right_probs]
    skewness = _skewness(values)
    skewness_sign = skewness_sign_override if skewness_sign_override in {-1, 0, 1} else _sign(skewness)
    skew_scores = _skew_tail_scores(left_scores, right_scores, skewness_sign)
    outlier_scores = [
        max(left_score, right_score, skew_score)
        for left_score, right_score, skew_score in zip(left_scores, right_scores, skew_scores)
    ]
    return ECODScoreResult(
        left_tail_scores=left_scores,
        right_tail_scores=right_scores,
        skew_tail_scores=skew_scores,
        outlier_scores=outlier_scores,
        skewness=skewness,
        skewness_sign=skewness_sign,
    )


def _score_tail_mode(metric_name: str, tail_overrides: dict[Any, Any]) -> str:
    mode = str(tail_overrides.get(metric_name, "auto")).lower()
    return mode if mode in {"auto", "left", "right", "skew"} else "auto"


def _scores_for_tail_mode(scores: ECODScoreResult, tail_mode: str) -> list[float]:
    if tail_mode == "left":
        return scores.left_tail_scores
    if tail_mode == "right":
        return scores.right_tail_scores
    if tail_mode == "skew":
        return scores.skew_tail_scores
    return scores.outlier_scores


def _scores_for_aggregate_tail_mode(
    scores: ECODScoreResult,
    tail_mode: str,
) -> ECODScoreResult:
    """Apply a directional metric override to every aggregate ECOD branch."""

    if tail_mode == "auto":
        return scores
    selected_scores = _scores_for_tail_mode(scores, tail_mode)
    return ECODScoreResult(
        left_tail_scores=list(selected_scores),
        right_tail_scores=list(selected_scores),
        skew_tail_scores=list(selected_scores),
        outlier_scores=list(selected_scores),
        skewness=scores.skewness,
        skewness_sign=scores.skewness_sign,
    )


def _metric_component_values(
    metric: MetricResult,
    expected_count: int,
) -> dict[str, list[float]]:
    """Return complete, numeric per-component source series for ECOD scoring."""

    raw_components = metric.metadata.get("ecod_component_values")
    if not isinstance(raw_components, dict):
        return {}

    components: dict[str, list[float]] = {}
    for raw_name, raw_values in raw_components.items():
        if not isinstance(raw_name, str) or not isinstance(raw_values, (list, tuple)):
            continue
        if len(raw_values) != expected_count:
            continue
        if not all(isinstance(value, (int, float, bool)) for value in raw_values):
            continue
        components[raw_name] = [float(value) for value in raw_values]
    return components if len(components) > 1 else {}


def _max_component_scores(
    component_scores: dict[str, ECODScoreResult],
) -> ECODScoreResult:
    """Collapse independent component ECOD scores with a pointwise maximum."""

    scores = list(component_scores.values())
    if not scores:
        return ECODScoreResult([], [], [], [], 0.0, 0)
    count = min(len(score.outlier_scores) for score in scores)
    left_scores = [
        max(score.left_tail_scores[index] for score in scores)
        for index in range(count)
    ]
    right_scores = [
        max(score.right_tail_scores[index] for score in scores)
        for index in range(count)
    ]
    skew_scores = [
        max(score.skew_tail_scores[index] for score in scores)
        for index in range(count)
    ]
    return ECODScoreResult(
        left_tail_scores=left_scores,
        right_tail_scores=right_scores,
        skew_tail_scores=skew_scores,
        outlier_scores=[
            max(left_score, right_score, skew_score)
            for left_score, right_score, skew_score in zip(
                left_scores,
                right_scores,
                skew_scores,
            )
        ],
        skewness=0.0,
        skewness_sign=0,
    )


def _skewness_sign_override(metric_name: str, overrides: dict[Any, Any]) -> int | None:
    value = overrides.get(metric_name)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {-1, 0, 1}:
        return value
    if isinstance(value, float) and value in {-1.0, 0.0, 1.0}:
        return int(value)
    return None


def _empirical_cdf(values: list[float]) -> list[float]:
    count = len(values)
    sorted_values = sorted(values)
    return [
        bisect_right(sorted_values, value) / count
        for value in values
    ]


def _align_scalar_metric_samples(metrics: list[MetricResult]) -> dict[str, Any]:
    """Align scalar metric samples onto the densest available timestamp grid."""

    # Preserve source-metric order as the deterministic tie-breaker. Choosing the
    # metric with the most samples prevents a coarse source (for example, a 1 Hz
    # cached VLM score) from reducing an otherwise high-rate aggregate timeline.
    anchor_metric = max(metrics, key=lambda metric: len(_scalar_samples(metric)))
    anchor_samples = _scalar_samples(anchor_metric)
    timestamps = [sample.timestamp_sec for sample in anchor_samples]
    feature_values: list[list[float]] = []
    feature_sample_indices: list[list[int]] = []
    alignment: dict[str, Any] = {
        "method": "nearest_timestamp_to_densest_metric_grid",
        "anchor_metric_name": anchor_metric.name,
        "anchor_timestamp_count": len(timestamps),
        "per_metric_max_abs_time_delta_sec": {},
    }

    for metric in metrics:
        samples = _scalar_samples(metric)
        metric_times = [sample.timestamp_sec for sample in samples]
        metric_values = [float(sample.value) for sample in samples]
        nearest_indices = _nearest_sample_indices(timestamps, metric_times)
        aligned_values = [metric_values[index] for index in nearest_indices]
        max_abs_delta = max(
            (
                abs(metric_times[index] - timestamp)
                for timestamp, index in zip(timestamps, nearest_indices)
            ),
            default=0.0,
        )
        feature_values.append(aligned_values)
        feature_sample_indices.append(nearest_indices)
        alignment["per_metric_max_abs_time_delta_sec"][metric.name] = max_abs_delta

    return {
        "timestamps": timestamps,
        "metric_names": [metric.name for metric in metrics],
        "feature_values": feature_values,
        "feature_sample_indices": feature_sample_indices,
        "alignment": alignment,
    }


def _nearest_sample_indices(
    anchor_times: list[float],
    metric_times: list[float],
) -> list[int]:
    """Vectorized nearest indices with the original lower-index tie rule."""

    if not metric_times:
        return []
    times = np.asarray(metric_times, dtype=np.float64)
    anchors = np.asarray(anchor_times, dtype=np.float64)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    insertion = np.searchsorted(sorted_times, anchors, side="left")
    left_positions = np.clip(insertion - 1, 0, len(sorted_times) - 1)
    right_positions = np.clip(insertion, 0, len(sorted_times) - 1)
    left_indices = order[left_positions]
    right_indices = order[right_positions]
    left_delta = np.abs(times[left_indices] - anchors)
    right_delta = np.abs(times[right_indices] - anchors)
    choose_right = (right_delta < left_delta) | (
        (right_delta == left_delta) & (right_indices < left_indices)
    )
    return np.where(choose_right, right_indices, left_indices).astype(int).tolist()


def _parse_aggregation_groups(value: dict[Any, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for raw_name, raw_metrics in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_metrics, (list, tuple)):
            continue
        metrics = [
            str(metric_name)
            for metric_name in raw_metrics
            if isinstance(metric_name, str)
        ]
        if metrics:
            groups[raw_name] = metrics
    return groups


def _active_aggregation_groups(
    metric_names: list[str],
    configured_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Resolve configured groups and add singleton groups for unlisted metrics."""

    active: dict[str, list[str]] = {}
    claimed: set[str] = set()
    available = set(metric_names)
    for group_name, configured_names in configured_groups.items():
        members = [
            name
            for name in configured_names
            if name in available and name not in claimed
        ]
        if members:
            active[group_name] = members
            claimed.update(members)
    for metric_name in metric_names:
        if metric_name not in claimed:
            group_name = metric_name
            suffix = 2
            while group_name in active:
                group_name = f"{metric_name}_{suffix}"
                suffix += 1
            active[group_name] = [metric_name]
    return active


def _group_branch_scores(
    scores_by_metric: dict[str, ECODScoreResult],
    groups: dict[str, list[str]],
    *,
    branch: str,
    reduction: str,
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for group_name, metric_names in groups.items():
        branches = [
            getattr(scores_by_metric[metric_name], branch)
            for metric_name in metric_names
        ]
        output[group_name] = [
            _reduce_values(
                [values[index] for values in branches],
                reduction,
            )
            for index in range(len(branches[0]))
        ]
    return output


def _reduce_group_branches(
    group_scores: dict[str, list[float]],
    reduction: str,
) -> list[float]:
    branches = list(group_scores.values())
    return [
        _reduce_values([values[index] for values in branches], reduction)
        for index in range(len(branches[0]))
    ]


def _reduce_values(values: list[float], reduction: str) -> float:
    if reduction == "mean":
        return sum(values) / len(values)
    if reduction == "max":
        return max(values)
    return sum(values)


def _scalar_samples(metric: MetricResult) -> list[MetricSample]:
    return [
        sample
        for sample in metric.samples
        if isinstance(sample.value, (int, float, bool))
    ]


def _skew_tail_scores(
    left_scores: list[float],
    right_scores: list[float],
    skewness_sign: int,
) -> list[float]:
    # Mirrors PyOD ECOD: U_skew = U_l * -sign(skew_sign - 1) + U_r * sign(skew_sign + 1).
    left_factor = -_sign(skewness_sign - 1)
    right_factor = _sign(skewness_sign + 1)
    return [
        left_factor * left_score + right_factor * right_score
        for left_score, right_score in zip(left_scores, right_scores)
    ]


def _skewness(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    variance = sum(value * value for value in centered) / len(values)
    if variance <= 0:
        return 0.0
    third_moment = sum(value**3 for value in centered) / len(values)
    return third_moment / (variance ** 1.5)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
