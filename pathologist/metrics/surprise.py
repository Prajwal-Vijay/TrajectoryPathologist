"""ECOD-style surprise scoring for scalar metric time series."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pathologist.metrics.helpers import summarize
from pathologist.schemas.metrics import MetricProfile, MetricResult, MetricSample


DEFAULT_SURPRISE_SOURCE_METRICS = [
    "action_discontinuity",
    "motion_smoothness",
    "velocity_spike",
    "stalling",
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
        self.score_tail_overrides = {"stalling": "left"}
        if isinstance(tail_overrides, dict):
            self.score_tail_overrides.update(tail_overrides)
        self.score_tail_overrides["stalling"] = "left"

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
        scores = ecod_scores(values, epsilon=self.epsilon, skewness_sign_override=skewness_sign_override)
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
            },
        )

    def score_aggregate_metric(self, profile: MetricProfile) -> MetricResult | None:
        """Build the ECOD paper-style row score by summing feature scores."""

        source_metrics = [
            metric
            for metric_name in self.source_metrics
            for metric in profile.metrics
            if metric.name == metric_name and _scalar_samples(metric)
        ]
        if len(source_metrics) < 2:
            return MetricResult(
                name=self.aggregate_metric_name,
                description="Aggregate ECOD-style surprise score across configured source metrics.",
                unit="ecod_surprise",
                samples=[],
                display_hint="ecod_aggregate_surprise_timeseries",
                metadata={
                    "available": False,
                    "unavailable_reason": "requires at least two source metrics with scalar samples",
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

        per_feature_scores = [
            ecod_scores(
                feature_values,
                epsilon=self.epsilon,
                skewness_sign_override=_skewness_sign_override(metric_name, self.skewness_sign_overrides),
            )
            for metric_name, feature_values in zip(aligned["metric_names"], aligned["feature_values"])
        ]
        aggregate_left_scores = [
            sum(feature_score.left_tail_scores[index] for feature_score in per_feature_scores)
            for index in range(len(aligned["timestamps"]))
        ]
        aggregate_right_scores = [
            sum(feature_score.right_tail_scores[index] for feature_score in per_feature_scores)
            for index in range(len(aligned["timestamps"]))
        ]
        aggregate_skew_scores = [
            sum(feature_score.skew_tail_scores[index] for feature_score in per_feature_scores)
            for index in range(len(aligned["timestamps"]))
        ]
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
                "ECOD-style aggregate trajectory-relative surprise score; each timestamp row "
                "is scored by summing feature-wise ECOD outlier scores across configured metrics."
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
                "score_formula": "O_i=max(sum_j U_l_ij, sum_j U_r_ij, sum_j U_skew_ij)",
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
    return [
        sum(1 for candidate in values if candidate <= value) / count
        for value in values
    ]


def _align_scalar_metric_samples(metrics: list[MetricResult]) -> dict[str, Any]:
    """Align scalar metric samples onto the first metric's timestamp grid."""

    anchor_metric = metrics[0]
    anchor_samples = _scalar_samples(anchor_metric)
    timestamps = [sample.timestamp_sec for sample in anchor_samples]
    feature_values: list[list[float]] = []
    alignment: dict[str, Any] = {
        "method": "nearest_timestamp_to_first_metric_grid",
        "anchor_metric_name": anchor_metric.name,
        "anchor_timestamp_count": len(timestamps),
        "per_metric_max_abs_time_delta_sec": {},
    }

    for metric in metrics:
        samples = _scalar_samples(metric)
        metric_times = [sample.timestamp_sec for sample in samples]
        metric_values = [float(sample.value) for sample in samples]
        aligned_values = []
        max_abs_delta = 0.0
        for timestamp in timestamps:
            nearest_index = min(
                range(len(metric_times)),
                key=lambda index: abs(metric_times[index] - timestamp),
            )
            max_abs_delta = max(max_abs_delta, abs(metric_times[nearest_index] - timestamp))
            aligned_values.append(metric_values[nearest_index])
        feature_values.append(aligned_values)
        alignment["per_metric_max_abs_time_delta_sec"][metric.name] = max_abs_delta

    return {
        "timestamps": timestamps,
        "metric_names": [metric.name for metric in metrics],
        "feature_values": feature_values,
        "alignment": alignment,
    }


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
