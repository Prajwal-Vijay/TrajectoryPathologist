"""Metric engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathologist.metrics.audio import AudioSpectralFluxMetric
from pathologist.metrics.base import MetricComputationContext, MetricComputer
from pathologist.metrics.gripper import (
    GripperApertureSpeedMetric,
    GripperChatterMetric,
    GripperTransitionMetric,
)
from pathologist.metrics.kinematics import (
    EefAngularJerkMetric,
    EefAngularVelocitySpikeMetric,
    EefLinearVelocitySpikeMetric,
    EefRotationalStallingMetric,
    EefTranslationalJerkMetric,
    EefTranslationalStallingMetric,
    EndEffectorCartesianPathLengthMetric,
    JointPathLengthMetric,
    MotionSmoothnessMetric,
    StallingMetric,
    VelocitySpikeMetric,
)
from pathologist.metrics.signals import (
    ActionDiscontinuityMetric,
    ActionDistributionEntropyMetric,
    ActionRangeSaturationMetric,
    ForceNormMetric,
)
from pathologist.metrics.surprise import ECODSurpriseScorer
from pathologist.metrics.timing import TimestampRegularityMetric
from pathologist.metrics.vision import VisualFlowMetric
from pathologist.schemas.metrics import MetricProfile
from pathologist.schemas.trajectory_bundle import TrajectoryBundle


def default_metric_computers() -> list[MetricComputer]:
    """Return the default metric set from the project plan."""

    return [
        TimestampRegularityMetric(),
        ActionRangeSaturationMetric(),
        ActionDiscontinuityMetric(),
        ActionDistributionEntropyMetric(),
        JointPathLengthMetric(),
        EndEffectorCartesianPathLengthMetric(),
        MotionSmoothnessMetric(),
        VelocitySpikeMetric(),
        StallingMetric(),
        EefLinearVelocitySpikeMetric(),
        EefAngularVelocitySpikeMetric(),
        EefTranslationalStallingMetric(),
        EefRotationalStallingMetric(),
        EefTranslationalJerkMetric(),
        EefAngularJerkMetric(),
        ForceNormMetric(),
        VisualFlowMetric(),
        AudioSpectralFluxMetric(),
        GripperApertureSpeedMetric(),
        GripperTransitionMetric(),
        GripperChatterMetric(),
    ]


@dataclass(slots=True)
class MetricEngine:
    """Compute a MetricProfile from a TrajectoryBundle."""

    metrics: list[MetricComputer] = field(default_factory=default_metric_computers)
    max_events_per_metric: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def compute(self, bundle: TrajectoryBundle) -> MetricProfile:
        """Compute all configured metrics."""

        max_events_per_metric = self.max_events_per_metric
        if max_events_per_metric is None:
            metrics_config = self.config.get("metrics")
            if isinstance(metrics_config, dict):
                max_events_per_metric = int(metrics_config.get("max_events_per_metric", 20))
            else:
                max_events_per_metric = 20

        context = MetricComputationContext(
            bundle=bundle,
            max_events_per_metric=max_events_per_metric,
            config=self.config,
        )
        results = [metric.compute(context) for metric in self.metrics]
        global_events = []
        for result in results:
            global_events.extend(result.events)
        profile = MetricProfile(
            bundle_id=bundle.bundle_id,
            metrics=results,
            global_events=global_events,
        )
        return ECODSurpriseScorer(config=self.config).append_to_profile(profile)
