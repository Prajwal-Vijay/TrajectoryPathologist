"""Metric engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from pathologist.metrics.base import MetricComputationContext, MetricComputer
from pathologist.metrics.gripper import GripperChatterMetric
from pathologist.metrics.kinematics import (
    EndEffectorCartesianPathLengthMetric,
    JointPathLengthMetric,
    MotionSmoothnessMetric,
    StallingMetric,
)
from pathologist.metrics.signals import ActionDistributionEntropyMetric, ActionRangeSaturationMetric
from pathologist.metrics.timing import TimestampRegularityMetric
from pathologist.schemas.metrics import MetricProfile
from pathologist.schemas.trajectory_bundle import TrajectoryBundle


def default_metric_computers() -> list[MetricComputer]:
    """Return the default metric set from the project plan."""

    return [
        TimestampRegularityMetric(),
        ActionRangeSaturationMetric(),
        ActionDistributionEntropyMetric(),
        JointPathLengthMetric(),
        EndEffectorCartesianPathLengthMetric(),
        MotionSmoothnessMetric(),
        StallingMetric(),
        GripperChatterMetric(),
    ]


@dataclass(slots=True)
class MetricEngine:
    """Compute a MetricProfile from a TrajectoryBundle."""

    metrics: list[MetricComputer] = field(default_factory=default_metric_computers)
    max_events_per_metric: int = 20

    def compute(self, bundle: TrajectoryBundle) -> MetricProfile:
        """Compute all configured metrics."""

        context = MetricComputationContext(
            bundle=bundle,
            max_events_per_metric=self.max_events_per_metric,
        )
        results = [metric.compute(context) for metric in self.metrics]
        global_events = []
        for result in results:
            global_events.extend(result.events)
        return MetricProfile(
            bundle_id=bundle.bundle_id,
            metrics=results,
            global_events=global_events,
        )
