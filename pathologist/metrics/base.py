"""Base types for deterministic trajectory metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pathologist.schemas.metrics import MetricResult
from pathologist.schemas.trajectory_bundle import TrajectoryBundle


@dataclass(slots=True)
class MetricComputationContext:
    """Runtime options shared by metric computations."""

    bundle: TrajectoryBundle
    max_events_per_metric: int = 20
    config: dict[str, object] = field(default_factory=dict)


class MetricComputer(Protocol):
    """Protocol implemented by every deterministic metric."""

    name: str

    def compute(self, context: MetricComputationContext) -> MetricResult:
        """Compute one metric for the trajectory bundle."""
