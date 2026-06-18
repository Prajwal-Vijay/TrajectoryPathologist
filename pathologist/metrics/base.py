"""Base types for deterministic trajectory metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

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


def metric_config(context: MetricComputationContext, metric_name: str) -> dict[str, object]:
    """Return config for one metric from the shared runtime config."""

    metrics = context.config.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    config = metrics.get(metric_name)
    return config if isinstance(config, dict) else {}


def metric_option(
    config: dict[str, object],
    stream_metadata: dict[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    """Read a metric option, preferring config over stream metadata."""

    if key in config:
        return config[key]
    if stream_metadata is not None and key in stream_metadata:
        return stream_metadata[key]
    return default
