"""Metric result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathologist.schemas.common import (
    EvidenceRef,
    Modality,
    NumericSummary,
    SchemaBase,
    Severity,
    Threshold,
    TimeWindow,
    require_non_empty,
)


@dataclass(slots=True)
class MetricSample:
    """One timestamped metric value."""

    timestamp_sec: float
    value: float | int | bool | list[float]
    window: TimeWindow | None = None

    def __post_init__(self) -> None:
        if self.timestamp_sec < 0:
            raise ValueError("MetricSample.timestamp_sec must be non-negative.")


@dataclass(slots=True)
class MetricEvent:
    """A metric-derived noteworthy event or violation."""

    metric_name: str
    time_window: TimeWindow
    value: Any
    severity: Severity | str
    reason: str
    threshold: Threshold | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.metric_name, "MetricEvent.metric_name")
        require_non_empty(self.severity, "MetricEvent.severity")
        require_non_empty(self.reason, "MetricEvent.reason")


@dataclass(slots=True)
class MetricResult(SchemaBase):
    """Output of a deterministic metric computation."""

    name: str = ""
    description: str = ""
    source_modalities: list[Modality] = field(default_factory=list)
    unit: str | None = None
    samples: list[MetricSample] = field(default_factory=list)
    summary: NumericSummary = field(default_factory=NumericSummary)
    thresholds: list[Threshold] = field(default_factory=list)
    events: list[MetricEvent] = field(default_factory=list)
    display_hint: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.name, "MetricResult.name")
        require_non_empty(self.description, "MetricResult.description")


@dataclass(slots=True)
class MetricProfile(SchemaBase):
    """Metric collection for one trajectory."""

    bundle_id: str = ""
    metrics: list[MetricResult] = field(default_factory=list)
    global_events: list[MetricEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.bundle_id, "MetricProfile.bundle_id")
