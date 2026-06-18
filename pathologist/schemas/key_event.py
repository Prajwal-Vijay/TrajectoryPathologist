"""Sparse key-event schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathologist.schemas.common import (
    EventProposalSource,
    EvidenceRef,
    ProgressAssessment,
    SchemaBase,
    TimeWindow,
    require_non_empty,
)
from pathologist.schemas.metrics import MetricEvent, MetricResult
from pathologist.schemas.trajectory_bundle import CameraFrameRef


@dataclass(slots=True)
class VisualEventLabel(SchemaBase):
    """Visual-only interpretation of a selected key-event window."""

    event_id: str = ""
    visual_observation: str = ""
    phase: str | None = None
    event_type: str | None = None
    objects_involved: list[str] = field(default_factory=list)
    contact_or_grasp_state: str | None = None
    progress_assessment: ProgressAssessment | str | None = None
    visible_pathology_labels: list[str] = field(default_factory=list)
    visual_evidence_ids: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.event_id, "VisualEventLabel.event_id")
        require_non_empty(self.visual_observation, "VisualEventLabel.visual_observation")


@dataclass(slots=True)
class MetricGroundingCard(SchemaBase):
    """Deterministic numeric grounding for one metric-triggered event."""

    evidence_id: str = ""
    metric_name: str = ""
    time_window: TimeWindow | None = None
    observed_value: Any = None
    severity: str | None = None
    threshold_context: dict[str, Any] = field(default_factory=dict)
    relative_level: str = ""
    metric_interpretation: str = ""
    allowed_metric_claims: list[str] = field(default_factory=list)
    forbidden_semantic_claims: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.evidence_id, "MetricGroundingCard.evidence_id")
        require_non_empty(self.metric_name, "MetricGroundingCard.metric_name")
        require_non_empty(self.relative_level, "MetricGroundingCard.relative_level")


@dataclass(slots=True)
class KeyEventProposal(SchemaBase):
    """A candidate analysis window selected before VLM interpretation."""

    event_id: str = ""
    time_window: TimeWindow | None = None
    source: EventProposalSource | str = ""
    trigger_reason: str = ""
    candidate_frames: list[CameraFrameRef] = field(default_factory=list)
    metric_events: list[MetricEvent] = field(default_factory=list)
    metric_packet: list[MetricResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.event_id, "KeyEventProposal.event_id")
        require_non_empty(self.source, "KeyEventProposal.source")
        require_non_empty(self.trigger_reason, "KeyEventProposal.trigger_reason")
        if self.time_window is None:
            raise ValueError("KeyEventProposal.time_window is required.")


@dataclass(slots=True)
class KeyEventLabel(SchemaBase):
    """Interpretation of a selected key event.

    Object identities are intentionally not a required field because many
    datasets expose only robot-side signals. If objects are inferred from
    images, cite them in visual_evidence and explanation instead.
    """

    event_id: str = ""
    phase: str | None = None
    event_type: str | None = None
    objects_involved: list[str] = field(default_factory=list)
    contact_or_grasp_state: str | None = None
    progress_assessment: ProgressAssessment | str | None = None # Did it move to the goal or away from it?
    pathology_labels: list[str] = field(default_factory=list) # What diagnosis of event
    metric_evidence: list[EvidenceRef] = field(default_factory=list)
    visual_evidence: list[EvidenceRef] = field(default_factory=list)
    explanation: str = ""
    uncertainty_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.event_id, "KeyEventLabel.event_id")
        require_non_empty(self.explanation, "KeyEventLabel.explanation")


@dataclass(slots=True)
class KeyEvent(SchemaBase):
    """Final event object used in reports."""

    proposal: KeyEventProposal | None = None
    label: KeyEventLabel | None = None

    def __post_init__(self) -> None:
        if self.proposal is None:
            raise ValueError("KeyEvent.proposal is required.")
