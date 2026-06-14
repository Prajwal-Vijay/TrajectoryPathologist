"""Final trajectory report schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from pathologist.schemas.common import EvidenceRef, OutcomeLabel, SchemaBase, require_non_empty
from pathologist.schemas.key_event import KeyEvent
from pathologist.schemas.metrics import MetricProfile
from pathologist.schemas.task_contract import TaskContract


@dataclass(slots=True)
class OutcomeAssessment(SchemaBase):
    """Episode-level outcome classification."""

    outcome: OutcomeLabel | str = ""
    explanation: str = ""
    issue_labels: list[str] = field(default_factory=list)
    confidence: float | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.outcome, "OutcomeAssessment.outcome")
        require_non_empty(self.explanation, "OutcomeAssessment.explanation")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("OutcomeAssessment.confidence must be in [0, 1].")


@dataclass(slots=True)
class TrajectoryReport(SchemaBase):
    """Grounded pathology report for one trajectory."""

    report_id: str = ""
    bundle_id: str = ""
    task_contract: TaskContract | None = None
    outcome_assessment: OutcomeAssessment | None = None
    key_events: list[KeyEvent] = field(default_factory=list)
    metric_profile: MetricProfile | None = None
    overall_explanation: str = ""
    uncertainty_notes: list[str] = field(default_factory=list)
    review_priority_score: float | None = None # Not trajectory quality score exactly, more like indicator how important should it be reviewed.
    verifier_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.report_id, "TrajectoryReport.report_id")
        require_non_empty(self.bundle_id, "TrajectoryReport.bundle_id")
        require_non_empty(self.overall_explanation, "TrajectoryReport.overall_explanation")
        if self.review_priority_score is not None and not 0 <= self.review_priority_score <= 1:
            raise ValueError("TrajectoryReport.review_priority_score must be in [0, 1].")
