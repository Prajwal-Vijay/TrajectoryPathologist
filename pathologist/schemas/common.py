"""Shared schema primitives for trajectory analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a schema object is missing required information."""


class StrEnum(str, Enum):
    """String enum that serializes cleanly to JSON-like dictionaries."""

    def __str__(self) -> str:
        return self.value


class Modality(StrEnum):
    """Supported input modalities."""

    TELEMETRY = "telemetry"
    ACTION = "action"
    PROPRIOCEPTION = "proprioception"
    GRIPPER = "gripper"
    RGB = "rgb"
    DEPTH = "depth"
    FORCE = "force"
    AUDIO = "audio"
    LANGUAGE = "language"
    METADATA = "metadata"


class EvidenceSource(StrEnum):
    """Where an evidence item came from."""

    METRIC = "metric"
    VISUAL = "visual"
    TELEMETRY = "telemetry"
    TASK_CONTRACT = "task_contract"
    MODEL = "model"


class Severity(StrEnum):
    """Common severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThresholdBasis(StrEnum):
    """Where a threshold or severity rule came from."""

    HARD_LIMIT = "hard_limit"
    TASK_CONTRACT = "task_contract"
    INTRA_EPISODE_PERCENTILE = "intra_episode_percentile"
    INTRA_EPISODE_ROBUST_STAT = "intra_episode_robust_stat"
    PHASE_RELATIVE_PERCENTILE = "phase_relative_percentile"
    DATASET_RELATIVE_PERCENTILE = "dataset_relative_percentile"
    LOCAL_WINDOW_CHANGE = "local_window_change"
    HEURISTIC_DEFAULT = "heuristic_default"
    MANUAL_RULE = "manual_rule"
    AVAILABILITY_RULE = "availability_rule"


class ThresholdOperator(StrEnum):
    """Supported threshold condition operators."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    INSIDE_RANGE = "inside_range"
    OUTSIDE_RANGE = "outside_range"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    ABOVE_PERCENTILE = "above_percentile"
    ABOVE_ZSCORE = "above_zscore"
    ABOVE_MAD = "above_mad"
    CHANGE_ABOVE = "change_above"
    UNAVAILABLE = "unavailable"


class OutcomeLabel(StrEnum):
    """Episode-level outcome labels."""

    SUCCESS = "success"
    SUBOPTIMAL_SUCCESS = "suboptimal_success"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"


class EventProposalSource(StrEnum):
    """Why a key-event window was selected."""

    CHECKPOINT = "checkpoint"
    METRIC_TRIGGER = "metric_trigger" # jerk spike, stall, gripper chatter, action saturation, collision count jump, path deviation
    SEMANTIC_TRIGGER = "semantic_trigger" # gripper closes, gripper opens, wrist moves near object, object starts moving, lift begins, release happens, final object state
    MANUAL = "manual"


class ProgressAssessment(StrEnum):
    """Progress/regression label for a key event."""
    """Essentially it says whether the progress was towards success, or away"""
    PROGRESS = "progress"
    REGRESSION = "regression"
    NO_PROGRESS = "no_progress"
    RECOVERY = "recovery"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class TimeWindow:
    """A timestamp or time interval in seconds."""

    start_sec: float
    end_sec: float | None = None

    def __post_init__(self) -> None:
        if self.start_sec < 0:
            raise SchemaValidationError("start_sec must be non-negative.")
        if self.end_sec is not None and self.end_sec < self.start_sec:
            raise SchemaValidationError("end_sec must be >= start_sec.")

    @property
    def center_sec(self) -> float:
        """Return the interval midpoint, or the single timestamp."""
        if self.end_sec is None:
            return self.start_sec
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(slots=True)
class EvidenceRef:
    """Reference to metric, visual, telemetry, or task-contract evidence."""

    source: EvidenceSource
    reference_id: str
    time_window: TimeWindow | None = None
    modality: Modality | None = None
    description: str | None = None
    value: Any | None = None


@dataclass(slots=True)
class NumericSummary:
    """Common summary statistics for a metric series."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p95: float | None = None
    count: int | None = None


@dataclass(slots=True)
class Threshold:
    """Threshold or rule used to flag metric events.

    A threshold is not assumed to be universal. Its basis must say whether it
    came from a hard robot limit, the task contract, intra-episode profiling,
    dataset calibration, or a heuristic default.
    """

    name: str
    basis: ThresholdBasis | str
    condition: ThresholdCondition
    severity_bands: list["SeverityBand"] = field(default_factory=list)
    description: str | None = None
    computed_from: list[EvidenceRef] = field(default_factory=list)
    numeric_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.name, "Threshold.name")
        require_non_empty(self.basis, "Threshold.basis")


@dataclass(slots=True)
class ThresholdCondition:
    """Concrete condition used by a threshold.

    Examples:
    - value > episode_p95
    - 3 < value < 4
    - value is outside [low, high]
    - metric is unavailable, so do not make a grounded claim
    """

    operator: ThresholdOperator | str
    threshold_value: float | int | str | bool | None = None # For single comparison targets
    lower: float | int | None = None
    upper: float | int | None = None
    inclusive_lower: bool = False
    inclusive_upper: bool = False
    reference_statistic: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.operator, "ThresholdCondition.operator")
        if self.lower is not None and self.upper is not None and self.upper < self.lower:
            raise SchemaValidationError("ThresholdCondition.upper must be >= lower.")


@dataclass(slots=True)
class SeverityBand:
    """Mapping from a numeric interval to a severity label."""

    severity: Severity | str
    lower: float | int | None = None
    upper: float | int | None = None
    inclusive_lower: bool = True
    inclusive_upper: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.severity, "SeverityBand.severity")
        if self.lower is not None and self.upper is not None and self.upper < self.lower:
            raise SchemaValidationError("SeverityBand.upper must be >= lower.")


@dataclass(slots=True)
class SchemaBase:
    """Base mixin for lightweight schema objects."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert a schema object to a JSON-compatible dictionary."""
        return _to_jsonable(asdict(self))


def require_non_empty(value: str | list[Any], field_name: str) -> None:
    """Validate that a string or list is present."""
    if not value:
        raise SchemaValidationError(f"{field_name} must be non-empty.")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
