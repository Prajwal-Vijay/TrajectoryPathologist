"""Task contract schema for v1 trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathologist.schemas.common import SchemaBase, require_non_empty


@dataclass(slots=True)
class FailureCase:
    """Known or likely way a task can fail or become suboptimal."""

    label: str
    description: str
    severity_hint: str | None = None
    observable_signals: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_empty(self.label, "FailureCase.label")
        require_non_empty(self.description, "FailureCase.description")


@dataclass(slots=True)
class TaskContract(SchemaBase):
    """Human-provided task semantics required by the v1 analyzer."""

    task_name: str = ""
    task_description: str = ""
    success_criteria: str = ""
    failure_criteria: str = ""
    objects_of_interest: list[str] = field(default_factory=list)
    known_failure_cases: list[FailureCase] = field(default_factory=list)
    subtask_phases: list[str] = field(default_factory=list)
    extra_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.task_name, "TaskContract.task_name")
        require_non_empty(self.task_description, "TaskContract.task_description")
        require_non_empty(self.success_criteria, "TaskContract.success_criteria")
        require_non_empty(self.failure_criteria, "TaskContract.failure_criteria")
