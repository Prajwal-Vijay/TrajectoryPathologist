"""VLM-based key-event labelling."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pathologist.schemas.common import EvidenceRef, EvidenceSource, Modality, ProgressAssessment
from pathologist.schemas.key_event import KeyEvent, KeyEventLabel, KeyEventProposal
from pathologist.schemas.metrics import MetricEvent
from pathologist.schemas.task_contract import TaskContract
from pathologist.schemas.trajectory_bundle import CameraFrameRef


class EventLabelVLMClient(Protocol):
    """Provider-neutral client used by the event labeler."""

    def label_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON-like event label dictionary."""


class VLMEventLabeler:
    """Populate KeyEventLabel objects from selected key-event proposals."""

    def __init__(self, client: EventLabelVLMClient) -> None:
        self.client = client

    def label(
        self,
        proposal: KeyEventProposal,
        task_contract: TaskContract,
    ) -> KeyEvent:
        """Call the VLM client and attach a structured KeyEventLabel."""

        payload = build_event_label_payload(proposal, task_contract)
        response = self.client.label_event(payload)
        label = parse_key_event_label(
            proposal=proposal,
            response=response,
            metric_evidence_refs=_metric_evidence_refs(proposal),
            visual_evidence_refs=_visual_evidence_refs(proposal),
        )
        return KeyEvent(proposal=proposal, label=label)

    def label_many(
        self,
        proposals: list[KeyEventProposal],
        task_contract: TaskContract,
    ) -> list[KeyEvent]:
        """Label multiple proposals in rank order."""

        return [self.label(proposal, task_contract) for proposal in proposals]


def build_event_label_payload(
    proposal: KeyEventProposal,
    task_contract: TaskContract,
) -> dict[str, Any]:
    """Build a compact, grounded payload for event interpretation."""

    return {
        "instructions": (
            "Interpret this selected robot trajectory event. Use metric values and frame "
            "references as evidence. Do not invent numeric values. If object/contact state "
            "is not visible or not supported, mark it uncertain."
        ),
        "output_schema": {
            "phase": "string or null",
            "event_type": "string or null",
            "objects_involved": "list of strings; empty if unknown",
            "contact_or_grasp_state": "string or null",
            "progress_assessment": [
                ProgressAssessment.PROGRESS.value,
                ProgressAssessment.REGRESSION.value,
                ProgressAssessment.NO_PROGRESS.value,
                ProgressAssessment.RECOVERY.value,
                ProgressAssessment.UNCERTAIN.value,
            ],
            "pathology_labels": "list of strings",
            "metric_evidence_ids": "list of metric evidence IDs used",
            "visual_evidence_ids": "list of visual evidence IDs used",
            "explanation": "short grounded explanation",
            "uncertainty_notes": "list of uncertainty notes",
        },
        "task_contract": _jsonable(task_contract.to_dict()),
        "proposal": {
            "event_id": proposal.event_id,
            "time_window": _time_window_to_dict(proposal.time_window),
            "trigger_reason": proposal.trigger_reason,
            "source": str(proposal.source),
            "metadata": _jsonable(proposal.metadata),
        },
        "metric_evidence": [
            _metric_event_payload(event, evidence_id=f"metric_event_{index:03d}")
            for index, event in enumerate(proposal.metric_events)
        ],
        "visual_evidence": [
            _frame_payload(frame, evidence_id=f"frame_{index:03d}")
            for index, frame in enumerate(proposal.candidate_frames)
        ],
    }


def parse_key_event_label(
    proposal: KeyEventProposal,
    response: dict[str, Any],
    metric_evidence_refs: dict[str, EvidenceRef],
    visual_evidence_refs: dict[str, EvidenceRef],
) -> KeyEventLabel:
    """Parse a JSON-like VLM response into a KeyEventLabel."""

    label_data = response.get("label", response)
    if not isinstance(label_data, dict):
        raise ValueError("VLM event label response must be a dictionary.")

    metric_ids = _string_list(label_data.get("metric_evidence_ids"))
    visual_ids = _string_list(label_data.get("visual_evidence_ids"))

    return KeyEventLabel(
        event_id=proposal.event_id,
        phase=_optional_string(label_data.get("phase")),
        event_type=_optional_string(label_data.get("event_type")),
        objects_involved=_string_list(label_data.get("objects_involved")),
        contact_or_grasp_state=_optional_string(label_data.get("contact_or_grasp_state")),
        progress_assessment=_progress_assessment(label_data.get("progress_assessment")),
        pathology_labels=_string_list(label_data.get("pathology_labels")),
        metric_evidence=_select_refs(metric_ids, metric_evidence_refs),
        visual_evidence=_select_refs(visual_ids, visual_evidence_refs),
        explanation=_required_string(label_data.get("explanation"), "explanation"),
        uncertainty_notes=_string_list(label_data.get("uncertainty_notes")),
        metadata={
            "raw_vlm_response": _jsonable(response),
            "metric_evidence_ids": metric_ids,
            "visual_evidence_ids": visual_ids,
        },
    )


def _metric_event_payload(event: MetricEvent, evidence_id: str) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "metric_name": event.metric_name,
        "time_window": _time_window_to_dict(event.time_window),
        "value": _jsonable(event.value),
        "severity": str(event.severity),
        "reason": event.reason,
        "threshold": _jsonable(event.threshold),
    }


def _frame_payload(frame: CameraFrameRef, evidence_id: str) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "camera_name": frame.camera_name,
        "timestamp_sec": frame.timestamp_sec,
        "image_path": str(frame.image_path),
        "frame_index": frame.frame_index,
        "time_window": _time_window_to_dict(frame.time_window),
    }


def _metric_evidence_refs(proposal: KeyEventProposal) -> dict[str, EvidenceRef]:
    return {
        f"metric_event_{index:03d}": EvidenceRef(
            source=EvidenceSource.METRIC,
            reference_id=f"{proposal.event_id}:metric_event_{index:03d}",
            time_window=event.time_window,
            description=event.reason,
            value=_jsonable(event.value),
        )
        for index, event in enumerate(proposal.metric_events)
    }


def _visual_evidence_refs(proposal: KeyEventProposal) -> dict[str, EvidenceRef]:
    return {
        f"frame_{index:03d}": EvidenceRef(
            source=EvidenceSource.VISUAL,
            reference_id=str(frame.image_path),
            time_window=frame.time_window,
            modality=Modality.RGB,
            description=f"{frame.camera_name} frame at {frame.timestamp_sec:.3f}s",
            value={
                "camera_name": frame.camera_name,
                "timestamp_sec": frame.timestamp_sec,
                "frame_index": frame.frame_index,
            },
        )
        for index, frame in enumerate(proposal.candidate_frames)
    }


def _select_refs(
    selected_ids: list[str],
    refs_by_id: dict[str, EvidenceRef],
) -> list[EvidenceRef]:
    if not selected_ids:
        return list(refs_by_id.values())
    return [
        refs_by_id[item]
        for item in selected_ids
        if item in refs_by_id
    ]


def _progress_assessment(value: Any) -> ProgressAssessment | str:
    if isinstance(value, ProgressAssessment):
        return value
    if isinstance(value, str):
        allowed = {item.value for item in ProgressAssessment}
        if value in allowed:
            return value
    return ProgressAssessment.UNCERTAIN


def _required_string(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"VLM event label response requires non-empty {field_name}.")


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _time_window_to_dict(time_window: Any) -> dict[str, float | None] | None:
    if time_window is None:
        return None
    return {
        "start_sec": time_window.start_sec,
        "end_sec": time_window.end_sec,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
