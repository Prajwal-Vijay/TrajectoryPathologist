"""VLM-based key-event labelling."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pathologist.schemas.common import EvidenceRef, EvidenceSource, Modality, ProgressAssessment
from pathologist.schemas.key_event import (
    KeyEvent,
    KeyEventLabel,
    KeyEventProposal,
    MetricGroundingCard,
    VisualEventLabel,
)
from pathologist.schemas.metrics import MetricEvent
from pathologist.schemas.task_contract import TaskContract
from pathologist.schemas.trajectory_bundle import CameraFrameRef


class EventLabelVLMClient(Protocol):
    """Provider-neutral client used by the event labeler."""

    def label_visual_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a visual-only event label dictionary."""

    def integrate_event_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a final integrated event label dictionary."""


class VLMEventLabeler:
    """Populate KeyEventLabel objects from selected key-event proposals."""

    def __init__(self, client: EventLabelVLMClient) -> None:
        self.client = client

    def label(
        self,
        proposal: KeyEventProposal,
        task_contract: TaskContract,
    ) -> KeyEvent:
        """Label an event in visual, metric-grounding, and integration stages."""

        visual_payload = build_visual_label_payload(proposal, task_contract)
        visual_response = self.client.label_visual_event(visual_payload)
        visual_label = parse_visual_event_label(proposal, visual_response)

        metric_grounding_cards = build_metric_grounding_cards(proposal.metric_events)
        integration_payload = build_evidence_integration_payload(
            proposal=proposal,
            task_contract=task_contract,
            visual_label=visual_label,
            metric_grounding_cards=metric_grounding_cards,
        )
        integration_response = self.client.integrate_event_evidence(integration_payload)
        label = parse_key_event_label(
            proposal=proposal,
            response=integration_response,
            metric_evidence_refs=_metric_evidence_refs(proposal),
            visual_evidence_refs=_visual_evidence_refs(proposal),
            visual_label=visual_label,
            metric_grounding_cards=metric_grounding_cards,
            raw_visual_response=visual_response,
        )
        return KeyEvent(proposal=proposal, label=label)

    def label_many(
        self,
        proposals: list[KeyEventProposal],
        task_contract: TaskContract,
    ) -> list[KeyEvent]:
        """Label multiple proposals in rank order."""

        return [self.label(proposal, task_contract) for proposal in proposals]


def build_visual_label_payload(
    proposal: KeyEventProposal,
    task_contract: TaskContract,
) -> dict[str, Any]:
    """Build a visual-only payload with no metric values, names, thresholds, or severity."""

    return {
        "instructions": (
            "Interpret only the visible image evidence for this selected robot trajectory window. "
            "Do not infer anything from robot metrics; they are intentionally omitted. "
            "Describe what is visibly happening, especially the head-camera sequence when available. "
            "Compare the first, middle, and last frames before deciding whether the scene is unchanged. "
            "Explicitly look for gripper open/close changes, one manipulator releasing a handle, "
            "loss or gain of contact, and object motion across the frame sequence. "
            "Judge visible task phase, grasp/contact state, object state, and progress/regression "
            "only from frames. "
            "Use uncertain only when the relevant object/contact state is occluded, outside the "
            "frame, or genuinely visually ambiguous. If the object is visibly held, grasped, "
            "released, tilted, dropped, or stable, state that visual observation directly."
        ),
        "output_schema": {
            "visual_observation": "short description of what the images visibly show before metric interpretation",
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
            "visible_pathology_labels": "list of visual issue labels; empty if no visible issue",
            "visual_evidence_ids": "list of visual evidence IDs used",
            "uncertainty_notes": "list of uncertainty notes",
        },
        "task_contract": _jsonable(task_contract.to_dict()),
        "proposal": {
            "event_id": proposal.event_id,
            "time_window": _time_window_to_dict(proposal.time_window),
            "selection_note": "This window was selected upstream; visual labeling should not assume why.",
        },
        "visual_evidence": [
            _frame_payload(frame, evidence_id=f"frame_{index:03d}")
            for index, frame in enumerate(proposal.candidate_frames)
        ],
    }


def build_event_label_payload(
    proposal: KeyEventProposal,
    task_contract: TaskContract,
) -> dict[str, Any]:
    """Backward-compatible alias for the visual-only payload builder."""

    return build_visual_label_payload(proposal, task_contract)


def build_evidence_integration_payload(
    proposal: KeyEventProposal,
    task_contract: TaskContract,
    visual_label: VisualEventLabel,
    metric_grounding_cards: list[MetricGroundingCard],
) -> dict[str, Any]:
    """Build a text-only integration payload from visual semantics and metric cards."""

    return {
        "instructions": (
            "Integrate visual evidence and metric grounding for one robot trajectory event. "
            "The visual label is authoritative for visible grasp/contact, object state, and "
            "task progress. Metric cards are authoritative for numeric motion/control anomalies. "
            "High metric severity alone cannot imply failure, grasp loss, object drop, or regression. "
            "Use metric cards to numerically ground why the event is notable. If visual and metric "
            "evidence disagree, state the disagreement clearly instead of forcing a failure label."
        ),
        "output_schema": {
            "visual_observation": "visual observation copied or refined from the visual label",
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
            "pathology_labels": "list of integrated issue labels",
            "metric_evidence_ids": "list of metric grounding IDs used",
            "visual_evidence_ids": "list of visual evidence IDs used",
            "explanation": "short explanation integrating visual label and numeric metric cards",
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
        "visual_label": _jsonable(visual_label.to_dict()),
        "metric_grounding_cards": [
            _jsonable(card.to_dict())
            for card in metric_grounding_cards
        ],
    }


def parse_visual_event_label(
    proposal: KeyEventProposal,
    response: dict[str, Any],
) -> VisualEventLabel:
    """Parse a visual-only VLM response into an intermediate label."""

    label_data = response.get("label", response)
    if not isinstance(label_data, dict):
        raise ValueError("Visual event label response must be a dictionary.")

    return VisualEventLabel(
        event_id=proposal.event_id,
        visual_observation=_required_string(label_data.get("visual_observation"), "visual_observation"),
        phase=_optional_string(label_data.get("phase")),
        event_type=_optional_string(label_data.get("event_type")),
        objects_involved=_string_list(label_data.get("objects_involved")),
        contact_or_grasp_state=_optional_string(label_data.get("contact_or_grasp_state")),
        progress_assessment=_progress_assessment(label_data.get("progress_assessment")),
        visible_pathology_labels=_string_list(
            label_data.get("visible_pathology_labels", label_data.get("pathology_labels"))
        ),
        visual_evidence_ids=_string_list(label_data.get("visual_evidence_ids")),
        uncertainty_notes=_string_list(label_data.get("uncertainty_notes")),
        metadata={"raw_vlm_response": _jsonable(response)},
    )


def parse_key_event_label(
    proposal: KeyEventProposal,
    response: dict[str, Any],
    metric_evidence_refs: dict[str, EvidenceRef],
    visual_evidence_refs: dict[str, EvidenceRef],
    visual_label: VisualEventLabel | None = None,
    metric_grounding_cards: list[MetricGroundingCard] | None = None,
    raw_visual_response: dict[str, Any] | None = None,
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
            "raw_integration_response": _jsonable(response),
            "raw_visual_response": _jsonable(raw_visual_response),
            "metric_evidence_ids": metric_ids,
            "visual_evidence_ids": visual_ids,
            "visual_observation": _optional_string(label_data.get("visual_observation")),
            "visual_label": _jsonable(visual_label.to_dict()) if visual_label is not None else None,
            "metric_grounding_cards": [
                _jsonable(card.to_dict())
                for card in (metric_grounding_cards or [])
            ],
        },
    )


def build_metric_grounding_cards(events: list[MetricEvent]) -> list[MetricGroundingCard]:
    """Build deterministic metric cards with explicit claim boundaries."""

    return [
        MetricGroundingCard(
            evidence_id=f"metric_event_{index:03d}",
            metric_name=event.metric_name,
            time_window=event.time_window,
            observed_value=_jsonable(event.value),
            severity=str(event.severity),
            threshold_context=_threshold_context(event),
            relative_level=_relative_level(event),
            metric_interpretation=_metric_interpretation(event.metric_name),
            allowed_metric_claims=_allowed_metric_claims(event.metric_name),
            forbidden_semantic_claims=_forbidden_semantic_claims(),
        )
        for index, event in enumerate(events)
    ]


def _threshold_context(event: MetricEvent) -> dict[str, Any]:
    threshold = event.threshold
    if threshold is None:
        return {"metric_reason": event.reason}
    return {
        "metric_reason": event.reason,
        "threshold_name": threshold.name,
        "threshold_basis": str(threshold.basis),
        "condition": _jsonable(threshold.condition),
        "numeric_context": _jsonable(threshold.numeric_context),
        "severity_bands": _jsonable(threshold.severity_bands),
    }


def _relative_level(event: MetricEvent) -> str:
    threshold = event.threshold
    severity = str(event.severity)
    if threshold is None:
        return f"{severity}_metric_trigger"
    threshold_value = getattr(threshold.condition, "threshold_value", None)
    if isinstance(event.value, (int, float)) and isinstance(threshold_value, (int, float)) and threshold_value != 0:
        return f"{severity}_metric_trigger; observed_value_is_{event.value / threshold_value:.3f}x_threshold"
    return f"{severity}_metric_trigger_relative_to_{threshold.name}"


def _metric_interpretation(metric_name: str) -> str:
    if metric_name == "motion_smoothness":
        return "sudden or rough robot motion; possible dynamic disturbance"
    if metric_name == "gripper_chatter":
        return "rapid gripper open/close transitions; possible unstable command or contact adjustment"
    if metric_name == "stalling":
        return "low robot motion over the event window; possible pause, stall, or waiting"
    if metric_name == "action_range_saturation":
        return "action command reached configured control bounds"
    if metric_name == "action_distribution_entropy":
        return "unusual action distribution for review"
    return "metric anomaly selected this event for review"


def _allowed_metric_claims(metric_name: str) -> list[str]:
    if metric_name == "motion_smoothness":
        return ["rough motion", "jerk spike", "dynamic anomaly", "possible impact or abrupt correction"]
    if metric_name == "gripper_chatter":
        return ["rapid gripper transitions", "possible command chatter", "possible grasp adjustment"]
    if metric_name == "stalling":
        return ["stalling", "pause", "low progress in telemetry"]
    if metric_name == "action_range_saturation":
        return ["control saturation", "action at configured bounds"]
    return ["metric-triggered anomaly", "review-worthy telemetry pattern"]


def _forbidden_semantic_claims() -> list[str]:
    return [
        "grasp failed unless visual evidence supports it",
        "object dropped unless visual evidence supports it",
        "task failed unless visual evidence or task state supports it",
        "regression unless visual progress assessment supports it",
    ]


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
