"""Smoke tests for VLM event labelling."""

from __future__ import annotations

from pathlib import Path

from pathologist.schemas import (
    CameraFrameRef,
    EventProposalSource,
    MetricEvent,
    ProgressAssessment,
    Severity,
    TaskContract,
    TimeWindow,
    VisualEventLabel,
)
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.vlm import (
    VLMEventLabeler,
    build_evidence_integration_payload,
    build_metric_grounding_cards,
    build_visual_label_payload,
)


class FakeEventLabelClient:
    """Small deterministic client for parser tests."""

    def __init__(self) -> None:
        self.visual_payload = None
        self.integration_payload = None

    def label_visual_event(self, payload):
        self.visual_payload = payload
        return {
            "visual_observation": "The head camera shows the pot being contacted by the gripper.",
            "phase": "grasp",
            "event_type": "successful_grasp",
            "objects_involved": ["pot"],
            "contact_or_grasp_state": "gripper appears engaged with the pot handle",
            "progress_assessment": "progress",
            "visible_pathology_labels": [],
            "visual_evidence_ids": ["frame_000"],
            "uncertainty_notes": [],
        }

    def integrate_event_evidence(self, payload):
        self.integration_payload = payload
        return {
            "visual_observation": "The head camera shows the pot being contacted by the gripper.",
            "phase": "grasp",
            "event_type": "successful_grasp_with_rough_motion",
            "objects_involved": ["pot"],
            "contact_or_grasp_state": "gripper appears engaged with the pot handle",
            "progress_assessment": "progress",
            "pathology_labels": ["rough_motion"],
            "metric_evidence_ids": ["metric_event_000"],
            "visual_evidence_ids": ["frame_000"],
            "explanation": "Visual evidence shows a maintained grasp while the metric card grounds this as a rough-motion event, not grasp failure.",
            "uncertainty_notes": [],
        }


def test_vlm_event_labeler_populates_key_event_label() -> None:
    proposal = KeyEventProposal(
        event_id="demo_metric_overlap_001",
        time_window=TimeWindow(1.0, 1.2),
        source=EventProposalSource.METRIC_TRIGGER,
        trigger_reason="2 metrics overlap",
        metric_events=[
            MetricEvent(
                metric_name="motion_smoothness",
                time_window=TimeWindow(1.0, 1.2),
                value=25.0,
                severity=Severity.HIGH,
                reason="jerk norm exceeded p95",
            )
        ],
        candidate_frames=[
            CameraFrameRef(
                camera_name="head",
                timestamp_sec=1.1,
                image_path=Path("/tmp/frame.jpg"),
                frame_index=10,
            )
        ],
    )
    task_contract = TaskContract(
        task_name="Lift pot",
        task_description="Lift the pot using both hands.",
        success_criteria="The pot is lifted and held stably by both hands.",
        failure_criteria="The pot is dropped or only held unstably by one hand.",
        objects_of_interest=["pot"],
        subtask_phases=["approach", "grasp", "lift"],
    )
    client = FakeEventLabelClient()

    key_event = VLMEventLabeler(client).label(proposal, task_contract)

    assert key_event.label is not None
    assert key_event.label.phase == "grasp"
    assert key_event.label.event_type == "successful_grasp_with_rough_motion"
    assert key_event.label.objects_involved == ["pot"]
    assert key_event.label.progress_assessment == ProgressAssessment.PROGRESS.value
    assert len(key_event.label.metric_evidence) == 1
    assert len(key_event.label.visual_evidence) == 1
    assert key_event.label.metadata["visual_observation"] == "The head camera shows the pot being contacted by the gripper."
    assert key_event.label.metadata["visual_label"]["event_type"] == "successful_grasp"
    assert key_event.label.metadata["metric_grounding_cards"][0]["metric_name"] == "motion_smoothness"
    assert key_event.label.metadata["metric_grounding_cards"][0]["observed_value"] == 25.0
    assert client.visual_payload is not None
    assert client.integration_payload is not None
    assert "metric_events" not in client.visual_payload
    assert "metric_trigger_summary" not in client.visual_payload
    assert "metric_grounding_cards" in client.integration_payload


def test_event_label_payload_contains_task_and_proposal_context() -> None:
    proposal = KeyEventProposal(
        event_id="demo_metric_overlap_001",
        time_window=TimeWindow(1.0, 1.2),
        source=EventProposalSource.METRIC_TRIGGER,
        trigger_reason="1 metric overlaps",
    )
    task_contract = TaskContract(
        task_name="Lift pot",
        task_description="Lift the pot using both hands.",
        success_criteria="The pot is lifted and held stably by both hands.",
        failure_criteria="The pot is dropped or unstable.",
    )

    payload = build_visual_label_payload(proposal, task_contract)

    assert payload["task_contract"]["task_name"] == "Lift pot"
    assert payload["proposal"]["event_id"] == "demo_metric_overlap_001"
    assert "output_schema" in payload
    assert "visual_observation" in payload["output_schema"]
    serialized_payload = str(payload)
    assert "metric_name" not in serialized_payload
    assert "severity" not in serialized_payload
    assert "threshold" not in serialized_payload


def test_metric_grounding_cards_preserve_numeric_grounding() -> None:
    event = MetricEvent(
        metric_name="motion_smoothness",
        time_window=TimeWindow(1.0),
        value=25.0,
        severity=Severity.HIGH,
        reason="jerk norm 25.0 exceeded threshold 10.0",
    )

    cards = build_metric_grounding_cards([event])

    assert cards[0].metric_name == "motion_smoothness"
    assert cards[0].observed_value == 25.0
    assert cards[0].severity == "high"
    assert "rough motion" in cards[0].allowed_metric_claims
    assert "grasp failed unless visual evidence supports it" in cards[0].forbidden_semantic_claims


def test_integration_payload_uses_visual_label_and_metric_cards() -> None:
    proposal = KeyEventProposal(
        event_id="demo_metric_overlap_001",
        time_window=TimeWindow(1.0, 1.2),
        source=EventProposalSource.METRIC_TRIGGER,
        trigger_reason="1 metric overlaps",
        metric_events=[
            MetricEvent(
                metric_name="motion_smoothness",
                time_window=TimeWindow(1.1),
                value=25.0,
                severity=Severity.HIGH,
                reason="jerk norm exceeded p95",
            )
        ],
    )
    task_contract = TaskContract(
        task_name="Lift pot",
        task_description="Lift the pot using both hands.",
        success_criteria="The pot is lifted and held stably by both hands.",
        failure_criteria="The pot is dropped or unstable.",
    )
    visual_response = FakeEventLabelClient().label_visual_event({})
    visual_label = VisualEventLabel(
        event_id=proposal.event_id,
        visual_observation=visual_response["visual_observation"],
        phase=visual_response["phase"],
        event_type=visual_response["event_type"],
        objects_involved=visual_response["objects_involved"],
        contact_or_grasp_state=visual_response["contact_or_grasp_state"],
        progress_assessment=visual_response["progress_assessment"],
        visible_pathology_labels=visual_response["visible_pathology_labels"],
        visual_evidence_ids=visual_response["visual_evidence_ids"],
        uncertainty_notes=visual_response["uncertainty_notes"],
    )
    cards = build_metric_grounding_cards(proposal.metric_events)
    payload = build_evidence_integration_payload(
        proposal=proposal,
        task_contract=task_contract,
        visual_label=visual_label,
        metric_grounding_cards=cards,
    )

    assert payload["visual_label"]["event_type"] == visual_response["event_type"]
    assert payload["metric_grounding_cards"][0]["observed_value"] == 25.0
    assert "metric_packet" not in payload
