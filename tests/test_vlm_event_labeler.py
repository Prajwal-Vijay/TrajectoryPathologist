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
)
from pathologist.schemas.key_event import KeyEventProposal
from pathologist.vlm import VLMEventLabeler, build_event_label_payload


class FakeEventLabelClient:
    """Small deterministic client for parser tests."""

    def __init__(self) -> None:
        self.last_payload = None

    def label_event(self, payload):
        self.last_payload = payload
        return {
            "visual_observation": "The head camera shows the pot being contacted by the gripper.",
            "phase": "grasp",
            "event_type": "unstable_grasp",
            "objects_involved": ["pot"],
            "contact_or_grasp_state": "left hand appears engaged; right hand uncertain",
            "progress_assessment": "regression",
            "pathology_labels": ["one_handed_grasp", "instability"],
            "metric_evidence_ids": ["metric_event_000"],
            "visual_evidence_ids": ["frame_000"],
            "explanation": "The event overlaps a gripper chatter burst and the visible frame suggests an unstable one-handed grasp.",
            "uncertainty_notes": ["Object contact is inferred from the selected frame, not force data."],
        }


def test_vlm_event_labeler_populates_key_event_label() -> None:
    proposal = KeyEventProposal(
        event_id="demo_metric_overlap_001",
        time_window=TimeWindow(1.0, 1.2),
        source=EventProposalSource.METRIC_TRIGGER,
        trigger_reason="2 metrics overlap",
        metric_events=[
            MetricEvent(
                metric_name="gripper_chatter",
                time_window=TimeWindow(1.0, 1.2),
                value={"local_transition_rate": 3.0},
                severity=Severity.MEDIUM,
                reason="local gripper transition rate exceeded p95",
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
    assert key_event.label.event_type == "unstable_grasp"
    assert key_event.label.objects_involved == ["pot"]
    assert key_event.label.progress_assessment == ProgressAssessment.REGRESSION.value
    assert len(key_event.label.metric_evidence) == 1
    assert len(key_event.label.visual_evidence) == 1
    assert key_event.label.metadata["visual_observation"] == "The head camera shows the pot being contacted by the gripper."
    assert client.last_payload is not None
    assert client.last_payload["metric_trigger_summary"][0]["evidence_id"] == "metric_event_000"
    assert client.last_payload["metric_trigger_summary"][0]["selection_role"] == "metric trigger for visual inspection"
    assert "severity" not in client.last_payload["metric_trigger_summary"][0]
    assert "value" not in client.last_payload["metric_trigger_summary"][0]


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

    payload = build_event_label_payload(proposal, task_contract)

    assert payload["task_contract"]["task_name"] == "Lift pot"
    assert payload["proposal"]["event_id"] == "demo_metric_overlap_001"
    assert "output_schema" in payload
    assert "visual_observation" in payload["output_schema"]
