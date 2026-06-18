"""Smoke tests for OpenAI event-label client request construction."""

from __future__ import annotations

from pathologist.vlm.openai_event_client import OpenAIEventLabelClient


def test_openai_event_client_builds_responses_request() -> None:
    client = OpenAIEventLabelClient(api_key="test-key", model="test-model")
    request_body = client._build_request_body(
        {
            "instructions": "label event",
            "visual_evidence": [
                {
                    "evidence_id": "frame_000",
                    "image_path": "/tmp/does_not_exist.jpg",
                }
            ],
        }
    )

    assert request_body["model"] == "test-model"
    assert request_body["input"][0]["content"][0]["type"] == "input_text"
    assert request_body["text"]["format"]["type"] == "json_schema"
    assert request_body["text"]["format"]["schema"]["required"] == [
        "visual_observation",
        "phase",
        "event_type",
        "objects_involved",
        "contact_or_grasp_state",
        "progress_assessment",
        "pathology_labels",
        "metric_evidence_ids",
        "visual_evidence_ids",
        "explanation",
        "uncertainty_notes",
    ]


def test_openai_event_client_builds_visual_schema_request() -> None:
    client = OpenAIEventLabelClient(api_key="test-key", model="test-model")
    request_body = client._build_request_body(
        {"instructions": "label visual event", "visual_evidence": []},
        schema_name="trajectory_visual_event_label",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"visual_observation": {"type": "string"}},
            "required": ["visual_observation"],
        },
        include_images=True,
    )

    assert request_body["text"]["format"]["schema"]["required"] == ["visual_observation"]
    assert request_body["text"]["format"]["name"] == "trajectory_visual_event_label"
