"""OpenAI Responses API client for event labelling."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class OpenAIEventLabelClient:
    """Concrete EventLabelVLMClient backed by OpenAI's Responses API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str = "https://api.openai.com/v1/responses",
        timeout_sec: float = 120.0,
        max_output_tokens: int = 1200,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")
        self.model = model or os.environ.get("OPENAI_EVENT_LABEL_MODEL", "gpt-4o")
        self.api_url = api_url
        self.timeout_sec = timeout_sec
        self.max_output_tokens = max_output_tokens

    def label_visual_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI for visual-only event interpretation."""

        return self._call_structured(
            payload=payload,
            schema_name="trajectory_visual_event_label",
            schema=_visual_event_json_schema(),
            include_images=True,
        )

    def integrate_event_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI to integrate visual labels and metric grounding cards."""

        return self._call_structured(
            payload=payload,
            schema_name="trajectory_key_event_label",
            schema=_event_label_json_schema(),
            include_images=False,
        )

    def label_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Backward-compatible single-stage label call."""
        ''' THis is the old version '''
        return self._call_structured(
            payload=payload,
            schema_name="trajectory_key_event_label",
            schema=_event_label_json_schema(),
            include_images=True,
        )

    def _call_structured(
        self,
        payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
        include_images: bool,
    ) -> dict[str, Any]:
        """Call OpenAI and return parsed JSON matching the provided schema."""

        request_body = self._build_request_body(
            payload,
            schema_name=schema_name,
            schema=schema,
            include_images=include_images,
        )
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI event labelling request failed: {exc.code} {error_body}") from exc

        output_text = _extract_output_text(response_body)
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI event labelling response was not valid JSON: {output_text}") from exc

    def _build_request_body(
        self,
        payload: dict[str, Any],
        schema_name: str = "trajectory_key_event_label",
        schema: dict[str, Any] | None = None,
        include_images: bool = True,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Return only JSON matching the schema. The event payload follows:\n"
                    f"{json.dumps(payload, indent=2)}"
                ),
            }
        ]
        if include_images:
            content.extend(_image_inputs_from_payload(payload))
        return {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema or _event_label_json_schema(),
                }
            },
            "max_output_tokens": self.max_output_tokens,
        }


def _image_inputs_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    image_inputs: list[dict[str, Any]] = []
    for frame in payload.get("visual_evidence", []):
        image_path = frame.get("image_path")
        if not image_path:
            continue
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            continue
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
        image_inputs.append(
            {
                "type": "input_text",
                "text": (
                    f"Image for {frame.get('evidence_id')}: "
                    f"camera={frame.get('camera_name')}, "
                    f"timestamp_sec={frame.get('timestamp_sec')}, "
                    f"frame_index={frame.get('frame_index')}"
                ),
            }
        )
        image_inputs.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded_image}",
            }
        )
    return image_inputs


def _extract_output_text(response_body: dict[str, Any]) -> str:
    if isinstance(response_body.get("output_text"), str):
        return response_body["output_text"]
    output_parts: list[str] = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                output_parts.append(content["text"])
    if output_parts:
        return "".join(output_parts)
    raise ValueError(f"Could not find output text in OpenAI response: {response_body}")


def _event_label_json_schema() -> dict[str, Any]:
    string_array = {
        "type": "array",
        "items": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "visual_observation": {"type": "string"},
            "phase": {"type": "string"},
            "event_type": {"type": "string"},
            "objects_involved": string_array,
            "contact_or_grasp_state": {"type": "string"},
            "progress_assessment": {
                "type": "string",
                "enum": ["progress", "regression", "no_progress", "recovery", "uncertain"],
            },
            "pathology_labels": string_array,
            "metric_evidence_ids": string_array,
            "visual_evidence_ids": string_array,
            "explanation": {"type": "string"},
            "uncertainty_notes": string_array,
        },
        "required": [
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
        ],
    }


def _visual_event_json_schema() -> dict[str, Any]:
    string_array = {
        "type": "array",
        "items": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "visual_observation": {"type": "string"},
            "phase": {"type": "string"},
            "event_type": {"type": "string"},
            "objects_involved": string_array,
            "contact_or_grasp_state": {"type": "string"},
            "progress_assessment": {
                "type": "string",
                "enum": ["progress", "regression", "no_progress", "recovery", "uncertain"],
            },
            "visible_pathology_labels": string_array,
            "visual_evidence_ids": string_array,
            "uncertainty_notes": string_array,
        },
        "required": [
            "visual_observation",
            "phase",
            "event_type",
            "objects_involved",
            "contact_or_grasp_state",
            "progress_assessment",
            "visible_pathology_labels",
            "visual_evidence_ids",
            "uncertainty_notes",
        ],
    }
