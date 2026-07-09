"""VLM clients for RoboFAC QA evaluation."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from pathologist.schemas.trajectory_bundle import CameraFrameRef


class RoboFACQAClient(Protocol):
    """Protocol for clients that answer RoboFAC questions from selected frames."""

    def answer(self, prompt: str, frames: list[CameraFrameRef]) -> str:
        """Return a raw text answer."""


class OpenAIRoboFACQAClient:
    """OpenAI Responses API client for RoboFAC QA."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str = "https://api.openai.com/v1/responses",
        timeout_sec: float = 120.0,
        max_output_tokens: int = 512,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")
        self.model = model or os.environ.get("OPENAI_ROBOFAC_QA_MODEL", "gpt-4o")
        self.api_url = api_url
        self.timeout_sec = timeout_sec
        self.max_output_tokens = max_output_tokens

    def answer(self, prompt: str, frames: list[CameraFrameRef]) -> str:
        """Answer one RoboFAC question from selected RGB frames."""

        request_body = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        *self._image_inputs(frames),
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_output_tokens": self.max_output_tokens,
        }
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
            raise RuntimeError(f"OpenAI RoboFAC QA request failed: {exc.code} {error_body}") from exc
        return _extract_output_text(response_body)

    def _image_inputs(self, frames: list[CameraFrameRef]) -> list[dict[str, str]]:
        content: list[dict[str, str]] = []
        for index, frame in enumerate(frames):
            path = Path(frame.image_path)
            if not path.exists() or not path.is_file():
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"frame_{index}: timestamp_sec={frame.timestamp_sec:.3f}, "
                        f"original_frame_index={frame.frame_index}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded_image}",
                }
            )
        return content


class StaticRoboFACQAClient:
    """Deterministic client for smoke tests and dry runs."""

    def __init__(self, answer_text: str = "") -> None:
        self.answer_text = answer_text

    def answer(self, prompt: str, frames: list[CameraFrameRef]) -> str:
        return self.answer_text


def _extract_output_text(response_body: dict[str, object]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output_parts: list[str] = []
    output = response_body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    output_parts.append(part["text"])
    if output_parts:
        return "".join(output_parts)
    raise ValueError(f"Could not find output text in OpenAI response: {response_body}")
