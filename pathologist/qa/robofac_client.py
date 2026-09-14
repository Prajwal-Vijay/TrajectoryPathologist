"""VLM clients for RoboFAC QA evaluation."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from pathologist.schemas.trajectory_bundle import CameraFrameRef


class RoboFACQAClient(Protocol):
    """Protocol for clients that answer RoboFAC questions from selected frames."""

    def answer(
        self,
        prompt: str,
        frames: list[CameraFrameRef],
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Return a raw text answer."""


class OpenAIRoboFACQAError(RuntimeError):
    """OpenAI RoboFAC QA request failure with retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class OpenAIRoboFACQAClient:
    """OpenAI Responses API client for RoboFAC QA."""

    RETRYABLE_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_url: str = "https://api.openai.com/v1/responses",
        timeout_sec: float | None = None,
        max_output_tokens: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        include_frame_metadata: bool = True,
        include_frame_timing_metadata: bool = True,
        max_retries: int | None = None,
        retry_backoff_sec: float | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")
        self.model = model or os.environ.get("OPENAI_ROBOFAC_QA_MODEL", "gpt-4o")
        self.api_url = api_url
        self.timeout_sec = _resolve_timeout_sec(timeout_sec)
        self.max_output_tokens = max_output_tokens
        self.temperature = _resolve_temperature(temperature)
        self.top_p = _resolve_top_p(top_p)
        self.include_frame_metadata = bool(include_frame_metadata)
        self.include_frame_timing_metadata = bool(include_frame_timing_metadata)
        self.max_retries = _resolve_nonnegative_int(
            max_retries,
            env_var="OPENAI_ROBOFAC_QA_MAX_RETRIES",
            default=2,
            label="OpenAI RoboFAC QA max retries",
        )
        self.retry_backoff_sec = _resolve_nonnegative_float(
            retry_backoff_sec,
            env_var="OPENAI_ROBOFAC_QA_RETRY_BACKOFF_SEC",
            default=2.0,
            label="OpenAI RoboFAC QA retry backoff seconds",
        )

    def answer(
        self,
        prompt: str,
        frames: list[CameraFrameRef],
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Answer one RoboFAC question from selected RGB frames."""

        request_body = self._build_request_body(
            prompt=prompt,
            frames=frames,
            system_prompt=system_prompt,
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
        for attempt in range(self.max_retries + 1):
            try:
                return self._send_request(request)
            except OpenAIRoboFACQAError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay_sec = min(self.retry_backoff_sec * (2**attempt), 30.0)
                if delay_sec > 0:
                    time.sleep(delay_sec)
        raise RuntimeError("OpenAI RoboFAC QA request retry loop exited unexpectedly.")

    def _send_request(self, request: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in self.RETRYABLE_HTTP_STATUS_CODES
            raise OpenAIRoboFACQAError(
                f"OpenAI RoboFAC QA request failed: HTTP {exc.code} {error_body}",
                retryable=retryable,
                status_code=exc.code,
            ) from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError, OSError) as exc:
            raise OpenAIRoboFACQAError(
                f"OpenAI RoboFAC QA request failed: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        return _extract_output_text(response_body)

    def _build_request_body(
        self,
        prompt: str,
        frames: list[CameraFrameRef],
        system_prompt: str | None = None,
    ) -> dict[str, object]:
        """Build the Responses API request body."""

        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                }
            )
        messages.append(
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
        )
        request_body = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            request_body["temperature"] = self.temperature
        if self.top_p is not None:
            request_body["top_p"] = self.top_p
        return request_body

    def _image_inputs(self, frames: list[CameraFrameRef]) -> list[dict[str, str]]:
        content: list[dict[str, str]] = []
        for index, frame in enumerate(frames):
            path = Path(frame.image_path)
            if not path.exists() or not path.is_file():
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
            if self.include_frame_metadata:
                label = f"[View: {frame.camera_name}]"
                if self.include_frame_timing_metadata:
                    label += (
                        f" frame_{index}: timestamp_sec={frame.timestamp_sec:.3f}, "
                        f"original_frame_index={frame.frame_index}"
                    )
                content.append(
                    {
                        "type": "input_text",
                        "text": label,
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

    def answer(
        self,
        prompt: str,
        frames: list[CameraFrameRef],
        *,
        system_prompt: str | None = None,
    ) -> str:
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


def _resolve_temperature(temperature: float | None) -> float | None:
    if temperature is None:
        env_temperature = os.environ.get("OPENAI_ROBOFAC_QA_TEMPERATURE")
        if env_temperature in {None, ""}:
            return None
        try:
            temperature = float(env_temperature)
        except ValueError as exc:
            raise ValueError(
                "OPENAI_ROBOFAC_QA_TEMPERATURE must be a number between 0 and 2."
            ) from exc
    if not 0 <= temperature <= 2:
        raise ValueError("OpenAI RoboFAC QA temperature must be between 0 and 2.")
    return float(temperature)


def _resolve_top_p(top_p: float | None) -> float | None:
    if top_p is None:
        return None
    if not 0 <= top_p <= 1:
        raise ValueError("OpenAI RoboFAC QA Top-P must be between 0 and 1.")
    return float(top_p)


def _resolve_timeout_sec(timeout_sec: float | None) -> float:
    return _resolve_nonnegative_float(
        timeout_sec,
        env_var="OPENAI_ROBOFAC_QA_TIMEOUT_SEC",
        default=120.0,
        label="OpenAI RoboFAC QA timeout seconds",
        allow_zero=False,
    )


def _resolve_nonnegative_int(
    value: int | None,
    *,
    env_var: str,
    default: int,
    label: str,
) -> int:
    if value is None:
        env_value = os.environ.get(env_var)
        if env_value in {None, ""}:
            value = default
        else:
            try:
                value = int(env_value)
            except ValueError as exc:
                raise ValueError(f"{env_var} must be a non-negative integer.") from exc
    if value < 0:
        raise ValueError(f"{label} must be non-negative.")
    return int(value)


def _resolve_nonnegative_float(
    value: float | None,
    *,
    env_var: str,
    default: float,
    label: str,
    allow_zero: bool = True,
) -> float:
    if value is None:
        env_value = os.environ.get(env_var)
        if env_value in {None, ""}:
            value = default
        else:
            try:
                value = float(env_value)
            except ValueError as exc:
                raise ValueError(f"{env_var} must be a number.") from exc
    if allow_zero:
        valid = value >= 0
        qualifier = "non-negative"
    else:
        valid = value > 0
        qualifier = "positive"
    if not valid:
        raise ValueError(f"{label} must be {qualifier}.")
    return float(value)
