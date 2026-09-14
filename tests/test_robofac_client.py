"""Tests for RoboFAC QA client request construction."""

from __future__ import annotations

from pathlib import Path

from pathologist.qa.robofac_client import OpenAIRoboFACQAClient
from pathologist.schemas import CameraFrameRef


def test_openai_robofac_client_includes_temperature_when_set() -> None:
    client = OpenAIRoboFACQAClient(
        api_key="test-key",
        model="test-model",
        temperature=0.2,
    )

    request_body = client._build_request_body(prompt="answer", frames=[])

    assert request_body["model"] == "test-model"
    assert request_body["temperature"] == 0.2


def test_openai_robofac_client_omits_temperature_by_default() -> None:
    client = OpenAIRoboFACQAClient(api_key="test-key", model="test-model")

    request_body = client._build_request_body(prompt="answer", frames=[])

    assert "temperature" not in request_body


def test_openai_robofac_client_can_omit_frame_metadata_and_set_top_p(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image-data")
    frame = CameraFrameRef(
        camera_name="rgb",
        timestamp_sec=1.25,
        image_path=image_path,
        frame_index=12,
    )
    client = OpenAIRoboFACQAClient(
        api_key="test-key",
        model="test-model",
        top_p=1.0,
        include_frame_metadata=False,
    )

    request_body = client._build_request_body(prompt="answer", frames=[frame])
    content = request_body["input"][0]["content"]

    assert request_body["top_p"] == 1.0
    assert [part["type"] for part in content] == ["input_image", "input_text"]
    assert "timestamp" not in str(content)
    assert "frame_index" not in str(content)


def test_openai_robofac_client_sends_system_prompt_as_separate_message() -> None:
    client = OpenAIRoboFACQAClient(api_key="test-key", model="test-model")

    request_body = client._build_request_body(
        prompt="user instructions",
        frames=[],
        system_prompt="system instructions",
    )

    assert [message["role"] for message in request_body["input"]] == ["system", "user"]
    assert request_body["input"][0]["content"] == [
        {"type": "input_text", "text": "system instructions"}
    ]
    assert request_body["input"][1]["content"][-1] == {
        "type": "input_text",
        "text": "user instructions",
    }


def test_openai_robofac_client_can_label_view_without_timing_metadata(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image-data")
    frame = CameraFrameRef(
        camera_name="head_rgb",
        timestamp_sec=1.25,
        image_path=image_path,
        frame_index=12,
    )
    client = OpenAIRoboFACQAClient(
        api_key="test-key",
        model="test-model",
        include_frame_timing_metadata=False,
    )

    content = client._build_request_body(prompt="answer", frames=[frame])["input"][0]["content"]

    assert content[0] == {"type": "input_text", "text": "[View: head_rgb]"}
    assert "timestamp" not in str(content)
    assert "frame_index" not in str(content)
