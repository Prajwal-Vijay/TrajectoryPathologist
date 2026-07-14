"""Tests for RoboFAC QA client request construction."""

from __future__ import annotations

from pathologist.qa.robofac_client import OpenAIRoboFACQAClient


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
