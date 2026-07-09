"""Tests for RoboFAC QA parsing and prompt construction."""

from __future__ import annotations

from pathlib import Path

from pathologist.qa.robofac import (
    RoboFACQAItem,
    build_qa_prompt,
    extract_options_from_question,
    normalize_answer,
    parse_option_from_answer,
    score_mcq_answer,
)
from pathologist.schemas import CameraFrameRef


def test_extract_options_from_robofac_question() -> None:
    question = (
        "<image>\nThis is a video of a robotic arm. "
        "Choose one of the following options: ['Open the microwave door.', 'Grasp the spoon.']"
    )

    assert extract_options_from_question(question) == [
        "Open the microwave door.",
        "Grasp the spoon.",
    ]


def test_score_mcq_answer_uses_normalized_containment() -> None:
    correct, parsed, score = score_mcq_answer(
        prediction="The answer is: grasp the spoon.",
        reference="Grasp the spoon.",
        options=["Open the microwave door.", "Grasp the spoon."],
    )

    assert correct is True
    assert parsed == "Grasp the spoon."
    assert score == 1.0
    assert normalize_answer("The Grasp, Spoon.") == "grasp spoon"


def test_parse_lettered_option_answer() -> None:
    assert parse_option_from_answer(
        "(B)",
        ["Open the microwave door.", "Grasp the spoon."],
    ) == "Grasp the spoon."


def test_build_qa_prompt_has_shared_context_but_no_metric_values() -> None:
    item = RoboFACQAItem(
        video_id="video-1",
        video_rel_path="Task/view/video-1.mp4",
        task="MicrowaveTask",
        question_type="Failure locating",
        question="<image>\nWhere did the failure occur?",
        reference_answer="Grasp the spoon.",
    )
    frames = [
        CameraFrameRef(
            camera_name="rgb",
            timestamp_sec=1.0,
            image_path=Path("/tmp/frame.jpg"),
            frame_index=30,
        )
    ]

    prompt = build_qa_prompt(item, frames)

    assert "<image>" not in prompt
    assert "timestamp_sec=1.000" in prompt
    assert "Metric" not in prompt
    assert "threshold" not in prompt.lower()
    assert "BEV" not in prompt
