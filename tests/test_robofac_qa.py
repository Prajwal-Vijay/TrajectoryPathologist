"""Tests for RoboFAC QA parsing and prompt construction."""

from __future__ import annotations

from pathlib import Path

from pathologist.qa.robofac import (
    RoboFACQAItem,
    build_qa_prompt,
    extract_options_from_question,
    normalize_answer,
    parse_failure_detection_answer,
    parse_option_from_answer,
    score_failure_detection_answer,
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


def test_parse_failure_detection_answer_explicit_yes_no_uncertain() -> None:
    assert parse_failure_detection_answer("No.") == "No."
    assert parse_failure_detection_answer("Yes") == "Yes."
    assert parse_failure_detection_answer("Uncertain.") == "Uncertain."
    assert parse_failure_detection_answer("I cannot determine.") == "Uncertain."
    assert parse_failure_detection_answer("The task failed.") is None
    assert parse_failure_detection_answer("Yes or no") is None


def test_score_failure_detection_answer_is_strict() -> None:
    correct, parsed, score = score_failure_detection_answer(
        prediction="No, the task did not succeed.",
        reference="No.",
    )

    assert correct is True
    assert parsed == "No."
    assert score == 1.0

    correct, parsed, score = score_failure_detection_answer(
        prediction="The task failed.",
        reference="No.",
    )

    assert correct is False
    assert parsed is None
    assert score == 0.0


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


def test_build_failure_detection_prompt_forces_exact_option_string() -> None:
    item = RoboFACQAItem(
        video_id="video-1",
        video_rel_path="Task/view/video-1.mp4",
        task="PickCube",
        question_type="Failure detection",
        question="<image>\nWas the task successfully completed?",
        reference_answer="No.",
    )
    frames = [
        CameraFrameRef(
            camera_name="rgb",
            timestamp_sec=0.0,
            image_path=Path("/tmp/frame.jpg"),
            frame_index=0,
        )
    ]

    prompt = build_qa_prompt(item, frames)

    assert "- Yes." in prompt
    assert "- No." in prompt
    assert "- Uncertain." in prompt
    assert "Return exactly the option string and nothing else." in prompt
    assert "Do not include reasoning" in prompt
