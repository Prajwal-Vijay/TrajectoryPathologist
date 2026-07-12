"""RoboFAC QA annotation, prompt, parsing, and scoring utilities."""

from __future__ import annotations

import ast
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathologist.schemas.trajectory_bundle import CameraFrameRef


DEFAULT_ROBOFAC_MCQ_TYPES = (
    "Failure detection",
    "Failure identification",
    "Failure locating",
)

FAILURE_DETECTION_OPTIONS = ("Yes.", "No.", "Uncertain.")


@dataclass(slots=True)
class RoboFACQAItem:
    """One RoboFAC question/answer item for a video."""

    video_id: str
    video_rel_path: str
    task: str
    question_type: str
    question: str
    reference_answer: str
    options: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RoboFACPrediction:
    """One selector/model prediction for a RoboFAC QA item."""

    video_id: str
    question_type: str
    selector_name: str
    question: str
    reference_answer: str
    raw_prediction: str
    parsed_prediction: str | None
    correct: bool | None
    score: float | None
    selected_frame_indices: list[int | None] = field(default_factory=list)
    selected_timestamps_sec: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "question_type": self.question_type,
            "selector_name": self.selector_name,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "raw_prediction": self.raw_prediction,
            "parsed_prediction": self.parsed_prediction,
            "correct": self.correct,
            "score": self.score,
            "selected_frame_indices": self.selected_frame_indices,
            "selected_timestamps_sec": self.selected_timestamps_sec,
            "metadata": self.metadata,
        }


def load_robofac_qa_items(
    annotation_path: str | Path,
    question_types: set[str] | None = None,
) -> list[RoboFACQAItem]:
    """Load RoboFAC annotation JSON into flat QA items."""

    allowed_types = question_types or set(DEFAULT_ROBOFAC_MCQ_TYPES)
    payload = json.loads(Path(annotation_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RoboFAC annotation JSON must contain an object: {annotation_path}")
    items: list[RoboFACQAItem] = []
    for video_id, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        annos = entry.get("annos")
        if not isinstance(annos, dict):
            continue
        for question_type, qa_turns in annos.items():
            if question_type not in allowed_types:
                continue
            question, reference = _question_and_reference(qa_turns)
            if not question or reference is None:
                continue
            items.append(
                RoboFACQAItem(
                    video_id=str(video_id),
                    video_rel_path=str(entry.get("video", "")),
                    task=str(entry.get("task", "")),
                    question_type=str(question_type),
                    question=question,
                    reference_answer=reference,
                    options=extract_options_from_question(question),
                )
            )
    return items


def build_qa_prompt(
    item: RoboFACQAItem,
    selected_frames: list[CameraFrameRef],
    include_shared_context: bool = True,
) -> str:
    """Build a selector-agnostic RoboFAC prompt with no metric values."""

    question = item.question.replace("<image>", "").strip()
    answer_options = response_options_for_item(item)
    answer_instruction = _strict_option_instruction(answer_options)
    if not include_shared_context:
        return f"{question}\n\n{answer_instruction}"
    frame_lines = [
        f"frame_{index}: timestamp_sec={frame.timestamp_sec:.3f}, original_frame_index={frame.frame_index}"
        for index, frame in enumerate(selected_frames)
    ]
    context = "\n".join(
        [
            "You are given selected RGB keyframes from a robot task execution.",
            "Use the frames in chronological order. Answer the RoboFAC question directly.",
            f"Task family: {item.task or 'unknown'}.",
            "Robot profile: single-arm Panda-style manipulator with parallel-jaw gripper.",
            "Selected frame timestamps:",
            *frame_lines,
        ]
    )
    return f"{question}\n\n{answer_instruction}\n\n[CONTEXT]\n{context}"


def extract_options_from_question(question_text: str) -> list[str]:
    """Extract MCQ options from RoboFAC question text."""

    match = re.search(r"\[(.*?)\]", question_text, re.DOTALL)
    if not match:
        return []
    literal = "[" + match.group(1) + "]"
    try:
        value = ast.literal_eval(literal)
    except Exception:
        value = None
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", literal)
    return [(left or right).strip() for left, right in quoted if (left or right).strip()]


def parse_option_from_answer(answer_text: str, options: list[str]) -> str | None:
    """Parse the selected option from an answer string."""

    if not answer_text:
        return None
    stripped = answer_text.strip()
    letter_match = re.search(r"(?:^|\b|\()([A-E])(?:\)|\.|\b)", stripped)
    if letter_match and options:
        index = ord(letter_match.group(1)) - ord("A")
        if 0 <= index < len(options):
            return options[index]
    normalized_prediction = normalize_answer(stripped)
    for option in options:
        normalized_option = normalize_answer(option)
        if normalized_prediction == normalized_option:
            return option
    for option in options:
        normalized_option = normalize_answer(option)
        if normalized_option and normalized_option in normalized_prediction:
            return option
    return stripped


def score_mcq_answer(prediction: str, reference: str, options: list[str] | None = None) -> tuple[bool, str | None, float]:
    """Score FD/FI/FL answers with KITE-style normalized containment."""

    parsed = parse_option_from_answer(prediction, options or [])
    normalized_prediction = normalize_answer(parsed or prediction)
    normalized_reference = normalize_answer(reference)
    correct = bool(
        normalized_prediction
        and normalized_reference
        and (
            normalized_prediction == normalized_reference
            or normalized_reference in normalized_prediction
            or normalized_prediction in normalized_reference
        )
    )
    return correct, parsed, 1.0 if correct else 0.0


def score_failure_detection_answer(prediction: str, reference: str) -> tuple[bool, str | None, float]:
    """Score failure-detection answers by explicitly parsing yes/no/uncertain."""

    parsed_prediction = parse_failure_detection_answer(prediction)
    parsed_reference = parse_failure_detection_answer(reference)
    correct = (
        parsed_prediction is not None
        and parsed_reference is not None
        and parsed_prediction == parsed_reference
    )
    return correct, parsed_prediction, 1.0 if correct else 0.0


def parse_failure_detection_answer(answer_text: str) -> str | None:
    """Parse failure-detection answers into Yes., No., or Uncertain."""

    if not answer_text:
        return None
    normalized = normalize_answer(answer_text)
    if not normalized:
        return None

    exact_map = {
        "yes": "Yes.",
        "no": "No.",
        "uncertain": "Uncertain.",
        "unsure": "Uncertain.",
        "unknown": "Uncertain.",
        "unclear": "Uncertain.",
    }
    if normalized in exact_map:
        return exact_map[normalized]

    categories: set[str] = set()
    if re.search(r"\byes\b", normalized):
        categories.add("Yes.")
    if re.search(r"\bno\b", normalized):
        categories.add("No.")
    if re.search(r"\b(uncertain|unsure|unknown|unclear)\b", normalized):
        categories.add("Uncertain.")
    if re.search(r"\b(can ?not|cant|cannot) (determine|tell|say)\b", normalized):
        categories.add("Uncertain.")

    if len(categories) == 1:
        return next(iter(categories))
    return None


def response_options_for_item(item: RoboFACQAItem) -> list[str]:
    """Return the exact response options the QA model should choose from."""

    if item.question_type == "Failure detection":
        return list(FAILURE_DETECTION_OPTIONS)
    return item.options


def normalize_answer(text: str) -> str:
    """Normalize text similarly to common QA/MCQ evaluation scripts."""

    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strict_option_instruction(options: list[str]) -> str:
    if not options:
        return (
            "Answer format: return only the final answer string. "
            "Do not include reasoning, explanation, labels, or extra text."
        )
    option_lines = "\n".join(f"- {option}" for option in options)
    return "\n".join(
        [
            "Answer format: choose exactly one option below.",
            option_lines,
            "Return exactly the option string and nothing else.",
            "Do not include reasoning, explanation, labels, punctuation changes, or extra text.",
        ]
    )


def aggregate_predictions(predictions: list[RoboFACPrediction]) -> dict[str, Any]:
    """Aggregate MCQ scores by selector and question type."""

    buckets: dict[str, dict[str, list[float]]] = {}
    for prediction in predictions:
        if prediction.score is None:
            continue
        selector_bucket = buckets.setdefault(prediction.selector_name, {})
        selector_bucket.setdefault(prediction.question_type, []).append(float(prediction.score))
    stats: dict[str, Any] = {}
    for selector, by_type in buckets.items():
        stats[selector] = {}
        all_scores: list[float] = []
        for question_type, scores in by_type.items():
            all_scores.extend(scores)
            stats[selector][question_type] = {
                "count": len(scores),
                "accuracy": sum(scores) / len(scores) if scores else None,
            }
        stats[selector]["overall"] = {
            "count": len(all_scores),
            "accuracy": sum(all_scores) / len(all_scores) if all_scores else None,
        }
    return stats


def _question_and_reference(qa_turns: Any) -> tuple[str | None, str | None]:
    if not isinstance(qa_turns, list) or len(qa_turns) < 2:
        return None, None
    question_turn = qa_turns[0]
    answer_turn = qa_turns[1]
    if not isinstance(question_turn, dict) or not isinstance(answer_turn, dict):
        return None, None
    question = question_turn.get("value")
    answer = answer_turn.get("value")
    return (
        str(question) if isinstance(question, str) else None,
        str(answer) if isinstance(answer, str) else None,
    )
