"""RoboFine-Bench VQA loading, prompting, and deterministic scoring."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VQA_SYSTEM_PROMPT = (
    "You are an expert robot manipulation video analyst. You will be shown frames from a "
    "robot manipulation video with one or more camera views. Each view is labeled "
    "(e.g., [View: head_rgb], [View: left_wrist_rgb]). Use information from ALL views to "
    "answer the questions. Answer based ONLY on what you observe in the video frames. "
    "Be precise and concise."
)

CAPABILITY_AXES = {
    "gnd": ("active_actor", "target_object", "initial_configuration"),
    "act": (
        "action_sequence",
        "contact_and_approach",
        "trajectory_and_orientation",
        "body_motion",
    ),
    "state": ("object_interaction", "final_configuration", "failure_and_recovery"),
}

AXIS_LABELS = {
    "gnd": "Entity & Scene Grounding",
    "act": "Action & Motion Understanding",
    "state": "Interaction & State Reasoning",
}

OPTION_LETTERS = string.ascii_uppercase


@dataclass(slots=True)
class RoboFineQAItem:
    """One RoboFine-Bench VQA question."""

    sample_id: str
    dataset: str
    question_id: str
    source_question_id: str
    question: str
    answer: str
    answer_type: str
    options: list[str] = field(default_factory=list)
    capability: str = ""
    mode: str = ""
    reference_text: str = ""
    instruction_raw: str | None = None


@dataclass(slots=True)
class ShuffledChoiceQuestion:
    """Multiple-choice question with option letters shuffled from the question id."""

    question_id: str
    question: str
    answer_type: str
    labeled_options: list[tuple[str, str]]
    letter_to_option: dict[str, str]
    correct_letter: str | None


@dataclass(slots=True)
class RoboFinePrediction:
    """One selector prediction for one RoboFine question."""

    sample_id: str
    dataset: str
    question_id: str
    selector_name: str
    capability: str
    answer_type: str
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
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "question_id": self.question_id,
            "selector_name": self.selector_name,
            "capability": self.capability,
            "answer_type": self.answer_type,
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


def load_robofine_eval_samples(eval_sets_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(eval_sets_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"EvalSets.json must contain a list: {eval_sets_path}")
    samples: dict[str, dict[str, Any]] = {}
    for entry in payload:
        if isinstance(entry, dict) and entry.get("sample_id"):
            samples[str(entry["sample_id"])] = entry
    return samples


def load_robofine_qa_items(
    qa_eval_sets_path: str | Path,
    eval_samples: dict[str, dict[str, Any]],
    allowed_sample_ids: set[str] | None = None,
) -> list[RoboFineQAItem]:
    """Load VQA items, keeping only samples present in EvalSets and the allow-list."""

    payload = json.loads(Path(qa_eval_sets_path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"QAEvalSets.json must contain a list: {qa_eval_sets_path}")
    items: list[RoboFineQAItem] = []
    question_id_occurrences: dict[tuple[str, str], int] = {}
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        sample_id = str(entry.get("sample_id") or "")
        if not sample_id:
            continue
        if allowed_sample_ids is not None and sample_id not in allowed_sample_ids:
            continue
        sample = eval_samples.get(sample_id)
        if sample is None:
            continue
        qas = entry.get("qas")
        if not isinstance(qas, list):
            continue
        for question in qas:
            if not isinstance(question, dict):
                continue
            source_question_id = str(question.get("question_id") or "")
            prompt = str(question.get("question") or "").strip()
            if not source_question_id or not prompt:
                continue
            occurrence_key = (sample_id, source_question_id)
            occurrence = question_id_occurrences.get(occurrence_key, 0) + 1
            question_id_occurrences[occurrence_key] = occurrence
            question_id = (
                source_question_id
                if occurrence == 1
                else f"{source_question_id}__duplicate_{occurrence}"
            )
            options = question.get("options") or []
            items.append(
                RoboFineQAItem(
                    sample_id=sample_id,
                    dataset=str(sample.get("dataset") or ""),
                    question_id=question_id,
                    source_question_id=source_question_id,
                    question=prompt,
                    answer=str(question.get("answer") or ""),
                    answer_type=str(question.get("answer_type") or "multiple_choice"),
                    options=[str(option) for option in options] if isinstance(options, list) else [],
                    capability=str(question.get("capability") or ""),
                    mode=str(question.get("mode") or ""),
                    reference_text=str(question.get("reference_text") or ""),
                    instruction_raw=_instruction_text(sample.get("instruction_raw")),
                )
            )
    return items


def shuffle_choice_question(item: RoboFineQAItem) -> ShuffledChoiceQuestion:
    """Assign option letters with a stable shuffle seeded by question_id."""

    options = list(item.options)
    rng = _stable_rng(item.source_question_id)
    rng.shuffle(options)
    labeled = [(OPTION_LETTERS[index], option) for index, option in enumerate(options)]
    letter_to_option = {letter: option for letter, option in labeled}
    correct_letter = None
    normalized_answer = _normalize_text(item.answer)
    for letter, option in labeled:
        if _normalize_text(option) == normalized_answer:
            correct_letter = letter
            break
    return ShuffledChoiceQuestion(
        question_id=item.question_id,
        question=item.question,
        answer_type=item.answer_type,
        labeled_options=labeled,
        letter_to_option=letter_to_option,
        correct_letter=correct_letter,
    )


def build_vqa_prompt(
    items: list[RoboFineQAItem],
) -> tuple[str, str, dict[str, ShuffledChoiceQuestion]]:
    """Build the exact Appendix A.3.4 system and user prompts for one sample."""

    shuffled: dict[str, ShuffledChoiceQuestion] = {}
    question_blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        choice = shuffle_choice_question(item)
        shuffled[item.question_id] = choice
        question_blocks.append(_format_question_block(index, item, choice))

    questions_block = "\n\n".join(question_blocks)
    user_prompt = (
        "Watch the video frames carefully and answer ALL of the following questions "
        "about this robot manipulation video.\n\n"
        f"{questions_block}\n\n"
        "You MUST return your answers as a JSON object. Use the FULL question_id shown "
        "in brackets [] as the key (NOT the short Q1/Q2 label):\n\n"
        '{"answers": {"<full_question_id>": "<your_answer>", ...}}\n\n'
        "Rules:\n"
        '-- For yes_no questions: answer EXACTLY "yes" or "no"\n'
        '-- For number questions: answer with JUST a number (e.g. "3")\n'
        '-- For multiple_choice questions: answer with JUST the option letter (e.g. "B")\n'
        "-- Do NOT include explanations in the JSON values --- only the short answer\n"
        "-- You MUST answer ALL questions"
    )
    return VQA_SYSTEM_PROMPT, user_prompt, shuffled


def parse_vqa_answers(raw_prediction: str) -> dict[str, str]:
    """Parse `{\"answers\": {question_id: value}}` from a VLM response."""

    payload = _load_json_object(raw_prediction)
    answers = payload.get("answers", payload)
    if not isinstance(answers, dict):
        return {}
    parsed: dict[str, str] = {}
    for key, value in answers.items():
        if value is None:
            continue
        parsed[str(key)] = str(value).strip()
    return parsed


def score_robofine_answer(
    item: RoboFineQAItem,
    raw_value: str | None,
    shuffled: ShuffledChoiceQuestion | None,
) -> tuple[bool | None, str | None, float | None]:
    """Score one parsed answer against the ground-truth label."""

    if raw_value is None or not str(raw_value).strip():
        return False, None, 0.0
    answer_type = item.answer_type.lower()
    if answer_type == "yes_no":
        parsed = _parse_yes_no(raw_value)
        correct = parsed is not None and parsed == _parse_yes_no(item.answer)
        return correct, parsed, 1.0 if correct else 0.0
    if answer_type == "number":
        parsed_number = _parse_number(raw_value)
        reference_number = _parse_number(item.answer)
        parsed = None if parsed_number is None else _format_number(parsed_number)
        correct = (
            parsed_number is not None
            and reference_number is not None
            and abs(parsed_number - reference_number) < 1e-6
        )
        return correct, parsed, 1.0 if correct else 0.0
    parsed_letter, parsed_option = _parse_multiple_choice(raw_value, shuffled)
    parsed = parsed_letter or parsed_option
    if shuffled is not None and parsed_letter is not None:
        predicted_option = shuffled.letter_to_option.get(parsed_letter, parsed_option)
    else:
        predicted_option = parsed_option
    correct = False
    if predicted_option is not None and _normalize_text(predicted_option) == _normalize_text(item.answer):
        correct = True
    elif parsed_letter is not None and shuffled is not None and parsed_letter == shuffled.correct_letter:
        correct = True
    return correct, parsed, 1.0 if correct else 0.0


def axis_for_capability(capability: str) -> str | None:
    for axis, capabilities in CAPABILITY_AXES.items():
        if capability in capabilities:
            return axis
    return None


def aggregate_robofine_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy overall, by capability, by reporting axis, dataset, and answer type."""

    running: dict[str, dict[str, dict[str, float | int]]] = {}
    for row in predictions:
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        selector = row.get("selector_name")
        if not isinstance(selector, str):
            continue
        selector_bucket = running.setdefault(selector, {})
        _add_score(selector_bucket, "overall", float(score))
        capability = row.get("capability")
        if isinstance(capability, str) and capability:
            _add_score(selector_bucket, f"capability:{capability}", float(score))
            axis = axis_for_capability(capability)
            if axis is not None:
                _add_score(selector_bucket, f"axis:{axis}", float(score))
        dataset = row.get("dataset")
        if isinstance(dataset, str) and dataset:
            _add_score(selector_bucket, f"dataset:{dataset}", float(score))
        answer_type = row.get("answer_type")
        if isinstance(answer_type, str) and answer_type:
            _add_score(selector_bucket, f"answer_type:{answer_type}", float(score))

    stats: dict[str, Any] = {}
    for selector, buckets in running.items():
        stats[selector] = {
            "overall": _finalize_bucket(buckets.get("overall")),
            "by_capability": {},
            "by_axis": {},
            "by_dataset": {},
            "by_answer_type": {},
        }
        for key, bucket in buckets.items():
            finalized = _finalize_bucket(bucket)
            if key.startswith("capability:"):
                stats[selector]["by_capability"][key.split(":", 1)[1]] = finalized
            elif key.startswith("axis:"):
                axis = key.split(":", 1)[1]
                stats[selector]["by_axis"][axis] = {
                    **finalized,
                    "label": AXIS_LABELS.get(axis, axis),
                }
            elif key.startswith("dataset:"):
                stats[selector]["by_dataset"][key.split(":", 1)[1]] = finalized
            elif key.startswith("answer_type:"):
                stats[selector]["by_answer_type"][key.split(":", 1)[1]] = finalized
    return stats


def _format_question_block(index: int, item: RoboFineQAItem, choice: ShuffledChoiceQuestion) -> str:
    header = f"[Q{index}] [{item.question_id}] ({item.answer_type}) {item.question}"
    if item.answer_type == "multiple_choice" and choice.labeled_options:
        option_lines = [f"  {letter}. {option}" for letter, option in choice.labeled_options]
        return header + "\n" + "\n".join(option_lines)
    if item.answer_type == "yes_no":
        return header + "\n  Answer yes or no."
    if item.answer_type == "number":
        return header + "\n  Answer with a number."
    return header


def _parse_multiple_choice(
    raw_value: str,
    shuffled: ShuffledChoiceQuestion | None,
) -> tuple[str | None, str | None]:
    stripped = raw_value.strip()
    match = re.match(r"^\s*([A-Za-z])[\).:\s]*", stripped)
    if match:
        letter = match.group(1).upper()
        option = shuffled.letter_to_option.get(letter) if shuffled is not None else None
        return letter, option
    normalized = _normalize_text(stripped)
    if shuffled is not None:
        for letter, option in shuffled.labeled_options:
            if _normalize_text(option) == normalized:
                return letter, option
    return None, stripped


def _parse_yes_no(value: str) -> str | None:
    normalized = _normalize_text(value)
    if normalized in {"yes", "y"}:
        return "yes"
    if normalized in {"no", "n"}:
        return "no"
    return None


def _parse_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    return float(match.group(0))


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def _load_json_object(raw_prediction: str) -> dict[str, Any]:
    text = raw_prediction.strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _stable_rng(question_id: str) -> random.Random:
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _instruction_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _add_score(buckets: dict[str, dict[str, float | int]], key: str, score: float) -> None:
    bucket = buckets.setdefault(key, {"count": 0, "sum": 0.0})
    bucket["count"] = int(bucket["count"]) + 1
    bucket["sum"] = float(bucket["sum"]) + score


def _finalize_bucket(bucket: dict[str, float | int] | None) -> dict[str, Any]:
    if not bucket:
        return {"count": 0, "accuracy": None}
    count = int(bucket["count"])
    score_sum = float(bucket["sum"])
    return {"count": count, "accuracy": score_sum / count if count else None}
