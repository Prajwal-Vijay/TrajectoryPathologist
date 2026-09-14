"""Prompts, judge parsing, and clients for open-ended RoboFAC evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pathologist.qa.robofac_client import OpenAIRoboFACQAClient


SUPPORTED_REASONING_QUESTION_TYPES = (
    "Failure explanation",
    "Task planning",
)
JUDGE_CRITERIA = ("correctness", "relevance", "completeness")


@dataclass(frozen=True, slots=True)
class RoboFACJudgeResult:
    """Validated RoboFAC appendix judge result."""

    criteria: dict[str, dict[str, Any]]
    average_score: float
    normalized_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": self.criteria,
            "average_score": self.average_score,
            "normalized_score": self.normalized_score,
        }


class RoboFACJudgeClient(Protocol):
    """Protocol for text-only RoboFAC answer judges."""

    def judge(self, prompt: str) -> str:
        """Return the raw judge response."""


class OpenAIRoboFACJudgeClient:
    """Text-only OpenAI judge configured independently from answer generation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
        retry_backoff_sec: float | None = None,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._client = OpenAIRoboFACQAClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            include_frame_metadata=False,
            max_output_tokens=1024,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )

    def judge(self, prompt: str) -> str:
        return self._client.answer(prompt, [])


class StaticRoboFACJudgeClient:
    """Deterministic judge client for tests and local smoke checks."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def judge(self, prompt: str) -> str:
        return self.response_text


def validate_reasoning_question_types(question_types: list[str] | tuple[str, ...]) -> list[str]:
    """Validate and deduplicate the only supported open-ended question types."""

    if not question_types:
        raise ValueError("At least one RoboFAC reasoning question type is required.")
    supported = set(SUPPORTED_REASONING_QUESTION_TYPES)
    unsupported = [question_type for question_type in question_types if question_type not in supported]
    if unsupported:
        raise ValueError(
            "Unsupported RoboFAC reasoning question type(s): "
            f"{unsupported}. Allowed values: {list(SUPPORTED_REASONING_QUESTION_TYPES)}."
        )
    return list(dict.fromkeys(question_types))


def build_reasoning_prompt(question: str) -> str:
    """Build the intentionally minimal selector-comparison prompt."""

    clean_question = question.replace("<image>", "").strip()
    return (
        "Look at the provided frames from the robot task execution, shown in "
        "chronological order, and answer the question.\n\n"
        f"Question: {clean_question}\n\n"
        "Answer:"
    )


def build_robofac_judge_prompt(question: str, reference: str, prediction: str) -> str:
    """Build the LLM-scoring prompt from Appendix E of the RoboFAC paper."""

    clean_question = question.replace("<image>", "").strip()
    return f"""You are an expert evaluator. Assess the quality of a model's response to the user's query.

Question: {clean_question}

Reference answer: {reference}

Model's response: {prediction}

Evaluate the model's response on the following criteria:
- correctness: factual accuracy and consistency with the reference answer.
- relevance: how well the model's response addresses the question.
- completeness: whether all key aspects of the reference answer are covered.

For each criterion, provide a score from 0 to 5 and a **brief** explanation, the score should be an integer. The score you give needs to be strict and demanding.

Output ONLY the JSON object in the following format:
{{
  "criteria": {{
    "correctness": {{"score": <0-5>, "explanation": <brief explanation>}},
    "relevance": {{"score": <0-5>, "explanation": <brief explanation>}},
    "completeness": {{"score": <0-5>, "explanation": <brief explanation>}}
  }}
}}"""


def parse_robofac_judge_response(response_text: str) -> RoboFACJudgeResult:
    """Parse and validate the judge JSON, then normalize its score to 100."""

    payload = _parse_json_object(response_text)
    criteria_payload = payload.get("criteria")
    if not isinstance(criteria_payload, dict):
        raise ValueError("RoboFAC judge response must contain a criteria object.")
    for criterion in JUDGE_CRITERIA:
        if not isinstance(criteria_payload.get(criterion), dict):
            raise ValueError(f"RoboFAC judge response is missing criterion: {criterion}.")

    criteria: dict[str, dict[str, Any]] = {}
    scores = []
    for criterion in JUDGE_CRITERIA:
        value = criteria_payload[criterion]
        score = value.get("score")
        explanation = value.get("explanation")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
            raise ValueError(f"Judge score for {criterion} must be an integer from 0 to 5.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Judge explanation for {criterion} must be a non-empty string.")
        criteria[criterion] = {
            "score": score,
            "explanation": explanation.strip(),
        }
        scores.append(score)

    average_score = sum(scores) / len(scores)
    return RoboFACJudgeResult(
        criteria=criteria,
        average_score=average_score,
        normalized_score=100.0 * sum(scores) / (5 * len(scores)),
    )


def _parse_json_object(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("RoboFAC judge response did not contain a JSON object.") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid RoboFAC judge JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("RoboFAC judge response must be a JSON object.")
    return payload
