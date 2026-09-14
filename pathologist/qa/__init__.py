"""RoboFAC QA evaluation helpers."""

from pathologist.qa.robofac import (
    RoboFACPrediction,
    RoboFACQAItem,
    build_qa_prompt,
    extract_options_from_question,
    load_robofac_qa_items,
    normalize_answer,
    parse_failure_detection_answer,
    parse_option_from_answer,
    score_failure_detection_answer,
    score_mcq_answer,
)
from pathologist.qa.robofac_client import OpenAIRoboFACQAClient
from pathologist.qa.robofac_reasoning import (
    SUPPORTED_REASONING_QUESTION_TYPES,
    OpenAIRoboFACJudgeClient,
    build_reasoning_prompt,
    build_robofac_judge_prompt,
    parse_robofac_judge_response,
)
from pathologist.qa.robofac_selectors import select_keyframes
from pathologist.qa.robofine import (
    RoboFinePrediction,
    aggregate_robofine_stats,
    build_vqa_prompt,
    load_robofine_eval_samples,
    load_robofine_qa_items,
    parse_vqa_answers,
    score_robofine_answer,
)

__all__ = [
    "OpenAIRoboFACQAClient",
    "OpenAIRoboFACJudgeClient",
    "RoboFACPrediction",
    "RoboFACQAItem",
    "RoboFinePrediction",
    "SELECTOR_ALIASES",
    "SELECTOR_NAMES",
    "aggregate_event_stats",
    "aggregate_robofine_stats",
    "build_event_classification_prompt",
    "build_qa_prompt",
    "build_reasoning_prompt",
    "build_robofac_judge_prompt",
    "build_vqa_prompt",
    "extract_options_from_question",
    "load_robofac_qa_items",
    "nearest_event",
    "parse_event_annotations",
    "parse_event_prediction",
    "resolve_selector_names",
    "load_robofine_eval_samples",
    "load_robofine_qa_items",
    "normalize_answer",
    "parse_failure_detection_answer",
    "parse_option_from_answer",
    "parse_robofac_judge_response",
    "parse_vqa_answers",
    "score_failure_detection_answer",
    "score_mcq_answer",
    "score_robofine_answer",
    "select_keyframes",
    "SUPPORTED_REASONING_QUESTION_TYPES",
]
