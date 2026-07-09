"""RoboFAC QA evaluation helpers."""

from pathologist.qa.robofac import (
    RoboFACPrediction,
    RoboFACQAItem,
    build_qa_prompt,
    extract_options_from_question,
    load_robofac_qa_items,
    normalize_answer,
    parse_option_from_answer,
    score_mcq_answer,
)
from pathologist.qa.robofac_client import OpenAIRoboFACQAClient
from pathologist.qa.robofac_selectors import select_keyframes

__all__ = [
    "OpenAIRoboFACQAClient",
    "RoboFACPrediction",
    "RoboFACQAItem",
    "build_qa_prompt",
    "extract_options_from_question",
    "load_robofac_qa_items",
    "normalize_answer",
    "parse_option_from_answer",
    "score_mcq_answer",
    "select_keyframes",
]
