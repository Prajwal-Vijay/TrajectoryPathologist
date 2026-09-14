"""Tests for RoboFine-Bench prompt construction and question loading."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.qa.robofine import (
    VQA_SYSTEM_PROMPT,
    build_vqa_prompt,
    load_robofine_qa_items,
)


def test_duplicate_question_ids_get_unique_prompt_ids(tmp_path: Path) -> None:
    qa_path = tmp_path / "QAEvalSets.json"
    qa_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "status": "ok",
                    "qas": [
                        {
                            "question_id": "sample-1_q1",
                            "question": "Which object moved?",
                            "answer_type": "multiple_choice",
                            "options": ["cup", "plate"],
                            "answer": "cup",
                        },
                        {
                            "question_id": "sample-1_q1",
                            "question": "Did the gripper close?",
                            "answer_type": "yes_no",
                            "options": [],
                            "answer": "yes",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    eval_samples = {"sample-1": {"sample_id": "sample-1", "dataset": "BC-Z"}}

    items = load_robofine_qa_items(qa_path, eval_samples)

    assert [item.question_id for item in items] == [
        "sample-1_q1",
        "sample-1_q1__duplicate_2",
    ]
    assert [item.source_question_id for item in items] == ["sample-1_q1", "sample-1_q1"]

    system_prompt, user_prompt, shuffled = build_vqa_prompt(items)
    assert system_prompt == VQA_SYSTEM_PROMPT
    assert VQA_SYSTEM_PROMPT not in user_prompt
    assert "Selected frames:" not in user_prompt
    assert "[sample-1_q1]" in user_prompt
    assert "[sample-1_q1__duplicate_2]" in user_prompt
    assert set(shuffled) == {"sample-1_q1", "sample-1_q1__duplicate_2"}
