from pathlib import Path

import pytest

from VL_implemenation.abc_data import semantic_event_type
from VL_implemenation.evaluation import evaluate_localizations, score_row
from VL_implemenation.mosaic import (
    build_mosaic_timeline,
    video_sec_to_trajectory_sec,
)
from VL_implemenation.qwen_localizer import parse_localization_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Insert the plug into the socket.", "insertion"),
        ("Press the plug into the socket.", "insertion"),
        ("Remove the plug from the socket.", "removal"),
        ("Take out the item from the bag.", "removal"),
        ("Pick up the plug.", None),
        ("mistake", None),
    ],
)
def test_semantic_event_type(text: str, expected: str | None) -> None:
    assert semantic_event_type(text) == expected


def test_parser_strips_fences_clips_and_keeps_only_first_event() -> None:
    parsed = parse_localization_answer(
        """```json
{"events":[
  {"type":"removal","start_sec":12,"end_sec":8},
  {"type":"insertion","start_sec":15,"end_sec":16}
]}
```""",
        video_duration_sec=10.0,
    )

    assert parsed["parse_status"] == "valid_event"
    assert parsed["event"] == {
        "type": "removal",
        "start_sec": 8.0,
        "end_sec": 10.0,
        "midpoint_sec": 9.0,
    }
    assert parsed["returned_event_count"] == 2
    assert parsed["extra_events_ignored"] == 1


def test_unparseable_and_valid_empty_are_distinct() -> None:
    invalid = parse_localization_answer("not JSON", video_duration_sec=10.0)
    empty = parse_localization_answer('{"events":[]}', video_duration_sec=10.0)

    assert invalid["parse_status"] == "unparseable_json"
    assert empty["parse_status"] == "valid_no_event"


def test_mosaic_sidecar_maps_video_to_irregular_trajectory_clock(
    tmp_path: Path,
) -> None:
    timelines = {
        "top": [(0.1, tmp_path / "t0.jpg"), (0.31, tmp_path / "t1.jpg")],
        "left_wrist": [(0.09, tmp_path / "l0.jpg"), (0.32, tmp_path / "l1.jpg")],
        "right_wrist": [(0.11, tmp_path / "r0.jpg"), (0.30, tmp_path / "r1.jpg")],
    }

    records = build_mosaic_timeline(timelines, output_fps=5.0)

    assert records[0]["left_wrist_src_sec"] == pytest.approx(0.09)
    assert records[1]["right_wrist_src_sec"] == pytest.approx(0.30)
    assert video_sec_to_trajectory_sec(records, 0.1) == pytest.approx(0.205)


def test_half_open_type_aware_midpoint_scoring_and_no_event_policy() -> None:
    ground = [
        {
            "event_type": "insertion",
            "start_sec": 2.0,
            "end_sec": 4.0,
            "annotation_text": "Insert.",
        }
    ]
    hit = score_row(
        {
            "bundle_id": "hit",
            "status": "complete",
            "parse_status": "valid_event",
            "event": {
                "type": "insertion",
                "trajectory_midpoint_sec": 3.0,
            },
            "ground_intervals": ground,
        }
    )
    wrong_type = score_row(
        {
            "bundle_id": "wrong",
            "status": "complete",
            "parse_status": "valid_event",
            "event": {
                "type": "removal",
                "trajectory_midpoint_sec": 3.0,
            },
            "ground_intervals": ground,
        }
    )
    boundary = score_row(
        {
            "bundle_id": "boundary",
            "status": "complete",
            "parse_status": "valid_event",
            "event": {
                "type": "insertion",
                "trajectory_midpoint_sec": 4.0,
            },
            "ground_intervals": ground,
        }
    )
    parse_failure_without_ground = score_row(
        {
            "bundle_id": "failure",
            "status": "complete",
            "parse_status": "unparseable_json",
            "event": None,
            "ground_intervals": [],
        }
    )

    assert hit["type_aware_hit"]
    assert wrong_type["type_agnostic_hit"]
    assert not wrong_type["type_aware_hit"]
    assert not boundary["type_aware_hit"]
    assert not parse_failure_without_ground["true_no_event"]
    assert not parse_failure_without_ground["decision_correct"]


def test_aggregate_metrics() -> None:
    rows = [
        {
            "bundle_id": "event",
            "status": "complete",
            "parse_status": "valid_event",
            "event": {
                "type": "removal",
                "trajectory_midpoint_sec": 2.0,
            },
            "ground_intervals": [
                {
                    "event_type": "removal",
                    "start_sec": 1.0,
                    "end_sec": 3.0,
                }
            ],
        },
        {
            "bundle_id": "none",
            "status": "complete",
            "parse_status": "valid_no_event",
            "event": None,
            "ground_intervals": [],
        },
    ]

    summary, _ = evaluate_localizations(rows, bootstrap_samples=0, seed=0)

    assert summary["point_precision"] == 1.0
    assert summary["event_episode_hit_recall"] == 1.0
    assert summary["no_event_specificity"] == 1.0
    assert summary["all_episode_decision_accuracy"] == 1.0
