"""Tests for ViFailBack exact-frame localization evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathologist.run_vifailback_localization_eval import build_arg_parser
from pathologist.vifailback.evaluation import (
    _exact_frame_index,
    _select_random_keyframes,
    _vifailback_config_override,
    aggregate_vifailback_localization,
    discover_vifailback_failure_episodes,
    evaluate_exact_frame_predictions,
)
from pathologist.schemas import CameraFrameRef


def test_exact_frame_scoring_preserves_signed_and_absolute_error() -> None:
    result = evaluate_exact_frame_predictions(
        ground_truth_frames=[100],
        predicted_frames=[93, 140],
        tolerance_frames=[0, 6, 7, 25],
    )

    assert result["nearest_absolute_frame_error"] == 7
    assert result["nearest_absolute_error_sec"] == pytest.approx(0.28)
    assert result["ground_truth_matches"] == [
        {
            "ground_truth_frame_index": 100,
            "nearest_predicted_frame_index": 93,
            "signed_frame_error": -7,
            "absolute_frame_error": 7,
        }
    ]
    assert result["ground_truth_hits_by_tolerance_frames"] == {
        "0": 0,
        "6": 0,
        "7": 1,
        "25": 1,
    }


def test_discovery_uses_failure_avoid_keyframe_and_existing_hdf5(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ViFailback-Dataset"
    raw_task = root / "raw_data" / "task"
    raw_task.mkdir(parents=True)
    (raw_task / "episode_3.hdf5").touch()
    annotation_task = root / "annotated_data" / "task"
    annotation_task.mkdir(parents=True)
    annotation_path = annotation_task / "task_annotations.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "video": "annotated_data/task/videos/episode_3_cam_high.mp4",
                    "failure_detection": "yes",
                    "failure_type": "gripper 6d-pose",
                    "failure_subtask": "2",
                    "keyframe": [
                        {
                            "avoid_keyframe": (
                                "annotated_data/task/frames/"
                                "episode_3_cam_high/117_avoid_keyframe.jpg"
                            )
                        }
                    ],
                },
                {
                    "video": "annotated_data/task/videos/episode_4_cam_high.mp4",
                    "failure_detection": "no",
                    "keyframe": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    records = discover_vifailback_failure_episodes(root)

    assert len(records) == 1
    assert records[0].bundle_id == "task_episode_3"
    assert records[0].avoid_keyframe_indices == (117,)
    assert records[0].failure_type == "gripper 6d-pose"


def test_localization_aggregate_reports_exact_hit_and_frame_error() -> None:
    rows = [
        _row("uniform", target=100, predictions=[100, 200]),
        _row("uniform", target=100, predictions=[90, 200]),
    ]
    rows[0]["failure_type"] = "gripper state"
    rows[1]["failure_type"] = "task planning"

    aggregate = aggregate_vifailback_localization(
        rows,
        tolerance_frames=[0, 10],
        bootstrap_samples=0,
    )
    stats = aggregate["uniform"]

    assert stats["trajectory_count"] == 2
    assert stats["mean_nearest_absolute_frame_error"] == pytest.approx(5.0)
    assert stats["mean_nearest_absolute_error_sec"] == pytest.approx(0.2)
    assert stats["keyframe_recall_at_k_by_tolerance_frames"] == {
        "0": 0.5,
        "10": 1.0,
    }
    assert stats["point_precision_at_k_by_tolerance_frames"] == {
        "0": 0.25,
        "10": 0.5,
    }
    assert (
        aggregate["by_failure_type"]["gripper state"]["uniform"][
            "mean_nearest_absolute_frame_error"
        ]
        == 0.0
    )
    assert (
        aggregate["by_failure_type"]["task planning"]["uniform"][
            "mean_nearest_absolute_frame_error"
        ]
        == 10.0
    )


def test_vifailback_config_and_cli_fix_native_protocol_to_k2() -> None:
    config = _vifailback_config_override(top_k=2)
    args = build_arg_parser().parse_args([])

    assert args.top_k == 2
    assert args.smoothing_window == 11
    assert args.selectors == ["random", "kite_flow", "metric"]
    assert args.tolerance_frames == [0, 6, 12, 25]
    assert config["selector"]["surprise_flood"] == {
        "smoothing_window": 11,
        "ranking": "persistence",
        "max_proposals": 2,
    }
    assert config["robofac_qa"]["kite_flow"]["camera_name"] == "cam_high"
    assert config["metrics"]["visual_flow"]["camera_names"] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]


def test_exact_frame_index_rejects_off_grid_timestamp(tmp_path: Path) -> None:
    frame = CameraFrameRef(
        camera_name="cam_high",
        timestamp_sec=0.041,
        image_path=tmp_path / "frame.jpg",
        frame_index=1,
    )

    with pytest.raises(ValueError, match="timestamp/index mismatch"):
        _exact_frame_index(frame)


def test_random_baseline_is_seeded_without_replacement(tmp_path: Path) -> None:
    frames = [
        CameraFrameRef(
            camera_name="cam_high",
            timestamp_sec=index / 25.0,
            image_path=tmp_path / f"{index}.jpg",
            frame_index=index,
        )
        for index in range(20)
    ]

    first = _select_random_keyframes(frames, 2, seed="0:episode_1:random")
    repeated = _select_random_keyframes(frames, 2, seed="0:episode_1:random")
    other_episode = _select_random_keyframes(frames, 2, seed="0:episode_2:random")

    assert [frame.frame_index for frame in first] == [
        frame.frame_index for frame in repeated
    ]
    assert len({frame.frame_index for frame in first}) == 2
    assert [frame.frame_index for frame in first] != [
        frame.frame_index for frame in other_episode
    ]


def _row(selector: str, target: int, predictions: list[int]) -> dict:
    scoring = evaluate_exact_frame_predictions(
        ground_truth_frames=[target],
        predicted_frames=predictions,
        tolerance_frames=[0, 10],
    )
    return {
        "bundle_id": f"episode_{target}_{predictions[0]}",
        "selector_name": selector,
        "ground_truth_frame_indices": [target],
        "selected_prediction_count": len(predictions),
        **scoring,
    }
