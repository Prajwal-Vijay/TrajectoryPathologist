from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from pathologist.adapters.robofail_bundle import (
    RoboFailRealBundleAdapter,
    RoboFailSimBundleAdapter,
    load_real_world_task_map,
)
from pathologist.robofail.evaluation import parse_failure_steps, run_robofail_eval_pipeline
from pathologist.run_robofail_eval_pipeline import build_arg_parser
from pathologist.schemas import BundleFormat


def test_parse_failure_steps_accepts_strings_numbers_and_lists() -> None:
    assert parse_failure_steps("00:44") == [44.0]
    assert parse_failure_steps("2:41") == [161.0]
    assert parse_failure_steps("1:02:03") == [3723.0]
    assert parse_failure_steps(12) == [12.0]
    assert parse_failure_steps(["00:01", "2:00", 3]) == [1.0, 120.0, 3.0]
    assert parse_failure_steps("00:01, 00:03") == [1.0, 3.0]


def test_parse_failure_steps_rejects_bad_timestamp() -> None:
    with pytest.raises(ValueError):
        parse_failure_steps("not-a-time")


def test_sim_adapter_loads_task_frames_and_action_labels(tmp_path: Path) -> None:
    episode_dir = _write_sim_episode(tmp_path, gt_failure_step="00:02", frame_steps=[1, 2, 3])
    (episode_dir / "interact_actions.pickle").write_bytes(pickle.dumps({2: "Pick"}))

    bundle = RoboFailSimBundleAdapter().load(episode_dir)

    assert bundle.bundle_id == "task-1"
    assert bundle.source_format == BundleFormat.ROBOFAIL_SIM
    assert [frame.timestamp_sec for frame in bundle.camera_frames] == [1.0, 2.0, 3.0]
    assert bundle.raw_metadata["gt_failure_step"] == "00:02"
    assert [stream.name for stream in bundle.streams] == ["action", "action_labels"]
    assert bundle.streams[0].records[0]["action"] == [0.0]
    assert bundle.streams[1].records[0]["action_label"] == "Pick"


def test_real_task_metadata_maps_general_folder_name(tmp_path: Path) -> None:
    metadata_path = tmp_path / "tasks_real_world.json"
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-a",
                    "general_folder_name": "putAppleBowl1",
                    "gt_failure_step": "00:12",
                }
            ]
        ),
        encoding="utf-8",
    )

    task_map = load_real_world_task_map(metadata_path)

    assert set(task_map) == {"putAppleBowl1"}
    assert task_map["putAppleBowl1"].task_id == "task-a"
    assert task_map["putAppleBowl1"].payload["gt_failure_step"] == "00:12"


def test_real_task_metadata_accepts_task_name_keyed_dict(tmp_path: Path) -> None:
    metadata_path = tmp_path / "tasks_real_world.json"
    metadata_path.write_text(
        json.dumps(
            {
                "Task 1": {
                    "general_folder_name": "putAppleBowl1",
                    "gt_failure_step": "00:12",
                },
                "Task 2": {
                    "general_folder_name": "boilWater1",
                    "gt_failure_step": ["01:58", "02:22"],
                },
            }
        ),
        encoding="utf-8",
    )

    task_map = load_real_world_task_map(metadata_path)

    assert set(task_map) == {"putAppleBowl1", "boilWater1"}
    assert task_map["putAppleBowl1"].task_id == "Task 1"
    assert task_map["boilWater1"].payload["gt_failure_step"] == ["01:58", "02:22"]


@pytest.mark.parametrize("split", ["sim", "real"])
def test_cli_accepts_sim_and_real_splits(split: str) -> None:
    args = build_arg_parser().parse_args(["--dataset-root", "Robofail", "--split", split])
    assert args.split == split
    assert args.plot is False


def test_cli_accepts_plot_flag() -> None:
    args = build_arg_parser().parse_args(
        ["--dataset-root", "Robofail", "--split", "sim", "--plot"]
    )
    assert args.plot is True


def test_real_adapter_requires_zarr_for_real_load(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")

    metadata_path = tmp_path / "tasks_real_world.json"
    metadata_path.write_text(
        json.dumps([{"general_folder_name": "episode1", "gt_failure_step": "00:01"}]),
        encoding="utf-8",
    )
    episode_dir = tmp_path / "episode1"
    (episode_dir / "replay_buffer.zarr").mkdir(parents=True)

    with pytest.raises(Exception):
        RoboFailRealBundleAdapter().load(episode_dir, metadata_path, extract_frames=False)


def test_robofail_eval_pipeline_scores_multivalue_gt_with_uniform_selector(tmp_path: Path) -> None:
    dataset_root = tmp_path / "Robofail"
    _write_sim_episode(
        dataset_root / "sim_data" / "heatPotato",
        gt_failure_step=["00:05", "00:15"],
        frame_steps=[0, 10, 20],
    )

    output_dir = tmp_path / "outputs"
    summary = run_robofail_eval_pipeline(
        dataset_root=dataset_root,
        split="sim",
        output_dir=output_dir,
        top_k=2,
        selectors=["uniform"],
    )

    stats = summary["stats"]["uniform"]
    assert stats["trajectory_count"] == 1
    assert stats["gt_failure_timestamp_count"] == 2
    assert stats["sum_abs_error_sec"] == 10.0
    assert stats["mae_sec"] == 5.0
    assert stats["topk_expected_peak_count"] == 2
    assert stats["topk_selected_peak_count"] == 2
    assert stats["topk_missing_peak_count"] == 0
    assert stats["topk_sum_abs_error_sec"] == 10.0
    assert stats["topk_mean_abs_error_sec"] == 5.0
    assert stats["topk_median_abs_error_sec"] == 5.0
    assert stats["topk_duration_threshold_peak_count"] == 2
    assert stats["topk_within_5pct_duration_count"] == 0
    assert stats["topk_within_5pct_duration_fraction"] == 0.0
    assert stats["topk_within_10pct_duration_count"] == 0
    assert stats["topk_within_10pct_duration_fraction"] == 0.0

    rows = [
        json.loads(line)
        for line in (output_dir / "robofail_eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["gt_failure_steps_sec"] == [5.0, 15.0]
    assert rows[0]["selected_peaks"][0]["peak_sec"] == 0.0
    assert rows[0]["selected_peaks"][1]["peak_sec"] == 20.0
    assert rows[0]["duration_sec"] == 20.0
    assert rows[0]["per_selected_peak_nearest_gt"] == [
        {"peak_sec": 0.0, "nearest_gt_sec": 5.0, "error_sec": 5.0},
        {"peak_sec": 20.0, "nearest_gt_sec": 15.0, "error_sec": 5.0},
    ]
    assert rows[0]["topk_sum_abs_error_sec"] == 10.0
    assert rows[0]["topk_mean_abs_error_sec"] == 5.0
    assert rows[0]["topk_within_5pct_duration_threshold_sec"] == 1.0
    assert rows[0]["topk_within_10pct_duration_threshold_sec"] == 2.0
    assert rows[0]["topk_within_5pct_duration_fraction"] == 0.0
    assert rows[0]["topk_within_10pct_duration_fraction"] == 0.0


def _write_sim_episode(
    root: Path,
    gt_failure_step: object,
    frame_steps: list[int],
) -> Path:
    episode_dir = root / "task-1"
    image_dir = episode_dir / "ego_img"
    image_dir.mkdir(parents=True)
    (episode_dir / "task.json").write_text(
        json.dumps(
            {
                "task_name": "heatPotato",
                "gt_failure_step": gt_failure_step,
            }
        ),
        encoding="utf-8",
    )
    for step in frame_steps:
        (image_dir / f"img_step_{step}.png").write_bytes(b"fake image bytes")
    return episode_dir
