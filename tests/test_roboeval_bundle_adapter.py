"""Tests for RoboEval bundle ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.adapters import RoboEvalBundleAdapter
from pathologist.schemas import TaskContract


def test_roboeval_bundle_adapter_loads_streams_and_frames(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "left_handle_failure"
    bundle_dir.mkdir()
    _write_jsonl(
        bundle_dir / "telemetry.jsonl",
        [
            {
                "timestamp_sec": 0.0,
                "action": [0.0, 0.0, 0.0, 0.0],
                "proprioception": [0.0, 0.0, 0.0],
                "proprioception_grippers": [0.0, 0.0],
                "wrist_poses": {
                    "handside.left": {"position": [0.5, 0.2, 1.0]},
                    "handside.right": {"position": [0.5, -0.2, 1.0]},
                },
            },
            {
                "timestamp_sec": 0.1,
                "action": [1.0, 0.0, 0.0, 1.0],
                "proprioception": [0.1, 0.0, 0.0],
                "proprioception_grippers": [1.0, 1.0],
                "wrist_poses": {
                    "handside.left": {"position": [0.6, 0.2, 1.0]},
                    "handside.right": {"position": [0.6, -0.2, 1.0]},
                },
            },
        ],
    )
    for camera_name in ("head", "left", "right"):
        camera_dir = bundle_dir / camera_name
        camera_dir.mkdir()
        for index in range(2):
            (camera_dir / f"{index:06d}.jpg").write_bytes(b"jpeg")

    bundle = RoboEvalBundleAdapter().load(
        bundle_dir,
        task_contract=TaskContract(
            task_name="Lift handle",
            task_description="Lift the left handle.",
            success_criteria="Handle is lifted.",
            failure_criteria="Handle is not lifted.",
        ),
    )

    assert bundle.bundle_id == "left_handle_failure"
    assert {stream.name for stream in bundle.streams} == {"telemetry", "action", "proprioception", "gripper"}
    assert len(bundle.camera_frames) == 6
    assert "action_space" not in bundle.raw_metadata
    action_stream = next(stream for stream in bundle.streams if stream.name == "action")
    assert "action_low" not in action_stream.metadata
    assert "action_high" not in action_stream.metadata


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
