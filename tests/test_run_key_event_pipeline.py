"""Smoke test for the end-to-end key event pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path

from pathologist.run_key_event_pipeline import run_key_event_pipeline
from pathologist.schemas import TaskContract


class FakeEventLabelClient:
    def label_visual_event(self, payload):
        return {
            "visual_observation": "The handle appears grasped in the sampled frames.",
            "phase": "grasp",
            "event_type": "visible_grasp",
            "objects_involved": ["handle"],
            "contact_or_grasp_state": "handle appears grasped",
            "progress_assessment": "progress",
            "visible_pathology_labels": [],
            "visual_evidence_ids": ["frame_000"],
            "uncertainty_notes": [],
        }

    def integrate_event_evidence(self, payload):
        return {
            "visual_observation": "The handle appears grasped in the sampled frames.",
            "phase": "grasp",
            "event_type": "grasp_with_metric_anomaly",
            "objects_involved": ["handle"],
            "contact_or_grasp_state": "handle appears grasped",
            "progress_assessment": "progress",
            "pathology_labels": ["metric_anomaly"],
            "metric_evidence_ids": ["metric_event_000"],
            "visual_evidence_ids": ["frame_000"],
            "explanation": "Visual evidence shows a grasp while metric cards mark the event for review.",
            "uncertainty_notes": [],
        }


def test_run_key_event_pipeline_writes_outputs(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "left_handle_failure"
    bundle_dir.mkdir()
    _write_jsonl(bundle_dir / "telemetry.jsonl", _telemetry_rows())
    for camera_name in ("head", "left", "right"):
        camera_dir = bundle_dir / camera_name
        camera_dir.mkdir()
        for index in range(20):
            (camera_dir / f"{index:06d}.jpg").write_bytes(b"jpeg")

    summary = run_key_event_pipeline(
        bundle_dir=bundle_dir,
        output_dir=tmp_path / "outputs",
        task_contract=TaskContract(
            task_name="Lift handle",
            task_description="Lift the left handle using the robot.",
            success_criteria="The handle is lifted stably.",
            failure_criteria="The handle is dropped or handled unstably.",
        ),
        point_event_delta_sec=0.1,
        frame_stride_within_window=10,
        max_events_to_label=2,
        event_label_client=FakeEventLabelClient(),
    )

    assert Path(summary["metric_profile_path"]).exists()
    assert Path(summary["timeline_debug_path"]).exists()
    assert Path(summary["proposals_path"]).exists()
    assert Path(summary["key_events_path"]).exists()
    assert Path(summary["visual_event_labels_path"]).exists()
    assert Path(summary["metric_grounding_cards_path"]).exists()

    proposals = json.loads(Path(summary["proposals_path"]).read_text(encoding="utf-8"))
    key_events = json.loads(Path(summary["key_events_path"]).read_text(encoding="utf-8"))
    visual_event_labels = json.loads(Path(summary["visual_event_labels_path"]).read_text(encoding="utf-8"))
    metric_grounding_cards = json.loads(Path(summary["metric_grounding_cards_path"]).read_text(encoding="utf-8"))
    assert proposals
    assert key_events
    assert visual_event_labels
    assert metric_grounding_cards
    assert key_events[0]["label"]["event_type"] == "grasp_with_metric_anomaly"
    assert "visual_label" in key_events[0]["label"]["metadata"]
    assert "metric_grounding_cards" in key_events[0]["label"]["metadata"]
    assert Path(summary["effective_config_path"]).exists()


def test_run_key_event_pipeline_uses_bundle_config(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "left_handle_failure"
    bundle_dir.mkdir()
    _write_jsonl(bundle_dir / "telemetry.jsonl", _telemetry_rows())
    (bundle_dir / "pathologist_config.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "max_events_per_metric": 1,
                    "gripper_chatter": {
                        "chatter_bin_steps": 10,
                        "min_chatter_transitions": 10
                    }
                },
                "selector": {
                    "point_event_delta_sec": 0.05,
                    "frame_stride_within_window": 1
                },
                "labeling": {
                    "max_events_to_label": 1
                }
            }
        ),
        encoding="utf-8",
    )

    summary = run_key_event_pipeline(
        bundle_dir=bundle_dir,
        output_dir=tmp_path / "outputs",
        task_contract=TaskContract(
            task_name="Lift handle",
            task_description="Lift the left handle using the robot.",
            success_criteria="The handle is lifted stably.",
            failure_criteria="The handle is dropped or handled unstably.",
        ),
        event_label_client=FakeEventLabelClient(),
    )

    effective_config = json.loads(Path(summary["effective_config_path"]).read_text(encoding="utf-8"))
    timeline_debug = json.loads(Path(summary["timeline_debug_path"]).read_text(encoding="utf-8"))
    key_events = json.loads(Path(summary["key_events_path"]).read_text(encoding="utf-8"))

    assert effective_config["metrics"]["gripper_chatter"]["min_chatter_transitions"] == 10
    assert timeline_debug["point_event_delta_sec"] == 0.05
    assert len(key_events) <= 1


def _telemetry_rows() -> list[dict]:
    timestamps = [index * 0.05 for index in range(20)]
    actions = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0, 1.0],
        [0.3, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.4, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 1.0],
        [0.6, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0, 1.0],
        [0.7, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0, 1.0],
        [0.8, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 1.0],
        [0.6, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 0.0],
    ]
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "timestamp_sec": timestamp,
                "action": actions[index],
                "proprioception": [index * 0.01, 0.0, 0.0],
                "proprioception_grippers": [actions[index][-1], actions[index][-1]],
                "wrist_poses": {
                    "handside.left": {"position": [0.5 + index * 0.01, 0.2, 1.0]},
                    "handside.right": {"position": [0.5 + index * 0.02, -0.2, 1.0]},
                },
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
