"""Tests for REASSEMBLE result viewer loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathologist.reassemble.viewer import (
    _gripper_transitions_from_series,
    build_camera_frame_index,
    flatten_peaks,
    frames_in_time_window,
    infer_full_timeline_offset_sec,
    load_ecod_aggregate,
    load_reassemble_telemetry_series_from_h5,
    load_viewer_dataset,
    nearest_frame,
    sample_frames_evenly,
)
from pathologist.reassemble.evaluation import _load_resume_state
from pathologist.run_reassemble_viewer_prep import build_arg_parser
from pathologist.run_reassemble_success_viewer_prep import (
    build_arg_parser as build_success_viewer_arg_parser,
)
from pathologist.reassemble.viewer_app import (
    _chronological_peaks,
    _format_timestamp,
    _peak_segment_context,
)


def test_viewer_loader_groups_selector_rows_by_bundle(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)

    dataset = load_viewer_dataset(output_dir)

    assert [bundle.bundle_id for bundle in dataset.bundles] == ["demo_high_1"]
    bundle = dataset.bundles[0]
    assert bundle.source_file_bundle_id == "demo"
    assert bundle.high_level_task_kind == "insert"
    assert set(bundle.rows_by_selector) == {"uniform", "kite_flow", "metric"}
    assert [peak["selector_name"] for peak in flatten_peaks(bundle)] == [
        "uniform",
        "kite_flow",
        "metric",
    ]


def test_viewer_loads_ecod_aggregate_and_missing_aggregate(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)

    aggregate = load_ecod_aggregate(output_dir, "demo_high_1")

    assert aggregate is not None
    assert aggregate["timestamps_sec"] == [0.0, 1.0, 2.0]
    assert aggregate["selected_local_maxima"][0]["peak_sec"] == 1.0
    assert load_ecod_aggregate(output_dir, "missing") is None


def test_viewer_loads_raw_telemetry_series_for_high_level_window(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "demo.h5"
    with h5py.File(h5_path, "w") as handle:
        robot_state = handle.create_group("robot_state")
        timestamps = handle.create_group("timestamps")
        robot_state.create_dataset(
            "measured_force",
            data=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
        )
        timestamps.create_dataset("measured_force", data=[10.0, 11.0, 12.0, 13.0])

    streams = load_reassemble_telemetry_series_from_h5(
        h5_path=h5_path,
        start_sec=1.0,
        duration_sec=2.0,
        max_samples=100,
    )

    assert len(streams) == 1
    stream = streams[0]
    assert stream.name == "measured_force"
    assert stream.timestamps_sec == [0.0, 1.0, 2.0]
    assert stream.values_by_component["x"] == [1.0, 0.0, 0.0]
    assert stream.values_by_component["y"] == [0.0, 2.0, 0.0]
    assert stream.values_by_component["magnitude"] == [1.0, 2.0, 3.0]


def test_viewer_frame_index_selects_nearest_frame_per_camera(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)

    frame_index = build_camera_frame_index(output_dir, "demo")
    match = nearest_frame(frame_index, "hama2", 101.4)

    assert sorted(frame_index) == ["hama1", "hama2", "hand"]
    assert match is not None
    assert match.frame.image_path.name == "000001_t0101.000.jpg"
    assert match.delta_sec == -0.4000000000000057


def test_viewer_filters_and_samples_relevant_high_level_frames(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)
    frame_index = build_camera_frame_index(output_dir, "demo")

    frames = frames_in_time_window(frame_index, "hand", 100.2, 102.0)
    sampled = sample_frames_evenly(frames, 2)

    assert [frame.image_path.name for frame in frames] == [
        "000001_t0101.000.jpg",
        "000002_t0102.000.jpg",
    ]
    assert [frame.image_path.name for frame in sampled] == [
        "000001_t0101.000.jpg",
        "000002_t0102.000.jpg",
    ]


def test_rapid_review_orders_events_by_trajectory_time(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)
    bundle = load_viewer_dataset(output_dir).bundles[0]

    peaks = _chronological_peaks(bundle, ["metric", "kite_flow", "uniform"])

    assert [peak["selector_name"] for peak in peaks] == ["uniform", "metric", "kite_flow"]
    assert [_format_timestamp(float(peak["peak_sec"])) for peak in peaks] == [
        "0:00.00",
        "0:01.00",
        "0:02.00",
    ]


def test_rapid_review_labels_peaks_inside_failed_segments(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)
    bundle = load_viewer_dataset(output_dir).bundles[0]

    assert _peak_segment_context(bundle, 0.75) == ("Align", True)
    assert _peak_segment_context(bundle, 1.5) == ("Candidate key event", False)


def test_viewer_infers_full_timeline_offset_from_selected_frame_paths(tmp_path: Path) -> None:
    output_dir = _write_viewer_fixture(tmp_path)
    bundle = load_viewer_dataset(output_dir).bundles[0]

    assert infer_full_timeline_offset_sec(bundle) == 100.0


def test_viewer_missing_frame_cache_is_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    assert build_camera_frame_index(output_dir, "missing") == {}


def test_viewer_gripper_transition_marks_label_open_and_closed() -> None:
    marks = _gripper_transitions_from_series(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [0.1, 0.2, 0.8, 0.9, 0.3],
        threshold=0.5,
        hysteresis=0.05,
    )

    assert [(mark.timestamp_sec, mark.label, mark.previous_state, mark.state) for mark in marks] == [
        (1.0, "O", "closed", "open"),
        (2.0, "C", "open", "closed"),
    ]


def test_viewer_gripper_transition_marks_ignore_threshold_jitter() -> None:
    marks = _gripper_transitions_from_series(
        [0.0, 0.5, 1.0, 1.5],
        [0.49, 0.51, 0.495, 0.505],
        threshold=0.5,
        hysteresis=0.03,
    )

    assert marks == []


def test_reassemble_viewer_prep_defaults_to_all_viewer_assets() -> None:
    args = build_arg_parser().parse_args(["--data-root", "REASSEMBLE", "--resume"])

    assert args.output_dir == "REASSEMBLE__outputs_persistence"
    assert args.metric_ranking == "persistence"
    assert args.evaluation_unit == "failed_high_level_action"
    assert args.selectors == ["uniform", "kite_flow", "metric"]
    assert args.resume is True


def test_reassemble_success_viewer_prep_defaults_to_successful_demos() -> None:
    args = build_success_viewer_arg_parser().parse_args(["--data-root", "REASSEMBLE"])

    assert args.output_dir == "REASSEMBLE_Successful"
    assert args.max_demos == 100
    assert args.metric_ranking == "persistence"
    assert args.selectors == ["uniform", "kite_flow", "metric"]


def test_reassemble_resume_state_keeps_only_complete_checkpointed_bundles(tmp_path: Path) -> None:
    manifest_path = tmp_path / "reassemble_manifest.json"
    eval_path = tmp_path / "reassemble_eval.jsonl"
    alignment_path = tmp_path / "reassemble_alignment_debug.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "bundle_id": "complete_high_1",
                    "h5_path": "/tmp/complete.h5",
                    "included": True,
                    "failed_low_level_coverage": {
                        "high_level_duration_sec": 2.0,
                        "failed_low_level_duration_sec": 1.0,
                        "failed_low_level_coverage_fraction": 0.5,
                    },
                },
                {
                    "bundle_id": "partial_high_1",
                    "h5_path": "/tmp/partial.h5",
                    "included": True,
                },
                {
                    "bundle_id": "errored",
                    "h5_path": "/tmp/errored.h5",
                    "included": False,
                    "skip_reason": "error",
                    "error": "No space left on device",
                },
                {
                    "bundle_id": "skipped",
                    "h5_path": "/tmp/skipped.h5",
                    "included": False,
                    "skip_reason": "no_failed_high_level_actions_with_failed_low_level_segments",
                },
            ]
        ),
        encoding="utf-8",
    )
    eval_rows = [
        {"bundle_id": "complete_high_1", "selector_name": "uniform", "recall_by_level": {}},
        {"bundle_id": "complete_high_1", "selector_name": "kite_flow", "recall_by_level": {}},
        {"bundle_id": "complete_high_1", "selector_name": "metric", "recall_by_level": {}},
        {"bundle_id": "partial_high_1", "selector_name": "uniform", "recall_by_level": {}},
    ]
    eval_path.write_text("\n".join(json.dumps(row) for row in eval_rows), encoding="utf-8")
    alignment_path.write_text(
        json.dumps(
            [
                {"bundle_id": "complete_high_1", "h5_path": "/tmp/complete.h5"},
                {"bundle_id": "partial_high_1", "h5_path": "/tmp/partial.h5"},
                {"bundle_id": "skipped", "h5_path": "/tmp/skipped.h5"},
            ]
        ),
        encoding="utf-8",
    )

    state = _load_resume_state(
        manifest_path=manifest_path,
        eval_path=eval_path,
        alignment_path=alignment_path,
        selectors=["uniform", "kite_flow", "metric"],
    )

    assert state["completed_demo_bundle_ids"] == {"complete_high_1"}
    assert state["completed_skipped_h5_paths"] == {"/tmp/skipped.h5"}
    assert [entry["bundle_id"] for entry in state["manifest"]] == ["complete_high_1", "skipped"]
    assert [row["bundle_id"] for row in state["eval_rows"]] == [
        "complete_high_1",
        "complete_high_1",
        "complete_high_1",
    ]
    assert state["rewrite_eval_rows"] is True


def _write_viewer_fixture(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    image_root = output_dir / "frame_cache" / "demo" / "images"
    for camera_name in ("hand", "hama1", "hama2"):
        camera_dir = image_root / camera_name
        camera_dir.mkdir(parents=True)
        for index, timestamp in enumerate((100.0, 101.0, 102.0)):
            (camera_dir / f"{index:06d}_t{timestamp:08.3f}.jpg").write_bytes(b"fake-jpg")

    manifest = [
        {
            "h5_path": "/tmp/demo.h5",
            "bundle_id": "demo_high_1",
            "source_file_bundle_id": "demo",
            "duration_sec": 2.0,
            "included": True,
            "high_level_task_kind": "insert",
            "high_level_segment": {"text": "Insert object", "start_sec": 0.0, "end_sec": 2.0},
            "failed_low_level_coverage": {
                "failed_low_level_coverage_fraction": 0.25,
            },
        }
    ]
    (output_dir / "reassemble_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = [
        _eval_row(
            "uniform",
            [
                {
                    "event_id": "uniform_frame_001",
                    "timeline_rank": 1,
                    "peak_sec": 0.0,
                    "image_path": str(image_root / "hama1" / "000000_t0100.000.jpg"),
                    "camera_name": "hama1",
                    "frame_index": 0,
                }
            ],
        ),
        _eval_row(
            "kite_flow",
            [
                {
                    "event_id": "kite_flow_frame_001",
                    "timeline_rank": 1,
                    "peak_sec": 2.0,
                    "image_path": str(image_root / "hama1" / "000002_t0102.000.jpg"),
                    "camera_name": "hama1",
                    "frame_index": 2,
                }
            ],
        ),
        _eval_row(
            "metric",
            [
                {
                    "event_id": "metric_001",
                    "timeline_rank": 1,
                    "peak_sec": 1.0,
                    "persistence": 5.0,
                    "area_above_base": 2.0,
                    "width_sec": 0.2,
                }
            ],
        ),
    ]
    (output_dir / "reassemble_eval.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    aggregate_dir = output_dir / "ecod_aggregates"
    aggregate_dir.mkdir()
    aggregate = {
        "bundle_id": "demo_high_1",
        "available": True,
        "timestamps_sec": [0.0, 1.0, 2.0],
        "raw_scores": [0.0, 2.0, 0.0],
        "smoothed_scores": [0.0, 3.0, 0.0],
        "local_maxima": [{"peak_sec": 1.0, "selected": True}],
        "selected_local_maxima": [{"peak_sec": 1.0}],
    }
    (aggregate_dir / "demo_high_1.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return output_dir


def _eval_row(selector_name: str, peaks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "bundle_id": "demo_high_1",
        "h5_path": "/tmp/demo.h5",
        "selector_name": selector_name,
        "high_level_task_text": "Insert object",
        "high_level_task_kind": "insert",
        "selected_peaks": peaks,
        "failed_segments": {
            "high": [],
            "low": [
                {
                    "segment_id": "1/0",
                    "level": "low",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "success": False,
                    "text": "Align",
                }
            ],
            "combined": [],
        },
        "recall_by_level": {},
    }
