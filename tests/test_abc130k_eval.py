from pathlib import Path
from types import SimpleNamespace

import pytest

from pathologist.abc130k.evaluation import (
    ABC_PRISM_SOURCE_METRICS,
    _abc_config_override,
    _aggregate_rows,
    _contains,
    _evaluate_selector,
    _running_metrics_message,
)
from pathologist.adapters.abc130k_bundle import (
    _build_robot_streams,
    _segments_from_annotations,
)
from pathologist.run_abc130k_eval_pipeline import build_arg_parser
from pathologist.schemas import BundleFormat, CameraFrameRef, TimeWindow, TrajectoryBundle


def test_abc_annotations_define_half_open_mistake_intervals() -> None:
    segments = _segments_from_annotations(
        [
            (100_000_000, "pick object"),
            (2_000_000_000, "mistake"),
            (4_500_000_000, "recover"),
            (5_500_000_000, "reset both arms"),
        ],
        episode_origin_ns=0,
        episode_end_ns=6_000_000_000,
    )

    failure = segments[1]
    assert failure.failure is True
    assert failure.start_sec == pytest.approx(2.0)
    assert failure.end_sec == pytest.approx(4.5)
    assert _contains(failure.to_dict(), 2.0)
    assert _contains(failure.to_dict(), 4.499)
    assert not _contains(failure.to_dict(), 4.5)
    assert segments[-1].ignored is True
    assert segments[-1].to_dict()["success"] is None


def test_abc_flow_strides_and_robot_masking_are_configured_independently(
    tmp_path: Path,
) -> None:
    config = _abc_config_override(
        camera_names=["top", "left_wrist", "right_wrist"],
        top_k=10,
        metric_ranking="persistence",
        smoothing_window=3,
        visual_flow_frame_stride=1,
        kite_frame_stride=4,
        robot_mask_visual_flow=True,
        robot_mask_prompt="robot arm. robot gripper.",
        robot_mask_box_threshold=0.35,
        robot_mask_dilation_px=8,
        robot_mask_cache_dir=tmp_path / "robot_masks",
    )

    assert config["metrics"]["visual_flow"]["frame_stride"] == 1
    assert config["metrics"]["visual_flow"]["camera_names"] == [
        "top",
        "left_wrist",
        "right_wrist",
    ]
    assert config["metrics"]["visual_flow"]["resize"] == [128, 128]
    assert config["metrics"]["visual_flow"]["spatial_aggregation"] == "mean"
    assert config["surprise"]["source_metrics"] == list(ABC_PRISM_SOURCE_METRICS)
    assert config["robofac_qa"]["kite_flow"]["frame_stride"] == 4
    assert config["robofac_qa"]["kite_flow"]["camera_name"] == "top"
    assert config["metrics"]["visual_flow"]["robot_masking"]["enabled"] is True
    assert (
        config["metrics"]["visual_flow"]["robot_masking"]["box_threshold"]
        == pytest.approx(0.35)
    )
    assert config["metrics"]["visual_flow"]["robot_masking"]["dilation_px"] == 8
    assert config["metrics"]["visual_flow"]["robot_masking"]["cache_dir"] == str(
        tmp_path / "robot_masks"
    )
    assert config["metrics"]["visual_flow"]["robot_masking"]["chunk_size_frames"] == 20
    assert (
        config["metrics"]["visual_flow"]["robot_masking"]["duplicate_iou_threshold"]
        == pytest.approx(0.8)
    )


def test_abc_cli_accepts_independent_flow_strides_and_robot_masking() -> None:
    args = build_arg_parser().parse_args(
        [
            "--visual-flow-frame-stride",
            "1",
            "--kite-frame-stride",
            "3",
            "--robot-mask-visual-flow",
            "--robot-mask-box-threshold",
            "0.4",
            "--robot-mask-dilation-px",
            "6",
            "--robot-mask-cache-dir",
            "/tmp/shared-robot-masks",
        ]
    )

    assert args.visual_flow_frame_stride == 1
    assert args.kite_frame_stride == 3
    assert args.robot_mask_visual_flow is True
    assert args.robot_mask_box_threshold == pytest.approx(0.4)
    assert args.robot_mask_dilation_px == 6
    assert args.robot_mask_cache_dir == "/tmp/shared-robot-masks"


def test_abc_cli_accepts_multiple_cameras_and_defaults_to_all() -> None:
    parser = build_arg_parser()

    assert parser.parse_args([]).camera_names is None
    args = parser.parse_args(
        [
            "--camera",
            "top",
            "--camera",
            "left_wrist",
            "--camera",
            "right_wrist",
            "--frame-cache-dir",
            "/tmp/shared-abc-frames",
        ]
    )
    assert args.camera_names == ["top", "left_wrist", "right_wrist"]
    assert args.frame_cache_dir == "/tmp/shared-abc-frames"


def test_abc_cli_accepts_selector_specific_padding() -> None:
    args = build_arg_parser().parse_args(["--pad", "prism", "kite"])

    assert args.pad == ["prism", "kite"]


def test_abc_cli_accepts_both_ronar_protocols() -> None:
    args = build_arg_parser().parse_args(
        [
            "--selectors",
            "ronar",
            "ronar_k",
            "--ronar-sample-period-sec",
            "0.25",
            "--ronar-threshold",
            "60",
        ]
    )

    assert args.selectors == ["ronar", "ronar_k"]
    assert args.ronar_sample_period_sec == pytest.approx(0.25)
    assert args.ronar_threshold == pytest.approx(60.0)


def test_abc_running_metrics_message_reports_both_recalls() -> None:
    message = _running_metrics_message(
        {
            "metric": {
                "point_precision_at_k": 0.125,
                "failed_segment_recall_at_k": 0.4,
                "episode_recall_at_k": 0.6,
            }
        },
        selectors=["metric"],
        completed_count=5,
        total_count=10,
    )

    assert message == "  running [5/10] metric: P=12.5% SegR=40.0% TrajR=60.0%"


def test_abc_robot_topics_map_to_metric_streams(tmp_path: Path) -> None:
    samples = {
        "/left-arm-action": {0: [1.0], 1_000_000_000: [2.0]},
        "/right-arm-action": {0: [3.0], 1_000_000_000: [4.0]},
        "/left-arm-state": {0: [5.0], 1_000_000_000: [6.0]},
        "/right-arm-state": {0: [7.0], 1_000_000_000: [8.0]},
        "/left-arm-state:velocity": {0: [0.1], 1_000_000_000: [0.2]},
        "/right-arm-state:velocity": {0: [0.3], 1_000_000_000: [0.4]},
        "/left-arm-state:torque": {0: [1.1], 1_000_000_000: [1.2]},
        "/right-arm-state:torque": {0: [1.3], 1_000_000_000: [1.4]},
        "/left-arm-state:pose": {
            0: [1, 0, 0, 10, 0, 1, 0, 20, 0, 0, 1, 30, 0, 0, 0, 1],
            1_000_000_000: [1, 0, 0, 11, 0, 1, 0, 21, 0, 0, 1, 31, 0, 0, 0, 1],
        },
        "/right-arm-state:pose": {
            0: [1, 0, 0, 40, 0, 1, 0, 50, 0, 0, 1, 60, 0, 0, 0, 1],
            1_000_000_000: [1, 0, 0, 41, 0, 1, 0, 51, 0, 0, 1, 61, 0, 0, 0, 1],
        },
        "/left-ee-action": {0: [0.2], 1_000_000_000: [0.3]},
        "/right-ee-action": {0: [0.4], 1_000_000_000: [0.5]},
        "/left-ee-state": {0: [0.0], 1_000_000_000: [1.0]},
        "/right-ee-state": {0: [1.0], 1_000_000_000: [0.0]},
    }

    streams = _build_robot_streams(
        samples,
        origin_ns=0,
        target_hz=None,
        episode_path=tmp_path / "episode.mcap",
    )
    by_name = {stream.name: stream for stream in streams}

    assert set(by_name) == {
        "action",
        "proprioception",
        "velocity",
        "gripper_action",
        "gripper",
        "joint_torque",
        "eef_velocity",
        "telemetry",
    }
    assert by_name["action"].records[0]["action"] == [1.0, 3.0]
    assert by_name["proprioception"].records[1]["state"] == [6.0, 8.0]
    assert by_name["velocity"].records[1]["qvel"] == [0.2, 0.4]
    assert by_name["gripper_action"].records[0]["gripper_action"] == [0.2, 0.4]
    assert by_name["gripper"].records[0]["gripper"] == [0.0, 1.0]
    assert by_name["joint_torque"].records[1]["torque"] == [1.2, 1.4]
    assert by_name["eef_velocity"].records[0]["eef_linear_velocity"] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert by_name["eef_velocity"].records[0]["eef_angular_velocity"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert by_name["eef_velocity"].metadata["vector_groups"] == {
        "left": [0, 1, 2],
        "right": [3, 4, 5],
    }
    assert by_name["telemetry"].records[0]["wrist_poses"]["left"]["position"] == [10, 20, 30]
    assert by_name["action"].metadata["vector_groups"] == {
        "left": [0],
        "right": [1],
    }
    assert by_name["proprioception"].metadata["vector_groups"] == {
        "left": [0],
        "right": [1],
    }
    assert by_name["gripper"].metadata["vector_groups"] == {
        "left": [0],
        "right": [1],
    }


def test_uniform_k5_is_scored_against_mistake_intervals(tmp_path: Path) -> None:
    frames = [
        CameraFrameRef(
            camera_name="top",
            timestamp_sec=float(index),
            image_path=tmp_path / f"{index}.jpg",
            frame_index=index,
        )
        for index in range(9)
    ]
    bundle = TrajectoryBundle(
        bundle_id="episode",
        source_format=BundleFormat.ABC130K_MCAP,
        root_path=tmp_path / "episode.mcap",
        duration_sec=8.0,
        camera_frames=frames,
        raw_metadata={
            "task_name": "task",
            "abc130k_segments": [
                {
                    "segment_id": "subtask_1",
                    "start_sec": 1.5,
                    "end_sec": 4.5,
                    "failure": True,
                    "text": "mistake",
                }
            ],
        },
    )

    row = _evaluate_selector(
        selector_name="uniform",
        bundle=bundle,
        proposals=[],
        top_k=5,
        config={},
    )

    assert [peak["peak_sec"] for peak in row["selected_peaks"]] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert row["selected_peak_hit_count"] == 2
    assert row["point_precision_at_k"] == pytest.approx(0.4)
    assert row["failed_segment_recall_at_k"] == pytest.approx(1.0)


def test_abc_metric_padding_spans_the_full_remaining_trajectory(tmp_path: Path) -> None:
    frames = [
        CameraFrameRef(
            camera_name="top",
            timestamp_sec=float(index),
            image_path=tmp_path / f"{index}.jpg",
            frame_index=index,
        )
        for index in range(9)
    ]
    bundle = TrajectoryBundle(
        bundle_id="episode",
        source_format=BundleFormat.ABC130K_MCAP,
        root_path=tmp_path / "episode.mcap",
        duration_sec=8.0,
        camera_frames=frames,
        raw_metadata={"task_name": "task", "abc130k_segments": []},
    )
    proposals = [
        SimpleNamespace(
            metadata={"peak_sec": float(index)},
            metric_events=[],
            time_window=TimeWindow(float(index), float(index)),
        )
        for index in (2, 6)
    ]

    unpadded_row = _evaluate_selector(
        selector_name="metric",
        bundle=bundle,
        proposals=proposals,
        top_k=5,
        config={},
    )
    assert unpadded_row["selected_peak_count"] == 2
    assert unpadded_row["uniform_padded_peak_count"] == 0

    row = _evaluate_selector(
        selector_name="metric",
        bundle=bundle,
        proposals=proposals,
        top_k=5,
        config={},
        pad_selectors={"metric"},
    )

    assert sorted(peak["peak_sec"] for peak in row["selected_peaks"]) == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]
    assert row["method_selected_peak_count"] == 2
    assert row["uniform_padded_peak_count"] == 3
    assert [peak["selection_source"] for peak in row["selected_peaks"]] == [
        "metric",
        "metric",
        "uniform_padding",
        "uniform_padding",
        "uniform_padding",
    ]


def test_abc_ronar_k_preserves_protocol_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = TrajectoryBundle(
        bundle_id="episode",
        source_format=BundleFormat.ABC130K_MCAP,
        root_path=tmp_path / "episode.mcap",
        duration_sec=4.0,
        raw_metadata={
            "task_name": "task",
            "abc130k_segments": [
                {
                    "segment_id": "failure",
                    "start_sec": 1.0,
                    "end_sec": 3.0,
                    "failure": True,
                }
            ],
        },
    )

    class FakeEvent:
        def __init__(self, timestamp: float) -> None:
            self.timestamp = timestamp

        def to_dict(self, rank: int) -> dict[str, object]:
            return {
                "rank": rank,
                "peak_sec": self.timestamp,
                "selection_source": "ronar_k_crossing",
            }

    protocol = {
        "protocol": "per_trajectory_exact_k",
        "genuine_selected_event_count": 5,
        "uniform_padded_event_count": 0,
    }
    monkeypatch.setattr(
        "pathologist.abc130k.evaluation.select_ronar_keyevents",
        lambda *args, **kwargs: SimpleNamespace(
            events=[
                FakeEvent(timestamp)
                for timestamp in (0.0, 1.0, 2.0, 3.0, 4.0)
            ],
            metadata=protocol,
        ),
    )

    row = _evaluate_selector(
        selector_name="ronar_k",
        bundle=bundle,
        proposals=[],
        top_k=5,
        config={},
    )

    assert row["selected_peak_count"] == 5
    assert row["selected_peak_hit_count"] == 2
    assert row["point_precision_at_k"] == pytest.approx(0.4)
    assert row["selection_protocol"] == protocol

    strict_row = _evaluate_selector(
        selector_name="ronar",
        bundle=bundle,
        proposals=[],
        top_k=5,
        config={},
    )
    assert strict_row["top_k"] is None
    assert strict_row["requested_top_k"] == 5
    assert strict_row["point_precision"] == pytest.approx(0.4)
    assert "point_precision_at_k" not in strict_row
    assert "failed_segment_recall_at_k" not in strict_row


def test_abc_aggregate_uses_pooled_peak_precision() -> None:
    rows = [
        {
            "episode_hit_at_k": True,
            "failed_segment_count": 2,
            "hit_failed_segment_count": 1,
            "selected_peak_count": 5,
            "selected_peak_hit_count": 2,
            "point_precision_at_k": 0.4,
            "failed_segment_recall_at_k": 0.5,
        },
        {
            "episode_hit_at_k": False,
            "failed_segment_count": 1,
            "hit_failed_segment_count": 0,
            "selected_peak_count": 4,
            "selected_peak_hit_count": 0,
            "point_precision_at_k": 0.0,
            "failed_segment_recall_at_k": 0.0,
        },
    ]

    stats = _aggregate_rows(rows)

    assert stats["episode_recall_at_k"] == pytest.approx(0.5)
    assert stats["failed_segment_recall_at_k"] == pytest.approx(1 / 3)
    assert stats["point_precision_at_k"] == pytest.approx(2 / 9)
    assert stats["macro_point_precision_at_k"] == pytest.approx(0.2)
    assert stats["mean_selected_events_per_trajectory"] == pytest.approx(4.5)
