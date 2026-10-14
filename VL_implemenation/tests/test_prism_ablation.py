from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from VL_implemenation.category_evaluation import (
    VLM_METRIC_NAME,
    _method_configs,
)
from VL_implemenation.category_protocol import load_experiment_config
from VL_implemenation.coarse_to_fine import fine_cache_paths
from VL_implemenation.nms_ablation import select_nms_peaks
from VL_implemenation.prism_ablation import _chance_null, _score_rows
from VL_implemenation.vlm_peak_gate import (
    _paired_bootstrap_safe,
    build_gated_rows,
    gate_peaks,
    nearest_time_index,
    threshold_key,
)


def _rows() -> list[dict]:
    return [
        {
            "bundle_id": "one",
            "duration_sec": 10.0,
            "ground_intervals": [
                {
                    "event_type": "insertion",
                    "start_sec": 2.0,
                    "end_sec": 4.0,
                },
                {
                    "event_type": "removal",
                    "start_sec": 7.0,
                    "end_sec": 8.0,
                },
            ],
            "peaks": {
                "method": [
                    {"time_sec": 3.0},
                    {"time_sec": 7.5},
                    {"time_sec": 9.0},
                ]
            },
        },
        {
            "bundle_id": "two",
            "duration_sec": 5.0,
            "ground_intervals": [
                {
                    "event_type": "insertion",
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                }
            ],
            "peaks": {"method": [{"time_sec": 2.0}]},
        },
    ]


def test_category_configs_include_exact_requested_metrics() -> None:
    experiment = load_experiment_config()
    for details in experiment["categories"].values():
        telemetry = details["telemetry_metrics"]
        methods = _method_configs(telemetry, experiment=experiment)

        assert methods["vlm_only"]["surprise"]["source_metrics"] == [
            VLM_METRIC_NAME
        ]
        fused = methods["telemetry_vlm"]["surprise"]
        assert fused["source_metrics"] == [*telemetry, VLM_METRIC_NAME]
        assert fused["aggregation_groups"] == {
            "telemetry": telemetry,
            "vlm": [VLM_METRIC_NAME],
        }
        assert fused["within_group_reduction"] == "mean"
        assert fused["across_group_reduction"] == "sum"


def test_category_fv_cache_path_has_no_legacy_subdirectory(
    tmp_path: Path,
) -> None:
    episode = type(
        "EpisodeStub",
        (),
        {"task_name": "task", "bundle_id": "bundle"},
    )()

    npz_path, json_path = fine_cache_paths(
        tmp_path / "category",
        episode,
        subdirectory=None,
    )

    assert npz_path == tmp_path / "category" / "task" / "bundle.npz"
    assert json_path == tmp_path / "category" / "task" / "bundle.json"


def test_score_rows_uses_half_open_merged_intervals() -> None:
    metrics = _score_rows(_rows(), method_name="method")

    assert metrics["trajectory_hit_at_1"] == 0.5
    assert metrics["trajectory_hit_at_3"] == 0.5
    assert metrics["trajectory_hit_at_5"] == 0.5
    assert metrics["segment_recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["precision_at_5"] == 0.5
    assert metrics["per_type"]["insertion"]["segment_recall_at_5"] == 0.5
    assert metrics["per_type"]["removal"]["segment_recall_at_5"] == 1.0


def test_chance_null_is_seeded_and_reports_all_metrics() -> None:
    first, intervals = _chance_null(_rows(), draws=50, seed=7)
    second, _ = _chance_null(_rows(), draws=50, seed=7)

    assert first == second
    assert set(intervals) == {
        "trajectory_hit_at_1",
        "trajectory_hit_at_3",
        "trajectory_hit_at_5",
        "segment_recall_at_5",
        "precision_at_5",
    }
    assert 0.0 <= first["trajectory_hit_at_5"] <= 1.0


def test_nearest_time_index_chooses_lower_index_on_tie() -> None:
    times = np.asarray([1.0, 2.0, 3.0])

    assert nearest_time_index(times, 0.0) == 0
    assert nearest_time_index(times, 1.5) == 0
    assert nearest_time_index(times, 2.6) == 2
    assert nearest_time_index(times, 4.0) == 2


def test_gate_peaks_is_strict_and_preserves_source_order() -> None:
    peaks = [
        {"rank": 1, "time_sec": 1.5, "persistence": 3.0},
        {"rank": 2, "time_sec": 2.0, "persistence": 2.0},
        {"rank": 3, "time_sec": 2.6, "persistence": 1.0},
    ]

    gated = gate_peaks(
        peaks,
        centers=np.asarray([1.0, 2.0, 3.0]),
        margins=np.asarray([-2.0, 0.0, 2.0]),
        yes_no_masses=np.asarray([0.8, 0.9, 1.0]),
        threshold=0.0,
    )

    assert [peak["time_sec"] for peak in gated] == [2.6]
    assert gated[0]["rank"] == 1
    assert gated[0]["source_rank"] == 3
    assert gated[0]["vlm_margin"] == 2.0
    assert gated[0]["vlm_center_sec"] == 3.0


def test_gate_considers_candidates_beyond_original_top_five() -> None:
    peaks = [
        {"rank": rank, "time_sec": float(rank), "persistence": 10.0 - rank}
        for rank in range(1, 7)
    ]

    gated = gate_peaks(
        peaks,
        centers=np.arange(1.0, 7.0),
        margins=np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0]),
        yes_no_masses=np.ones(6),
        threshold=0.0,
        max_peaks=5,
    )

    assert [peak["source_rank"] for peak in gated] == [6]


def test_gate_runs_before_peak_height_nms() -> None:
    peaks = [
        {"rank": 1, "time_sec": 0.0, "peak_score": 5.0, "persistence": 9.0},
        {"rank": 2, "time_sec": 4.0, "peak_score": 10.0, "persistence": 8.0},
        {"rank": 3, "time_sec": 20.0, "peak_score": 8.0, "persistence": 7.0},
        {"rank": 4, "time_sec": 25.0, "peak_score": 7.0, "persistence": 6.0},
    ]

    gated = gate_peaks(
        peaks,
        centers=np.asarray([0.0, 4.0, 20.0, 25.0]),
        margins=np.asarray([1.0, -1.0, 1.0, 1.0]),
        yes_no_masses=np.ones(4),
        threshold=0.0,
        max_peaks=5,
        selection_mode="nms",
        nms_radius_sec=8.0,
    )

    # The rejected 4 s maximum suppresses nothing. Of the surviving peaks,
    # 20 s wins by height, suppresses 25 s, and 0 s remains spatially distinct.
    assert [peak["time_sec"] for peak in gated] == [20.0, 0.0]
    assert [peak["source_rank"] for peak in gated] == [3, 1]


def test_peak_height_nms_suppresses_nearby_local_maxima() -> None:
    peaks = [
        {"rank": 1, "time_sec": 2.0, "peak_score": 7.0, "persistence": 9.0},
        {"rank": 2, "time_sec": 5.0, "peak_score": 10.0, "persistence": 3.0},
        {"rank": 3, "time_sec": 12.0, "peak_score": 8.0, "persistence": 8.0},
        {"rank": 4, "time_sec": 20.0, "peak_score": 6.0, "persistence": 7.0},
    ]

    selected = select_nms_peaks(peaks, radius_sec=8.0, top_k=5)

    # The 5 s maximum suppresses 2 and 12 s; exactly 8 s is suppressed.
    assert [peak["time_sec"] for peak in selected] == [5.0, 20.0]
    assert [peak["source_persistence_rank"] for peak in selected] == [2, 4]


def test_build_gated_rows_loads_cache_once_and_fails_closed(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "scores"
    cache_directory = cache_root / "task"
    cache_directory.mkdir(parents=True)
    np.savez(
        cache_directory / "present.npz",
        center_sec=np.asarray([1.0, 2.0]),
        margin=np.asarray([-1.0, 1.0]),
        yes_no_mass=np.asarray([0.9, 0.8]),
    )
    rows = [
        {
            "task_name": "task",
            "bundle_id": bundle_id,
            "duration_sec": 3.0,
            "ground_intervals": [],
            "peaks": {
                "telemetry": [
                    {"rank": 1, "time_sec": 1.0},
                    {"rank": 2, "time_sec": 2.0},
                ]
            },
            "all_peaks": {
                "telemetry": [
                    {"rank": 1, "time_sec": 1.0},
                    {"rank": 2, "time_sec": 2.0},
                ]
            },
        }
        for bundle_id in ("present", "missing")
    ]

    gated_rows = build_gated_rows(
        rows,
        cache_root=cache_root,
        thresholds=[0.0],
    )

    key = threshold_key(0.0)
    assert gated_rows[0]["gate_cache_available"] is True
    assert len(gated_rows[0]["gated_peaks"][key]) == 1
    assert gated_rows[1]["gate_cache_available"] is False
    assert gated_rows[1]["gated_peaks"][key] == []


def test_paired_bootstrap_allows_undefined_gated_precision() -> None:
    rows = [
        {
            **row,
            "peaks": {
                "telemetry": row["peaks"]["method"],
                "telemetry_vlm_gate": [],
            },
        }
        for row in _rows()
    ]

    comparison = _paired_bootstrap_safe(
        rows,
        left="telemetry_vlm_gate",
        right="telemetry",
        samples=10,
        seed=3,
    )

    assert comparison["precision_at_5"]["difference"] is None
    assert comparison["precision_at_5"]["valid_bootstrap_samples"] == 0
    assert comparison["trajectory_hit_at_5"]["valid_bootstrap_samples"] == 10
