from __future__ import annotations

import numpy as np
import pytest

from VL_implemenation.coarse_to_fine import (
    CoarsePlan,
    build_coarse_plan,
    greedy_temporal_nms,
    select_final_peak,
    select_probe_frame_indices,
)


def test_greedy_temporal_nms_ranks_by_score_and_suppresses_radius() -> None:
    selected = greedy_temporal_nms(
        times=[0.0, 4.0, 8.0, 12.0, 20.0],
        scores=[0.5, 0.9, 0.8, 0.7, 0.6],
        radius_sec=8.0,
        max_candidates=3,
    )

    # 4 s wins; 0, 8, and 12 s are suppressed because Eq. 2 requires > r_f.
    assert selected == [1, 4]


def test_coarse_plan_places_eight_clipped_uniform_fine_probes() -> None:
    plan = build_coarse_plan(
        centers=[1.0, 10.0, 19.0],
        margins=[3.0, 2.0, 1.0],
        duration_sec=20.0,
        radius_sec=8.0,
        candidate_count=3,
        fine_count=8,
    )

    assert plan.centers.tolist() == [1.0, 10.0, 19.0]
    assert plan.fine_centers.shape == (3, 8)
    assert plan.fine_centers[0].tolist() == pytest.approx(
        np.linspace(0.0, 9.0, 8)
    )
    assert plan.fine_centers[-1].tolist() == pytest.approx(
        np.linspace(11.0, 20.0, 8)
    )


def test_final_peak_uses_joint_candidate_score_then_fine_argmax() -> None:
    plan = CoarsePlan(
        indices=np.asarray([2, 7]),
        centers=np.asarray([2.0, 7.0]),
        margins=np.asarray([2.0, 0.0]),
        probabilities=np.asarray([0.8, 0.5]),
        fine_centers=np.asarray([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
    )
    fine_margins = np.asarray(
        [
            [-2.0, -1.0, 0.0],
            [-2.0, 4.0, -1.0],
        ]
    )

    result = select_final_peak(plan, fine_margins, alpha=0.5)

    assert result["winning_candidate_index"] == 1
    assert result["winning_fine_index"] == 1
    assert result["prediction_sec"] == 7.0


def test_probe_frames_have_fixed_even_count_and_nearest_mean() -> None:
    times = np.arange(30, dtype=np.float64) / 5.0

    indices = select_probe_frame_indices(
        times,
        center_sec=3.0,
        frame_count=10,
    )

    assert len(indices) == 10
    assert indices == tuple(range(indices[0], indices[0] + 10))
    assert np.mean(times[list(indices)]) == pytest.approx(2.9)
