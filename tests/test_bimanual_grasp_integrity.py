"""Regression tests for per-arm slip/drop telemetry and signal math."""

from __future__ import annotations

import pytest

from pathologist.metrics.grasp_integrity import (
    HeldObjectCandidate,
    HoldWindow,
    aperture_collapse,
    attachment_loss,
    detect_hold_windows,
    max_pool_resample,
    relative_motion_from_flow,
    select_held_object_candidate,
)
from pathologist.metrics.grasp_initialization import (
    _aperture_adjusted_finger_polygons,
    _fingertip_band_points,
    _inner_finger_gap_band,
)
from pathologist.metrics.grasp_tracking import (
    HoldSignalRecord,
    _apply_bimanual_suppression,
)


np = pytest.importorskip("numpy")


def _wide_object_hold() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(0.0, 2.01, 0.02)
    command = np.ones(len(timestamps))
    command[(timestamps >= 0.2) & (timestamps < 1.4)] = 0.2
    command[(timestamps >= 1.4) & (timestamps < 1.6)] = 0.26
    state = np.ones(len(timestamps))
    closing = (timestamps >= 0.2) & (timestamps < 0.5)
    state[closing] = np.linspace(1.0, 0.6, int(np.count_nonzero(closing)))
    state[(timestamps >= 0.5) & (timestamps < 1.4)] = 0.6
    state[(timestamps >= 1.4) & (timestamps < 1.6)] = 0.62
    return timestamps, command, state


def test_wide_object_hold_uses_command_not_measured_aperture() -> None:
    timestamps, command, measured = _wide_object_hold()
    windows = detect_hold_windows(
        timestamps,
        command,
        timestamps,
        measured,
        arm="left",
        empty_closed=0.01,
    )

    assert len(windows) == 1
    window = windows[0]
    assert window.state_plateau == pytest.approx(0.6)
    assert window.state_plateau > 0.5
    assert window.command_plateau == pytest.approx(0.2)
    assert window.settle_sec == pytest.approx(0.5, abs=0.04)
    assert window.open_sec == pytest.approx(1.4)


def test_per_arm_commands_are_never_averaged() -> None:
    timestamps = np.arange(0.0, 1.01, 0.02)
    left_command = np.full(len(timestamps), 0.1)
    right_command = np.full(len(timestamps), 0.9)
    left_state = np.full(len(timestamps), 0.6)
    right_state = np.full(len(timestamps), 0.9)

    left = detect_hold_windows(
        timestamps,
        left_command,
        timestamps,
        left_state,
        arm="left",
        empty_closed=0.01,
    )
    right = detect_hold_windows(
        timestamps,
        right_command,
        timestamps,
        right_state,
        arm="right",
        empty_closed=0.01,
    )

    assert len(left) == 1
    assert right == []
    assert np.mean([left_command[0], right_command[0]]) == pytest.approx(0.5)


def test_command_hysteresis_does_not_chatter_around_half() -> None:
    timestamps = np.arange(0.0, 1.41, 0.1)
    command = np.asarray(
        [0.9, 0.2, 0.45, 0.55, 0.48, 0.52, 0.45, 0.55, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9]
    )
    measured = np.full(len(timestamps), 0.3)

    windows = detect_hold_windows(
        timestamps,
        command,
        timestamps,
        measured,
        arm="left",
        empty_closed=0.01,
        open_rise=0.5,
    )

    assert len(windows) == 1


def test_settle_comes_from_measured_state_and_open_uses_small_command_rise() -> None:
    timestamps, command, measured = _wide_object_hold()
    window = detect_hold_windows(
        timestamps,
        command,
        timestamps,
        measured,
        arm="right",
        empty_closed=0.01,
    )[0]

    assert window.close_intent_sec == pytest.approx(0.2)
    assert window.settle_sec > window.close_intent_sec
    assert window.open_sec < 1.6


def test_short_hold_is_discarded() -> None:
    timestamps = np.arange(0.0, 1.01, 0.02)
    command = np.ones(len(timestamps))
    command[(timestamps >= 0.2) & (timestamps < 0.4)] = 0.1
    measured = command.copy()

    assert detect_hold_windows(
        timestamps,
        command,
        timestamps,
        measured,
        arm="left",
        empty_closed=0.01,
    ) == []


def test_thin_object_is_empty_and_telemetry_blind() -> None:
    timestamps = np.arange(0.0, 1.01, 0.02)
    command = np.full(len(timestamps), 0.01)
    measured = np.full(len(timestamps), 0.025)

    window = detect_hold_windows(
        timestamps,
        command,
        timestamps,
        measured,
        arm="left",
        empty_closed=0.01,
    )[0]

    assert window.grasp_width == pytest.approx(0.015)
    assert window.capacity == pytest.approx(0.015)
    assert window.telemetry_empty is True
    assert window.telemetry_blind is True


def test_aperture_collapse_uses_available_capacity() -> None:
    collapse = aperture_collapse(
        [0.6, 0.6, 0.35, 0.1],
        state_plateau=0.6,
        capacity=0.5,
        telemetry_blind=False,
    )
    blind = aperture_collapse(
        [0.6, 0.1],
        state_plateau=0.6,
        capacity=0.0,
        telemetry_blind=True,
    )

    assert collapse.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert blind.tolist() == [0.0, 0.0]


def test_attachment_loss_invalid_and_occlusion_rules() -> None:
    band = np.zeros((8, 8), dtype=bool)
    band[3:5, 3:5] = True
    touching = np.zeros_like(band)
    touching[3:5, 3:5] = True
    detached = np.zeros_like(band)
    other_arm = np.zeros_like(band)
    other_arm[3:5, 3:5] = True

    loss, occluded = attachment_loss(
        [touching, detached, detached],
        [True, True, False],
        band,
        other_arm_masks=[
            np.zeros_like(band),
            np.zeros_like(band),
            other_arm,
        ],
    )

    assert loss.tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert occluded.tolist() == [False, False, True]


def test_relative_motion_has_expected_units() -> None:
    flow = np.zeros((60, 80, 2), dtype=float)
    fingers = np.zeros((60, 80), dtype=bool)
    fingers[10:20, 10:20] = True
    obj = np.zeros((60, 80), dtype=bool)
    obj[30:40, 30:40] = True
    flow[obj, 0] = 5.0

    value = relative_motion_from_flow(
        flow,
        obj,
        fingers,
        native_fps=30.0,
    )

    assert value == pytest.approx(1.5)


def test_max_pool_resampling_preserves_two_frame_spike() -> None:
    source_timestamps = np.arange(0.0, 1.0, 1.0 / 30.0)
    values = np.zeros(len(source_timestamps))
    values[10:12] = 1.0
    target_timestamps = np.arange(0.0, 1.0, 0.2)

    pooled = max_pool_resample(
        source_timestamps,
        values,
        target_timestamps,
    )

    assert max(pooled) == 1.0


def test_comotion_prefers_candidate_moving_with_fingers() -> None:
    fingers = np.zeros((100, 100), dtype=bool)
    fingers[80:95, 40:60] = True
    band = np.zeros_like(fingers)
    band[70:80, 35:65] = True
    held = np.zeros_like(fingers)
    held[68:78, 42:58] = True
    table = np.zeros_like(fingers)
    table[70:80, 20:40] = True
    flows = []
    for _ in range(4):
        flow = np.zeros((100, 100, 2), dtype=float)
        flow[..., 0] = 2.0
        flow[fingers, 0] = 0.0
        flow[held, 0] = 0.0
        flows.append(flow)

    selection = select_held_object_candidate(
        [
            HeldObjectCandidate(table, "table"),
            HeldObjectCandidate(held, "held"),
        ],
        flow_fields=flows,
        finger_mask=fingers,
        contact_band=band,
    )

    assert selection.candidate is not None
    assert selection.candidate.source == "held"
    assert selection.comotion == pytest.approx(1.0)
    assert selection.init_confidence == "high"


def test_static_background_falls_back_to_largest_contact() -> None:
    fingers = np.zeros((100, 100), dtype=bool)
    fingers[80:95, 40:60] = True
    band = np.zeros_like(fingers)
    band[70:80, 30:70] = True
    small = np.zeros_like(fingers)
    small[70:75, 32:38] = True
    large = np.zeros_like(fingers)
    large[70:78, 45:60] = True
    flows = [np.zeros((100, 100, 2), dtype=float) for _ in range(4)]

    selection = select_held_object_candidate(
        [
            HeldObjectCandidate(small, "small"),
            HeldObjectCandidate(large, "large"),
        ],
        flow_fields=flows,
        finger_mask=fingers,
        contact_band=band,
    )

    assert selection.candidate is not None
    assert selection.candidate.source == "large"
    assert selection.comotion is None
    assert selection.init_confidence == "low"


def test_fingertip_prompts_stay_inside_two_finger_gap() -> None:
    fingers = np.zeros((100, 100), dtype=bool)
    fingers[40:, :35] = True
    fingers[40:, 65:] = True
    band = np.zeros_like(fingers)
    band[40:90, 35:38] = True
    band[40:90, 62:65] = True
    band[37:40, :] = True

    points = _fingertip_band_points(
        band,
        finger_mask=fingers,
        count=5,
    )

    assert len(points) == 5
    assert all(35 <= x < 65 for x, _ in points)
    assert all(y <= 85 for _, y in points)


def test_contact_band_excludes_outer_jaw_faces() -> None:
    fingers = np.zeros((60, 100), dtype=bool)
    fingers[20:, 20:35] = True
    fingers[20:, 65:80] = True
    full_band = np.zeros_like(fingers)
    full_band[20:, 18:20] = True
    full_band[20:, 35:37] = True
    full_band[20:, 63:65] = True
    full_band[20:, 80:82] = True

    inner = _inner_finger_gap_band(
        full_band,
        finger_mask=fingers,
    )

    assert np.any(inner[:, 35:37])
    assert np.any(inner[:, 63:65])
    assert not np.any(inner[:, :20])
    assert not np.any(inner[:, 80:])


def test_finger_templates_follow_measured_aperture_per_arm() -> None:
    templates = [
        [[0.45, 1.0], [0.49, 0.8]],
        [[0.51, 0.8], [0.55, 1.0]],
    ]

    adjusted = _aperture_adjusted_finger_polygons(
        templates,
        aperture=0.6,
        shift_fraction=0.25,
    )

    assert np.asarray(adjusted[0])[:, 0].tolist() == pytest.approx([0.30, 0.34])
    assert np.asarray(adjusted[1])[:, 0].tolist() == pytest.approx([0.66, 0.70])


def test_handover_suppresses_nearby_transfer_but_not_late_hold() -> None:
    def record(arm: str, settle: float, event_time: float | None) -> HoldSignalRecord:
        timestamps = np.asarray([1.0, 1.1, 1.2])
        event = (
            {"label": "drop", "timestamp_sec": event_time, "source": "vision"}
            if event_time is not None
            else None
        )
        return HoldSignalRecord(
            arm=arm,
            window_index=0,
            window=HoldWindow(
                arm=arm,
                close_intent_sec=settle - 0.2,
                settle_sec=settle,
                open_sec=3.0,
                state_plateau=0.5,
                command_plateau=0.1,
                empty_closed=0.0,
                grasp_width=0.5,
                capacity=0.4,
                telemetry_empty=False,
                telemetry_blind=False,
            ),
            timestamps=timestamps,
            attach_loss=np.ones(3),
            rel_motion=np.zeros(3),
            aperture_collapse=np.ones(3),
            valid=np.ones(3, dtype=bool),
            occluded=np.zeros(3, dtype=bool),
            cache_hit=False,
            other_arm_cache_hit=False,
            initialization_found=True,
            event=event,
        )

    sender = record("left", 0.5, 1.0)
    receiver = record("right", 1.3, None)
    _apply_bimanual_suppression([sender, receiver])
    assert sender.event["label"] == "transfer"
    assert sender.attach_loss.tolist() == [0.0, 0.0, 0.0]

    late_sender = record("left", 0.5, 1.0)
    late_receiver = record("right", 3.0, None)
    _apply_bimanual_suppression([late_sender, late_receiver])
    assert late_sender.event["label"] == "drop"


def test_runtime_module_is_task_agnostic() -> None:
    from pathlib import Path

    source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "pathologist/metrics/grasp_integrity.py",
            "pathologist/metrics/grasp_initialization.py",
            "pathologist/metrics/grasp_tracking.py",
        )
    )
    assert "task_name" not in source
    assert "subtask" not in source
