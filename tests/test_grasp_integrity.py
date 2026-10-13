"""Tests for the bimanual wrist-camera grasp-integrity metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pathologist.metrics import MetricEngine  # noqa: E402
from pathologist.metrics import grasp_integrity as gi  # noqa: E402
from pathologist.metrics.base import MetricComputationContext  # noqa: E402
from pathologist.schemas import (  # noqa: E402
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TrajectoryBundle,
)

CONFIG = gi.resolve_config({"enabled": True})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _windows(t, cmd, state, empty_closed=0.0, **overrides):
    config = gi.resolve_config({"enabled": True, **overrides})
    return gi.find_hold_windows(
        np.asarray(t), np.asarray(cmd), np.asarray(state),
        arm="left", empty_closed=empty_closed, config=config,
    )


def _telemetry_bundle(t, left_cmd, right_cmd, left_state, right_state, frames=(), bundle_id="b"):
    groups = {"left": [0], "right": [1]}
    return TrajectoryBundle(
        bundle_id=bundle_id,
        source_format=BundleFormat.CUSTOM,
        streams=[
            ModalityStream(
                name="gripper_action",
                modality=Modality.GRIPPER,
                timestamps_sec=list(t),
                records=[{"gripper_action": [float(a), float(b)]} for a, b in zip(left_cmd, right_cmd)],
                metadata={"vector_groups": groups},
            ),
            ModalityStream(
                name="gripper",
                modality=Modality.GRIPPER,
                timestamps_sec=list(t),
                records=[{"gripper": [float(a), float(b)]} for a, b in zip(left_state, right_state)],
                metadata={"vector_groups": groups},
            ),
        ],
        camera_frames=list(frames),
    )


# ---------------------------------------------------------------------------
# 1. Hold windows from telemetry
# ---------------------------------------------------------------------------


def test_wide_object_hold_is_a_window_even_though_state_stays_open() -> None:
    t = np.arange(0, 4, 0.02)
    cmd = np.where((t >= 0.5) & (t < 3.0), 0.2, 1.0)
    state = np.where(t < 0.5, 1.0, np.where(t < 3.0, 0.6, 1.0))
    windows = _windows(t, cmd, state)
    assert len(windows) == 1
    window = windows[0]
    assert window.state_plateau == pytest.approx(0.6)
    assert window.cmd_plateau == pytest.approx(0.2)
    assert window.capacity == pytest.approx(0.4)
    assert not window.telemetry_blind
    assert window.t_open == pytest.approx(3.0, abs=0.03)


def test_command_hysteresis_does_not_chatter() -> None:
    t = np.arange(0, 3, 0.02)
    cmd = np.where(t < 0.5, 1.0, np.where(np.arange(len(t)) % 2 == 0, 0.35, 0.45))
    state = np.where(t < 0.5, 1.0, 0.3)
    assert len(_windows(t, cmd, state)) == 1


def test_open_onset_triggers_on_small_rise_not_at_hysteresis_exit() -> None:
    t = np.arange(0, 4, 0.02)
    cmd = np.full(len(t), 1.0)
    cmd[(t >= 0.5) & (t < 2.0)] = 0.1
    ramp = (t >= 2.0) & (t < 3.0)
    cmd[ramp] = 0.1 + (t[ramp] - 2.0)  # rises 1.0 per second
    state = np.where(t < 0.5, 1.0, 0.5)
    window = _windows(t, cmd, state)[0]
    # first sample above 0.1 + 0.05 is ~2.06 s; the 0.6 exit would be ~2.5 s
    assert window.t_open == pytest.approx(2.06, abs=0.03)


def test_short_windows_are_discarded() -> None:
    t = np.arange(0, 2, 0.02)
    cmd = np.where((t >= 0.5) & (t < 0.7), 0.1, 1.0)
    state = np.where((t >= 0.5) & (t < 0.7), 0.4, 1.0)
    assert _windows(t, cmd, state) == []


def test_settle_uses_measured_state() -> None:
    t = np.arange(0, 3, 0.01)
    cmd = np.where(t >= 0.5, 0.0, 1.0)
    state = np.clip(1.0 - np.maximum(0.0, t - 0.5) * 2.0, 0.5, 1.0)  # closes until 0.75 s
    window = _windows(t, cmd, state)[0]
    assert window.t_close == pytest.approx(0.5, abs=0.011)
    assert window.t_settle == pytest.approx(0.75, abs=0.03)


def test_thin_object_is_telemetry_blind() -> None:
    t = np.arange(0, 3, 0.02)
    cmd = np.where(t >= 0.5, 0.0, 1.0)
    state = np.where(t >= 0.5, 0.02, 1.0)
    window = _windows(t, cmd, state, empty_closed=0.0)[0]
    assert window.capacity == pytest.approx(0.02)
    assert window.telemetry_blind
    assert window.telemetry_empty


def test_aperture_collapse_is_normalized_by_capacity() -> None:
    t = np.arange(0, 4, 0.02)
    cmd = np.where((t >= 0.5) & (t < 3.5), 0.1, 1.0)
    state = np.where(t < 0.5, 1.0, 0.6)
    state[(t >= 2.0) & (t < 3.5)] = 0.1  # object leaves: fingers close toward the command
    windows = _windows(t, cmd, state, empty_closed=0.05)
    collapse = gi.aperture_collapse_series(t, state, windows)
    assert collapse[(t > 2.1) & (t < 3.4)] == pytest.approx(1.0)
    assert collapse[(t > 0.9) & (t < 1.9)] == pytest.approx(0.0)
    assert collapse[t > 3.6].max() == 0.0


def test_arms_are_gated_independently() -> None:
    """Regression for the averaging bug: left closed + right open must not read as 0.5."""

    t = np.arange(0, 3, 0.02)
    bundle = _telemetry_bundle(
        t,
        left_cmd=np.where(t >= 0.5, 0.1, 1.0),
        right_cmd=np.full(len(t), 0.9),
        left_state=np.where(t >= 0.5, 0.5, 1.0),
        right_state=np.full(len(t), 0.9),
    )
    analysis = gi.analyze_bundle(bundle, {"enabled": True, "vision": {"enabled": False}})
    windows = analysis.metadata["hold_windows"]
    assert len(windows["left"]) == 1
    assert windows["right"] == []


# ---------------------------------------------------------------------------
# 2. Geometry, selection, signals, pooling
# ---------------------------------------------------------------------------


def test_contact_band_is_a_ring_outside_the_fingers() -> None:
    finger = np.zeros((40, 40), bool)
    finger[30:, 15:25] = True
    band = gi.contact_band(finger, 2)
    assert not (band & finger).any()
    assert band[28, 20] and band[30, 13]
    assert not band[20, 20]


def _flow(shape, vector):
    flow = np.zeros((*shape, 2), np.float32)
    flow[...] = vector
    return flow


SEL_SHAPE = (60, 80)


def _selection_scene(n_frames=6, pan_px=3, wall=False):
    """Grey wrist frames: textured background panning ``pan_px``/frame, fixed fingers and held object.

    With ``wall``, the left part of the background is a plain grey surface whose
    right edge pans with the background (like the green wall in the overlays).
    Returns frames, finger, held, table (a textured background patch beside the
    fingers) and wall masks, all from frame 0.
    """

    rng = np.random.default_rng(1)
    height, width = SEL_SHAPE
    background = cv2.GaussianBlur(
        rng.uniform(40, 220, (height, width + pan_px * n_frames + 10)).astype(np.float32), (0, 0), 0.8
    )
    held_tex = rng.uniform(150, 255, (10, 10)).astype(np.float32)
    finger = np.zeros(SEL_SHAPE, bool)
    finger[45:, 30:50] = True
    held = np.zeros(SEL_SHAPE, bool)
    held[35:45, 35:45] = True
    frames = []
    for k in range(n_frames):
        image = background[:, pan_px * k : pan_px * k + width].copy()
        if wall:
            image[:, : max(0, 30 - pan_px * k)] = 128.0  # wall edge pans left with the background
        image[finger] = 20.0
        image[35:45, 35:45] = held_tex
        frames.append(np.clip(image, 0, 255).astype(np.uint8))
    table = np.zeros(SEL_SHAPE, bool)
    table[38:52, 18:28] = True
    wall_mask = np.zeros(SEL_SHAPE, bool)
    wall_mask[:, :30] = True
    wall_mask &= ~finger
    return frames, finger, held, table, wall_mask


def test_selection_prefers_rigid_object_over_moving_table() -> None:
    frames, finger, held, table, _ = _selection_scene()
    band = gi.contact_band(finger, 3)
    index, info = gi.select_held_object(
        [table, held], finger=finger, band=band, grays=frames, vision_config=CONFIG["vision"],
    )
    assert info["confidence"] == "high"
    assert info["masks"][index][40, 40]
    rigid, moving = info["rigidity"][1], info["rigidity"][0]
    assert rigid > 0.95 and moving < 0.5


def test_plain_moving_wall_is_not_mistaken_for_the_held_object() -> None:
    """Regression for the overlays: zero optical flow on a plain wall read as 'moves with the fingers'."""

    frames, finger, held, _, wall = _selection_scene(wall=True)
    band = gi.contact_band(finger, 3)
    assert (wall & band).sum() > (held & band).sum()  # the wall has more contact
    flow = cv2.calcOpticalFlowFarneback(frames[0], frames[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    interior = gi._erode(wall, 8)
    assert np.median(np.linalg.norm(flow[interior], axis=-1)) < 0.5  # what fooled co-motion
    index, info = gi.select_held_object(
        [wall, held], finger=finger, band=band, grays=frames, vision_config=CONFIG["vision"],
    )
    assert info["confidence"] == "high"
    assert info["masks"][index][40, 40]
    assert info["rigidity"][0] is None or info["rigidity"][0] < 0.6


def test_candidate_without_evidence_is_not_eligible() -> None:
    """A thin, plain candidate has no textured interior and no ring, so it cannot be judged or chosen."""

    frames, finger, held, _, _ = _selection_scene()
    plain = np.zeros(SEL_SHAPE, bool)
    plain[46:, 50:54] = True  # a 4-px plain strip beside the fingers, e.g. a cable
    for frame in frames:
        frame[plain] = 90
    band = gi.contact_band(finger, 3)
    scores, counts = gi.rigidity_scores(
        [plain, held], frames, [1, 2, 3], finger=finger, other_arm_seq=None,
        vision_config=CONFIG["vision"],
    )
    assert scores[0] is None and counts[0] < CONFIG["vision"]["rigidity_min_evidence_px"]
    index, info = gi.select_held_object(
        [plain, held], finger=finger, band=band, grays=frames, vision_config=CONFIG["vision"],
    )
    assert info["masks"][index][40, 40]


def test_rigidity_is_judged_after_the_arm_starts_moving() -> None:
    """Right after the grasp the arm is still; the evidence comes later in the window."""

    frames, finger, held, table, _ = _selection_scene(n_frames=10)
    still = [frames[0]] * 5 + frames[1:6]  # 5 still frames, then the pan starts
    band = gi.contact_band(finger, 3)
    index, info = gi.select_held_object(
        [table, held], finger=finger, band=band, grays=still, vision_config=CONFIG["vision"],
    )
    assert min(info["informative_frames"]) >= 5
    assert info["confidence"] == "high" and info["masks"][index][40, 40]


def test_selection_without_background_change_falls_back_to_contact() -> None:
    frames, finger, _, _, _ = _selection_scene()
    band = gi.contact_band(finger, 3)
    small = np.zeros(SEL_SHAPE, bool)
    small[41:45, 46:52] = True
    large = np.zeros(SEL_SHAPE, bool)
    large[33:45, 33:47] = True
    index, info = gi.select_held_object(
        [small, large], finger=finger, band=band, grays=[frames[0]] * 6,
        vision_config=CONFIG["vision"],
    )
    assert info["confidence"] == "low" and info["informative_frames"] == []
    assert info["masks"][index][35, 40]


def test_static_arm_fallback_prefers_fingertip_contact() -> None:
    """Without background change, a table object touching the side of the fingers must not win."""

    frames, finger, _, _, _ = _selection_scene()
    band = gi.contact_band(finger, 3)
    at_tips = np.zeros(SEL_SHAPE, bool)
    at_tips[38:45, 36:44] = True
    beside = np.zeros(SEL_SHAPE, bool)
    beside[40:60, 22:30] = True  # larger contact, but along the side of the fingers
    assert (beside & band).sum() > (at_tips & band).sum()
    index, info = gi.select_held_object(
        [beside, at_tips], finger=finger, band=band, grays=[frames[0]] * 6,
        vision_config=CONFIG["vision"], tip=gi.tip_band(band, "bottom"),
    )
    assert info["confidence"] == "low"
    assert info["masks"][index][40, 40]


def test_window_signals_lost_track_and_occlusion() -> None:
    shape = (60, 80)
    finger = np.zeros(shape, bool)
    finger[45:, 30:50] = True
    band = gi.contact_band(finger, 3)
    held = np.zeros(shape, bool)
    held[35:45, 35:45] = True
    empty = np.zeros(shape, bool)
    covering = band.copy()
    times = [0.0, 0.1, 0.2, 0.3]
    result = gi.window_signals(
        times=times,
        object_masks=[held, held, empty, held],
        scores=None,
        finger=finger,
        band=band,
        other_arm_seq=[None, None, None, covering],
        flows=[_flow(shape, (0.0, 0.0))] * 3,
        t_settle=0.0,
        init_contact=int((held & band).sum()),
        vision_config=CONFIG["vision"],
    )
    assert result.attach_loss[:2] == pytest.approx([0.0, 0.0])
    assert result.attach_loss[2] == 1.0 and not result.valid[2]
    assert result.gap[2] == pytest.approx(CONFIG["vision"]["gap_cap_frac"])
    assert result.occluded[3] and result.attach_loss[3] == 0.0


def test_two_frame_spike_survives_downsampling() -> None:
    frame_t = list(np.arange(0, 2, 1 / 30))
    values = [0.0] * len(frame_t)
    values[31] = values[32] = 1.0
    intervals = gi.frame_hold_intervals(frame_t, 0.0, 2.0)
    pooled = gi.pool_onto_grid(list(np.arange(0, 2, 0.2)), intervals, values)
    assert pooled.max() == 1.0
    nearest = [values[int(np.argmin(np.abs(np.asarray(frame_t) - g)))] for g in np.arange(0, 2, 0.2)]
    assert max(nearest) == 0.0  # nearest-neighbour sampling would have missed it


def test_slow_signal_becomes_step_when_upsampled() -> None:
    frame_t = [0.0, 0.2, 0.4]
    intervals = gi.frame_hold_intervals(frame_t, 0.0, 0.6)
    grid = list(np.arange(0, 0.6, 0.02))
    pooled = gi.pool_onto_grid(grid, intervals, [0.0, 1.0, 0.0])
    on = [g for g, v in zip(grid, pooled) if v == 1.0]
    assert min(on) == pytest.approx(0.1, abs=0.011) and max(on) == pytest.approx(0.3, abs=0.011)


def test_transfer_rule() -> None:
    def window(arm, settle, open_):
        return gi.HoldWindow(arm, settle - 0.2, settle, open_, 0.1, 0.5, 0.5, 0.4, False, False)

    drop = gi.GraspEvent("left", "drop", 2.5, "vision", 0)
    windows = {"left": [window("left", 1.0, 4.0)], "right": [window("right", 2.8, 5.0)]}
    found = gi.WindowVision(status="ok")
    events = gi.apply_transfer_rule([drop], windows, {"left": [found], "right": [found]}, CONFIG["events"])
    assert events[0].kind == "transfer"

    late = gi.GraspEvent("left", "drop", 2.5, "vision", 0)
    windows["right"] = [window("right", 4.5, 6.0)]
    events = gi.apply_transfer_rule([late], windows, {"left": [found], "right": [found]}, CONFIG["events"])
    assert events[0].kind == "drop"


# ---------------------------------------------------------------------------
# 3. End-to-end on a synthetic wrist-camera drop
# ---------------------------------------------------------------------------

H, W = 120, 160
DROP_T = 2.5
SETTLE_T = 0.5


def _object_mask(image_rgb):  # red, textured
    r, g, b = image_rgb[..., 0].astype(int), image_rgb[..., 1].astype(int), image_rgb[..., 2].astype(int)
    return (r >= 140) & (g <= 40) & (b <= 40)


def _decoy_mask(image_rgb):  # green, textured, painted on the moving background
    r, g, b = image_rgb[..., 0].astype(int), image_rgb[..., 1].astype(int), image_rgb[..., 2].astype(int)
    return (g >= 140) & (r <= 40) & (b <= 40)


def _finger_mask(image_rgb):  # dark grey, textured
    channels = image_rgb.astype(int)
    return (channels.max(axis=-1) <= 70) & (channels.max(axis=-1) - channels.min(axis=-1) <= 5)


class ColorSegmenter:
    """Stand-in for SAM2: segments the synthetic scene by colour."""

    def identity(self) -> str:
        return "color-test"

    def finger_mask(self, image_rgb, entry_edge, edge_fraction):
        mask = _finger_mask(image_rgb)
        return mask if mask.any() else None

    def robot_masks(self, image_paths):
        return [_finger_mask(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in image_paths]

    def candidate_masks(self, image_rgb, positive_points, negative_points, text_prompt):
        assert positive_points and negative_points
        return [m for m in (_object_mask(image_rgb), _decoy_mask(image_rgb)) if m.any()]

    def track(self, image_paths, positive_points, negative_points):
        """Track whichever coloured region contains the first positive point."""

        first = cv2.cvtColor(cv2.imread(str(image_paths[0])), cv2.COLOR_BGR2RGB)
        x, y = (int(round(v)) for v in positive_points[0])
        segment = _object_mask if _object_mask(first)[y, x] else _decoy_mask
        self.tracked = segment.__name__
        masks, scores = [], []
        for path in image_paths:
            mask = segment(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
            masks.append(mask)
            scores.append(1.0 if mask.any() else -1.0)
        return masks, scores


SEGMENTER = ColorSegmenter()
gi.register_grasp_segmenter("color_test", lambda config: SEGMENTER)


def _noise_patch(rng, height, width, channel):
    """Non-periodic texture in one colour channel (BGR index), so flow is unambiguous."""

    patch = np.zeros((height, width, 3), np.uint8)
    patch[..., channel] = rng.integers(150, 256, (height, width))
    return patch


FINGER_TEX = np.random.default_rng(7).integers(15, 65, (H - 95, 31)).astype(np.uint8)


def _write_scene(directory: Path, fps: float = 10.0, duration: float = 5.0) -> list[CameraFrameRef]:
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.uniform(110, 230, (H, W + 400)).astype(np.float32), (0, 0), 2.0)
    texture = np.clip(texture, 110, 230).astype(np.uint8)
    finger_tex = FINGER_TEX
    obj = _noise_patch(rng, 20, 13, channel=2)
    decoy = _noise_patch(rng, 30, 10, channel=1)
    frames = []
    for k, t in enumerate(np.arange(0, duration, 1 / fps)):
        shift = 3 * k  # the arm moves, so the background slides 3 px/frame in the wrist view
        image = cv2.cvtColor(texture[:, shift : shift + W], cv2.COLOR_GRAY2BGR)
        # Decoy on the background: touches the fingers when the grasp settles, then slides away.
        dx = int(round(55 - 3 * (t - SETTLE_T) * fps))
        lo, hi = max(0, dx), min(W, dx + 10)
        if hi > lo:
            image[80:110, lo:hi] = decoy[:, lo - dx : hi - dx]
        image[95:, 65:96] = cv2.cvtColor(finger_tex, cv2.COLOR_GRAY2BGR)
        if t < DROP_T:
            image[75:95, 74:87] = obj
        elif t < DROP_T + 0.3:
            off = int(round(8 * (t - DROP_T) * fps + 8))
            image[75 - off : 95 - off, 74 - off : 87 - off] = obj
        path = directory / f"{k:06d}.png"
        cv2.imwrite(str(path), image)
        frames.append(CameraFrameRef(camera_name="left_wrist", timestamp_sec=float(t), image_path=path, frame_index=k))
    return frames


def _drop_bundle(tmp_path: Path):
    frames = _write_scene(tmp_path)
    t = np.arange(0, 5, 0.02)
    left_cmd = np.where((t >= 0.3) & (t < 4.0), 0.0, 1.0)
    left_state = np.where(t < 0.3, 1.0, np.where(t < 4.0, 0.5, 1.0))
    left_state[(t >= 0.3) & (t < 0.5)] = 1.0 - (t[(t >= 0.3) & (t < 0.5)] - 0.3) * 2.5
    return _telemetry_bundle(
        t, left_cmd, np.ones(len(t)), left_state, np.ones(len(t)), frames=frames, bundle_id="drop"
    )


def _calibration_episode(directory: Path, rng, n_frames=12, finger_at=None, held=None):
    """Random scenes (a new one every frame) seen through a wrist camera with fixed fingers.

    ``finger_at(aperture)`` gives the finger patch's column offset; ``held``
    paints a fixed patch (an object held in every episode) on frames with
    aperture below 0.7.
    """

    directory.mkdir(parents=True, exist_ok=True)
    times = np.arange(n_frames) * 0.5
    apertures = np.where(np.arange(n_frames) < n_frames // 2, 1.0, 0.5)
    paths = []
    for k, (t, aperture) in enumerate(zip(times, apertures)):
        scene = cv2.GaussianBlur(rng.uniform(60, 240, (H, W)).astype(np.float32), (0, 0), 1.5)
        image = cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        x0 = 65 + (finger_at(aperture) if finger_at else 0)
        image[95:, x0 : x0 + 31] = cv2.cvtColor(FINGER_TEX, cv2.COLOR_GRAY2BGR)
        if held is not None and aperture < 0.7:
            image[75:95, x0 + 9 : x0 + 22] = held
        path = directory / f"{k:04d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths, list(times), list(times), list(apertures)


def _build_calibration(tmp_path: Path, n_episodes=10, **scene):
    from pathologist.metrics.wrist_finger_calibration import FingerCalibrator

    rng = np.random.default_rng(3)
    calibrator = FingerCalibrator(work_width=W, bin_width=0.1)
    for e in range(n_episodes):
        paths, times, ap_t, ap = _calibration_episode(tmp_path / f"ep{e}", rng, **scene.get("episode", {}))
        calibrator.add_episode(paths, times, ap_t, ap, exclude=scene.get("exclude", ()))
    return calibrator.finish("left_wrist", min_episodes=scene.get("min_episodes", 8))


def _grasp_config(tmp_path: Path | None = None, calibration_dir: Path | None = None) -> dict:
    vision = {"backend": "color_test", "work_width": 160}
    if tmp_path is not None:
        vision["cache_dir"] = str(tmp_path / "cache")
    calibration_dir = calibration_dir or tmp_path
    if calibration_dir is not None:
        path = calibration_dir / "left_wrist_fingers.npz"
        if not path.exists():
            _build_calibration(calibration_dir / "calib").save(path)
        vision["finger_calibration"] = {"left_wrist": str(path)}
    else:
        vision["finger_fallback"] = "segmented"
    return {"metrics": {"grasp_integrity": {"enabled": True, "vision": vision}}}


def _metric(bundle, metric_class, config):
    return metric_class().compute(MetricComputationContext(bundle=bundle, config=config))


def test_end_to_end_drop_is_localized(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    config = _grasp_config(calibration_dir=tmp_path)
    attach = _metric(bundle, gi.GraspAttachLossMetric, config)
    gap = _metric(bundle, gi.GraspObjectGapMetric, config)
    rel = _metric(bundle, gi.GraspRelativeMotionMetric, config)
    collapse = _metric(bundle, gi.GraspApertureCollapseMetric, config)

    window = attach.metadata["hold_windows"]["left"][0]
    assert window["vision_status"] == "ok"
    info = window["vision_info"]
    assert info["finger_source"] == "calibration"
    assert info["finger_bin"] == pytest.approx(0.5)  # hold aperture 0.5
    assert info["confidence"] == "high"
    assert info["n_kept"] == 2  # both touch the fingers when the grasp settles
    assert min(info["rigidity"]) < 0.5
    assert max(info["rigidity"]) > 0.95
    assert SEGMENTER.tracked == "_object_mask"  # selection picked the rigid object

    def values(result, lo, hi):
        return [s.value for s in result.samples if lo <= s.timestamp_sec < hi]

    assert max(values(attach, 0.7, DROP_T - 0.1)) == pytest.approx(0.0)
    assert max(values(attach, DROP_T, DROP_T + 0.5)) >= 0.8
    # A drop is a pulse at the detachment, not a plateau until the gripper opens.
    assert max(values(attach, DROP_T + 0.7, 3.95)) == 0.0
    assert max(values(gap, DROP_T + 0.7, 3.95)) == 0.0
    assert max(values(gap, 0.7, DROP_T - 0.1)) <= 0.01  # object touches the fingers (~1 px)
    assert max(values(gap, DROP_T, DROP_T + 0.5)) > 0.03
    peak_t = max(rel.samples, key=lambda s: s.value).timestamp_sec
    assert DROP_T - 0.15 <= peak_t <= DROP_T + 0.5
    assert all(s.value == 0.0 for s in collapse.samples)  # state never fell
    drops = [e for e in attach.metadata["events"] if e["kind"] == "drop"]
    assert len(drops) == 1 and abs(drops[0]["time_sec"] - DROP_T) <= 0.2
    assert drops[0]["source"] == "vision"
    assert all(arm in (None, "left") for arm in attach.metadata["winning_arm"])


def test_window_cache_round_trip(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    first = _metric(bundle, gi.GraspAttachLossMetric, _grasp_config(tmp_path))
    assert list((tmp_path / "cache" / "left_wrist").glob("*.json"))
    gi._ANALYSIS_CACHE.clear()
    second = _metric(bundle, gi.GraspAttachLossMetric, _grasp_config(tmp_path))
    assert second.metadata["hold_windows"]["left"][0]["vision_status"] == "ok"
    assert [s.value for s in first.samples] == [s.value for s in second.samples]


def test_engine_integration_and_ecod(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    config = _grasp_config(calibration_dir=tmp_path)
    config["surprise"] = {"source_metrics": list(gi.SIGNAL_NAMES)}
    profile = MetricEngine(config=config).compute(bundle)
    names = {metric.name for metric in profile.metrics}
    assert set(gi.SIGNAL_NAMES) <= names
    aggregate = next(m for m in profile.metrics if m.name == "ecod_aggregate_score")
    assert aggregate.samples
    peak = max(aggregate.samples, key=lambda s: s.value)
    assert DROP_T - 0.2 <= peak.timestamp_sec <= DROP_T + 0.5


# ---------------------------------------------------------------------------
# 3b. Finger calibration
# ---------------------------------------------------------------------------


def _patch_mask(x0):
    mask = np.zeros((H, W), bool)
    mask[95:, x0 : x0 + 31] = True
    return mask


def test_calibration_recovers_aperture_dependent_fingers(tmp_path: Path) -> None:
    # Fingers sit 10 px further right when half closed (a stand-in for closing).
    calibration = _build_calibration(
        tmp_path, episode={"finger_at": lambda a: 0 if a > 0.7 else 10}
    )
    assert sorted(calibration.centers.tolist()) == [0.5, 1.0]
    for aperture, x0 in ((0.97, 65), (0.52, 75)):
        mask, center, distance = calibration.mask_for(aperture)
        assert gi._mask_iou(mask, _patch_mask(x0)) > 0.9
        assert distance < 0.05
    # Nearest bin when the hold aperture was never seen empty.
    _, center, distance = calibration.mask_for(0.2)
    assert center == pytest.approx(0.5) and distance == pytest.approx(0.3)


def test_calibration_round_trips_and_needs_enough_episodes(tmp_path: Path) -> None:
    from pathologist.metrics.wrist_finger_calibration import FingerCalibration

    calibration = _build_calibration(tmp_path / "a")
    path = calibration.save(tmp_path / "c.npz")
    loaded = FingerCalibration.load(path)
    assert loaded.digest() == calibration.digest()
    assert loaded.work_shape == (H, W)
    with pytest.raises(ValueError, match="only 3 episodes"):
        _build_calibration(tmp_path / "b", n_episodes=3)


def test_calibration_excludes_held_objects(tmp_path: Path) -> None:
    """An object held in every episode is rigid with the camera too; hold frames must be left out."""

    held = np.full((20, 13, 3), (0, 0, 200), np.uint8)
    polluted = _build_calibration(tmp_path / "p", episode={"held": held})
    mask, _, _ = polluted.mask_for(0.5)
    assert mask[85, 80]  # the held object was absorbed into the "fingers"
    clean = _build_calibration(tmp_path / "c", episode={"held": held}, exclude=[(2.9, 100.0)])
    mask, center, _ = clean.mask_for(0.5)
    assert center == pytest.approx(1.0)  # only open frames remain
    assert not mask[85, 80] and gi._mask_iou(mask, _patch_mask(65)) > 0.9


def test_missing_finger_calibration_skips_vision_but_keeps_telemetry(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    config = {"metrics": {"grasp_integrity": {"enabled": True, "vision": {"backend": "color_test", "work_width": 160}}}}
    attach = _metric(bundle, gi.GraspAttachLossMetric, config)
    window = attach.metadata["hold_windows"]["left"][0]
    assert window["vision_status"] == "no_finger_calibration"
    collapse = _metric(bundle, gi.GraspApertureCollapseMetric, config)
    assert collapse.metadata["available"] is True


def test_calibration_inputs_exclude_hold_windows(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    paths, times, _, _, exclude = gi.calibration_inputs(bundle, {"enabled": True}, "left")
    assert len(paths) == len(times) == 50
    assert len(exclude) == 1
    start, end = exclude[0]
    assert start == pytest.approx(0.3 - 0.5) and end == pytest.approx(4.0 + 0.5, abs=0.03)


def test_overlay_is_written(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    config = _grasp_config(calibration_dir=tmp_path)
    config["metrics"]["grasp_integrity"]["vision"]["debug_dir"] = str(tmp_path / "overlays")
    _metric(bundle, gi.GraspAttachLossMetric, config)
    written = list((tmp_path / "overlays" / "left_wrist").glob("*_pick*_high.jpg"))
    assert len(written) == 1
    image = cv2.imread(str(written[0]))
    assert image.shape[1] == 2 * W  # settle frame | later frame


# ---------------------------------------------------------------------------
# 4. Availability and task-agnosticism
# ---------------------------------------------------------------------------


def test_telemetry_only_drop_of_wide_object() -> None:
    """Vision off: a wide object leaving the fingers shows up as aperture collapse."""

    t = np.arange(0, 4, 0.02)
    left_cmd = np.where((t >= 0.3) & (t < 3.5), 0.0, 1.0)
    left_state = np.where(t < 0.3, 1.0, 0.6)
    left_state[(t >= 2.0) & (t < 3.5)] = 0.02  # object gone, fingers close fully
    left_state[t >= 3.5] = 1.0
    bundle = _telemetry_bundle(t, left_cmd, np.ones(len(t)), left_state, np.ones(len(t)), bundle_id="wide")
    config = {"metrics": {"grasp_integrity": {"enabled": True, "vision": {"enabled": False}}}}
    collapse = _metric(bundle, gi.GraspApertureCollapseMetric, config)
    attach = _metric(bundle, gi.GraspAttachLossMetric, config)
    peak = max(collapse.samples, key=lambda s: s.value)
    assert peak.value == pytest.approx((0.6 - 0.02) / 0.6)  # capacity = plateau - command
    assert peak.timestamp_sec == pytest.approx(2.0, abs=0.03)
    assert max(s.value for s in collapse.samples if 2.6 <= s.timestamp_sec < 3.4) == 0.0
    drops = [e for e in collapse.metadata["events"] if e["kind"] == "drop"]
    assert len(drops) == 1 and drops[0]["source"] == "telemetry"
    assert attach.metadata["available"] is False  # vision disabled


def test_disabled_by_default() -> None:
    t = np.arange(0, 2, 0.02)
    bundle = _telemetry_bundle(t, np.ones(len(t)), np.ones(len(t)), np.ones(len(t)), np.ones(len(t)))
    result = _metric(bundle, gi.GraspAttachLossMetric, {})
    assert result.metadata["available"] is False
    assert "disabled" in result.metadata["unavailable_reason"]


def test_missing_backend_keeps_telemetry_signal(tmp_path: Path) -> None:
    bundle = _drop_bundle(tmp_path)
    config = {"metrics": {"grasp_integrity": {"enabled": True, "vision": {"backend": "does_not_exist"}}}}
    attach = _metric(bundle, gi.GraspAttachLossMetric, config)
    collapse = _metric(bundle, gi.GraspApertureCollapseMetric, config)
    assert attach.metadata["available"] is False
    assert "backend" in attach.metadata["unavailable_reason"]
    assert collapse.metadata["available"] is True and collapse.samples


def test_module_never_reads_task_text() -> None:
    source = Path(gi.__file__).read_text(encoding="utf-8")
    for forbidden in ("task_contract", ".instruction", "annotation_text", "raw_metadata"):
        assert forbidden not in source
