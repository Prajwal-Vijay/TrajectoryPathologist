"""Bimanual grasp-integrity metrics from wrist cameras and gripper telemetry.

The metrics detect slips and drops of an object held by either arm. They are
task-agnostic: nothing here reads task names, instructions, or annotations.

Per arm, independently:

1. Hold windows come from that arm's *commanded* aperture (intent). Settling
   comes from the *measured* aperture (the fingers stop moving), and the window
   ends as soon as the command starts to open. The measured aperture is never
   thresholded to decide "grasping", because a wide object stops the fingers at,
   say, 0.6 while the grasp is firm.
2. In that arm's wrist camera the fingers are rigid with the camera, so their
   image region depends only on the aperture. It is calibrated once per camera
   from many episodes (:mod:`pathologist.metrics.wrist_finger_calibration`):
   finger pixels are the ones whose appearance does not change across scenes.
   The held object is also rigid with the camera, so during the hold it stays
   on the same pixels while the arm moves and the background changes. It is the
   candidate segment that touches a thin band around the fingers and whose
   pixels keep their appearance once the background has visibly changed
   ("rigidity"). Rigidity is measured only on pixels that carry evidence -
   textured interior and inner boundary - because a plain, untextured surface
   looks unchanged whether or not it moves.
3. Four signals, all large on a slip or drop and near zero during a normal hold:

   - ``grasp_attach_loss``: 1 - (object/finger-band contact) / (contact at
     settle). A lost track counts as full loss, because a dropped object is
     usually motion-blurred or leaves the view.
   - ``grasp_object_gap``: shortest distance from the object to the fingers, as
     a fraction of the image diagonal. Direction-free, so it does not depend on
     where gravity points in the wrist image.
   - ``grasp_relative_motion``: object image motion relative to the fingers, in
     image diagonals per second (slip).
   - ``grasp_aperture_collapse``: how far the measured aperture fell from its
     hold plateau toward the command, normalized by how far it *could* fall.
     Telemetry only; strong for wide objects, blind for thin ones.

Bimanual rules: each arm uses only its own wrist camera and telemetry. A drop is
relabelled ``transfer`` (and its signals zeroed from that moment) when the
other arm settles a grasp on an object shortly after. Per timestamp, each signal
is the max over the two arms. Values are max-pooled onto the gripper telemetry
grid so that 1-3 frame events survive alignment with other metrics.

Segmentation sits behind a small backend interface (``GraspSegmenter``). The
default backend uses Grounding DINO + SAM2 through ``transformers``, sharing the
models already loaded by :mod:`pathologist.metrics.robot_masking`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import extract_vector_series, summarize
from pathologist.schemas.common import Modality
from pathologist.schemas.metrics import MetricResult, MetricSample
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle

CONFIG_NAME = "grasp_integrity"
SIGNAL_NAMES = (
    "grasp_attach_loss",
    "grasp_object_gap",
    "grasp_relative_motion",
    "grasp_aperture_collapse",
)
VISION_SIGNALS = SIGNAL_NAMES[:3]
WINDOW_CACHE_VERSION = 2

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "arms": {
        "left": {"group": "left", "camera": "left_wrist"},
        "right": {"group": "right", "camera": "right_wrist"},
    },
    "command_stream": "gripper_action",
    "command_key": "gripper_action",
    "state_stream": "gripper",
    "state_key": "gripper",
    "component_index": 0,
    # Measured aperture when closed on nothing (0 = fully closed). Calibrate per
    # arm once per dataset; it bounds how far the fingers can collapse.
    "empty_closed": {"left": 0.0, "right": 0.0},
    "close_enter": 0.4,
    "close_exit": 0.6,
    "settle_speed_per_sec": 0.1,
    "settle_span_sec": 0.1,
    "settle_cap_sec": 1.0,
    "open_rise": 0.05,
    "plateau_sec": 0.2,
    "min_window_sec": 0.3,
    "telemetry_empty_width": 0.03,
    "telemetry_blind_capacity": 0.05,
    "vision": {
        "enabled": True,
        "backend": "grounded_sam2",
        "work_width": 320,
        # Finger mask sources, in order: a fixed mask image per camera
        # (finger_masks), else a calibration file per camera (finger_calibration,
        # built by scripts/check_grasp_integrity.py), else finger_fallback:
        # "none" (window skipped, status no_finger_calibration) or "segmented"
        # (text-prompted Grounding DINO + SAM2; unreliable in wrist views).
        "finger_masks": {},
        "finger_calibration": {},
        "finger_fallback": "none",
        "finger_entry_edge": "bottom",
        "finger_edge_fraction": 0.05,
        "band_radius_frac": 0.02,
        "num_positive_points": 5,
        "object_text_prompt": "object. item. tool.",
        "min_area_frac": 0.001,
        # A held object right in front of the wrist camera can fill half the view.
        "max_area_frac": 0.6,
        "max_finger_overlap": 0.2,
        "max_other_arm_overlap": 0.5,
        "min_contact_px": 6,
        "merge_iou": 0.7,
        # Held-object selection by rigidity (see select_held_object).
        "rigidity_blur_sigma": 1.0,
        # A pixel "kept its appearance" if it changed by at most this many grey
        # levels: above sensor/JPEG noise after the blur, well below what a moved
        # textured pixel changes by.
        "rigidity_tolerance": 10.0,
        # "Textured" = local (7x7) grey-level std at least this. After a move, a
        # non-rigid pixel is roughly a fresh draw from its neighbourhood, so it
        # stays within the tolerance with probability <= 1 - 2(1 - Phi(10 / (12
        # sqrt 2))) = 0.44, versus ~1 for a rigid pixel.
        "rigidity_min_contrast": 12.0,
        "rigidity_ring_px": 3,  # inner-boundary ring width
        "rigidity_far_px": 5,  # ring pixels this close to fingers/border carry no evidence
        "rigidity_min_evidence_px": 15,
        "informative_change_frac": 0.15,  # background visibly changed in this frame
        "max_selection_frames": 6,
        "min_informative_frames": 2,
        "min_rigidity": 0.6,
        "rigidity_tie_margin": 0.1,
        # Half the Farneback window (15 px): flow closer than this to another
        # region's boundary is contaminated by that region's motion.
        "flow_boundary_px": 7.5,
        "other_arm_period_sec": 0.2,
        "occlusion_band_fraction": 0.3,
        "max_area_jump": 3.0,
        "object_score_threshold": 0.0,
        "gap_cap_frac": 0.25,
        "contact_ref_sec": 0.2,
        "cache_dir": None,
        # If set, write one initialization overlay per hold window (fingers,
        # contact band, candidates, chosen object) for manual QA.
        "debug_dir": None,
        "dino_model_id": "IDEA-Research/grounding-dino-tiny",
        "sam2_model_id": "facebook/sam2.1-hiera-tiny",
        "device": "auto",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "mask_threshold": 0.0,
        "finger_prompt": "robot gripper.",
        "robot_prompt": "robot arm. robot gripper.",
    },
    "events": {
        "drop_threshold": 0.8,
        "collapse_threshold": 0.8,
        "slip_speed": 0.05,
        "slip_attach_max": 0.5,
        "min_duration_sec": 0.1,
        "transfer_before_sec": 1.0,
        "transfer_after_sec": 0.5,
        # After a confirmed drop, keep the signals up this long, then zero frames
        # still in the lost state: the event is the detachment, not the absence.
        "post_drop_pulse_sec": 0.5,
    },
}


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class HoldWindow:
    """One grasp of one arm, from telemetry."""

    arm: str
    t_close: float
    t_settle: float
    t_open: float
    cmd_plateau: float
    state_plateau: float
    grasp_width: float
    capacity: float
    telemetry_empty: bool
    telemetry_blind: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WindowVision:
    """Frame-level vision signals for one hold window."""

    status: str
    times: list[float] = field(default_factory=list)
    attach_loss: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    rel_motion: list[float] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    occluded: list[bool] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def object_found(self) -> bool:
        return self.status == "ok"


@dataclass(slots=True)
class GraspEvent:
    """Discrete diagnostic event (PRISM itself uses the continuous signals)."""

    arm: str
    kind: str
    time_sec: float
    source: str
    window_index: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Segmentation backend interface
# --------------------------------------------------------------------------


class GraspSegmenter(Protocol):
    """Segmentation backend. All masks are boolean arrays at full image size."""

    def identity(self) -> str:
        """Stable string identifying models and settings, used in cache keys."""

    def finger_mask(self, image_rgb: Any, entry_edge: str, edge_fraction: float) -> Any | None:
        """Mask of this arm's closed fingers in its wrist image, or None."""

    def robot_masks(self, image_paths: list[Path]) -> list[Any]:
        """Union of all robot-part masks per frame (own fingers included)."""

    def candidate_masks(
        self,
        image_rgb: Any,
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
        text_prompt: str,
    ) -> list[Any]:
        """Candidate held-object masks: one per positive point, plus text detections."""

    def track(
        self,
        image_paths: list[Path],
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
    ) -> tuple[list[Any], list[float] | None]:
        """Track one object from frame 0; returns per-frame masks and presence scores."""


_SEGMENTER_FACTORIES: dict[str, Callable[[dict[str, Any]], GraspSegmenter]] = {}


def register_grasp_segmenter(
    name: str,
    factory: Callable[[dict[str, Any]], GraspSegmenter],
) -> None:
    """Register a segmentation backend selectable via ``vision.backend``."""

    _SEGMENTER_FACTORIES[name] = factory


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def resolve_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge user config over the defaults."""

    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in {
                "arms",
                "finger_masks",
                "finger_calibration",
                "empty_closed",
            }:
                out[key] = merge(out[key], value)
            else:
                out[key] = copy.deepcopy(value)
        return out

    return merge(DEFAULT_CONFIG, raw or {})


def _config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------------------------------
# Telemetry: per-arm series and hold windows
# --------------------------------------------------------------------------


def per_arm_series(
    bundle: TrajectoryBundle,
    stream_name: str,
    key: str,
    arms: dict[str, dict[str, Any]],
    component_index: int,
) -> tuple[list[float], dict[str, list[float]], str | None]:
    """Return timestamps and one scalar series per arm from a paired vector stream."""

    stream = next((s for s in bundle.streams if s.name == stream_name), None)
    if stream is None:
        return [], {}, f"missing stream {stream_name!r}"
    timestamps, vectors = extract_vector_series(stream, [key])
    if len(vectors) < 2:
        return [], {}, f"stream {stream_name!r} has fewer than two samples"
    width = min(len(vector) for vector in vectors)
    groups = stream.metadata.get("vector_groups") if stream.metadata else None
    if not isinstance(groups, dict) or not groups:
        groups = {"left": [0], "right": [1]} if width == 2 else {}
    series: dict[str, list[float]] = {}
    for arm, arm_config in arms.items():
        indices = groups.get(str(arm_config.get("group", arm)))
        if not indices or component_index >= len(indices):
            continue
        index = int(indices[component_index])
        if index >= width:
            continue
        series[arm] = [float(vector[index]) for vector in vectors]
    if not series:
        return [], {}, f"no configured arm group found in stream {stream_name!r}"
    return [float(t) for t in timestamps[: len(vectors)]], series, None


def find_hold_windows(
    t: Any,
    cmd: Any,
    state: Any,
    *,
    arm: str,
    empty_closed: float,
    config: dict[str, Any],
) -> list[HoldWindow]:
    """Find grasp hold windows for one arm.

    ``t``, ``cmd`` and ``state`` share one time grid (command interpolated onto
    the state timestamps). Close intent uses hysteresis on the command; settle
    uses the measured state; the window ends when the command starts to rise.
    """

    import numpy as np

    t = np.asarray(t, dtype=float)
    cmd = np.asarray(cmd, dtype=float)
    state = np.asarray(state, dtype=float)
    n = len(t)
    enter = float(config["close_enter"])
    exit_ = float(config["close_exit"])

    runs: list[tuple[int, int]] = []
    closed = False
    start = 0
    for i in range(n):
        if not closed and cmd[i] < enter:
            closed, start = True, i
        elif closed and cmd[i] > exit_:
            closed = False
            runs.append((start, i))
    if closed:
        runs.append((start, n))

    speed = float(config["settle_speed_per_sec"])
    span = float(config["settle_span_sec"])
    cap = float(config["settle_cap_sec"])
    windows: list[HoldWindow] = []
    for start, end in runs:
        t_close = float(t[start])
        settle_idx = None
        for j in range(start, end):
            if t[j] > t_close + cap:
                break
            k = int(np.searchsorted(t, t[j] + span - 1e-9))
            if k >= end:
                break
            dt = t[k] - t[j]
            if dt > 0 and abs(state[k] - state[j]) / dt < speed:
                settle_idx = j
                break
        if settle_idx is None:
            settle_idx = min(int(np.searchsorted(t, t_close + cap)), end - 1)
        settle_idx = max(start, settle_idx)
        t_settle = float(t[settle_idx])

        plateau_end = max(
            settle_idx + 1,
            min(end, int(np.searchsorted(t, t_settle + float(config["plateau_sec"])))),
        )
        cmd_plateau = float(np.median(cmd[settle_idx:plateau_end]))
        state_plateau = float(np.median(state[settle_idx:plateau_end]))

        open_idx = end
        for j in range(settle_idx, end):
            if cmd[j] > cmd_plateau + float(config["open_rise"]):
                open_idx = j
                break
        t_open = float(t[open_idx]) if open_idx < n else float(t[-1])
        if t_open - t_settle < float(config["min_window_sec"]):
            continue
        grasp_width = state_plateau - empty_closed
        capacity = state_plateau - max(cmd_plateau, empty_closed)
        windows.append(
            HoldWindow(
                arm=arm,
                t_close=t_close,
                t_settle=t_settle,
                t_open=t_open,
                cmd_plateau=cmd_plateau,
                state_plateau=state_plateau,
                grasp_width=grasp_width,
                capacity=capacity,
                telemetry_empty=grasp_width < float(config["telemetry_empty_width"]),
                telemetry_blind=capacity < float(config["telemetry_blind_capacity"]),
            )
        )
    return windows


def aperture_collapse_series(t: Any, state: Any, windows: list[HoldWindow]) -> Any:
    """Normalized aperture collapse on the state grid; zero outside hold windows."""

    import numpy as np

    t = np.asarray(t, dtype=float)
    state = np.asarray(state, dtype=float)
    out = np.zeros(len(t))
    for window in windows:
        if window.telemetry_blind or window.capacity <= 0:
            continue
        inside = (t >= window.t_settle) & (t < window.t_open)
        out[inside] = np.clip(
            (window.state_plateau - state[inside]) / window.capacity, 0.0, 1.0
        )
    return out


# --------------------------------------------------------------------------
# Geometry helpers (work resolution)
# --------------------------------------------------------------------------


def contact_band(finger_mask: Any, radius_px: int) -> Any:
    """Thin ring around the fingers: dilate(F, r) minus F."""

    import cv2
    import numpy as np

    finger = np.asarray(finger_mask, dtype=bool)
    radius = max(1, int(radius_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(finger.astype(np.uint8), kernel).astype(bool)
    return dilated & ~finger


def _resize_mask(mask: Any, shape_hw: tuple[int, int]) -> Any:
    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape_hw:
        return mask
    return cv2.resize(
        mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _separated_maxima(mask: Any, count: int) -> list[tuple[int, int]]:
    """Up to ``count`` well-separated interior points (x, y) of a mask."""

    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = max(3, int(0.25 * math.sqrt(float(mask.sum()))))
    points: list[tuple[int, int]] = []
    for _ in range(count):
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        if dist[y, x] <= 0:
            break
        points.append((int(x), int(y)))
        cv2.circle(dist, (int(x), int(y)), radius, 0.0, thickness=-1)
    return points


def tip_band(band: Any, entry_edge: str) -> Any:
    """Fingertip end of the contact band: the 30% of band pixels deepest into the image.

    A held object sits at the fingertips; a table object next to the gripper
    usually touches the side of the fingers instead.
    """

    import numpy as np

    band = np.asarray(band, dtype=bool)
    ys, xs = np.nonzero(band)
    tip = np.zeros_like(band)
    if len(xs) == 0:
        return tip
    height, width = band.shape
    depth = {
        "bottom": (height - 1) - ys,
        "top": ys,
        "left": xs,
        "right": (width - 1) - xs,
    }.get(entry_edge, (height - 1) - ys)
    keep = depth >= np.percentile(depth, 70)
    tip[ys[keep], xs[keep]] = True
    return tip


def _band_points(band: Any, count: int, entry_edge: str) -> list[tuple[int, int]]:
    """Points spread along the fingertip end of the contact band."""

    import numpy as np

    ys, xs = np.nonzero(tip_band(band, entry_edge))
    if len(xs) == 0:
        return []
    along = xs if entry_edge in {"bottom", "top"} else ys
    order = np.argsort(along, kind="stable")
    xs, ys = xs[order], ys[order]
    picks = np.unique(np.linspace(0, len(xs) - 1, num=min(count, len(xs))).round().astype(int))
    return [(int(xs[i]), int(ys[i])) for i in picks]


def _mask_iou(first: Any, second: Any) -> float:
    inter = float((first & second).sum())
    union = float((first | second).sum())
    return inter / union if union > 0 else 0.0


def _median_flow(flow: Any, mask: Any) -> Any | None:
    import numpy as np

    if mask is None or not mask.any():
        return None
    return np.median(flow[mask], axis=0)


def _interior(mask: Any, others: Any, distance_px: float) -> Any:
    """Pixels of ``mask`` at least ``distance_px`` from ``others``.

    Dense flow is smoothed over its window, so pixels near a motion boundary
    inherit the neighbouring region's motion. If no pixel is far enough, the
    farther half of the mask is used.
    """

    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    others = np.asarray(others, dtype=bool)
    if not mask.any() or not others.any():
        return mask
    dist = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
    core = mask & (dist >= distance_px)
    if core.any():
        return core
    values = dist[mask]
    return mask & (dist >= np.median(values))


# --------------------------------------------------------------------------
# Held-object selection
# --------------------------------------------------------------------------


def _blurred(gray: Any, sigma: float) -> Any:
    import cv2
    import numpy as np

    image = np.asarray(gray, dtype=np.float32)
    return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image


def _local_contrast(image: Any, size: int = 7) -> Any:
    """Grey-level standard deviation in a ``size`` x ``size`` window."""

    import cv2
    import numpy as np

    image = np.asarray(image, dtype=np.float32)
    mean = cv2.boxFilter(image, -1, (size, size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(image * image, -1, (size, size), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _erode(mask: Any, radius: int) -> Any:
    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def evidence_pixels(mask: Any, finger: Any, contrast: Any, vision_config: dict[str, Any]) -> Any:
    """Pixels of a candidate whose appearance would change if the candidate moved.

    Two kinds: textured interior pixels, and an inner ring along the
    candidate's boundary. A plain interior carries no evidence (a uniform wall
    looks the same after the camera pans), but the region's boundary does,
    because it shifts in the image unless the region is rigid with the camera.
    Ring pixels next to the fingers or the image border are excluded: those
    boundaries are fixed for every region.
    """

    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    finger = np.asarray(finger, dtype=bool)
    min_contrast = float(vision_config["rigidity_min_contrast"])
    ring_px = int(vision_config["rigidity_ring_px"])
    far_px = float(vision_config["rigidity_far_px"])
    # Start 3 px inside the boundary (interior) / 2 px (ring) so the blur does not
    # mix in the neighbouring region, whose change says nothing about this one.
    interior = _erode(mask, 3)
    textured = interior & (np.asarray(contrast) >= min_contrast)
    ring = _erode(mask, 2) & ~_erode(mask, 2 + ring_px)
    fixed = finger.copy()
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    distance = cv2.distanceTransform((~fixed).astype(np.uint8), cv2.DIST_L2, 5)
    return textured | (ring & (distance >= far_px))


def informative_frames(
    grays: list[Any],
    finger: Any,
    other_arm_seq: list[Any] | None,
    vision_config: dict[str, Any],
) -> list[int]:
    """Frames (after frame 0) in which the background has visibly changed.

    Only such frames can tell a rigid region from a static one. A frame counts
    when at least ``informative_change_frac`` of the textured non-finger pixels
    differ from frame 0 by more than the tolerance. The held object itself is
    among those pixels and does not change, so the fraction is diluted by the
    object's share of the textured scene, not inflated by it.
    """

    import cv2
    import numpy as np

    if len(grays) < 2:
        return []
    sigma = float(vision_config["rigidity_blur_sigma"])
    tolerance = float(vision_config["rigidity_tolerance"])
    reference = _blurred(grays[0], sigma)
    kernel = np.ones((7, 7), np.uint8)
    near_finger = cv2.dilate(np.asarray(finger, dtype=np.uint8), kernel).astype(bool)
    scene = ~near_finger & (_local_contrast(reference) >= float(vision_config["rigidity_min_contrast"]))
    minimum = int(vision_config["rigidity_min_evidence_px"])
    picks: list[int] = []
    for k in range(1, len(grays)):
        pixels = scene
        if other_arm_seq is not None and other_arm_seq[k] is not None:
            pixels = scene & ~np.asarray(other_arm_seq[k], dtype=bool)
        if int(pixels.sum()) < minimum:
            continue
        changed = np.abs(_blurred(grays[k], sigma) - reference) > tolerance
        if float(changed[pixels].mean()) >= float(vision_config["informative_change_frac"]):
            picks.append(k)
            if len(picks) >= int(vision_config["max_selection_frames"]):
                break
    return picks


def rigidity_scores(
    candidates: list[Any],
    grays: list[Any],
    frames: list[int],
    *,
    finger: Any,
    other_arm_seq: list[Any] | None,
    vision_config: dict[str, Any],
) -> tuple[list[float | None], list[int]]:
    """Per candidate: median over ``frames`` of the share of its evidence pixels unchanged since frame 0.

    Returns (scores, evidence pixel counts). A score is None when the candidate
    has too few evidence pixels to be judged.
    """

    import numpy as np

    sigma = float(vision_config["rigidity_blur_sigma"])
    tolerance = float(vision_config["rigidity_tolerance"])
    minimum = int(vision_config["rigidity_min_evidence_px"])
    reference = _blurred(grays[0], sigma)
    contrast = _local_contrast(reference)
    unchanged = {
        k: np.abs(_blurred(grays[k], sigma) - reference) <= tolerance for k in frames
    }
    scores: list[float | None] = []
    counts: list[int] = []
    for mask in candidates:
        evidence = evidence_pixels(mask, finger, contrast, vision_config)
        counts.append(int(evidence.sum()))
        per_frame = []
        for k in frames:
            pixels = evidence
            if other_arm_seq is not None and other_arm_seq[k] is not None:
                pixels = evidence & ~np.asarray(other_arm_seq[k], dtype=bool)
            if int(pixels.sum()) >= minimum:
                per_frame.append(float(unchanged[k][pixels].mean()))
        scores.append(float(np.median(per_frame)) if per_frame else None)
    return scores, counts


def select_held_object(
    candidates: list[Any],
    *,
    finger: Any,
    band: Any,
    grays: list[Any],
    vision_config: dict[str, Any],
    other_arm_seq: list[Any] | None = None,
    tip: Any | None = None,
) -> tuple[int | None, dict[str, Any]]:
    """Pick the candidate that touches the fingers and is rigid with the camera.

    ``grays`` are the window's work-resolution grey frames, frame 0 at settle.
    The held object keeps its pixels and appearance while the background
    changes; a table object or wall next to the fingers does not. Rigidity is
    judged on frames anywhere in the window where the background has visibly
    changed, because right after the grasp the arm is usually still and every
    region looks static. Returns the chosen index (into the filtered, merged
    list stored in ``info['masks']``) or None.
    """

    import numpy as np

    finger = np.asarray(finger, dtype=bool)
    band = np.asarray(band, dtype=bool)
    first_other = other_arm_seq[0] if other_arm_seq else None
    other_arm = (
        np.zeros_like(finger) if first_other is None else np.asarray(first_other, dtype=bool)
    )
    image_area = float(finger.size)
    kept: list[Any] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for raw in candidates:
        mask = np.asarray(raw, dtype=bool)
        area = float(mask.sum())
        if area < float(vision_config["min_area_frac"]) * image_area:
            reject("too_small")
            continue
        if area > float(vision_config["max_area_frac"]) * image_area:
            reject("too_large")
            continue
        if float((mask & finger).sum()) / area > float(vision_config["max_finger_overlap"]):
            reject("mostly_finger")
            continue
        if float((mask & other_arm).sum()) / area > float(vision_config["max_other_arm_overlap"]):
            reject("mostly_other_arm")
            continue
        if int((mask & band).sum()) < int(vision_config["min_contact_px"]):
            reject("no_finger_contact")
            continue
        merged = False
        for index, existing in enumerate(kept):
            if _mask_iou(existing, mask) > float(vision_config["merge_iou"]):
                kept[index] = existing | mask
                merged = True
                break
        if not merged:
            kept.append(mask)

    info: dict[str, Any] = {
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "rejected": rejected,
        "masks": kept,
    }
    if not kept:
        info["confidence"] = "none"
        return None, info

    # Contact with the fingertip end of the band (all of the band if not given).
    tip = band if tip is None else np.asarray(tip, dtype=bool)
    contacts = [int((mask & tip).sum()) for mask in kept]
    frames = informative_frames(grays, finger, other_arm_seq, vision_config)
    info["contacts"] = contacts
    info["informative_frames"] = frames
    if len(frames) >= int(vision_config["min_informative_frames"]):
        rigidity, evidence = rigidity_scores(
            kept, grays, frames, finger=finger, other_arm_seq=other_arm_seq,
            vision_config=vision_config,
        )
        info["rigidity"] = rigidity
        info["evidence_px"] = evidence
        eligible = [
            index
            for index, score in enumerate(rigidity)
            if score is not None and score >= float(vision_config["min_rigidity"])
        ]
        if not eligible:
            info["confidence"] = "none"
            return None, info
        # Rigidity is the defining property of the held object, so rank by it;
        # fingertip contact only breaks near-ties.
        best = max(rigidity[index] for index in eligible)
        tied = [
            index
            for index in eligible
            if rigidity[index] >= best - float(vision_config["rigidity_tie_margin"])
        ]
        chosen = max(tied, key=lambda index: contacts[index])
        info["confidence"] = "high"
        info["selected"] = chosen
        return chosen, info
    # The background never changed visibly (arm still for the whole hold, or a
    # plain backdrop): no evidence about rigidity, so fall back to fingertip contact.
    info["confidence"] = "low"
    chosen = int(np.argmax(contacts))
    info["selected"] = chosen
    return chosen, info


# --------------------------------------------------------------------------
# Per-window vision signals
# --------------------------------------------------------------------------


def window_signals(
    *,
    times: list[float],
    object_masks: list[Any],
    scores: list[float] | None,
    finger: Any,
    band: Any,
    other_arm_seq: list[Any] | None,
    flows: list[Any],
    t_settle: float,
    init_contact: int,
    vision_config: dict[str, Any],
) -> WindowVision:
    """Compute attach loss, object gap and relative motion for one window."""

    import cv2
    import numpy as np

    finger = np.asarray(finger, dtype=bool)
    band = np.asarray(band, dtype=bool)
    image_area = float(finger.size)
    diagonal = math.hypot(*finger.shape)
    band_area = max(1.0, float(band.sum()))
    gap_cap = float(vision_config["gap_cap_frac"])
    dist_to_finger = cv2.distanceTransform((~finger).astype(np.uint8), cv2.DIST_L2, 5)
    edge = float(vision_config["flow_boundary_px"])
    finger_core = _interior(finger, ~finger, edge)

    n = len(times)
    valid: list[bool] = []
    previous_area: float | None = None
    for k in range(n):
        mask = np.asarray(object_masks[k], dtype=bool)
        area = float(mask.sum())
        ok = area >= float(vision_config["min_area_frac"]) * image_area
        if ok and float((mask & finger).sum()) / area > float(vision_config["max_finger_overlap"]):
            ok = False
        if ok and previous_area:
            jump = max(area / previous_area, previous_area / area)
            if jump > float(vision_config["max_area_jump"]):
                ok = False
        if ok and scores is not None and scores[k] is not None:
            ok = float(scores[k]) >= float(vision_config["object_score_threshold"])
        if ok:
            previous_area = area
        valid.append(bool(ok))

    occluded = []
    for k in range(n):
        other = other_arm_seq[k] if other_arm_seq else None
        if other is None:
            occluded.append(False)
        else:
            covered = float((np.asarray(other, dtype=bool) & band).sum()) / band_area
            occluded.append(covered > float(vision_config["occlusion_band_fraction"]))

    contacts = [
        int((np.asarray(object_masks[k], dtype=bool) & band).sum()) if valid[k] else 0
        for k in range(n)
    ]
    ref_frames = [
        contacts[k]
        for k in range(n)
        if valid[k] and times[k] <= t_settle + float(vision_config["contact_ref_sec"])
    ]
    contact_ref = float(np.median(ref_frames)) if ref_frames else float(init_contact)
    contact_ref = max(1.0, contact_ref)

    attach, gap, rel = [], [], []
    for k in range(n):
        mask = np.asarray(object_masks[k], dtype=bool)
        if occluded[k]:
            attach.append(0.0)
            gap.append(0.0)
        elif not valid[k]:
            attach.append(1.0)
            gap.append(gap_cap)
        else:
            attach.append(float(1.0 - min(1.0, contacts[k] / contact_ref)))
            gap.append(float(min(gap_cap, float(dist_to_finger[mask].min()) / diagonal)))
        if k == 0 or occluded[k] or not (valid[k] and valid[k - 1]):
            rel.append(0.0)
            continue
        flow = flows[k - 1]
        previous_mask = np.asarray(object_masks[k - 1], dtype=bool)
        object_flow = _median_flow(flow, _interior(previous_mask, finger, edge))
        finger_flow = _median_flow(flow, finger_core)
        dt = max(1e-6, times[k] - times[k - 1])
        if object_flow is None or finger_flow is None:
            rel.append(0.0)
        else:
            rel.append(float(np.linalg.norm(object_flow - finger_flow)) / diagonal / dt)

    return WindowVision(
        status="ok",
        times=list(times),
        attach_loss=attach,
        gap=gap,
        rel_motion=rel,
        valid=valid,
        occluded=occluded,
        info={"contact_ref": contact_ref, "valid_fraction": float(np.mean(valid)) if n else 0.0},
    )


# --------------------------------------------------------------------------
# Max-pooling onto the output grid
# --------------------------------------------------------------------------


def frame_hold_intervals(times: list[float], start: float, end: float) -> list[tuple[float, float]]:
    """Interval each frame value holds for: midpoints between frames, clipped to the window."""

    intervals = []
    n = len(times)
    for k in range(n):
        lo = start if k == 0 else 0.5 * (times[k - 1] + times[k])
        hi = end if k == n - 1 else 0.5 * (times[k] + times[k + 1])
        intervals.append((max(start, lo), min(end, hi)))
    return intervals


def pool_onto_grid(
    grid: list[float],
    intervals: list[tuple[float, float]],
    values: list[float],
) -> Any:
    """Max of every value whose hold interval overlaps each grid bin.

    Bins are bounded by midpoints between grid timestamps, so a brief spike is
    never skipped when downsampling, and a slow signal becomes a step function
    when upsampling.
    """

    import numpy as np

    grid = np.asarray(grid, dtype=float)
    out = np.zeros(len(grid))
    if len(grid) == 0:
        return out
    edges = np.concatenate(([-np.inf], 0.5 * (grid[1:] + grid[:-1]), [np.inf]))
    for (lo, hi), value in zip(intervals, values):
        if hi <= lo:
            continue
        first = int(np.searchsorted(edges, lo, side="right")) - 1
        last = int(np.searchsorted(edges, hi, side="left")) - 1
        first = max(0, first)
        last = min(len(grid) - 1, last)
        if last >= first:
            out[first : last + 1] = np.maximum(out[first : last + 1], value)
    return out


# --------------------------------------------------------------------------
# Discrete events and bimanual rules
# --------------------------------------------------------------------------


def _runs(times: list[float], flags: list[bool], min_duration: float) -> list[float]:
    """Start times of runs of True lasting at least ``min_duration``."""

    starts = []
    n = len(times)
    step = (
        float(sorted(b - a for a, b in zip(times[:-1], times[1:]))[(n - 1) // 2])
        if n > 1
        else 0.0
    )
    k = 0
    while k < n:
        if not flags[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and flags[j + 1]:
            j += 1
        if times[j] - times[k] + step >= min_duration - 1e-9:
            starts.append(times[k])
        k = j + 1
    return starts


def detect_window_events(
    window: HoldWindow,
    window_index: int,
    vision: WindowVision | None,
    grid: list[float],
    collapse: Any,
    events_config: dict[str, Any],
) -> list[GraspEvent]:
    """Drop (vision or telemetry) and slip events inside one hold window."""

    min_duration = float(events_config["min_duration_sec"])
    events: list[GraspEvent] = []
    if vision is not None and vision.object_found:
        drop_flags = [
            loss >= float(events_config["drop_threshold"]) and not occ
            for loss, occ in zip(vision.attach_loss, vision.occluded)
        ]
        for start in _runs(vision.times, drop_flags, min_duration):
            events.append(GraspEvent(window.arm, "drop", start, "vision", window_index))
        slip_flags = [
            speed > float(events_config["slip_speed"])
            and loss < float(events_config["slip_attach_max"])
            for speed, loss in zip(vision.rel_motion, vision.attach_loss)
        ]
        for start in _runs(vision.times, slip_flags, min_duration):
            events.append(GraspEvent(window.arm, "slip", start, "vision", window_index))
    inside = [i for i, t in enumerate(grid) if window.t_settle <= t < window.t_open]
    if inside:
        times = [grid[i] for i in inside]
        flags = [float(collapse[i]) >= float(events_config["collapse_threshold"]) for i in inside]
        for start in _runs(times, flags, min_duration):
            events.append(GraspEvent(window.arm, "drop", start, "telemetry", window_index))
    # Merge vision and telemetry drops that refer to the same moment.
    events.sort(key=lambda event: (event.kind, event.time_sec))
    merged: list[GraspEvent] = []
    for event in events:
        if (
            merged
            and merged[-1].kind == event.kind == "drop"
            and abs(merged[-1].time_sec - event.time_sec) <= 0.5
            and merged[-1].source != event.source
        ):
            merged[-1].source = "both"
            merged[-1].time_sec = min(merged[-1].time_sec, event.time_sec)
            continue
        merged.append(event)
    return merged


def apply_transfer_rule(
    events: list[GraspEvent],
    windows: dict[str, list[HoldWindow]],
    visions: dict[str, list[WindowVision | None]],
    events_config: dict[str, Any],
) -> list[GraspEvent]:
    """Relabel a drop as ``transfer`` when the other arm settles on an object right after."""

    before = float(events_config["transfer_before_sec"])
    after = float(events_config["transfer_after_sec"])
    for event in events:
        if event.kind != "drop":
            continue
        for other_arm, other_windows in windows.items():
            if other_arm == event.arm:
                continue
            for index, other in enumerate(other_windows):
                if not (event.time_sec - before <= other.t_settle <= event.time_sec + after):
                    continue
                other_vision = visions.get(other_arm, [None] * len(other_windows))[index]
                if other_vision is not None and other_vision.status != "unavailable":
                    holds_object = other_vision.object_found
                else:
                    holds_object = not other.telemetry_empty
                if holds_object:
                    event.kind = "transfer"
                    event.details = {"receiving_arm": other_arm, "receiving_window": index}
                    break
            if event.kind == "transfer":
                break
        if event.kind == "drop":
            event.details["other_arm_holding"] = any(
                w.t_settle <= event.time_sec < w.t_open
                for arm, ws in windows.items()
                if arm != event.arm
                for w in ws
            )
    return events


# --------------------------------------------------------------------------
# Full analysis (shared by the four metric classes)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GraspAnalysis:
    available: bool
    reason: str | None
    grid: list[float] = field(default_factory=list)
    signals: dict[str, list[float]] = field(default_factory=dict)
    winning_arm: dict[str, list[str | None]] = field(default_factory=dict)
    vision_available: bool = False
    vision_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_ANALYSIS_CACHE: OrderedDict[tuple[Any, ...], GraspAnalysis] = OrderedDict()
_SEGMENTER_CACHE: dict[str, GraspSegmenter] = {}


def analyze_bundle(bundle: TrajectoryBundle, raw_config: dict[str, Any]) -> GraspAnalysis:
    """Run (or reuse) the grasp-integrity analysis for one bundle."""

    config = resolve_config(raw_config)
    key = (
        bundle.bundle_id,
        id(bundle),
        len(bundle.camera_frames),
        len(bundle.streams),
        _config_digest(config),
    )
    cached = _ANALYSIS_CACHE.get(key)
    if cached is not None:
        return cached
    analysis = _analyze(bundle, config)
    _ANALYSIS_CACHE[key] = analysis
    while len(_ANALYSIS_CACHE) > 4:
        _ANALYSIS_CACHE.popitem(last=False)
    return analysis


def _analyze(bundle: TrajectoryBundle, config: dict[str, Any]) -> GraspAnalysis:
    import numpy as np

    if not config.get("enabled", False):
        return GraspAnalysis(False, f"disabled by metrics.{CONFIG_NAME}.enabled")
    arms_config: dict[str, dict[str, Any]] = config["arms"]
    component_index = int(config["component_index"])
    cmd_t, cmd_series, error = per_arm_series(
        bundle, config["command_stream"], config["command_key"], arms_config, component_index
    )
    if error:
        return GraspAnalysis(False, f"commanded aperture unavailable: {error}")
    state_t, state_series, error = per_arm_series(
        bundle, config["state_stream"], config["state_key"], arms_config, component_index
    )
    if error:
        return GraspAnalysis(False, f"measured aperture unavailable: {error}")
    arms = [arm for arm in arms_config if arm in cmd_series and arm in state_series]
    if not arms:
        return GraspAnalysis(False, "no arm has both commanded and measured aperture")

    grid = list(state_t)
    grid_arr = np.asarray(grid)
    cmd_on_grid = {
        arm: np.interp(grid_arr, np.asarray(cmd_t), np.asarray(cmd_series[arm])) for arm in arms
    }
    empty_closed = config.get("empty_closed") or {}
    windows: dict[str, list[HoldWindow]] = {}
    collapse: dict[str, Any] = {}
    for arm in arms:
        windows[arm] = find_hold_windows(
            grid_arr,
            cmd_on_grid[arm],
            state_series[arm],
            arm=arm,
            empty_closed=float(empty_closed.get(arm, 0.0)),
            config=config,
        )
        collapse[arm] = aperture_collapse_series(grid_arr, state_series[arm], windows[arm])

    vision_config = config["vision"]
    visions: dict[str, list[WindowVision | None]] = {arm: [None] * len(windows[arm]) for arm in arms}
    vision_reason: str | None = None
    vision_available = False
    if not vision_config.get("enabled", True):
        vision_reason = "disabled by vision.enabled"
    else:
        try:
            segmenter = _get_segmenter(vision_config)
        except Exception as exc:  # noqa: BLE001 - surface any backend failure as unavailability
            segmenter = None
            vision_reason = f"segmentation backend unavailable: {type(exc).__name__}: {exc}"
        if segmenter is not None:
            for arm in arms:
                camera = str(arms_config[arm].get("camera", ""))
                frames = sorted(
                    (f for f in bundle.camera_frames if f.camera_name == camera),
                    key=lambda f: (float(f.timestamp_sec), int(f.frame_index or 0)),
                )
                if not frames:
                    continue
                vision_available = True
                for index, window in enumerate(windows[arm]):
                    try:
                        visions[arm][index] = _window_vision(
                            frames, window, camera, segmenter, vision_config
                        )
                    except Exception as exc:  # noqa: BLE001
                        visions[arm][index] = WindowVision(
                            status="error",
                            info={"error": f"{type(exc).__name__}: {exc}"},
                        )
            if not vision_available:
                vision_reason = "no wrist-camera frames for any configured arm"

    events: list[GraspEvent] = []
    for arm in arms:
        for index, window in enumerate(windows[arm]):
            events.extend(
                detect_window_events(
                    window, index, visions[arm][index], grid, collapse[arm], config["events"]
                )
            )
    events = apply_transfer_rule(events, windows, visions, config["events"])

    # Zero a transferring arm's signals from the transfer onward.
    for event in events:
        if event.kind != "transfer":
            continue
        window = windows[event.arm][event.window_index]
        collapse[event.arm][(grid_arr >= event.time_sec) & (grid_arr < window.t_open)] = 0.0
        vision = visions[event.arm][event.window_index]
        if vision is not None and vision.object_found:
            for k, t in enumerate(vision.times):
                if t >= event.time_sec:
                    vision.attach_loss[k] = 0.0
                    vision.gap[k] = 0.0
                    vision.rel_motion[k] = 0.0

    # Turn each confirmed drop into a pulse: after post_drop_pulse_sec, frames that
    # are still "lost" carry no new evidence and would otherwise form a plateau whose
    # persistence peak can land anywhere up to the gripper opening.
    pulse = float(config["events"]["post_drop_pulse_sec"])
    threshold = float(config["events"]["drop_threshold"])
    collapse_threshold = float(config["events"]["collapse_threshold"])
    for event in events:
        if event.kind != "drop":
            continue
        window = windows[event.arm][event.window_index]
        after = event.time_sec + pulse
        tail = (grid_arr >= after) & (grid_arr < window.t_open)
        lost = collapse[event.arm] >= collapse_threshold
        collapse[event.arm][tail & lost] = 0.0
        vision = visions[event.arm][event.window_index]
        if vision is not None and vision.object_found:
            for k, t in enumerate(vision.times):
                if t >= after and vision.attach_loss[k] >= threshold:
                    vision.attach_loss[k] = 0.0
                    vision.gap[k] = 0.0
                    vision.rel_motion[k] = 0.0

    per_arm: dict[str, dict[str, Any]] = {arm: {} for arm in arms}
    for arm in arms:
        per_arm[arm]["grasp_aperture_collapse"] = collapse[arm]
        for name, attr in zip(VISION_SIGNALS, ("attach_loss", "gap", "rel_motion")):
            pooled = np.zeros(len(grid))
            for window, vision in zip(windows[arm], visions[arm]):
                if vision is None or not vision.object_found or not vision.times:
                    continue
                intervals = frame_hold_intervals(vision.times, window.t_settle, window.t_open)
                pooled = np.maximum(
                    pooled, pool_onto_grid(grid, intervals, getattr(vision, attr))
                )
            per_arm[arm][name] = pooled

    signals: dict[str, list[float]] = {}
    winning: dict[str, list[str | None]] = {}
    for name in SIGNAL_NAMES:
        stack = np.vstack([per_arm[arm][name] for arm in arms])
        best = np.argmax(stack, axis=0)
        values = stack[best, np.arange(len(grid))]
        signals[name] = [float(v) for v in values]
        winning[name] = [arms[int(b)] if values[i] > 0 else None for i, b in enumerate(best)]

    window_meta = {
        arm: [
            {
                **window.to_dict(),
                "vision_status": None if visions[arm][i] is None else visions[arm][i].status,
                "vision_info": None
                if visions[arm][i] is None
                else _jsonable_info(visions[arm][i].info),
            }
            for i, window in enumerate(windows[arm])
        ]
        for arm in arms
    }
    return GraspAnalysis(
        available=True,
        reason=None,
        grid=grid,
        signals=signals,
        winning_arm=winning,
        vision_available=vision_available,
        vision_reason=vision_reason,
        metadata={
            "arms": arms,
            "hold_windows": window_meta,
            "events": [event.to_dict() for event in events],
            "config_digest": _config_digest(config),
            "empty_closed": {arm: float(empty_closed.get(arm, 0.0)) for arm in arms},
        },
    )


def _jsonable_info(info: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key != "masks"}


def _get_segmenter(vision_config: dict[str, Any]) -> GraspSegmenter:
    name = str(vision_config.get("backend", "grounded_sam2"))
    factory = _SEGMENTER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"unknown grasp segmentation backend {name!r}")
    key = name + ":" + _config_digest(vision_config)
    if key not in _SEGMENTER_CACHE:
        _SEGMENTER_CACHE[key] = factory(vision_config)
    return _SEGMENTER_CACHE[key]


def _window_vision(
    frames: list[CameraFrameRef],
    window: HoldWindow,
    camera: str,
    segmenter: GraspSegmenter,
    vision_config: dict[str, Any],
) -> WindowVision:
    import numpy as np

    window_frames = [f for f in frames if window.t_settle <= float(f.timestamp_sec) < window.t_open]
    if len(window_frames) < 2:
        return WindowVision(status="too_few_frames", info={"n_frames": len(window_frames)})

    cache_path = _window_cache_path(window_frames, window, camera, segmenter, vision_config)
    if cache_path is not None and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return WindowVision(**payload)
        except (OSError, ValueError, TypeError):
            pass

    result = _compute_window_vision(window_frames, window, camera, segmenter, vision_config)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "status": result.status,
                    "times": result.times,
                    "attach_loss": result.attach_loss,
                    "gap": result.gap,
                    "rel_motion": result.rel_motion,
                    "valid": result.valid,
                    "occluded": result.occluded,
                    "info": _jsonable_info(result.info),
                },
                default=lambda value: np.asarray(value).tolist(),
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    return result


def _window_cache_path(
    frames: list[CameraFrameRef],
    window: HoldWindow,
    camera: str,
    segmenter: GraspSegmenter,
    vision_config: dict[str, Any],
) -> Path | None:
    cache_dir = vision_config.get("cache_dir")
    if not cache_dir:
        return None
    digest = hashlib.sha256(
        json.dumps(
            {
                "version": WINDOW_CACHE_VERSION,
                "camera": camera,
                "frames": [[str(f.image_path), float(f.timestamp_sec)] for f in frames],
                "window": [window.t_settle, window.t_open],
                "vision": {k: v for k, v in vision_config.items() if k != "cache_dir"},
                "backend": segmenter.identity(),
                # Content, not just path: recalibrating must invalidate the cache.
                "finger_calibration": _finger_calibration_digest(camera, vision_config),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return Path(str(cache_dir)).expanduser() / camera / f"{digest}.json"


def _compute_window_vision(
    frames: list[CameraFrameRef],
    window: HoldWindow,
    camera: str,
    segmenter: GraspSegmenter,
    vision_config: dict[str, Any],
) -> WindowVision:
    import cv2

    paths = [Path(f.image_path) for f in frames]
    times = [float(f.timestamp_sec) for f in frames]
    first_bgr = cv2.imread(str(paths[0]))
    if first_bgr is None:
        return WindowVision(status="unreadable_frames", info={"path": str(paths[0])})
    full_h, full_w = first_bgr.shape[:2]
    work_w = min(int(vision_config["work_width"]), full_w)
    work_h = max(2, int(round(full_h * work_w / full_w)))
    work_shape = (work_h, work_w)
    sx, sy = full_w / work_w, full_h / work_h

    grays = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            return WindowVision(status="unreadable_frames", info={"path": str(path)})
        grays.append(cv2.cvtColor(cv2.resize(image, (work_w, work_h)), cv2.COLOR_BGR2GRAY))
    flows = [
        cv2.calcOpticalFlowFarneback(grays[k], grays[k + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for k in range(len(grays) - 1)
    ]

    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    finger, finger_info = _finger_for_window(first_rgb, work_shape, camera, window, segmenter, vision_config)
    if finger is None:
        return WindowVision(status=str(finger_info.pop("status")), info=finger_info)
    diagonal = math.hypot(work_h, work_w)
    band = contact_band(finger, max(1, int(round(float(vision_config["band_radius_frac"]) * diagonal))))

    # Other arm: robot parts that do not touch this arm's fingers, carried forward.
    other_seq = _other_arm_sequence(paths, times, finger, work_shape, segmenter, vision_config)

    positives = _band_points(band, int(vision_config["num_positive_points"]), str(vision_config["finger_entry_edge"]))
    negatives = _separated_maxima(finger, 2)
    candidates_full = segmenter.candidate_masks(
        first_rgb,
        [(x * sx, y * sy) for x, y in positives],
        [(x * sx, y * sy) for x, y in negatives],
        str(vision_config["object_text_prompt"]),
    )
    candidates = [_resize_mask(mask, work_shape) for mask in candidates_full]
    chosen, select_info = select_held_object(
        candidates,
        finger=finger,
        band=band,
        tip=tip_band(band, str(vision_config["finger_entry_edge"])),
        other_arm_seq=other_seq,
        grays=grays,
        vision_config=vision_config,
    )
    info = {key: value for key, value in select_info.items() if key != "masks"}
    info.update({**finger_info, "n_frames": len(frames)})
    _write_init_overlay(
        vision_config, camera, window, paths, work_shape, finger, band,
        select_info["masks"], chosen, info,
    )
    if chosen is None:
        return WindowVision(status="no_object_found", info=info)
    selected = select_info["masks"][chosen]
    init_contact = int((selected & band).sum())

    track_pos = _separated_maxima(selected, 3)
    masks_full, scores = segmenter.track(
        paths,
        [(x * sx, y * sy) for x, y in track_pos],
        [(x * sx, y * sy) for x, y in negatives],
    )
    object_masks = [_resize_mask(mask, work_shape) for mask in masks_full]
    if len(object_masks) != len(frames):
        return WindowVision(status="tracking_length_mismatch", info=info)
    result = window_signals(
        times=times,
        object_masks=object_masks,
        scores=scores,
        finger=finger,
        band=band,
        other_arm_seq=other_seq,
        flows=flows,
        t_settle=window.t_settle,
        init_contact=init_contact,
        vision_config=vision_config,
    )
    result.info.update(info)
    return result


_CALIBRATION_CACHE: dict[tuple[str, float], Any] = {}


def _load_finger_calibration(path: str | Path) -> Any:
    """Load a calibration file, cached by path and modification time."""

    from pathologist.metrics.wrist_finger_calibration import FingerCalibration

    resolved = Path(str(path)).expanduser()
    key = (str(resolved), resolved.stat().st_mtime)
    if key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[key] = FingerCalibration.load(resolved)
    return _CALIBRATION_CACHE[key]


def _finger_for_window(
    first_rgb: Any,
    work_shape: tuple[int, int],
    camera: str,
    window: HoldWindow,
    segmenter: GraspSegmenter,
    vision_config: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    """This arm's finger mask at work resolution, and where it came from.

    On failure returns (None, info) with ``info['status']`` set.
    """

    import cv2
    import numpy as np

    static_path = (vision_config.get("finger_masks") or {}).get(camera)
    if static_path:
        static = cv2.imread(str(static_path), cv2.IMREAD_GRAYSCALE)
        if static is None or not (static > 127).any():
            return None, {"status": "no_finger_mask", "finger_source": "static"}
        return _resize_mask(static > 127, work_shape), {"finger_source": "static"}

    calibration_path = (vision_config.get("finger_calibration") or {}).get(camera)
    if calibration_path:
        try:
            calibration = _load_finger_calibration(calibration_path)
        except (OSError, ValueError, KeyError) as exc:
            return None, {
                "status": "bad_finger_calibration",
                "finger_source": "calibration",
                "error": f"{type(exc).__name__}: {exc}",
            }
        # The fingers sit where the measured aperture holds them during the grasp.
        mask, center, distance = calibration.mask_for(window.state_plateau)
        return _resize_mask(mask, work_shape), {
            "finger_source": "calibration",
            "finger_bin": center,
            "finger_bin_distance": round(distance, 4),
        }

    if str(vision_config.get("finger_fallback", "none")) == "segmented":
        finger_full = segmenter.finger_mask(
            first_rgb,
            str(vision_config["finger_entry_edge"]),
            float(vision_config["finger_edge_fraction"]),
        )
        if finger_full is None or not np.asarray(finger_full).any():
            return None, {"status": "no_finger_mask", "finger_source": "segmented"}
        return _resize_mask(finger_full, work_shape), {"finger_source": "segmented"}

    return None, {"status": "no_finger_calibration", "finger_source": None}


def _finger_calibration_digest(camera: str, vision_config: dict[str, Any]) -> str | None:
    path = (vision_config.get("finger_calibration") or {}).get(camera)
    if not path:
        return None
    try:
        return _load_finger_calibration(path).digest()
    except (OSError, ValueError, KeyError):
        return "unreadable"


def calibration_inputs(
    bundle: TrajectoryBundle,
    config: dict[str, Any],
    arm: str,
    exclude_margin_sec: float = 0.5,
) -> tuple[list[Path], list[float], list[float], list[float], list[tuple[float, float]]] | None:
    """Inputs for :class:`FingerCalibrator.add_episode` for one arm of one bundle.

    Returns (frame paths, frame times, aperture times, measured apertures,
    excluded intervals), or None if the arm or its camera is missing. Hold
    windows, widened by ``exclude_margin_sec`` on both sides, are excluded so a
    held object is never averaged into the gripper.
    """

    import numpy as np

    config = resolve_config(config)
    arms_config = config["arms"]
    if arm not in arms_config:
        return None
    component = int(config["component_index"])
    cmd_t, cmd_series, error = per_arm_series(
        bundle, config["command_stream"], config["command_key"], arms_config, component
    )
    if error or arm not in cmd_series:
        return None
    state_t, state_series, error = per_arm_series(
        bundle, config["state_stream"], config["state_key"], arms_config, component
    )
    if error or arm not in state_series:
        return None
    camera = str(arms_config[arm].get("camera", ""))
    frames = sorted(
        (f for f in bundle.camera_frames if f.camera_name == camera),
        key=lambda f: (float(f.timestamp_sec), int(f.frame_index or 0)),
    )
    if not frames:
        return None
    grid = np.asarray(state_t)
    cmd = np.interp(grid, np.asarray(cmd_t), np.asarray(cmd_series[arm]))
    windows = find_hold_windows(
        grid,
        cmd,
        state_series[arm],
        arm=arm,
        empty_closed=float((config.get("empty_closed") or {}).get(arm, 0.0)),
        config=config,
    )
    exclude = [(w.t_close - exclude_margin_sec, w.t_open + exclude_margin_sec) for w in windows]
    return (
        [Path(f.image_path) for f in frames],
        [float(f.timestamp_sec) for f in frames],
        [float(v) for v in state_t],
        [float(v) for v in state_series[arm]],
        exclude,
    )


def _write_init_overlay(
    vision_config: dict[str, Any],
    camera: str,
    window: HoldWindow,
    paths: list[Path],
    work_shape: tuple[int, int],
    finger: Any,
    band: Any,
    candidates: list[Any],
    chosen: int | None,
    info: dict[str, Any],
) -> None:
    """Two panels for manual QA of held-object initialization.

    Left, the settle frame: fingers (blue), contact band (yellow), candidates
    that touched the band but were not chosen (red), chosen object (green), each
    candidate labelled with its rigidity. Right, the last frame used to judge
    rigidity, with the finger (blue) and chosen-object (green) outlines from the
    settle frame: if the choice is right, the object is still inside the green
    outline there while the background has moved.
    """

    debug_dir = vision_config.get("debug_dir")
    if not debug_dir:
        return
    import cv2
    import numpy as np

    height, width = work_shape

    def load(path: Path) -> Any:
        return cv2.resize(cv2.imread(str(path)), (width, height))

    image = load(paths[0])
    overlay = image.copy()
    overlay[finger] = (255, 0, 0)
    overlay[band] = (0, 255, 255)
    for index, mask in enumerate(candidates):
        overlay[mask] = (0, 200, 0) if index == chosen else (0, 0, 255)
    # Light fill so the scene stays visible, plus solid outlines: a tint alone is
    # ambiguous on red or green objects.
    left = cv2.addWeighted(image, 0.65, overlay, 0.35, 0.0)
    order = [i for i in range(len(candidates)) if i != chosen] + ([chosen] if chosen is not None else [])
    for index in order:
        colour = (0, 200, 0) if index == chosen else (0, 0, 255)
        contours, _ = cv2.findContours(candidates[index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(left, contours, -1, colour, 2 if index == chosen else 1)
    rigidity = info.get("rigidity") or [None] * len(candidates)
    used_rows: list[int] = []
    for index, mask in enumerate(candidates):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        score = rigidity[index] if index < len(rigidity) else None
        label = f"{index}:" + ("r=-" if score is None else f"r={score:.2f}")
        y = max(10, int(ys.min()) - 2)
        while any(abs(y - row) < 10 for row in used_rows):
            y += 10
        used_rows.append(y)
        position = (max(0, int(xs.min())), min(height - 2, y))
        colour = (0, 200, 0) if index == chosen else (0, 0, 255)
        cv2.putText(left, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.33, colour, 1, cv2.LINE_AA)

    frames = info.get("informative_frames") or []
    later = int(frames[-1]) if frames else len(paths) - 1
    right = load(paths[later])
    outlines = [(finger, (255, 0, 0))]
    if chosen is not None:
        outlines.append((candidates[chosen], (0, 200, 0)))
    for mask, colour in outlines:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(right, contours, -1, colour, 1)

    confidence = str(info.get("confidence", "none"))
    source = str(info.get("finger_source"))
    if info.get("finger_bin") is not None:
        source += f" a={window.state_plateau:.2f}->bin {float(info['finger_bin']):.2f}"
    header = np.full((16, 2 * width, 3), 255, np.uint8)
    text = f"conf={confidence} fingers={source} informative={len(frames)} | right: frame {later}"
    cv2.putText(header, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    panel = np.vstack([header, np.hstack([left, right])])

    status = "none" if chosen is None else f"pick{chosen}"
    out = Path(str(debug_dir)).expanduser() / camera
    out.mkdir(parents=True, exist_ok=True)
    # Frame caches are laid out <episode>/<camera>_<rate>/<frame>, so name by episode.
    episode = Path(paths[0]).parent.parent.name
    name = f"{episode}_{window.arm}_{window.t_settle:08.3f}_{status}_{confidence}.jpg"
    cv2.imwrite(str(out / name), np.ascontiguousarray(panel))


def _other_arm_sequence(
    paths: list[Path],
    times: list[float],
    finger: Any,
    work_shape: tuple[int, int],
    segmenter: GraspSegmenter,
    vision_config: dict[str, Any],
) -> list[Any] | None:
    """Other-arm masks per frame, sampled every ``other_arm_period_sec`` and carried forward."""

    import cv2
    import numpy as np

    period = float(vision_config["other_arm_period_sec"])
    sampled: list[int] = []
    last = -math.inf
    for k, t in enumerate(times):
        if t - last >= period - 1e-9:
            sampled.append(k)
            last = t
    robot = segmenter.robot_masks([paths[k] for k in sampled])
    if not robot:
        return None
    own = contact_band(finger, 2) | finger
    sampled_masks: dict[int, Any] = {}
    for k, mask in zip(sampled, robot):
        mask = _resize_mask(mask, work_shape)
        count, labels = cv2.connectedComponents(mask.astype(np.uint8))
        other = np.zeros(work_shape, dtype=bool)
        for label in range(1, count):
            component = labels == label
            if not (component & own).any():
                other |= component
        sampled_masks[k] = other
    sequence = []
    current = sampled_masks[sampled[0]]
    for k in range(len(times)):
        current = sampled_masks.get(k, current)
        sequence.append(current)
    return sequence


# --------------------------------------------------------------------------
# Default backend: Grounding DINO + SAM2 (transformers)
# --------------------------------------------------------------------------


class GroundedSAM2GraspSegmenter:
    """Grounding DINO boxes and SAM2 point/box prompts via ``transformers``.

    Shares the model cache with :class:`GroundedSAM2RobotArmMasker`, so the
    weights are loaded once per process.
    """

    def __init__(self, vision_config: dict[str, Any]) -> None:
        from pathologist.metrics.robot_masking import GroundedSAM2RobotArmMasker

        shared = {
            "dino_model_id": vision_config["dino_model_id"],
            "sam2_model_id": vision_config["sam2_model_id"],
            "box_threshold": vision_config["box_threshold"],
            "text_threshold": vision_config["text_threshold"],
            "mask_threshold": vision_config["mask_threshold"],
            "device": vision_config["device"],
        }
        robot_cache = vision_config.get("cache_dir")
        self._config = dict(vision_config)
        self._fingers = GroundedSAM2RobotArmMasker({**shared, "prompt": vision_config["finger_prompt"]})
        self._robot = GroundedSAM2RobotArmMasker(
            {
                **shared,
                "prompt": vision_config["robot_prompt"],
                "cache_dir": str(Path(str(robot_cache)) / "robot_masks") if robot_cache else None,
            }
        )
        self._objects = GroundedSAM2RobotArmMasker({**shared, "prompt": vision_config["object_text_prompt"]})
        self._fingers._load_models()
        self._robot._load_models()
        self._objects._load_models()
        self._torch = self._fingers._torch
        self._device = self._fingers._device
        self._dtype = self._fingers._dtype
        self._processor = self._fingers._sam2_processor
        self._model = self._fingers._sam2_model

    def identity(self) -> str:
        keys = ("dino_model_id", "sam2_model_id", "box_threshold", "text_threshold", "mask_threshold")
        return json.dumps({key: self._config[key] for key in keys}, sort_keys=True)

    def finger_mask(self, image_rgb: Any, entry_edge: str, edge_fraction: float) -> Any | None:
        from PIL import Image

        image = Image.fromarray(image_rgb)
        width, height = image.width, image.height
        margin_x, margin_y = edge_fraction * width, edge_fraction * height
        boxes = [
            box
            for box in self._fingers._detect_robot_boxes(image)
            if {
                "bottom": box[3] >= height - margin_y,
                "top": box[1] <= margin_y,
                "left": box[0] <= margin_x,
                "right": box[2] >= width - margin_x,
            }.get(entry_edge, box[3] >= height - margin_y)
        ]
        if not boxes:
            return None
        masks = self._single_frame(image, boxes=boxes)
        return _union(masks) if masks else None

    def robot_masks(self, image_paths: list[Path]) -> list[Any]:
        return list(self._robot.masks_for_paths(image_paths).masks)

    def candidate_masks(
        self,
        image_rgb: Any,
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
        text_prompt: str,
    ) -> list[Any]:
        from PIL import Image

        image = Image.fromarray(image_rgb)
        masks: list[Any] = []
        if positive_points:
            masks.extend(
                self._single_frame(
                    image, positive_points=positive_points, negative_points=negative_points
                )
            )
        boxes = self._objects._detect_robot_boxes(image)
        if boxes:
            masks.extend(self._single_frame(image, boxes=boxes))
        return masks

    def track(
        self,
        image_paths: list[Path],
        positive_points: list[tuple[float, float]],
        negative_points: list[tuple[float, float]],
    ) -> tuple[list[Any], list[float] | None]:
        from pathologist.metrics.robot_masking import _load_rgb_image

        points = [list(map(float, p)) for p in [*positive_points, *negative_points]]
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        session = self._new_session()
        masks: list[Any] = []
        scores: list[float] = []
        try:
            with self._torch.no_grad():
                for frame_idx, path in enumerate(image_paths):
                    image = _load_rgb_image(Path(path))
                    inputs = self._processor(images=image, return_tensors="pt").to(self._device)
                    if frame_idx == 0:
                        self._processor.add_inputs_to_inference_session(
                            inference_session=session,
                            frame_idx=0,
                            obj_ids=[1],
                            input_points=[[points]],
                            input_labels=[[labels]],
                            original_size=_original_size(inputs),
                        )
                    output = self._model(inference_session=session, frame=inputs.pixel_values[0])
                    per_object = _per_object_masks(
                        self._processor, output.pred_masks, inputs.original_sizes, self._config["mask_threshold"]
                    )
                    masks.append(per_object[0] if per_object else _empty_like(image))
                    logits = getattr(output, "object_score_logits", None)
                    scores.append(
                        float(logits.detach().float().flatten()[0].item()) if logits is not None else 1.0
                    )
        finally:
            self._close_session(session)
        return masks, scores

    # -- helpers ----------------------------------------------------------

    def _new_session(self) -> Any:
        return self._processor.init_video_session(
            inference_device=self._device,
            inference_state_device="cpu",
            max_vision_features_cache_size=1,
            dtype=self._dtype,
        )

    def _close_session(self, session: Any) -> None:
        session.reset_inference_session()
        if isinstance(getattr(session, "processed_frames", None), dict):
            session.processed_frames.clear()
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()

    def _single_frame(
        self,
        image: Any,
        *,
        boxes: list[list[float]] | None = None,
        positive_points: list[tuple[float, float]] | None = None,
        negative_points: list[tuple[float, float]] | None = None,
    ) -> list[Any]:
        """One mask per prompted object on a single frame."""

        session = self._new_session()
        try:
            with self._torch.no_grad():
                inputs = self._processor(images=image, return_tensors="pt").to(self._device)
                if boxes:
                    self._processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=list(range(1, len(boxes) + 1)),
                        input_boxes=[boxes],
                        original_size=_original_size(inputs),
                    )
                else:
                    negatives = [list(map(float, p)) for p in negative_points or []]
                    objects = [[list(map(float, p)), *negatives] for p in positive_points or []]
                    labels = [[1] + [0] * len(negatives) for _ in objects]
                    self._processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=list(range(1, len(objects) + 1)),
                        input_points=[objects],
                        input_labels=[labels],
                        original_size=_original_size(inputs),
                    )
                output = self._model(inference_session=session, frame=inputs.pixel_values[0])
                return _per_object_masks(
                    self._processor, output.pred_masks, inputs.original_sizes, self._config["mask_threshold"]
                )
        finally:
            self._close_session(session)


def _original_size(inputs: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in inputs.original_sizes[0].detach().cpu().tolist())


def _per_object_masks(processor: Any, pred_masks: Any, original_sizes: Any, threshold: float) -> list[Any]:
    import numpy as np

    processed = processor.post_process_masks([pred_masks], original_sizes=original_sizes, binarize=False)[0]
    array = processed.detach().cpu().float().numpy()
    if array.ndim == 2:
        return [array > threshold]
    masks = []
    for obj in array:
        obj = np.asarray(obj)
        while obj.ndim > 2:
            obj = obj.max(axis=0)
        masks.append(obj > threshold)
    return masks


def _union(masks: list[Any]) -> Any:
    import numpy as np

    out = np.zeros_like(np.asarray(masks[0], dtype=bool))
    for mask in masks:
        out |= np.asarray(mask, dtype=bool)
    return out


def _empty_like(image: Any) -> Any:
    import numpy as np

    return np.zeros((image.height, image.width), dtype=bool)


register_grasp_segmenter("grounded_sam2", GroundedSAM2GraspSegmenter)


# --------------------------------------------------------------------------
# Metric classes
# --------------------------------------------------------------------------

_DESCRIPTIONS = {
    "grasp_attach_loss": "Loss of contact between the held object and the fingers in the wrist camera (1 = detached or lost).",
    "grasp_object_gap": "Shortest distance from the held object to the fingers, as a fraction of the wrist-image diagonal.",
    "grasp_relative_motion": "Held-object image motion relative to the fingers, in wrist-image diagonals per second.",
    "grasp_aperture_collapse": "Measured aperture fall toward the command during a hold, normalized by the possible fall.",
}
_UNITS = {
    "grasp_attach_loss": "fraction",
    "grasp_object_gap": "image_diagonal_fraction",
    "grasp_relative_motion": "image_diagonals_per_sec",
    "grasp_aperture_collapse": "fraction",
}


class _GraspIntegritySignalMetric:
    name = ""

    def compute(self, context: MetricComputationContext) -> MetricResult:
        analysis = analyze_bundle(context.bundle, metric_config(context, CONFIG_NAME))
        modalities = [Modality.GRIPPER] if self.name == "grasp_aperture_collapse" else [Modality.RGB, Modality.GRIPPER]
        base_metadata = {
            "grasp_integrity_metric": True,
            "surprise_tail_interpretation": "right",
        }
        if not analysis.available:
            return self._unavailable(modalities, analysis.reason or "unavailable", base_metadata)
        if self.name in VISION_SIGNALS and not analysis.vision_available:
            return self._unavailable(
                modalities,
                analysis.vision_reason or "vision unavailable",
                {**base_metadata, **analysis.metadata},
            )
        values = analysis.signals[self.name]
        samples = [MetricSample(timestamp_sec=float(t), value=float(v)) for t, v in zip(analysis.grid, values)]
        return MetricResult(
            name=self.name,
            description=_DESCRIPTIONS[self.name],
            source_modalities=modalities,
            unit=_UNITS[self.name],
            samples=samples,
            summary=summarize(values),
            display_hint="timeseries",
            metadata={
                **base_metadata,
                "available": True,
                "winning_arm": analysis.winning_arm[self.name],
                "vision_reason": analysis.vision_reason,
                **analysis.metadata,
            },
        )

    def _unavailable(self, modalities: list[Modality], reason: str, metadata: dict[str, Any]) -> MetricResult:
        return MetricResult(
            name=self.name,
            description=_DESCRIPTIONS[self.name],
            source_modalities=modalities,
            unit=_UNITS[self.name],
            metadata={**metadata, "available": False, "unavailable_reason": reason},
        )


class GraspAttachLossMetric(_GraspIntegritySignalMetric):
    name = "grasp_attach_loss"


class GraspObjectGapMetric(_GraspIntegritySignalMetric):
    name = "grasp_object_gap"


class GraspRelativeMotionMetric(_GraspIntegritySignalMetric):
    name = "grasp_relative_motion"


class GraspApertureCollapseMetric(_GraspIntegritySignalMetric):
    name = "grasp_aperture_collapse"
