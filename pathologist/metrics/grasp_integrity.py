"""Task-agnostic bimanual grasp-integrity signals.

This module intentionally keeps the telemetry state machine and signal math
independent from the SAM 3 runtime.  The model-backed initialization and
tracking adapter can therefore fail closed without changing the definitions of
the continuous signals consumed by PRISM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import extract_vector_series, find_stream, summarize
from pathologist.schemas.common import Modality
from pathologist.schemas.metrics import MetricResult, MetricSample


ARM_NAMES = ("left", "right")
COMPONENT_NAMES = (
    "grasp_attach_loss",
    "grasp_rel_motion",
    "grasp_aperture_collapse",
)


class GraspIntegrityMetric:
    """Per-arm grasp integrity, with vision channels neutral until available."""

    name = "grasp_integrity"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = metric_config(context, self.name)
        if not bool(config.get("enabled", False)):
            return _unavailable_metric("disabled by metrics.grasp_integrity.enabled")
        arms_config = config.get("arms")
        if not isinstance(arms_config, dict):
            return _unavailable_metric("metrics.grasp_integrity.arms is required")

        command_stream = find_stream(
            context.bundle,
            [Modality.GRIPPER],
            preferred_names=["gripper_action"],
        )
        state_stream = find_stream(
            context.bundle,
            [Modality.GRIPPER],
            preferred_names=["gripper"],
        )
        command_ts, command_vectors = extract_vector_series(
            command_stream,
            ["gripper_action"],
        )
        state_ts, state_vectors = extract_vector_series(
            state_stream,
            ["gripper", "gripper_state"],
        )
        if (
            command_stream is None
            or state_stream is None
            or len(command_ts) < 2
            or len(state_ts) < 2
        ):
            return _unavailable_metric(
                "requires commanded and measured bimanual gripper streams"
            )

        command_groups = _stream_arm_indices(command_stream, command_vectors)
        state_groups = _stream_arm_indices(state_stream, state_vectors)
        if any(
            arm not in command_groups or arm not in state_groups
            for arm in ARM_NAMES
        ):
            return _unavailable_metric(
                "gripper streams must expose independent left/right vector groups"
            )

        output_hz = float(config.get("output_hz", 5.0))
        if output_hz <= 0.0:
            return _unavailable_metric("metrics.grasp_integrity.output_hz must be positive")
        grid = np.arange(
            float(command_ts[0]),
            float(command_ts[-1]) + 0.5 / output_hz,
            1.0 / output_hz,
            dtype=float,
        )
        per_arm_values: dict[str, dict[str, list[float]]] = {}
        windows_by_arm: dict[str, list[dict[str, Any]]] = {}
        window_objects_by_arm: dict[str, list[HoldWindow]] = {}
        state_by_arm: dict[str, np.ndarray] = {}
        for arm in ARM_NAMES:
            raw_arm_config = arms_config.get(arm)
            if not isinstance(raw_arm_config, dict):
                return _unavailable_metric(
                    f"metrics.grasp_integrity.arms.{arm} is required"
                )
            if "empty_closed" not in raw_arm_config:
                return _unavailable_metric(
                    f"metrics.grasp_integrity.arms.{arm}.empty_closed is required"
                )
            command = _vector_component(command_vectors, command_groups[arm])
            measured_state = _vector_component(state_vectors, state_groups[arm])
            windows = detect_hold_windows(
                command_ts,
                command,
                state_ts,
                measured_state,
                arm=arm,
                empty_closed=float(raw_arm_config["empty_closed"]),
                enter=float(config.get("enter", 0.4)),
                exit=float(config.get("exit", 0.6)),
                settle_speed_per_sec=float(
                    config.get("settle_speed_per_sec", 0.05)
                ),
                settle_duration_sec=float(
                    config.get("settle_duration_sec", 0.1)
                ),
                settle_timeout_sec=float(
                    config.get("settle_timeout_sec", 1.0)
                ),
                plateau_duration_sec=float(
                    config.get("plateau_duration_sec", 0.2)
                ),
                open_rise=float(config.get("open_rise", 0.05)),
                minimum_window_sec=float(
                    config.get("minimum_window_sec", 0.3)
                ),
                empty_width_threshold=float(
                    config.get("empty_width_threshold", 0.03)
                ),
                blind_capacity_threshold=float(
                    config.get("blind_capacity_threshold", 0.05)
                ),
            )
            collapse = np.zeros(len(grid), dtype=float)
            for window in windows:
                source_selected = (
                    (np.asarray(state_ts, dtype=float) >= window.settle_sec)
                    & (np.asarray(state_ts, dtype=float) < window.open_sec)
                )
                source_collapse = aperture_collapse(
                    np.asarray(measured_state, dtype=float)[source_selected],
                    state_plateau=window.state_plateau,
                    capacity=window.capacity,
                    telemetry_blind=window.telemetry_blind,
                )
                collapse = np.maximum(
                    collapse,
                    np.asarray(
                        max_pool_resample(
                            np.asarray(state_ts, dtype=float)[source_selected],
                            source_collapse,
                            grid,
                        ),
                        dtype=float,
                    ),
                )
            per_arm_values[arm] = {
                "grasp_attach_loss": [0.0] * len(grid),
                "grasp_rel_motion": [0.0] * len(grid),
                "grasp_aperture_collapse": collapse.tolist(),
            }
            windows_by_arm[arm] = [window.to_dict() for window in windows]
            window_objects_by_arm[arm] = windows
            state_by_arm[arm] = np.asarray(measured_state, dtype=float)

        vision_available = False
        vision_warning: str | None = None
        tracking_metadata: list[dict[str, Any]] = []
        if bool(config.get("tracking_enabled", False)):
            try:
                from pathologist.metrics.grasp_initialization import (
                    GraspObjectInitializer,
                )
                from pathologist.metrics.grasp_tracking import (
                    GraspIntegrityVisionTracker,
                    records_to_grid,
                )

                initializer = GraspObjectInitializer(
                    config=config,
                    output_dir=config.get(
                        "qa_output_dir",
                        ".cache/prism/grasp_integrity_qa",
                    ),
                )
                initialization = initializer.initialize_bundle(
                    context.bundle,
                    window_objects_by_arm,
                )
                records = GraspIntegrityVisionTracker(config=config).track_bundle(
                    context.bundle,
                    window_objects_by_arm,
                    initialization,
                    state_timestamps=state_ts,
                    state_by_arm=state_by_arm,
                )
                per_arm_values, tracking_metadata = records_to_grid(records, grid)
                vision_available = True
            except Exception as exc:
                vision_warning = f"{type(exc).__name__}: {exc}"

        component_values = {
            name: np.maximum(
                np.asarray(per_arm_values["left"][name], dtype=float),
                np.asarray(per_arm_values["right"][name], dtype=float),
            ).tolist()
            for name in COMPONENT_NAMES
        }
        parent_values = np.max(
            np.asarray([component_values[name] for name in COMPONENT_NAMES]),
            axis=0,
        ).tolist()
        return MetricResult(
            name=self.name,
            description=(
                "Per-arm grasp attachment, relative motion, and aperture collapse."
            ),
            source_modalities=[Modality.GRIPPER, Modality.RGB],
            unit="normalized_grasp_integrity",
            samples=[
                MetricSample(timestamp_sec=float(timestamp), value=float(value))
                for timestamp, value in zip(grid, parent_values)
            ],
            summary=summarize([float(value) for value in parent_values]),
            thresholds=[],
            events=[],
            display_hint="grasp_integrity_timeseries",
            metadata={
                "available": True,
                "vision_available": vision_available,
                "vision_warning": vision_warning,
                "component_values": component_values,
                "per_arm_values": per_arm_values,
                "hold_windows": windows_by_arm,
                "tracking_windows": tracking_metadata,
                "arm_reduction": "max",
                "resampling": "native-rate max pooling",
                "output_hz": output_hz,
                "runtime_task_text_used": False,
            },
        )


def grasp_integrity_component_metrics(
    grouped_metric: MetricResult,
) -> list[MetricResult]:
    """Expose the three max-over-arm grasp signals as scalar metrics."""

    raw_values = grouped_metric.metadata.get("component_values")
    available = (
        grouped_metric.metadata.get("available") is True
        and isinstance(raw_values, dict)
    )
    output: list[MetricResult] = []
    for name in COMPONENT_NAMES:
        values = raw_values.get(name) if isinstance(raw_values, dict) else None
        if available and isinstance(values, list) and len(values) == len(
            grouped_metric.samples
        ):
            numeric = [float(value) for value in values]
            output.append(
                MetricResult(
                    name=name,
                    description=f"Max-over-arms {name.replace('_', ' ')}.",
                    source_modalities=list(grouped_metric.source_modalities),
                    unit=(
                        "image_diagonals_per_second"
                        if name == "grasp_rel_motion"
                        else "normalized_grasp_integrity"
                    ),
                    samples=[
                        MetricSample(
                            timestamp_sec=float(sample.timestamp_sec),
                            value=value,
                        )
                        for sample, value in zip(grouped_metric.samples, numeric)
                    ],
                    summary=summarize(numeric),
                    thresholds=[],
                    events=[],
                    display_hint=f"{name}_timeseries",
                    metadata={
                        "available": True,
                        "parent_metric": grouped_metric.name,
                        "per_arm_values": {
                            arm: grouped_metric.metadata["per_arm_values"][arm][name]
                            for arm in ARM_NAMES
                        },
                        "hold_windows": grouped_metric.metadata.get(
                            "hold_windows",
                            {},
                        ),
                        "arm_reduction": "max",
                    },
                )
            )
        else:
            output.append(
                MetricResult(
                    name=name,
                    description=f"Max-over-arms {name.replace('_', ' ')}.",
                    source_modalities=[Modality.GRIPPER, Modality.RGB],
                    samples=[],
                    display_hint=f"{name}_timeseries",
                    metadata={
                        "available": False,
                        "unavailable_reason": (
                            "parent grasp-integrity metric is unavailable"
                        ),
                    },
                )
            )
    return output


@dataclass(frozen=True, slots=True)
class HoldWindow:
    """One command-defined hold interval for one arm."""

    arm: str
    close_intent_sec: float
    settle_sec: float
    open_sec: float
    state_plateau: float
    command_plateau: float
    empty_closed: float
    grasp_width: float
    capacity: float
    telemetry_empty: bool
    telemetry_blind: bool

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.open_sec - self.settle_sec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "close_intent_sec": self.close_intent_sec,
            "settle_sec": self.settle_sec,
            "open_sec": self.open_sec,
            "duration_sec": self.duration_sec,
            "state_plateau": self.state_plateau,
            "command_plateau": self.command_plateau,
            "empty_closed": self.empty_closed,
            "grasp_width": self.grasp_width,
            "capacity": self.capacity,
            "telemetry_empty": self.telemetry_empty,
            "telemetry_blind": self.telemetry_blind,
        }


@dataclass(frozen=True, slots=True)
class HeldObjectCandidate:
    """One task-agnostic held-object mask proposal."""

    mask: np.ndarray
    source: str
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """Held-object initialization result for one hold window."""

    candidate: HeldObjectCandidate | None
    comotion: float | None
    init_confidence: str
    n_candidates: int


def detect_hold_windows(
    command_timestamps: Sequence[float],
    command_values: Sequence[float],
    state_timestamps: Sequence[float],
    state_values: Sequence[float],
    *,
    arm: str,
    empty_closed: float,
    enter: float = 0.4,
    exit: float = 0.6,
    settle_speed_per_sec: float = 0.05,
    settle_duration_sec: float = 0.1,
    settle_timeout_sec: float = 1.0,
    plateau_duration_sec: float = 0.2,
    open_rise: float = 0.05,
    minimum_window_sec: float = 0.3,
    empty_width_threshold: float = 0.03,
    blind_capacity_threshold: float = 0.05,
) -> list[HoldWindow]:
    """Detect independent per-arm holds from command intent and measured motion."""

    if arm not in ARM_NAMES:
        raise ValueError(f"arm must be one of {ARM_NAMES}, got {arm!r}")
    command_ts, command = _clean_scalar_series(
        command_timestamps,
        command_values,
    )
    state_ts, state = _clean_scalar_series(state_timestamps, state_values)
    if len(command_ts) < 2 or len(state_ts) < 2:
        return []
    if not 0.0 <= enter < exit <= 1.0:
        raise ValueError("gripper hysteresis must satisfy 0 <= enter < exit <= 1")
    if settle_duration_sec < 0.0 or settle_timeout_sec < 0.0:
        raise ValueError("settle durations cannot be negative")
    if plateau_duration_sec <= 0.0 or minimum_window_sec < 0.0:
        raise ValueError("plateau duration must be positive")

    measured = np.interp(command_ts, state_ts, state)
    speed = _absolute_derivative(command_ts, measured)
    closed_runs = _closed_command_runs(command, enter=enter, exit=exit)
    windows: list[HoldWindow] = []
    for close_index, exit_index in closed_runs:
        close_sec = float(command_ts[close_index])
        settle_index = _settle_index(
            timestamps=command_ts,
            speed=speed,
            start_index=close_index,
            stop_index=exit_index,
            speed_threshold=settle_speed_per_sec,
            stable_duration_sec=settle_duration_sec,
            timeout_sec=settle_timeout_sec,
        )
        settle_sec = float(command_ts[settle_index])
        plateau_stop = int(
            np.searchsorted(
                command_ts,
                settle_sec + plateau_duration_sec,
                side="right",
            )
        )
        plateau_stop = max(settle_index + 1, min(plateau_stop, exit_index + 1))
        state_plateau = float(np.median(measured[settle_index:plateau_stop]))
        command_plateau = float(np.median(command[settle_index:plateau_stop]))

        open_threshold = command_plateau + open_rise
        open_index = exit_index
        for index in range(settle_index + 1, exit_index + 1):
            if float(command[index]) > open_threshold:
                open_index = index
                break
        open_sec = float(command_ts[open_index])
        if open_sec - settle_sec < minimum_window_sec:
            continue

        grasp_width = state_plateau - float(empty_closed)
        capacity = state_plateau - max(
            command_plateau,
            float(empty_closed),
        )
        windows.append(
            HoldWindow(
                arm=arm,
                close_intent_sec=close_sec,
                settle_sec=settle_sec,
                open_sec=open_sec,
                state_plateau=state_plateau,
                command_plateau=command_plateau,
                empty_closed=float(empty_closed),
                grasp_width=grasp_width,
                capacity=capacity,
                telemetry_empty=grasp_width < empty_width_threshold,
                telemetry_blind=capacity < blind_capacity_threshold,
            )
        )
    return windows


def filter_and_merge_candidates(
    candidates: Sequence[HeldObjectCandidate],
    *,
    finger_mask: np.ndarray,
    contact_band: np.ndarray,
    other_arm_mask: np.ndarray | None = None,
    minimum_area_fraction: float = 0.001,
    maximum_area_fraction: float = 0.30,
    maximum_finger_fraction: float = 0.20,
    maximum_other_arm_fraction: float = 0.50,
    minimum_contact_pixels: int = 20,
    merge_iou_threshold: float = 0.70,
) -> list[HeldObjectCandidate]:
    """Apply task-agnostic geometry filters and merge duplicate masks."""

    fingers = np.asarray(finger_mask, dtype=bool)
    band = np.asarray(contact_band, dtype=bool)
    other = (
        np.asarray(other_arm_mask, dtype=bool)
        if other_arm_mask is not None
        else np.zeros_like(fingers)
    )
    if fingers.shape != band.shape or other.shape != fingers.shape:
        raise ValueError("candidate-filter masks must share a shape")
    image_area = int(fingers.size)
    accepted: list[HeldObjectCandidate] = []
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != fingers.shape:
            continue
        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue
        area_fraction = area / image_area
        if not minimum_area_fraction <= area_fraction <= maximum_area_fraction:
            continue
        if np.count_nonzero(mask & fingers) / area > maximum_finger_fraction:
            continue
        if np.count_nonzero(mask & other) / area > maximum_other_arm_fraction:
            continue
        if np.count_nonzero(mask & band) < minimum_contact_pixels:
            continue
        accepted.append(
            HeldObjectCandidate(
                mask=mask,
                source=candidate.source,
                score=float(candidate.score),
            )
        )

    merged: list[HeldObjectCandidate] = []
    for candidate in accepted:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if mask_iou(candidate.mask, existing.mask) > merge_iou_threshold
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        existing = merged[duplicate_index]
        merged[duplicate_index] = HeldObjectCandidate(
            mask=np.asarray(existing.mask, dtype=bool)
            | np.asarray(candidate.mask, dtype=bool),
            source=f"{existing.source}+{candidate.source}",
            score=max(existing.score, candidate.score),
        )
    return merged


def select_held_object_candidate(
    candidates: Sequence[HeldObjectCandidate],
    *,
    flow_fields: Sequence[np.ndarray],
    finger_mask: np.ndarray,
    contact_band: np.ndarray,
    other_arm_masks: Sequence[np.ndarray] | None = None,
    background_motion_threshold_px: float = 1.5,
    minimum_informative_pairs: int = 3,
    minimum_comotion: float = 0.5,
) -> CandidateSelection:
    """Select the finger-contact candidate most consistent with wrist motion."""

    if not candidates:
        return CandidateSelection(None, None, "none", 0)
    fingers = np.asarray(finger_mask, dtype=bool)
    band = np.asarray(contact_band, dtype=bool)
    union = np.zeros_like(fingers)
    for candidate in candidates:
        union |= np.asarray(candidate.mask, dtype=bool)

    scores = [
        candidate_comotion(
            candidate.mask,
            flow_fields=flow_fields,
            finger_mask=fingers,
            excluded_candidates=union,
            other_arm_masks=other_arm_masks,
            background_motion_threshold_px=background_motion_threshold_px,
            minimum_informative_pairs=minimum_informative_pairs,
        )
        for candidate in candidates
    ]
    informative = any(score is not None for score in scores)
    if informative:
        eligible = [
            (index, score)
            for index, score in enumerate(scores)
            if score is not None and score >= minimum_comotion
        ]
        if not eligible:
            valid_scores = [score for score in scores if score is not None]
            return CandidateSelection(
                None,
                max(valid_scores) if valid_scores else None,
                "rejected",
                len(candidates),
            )
        index, score = max(
            eligible,
            key=lambda item: np.count_nonzero(candidates[item[0]].mask & band),
        )
        return CandidateSelection(
            candidates[index],
            float(score),
            "high",
            len(candidates),
        )

    index = max(
        range(len(candidates)),
        key=lambda item: np.count_nonzero(candidates[item].mask & band),
    )
    return CandidateSelection(candidates[index], None, "low", len(candidates))


def candidate_comotion(
    candidate_mask: np.ndarray,
    *,
    flow_fields: Sequence[np.ndarray],
    finger_mask: np.ndarray,
    excluded_candidates: np.ndarray,
    other_arm_masks: Sequence[np.ndarray] | None = None,
    background_motion_threshold_px: float = 1.5,
    minimum_informative_pairs: int = 3,
) -> float | None:
    """Measure how stationary a candidate is relative to wrist-mounted fingers."""

    candidate = np.asarray(candidate_mask, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    excluded = np.asarray(excluded_candidates, dtype=bool)
    if candidate.shape != fingers.shape or excluded.shape != fingers.shape:
        raise ValueError("co-motion masks must share a shape")
    ratios: list[float] = []
    for index, raw_flow in enumerate(flow_fields):
        flow = np.asarray(raw_flow, dtype=float)
        if flow.shape[:2] != candidate.shape or flow.shape[-1] != 2:
            raise ValueError("flow fields must have shape H x W x 2")
        other = (
            np.asarray(other_arm_masks[min(index, len(other_arm_masks) - 1)], dtype=bool)
            if other_arm_masks
            else np.zeros_like(candidate)
        )
        background = ~(fingers | other | excluded)
        if not np.any(candidate) or not np.any(fingers) or not np.any(background):
            continue
        object_relative = np.median(flow[candidate], axis=0) - np.median(
            flow[fingers],
            axis=0,
        )
        background_velocity = np.median(flow[background], axis=0)
        background_speed = float(np.linalg.norm(background_velocity))
        if background_speed <= background_motion_threshold_px:
            continue
        ratios.append(float(np.linalg.norm(object_relative)) / background_speed)
    if len(ratios) < minimum_informative_pairs:
        return None
    return float(np.clip(1.0 - np.median(ratios), 0.0, 1.0))


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return binary-mask intersection over union."""

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def aperture_collapse(
    state_values: Sequence[float],
    *,
    state_plateau: float,
    capacity: float,
    telemetry_blind: bool,
) -> np.ndarray:
    """Return normalized loss of object-supported finger aperture."""

    state = np.asarray(state_values, dtype=float)
    if telemetry_blind or capacity <= 0.0:
        return np.zeros_like(state, dtype=float)
    return np.clip((float(state_plateau) - state) / float(capacity), 0.0, 1.0)


def attachment_loss(
    object_masks: Sequence[np.ndarray],
    valid: Sequence[bool],
    contact_band: np.ndarray,
    *,
    timestamps: Sequence[float] | None = None,
    settle_sec: float | None = None,
    reference_duration_sec: float = 0.2,
    other_arm_masks: Sequence[np.ndarray] | None = None,
    occlusion_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return contact loss and the other-arm occlusion mask."""

    masks = [np.asarray(mask, dtype=bool) for mask in object_masks]
    band = np.asarray(contact_band, dtype=bool)
    if len(masks) != len(valid):
        raise ValueError("object_masks and valid must have equal length")
    if any(mask.shape != band.shape for mask in masks):
        raise ValueError("object masks and contact band must share a shape")
    if other_arm_masks is not None and len(other_arm_masks) != len(masks):
        raise ValueError("other_arm_masks must align with object_masks")
    contacts = np.asarray(
        [np.count_nonzero(mask & band) for mask in masks],
        dtype=float,
    )
    reference_indices = np.arange(min(len(masks), 1))
    if timestamps is not None and settle_sec is not None:
        ts = np.asarray(timestamps, dtype=float)
        if len(ts) != len(masks):
            raise ValueError("timestamps must align with object_masks")
        reference_indices = np.flatnonzero(
            (ts >= settle_sec) & (ts <= settle_sec + reference_duration_sec)
        )
    if len(reference_indices) == 0:
        reference_indices = np.arange(min(len(masks), 1))
    contact_ref = (
        float(np.median(contacts[reference_indices]))
        if len(reference_indices)
        else 0.0
    )
    loss = np.ones(len(masks), dtype=float)
    if contact_ref > 0.0:
        loss = 1.0 - np.minimum(1.0, contacts / contact_ref)
    loss[~np.asarray(valid, dtype=bool)] = 1.0

    occluded = np.zeros(len(masks), dtype=bool)
    band_area = int(np.count_nonzero(band))
    if other_arm_masks is not None and band_area > 0:
        occluded = np.asarray(
            [
                np.count_nonzero(np.asarray(mask, dtype=bool) & band)
                / band_area
                > occlusion_fraction
                for mask in other_arm_masks
            ],
            dtype=bool,
        )
        loss[occluded] = 0.0
    return np.clip(loss, 0.0, 1.0), occluded


def relative_motion_from_flow(
    flow: np.ndarray,
    object_mask: np.ndarray,
    finger_mask: np.ndarray,
    *,
    native_fps: float,
) -> float:
    """Return object-to-finger flow in image diagonals per second."""

    vectors = np.asarray(flow, dtype=float)
    obj = np.asarray(object_mask, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    if vectors.ndim != 3 or vectors.shape[-1] != 2:
        raise ValueError("flow must have shape H x W x 2")
    if obj.shape != vectors.shape[:2] or fingers.shape != vectors.shape[:2]:
        raise ValueError("flow and masks must share spatial dimensions")
    if native_fps <= 0.0:
        raise ValueError("native_fps must be positive")
    if not np.any(obj) or not np.any(fingers):
        return 0.0
    object_flow = np.median(vectors[obj], axis=0)
    finger_flow = np.median(vectors[fingers], axis=0)
    height, width = vectors.shape[:2]
    diagonal = float(np.hypot(width, height))
    if diagonal <= 0.0:
        return 0.0
    return float(np.linalg.norm(object_flow - finger_flow) / diagonal * native_fps)


def max_pool_resample(
    source_timestamps: Sequence[float],
    source_values: Sequence[float],
    target_timestamps: Sequence[float],
) -> list[float]:
    """Max-pool native-rate samples into consecutive PRISM grid intervals."""

    source_ts, source = _clean_scalar_series(source_timestamps, source_values)
    target = np.asarray(target_timestamps, dtype=float)
    if len(target) == 0:
        return []
    if np.any(np.diff(target) <= 0.0):
        raise ValueError("target_timestamps must be strictly increasing")
    if len(source_ts) == 0:
        return [0.0] * len(target)
    if len(target) == 1:
        return [float(np.max(source))]

    step = float(np.median(np.diff(target)))
    edges = np.concatenate([target, [target[-1] + step]])
    output: list[float] = []
    for start, stop in zip(edges, edges[1:]):
        indices = np.flatnonzero((source_ts >= start) & (source_ts < stop))
        output.append(float(np.max(source[indices])) if len(indices) else 0.0)
    return output


def contact_band_from_finger_mask(
    finger_mask: np.ndarray,
    *,
    radius_fraction: float = 0.02,
) -> np.ndarray:
    """Dilate a fixed finger mask and return only its exterior contact band."""

    mask = np.asarray(finger_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("finger_mask must be two-dimensional")
    if radius_fraction <= 0.0:
        raise ValueError("radius_fraction must be positive")
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("contact-band construction requires cv2") from exc
    height, width = mask.shape
    radius = max(1, int(round(radius_fraction * np.hypot(width, height))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    dilated = cv2.dilate(mask, kernel) > 0
    return dilated & ~(mask > 0)


def finger_mask_from_normalized_polygons(
    image_shape: tuple[int, int],
    polygons: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Rasterize hardware-calibrated finger polygons."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("finger-mask construction requires cv2") from exc
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    converted = []
    for polygon in polygons:
        points = np.asarray(
            [
                [
                    np.clip(float(point[0]), 0.0, 1.0) * (width - 1),
                    np.clip(float(point[1]), 0.0, 1.0) * (height - 1),
                ]
                for point in polygon
                if len(point) >= 2
            ],
            dtype=np.int32,
        )
        if len(points) >= 3:
            converted.append(points)
    if converted:
        cv2.fillPoly(mask, converted, 1)
    return mask > 0


def spread_mask_points(mask: np.ndarray, count: int) -> list[list[float]]:
    """Choose deterministic, spatially spread prompt points inside a mask."""

    if count <= 0:
        return []
    binary = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(binary)
    if len(coordinates) == 0:
        return []
    center = np.mean(coordinates, axis=0)
    first = int(np.argmin(np.linalg.norm(coordinates - center, axis=1)))
    selected = [coordinates[first]]
    while len(selected) < min(count, len(coordinates)):
        distances = np.min(
            np.stack(
                [
                    np.linalg.norm(coordinates - point, axis=1)
                    for point in selected
                ],
                axis=0,
            ),
            axis=0,
        )
        selected.append(coordinates[int(np.argmax(distances))])
    return [[float(point[1]), float(point[0])] for point in selected]


def interior_mask_points(mask: np.ndarray, count: int) -> list[list[float]]:
    """Choose distance-transform maxima for robust positive track prompts."""

    if count <= 0:
        return []
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("interior prompt construction requires cv2") from exc
    remaining = np.asarray(mask, dtype=np.uint8)
    if not np.any(remaining):
        return []
    points: list[list[float]] = []
    suppression_radius = max(
        1,
        int(round(0.05 * np.hypot(*remaining.shape))),
    )
    yy, xx = np.ogrid[: remaining.shape[0], : remaining.shape[1]]
    for _ in range(count):
        distance = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        _, maximum, _, location = cv2.minMaxLoc(distance)
        if maximum <= 0.0:
            break
        x, y = location
        points.append([float(x), float(y)])
        remaining[(xx - x) ** 2 + (yy - y) ** 2 <= suppression_radius**2] = 0
    return points


def dense_farneback_flows(frame_paths: Sequence[str]) -> list[np.ndarray]:
    """Return full-resolution Farneback vectors for consecutive RGB frames."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("dense flow requires cv2") from exc
    flows: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if previous is not None:
            flows.append(
                cv2.calcOpticalFlowFarneback(
                    previous,
                    image,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
            )
        previous = image
    return flows


def _clean_scalar_series(
    timestamps: Sequence[float],
    values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have equal length")
    ts = np.asarray(timestamps, dtype=float)
    x = np.asarray(values, dtype=float)
    keep = np.isfinite(ts) & np.isfinite(x)
    ts = ts[keep]
    x = x[keep]
    if len(ts) == 0:
        return ts, x
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    x = x[order]
    unique = np.concatenate([[True], np.diff(ts) > 0.0])
    return ts[unique], x[unique]


def _absolute_derivative(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=float)
    dt = np.diff(timestamps)
    dx = np.diff(values)
    derivative = np.divide(
        np.abs(dx),
        dt,
        out=np.full_like(dx, np.inf, dtype=float),
        where=dt > 0.0,
    )
    return np.concatenate([[derivative[0]], derivative])


def _closed_command_runs(
    command: np.ndarray,
    *,
    enter: float,
    exit: float,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(command):
        if float(command[index]) >= enter:
            index += 1
            continue
        start = index
        index += 1
        while index < len(command) and float(command[index]) <= exit:
            index += 1
        stop = min(index, len(command) - 1)
        runs.append((start, stop))
    return runs


def _settle_index(
    *,
    timestamps: np.ndarray,
    speed: np.ndarray,
    start_index: int,
    stop_index: int,
    speed_threshold: float,
    stable_duration_sec: float,
    timeout_sec: float,
) -> int:
    deadline = min(
        float(timestamps[stop_index]),
        float(timestamps[start_index]) + timeout_sec,
    )
    cap_index = int(np.searchsorted(timestamps, deadline, side="right") - 1)
    cap_index = max(start_index, min(cap_index, stop_index))
    for index in range(start_index, cap_index + 1):
        stable_stop_sec = float(timestamps[index]) + stable_duration_sec
        stable_stop = int(
            np.searchsorted(timestamps, stable_stop_sec, side="left")
        )
        if stable_stop > stop_index or stable_stop >= len(timestamps):
            break
        if float(timestamps[stable_stop]) + 1e-9 < stable_stop_sec:
            continue
        if np.all(speed[index : stable_stop + 1] < speed_threshold):
            return index
    return cap_index


def _stream_arm_indices(
    stream: Any,
    vectors: Sequence[Sequence[float]],
) -> dict[str, int]:
    raw_groups = stream.metadata.get("vector_groups")
    if not isinstance(raw_groups, dict):
        return {}
    width = min((len(vector) for vector in vectors), default=0)
    output: dict[str, int] = {}
    for arm in ARM_NAMES:
        raw_indices = raw_groups.get(arm)
        if not isinstance(raw_indices, (list, tuple)):
            continue
        valid = [
            int(index)
            for index in raw_indices
            if isinstance(index, int) and 0 <= index < width
        ]
        if valid:
            output[arm] = valid[0]
    return output


def _vector_component(
    vectors: Sequence[Sequence[float]],
    index: int,
) -> list[float]:
    return [float(vector[index]) for vector in vectors if index < len(vector)]


def _unavailable_metric(reason: str) -> MetricResult:
    return MetricResult(
        name=GraspIntegrityMetric.name,
        description=(
            "Per-arm grasp attachment, relative motion, and aperture collapse."
        ),
        source_modalities=[Modality.GRIPPER, Modality.RGB],
        samples=[],
        display_hint="grasp_integrity_timeseries",
        metadata={"available": False, "unavailable_reason": reason},
    )
