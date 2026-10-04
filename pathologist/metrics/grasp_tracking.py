"""Cached native-rate SAM 3 tracking for bimanual grasp-integrity signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pathologist.metrics.grasp_initialization import GraspInitializationResult
from pathologist.metrics.grasp_integrity import (
    HoldWindow,
    aperture_collapse,
    attachment_loss,
    dense_farneback_flows,
    interior_mask_points,
    max_pool_resample,
    relative_motion_from_flow,
)
from pathologist.metrics.grasp_integrity_sam3 import (
    Sam3MaskCache,
    Sam3Runtime,
)
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


@dataclass(slots=True)
class HoldSignalRecord:
    arm: str
    window_index: int
    window: HoldWindow
    timestamps: np.ndarray
    attach_loss: np.ndarray
    rel_motion: np.ndarray
    aperture_collapse: np.ndarray
    valid: np.ndarray
    occluded: np.ndarray
    cache_hit: bool
    other_arm_cache_hit: bool
    initialization_found: bool
    event: dict[str, Any] | None = None


class GraspIntegrityVisionTracker:
    """Track initialized held objects and produce the three continuous signals."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        runtime: Sam3Runtime | None = None,
    ) -> None:
        self.config = config
        sam_config = (
            config.get("sam3") if isinstance(config.get("sam3"), dict) else {}
        )
        self.score_threshold = float(
            sam_config.get("default_output_probability_threshold", 0.5)
        )
        self.runtime = runtime or Sam3Runtime(
            checkpoint_path=sam_config.get("video_checkpoint_path"),
            probability_threshold=self.score_threshold,
            version=str(sam_config.get("video_version", "sam3")),
        )
        cache_root = sam_config.get(
            "cache_dir",
            ".cache/prism/sam3_grasp_integrity",
        )
        self.cache = Sam3MaskCache(cache_root)
        self.native_fps = float(config.get("native_fps", 30.0))
        self.other_arm_hz = float(config.get("other_arm_mask_hz", 5.0))

    def track_bundle(
        self,
        bundle: TrajectoryBundle,
        windows_by_arm: dict[str, Sequence[HoldWindow]],
        initialization: Sequence[GraspInitializationResult],
        *,
        state_timestamps: Sequence[float],
        state_by_arm: dict[str, Sequence[float]],
    ) -> list[HoldSignalRecord]:
        init_by_key = {
            (result.arm, result.window_index): result for result in initialization
        }
        arms_config = self.config.get("arms")
        if not isinstance(arms_config, dict):
            raise ValueError("grasp-integrity arms config is required")
        records: list[HoldSignalRecord] = []
        for arm, windows in windows_by_arm.items():
            arm_config = arms_config.get(arm)
            if not isinstance(arm_config, dict):
                raise ValueError(f"missing grasp-integrity config for {arm}")
            camera_name = str(arm_config["wrist_camera"])
            camera_frames = sorted(
                [
                    frame
                    for frame in bundle.camera_frames
                    if frame.camera_name == camera_name
                ],
                key=lambda frame: (frame.timestamp_sec, frame.frame_index or 0),
            )
            for window_index, window in enumerate(windows):
                result = init_by_key.get((arm, window_index))
                records.append(
                    self._track_window(
                        bundle_id=bundle.bundle_id,
                        arm=arm,
                        window_index=window_index,
                        window=window,
                        frames=camera_frames,
                        initialization=result,
                        state_timestamps=state_timestamps,
                        state_values=state_by_arm[arm],
                    )
                )
        _apply_bimanual_suppression(records)
        return records

    def _track_window(
        self,
        *,
        bundle_id: str,
        arm: str,
        window_index: int,
        window: HoldWindow,
        frames: Sequence[CameraFrameRef],
        initialization: GraspInitializationResult | None,
        state_timestamps: Sequence[float],
        state_values: Sequence[float],
    ) -> HoldSignalRecord:
        selected_frames = [
            frame
            for frame in frames
            if window.settle_sec <= frame.timestamp_sec < window.open_sec
        ]
        timestamps = np.asarray(
            [frame.timestamp_sec for frame in selected_frames],
            dtype=float,
        )
        measured = (
            np.interp(timestamps, state_timestamps, state_values)
            if len(timestamps)
            else np.asarray([], dtype=float)
        )
        collapse = aperture_collapse(
            measured,
            state_plateau=window.state_plateau,
            capacity=window.capacity,
            telemetry_blind=window.telemetry_blind,
        )
        neutral = np.zeros(len(timestamps), dtype=float)
        if (
            initialization is None
            or initialization.no_object_found
            or initialization.selected_mask_path is None
            or len(selected_frames) < 2
        ):
            return HoldSignalRecord(
                arm=arm,
                window_index=window_index,
                window=window,
                timestamps=timestamps,
                attach_loss=neutral.copy(),
                rel_motion=neutral.copy(),
                aperture_collapse=collapse,
                valid=np.zeros(len(timestamps), dtype=bool),
                occluded=np.zeros(len(timestamps), dtype=bool),
                cache_hit=False,
                other_arm_cache_hit=False,
                initialization_found=False,
            )

        selected_path = Path(initialization.selected_mask_path)
        window_dir = selected_path.parent
        selected_mask = _read_mask(selected_path)
        finger_mask = _read_mask(
            window_dir / "finger_calibration" / "finger_mask.png"
        )
        contact_band = _read_mask(
            window_dir / "finger_calibration" / "contact_band.png"
        )
        positive_points = interior_mask_points(selected_mask, count=3)
        negative_points = interior_mask_points(finger_mask, count=2)
        frame_paths = [str(frame.image_path) for frame in selected_frames]
        key_payload = {
            "kind": "held_object_track",
            "bundle_id": bundle_id,
            "arm": arm,
            "window_index": window_index,
            "settle_sec": round(window.settle_sec, 6),
            "open_sec": round(window.open_sec, 6),
            "positive_points": positive_points,
            "negative_points": negative_points,
            "video_version": self.runtime.version,
            "prompt_mode": "points_then_selected_mask",
        }
        track_key = self.cache.key(key_payload)
        other_key = self.cache.key(
            {
                **key_payload,
                "kind": "other_arm_masks",
                "sample_hz": self.other_arm_hz,
            }
        )
        track_cached = self.cache.load(track_key)
        other_cached = self.cache.load(other_key)
        cache_hit = track_cached is not None
        other_cache_hit = other_cached is not None

        if track_cached is None or other_cached is None:
            track_cached, other_cached = self._run_session(
                frame_paths,
                finger_mask=finger_mask,
                selected_mask=selected_mask,
                positive_points=positive_points,
                negative_points=negative_points,
                track_key=track_key,
                other_key=other_key,
                need_track=track_cached is None,
                need_other=other_cached is None,
                existing_track=track_cached,
                existing_other=other_cached,
            )

        masks, scores = track_cached
        other_masks, _ = other_cached
        masks = _fit_mask_sequence(masks, len(timestamps), selected_mask.shape)
        other_masks = _fit_mask_sequence(
            other_masks,
            len(timestamps),
            selected_mask.shape,
        )
        scores = _fit_scores(scores, len(timestamps))
        minimum_area = max(1, int(round(0.001 * selected_mask.size)))
        # Point-prompted video objects expose no calibrated per-frame confidence:
        # SAM reports 1.0 while the object ID is present. Validity therefore uses
        # public object-ID/mask presence plus the specified minimum area.
        valid = np.count_nonzero(masks, axis=(1, 2)) >= minimum_area
        attach, occluded = attachment_loss(
            list(masks),
            valid.tolist(),
            contact_band,
            timestamps=timestamps,
            settle_sec=window.settle_sec,
            other_arm_masks=list(other_masks),
        )
        flows = dense_farneback_flows(frame_paths)
        relative = np.zeros(len(timestamps), dtype=float)
        for index, flow in enumerate(flows, start=1):
            if index < len(valid) and valid[index]:
                relative[index] = relative_motion_from_flow(
                    flow,
                    masks[index],
                    finger_mask,
                    native_fps=self.native_fps,
                )
        record = HoldSignalRecord(
            arm=arm,
            window_index=window_index,
            window=window,
            timestamps=timestamps,
            attach_loss=attach,
            rel_motion=relative,
            aperture_collapse=collapse,
            valid=valid,
            occluded=occluded,
            cache_hit=cache_hit,
            other_arm_cache_hit=other_cache_hit,
            initialization_found=True,
        )
        record.event = _derive_event(record)
        return record

    def _run_session(
        self,
        frame_paths: Sequence[str],
        *,
        finger_mask: np.ndarray,
        selected_mask: np.ndarray,
        positive_points: Sequence[Sequence[float]],
        negative_points: Sequence[Sequence[float]],
        track_key: str,
        other_key: str,
        need_track: bool,
        need_other: bool,
        existing_track: tuple[np.ndarray, np.ndarray] | None,
        existing_other: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        track_output = existing_track
        other_output = existing_other
        with self.runtime.open_session(frame_paths) as session:
            if need_other:
                other_masks = np.zeros(
                    (len(frame_paths), *finger_mask.shape),
                    dtype=bool,
                )
                stride = max(1, int(round(self.native_fps / self.other_arm_hz)))
                sampled = list(range(0, len(frame_paths), stride))
                for frame_index in sampled:
                    union = np.zeros_like(finger_mask)
                    for prompt in ("robot arm", "robot gripper"):
                        for prediction in session.prompt(
                            frame_index=frame_index,
                            text=prompt,
                            reset=True,
                        ):
                            mask = np.asarray(prediction.mask, dtype=bool)
                            if mask.shape == finger_mask.shape and not np.any(
                                mask & finger_mask
                            ):
                                union |= mask
                    stop = min(len(frame_paths), frame_index + stride)
                    other_masks[frame_index:stop] = union
                other_scores = np.ones(len(frame_paths), dtype=float)
                self.cache.save(
                    other_key,
                    masks=other_masks,
                    scores=other_scores,
                )
                other_output = (other_masks, other_scores)
            if need_track:
                session.prompt(
                    frame_index=0,
                    points_xy=[*positive_points, *negative_points],
                    point_labels=[
                        *([1] * len(positive_points)),
                        *([0] * len(negative_points)),
                    ],
                    object_id=0,
                    reset=True,
                )
                session.prompt_mask(
                    selected_mask,
                    frame_index=0,
                    object_id=0,
                )
                if self.runtime.version == "sam3.1":
                    observations = session.propagate_tracker(object_id=0)
                else:
                    observations = session.propagate(object_id=0)
                masks = np.zeros(
                    (len(frame_paths), *finger_mask.shape),
                    dtype=bool,
                )
                scores = np.zeros(len(frame_paths), dtype=float)
                for observation in observations:
                    if observation.mask is not None:
                        masks[observation.frame_index] = observation.mask
                        scores[observation.frame_index] = observation.score
                self.cache.save(track_key, masks=masks, scores=scores)
                track_output = (masks, scores)
        if track_output is None or other_output is None:
            raise RuntimeError("SAM 3 tracking session did not produce cache outputs")
        return track_output, other_output


def records_to_grid(
    records: Sequence[HoldSignalRecord],
    target_timestamps: Sequence[float],
) -> tuple[dict[str, dict[str, list[float]]], list[dict[str, Any]]]:
    """Max-pool native hold records onto the PRISM grid, independently per arm."""

    target = list(map(float, target_timestamps))
    output = {
        arm: {
            "grasp_attach_loss": [0.0] * len(target),
            "grasp_rel_motion": [0.0] * len(target),
            "grasp_aperture_collapse": [0.0] * len(target),
        }
        for arm in ("left", "right")
    }
    metadata: list[dict[str, Any]] = []
    for record in records:
        for name, values in (
            ("grasp_attach_loss", record.attach_loss),
            ("grasp_rel_motion", record.rel_motion),
            ("grasp_aperture_collapse", record.aperture_collapse),
        ):
            pooled = max_pool_resample(record.timestamps, values, target)
            output[record.arm][name] = np.maximum(
                np.asarray(output[record.arm][name], dtype=float),
                np.asarray(pooled, dtype=float),
            ).tolist()
        metadata.append(
            {
                "arm": record.arm,
                "window_index": record.window_index,
                "cache_hit": record.cache_hit,
                "other_arm_cache_hit": record.other_arm_cache_hit,
                "initialization_found": record.initialization_found,
                "valid_fraction": (
                    float(np.mean(record.valid)) if len(record.valid) else 0.0
                ),
                "occluded_fraction": (
                    float(np.mean(record.occluded)) if len(record.occluded) else 0.0
                ),
                "event": record.event,
            }
        )
    return output, metadata


def _derive_event(record: HoldSignalRecord) -> dict[str, Any] | None:
    vision_time = _first_sustained_time(
        record.timestamps,
        (record.attach_loss >= 0.8) & ~record.occluded,
        duration_sec=0.1,
    )
    telemetry_time = _first_sustained_time(
        record.timestamps,
        record.aperture_collapse >= 0.8,
        duration_sec=0.1,
    )
    drop_times = [value for value in (vision_time, telemetry_time) if value is not None]
    if drop_times:
        source = (
            "both"
            if vision_time is not None and telemetry_time is not None
            else ("vision" if vision_time is not None else "telemetry")
        )
        return {
            "label": "drop",
            "timestamp_sec": min(drop_times),
            "source": source,
        }
    slip_time = _first_sustained_time(
        record.timestamps,
        (record.rel_motion > 0.05) & (record.attach_loss < 0.5),
        duration_sec=0.1,
    )
    if slip_time is not None:
        return {
            "label": "slip",
            "timestamp_sec": slip_time,
            "source": "vision",
        }
    return None


def _apply_bimanual_suppression(records: Sequence[HoldSignalRecord]) -> None:
    for record in records:
        event = record.event
        if not event or event.get("label") != "drop":
            continue
        other_arm = "right" if record.arm == "left" else "left"
        drop_time = float(event["timestamp_sec"])
        receiver = next(
            (
                candidate
                for candidate in records
                if candidate.arm == other_arm
                and candidate.initialization_found
                and drop_time - 1.0
                <= candidate.window.settle_sec
                <= drop_time + 0.5
            ),
            None,
        )
        if receiver is None:
            continue
        event["label"] = "transfer"
        selected = record.timestamps >= drop_time
        record.attach_loss[selected] = 0.0
        record.aperture_collapse[selected] = 0.0

    transfers = [
        record
        for record in records
        if record.event and record.event.get("label") == "transfer"
    ]
    for record in transfers:
        other = next(
            (
                candidate
                for candidate in transfers
                if candidate.arm != record.arm
                and abs(
                    float(candidate.event["timestamp_sec"])
                    - float(record.event["timestamp_sec"])
                )
                <= 1.0
            ),
            None,
        )
        if other is None:
            continue
        record.event["label"] = "co_hold"
        if not np.any(record.aperture_collapse >= 0.8):
            record.attach_loss[:] = 0.0


def _first_sustained_time(
    timestamps: np.ndarray,
    active: np.ndarray,
    *,
    duration_sec: float,
) -> float | None:
    start: int | None = None
    for index, value in enumerate(np.asarray(active, dtype=bool)):
        if value and start is None:
            start = index
        if (not value or index == len(active) - 1) and start is not None:
            stop = index if not value else index + 1
            if stop > start and (
                float(timestamps[stop - 1]) - float(timestamps[start])
                >= duration_sec
            ):
                return float(timestamps[start])
            start = None
    return None


def _read_mask(path: str | Path) -> np.ndarray:
    import cv2  # type: ignore[import-not-found]

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def _fit_mask_sequence(
    masks: np.ndarray,
    count: int,
    shape: tuple[int, int],
) -> np.ndarray:
    source = np.asarray(masks, dtype=bool)
    output = np.zeros((count, *shape), dtype=bool)
    usable = min(count, len(source))
    if source.ndim == 3 and source.shape[1:] == shape and usable:
        output[:usable] = source[:usable]
    return output


def _fit_scores(scores: np.ndarray, count: int) -> np.ndarray:
    output = np.zeros(count, dtype=float)
    source = np.asarray(scores, dtype=float).reshape(-1)
    usable = min(count, len(source))
    output[:usable] = source[:usable]
    return output
