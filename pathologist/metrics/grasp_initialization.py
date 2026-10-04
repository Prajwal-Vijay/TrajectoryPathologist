"""SAM 3 held-object initialization for per-arm grasp windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pathologist.metrics.grasp_integrity import (
    CandidateSelection,
    HeldObjectCandidate,
    HoldWindow,
    contact_band_from_finger_mask,
    dense_farneback_flows,
    filter_and_merge_candidates,
    finger_mask_from_normalized_polygons,
    interior_mask_points,
    select_held_object_candidate,
    spread_mask_points,
)
from pathologist.metrics.grasp_integrity_sam3 import Sam3ImageRuntime
from pathologist.schemas.trajectory_bundle import CameraFrameRef, TrajectoryBundle


@dataclass(frozen=True, slots=True)
class GraspInitializationResult:
    """Initialization outcome and QA paths for one arm hold."""

    arm: str
    window_index: int
    settle_sec: float
    open_sec: float
    camera_name: str
    selected_mask_path: str | None
    qa_overlay_path: str
    comotion: float | None
    init_confidence: str
    n_candidates: int
    no_object_found: bool
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "window_index": self.window_index,
            "settle_sec": self.settle_sec,
            "open_sec": self.open_sec,
            "camera_name": self.camera_name,
            "selected_mask_path": self.selected_mask_path,
            "qa_overlay_path": self.qa_overlay_path,
            "comotion": self.comotion,
            "init_confidence": self.init_confidence,
            "n_candidates": self.n_candidates,
            "no_object_found": self.no_object_found,
            "frame_count": self.frame_count,
        }


class GraspObjectInitializer:
    """Initialize held objects without task or object names."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        output_dir: str | Path,
        runtime: Sam3ImageRuntime | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sam_config = (
            config.get("sam3") if isinstance(config.get("sam3"), dict) else {}
        )
        self.runtime = runtime or Sam3ImageRuntime(
            checkpoint_path=sam_config.get("image_checkpoint_path"),
            probability_threshold=float(
                sam_config.get(
                    "default_output_probability_threshold",
                    0.5,
                )
            ),
        )
        self.native_fps = float(config.get("native_fps", 30.0))
        self.contact_radius = float(
            config.get("contact_band_radius_fraction", 0.02)
        )

    def initialize_bundle(
        self,
        bundle: TrajectoryBundle,
        windows_by_arm: dict[str, Sequence[HoldWindow]],
    ) -> list[GraspInitializationResult]:
        """Initialize every left/right hold window in one trajectory."""

        output: list[GraspInitializationResult] = []
        arms = self.config.get("arms")
        if not isinstance(arms, dict):
            raise ValueError("grasp-integrity arms config is required")
        for arm, windows in windows_by_arm.items():
            arm_config = arms.get(arm)
            if not isinstance(arm_config, dict):
                raise ValueError(f"missing grasp-integrity config for {arm}")
            camera_name = str(arm_config["wrist_camera"])
            frames = sorted(
                [
                    frame
                    for frame in bundle.camera_frames
                    if frame.camera_name == camera_name
                ],
                key=lambda frame: (frame.timestamp_sec, frame.frame_index or 0),
            )
            _validate_native_rate(frames, expected_fps=self.native_fps)
            for index, window in enumerate(windows):
                result = self._initialize_window(
                    bundle_id=bundle.bundle_id,
                    arm=arm,
                    arm_config=arm_config,
                    camera_name=camera_name,
                    frames=frames,
                    window=window,
                    window_index=index,
                )
                output.append(result)
        return output

    def _initialize_window(
        self,
        *,
        bundle_id: str,
        arm: str,
        arm_config: dict[str, Any],
        camera_name: str,
        frames: Sequence[CameraFrameRef],
        window: HoldWindow,
        window_index: int,
    ) -> GraspInitializationResult:
        window_dir = (
            self.output_dir
            / bundle_id
            / arm
            / f"hold_{window_index:04d}_{window.settle_sec:.3f}"
        )
        window_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = window_dir / "initialization.json"
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            return GraspInitializationResult(**payload)

        selected_frames = [
            frame
            for frame in frames
            if window.settle_sec <= frame.timestamp_sec < window.open_sec
        ]
        if len(selected_frames) < 2:
            raise ValueError(
                f"{bundle_id}/{arm} hold at {window.settle_sec:.3f}s "
                "does not have native-rate wrist frames"
            )
        frame_paths = [str(frame.image_path) for frame in selected_frames]
        first_image = _read_image(frame_paths[0])
        height, width = first_image.shape[:2]
        polygons = arm_config.get("finger_polygons_normalized")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(
                f"{arm} finger_polygons_normalized must be manually calibrated"
            )
        shift_fraction = float(
            arm_config.get("finger_aperture_shift_fraction", 0.0)
        )
        adjusted_polygons = _aperture_adjusted_finger_polygons(
            polygons,
            aperture=window.state_plateau,
            shift_fraction=shift_fraction,
        )
        finger_mask = finger_mask_from_normalized_polygons(
            (height, width),
            adjusted_polygons,
        )
        full_contact_band = contact_band_from_finger_mask(
            finger_mask,
            radius_fraction=self.contact_radius,
        )
        contact_band = _inner_finger_gap_band(
            full_contact_band,
            finger_mask=finger_mask,
        )
        _store_hardware_masks(
            window_dir / "finger_calibration",
            first_image,
            finger_mask,
            contact_band,
        )

        init_indices = [
            _nearest_frame_index(selected_frames, window.settle_sec + offset)
            for offset in (0.0, 0.1, 0.2)
        ]
        other_masks: list[np.ndarray] = []
        candidates: list[HeldObjectCandidate] = []
        positive_points = _fingertip_band_points(
            contact_band,
            finger_mask=finger_mask,
            count=5,
        )
        negative_points = interior_mask_points(finger_mask, count=2)
        for frame_index in init_indices:
            image = _read_image(frame_paths[frame_index])
            point_prompts = [
                (
                    [positive, *negative_points],
                    [1, *([0] * len(negative_points))],
                )
                for positive in positive_points
            ]
            text_output, point_output = self.runtime.analyze_image(
                image,
                text_prompts=["robot arm", "robot gripper", "object", "item", "tool"],
                point_prompts=point_prompts,
            )
            frame_other = np.zeros((height, width), dtype=bool)
            for prompt in ("robot arm", "robot gripper"):
                for prediction in text_output[prompt]:
                    mask = np.asarray(prediction.mask, dtype=bool)
                    if mask.shape == finger_mask.shape and not np.any(
                        mask & finger_mask
                    ):
                        frame_other |= mask
            other_masks.append(frame_other)

            for positive_index, predictions in enumerate(point_output):
                for prediction in predictions:
                    candidates.append(
                        HeldObjectCandidate(
                            mask=prediction.mask,
                            source=f"point:f{frame_index}:p{positive_index}",
                            score=prediction.score,
                        )
                    )
            for prompt in ("object", "item", "tool"):
                for prediction in text_output[prompt]:
                    candidates.append(
                        HeldObjectCandidate(
                            mask=prediction.mask,
                            source=f"text:{prompt}:f{frame_index}",
                            score=prediction.score,
                        )
                    )

        other_union = np.logical_or.reduce(other_masks)
        candidates = filter_and_merge_candidates(
            candidates,
            finger_mask=finger_mask,
            contact_band=contact_band,
            other_arm_mask=other_union,
        )
        comotion_stop = max(
            2,
            int(round(min(0.5, window.duration_sec) * self.native_fps)) + 1,
        )
        flow_paths = frame_paths[:comotion_stop]
        flows = dense_farneback_flows(flow_paths)
        selection = select_held_object_candidate(
            candidates,
            flow_fields=flows,
            finger_mask=finger_mask,
            contact_band=contact_band,
            other_arm_masks=other_masks,
        )
        selected_mask_path: str | None = None
        if selection.candidate is not None:
            selected_path = window_dir / "selected_mask.png"
            _write_mask(selected_path, selection.candidate.mask)
            selected_mask_path = str(selected_path)
        overlay_path = window_dir / "initialization_overlay.jpg"
        _write_initialization_overlay(
            overlay_path,
            first_image,
            finger_mask=finger_mask,
            contact_band=contact_band,
            other_arm_mask=other_masks[0],
            selected_mask=(
                selection.candidate.mask
                if selection.candidate is not None
                else None
            ),
            positive_points=positive_points,
            negative_points=negative_points,
            selection=selection,
        )
        result = GraspInitializationResult(
            arm=arm,
            window_index=window_index,
            settle_sec=window.settle_sec,
            open_sec=window.open_sec,
            camera_name=camera_name,
            selected_mask_path=selected_mask_path,
            qa_overlay_path=str(overlay_path),
            comotion=selection.comotion,
            init_confidence=selection.init_confidence,
            n_candidates=selection.n_candidates,
            no_object_found=selection.candidate is None,
            frame_count=len(selected_frames),
        )
        metadata_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result


def _validate_native_rate(
    frames: Sequence[CameraFrameRef],
    *,
    expected_fps: float,
) -> None:
    if len(frames) < 3:
        raise ValueError("wrist camera requires at least three frames")
    timestamps = np.asarray([frame.timestamp_sec for frame in frames], dtype=float)
    positive = np.diff(timestamps)
    positive = positive[positive > 0.0]
    if len(positive) == 0:
        raise ValueError("wrist camera timestamps are not increasing")
    measured_fps = 1.0 / float(np.median(positive))
    if abs(measured_fps - expected_fps) > max(1.0, expected_fps * 0.05):
        raise ValueError(
            f"wrist camera is {measured_fps:.3f} Hz; expected native "
            f"{expected_fps:.3f} Hz"
        )


def _nearest_frame_index(
    frames: Sequence[CameraFrameRef],
    timestamp_sec: float,
) -> int:
    return min(
        range(len(frames)),
        key=lambda index: abs(frames[index].timestamp_sec - timestamp_sec),
    )


def _aperture_adjusted_finger_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    *,
    aperture: float,
    shift_fraction: float,
) -> list[list[list[float]]]:
    """Translate left/right jaw templates using measured normalized aperture."""

    shift = max(0.0, float(aperture)) * max(0.0, float(shift_fraction))
    output: list[list[list[float]]] = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("finger polygons must contain normalized [x, y] points")
        direction = -1.0 if float(np.mean(points[:, 0])) < 0.5 else 1.0
        adjusted = points.copy()
        adjusted[:, 0] = np.clip(
            adjusted[:, 0] + direction * shift,
            0.0,
            1.0,
        )
        output.append(adjusted.tolist())
    return output


def _fingertip_band_points(
    contact_band: np.ndarray,
    *,
    finger_mask: np.ndarray,
    count: int,
    inset_fraction: float = 0.015,
) -> list[list[float]]:
    """Spread prompts just inside the object-facing contact band."""

    band = np.asarray(contact_band, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    if band.shape != fingers.shape:
        raise ValueError("contact band and finger mask must have the same shape")
    height, _ = band.shape
    yy = np.indices(band.shape)[0]
    inner_band = _inner_finger_gap_band(
        band,
        finger_mask=fingers,
    ) & (yy <= 0.85 * height)
    points = spread_mask_points(inner_band, count)
    if len(points) != count:
        points = spread_mask_points(band, count)
    inset = max(0.0, float(inset_fraction)) * band.shape[1]
    center_x = band.shape[1] / 2.0
    return [
        [
            float(np.clip(x + np.sign(center_x - x) * inset, 0, band.shape[1] - 1)),
            float(y),
        ]
        for x, y in points
    ]


def _inner_finger_gap_band(
    contact_band: np.ndarray,
    *,
    finger_mask: np.ndarray,
) -> np.ndarray:
    """Keep contact pixels on the two jaw faces that oppose each other."""

    band = np.asarray(contact_band, dtype=bool)
    fingers = np.asarray(finger_mask, dtype=bool)
    if band.shape != fingers.shape:
        raise ValueError("contact band and finger mask must have the same shape")
    _, width = band.shape
    center_x = width // 2
    gap = np.zeros_like(band)
    for y in range(band.shape[0]):
        left = np.flatnonzero(fingers[y, :center_x])
        right = np.flatnonzero(fingers[y, center_x:]) + center_x
        if len(left) == 0 or len(right) == 0:
            continue
        left_edge = int(left[-1])
        right_edge = int(right[0])
        if left_edge + 1 < right_edge:
            gap[y, left_edge + 1 : right_edge] = True
    return band & gap


def _read_image(path: str | Path) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("grasp initialization requires cv2") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read wrist image: {path}")
    return image


def _write_mask(path: Path, mask: np.ndarray) -> None:
    import cv2  # type: ignore[import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)


def _store_hardware_masks(
    root: Path,
    image: np.ndarray,
    finger_mask: np.ndarray,
    contact_band: np.ndarray,
) -> None:
    import cv2  # type: ignore[import-not-found]

    root.mkdir(parents=True, exist_ok=True)
    finger_path = root / "finger_mask.png"
    band_path = root / "contact_band.png"
    overlay_path = root / "finger_contact_qa.jpg"
    if not finger_path.is_file():
        _write_mask(finger_path, finger_mask)
    if not band_path.is_file():
        _write_mask(band_path, contact_band)
    if not overlay_path.is_file():
        overlay = image.copy()
        overlay[finger_mask] = (
            0.4 * overlay[finger_mask] + 0.6 * np.asarray([255, 0, 0])
        ).astype(np.uint8)
        overlay[contact_band] = (
            0.3 * overlay[contact_band] + 0.7 * np.asarray([0, 255, 255])
        ).astype(np.uint8)
        cv2.imwrite(str(overlay_path), overlay)


def _write_initialization_overlay(
    path: Path,
    image: np.ndarray,
    *,
    finger_mask: np.ndarray,
    contact_band: np.ndarray,
    other_arm_mask: np.ndarray,
    selected_mask: np.ndarray | None,
    positive_points: Sequence[Sequence[float]],
    negative_points: Sequence[Sequence[float]],
    selection: CandidateSelection,
) -> None:
    import cv2  # type: ignore[import-not-found]

    overlay = image.copy()
    colors = [
        (finger_mask, np.asarray([255, 0, 0])),
        (contact_band, np.asarray([0, 255, 255])),
        (other_arm_mask, np.asarray([255, 0, 255])),
    ]
    if selected_mask is not None:
        colors.append((np.asarray(selected_mask, dtype=bool), np.asarray([0, 255, 0])))
    for mask, color in colors:
        overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
    for x, y in positive_points:
        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            5,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    for x, y in negative_points:
        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
    text = (
        f"init={selection.init_confidence} candidates={selection.n_candidates} "
        f"comotion={selection.comotion if selection.comotion is not None else 'NA'}"
    )
    cv2.putText(
        overlay,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
