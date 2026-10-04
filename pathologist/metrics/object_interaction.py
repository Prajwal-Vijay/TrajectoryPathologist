"""Object-centric motion metrics from Grounded SAM2 and CoTracker3."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import extract_vector_series, summarize
from pathologist.metrics.robot_masking import (
    DEFAULT_DINO_MODEL,
    DEFAULT_SAM2_MODEL,
    GroundedSAM2RobotArmMasker,
    _empty_masks,
    _load_rgb_image,
)
from pathologist.metrics.vision import _group_configured_frames
from pathologist.schemas.common import Modality, TimeWindow
from pathologist.schemas.metrics import MetricResult, MetricSample
from pathologist.schemas.trajectory_bundle import CameraFrameRef


LEGACY_COMPONENT_NAMES = (
    "object_residual_speed",
    "gripper_object_relative_speed",
    "gripper_object_separation_speed",
)
COMPONENT_NAMES = (
    *LEGACY_COMPONENT_NAMES,
    "unexplained_object_motion",
    "unexplained_object_motion_grasped",
    "object_visibility_drop",
    "object_visibility_drop_grasped",
)
COMPONENT_DESCRIPTIONS = {
    "object_residual_speed": (
        "Camera-compensated manipulated-object residual speed."
    ),
    "gripper_object_relative_speed": (
        "Camera-compensated relative speed between gripper and object."
    ),
    "gripper_object_separation_speed": (
        "Positive gripper-object separation speed along their connecting axis."
    ),
    "unexplained_object_motion": (
        "Object motion not explained by camera-compensated gripper motion."
    ),
    "unexplained_object_motion_grasped": (
        "Unexplained object motion gated to settled closed-gripper intervals."
    ),
    "object_visibility_drop": (
        "Positive loss of tracked-object visibility relative to a fixed time lag."
    ),
    "object_visibility_drop_grasped": (
        "Tracked-object visibility loss gated to settled closed-gripper intervals."
    ),
}
COMPONENT_UNITS = {
    "object_residual_speed": "image_diagonals_per_second",
    "gripper_object_relative_speed": "image_diagonals_per_second",
    "gripper_object_separation_speed": "image_diagonals_per_second",
    "unexplained_object_motion": "image_diagonals_per_second",
    "unexplained_object_motion_grasped": "image_diagonals_per_second",
    "object_visibility_drop": "visible_fraction_drop",
    "object_visibility_drop_grasped": "visible_fraction_drop",
}
_COTRACKER_CACHE: dict[tuple[str, str, str], Any] = {}
DEFAULT_COTRACKER_REPO = (
    "facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
)


@dataclass(slots=True)
class InteractionMaskBatch:
    """Separate propagated gripper and manipulated-object masks."""

    gripper_masks: list[Any]
    object_masks: list[Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class InteractionMotionRecord:
    """One camera-local interaction-motion observation."""

    timestamp_sec: float
    start_sec: float
    components: dict[str, float]
    homography_inlier_fraction: float
    visible_object_point_count: int
    visible_gripper_point_count: int
    visible_background_point_count: int
    object_track_valid: bool = True
    object_visibility_fraction: float = 1.0


class GroundedSAM2InteractionSegmenter:
    """Initialize gripper/object boxes with Grounding DINO and propagate with SAM2."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        shared = {
            "dino_model_id": config.get("dino_model_id", DEFAULT_DINO_MODEL),
            "sam2_model_id": config.get("sam2_model_id", DEFAULT_SAM2_MODEL),
            "box_threshold": config.get("box_threshold", 0.3),
            "text_threshold": config.get("text_threshold", 0.25),
            "mask_threshold": config.get("mask_threshold", 0.0),
            "duplicate_iou_threshold": config.get("duplicate_iou_threshold", 0.8),
            "chunk_size_frames": config.get("chunk_size_frames", 32),
            "device": config.get("device", "auto"),
        }
        raw_cache_dir = config.get("cache_dir")
        cache_dir = Path(str(raw_cache_dir)).expanduser().resolve() if raw_cache_dir else None
        gripper_prompt = config.get("gripper_prompt", "robot gripper.")
        object_prompt = config.get(
            "object_prompt",
            "object held by robot. manipulated object. tool. component. part.",
        )
        self.max_object_area_fraction = float(
            config.get("max_object_area_fraction", 0.35)
        )
        self.max_gripper_iou = float(config.get("max_gripper_iou", 0.7))
        self.object_ambiguity_margin_fraction = max(
            0.0,
            float(config.get("object_ambiguity_margin_fraction", 0.02)),
        )
        selection_digest = hashlib.sha256(
            json.dumps(
                {
                    "gripper_prompt": gripper_prompt,
                    "object_prompt": object_prompt,
                    "max_object_area_fraction": self.max_object_area_fraction,
                    "max_gripper_iou": self.max_gripper_iou,
                    "object_ambiguity_margin_fraction": (
                        self.object_ambiguity_margin_fraction
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        gripper_config = {
            **shared,
            "prompt": gripper_prompt,
            "cache_dir": str(cache_dir / "gripper") if cache_dir else None,
        }
        object_config = {
            **shared,
            "prompt": object_prompt,
            "cache_dir": (
                str(cache_dir / "object" / selection_digest)
                if cache_dir
                else None
            ),
        }
        self.gripper_masker = GroundedSAM2RobotArmMasker(gripper_config)
        self.object_masker = GroundedSAM2RobotArmMasker(object_config)
        self.chunk_size_frames = max(2, int(config.get("chunk_size_frames", 32)))
        self._previous_object_path: Path | None = None
        self._previous_object_mask: Any | None = None

    def reset_sequence(self) -> None:
        """Clear mask continuity before processing a different camera sequence."""

        self._previous_object_path = None
        self._previous_object_mask = None

    def masks_for_paths(self, image_paths: list[str | Path]) -> InteractionMaskBatch:
        """Return propagated gripper and nearest-object masks for one video chunk."""

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        if not resolved_paths:
            return InteractionMaskBatch([], [], {"available": False})

        gripper_batch = self.gripper_masker.masks_for_paths(resolved_paths)
        cached_object_masks = self.object_masker._read_cached_masks(resolved_paths)
        chunk_object_counts: list[int] = []
        selected_object_boxes: list[list[float] | None] = []
        continuity_seeded_chunks = 0
        if cached_object_masks is not None:
            object_masks = cached_object_masks
            cache_hit = True
        else:
            self.object_masker._load_models()
            object_masks = []
            cache_hit = False
            for start in range(0, len(resolved_paths), self.chunk_size_frames):
                chunk_paths = resolved_paths[start:start + self.chunk_size_frames]
                first_image = _load_rgb_image(chunk_paths[0])
                selected_box = None
                if (
                    self._previous_object_path == chunk_paths[0]
                    and self._previous_object_mask is not None
                ):
                    selected_box = _mask_bounding_box(self._previous_object_mask)
                    continuity_seeded_chunks += int(selected_box is not None)
                if selected_box is None:
                    object_boxes = self.object_masker._detect_robot_boxes(first_image)
                    chunk_object_counts.append(len(object_boxes))
                    gripper_box = _mask_bounding_box(gripper_batch.masks[start])
                    selected_box = _select_nearest_object_box(
                        object_boxes,
                        gripper_box=gripper_box,
                        image_width=first_image.width,
                        image_height=first_image.height,
                        max_area_fraction=self.max_object_area_fraction,
                        max_gripper_iou=self.max_gripper_iou,
                        ambiguity_margin_px=(
                            self.object_ambiguity_margin_fraction
                            * math.hypot(first_image.width, first_image.height)
                        ),
                    )
                selected_object_boxes.append(selected_box)
                if selected_box is None:
                    object_masks.extend(
                        _empty_masks(
                            count=len(chunk_paths),
                            height=first_image.height,
                            width=first_image.width,
                        )
                    )
                else:
                    object_masks.extend(
                        self.object_masker._track_boxes(chunk_paths, [selected_box])
                    )
            self.object_masker._write_cached_masks(resolved_paths, object_masks)

        import numpy as np

        separated_object_masks = [
            np.asarray(object_mask, dtype=bool)
            & ~np.asarray(gripper_mask, dtype=bool)
            for object_mask, gripper_mask in zip(
                object_masks,
                gripper_batch.masks,
            )
        ]
        if separated_object_masks:
            self._previous_object_path = resolved_paths[-1]
            self._previous_object_mask = separated_object_masks[-1]
        return InteractionMaskBatch(
            gripper_masks=[np.asarray(mask, dtype=bool) for mask in gripper_batch.masks],
            object_masks=separated_object_masks,
            metadata={
                "available": True,
                "method": "grounding_dino_nearest_object_plus_sam2",
                "gripper": gripper_batch.metadata,
                "object_prompt": self.object_masker.prompt,
                "object_mask_cache_hit": cache_hit,
                "chunk_object_detection_counts": chunk_object_counts,
                "selected_object_boxes": selected_object_boxes,
                "object_selection": "minimum_box_gap_to_gripper",
                "object_ambiguity_margin_fraction": (
                    self.object_ambiguity_margin_fraction
                ),
                "continuity_seeded_chunk_count": continuity_seeded_chunks,
            },
        )


class CoTrackerRunner:
    """Lazy torch-hub wrapper for frozen CoTracker3 offline inference."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.repo = str(config.get("repo", DEFAULT_COTRACKER_REPO))
        self.model_name = str(config.get("model_name", "cotracker3_offline"))
        self.requested_device = str(config.get("device", "auto"))
        self._torch = None
        self._device = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        self._torch = torch
        self._device = _resolve_torch_device(torch, self.requested_device)
        key = (self.repo, self.model_name, str(self._device))
        model = _COTRACKER_CACHE.get(key)
        if model is None:
            model = torch.hub.load(
                self.repo,
                self.model_name,
                trust_repo=True,
            ).to(self._device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            _COTRACKER_CACHE[key] = model
        self._model = model

    def track(self, video_rgb: Any, query_points_xy: Any) -> tuple[Any, Any]:
        """Track first-frame query points through an RGB uint8 video."""

        import numpy as np

        self.load()
        video = np.asarray(video_rgb)
        points = np.asarray(query_points_xy, dtype=np.float32)
        video_tensor = (
            self._torch.from_numpy(video)
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self._device)
        )
        queries = np.concatenate(
            [np.zeros((len(points), 1), dtype=np.float32), points],
            axis=1,
        )
        query_tensor = self._torch.from_numpy(queries)[None].to(self._device)
        with self._torch.inference_mode():
            tracks, visibility = self._model(
                video_tensor,
                queries=query_tensor,
                backward_tracking=False,
            )
        track_values = tracks[0].detach().float().cpu().numpy()
        visibility_values = visibility[0].detach().cpu().numpy()
        if visibility_values.ndim == 3 and visibility_values.shape[-1] == 1:
            visibility_values = visibility_values[..., 0]
        return track_values, visibility_values.astype(bool)

    @property
    def device(self) -> str:
        return str(self._device or self.requested_device)


class ObjectInteractionMotionMetric:
    """Camera-compensated object and gripper relative motion."""

    name = "object_interaction_motion"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = metric_config(context, self.name)
        enabled = bool(config.get("enabled", False))
        camera_names = _configured_camera_names(config)
        interaction_config = (
            config.get("interaction")
            if isinstance(config.get("interaction"), dict)
            else {}
        )
        gating_config = (
            interaction_config.get("gating")
            if isinstance(interaction_config.get("gating"), dict)
            else (
                config.get("gating")
                if isinstance(config.get("gating"), dict)
                else {}
            )
        )
        sample_period_sec = float(config.get("sample_period_sec", 0.5))
        chunk_size_frames = max(3, int(config.get("chunk_size_frames", 32)))
        tracker_resize = _configured_resize(config.get("tracker_resize", [384, 384]))
        object_point_count = max(1, int(config.get("object_point_count", 48)))
        gripper_point_count = max(1, int(config.get("gripper_point_count", 24)))
        background_point_count = max(4, int(config.get("background_point_count", 96)))
        ransac_threshold_px = float(config.get("ransac_threshold_px", 3.0))
        min_background_inliers = max(
            8,
            int(config.get("min_background_inliers", 12)),
        )
        background_point_count = max(
            background_point_count,
            min_background_inliers,
        )
        min_homography_inlier_fraction = min(
            1.0,
            max(
                0.0,
                float(config.get("min_homography_inlier_fraction", 0.5)),
            ),
        )
        foreground_erosion_px = max(
            0,
            int(config.get("foreground_erosion_px", 2)),
        )
        background_exclusion_dilation_px = max(
            0,
            int(config.get("background_exclusion_dilation_px", 6)),
        )
        max_camera_alignment_gap_sec = max(
            0.0,
            float(
                config.get(
                    "max_camera_alignment_gap_sec",
                    max(0.25, 2.0 * sample_period_sec),
                )
            ),
        )
        min_visible_object_points = max(
            1,
            int(config.get("min_visible_object_points", 8)),
        )
        min_visible_gripper_points = max(
            1,
            int(config.get("min_visible_gripper_points", 8)),
        )
        visibility_lag_sec = max(
            0.0,
            float(config.get("visibility_lag_sec", 0.5)),
        )

        if not enabled:
            return _unavailable_result(
                camera_names,
                reason="disabled by metrics.object_interaction_motion.enabled",
            )
        if sample_period_sec <= 0:
            raise ValueError(
                "object_interaction_motion.sample_period_sec must be positive"
            )

        grouped_frames = _group_configured_frames(
            context.bundle.camera_frames,
            camera_names,
        )
        if not grouped_frames:
            return _unavailable_result(
                camera_names,
                reason="no frames found for configured camera_names",
            )

        segmentation_config = config.get("segmentation", {})
        tracking_config = config.get("tracking", {})
        segmenter = GroundedSAM2InteractionSegmenter(
            dict(segmentation_config) if isinstance(segmentation_config, dict) else {}
        )
        tracker = CoTrackerRunner(
            dict(tracking_config) if isinstance(tracking_config, dict) else {}
        )
        try:
            tracker.load()
        except Exception as exc:
            return _unavailable_result(
                camera_names,
                matched_camera_names=sorted(grouped_frames),
                reason=f"CoTracker unavailable: {type(exc).__name__}: {exc}",
            )

        records_by_camera: dict[str, list[InteractionMotionRecord]] = {}
        segmentation_metadata: dict[str, list[dict[str, Any]]] = {}
        camera_errors: dict[str, list[str]] = {}
        image_read_failures: list[str] = []
        for camera_name, frames in grouped_frames.items():
            segmenter.reset_sequence()
            sampled_frames = _sample_frames_by_period(frames, sample_period_sec)
            try:
                records, mask_metadata = _compute_camera_records(
                    sampled_frames,
                    segmenter=segmenter,
                    tracker=tracker,
                    tracker_resize=tracker_resize,
                    chunk_size_frames=chunk_size_frames,
                    object_point_count=object_point_count,
                    gripper_point_count=gripper_point_count,
                    background_point_count=background_point_count,
                    ransac_threshold_px=ransac_threshold_px,
                    min_background_inliers=min_background_inliers,
                    min_homography_inlier_fraction=(
                        min_homography_inlier_fraction
                    ),
                    min_visible_object_points=min_visible_object_points,
                    min_visible_gripper_points=min_visible_gripper_points,
                    visibility_lag_sec=visibility_lag_sec,
                    foreground_erosion_px=foreground_erosion_px,
                    background_exclusion_dilation_px=(
                        background_exclusion_dilation_px
                    ),
                    image_read_failures=image_read_failures,
                )
            except Exception as exc:
                camera_errors.setdefault(camera_name, []).append(
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if records:
                records_by_camera[camera_name] = records
            segmentation_metadata[camera_name] = mask_metadata

        if not records_by_camera:
            return _unavailable_result(
                camera_names,
                matched_camera_names=sorted(grouped_frames),
                reason="no camera produced valid object/gripper/background tracks",
                camera_errors=camera_errors,
                image_read_failures=image_read_failures,
            )

        (
            samples,
            sample_camera_names,
            component_series,
            camera_aggregation_metadata,
        ) = _aggregate_camera_records(
            records_by_camera,
            max_alignment_gap_sec=max_camera_alignment_gap_sec,
        )
        if not samples:
            return _unavailable_result(
                camera_names,
                matched_camera_names=sorted(records_by_camera),
                reason="camera component alignment produced no samples",
            )

        gating_mask, gating_metadata = _interaction_grasp_mask(
            context,
            [float(sample.timestamp_sec) for sample in samples],
            gating_config,
        )
        ungated_pairs = (
            (
                "unexplained_object_motion",
                "unexplained_object_motion_grasped",
            ),
            (
                "object_visibility_drop",
                "object_visibility_drop_grasped",
            ),
        )
        for ungated_name, gated_name in ungated_pairs:
            source_values = component_series[ungated_name]
            component_series[gated_name] = [
                float(value) * float(gate)
                for value, gate in zip(source_values, gating_mask)
            ]
        per_camera_series = camera_aggregation_metadata.get(
            "per_camera_series",
            {},
        )
        if isinstance(per_camera_series, dict):
            for camera_series in per_camera_series.values():
                if not isinstance(camera_series, dict):
                    continue
                camera_timestamps = camera_series.get("timestamps_sec", [])
                camera_components = camera_series.get("components", {})
                if (
                    not isinstance(camera_timestamps, list)
                    or not isinstance(camera_components, dict)
                ):
                    continue
                camera_mask, _ = _interaction_grasp_mask(
                    context,
                    [float(value) for value in camera_timestamps],
                    gating_config,
                )
                for ungated_name, gated_name in ungated_pairs:
                    values_for_camera = camera_components.get(
                        ungated_name,
                        [],
                    )
                    if isinstance(values_for_camera, list):
                        camera_components[gated_name] = [
                            float(value) * float(gate)
                            for value, gate in zip(
                                values_for_camera,
                                camera_mask,
                            )
                        ]
        values = [
            max(component_series[name][index] for name in COMPONENT_NAMES)
            for index in range(len(samples))
        ]
        for sample, value in zip(samples, values):
            sample.value = value

        diagnostics = [
            record
            for records in records_by_camera.values()
            for record in records
        ]
        return MetricResult(
            name=self.name,
            description=(
                "Grounded-SAM2 object/gripper segmentation with CoTracker3 point "
                "tracks and background-RANSAC homography compensation."
            ),
            source_modalities=[Modality.RGB],
            unit="image_diagonals_per_second",
            samples=samples,
            summary=summarize(values),
            thresholds=[],
            events=[],
            display_hint="object_interaction_motion_timeseries",
            metadata={
                "available": True,
                "configured_camera_names": camera_names,
                "matched_camera_names": sorted(records_by_camera),
                "sample_period_sec": sample_period_sec,
                "chunk_size_frames": chunk_size_frames,
                "tracker_resize": list(tracker_resize),
                "object_point_count": object_point_count,
                "gripper_point_count": gripper_point_count,
                "background_point_count": background_point_count,
                "ransac_threshold_px": ransac_threshold_px,
                "min_background_inliers": min_background_inliers,
                "min_homography_inlier_fraction": (
                    min_homography_inlier_fraction
                ),
                "foreground_erosion_px": foreground_erosion_px,
                "background_exclusion_dilation_px": (
                    background_exclusion_dilation_px
                ),
                "max_camera_alignment_gap_sec": max_camera_alignment_gap_sec,
                "min_visible_object_points": min_visible_object_points,
                "min_visible_gripper_points": min_visible_gripper_points,
                "visibility_lag_sec": visibility_lag_sec,
                "cotracker_device": tracker.device,
                "ecod_component_values": component_series,
                "sample_camera_names": sample_camera_names,
                "component_camera_names": {
                    name: list(sample_camera_names) for name in COMPONENT_NAMES
                },
                "component_reduction": "max_after_independent_ecod",
                "camera_aggregation": camera_aggregation_metadata.get(
                    "camera_aggregation"
                ),
                "camera_reference_timeline": camera_aggregation_metadata.get(
                    "camera_reference_timeline"
                ),
                "per_camera_series": camera_aggregation_metadata.get(
                    "per_camera_series",
                    {},
                ),
                "object_track_valid": camera_aggregation_metadata.get(
                    "object_track_valid",
                    [],
                ),
                "gating_available": gating_metadata["gating_available"],
                "gating_config": gating_metadata["gating_config"],
                "gating_stream_name": gating_metadata.get("gating_stream_name"),
                "gated_in_fraction": gating_metadata["gated_in_fraction"],
                "grasp_closed_mask": [bool(value) for value in gating_mask],
                "gating_warning": gating_metadata.get("warning"),
                "segmentation_metadata": segmentation_metadata,
                "camera_errors": camera_errors,
                "image_read_failures": image_read_failures,
                "mean_homography_inlier_fraction": _mean_or_none(
                    record.homography_inlier_fraction for record in diagnostics
                ),
                "mean_visible_object_point_count": _mean_or_none(
                    record.visible_object_point_count for record in diagnostics
                ),
                "mean_visible_gripper_point_count": _mean_or_none(
                    record.visible_gripper_point_count for record in diagnostics
                ),
                "mean_visible_background_point_count": _mean_or_none(
                    record.visible_background_point_count for record in diagnostics
                ),
                "model_training": "pretrained_frozen_no_task_finetuning",
            },
        )


def object_interaction_component_metrics(
    grouped_metric: MetricResult,
) -> list[MetricResult]:
    """Expose interaction components as independently scoreable scalar metrics."""

    component_values = grouped_metric.metadata.get("ecod_component_values")
    available = (
        grouped_metric.metadata.get("available") is True
        and isinstance(component_values, dict)
    )
    output: list[MetricResult] = []
    for component_name in COMPONENT_NAMES:
        values = (
            component_values.get(component_name)
            if isinstance(component_values, dict)
            else None
        )
        aligned_values = (
            [float(value) for value in values]
            if isinstance(values, (list, tuple))
            and len(values) == len(grouped_metric.samples)
            else []
        )
        samples = [
            MetricSample(
                timestamp_sec=source.timestamp_sec,
                value=value,
                window=source.window,
            )
            for source, value in zip(grouped_metric.samples, aligned_values)
        ]
        component_available = available and bool(samples)
        metadata = {
            "available": component_available,
            "parent_metric_name": grouped_metric.name,
            "component_name": component_name,
            "surprise_tail_interpretation": "right",
            "object_track_valid": grouped_metric.metadata.get(
                "object_track_valid",
                [],
            ),
        }
        if component_available:
            metadata.update(
                {
                    "sample_camera_names": grouped_metric.metadata.get(
                        "sample_camera_names",
                        [],
                    ),
                    "model_training": grouped_metric.metadata.get(
                        "model_training"
                    ),
                    "gating_available": grouped_metric.metadata.get(
                        "gating_available"
                    ),
                    "gating_config": grouped_metric.metadata.get(
                        "gating_config"
                    ),
                    "gated_in_fraction": grouped_metric.metadata.get(
                        "gated_in_fraction"
                    ),
                }
            )
        else:
            metadata["unavailable_reason"] = grouped_metric.metadata.get(
                "unavailable_reason",
                "parent object-interaction component is unavailable",
            )
        output.append(
            MetricResult(
                name=component_name,
                description=COMPONENT_DESCRIPTIONS[component_name],
                source_modalities=(
                    [Modality.RGB, Modality.GRIPPER]
                    if component_name.endswith("_grasped")
                    else [Modality.RGB]
                ),
                unit=COMPONENT_UNITS[component_name],
                samples=samples,
                summary=summarize(
                    [float(sample.value) for sample in samples]
                ),
                thresholds=[],
                events=[],
                display_hint=f"{component_name}_timeseries",
                metadata=metadata,
            )
        )
    return output


def _aggregate_camera_records(
    records_by_camera: dict[str, list[InteractionMotionRecord]],
    *,
    max_alignment_gap_sec: float,
) -> tuple[list[MetricSample], list[str], dict[str, list[float]], dict[str, Any]]:
    """Select one camera per timestamp and keep all components from that camera."""

    camera_names = sorted(
        camera_name
        for camera_name, records in records_by_camera.items()
        if records
    )
    component_series = {name: [] for name in COMPONENT_NAMES}
    if not camera_names:
        return [], [], component_series, {
            "camera_aggregation": "scalar_max_same_camera_components",
            "camera_reference_timeline": None,
            "camera_sample_counts": {},
            "per_camera_series": {},
            "object_track_valid": [],
        }
    ordered_records = {
        camera_name: sorted(
            records_by_camera[camera_name],
            key=lambda record: record.timestamp_sec,
        )
        for camera_name in camera_names
    }
    reference_camera = max(
        camera_names,
        key=lambda camera_name: len(ordered_records[camera_name]),
    )
    samples: list[MetricSample] = []
    winning_cameras: list[str] = []
    winning_track_validity: list[bool] = []
    for reference_record in ordered_records[reference_camera]:
        timestamp = float(reference_record.timestamp_sec)
        candidates: list[tuple[float, str, dict[str, float]]] = []
        for camera_name in camera_names:
            components = _interpolated_record_components(
                ordered_records[camera_name],
                timestamp,
                max_alignment_gap_sec=max_alignment_gap_sec,
            )
            if components is None:
                continue
            candidates.append(
                (
                    max(components[name] for name in COMPONENT_NAMES),
                    camera_name,
                    components,
                )
            )
        if not candidates:
            continue
        scalar_value, winning_camera, winning_components = max(
            candidates,
            key=lambda candidate: candidate[0],
        )
        samples.append(
            MetricSample(
                timestamp_sec=timestamp,
                value=scalar_value,
                window=TimeWindow(
                    reference_record.start_sec,
                    reference_record.timestamp_sec,
                ),
            )
        )
        winning_cameras.append(winning_camera)
        winning_track_validity.append(
            _interpolated_record_track_validity(
                ordered_records[winning_camera],
                timestamp,
                max_alignment_gap_sec=max_alignment_gap_sec,
            )
        )
        for component_name in COMPONENT_NAMES:
            component_series[component_name].append(
                float(winning_components[component_name])
            )
    return samples, winning_cameras, component_series, {
        "camera_aggregation": "scalar_max_same_camera_components",
        "camera_reference_timeline": reference_camera,
        "camera_sample_counts": {
            camera_name: len(records) for camera_name, records in ordered_records.items()
        },
        "object_track_valid": winning_track_validity,
        "per_camera_series": {
            camera_name: {
                "timestamps_sec": [
                    float(record.timestamp_sec) for record in records
                ],
                "components": {
                    name: [
                        _record_component(record, name) for record in records
                    ]
                    for name in COMPONENT_NAMES
                },
                "object_track_valid": [
                    bool(record.object_track_valid) for record in records
                ],
                "object_visibility_fraction": [
                    float(record.object_visibility_fraction)
                    for record in records
                ],
            }
            for camera_name, records in ordered_records.items()
        },
    }


def _interpolated_record_components(
    records: list[InteractionMotionRecord],
    timestamp_sec: float,
    *,
    max_alignment_gap_sec: float,
) -> dict[str, float] | None:
    if not records:
        return None
    timestamps = [float(record.timestamp_sec) for record in records]
    right = bisect_left(timestamps, timestamp_sec)
    if right < len(records) and timestamps[right] == timestamp_sec:
        return {
            name: _record_component(records[right], name)
            for name in COMPONENT_NAMES
        }
    if right == 0 or right >= len(records):
        return None
    left = right - 1
    left_gap = timestamp_sec - timestamps[left]
    right_gap = timestamps[right] - timestamp_sec
    if (
        left_gap > max_alignment_gap_sec
        or right_gap > max_alignment_gap_sec
    ):
        return None
    span = timestamps[right] - timestamps[left]
    if span <= 0:
        return None
    weight = left_gap / span
    return {
        name: (
            _record_component(records[left], name)
            + weight
            * (
                _record_component(records[right], name)
                - _record_component(records[left], name)
            )
        )
        for name in COMPONENT_NAMES
    }


def _record_component(record: InteractionMotionRecord, name: str) -> float:
    if name in record.components:
        return float(record.components[name])
    ungated_name = name.removesuffix("_grasped")
    return float(record.components.get(ungated_name, 0.0))


def _interpolated_record_track_validity(
    records: list[InteractionMotionRecord],
    timestamp_sec: float,
    *,
    max_alignment_gap_sec: float,
) -> bool:
    """Return conservative validity for an exact or interpolated camera sample."""

    if not records:
        return False
    timestamps = [float(record.timestamp_sec) for record in records]
    right = bisect_left(timestamps, timestamp_sec)
    if right < len(records) and timestamps[right] == timestamp_sec:
        return bool(records[right].object_track_valid)
    if right == 0 or right >= len(records):
        return False
    left = right - 1
    if (
        timestamp_sec - timestamps[left] > max_alignment_gap_sec
        or timestamps[right] - timestamp_sec > max_alignment_gap_sec
    ):
        return False
    return bool(
        records[left].object_track_valid
        and records[right].object_track_valid
    )


def camera_compensated_motion_records(
    tracks_xy: Any,
    visibility: Any,
    *,
    object_indices: Any,
    gripper_indices: Any,
    background_indices: Any,
    timestamps_sec: list[float],
    frame_width: int,
    frame_height: int,
    ransac_threshold_px: float,
    min_background_inliers: int = 8,
    min_homography_inlier_fraction: float = 0.5,
    min_visible_object_points: int = 8,
    min_visible_gripper_points: int = 8,
    visibility_lag_sec: float = 0.5,
    object_masks: list[Any] | None = None,
    gripper_masks: list[Any] | None = None,
) -> list[InteractionMotionRecord]:
    """Compute robust object/gripper motion after background homography removal."""

    import cv2
    import numpy as np

    tracks = np.asarray(tracks_xy, dtype=np.float64)
    visible = np.asarray(visibility, dtype=bool)
    object_indices = np.asarray(object_indices, dtype=int)
    gripper_indices = np.asarray(gripper_indices, dtype=int)
    background_indices = np.asarray(background_indices, dtype=int)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("tracks_xy must have shape [time, points, 2]")
    if visible.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape [time, points]")
    if len(timestamps_sec) != tracks.shape[0]:
        raise ValueError("timestamps_sec must align with track time dimension")

    diagonal = max(1.0, math.hypot(frame_width, frame_height))
    object_visibility_fraction = (
        visible[:, object_indices].mean(axis=1)
        if len(object_indices)
        else np.zeros(tracks.shape[0], dtype=float)
    )
    timestamp_values = np.asarray(timestamps_sec, dtype=float)
    lag_indices = np.searchsorted(
        timestamp_values,
        timestamp_values - visibility_lag_sec,
        side="left",
    )
    lag_indices = np.clip(lag_indices, 0, len(timestamp_values) - 1)
    visibility_drops = np.maximum(
        0.0,
        object_visibility_fraction[lag_indices] - object_visibility_fraction,
    )
    records: list[InteractionMotionRecord] = []
    for time_index in range(1, tracks.shape[0]):
        dt = float(timestamps_sec[time_index] - timestamps_sec[time_index - 1])
        if dt <= 0:
            continue
        pair_visible = visible[time_index - 1] & visible[time_index]
        object_valid = object_indices[pair_visible[object_indices]]
        gripper_valid = gripper_indices[pair_visible[gripper_indices]]
        background_valid = background_indices[pair_visible[background_indices]]
        if object_masks is not None:
            object_valid = _indices_inside_mask(
                tracks[time_index],
                object_valid,
                object_masks[time_index],
            )
        if gripper_masks is not None:
            gripper_valid = _indices_inside_mask(
                tracks[time_index],
                gripper_valid,
                gripper_masks[time_index],
            )
        if len(background_valid) < min_background_inliers:
            continue

        homography, inlier_fraction = _background_homography(
            tracks[time_index - 1, background_valid],
            tracks[time_index, background_valid],
            ransac_threshold_px=ransac_threshold_px,
            min_inliers=min_background_inliers,
            min_inlier_fraction=min_homography_inlier_fraction,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if homography is None:
            continue
        object_residual_speed = 0.0
        relative_speed = 0.0
        separation_speed = 0.0
        unexplained_object_motion = 0.0
        object_track_valid = bool(
            visible[time_index - 1, object_indices].sum()
            >= min_visible_object_points
            and visible[time_index, object_indices].sum()
            >= min_visible_object_points
            and visible[time_index - 1, gripper_indices].sum()
            >= min_visible_gripper_points
            and visible[time_index, gripper_indices].sum()
            >= min_visible_gripper_points
            and len(object_valid) >= min_visible_object_points
            and len(gripper_valid) >= min_visible_gripper_points
        )
        if len(object_valid) >= 2:
            previous_object = _warp_points(
                tracks[time_index - 1, object_valid],
                homography,
            )
            current_object = tracks[time_index, object_valid]
            object_residuals = current_object - previous_object
            object_velocity = np.median(object_residuals, axis=0) / dt
            object_residual_speed = float(
                np.median(np.linalg.norm(object_residuals, axis=1))
                / dt
                / diagonal
            )
        else:
            previous_object = np.empty((0, 2), dtype=float)
            object_velocity = np.zeros(2, dtype=float)
        if len(gripper_valid) >= 2:
            previous_gripper = _warp_points(
                tracks[time_index - 1, gripper_valid],
                homography,
            )
            current_gripper = tracks[time_index, gripper_valid]
            gripper_residuals = current_gripper - previous_gripper
            gripper_velocity = np.median(gripper_residuals, axis=0) / dt
        else:
            previous_gripper = np.empty((0, 2), dtype=float)
            gripper_velocity = np.zeros(2, dtype=float)
        if len(object_valid) >= 2 and len(gripper_valid) >= 2:
            stabilized_axis = (
                np.median(previous_object, axis=0)
                - np.median(previous_gripper, axis=0)
            )
            axis_norm = float(np.linalg.norm(stabilized_axis))
            relative_velocity = object_velocity - gripper_velocity
            relative_speed = float(
                np.linalg.norm(relative_velocity) / diagonal
            )
            separation_speed = (
                max(
                    0.0,
                    float(
                        np.dot(
                            relative_velocity,
                            stabilized_axis / axis_norm,
                        )
                    ),
                )
                / diagonal
                if axis_norm > 1e-8
                else 0.0
            )
            if object_track_valid:
                unexplained_object_motion = (
                    unexplained_object_motion_from_velocities(
                        object_velocity,
                        gripper_velocity,
                        diagonal=diagonal,
                    )
                )
        records.append(
            InteractionMotionRecord(
                timestamp_sec=float(timestamps_sec[time_index]),
                start_sec=float(timestamps_sec[time_index - 1]),
                components={
                    "object_residual_speed": object_residual_speed,
                    "gripper_object_relative_speed": relative_speed,
                    "gripper_object_separation_speed": separation_speed,
                    "unexplained_object_motion": unexplained_object_motion,
                    "unexplained_object_motion_grasped": (
                        unexplained_object_motion
                    ),
                    "object_visibility_drop": float(
                        visibility_drops[time_index]
                    ),
                    "object_visibility_drop_grasped": float(
                        visibility_drops[time_index]
                    ),
                },
                homography_inlier_fraction=inlier_fraction,
                visible_object_point_count=len(object_valid),
                visible_gripper_point_count=len(gripper_valid),
                visible_background_point_count=len(background_valid),
                object_track_valid=object_track_valid,
                object_visibility_fraction=float(
                    object_visibility_fraction[time_index]
                ),
            )
        )
    return records


def unexplained_object_motion_from_velocities(
    object_velocity: Any,
    gripper_velocity: Any,
    *,
    diagonal: float = 1.0,
) -> float:
    """Return motion requiring both object movement and gripper disagreement."""

    import numpy as np

    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("diagonal must be positive and finite")
    object_values = np.asarray(object_velocity, dtype=float)
    gripper_values = np.asarray(gripper_velocity, dtype=float)
    if (
        object_values.shape != gripper_values.shape
        or object_values.ndim != 1
        or not np.isfinite(object_values).all()
        or not np.isfinite(gripper_values).all()
    ):
        return 0.0
    object_speed = float(np.linalg.norm(object_values)) / diagonal
    relative_speed = (
        float(np.linalg.norm(object_values - gripper_values)) / diagonal
    )
    return min(object_speed, relative_speed)


def _compute_camera_records(
    frames: list[CameraFrameRef],
    *,
    segmenter: GroundedSAM2InteractionSegmenter,
    tracker: CoTrackerRunner,
    tracker_resize: tuple[int, int],
    chunk_size_frames: int,
    object_point_count: int,
    gripper_point_count: int,
    background_point_count: int,
    ransac_threshold_px: float,
    min_background_inliers: int,
    min_homography_inlier_fraction: float,
    min_visible_object_points: int,
    min_visible_gripper_points: int,
    visibility_lag_sec: float,
    foreground_erosion_px: int,
    background_exclusion_dilation_px: int,
    image_read_failures: list[str],
) -> tuple[list[InteractionMotionRecord], list[dict[str, Any]]]:
    import cv2
    import numpy as np

    records: list[InteractionMotionRecord] = []
    segmentation_metadata: list[dict[str, Any]] = []
    step = max(2, chunk_size_frames - 1)
    for start in range(0, max(0, len(frames) - 1), step):
        chunk_frames = frames[start:start + chunk_size_frames]
        if len(chunk_frames) < 2:
            continue
        images_rgb = []
        readable = True
        for frame in chunk_frames:
            image = cv2.imread(str(frame.image_path))
            if image is None:
                image_read_failures.append(str(frame.image_path))
                readable = False
                break
            resized = cv2.resize(image, tracker_resize)
            images_rgb.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        if not readable:
            continue

        masks = segmenter.masks_for_paths(
            [frame.image_path for frame in chunk_frames]
        )
        segmentation_metadata.append(masks.metadata)
        object_masks = [
            cv2.resize(
                np.asarray(mask, dtype=np.uint8),
                tracker_resize,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks.object_masks
        ]
        gripper_masks = [
            cv2.resize(
                np.asarray(mask, dtype=np.uint8),
                tracker_resize,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks.gripper_masks
        ]
        if not object_masks or not gripper_masks:
            continue
        object_sampling_mask = _erode_mask(
            object_masks[0],
            foreground_erosion_px,
        )
        gripper_sampling_mask = _erode_mask(
            gripper_masks[0],
            foreground_erosion_px,
        )
        object_points = _sample_mask_points(
            object_sampling_mask,
            object_point_count,
        )
        gripper_points = _sample_mask_points(
            gripper_sampling_mask,
            gripper_point_count,
        )
        foreground = object_masks[0] | gripper_masks[0]
        foreground = _dilate_mask(
            foreground,
            background_exclusion_dilation_px,
        )
        background_points = _sample_mask_points(
            ~foreground,
            background_point_count,
            border_fraction=0.05,
        )
        if len(object_points) < 2 or len(gripper_points) < 2 or len(background_points) < 4:
            continue
        query_points = np.concatenate(
            [object_points, gripper_points, background_points],
            axis=0,
        )
        tracks, visibility = tracker.track(
            np.asarray(images_rgb, dtype=np.uint8),
            query_points,
        )
        object_end = len(object_points)
        gripper_end = object_end + len(gripper_points)
        records.extend(
            camera_compensated_motion_records(
                tracks,
                visibility,
                object_indices=np.arange(0, object_end),
                gripper_indices=np.arange(object_end, gripper_end),
                background_indices=np.arange(gripper_end, len(query_points)),
                timestamps_sec=[
                    float(frame.timestamp_sec) for frame in chunk_frames
                ],
                frame_width=tracker_resize[0],
                frame_height=tracker_resize[1],
                ransac_threshold_px=ransac_threshold_px,
                min_background_inliers=min_background_inliers,
                min_homography_inlier_fraction=(
                    min_homography_inlier_fraction
                ),
                min_visible_object_points=min_visible_object_points,
                min_visible_gripper_points=min_visible_gripper_points,
                visibility_lag_sec=visibility_lag_sec,
                object_masks=object_masks,
                gripper_masks=gripper_masks,
            )
        )
    return records, segmentation_metadata


def _background_homography(
    previous_points: Any,
    current_points: Any,
    *,
    ransac_threshold_px: float,
    min_inliers: int,
    min_inlier_fraction: float,
    frame_width: int,
    frame_height: int,
) -> tuple[Any | None, float]:
    import cv2
    import numpy as np

    previous = np.asarray(previous_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    if len(previous) < min_inliers or len(current) != len(previous):
        return None, 0.0
    homography, inliers = cv2.findHomography(
        previous,
        current,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
    )
    if (
        homography is None
        or not np.isfinite(homography).all()
        or abs(float(homography[2, 2])) < 1e-12
    ):
        return None, 0.0
    homography = homography / homography[2, 2]
    inlier_fraction = (
        float(np.asarray(inliers, dtype=bool).mean())
        if inliers is not None and len(inliers)
        else 0.0
    )
    inlier_count = int(np.asarray(inliers, dtype=bool).sum()) if inliers is not None else 0
    if inlier_count < min_inliers or inlier_fraction < min_inlier_fraction:
        return None, inlier_fraction
    if not np.isfinite(np.linalg.cond(homography)) or np.linalg.cond(homography) > 1e8:
        return None, inlier_fraction
    corners = np.asarray(
        [
            [0.0, 0.0],
            [float(frame_width - 1), 0.0],
            [float(frame_width - 1), float(frame_height - 1)],
            [0.0, float(frame_height - 1)],
        ],
        dtype=np.float64,
    )
    warped_corners = _warp_points(corners, homography)
    if not np.isfinite(warped_corners).all():
        return None, inlier_fraction
    frame_area = max(1.0, float((frame_width - 1) * (frame_height - 1)))
    warped_area_ratio = abs(float(cv2.contourArea(warped_corners.astype("float32")))) / frame_area
    if not 0.2 <= warped_area_ratio <= 5.0:
        return None, inlier_fraction
    max_extent = 4.0 * max(frame_width, frame_height)
    if float(np.abs(warped_corners).max()) > max_extent:
        return None, inlier_fraction
    return homography, inlier_fraction


def _warp_points(points: Any, homography: Any) -> Any:
    import cv2
    import numpy as np

    values = np.asarray(points, dtype=np.float64).reshape(1, -1, 2)
    return cv2.perspectiveTransform(values, homography).reshape(-1, 2)


def _indices_inside_mask(
    points: Any,
    indices: Any,
    mask: Any,
) -> Any:
    import numpy as np

    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return indices
    values = np.asarray(points)[indices]
    height, width = np.asarray(mask).shape[:2]
    x = np.rint(values[:, 0]).astype(int)
    y = np.rint(values[:, 1]).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    inside = np.zeros(len(indices), dtype=bool)
    inside[valid] = np.asarray(mask, dtype=bool)[y[valid], x[valid]]
    return indices[inside]


def _sample_mask_points(
    mask: Any,
    count: int,
    *,
    border_fraction: float = 0.0,
) -> Any:
    import numpy as np

    values = np.asarray(mask, dtype=bool).copy()
    height, width = values.shape
    border_y = int(round(height * border_fraction))
    border_x = int(round(width * border_fraction))
    if border_y > 0:
        values[:border_y] = False
        values[-border_y:] = False
    if border_x > 0:
        values[:, :border_x] = False
        values[:, -border_x:] = False
    coordinates_yx = np.argwhere(values)
    if len(coordinates_yx) <= count:
        chosen = coordinates_yx
    else:
        chosen = coordinates_yx[
            np.linspace(0, len(coordinates_yx) - 1, count, dtype=int)
        ]
    if len(chosen) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return chosen[:, [1, 0]].astype(np.float32)


def _erode_mask(mask: Any, radius_px: int) -> Any:
    if radius_px <= 0:
        return mask
    import cv2
    import numpy as np

    kernel_size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.erode(np.asarray(mask, dtype=np.uint8), kernel).astype(bool)


def _dilate_mask(mask: Any, radius_px: int) -> Any:
    if radius_px <= 0:
        return mask
    import cv2
    import numpy as np

    kernel_size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel).astype(bool)


def _mask_bounding_box(mask: Any) -> list[float] | None:
    import numpy as np

    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if len(x) == 0:
        return None
    return [
        float(x.min()),
        float(y.min()),
        float(x.max() + 1),
        float(y.max() + 1),
    ]


def _select_nearest_object_box(
    boxes: list[list[float]],
    *,
    gripper_box: list[float] | None,
    image_width: int,
    image_height: int,
    max_area_fraction: float,
    max_gripper_iou: float,
    ambiguity_margin_px: float = 0.0,
) -> list[float] | None:
    valid_boxes = [
        box
        for box in boxes
        if _box_area(box) / max(1.0, image_width * image_height)
        <= max_area_fraction
        and (
            gripper_box is None
            or _box_iou(box, gripper_box) <= max_gripper_iou
        )
    ]
    if not valid_boxes:
        return None
    if gripper_box is None:
        return valid_boxes[0] if len(valid_boxes) == 1 else None
    ranked = sorted(
        valid_boxes,
        key=lambda box: (
            _box_gap(box, gripper_box),
            -_box_area(box),
        ),
    )
    if (
        len(ranked) > 1
        and _box_gap(ranked[1], gripper_box)
        - _box_gap(ranked[0], gripper_box)
        <= ambiguity_margin_px
    ):
        return None
    return ranked[0]


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(first: list[float], second: list[float]) -> float:
    intersection = _box_area(
        [
            max(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            min(first[3], second[3]),
        ]
    )
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _box_gap(first: list[float], second: list[float]) -> float:
    dx = max(second[0] - first[2], first[0] - second[2], 0.0)
    dy = max(second[1] - first[3], first[1] - second[3], 0.0)
    return math.hypot(dx, dy)


def grasp_closed_mask(
    ts: Any,
    gripper_ts: Any,
    gripper_values: Any,
    *,
    closed_is_low: bool = True,
    enter: float = 0.4,
    exit: float = 0.6,
    settle_s: float = 0.2,
) -> Any | None:
    """Return a settled hysteretic closed-gripper mask on target timestamps."""

    import numpy as np

    target_ts = np.asarray(ts, dtype=float)
    source_ts = np.asarray(gripper_ts, dtype=float)
    values = np.asarray(gripper_values, dtype=float)
    if source_ts.ndim != 1 or values.ndim != 1 or len(source_ts) != len(values):
        raise ValueError("gripper timestamps and values must be aligned 1-D arrays")
    if len(source_ts) < 2:
        return None
    if not 0.0 <= enter < exit <= 1.0:
        raise ValueError("gripper hysteresis must satisfy 0 <= enter < exit <= 1")
    if settle_s < 0.0:
        raise ValueError("gripper settle_s cannot be negative")

    finite = np.isfinite(source_ts) & np.isfinite(values)
    source_ts = source_ts[finite]
    values = values[finite]
    if len(source_ts) < 2:
        return None
    order = np.argsort(source_ts, kind="stable")
    source_ts = source_ts[order]
    values = values[order]
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if not math.isfinite(span) or span < 1e-6:
        return None

    normalized = (values - low) / span
    if not closed_is_low:
        normalized = 1.0 - normalized
    closed = np.zeros(len(normalized), dtype=bool)
    state = False
    for index, value in enumerate(normalized):
        if not state and value < enter:
            state = True
        elif state and value > exit:
            state = False
        closed[index] = state

    index = 0
    while index < len(closed):
        if not closed[index]:
            index += 1
            continue
        end = index
        while end < len(closed) and closed[end]:
            end += 1
        closed[index:end] &= (
            source_ts[index:end] >= source_ts[index] + settle_s
        )
        index = end

    if len(target_ts) == 0:
        return np.zeros(0, dtype=bool)
    right = np.searchsorted(source_ts, target_ts)
    right = np.clip(right, 1, len(source_ts) - 1)
    nearest = np.where(
        np.abs(source_ts[right - 1] - target_ts)
        <= np.abs(source_ts[right] - target_ts),
        right - 1,
        right,
    )
    return closed[nearest]


def _interaction_grasp_mask(
    context: MetricComputationContext,
    timestamps_sec: list[float],
    config: dict[str, Any],
) -> tuple[list[bool], dict[str, Any]]:
    gripper_signal = str(config.get("gripper_signal", "auto")).strip().lower()
    closed_is_low = bool(config.get("closed_is_low", True))
    enter = float(config.get("enter", 0.4))
    exit = float(config.get("exit", 0.6))
    settle_s = float(config.get("settle_s", 0.2))
    gating_config = {
        "gripper_signal": gripper_signal,
        "closed_is_low": closed_is_low,
        "enter": enter,
        "exit": exit,
        "settle_s": settle_s,
    }
    stream, source_timestamps, source_values = _gripper_signal_series(
        context,
        gripper_signal,
    )
    mask = grasp_closed_mask(
        timestamps_sec,
        source_timestamps,
        source_values,
        closed_is_low=closed_is_low,
        enter=enter,
        exit=exit,
        settle_s=settle_s,
    )
    if mask is None:
        fallback = [True] * len(timestamps_sec)
        return fallback, {
            "gating_available": False,
            "gating_config": gating_config,
            "gating_stream_name": stream.name if stream is not None else None,
            "gated_in_fraction": 1.0 if fallback else 0.0,
            "warning": (
                "gripper gating unavailable; gated metrics equal ungated metrics"
            ),
        }
    output = [bool(value) for value in mask]
    return output, {
        "gating_available": True,
        "gating_config": gating_config,
        "gating_stream_name": stream.name if stream is not None else None,
        "gated_in_fraction": (
            sum(output) / len(output) if output else 0.0
        ),
    }


def _gripper_signal_series(
    context: MetricComputationContext,
    gripper_signal: str,
) -> tuple[Any | None, list[float], list[float]]:
    aliases = {
        "auto": ("gripper_action", "gripper"),
        "commanded": ("gripper_action",),
        "action": ("gripper_action",),
        "gripper_action": ("gripper_action",),
        "measured": ("gripper",),
        "aperture": ("gripper",),
        "gripper": ("gripper",),
    }
    preferred_names = aliases.get(gripper_signal, (gripper_signal,))
    stream = next(
        (
            candidate
            for name in preferred_names
            for candidate in context.bundle.streams
            if candidate.name == name and candidate.modality == Modality.GRIPPER
        ),
        None,
    )
    if stream is None:
        return None, [], []
    keys = (
        ("gripper_action",)
        if stream.name == "gripper_action"
        else ("gripper", "gripper_state", "gripper_aperture")
    )
    timestamps, vectors = extract_vector_series(stream, keys)
    values = [
        sum(float(value) for value in vector) / len(vector)
        for vector in vectors
        if vector
    ]
    return stream, timestamps[:len(values)], values


def _sample_frames_by_period(
    frames: list[CameraFrameRef],
    period_sec: float,
) -> list[CameraFrameRef]:
    if not frames:
        return []
    selected = [frames[0]]
    previous_timestamp = float(frames[0].timestamp_sec)
    for frame in frames[1:]:
        timestamp = float(frame.timestamp_sec)
        if timestamp - previous_timestamp + 1e-9 < period_sec:
            continue
        selected.append(frame)
        previous_timestamp = timestamp
    return selected


def _configured_camera_names(config: dict[str, object]) -> list[str]:
    interaction_config = config.get("interaction")
    raw_names = (
        interaction_config.get("cameras")
        if isinstance(interaction_config, dict)
        and interaction_config.get("cameras") is not None
        else config.get("camera_names", ["head"])
    )
    if isinstance(raw_names, str):
        names = [raw_names]
    elif isinstance(raw_names, list):
        names = [str(name) for name in raw_names if isinstance(name, str) and name]
    else:
        names = []
    return names or ["head"]


def _configured_resize(value: object) -> tuple[int, int]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        width, height = int(value[0]), int(value[1])
        if width > 0 and height > 0:
            return width, height
    return 384, 384


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_or_none(values: Any) -> float | None:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def _unavailable_result(
    camera_names: list[str],
    *,
    reason: str,
    matched_camera_names: list[str] | None = None,
    camera_errors: dict[str, list[str]] | None = None,
    image_read_failures: list[str] | None = None,
) -> MetricResult:
    return MetricResult(
        name=ObjectInteractionMotionMetric.name,
        description="Grounded-SAM2 and CoTracker object-interaction motion.",
        source_modalities=[Modality.RGB],
        unit="image_diagonals_per_second",
        samples=[],
        thresholds=[],
        events=[],
        display_hint="object_interaction_motion_timeseries",
        metadata={
            "available": False,
            "unavailable_reason": reason,
            "configured_camera_names": camera_names,
            "matched_camera_names": matched_camera_names or [],
            "camera_errors": camera_errors or {},
            "image_read_failures": image_read_failures or [],
            "model_training": "pretrained_frozen_no_task_finetuning",
        },
    )
