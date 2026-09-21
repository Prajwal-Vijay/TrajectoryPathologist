"""Visual metrics computed from RGB camera frames."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import summarize
from pathologist.metrics.robot_masking import (  # type: ignore[import-not-found]
    GroundedSAM2RobotArmMasker,
)
from pathologist.schemas.common import Modality, TimeWindow
from pathologist.schemas.metrics import MetricResult, MetricSample
from pathologist.schemas.trajectory_bundle import CameraFrameRef


class VisualFlowMetric:
    """Dense optical-flow magnitude between consecutive RGB frames."""

    name = "visual_flow"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        """Compute mean Farneback flow magnitude for configured cameras."""

        config = metric_config(context, self.name)
        camera_names = _configured_camera_names(config)
        frame_stride = max(1, int(config.get("frame_stride", 2)))
        resize_shape = _configured_resize_shape(config)
        spatial_aggregation = _configured_spatial_aggregation(config)
        top_fraction = _configured_top_fraction(config)
        robot_masking_config = _configured_robot_masking(config)
        robot_masking_enabled = bool(robot_masking_config.get("enabled", False))
        robot_mask_dilation_px = max(
            0,
            int(robot_masking_config.get("dilation_px", 0)),
        )
        grouped_frames = _group_configured_frames(context.bundle.camera_frames, camera_names)
        matched_camera_names = sorted(grouped_frames)
        if not matched_camera_names:
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                robot_masking_enabled=robot_masking_enabled,
                reason="no frames found for configured camera_names",
            )

        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                matched_camera_names=matched_camera_names,
                robot_masking_enabled=robot_masking_enabled,
                reason=f"opencv unavailable: {type(exc).__name__}: {exc}",
            )

        robot_masker = (
            GroundedSAM2RobotArmMasker(robot_masking_config)
            if robot_masking_enabled
            else None
        )
        dilation_kernel = None
        if robot_masker is not None and robot_mask_dilation_px > 0:
            kernel_size = 2 * robot_mask_dilation_px + 1
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
        samples_by_camera: dict[str, list[MetricSample]] = defaultdict(list)
        image_read_failures: list[str] = []
        skipped_frame_count = 0
        image_shape_examples: dict[str, list[int]] = {}
        robot_mask_metadata: dict[str, Any] = {}
        robot_mask_fractions: list[float] = []
        fully_masked_pair_count = 0

        for camera_name in matched_camera_names:
            frames = grouped_frames[camera_name]
            sampled_frames = frames[::frame_stride]
            if len(sampled_frames) < 2:
                skipped_frame_count += len(frames)
                continue
            robot_masks = None
            if robot_masker is not None:
                try:
                    mask_batch = robot_masker.masks_for_paths(
                        [frame.image_path for frame in sampled_frames]
                    )
                except Exception as exc:
                    return _unavailable_result(
                        configured_camera_names=camera_names,
                        frame_stride=frame_stride,
                        resize_shape=resize_shape,
                        matched_camera_names=matched_camera_names,
                        robot_masking_enabled=True,
                        reason=(
                            "robot masking unavailable: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                robot_masks = mask_batch.masks
                robot_mask_metadata[camera_name] = mask_batch.metadata
            previous_gray = None
            previous_robot_mask = None
            previous_frame_ref: CameraFrameRef | None = None
            for frame_index, frame_ref in enumerate(sampled_frames):
                image = cv2.imread(str(Path(frame_ref.image_path)))
                if image is None:
                    image_read_failures.append(str(frame_ref.image_path))
                    previous_gray = None
                    previous_robot_mask = None
                    previous_frame_ref = None
                    continue
                if camera_name not in image_shape_examples:
                    image_shape_examples[camera_name] = [int(item) for item in image.shape]
                if resize_shape is not None:
                    image = cv2.resize(image, resize_shape)
                robot_mask = None
                if robot_masks is not None and frame_index < len(robot_masks):
                    robot_mask = robot_masks[frame_index]
                    if resize_shape is not None:
                        robot_mask = cv2.resize(
                            robot_mask.astype("uint8"),
                            resize_shape,
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    else:
                        robot_mask = robot_mask.astype(bool)
                    if dilation_kernel is not None:
                        robot_mask = cv2.dilate(
                            robot_mask.astype("uint8"),
                            dilation_kernel,
                        ).astype(bool)
                    robot_mask_fractions.append(float(robot_mask.mean()))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if previous_gray is not None and previous_frame_ref is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        previous_gray,
                        gray,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0,
                    )
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    if robot_mask is not None and previous_robot_mask is not None:
                        valid_pixels = ~(robot_mask | previous_robot_mask)
                        if not valid_pixels.any():
                            fully_masked_pair_count += 1
                            previous_gray = gray
                            previous_robot_mask = robot_mask
                            previous_frame_ref = frame_ref
                            continue
                        valid_magnitudes = magnitude[valid_pixels]
                    else:
                        valid_magnitudes = magnitude.reshape(-1)
                    value = _aggregate_flow_magnitudes(
                        valid_magnitudes,
                        aggregation=spatial_aggregation,
                        top_fraction=top_fraction,
                    )
                    timestamp = float(frame_ref.timestamp_sec)
                    samples_by_camera[camera_name].append(
                        MetricSample(
                            timestamp_sec=timestamp,
                            value=value,
                            window=TimeWindow(
                                start_sec=float(previous_frame_ref.timestamp_sec),
                                end_sec=timestamp,
                            ),
                        )
                    )
                previous_gray = gray
                previous_robot_mask = robot_mask
                previous_frame_ref = frame_ref

        samples, sample_camera_names, camera_aggregation_metadata = (
            _max_across_camera_samples(samples_by_camera)
        )
        if not samples:
            reason = "requires at least two readable frames per configured camera"
            if image_read_failures:
                reason = "configured camera frames could not be read"
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                matched_camera_names=matched_camera_names,
                robot_masking_enabled=robot_masking_enabled,
                reason=reason,
                image_read_failures=image_read_failures,
                skipped_frame_count=skipped_frame_count,
                image_shape_examples=image_shape_examples,
            )

        values = [float(sample.value) for sample in samples]
        return MetricResult(
            name=self.name,
            description=(
                "Spatially aggregated dense optical-flow magnitude between consecutive "
                "RGB frames, "
                "used as a visual source for trajectory-relative ECDF surprise scoring."
            ),
            source_modalities=[Modality.RGB],
            unit="mean_optical_flow_magnitude",
            samples=samples,
            summary=summarize(values),
            thresholds=[],
            events=[],
            display_hint="visual_flow_timeseries",
            metadata={
                "available": True,
                "configured_camera_names": camera_names,
                "matched_camera_names": matched_camera_names,
                "frame_stride": frame_stride,
                "resize": list(resize_shape) if resize_shape is not None else None,
                "spatial_aggregation": spatial_aggregation,
                "top_fraction": (
                    top_fraction
                    if spatial_aggregation == "top_fraction_mean"
                    else None
                ),
                "flow_method": "farneback_dense_optical_flow",
                "farneback_params": {
                    "pyr_scale": 0.5,
                    "levels": 3,
                    "winsize": 15,
                    "iterations": 3,
                    "poly_n": 5,
                    "poly_sigma": 1.2,
                    "flags": 0,
                },
                "sample_camera_names": sample_camera_names,
                **camera_aggregation_metadata,
                "skipped_frame_count": skipped_frame_count,
                "image_read_failures": image_read_failures,
                "image_shape_examples": image_shape_examples,
                "robot_masking_enabled": robot_masking_enabled,
                "robot_masking_method": (
                    "grounding_dino_plus_sam2_video"
                    if robot_masking_enabled
                    else None
                ),
                "robot_mask_metadata": robot_mask_metadata,
                "robot_mask_dilation_px": (
                    robot_mask_dilation_px
                    if robot_masking_enabled
                    else None
                ),
                "mean_robot_mask_fraction": (
                    sum(robot_mask_fractions) / len(robot_mask_fractions)
                    if robot_mask_fractions
                    else None
                ),
                "fully_masked_pair_count": fully_masked_pair_count,
            },
        )


def _max_across_camera_samples(
    samples_by_camera: dict[str, list[MetricSample]],
    *,
    sample_count_metadata_key: str = "camera_flow_sample_counts",
) -> tuple[list[MetricSample], list[str], dict[str, Any]]:
    camera_names = sorted(
        camera_name
        for camera_name, samples in samples_by_camera.items()
        if samples
    )
    sample_counts = {
        camera_name: len(samples_by_camera[camera_name])
        for camera_name in camera_names
    }
    if not camera_names:
        return [], [], {
            "camera_aggregation": "max_after_linear_time_alignment",
            "camera_reference_timeline": None,
            sample_count_metadata_key: sample_counts,
        }
    if len(camera_names) == 1:
        camera_name = camera_names[0]
        samples = list(samples_by_camera[camera_name])
        return samples, [camera_name] * len(samples), {
            "camera_aggregation": "single_camera",
            "camera_reference_timeline": camera_name,
            sample_count_metadata_key: sample_counts,
        }

    reference_camera = max(
        camera_names,
        key=lambda camera_name: len(samples_by_camera[camera_name]),
    )
    camera_series = {
        camera_name: (
            [float(sample.timestamp_sec) for sample in samples_by_camera[camera_name]],
            [float(sample.value) for sample in samples_by_camera[camera_name]],
        )
        for camera_name in camera_names
    }
    aggregated_samples: list[MetricSample] = []
    winning_camera_names: list[str] = []
    for reference_sample in samples_by_camera[reference_camera]:
        timestamp = float(reference_sample.timestamp_sec)
        candidates: list[tuple[float, str]] = []
        for camera_name in camera_names:
            timestamps, values = camera_series[camera_name]
            value = _linearly_interpolated_value(timestamps, values, timestamp)
            if value is not None:
                candidates.append((value, camera_name))
        if not candidates:
            continue
        value, winning_camera = max(candidates, key=lambda item: item[0])
        aggregated_samples.append(
            MetricSample(
                timestamp_sec=timestamp,
                value=value,
                window=reference_sample.window,
            )
        )
        winning_camera_names.append(winning_camera)

    return aggregated_samples, winning_camera_names, {
        "camera_aggregation": "max_after_linear_time_alignment",
        "camera_reference_timeline": reference_camera,
        sample_count_metadata_key: sample_counts,
    }


def _linearly_interpolated_value(
    timestamps: list[float],
    values: list[float],
    timestamp: float,
) -> float | None:
    if not timestamps or len(timestamps) != len(values):
        return None
    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        return None
    right = bisect_left(timestamps, timestamp)
    if right < len(timestamps) and timestamps[right] == timestamp:
        return values[right]
    if right == 0 or right >= len(timestamps):
        return None
    left = right - 1
    span = timestamps[right] - timestamps[left]
    if span <= 0.0:
        return values[left]
    weight = (timestamp - timestamps[left]) / span
    return values[left] + weight * (values[right] - values[left])


def _configured_camera_names(config: dict[str, object]) -> list[str]:
    raw_camera_names = config.get("camera_names", ["head"])
    if isinstance(raw_camera_names, str):
        names = [raw_camera_names]
    elif isinstance(raw_camera_names, list):
        names = [str(name) for name in raw_camera_names if isinstance(name, str) and name]
    else:
        names = []
    return names or ["head"]


def _configured_resize_shape(config: dict[str, object]) -> tuple[int, int] | None:
    raw_resize = config.get("resize", [128, 128])
    if raw_resize is None:
        return None
    if (
        isinstance(raw_resize, list)
        and len(raw_resize) == 2
        and all(isinstance(value, (int, float)) for value in raw_resize)
    ):
        width = int(raw_resize[0])
        height = int(raw_resize[1])
        if width > 0 and height > 0:
            return (width, height)
    return (128, 128)


def _configured_spatial_aggregation(config: dict[str, object]) -> str:
    value = str(config.get("spatial_aggregation", "mean")).strip().lower()
    if value not in {"mean", "top_fraction_mean"}:
        raise ValueError(
            "visual_flow.spatial_aggregation must be 'mean' or 'top_fraction_mean'"
        )
    return value


def _configured_top_fraction(config: dict[str, object]) -> float:
    value = float(config.get("top_fraction", 0.02))
    if not 0.0 < value <= 1.0:
        raise ValueError("visual_flow.top_fraction must be in (0, 1]")
    return value


def _aggregate_flow_magnitudes(
    magnitudes: Any,
    *,
    aggregation: str,
    top_fraction: float,
) -> float:
    import numpy as np  # type: ignore[import-not-found]

    values = np.asarray(magnitudes).reshape(-1)
    if values.size == 0:
        return 0.0
    if aggregation == "mean":
        return float(values.mean())
    top_count = max(1, int(np.ceil(values.size * top_fraction)))
    partition_index = values.size - top_count
    return float(np.partition(values, partition_index)[partition_index:].mean())


def _configured_robot_masking(config: dict[str, object]) -> dict[str, Any]:
    raw_config = config.get("robot_masking", {})
    return dict(raw_config) if isinstance(raw_config, dict) else {}


def _group_configured_frames(
    frames: list[CameraFrameRef],
    camera_names: list[str],
) -> dict[str, list[CameraFrameRef]]:
    camera_name_set = set(camera_names)
    grouped: dict[str, list[CameraFrameRef]] = defaultdict(list)
    for frame in frames:
        if frame.camera_name in camera_name_set:
            grouped[frame.camera_name].append(frame)
    return {
        camera_name: sorted(camera_frames, key=_frame_sort_key)
        for camera_name, camera_frames in grouped.items()
    }


def _frame_sort_key(frame: CameraFrameRef) -> tuple[float, int]:
    return (float(frame.timestamp_sec), int(frame.frame_index or 0))


def _unavailable_result(
    configured_camera_names: list[str],
    frame_stride: int,
    resize_shape: tuple[int, int] | None,
    reason: str,
    matched_camera_names: list[str] | None = None,
    image_read_failures: list[str] | None = None,
    skipped_frame_count: int = 0,
    image_shape_examples: dict[str, list[int]] | None = None,
    robot_masking_enabled: bool = False,
) -> MetricResult:
    return MetricResult(
        name=VisualFlowMetric.name,
        description="Mean dense optical-flow magnitude between consecutive RGB frames.",
        source_modalities=[Modality.RGB],
        unit="mean_optical_flow_magnitude",
        samples=[],
        thresholds=[],
        events=[],
        display_hint="visual_flow_timeseries",
        metadata={
            "available": False,
            "unavailable_reason": reason,
            "configured_camera_names": configured_camera_names,
            "matched_camera_names": matched_camera_names or [],
            "frame_stride": frame_stride,
            "resize": list(resize_shape) if resize_shape is not None else None,
            "flow_method": "farneback_dense_optical_flow",
            "skipped_frame_count": skipped_frame_count,
            "image_read_failures": image_read_failures or [],
            "image_shape_examples": image_shape_examples or {},
            "robot_masking_enabled": robot_masking_enabled,
        },
    )
