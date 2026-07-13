"""Visual metrics computed from RGB camera frames."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import summarize
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
        grouped_frames = _group_configured_frames(context.bundle.camera_frames, camera_names)
        matched_camera_names = sorted(grouped_frames)
        if not matched_camera_names:
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                reason="no frames found for configured camera_names",
            )

        try:
            import cv2
        except Exception as exc:
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                matched_camera_names=matched_camera_names,
                reason=f"opencv unavailable: {type(exc).__name__}: {exc}",
            )

        samples: list[MetricSample] = []
        values: list[float] = []
        sample_camera_names: list[str] = []
        image_read_failures: list[str] = []
        skipped_frame_count = 0
        image_shape_examples: dict[str, list[int]] = {}

        for camera_name in matched_camera_names:
            frames = grouped_frames[camera_name]
            sampled_frames = frames[::frame_stride]
            if len(sampled_frames) < 2:
                skipped_frame_count += len(frames)
                continue
            previous_gray = None
            previous_frame_ref: CameraFrameRef | None = None
            for frame_ref in sampled_frames:
                image = cv2.imread(str(Path(frame_ref.image_path)))
                if image is None:
                    image_read_failures.append(str(frame_ref.image_path))
                    previous_gray = None
                    previous_frame_ref = None
                    continue
                if camera_name not in image_shape_examples:
                    image_shape_examples[camera_name] = [int(item) for item in image.shape]
                if resize_shape is not None:
                    image = cv2.resize(image, resize_shape)
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
                    value = float(magnitude.mean())
                    timestamp = float(frame_ref.timestamp_sec)
                    samples.append(
                        MetricSample(
                            timestamp_sec=timestamp,
                            value=value,
                            window=TimeWindow(
                                start_sec=float(previous_frame_ref.timestamp_sec),
                                end_sec=timestamp,
                            ),
                        )
                    )
                    values.append(value)
                    sample_camera_names.append(camera_name)
                previous_gray = gray
                previous_frame_ref = frame_ref

        if not samples:
            reason = "requires at least two readable frames per configured camera"
            if image_read_failures:
                reason = "configured camera frames could not be read"
            return _unavailable_result(
                configured_camera_names=camera_names,
                frame_stride=frame_stride,
                resize_shape=resize_shape,
                matched_camera_names=matched_camera_names,
                reason=reason,
                image_read_failures=image_read_failures,
                skipped_frame_count=skipped_frame_count,
                image_shape_examples=image_shape_examples,
            )

        return MetricResult(
            name=self.name,
            description=(
                "Mean dense optical-flow magnitude between consecutive RGB frames, "
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
                "skipped_frame_count": skipped_frame_count,
                "image_read_failures": image_read_failures,
                "image_shape_examples": image_shape_examples,
            },
        )


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
        },
    )
