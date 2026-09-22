"""Frozen DINOv2 patch-level temporal novelty metric."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import summarize
from pathologist.metrics.vision import (
    _group_configured_frames,
    _max_across_camera_samples,
)
from pathologist.schemas.common import Modality, TimeWindow
from pathologist.schemas.metrics import MetricResult, MetricSample
from pathologist.schemas.trajectory_bundle import CameraFrameRef


class FrozenDinoPatchNoveltyMetric:
    """Temporal novelty from locally matched frozen DINOv2 patch tokens."""

    name = "visual_patch_novelty"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = metric_config(context, self.name)
        enabled = bool(config.get("enabled", False))
        camera_names = _configured_camera_names(config)
        model_id = str(config.get("model_id", "facebook/dinov2-small"))
        sample_period_sec = float(config.get("sample_period_sec", 0.5))
        local_match_radius = int(config.get("local_match_radius", 2))
        top_fraction = float(config.get("top_fraction", 0.05))
        batch_size = int(config.get("batch_size", 16))
        requested_device = str(config.get("device", "auto")).strip().lower()

        if not enabled:
            return _unavailable_result(
                camera_names=camera_names,
                model_id=model_id,
                sample_period_sec=sample_period_sec,
                local_match_radius=local_match_radius,
                top_fraction=top_fraction,
                reason="disabled by metrics.visual_patch_novelty.enabled",
            )
        if sample_period_sec <= 0:
            raise ValueError("visual_patch_novelty.sample_period_sec must be positive")
        if local_match_radius < 0:
            raise ValueError("visual_patch_novelty.local_match_radius cannot be negative")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("visual_patch_novelty.top_fraction must be in (0, 1]")
        if batch_size <= 0:
            raise ValueError("visual_patch_novelty.batch_size must be positive")

        grouped_frames = _group_configured_frames(
            context.bundle.camera_frames,
            camera_names,
        )
        matched_camera_names = sorted(grouped_frames)
        if not matched_camera_names:
            return _unavailable_result(
                camera_names=camera_names,
                model_id=model_id,
                sample_period_sec=sample_period_sec,
                local_match_radius=local_match_radius,
                top_fraction=top_fraction,
                reason="no frames found for configured camera_names",
            )

        try:
            resources = _load_dinov2_resources(model_id, requested_device)
        except Exception as exc:
            return _unavailable_result(
                camera_names=camera_names,
                matched_camera_names=matched_camera_names,
                model_id=model_id,
                sample_period_sec=sample_period_sec,
                local_match_radius=local_match_radius,
                top_fraction=top_fraction,
                reason=f"DINOv2 unavailable: {type(exc).__name__}: {exc}",
            )

        samples_by_camera: dict[str, list[MetricSample]] = {}
        image_read_failures: list[str] = []
        sampled_frame_counts: dict[str, int] = {}
        patch_grid_shapes: dict[str, list[int]] = {}
        for camera_name in matched_camera_names:
            sampled_frames = _sample_frames_by_period(
                grouped_frames[camera_name],
                sample_period_sec,
            )
            sampled_frame_counts[camera_name] = len(sampled_frames)
            if len(sampled_frames) < 2:
                continue

            camera_samples: list[MetricSample] = []
            previous_frame: CameraFrameRef | None = None
            previous_grid = None
            for frame, patch_grid in _iter_patch_embeddings(
                sampled_frames,
                resources=resources,
                batch_size=batch_size,
                image_read_failures=image_read_failures,
            ):
                patch_grid_shapes[camera_name] = [
                    int(patch_grid.shape[0]),
                    int(patch_grid.shape[1]),
                ]
                if previous_frame is not None and previous_grid is not None:
                    patch_novelty = symmetric_local_patch_novelty(
                        previous_grid,
                        patch_grid,
                        radius=local_match_radius,
                    )
                    value = top_fraction_mean(patch_novelty, top_fraction)
                    camera_samples.append(
                        MetricSample(
                            timestamp_sec=float(frame.timestamp_sec),
                            value=value,
                            window=TimeWindow(
                                start_sec=float(previous_frame.timestamp_sec),
                                end_sec=float(frame.timestamp_sec),
                            ),
                        )
                    )
                previous_frame = frame
                previous_grid = patch_grid
            if camera_samples:
                samples_by_camera[camera_name] = camera_samples

        samples, sample_camera_names, camera_metadata = _max_across_camera_samples(
            samples_by_camera,
            sample_count_metadata_key="camera_patch_novelty_sample_counts",
        )
        if not samples:
            reason = "requires at least two readable sampled frames per configured camera"
            if image_read_failures:
                reason = "configured camera frames could not be read"
            return _unavailable_result(
                camera_names=camera_names,
                matched_camera_names=matched_camera_names,
                model_id=model_id,
                sample_period_sec=sample_period_sec,
                local_match_radius=local_match_radius,
                top_fraction=top_fraction,
                reason=reason,
                image_read_failures=image_read_failures,
            )

        values = [float(sample.value) for sample in samples]
        return MetricResult(
            name=self.name,
            description=(
                "Symmetric local-correspondence novelty of frozen DINOv2 patch tokens; "
                "the most novel patch fraction is averaged per frame pair and cameras "
                "are combined by a timestamp-aligned maximum."
            ),
            source_modalities=[Modality.RGB],
            unit="cosine_patch_novelty",
            samples=samples,
            summary=summarize(values),
            thresholds=[],
            events=[],
            display_hint="visual_patch_novelty_timeseries",
            metadata={
                "available": True,
                "model_id": model_id,
                "model_frozen": True,
                "task_specific_training": False,
                "device": resources["device"],
                "configured_camera_names": camera_names,
                "matched_camera_names": matched_camera_names,
                "sample_period_sec": sample_period_sec,
                "batch_size": batch_size,
                "local_match_radius": local_match_radius,
                "top_fraction": top_fraction,
                "matching": "symmetric_local_max_cosine",
                "score_formula": (
                    "TopMean_q({1-max_{j in N_r(i)} cos(z_ti,z_(t-1)j)} "
                    "union {1-max_{i in N_r(j)} cos(z_(t-1)j,z_ti)})"
                ),
                "camera_aggregation": camera_metadata["camera_aggregation"],
                "camera_reference_timeline": camera_metadata[
                    "camera_reference_timeline"
                ],
                "camera_patch_novelty_sample_counts": camera_metadata[
                    "camera_patch_novelty_sample_counts"
                ],
                "sample_camera_names": sample_camera_names,
                "sampled_frame_counts": sampled_frame_counts,
                "patch_grid_shapes": patch_grid_shapes,
                "image_read_failures": image_read_failures,
            },
        )


def symmetric_local_patch_novelty(
    previous_grid: Any,
    current_grid: Any,
    *,
    radius: int,
) -> Any:
    """Return bidirectional locally matched cosine novelty for two patch grids."""

    import torch
    import torch.nn.functional as functional

    if radius < 0:
        raise ValueError("radius cannot be negative")
    if previous_grid.ndim != 3 or current_grid.ndim != 3:
        raise ValueError("patch grids must have shape [height, width, feature_dim]")
    if previous_grid.shape != current_grid.shape:
        raise ValueError("patch grids must have identical shapes")

    previous = functional.normalize(previous_grid.float(), dim=-1)
    current = functional.normalize(current_grid.float(), dim=-1)
    current_to_previous = _directional_local_patch_novelty(
        source=current,
        target=previous,
        radius=radius,
    )
    previous_to_current = _directional_local_patch_novelty(
        source=previous,
        target=current,
        radius=radius,
    )
    return torch.cat(
        [current_to_previous.reshape(-1), previous_to_current.reshape(-1)]
    )


def top_fraction_mean(values: Any, fraction: float) -> float:
    """Average the largest configured fraction of a tensor-like score vector."""

    import torch

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        return 0.0
    count = max(1, int(math.ceil(tensor.numel() * fraction)))
    return float(torch.topk(tensor, count, largest=True).values.mean().item())


def _directional_local_patch_novelty(
    *,
    source: Any,
    target: Any,
    radius: int,
) -> Any:
    """For every source patch, find the best target match in a local window."""

    import torch

    height, width, _ = source.shape
    best_similarity = torch.full(
        (height, width),
        -torch.inf,
        dtype=source.dtype,
        device=source.device,
    )
    for y_offset in range(-radius, radius + 1):
        source_y0 = max(0, -y_offset)
        source_y1 = min(height, height - y_offset)
        target_y0 = source_y0 + y_offset
        target_y1 = source_y1 + y_offset
        for x_offset in range(-radius, radius + 1):
            source_x0 = max(0, -x_offset)
            source_x1 = min(width, width - x_offset)
            target_x0 = source_x0 + x_offset
            target_x1 = source_x1 + x_offset
            similarity = (
                source[source_y0:source_y1, source_x0:source_x1]
                * target[target_y0:target_y1, target_x0:target_x1]
            ).sum(dim=-1)
            best_similarity[source_y0:source_y1, source_x0:source_x1] = (
                torch.maximum(
                    best_similarity[
                        source_y0:source_y1,
                        source_x0:source_x1,
                    ],
                    similarity,
                )
            )
    novelty = (1.0 - best_similarity).clamp(min=0.0, max=2.0)
    return torch.where(novelty < 1e-6, torch.zeros_like(novelty), novelty)


def _sample_frames_by_period(
    frames: list[CameraFrameRef],
    period_sec: float,
) -> list[CameraFrameRef]:
    if not frames:
        return []
    selected = [frames[0]]
    last_timestamp = float(frames[0].timestamp_sec)
    for frame in frames[1:]:
        timestamp = float(frame.timestamp_sec)
        if timestamp - last_timestamp + 1e-9 < period_sec:
            continue
        selected.append(frame)
        last_timestamp = timestamp
    return selected


def _iter_patch_embeddings(
    frames: list[CameraFrameRef],
    *,
    resources: dict[str, Any],
    batch_size: int,
    image_read_failures: list[str],
) -> Iterator[tuple[CameraFrameRef, Any]]:
    from PIL import Image

    torch = resources["torch"]
    processor = resources["processor"]
    model = resources["model"]
    device = resources["device"]
    patch_size = int(resources["patch_size"])

    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        readable_frames: list[CameraFrameRef] = []
        images = []
        for frame in batch_frames:
            try:
                with Image.open(Path(frame.image_path)) as image:
                    images.append(image.convert("RGB"))
                readable_frames.append(frame)
            except Exception:
                image_read_failures.append(str(frame.image_path))
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("DINOv2 image processor did not return pixel_values")
        grid_height = int(pixel_values.shape[-2]) // patch_size
        grid_width = int(pixel_values.shape[-1]) // patch_size
        patch_count = grid_height * grid_width
        model_inputs = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**model_inputs)
            tokens = outputs.last_hidden_state
            if int(tokens.shape[1]) < patch_count:
                raise RuntimeError(
                    "DINOv2 output has fewer tokens than the inferred patch grid"
                )
            patch_tokens = tokens[:, -patch_count:, :]
            patch_tokens = torch.nn.functional.normalize(
                patch_tokens.float(),
                dim=-1,
            )
            patch_grids = patch_tokens.reshape(
                len(readable_frames),
                grid_height,
                grid_width,
                int(patch_tokens.shape[-1]),
            ).cpu()
        for frame, patch_grid in zip(readable_frames, patch_grids):
            yield frame, patch_grid


@lru_cache(maxsize=4)
def _load_dinov2_resources(
    model_id: str,
    requested_device: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = _resolve_device(requested_device, torch)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    patch_size = getattr(model.config, "patch_size", 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
        "patch_size": int(patch_size),
    }


def _resolve_device(requested_device: str, torch: Any) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested_device!r} but CUDA is unavailable")
    return requested_device


def _configured_camera_names(config: dict[str, object]) -> list[str]:
    raw_names = config.get("camera_names", ["head"])
    if isinstance(raw_names, str):
        names = [raw_names]
    elif isinstance(raw_names, list):
        names = [str(name) for name in raw_names if isinstance(name, str) and name]
    else:
        names = []
    return names or ["head"]


def _unavailable_result(
    *,
    camera_names: list[str],
    model_id: str,
    sample_period_sec: float,
    local_match_radius: int,
    top_fraction: float,
    reason: str,
    matched_camera_names: list[str] | None = None,
    image_read_failures: list[str] | None = None,
) -> MetricResult:
    return MetricResult(
        name=FrozenDinoPatchNoveltyMetric.name,
        description="Frozen DINOv2 patch-level temporal novelty.",
        source_modalities=[Modality.RGB],
        unit="cosine_patch_novelty",
        samples=[],
        thresholds=[],
        events=[],
        display_hint="visual_patch_novelty_timeseries",
        metadata={
            "available": False,
            "unavailable_reason": reason,
            "model_id": model_id,
            "model_frozen": True,
            "task_specific_training": False,
            "configured_camera_names": camera_names,
            "matched_camera_names": matched_camera_names or [],
            "sample_period_sec": sample_period_sec,
            "local_match_radius": local_match_radius,
            "top_fraction": top_fraction,
            "image_read_failures": image_read_failures or [],
        },
    )
