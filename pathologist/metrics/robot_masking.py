"""Optional zero-shot robot masking with Grounding DINO and SAM2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROBOT_PROMPT = "robot arm. robot gripper."
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
MASK_CACHE_VERSION = 2
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any, Any]] = {}


@dataclass(slots=True)
class RobotMaskBatch:
    """Robot masks and inference diagnostics for one camera sequence."""

    masks: list[Any]
    metadata: dict[str, Any]


class GroundedSAM2RobotArmMasker:
    """Detect robot parts with Grounding DINO and track them with SAM2."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.prompt = _normalized_prompt(config.get("prompt", DEFAULT_ROBOT_PROMPT))
        self.dino_model_id = str(config.get("dino_model_id", DEFAULT_DINO_MODEL))
        self.sam2_model_id = str(config.get("sam2_model_id", DEFAULT_SAM2_MODEL))
        self.box_threshold = float(config.get("box_threshold", 0.35))
        self.text_threshold = float(config.get("text_threshold", 0.25))
        self.mask_threshold = float(config.get("mask_threshold", 0.0))
        self.duplicate_iou_threshold = min(
            1.0,
            max(0.0, float(config.get("duplicate_iou_threshold", 0.8))),
        )
        self.chunk_size_frames = max(1, int(config.get("chunk_size_frames", 20)))
        self.requested_device = str(config.get("device", "auto"))
        raw_cache_dir = config.get("cache_dir")
        self.cache_dir = (
            Path(str(raw_cache_dir)).expanduser().resolve()
            if raw_cache_dir
            else None
        )
        self._torch: Any = None
        self._dino_processor: Any = None
        self._dino_model: Any = None
        self._sam2_processor: Any = None
        self._sam2_model: Any = None
        self._device: Any = None
        self._dtype: Any = None

    def masks_for_paths(self, image_paths: list[str | Path]) -> RobotMaskBatch:
        """Return one boolean robot mask for each chronologically ordered frame."""

        resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
        if not resolved_paths:
            return RobotMaskBatch(
                masks=[],
                metadata=self._metadata(detection_count=0, chunk_count=0),
            )

        cached_masks = self._read_cached_masks(resolved_paths)
        if cached_masks is not None:
            chunk_count = (
                len(resolved_paths) + self.chunk_size_frames - 1
            ) // self.chunk_size_frames
            return RobotMaskBatch(
                masks=cached_masks,
                metadata={
                    **self._metadata(detection_count=None, chunk_count=chunk_count),
                    "cache_hit": True,
                },
            )

        self._load_models()
        masks = []
        chunk_detection_counts = []
        for start_index in range(0, len(resolved_paths), self.chunk_size_frames):
            chunk_paths = resolved_paths[start_index:start_index + self.chunk_size_frames]
            first_image = _load_rgb_image(chunk_paths[0])
            boxes = self._detect_robot_boxes(first_image)
            chunk_detection_counts.append(len(boxes))
            if boxes:
                masks.extend(self._track_boxes(chunk_paths, boxes))
            else:
                masks.extend(
                    _empty_masks(
                        count=len(chunk_paths),
                        height=first_image.height,
                        width=first_image.width,
                    )
                )
        self._write_cached_masks(resolved_paths, masks)
        return RobotMaskBatch(
            masks=masks,
            metadata={
                **self._metadata(
                    detection_count=sum(chunk_detection_counts),
                    chunk_count=len(chunk_detection_counts),
                ),
                "chunk_detection_counts": chunk_detection_counts,
                "cache_hit": False,
            },
        )

    def _load_models(self) -> None:
        if self._dino_model is not None and self._sam2_model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2VideoModel,
                Sam2VideoProcessor,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Robot-masked visual flow requires torch, torchvision, and transformers "
                "with SAM2 support. Install requirements-robot-masking.txt in the active "
                "trajectory-pathologist environment."
            ) from exc

        self._torch = torch
        self._device = _resolve_device(torch, self.requested_device)
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        cache_key = (
            self.dino_model_id,
            self.sam2_model_id,
            str(self._device),
        )
        cached_models = _MODEL_CACHE.get(cache_key)
        if cached_models is not None:
            (
                self._dino_processor,
                self._dino_model,
                self._sam2_processor,
                self._sam2_model,
            ) = cached_models
            return
        self._dino_processor = AutoProcessor.from_pretrained(self.dino_model_id)
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.dino_model_id
        ).to(self._device)
        self._dino_model.eval()
        self._sam2_processor = Sam2VideoProcessor.from_pretrained(self.sam2_model_id)
        self._sam2_model = Sam2VideoModel.from_pretrained(self.sam2_model_id).to(
            self._device,
            dtype=self._dtype,
        )
        self._sam2_model.eval()
        _MODEL_CACHE[cache_key] = (
            self._dino_processor,
            self._dino_model,
            self._sam2_processor,
            self._sam2_model,
        )

    def _detect_robot_boxes(self, image: Any) -> list[list[float]]:
        inputs = self._dino_processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            outputs = self._dino_model(**inputs)
        detections = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = detections.get("boxes")
        scores = detections.get("scores")
        if boxes is None or len(boxes) == 0:
            return []
        box_values = boxes.detach().cpu().tolist()
        score_values = (
            scores.detach().cpu().tolist()
            if scores is not None
            else [1.0] * len(box_values)
        )
        return _deduplicate_detections(
            box_values,
            score_values,
            iou_threshold=self.duplicate_iou_threshold,
        )

    def _track_boxes(
        self,
        image_paths: list[Path],
        boxes: list[list[float]],
    ) -> list[Any]:
        session = self._sam2_processor.init_video_session(
            inference_device=self._device,
            inference_state_device="cpu",
            max_vision_features_cache_size=1,
            dtype=self._dtype,
        )
        masks = []
        try:
            with self._torch.no_grad():
                for frame_idx, image_path in enumerate(image_paths):
                    image = _load_rgb_image(image_path)
                    inputs = self._sam2_processor(
                        images=image,
                        return_tensors="pt",
                    ).to(self._device)
                    if frame_idx == 0:
                        original_size = tuple(
                            int(value)
                            for value in inputs.original_sizes[0].detach().cpu().tolist()
                        )
                        self._sam2_processor.add_inputs_to_inference_session(
                            inference_session=session,
                            frame_idx=0,
                            obj_ids=list(range(1, len(boxes) + 1)),
                            input_boxes=[boxes],
                            original_size=original_size,
                        )
                    output = self._sam2_model(
                        inference_session=session,
                        frame=inputs.pixel_values[0],
                    )
                    masks.append(
                        _union_sam2_masks(
                            self._sam2_processor,
                            output.pred_masks,
                            inputs.original_sizes,
                            self.mask_threshold,
                        )
                    )
        finally:
            session.reset_inference_session()
            if isinstance(session.processed_frames, dict):
                session.processed_frames.clear()
            del session
            if self._device.type == "cuda":
                self._torch.cuda.empty_cache()
        return masks

    def _read_cached_masks(self, image_paths: list[Path]) -> list[Any] | None:
        if self.cache_dir is None:
            return None
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image

        cache_paths = [self._cache_path(path) for path in image_paths]
        if not all(path.is_file() for path in cache_paths):
            return None
        return [
            np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
            for path in cache_paths
        ]

    def _write_cached_masks(
        self,
        image_paths: list[Path],
        masks: list[Any],
    ) -> None:
        if self.cache_dir is None:
            return
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for image_path, mask in zip(image_paths, masks):
            cache_path = self._cache_path(image_path)
            if not cache_path.is_file():
                Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(cache_path)

    def _cache_path(self, image_path: Path) -> Path:
        stat = image_path.stat()
        payload = {
            "cache_version": MASK_CACHE_VERSION,
            "path": str(image_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "prompt": self.prompt,
            "dino_model_id": self.dino_model_id,
            "sam2_model_id": self.sam2_model_id,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "mask_threshold": self.mask_threshold,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
            "chunk_size_frames": self.chunk_size_frames,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.png"

    def _metadata(
        self,
        detection_count: int | None,
        chunk_count: int | None = None,
    ) -> dict[str, Any]:
        return {
            "method": "grounding_dino_plus_sam2_video",
            "prompt": self.prompt,
            "dino_model_id": self.dino_model_id,
            "sam2_model_id": self.sam2_model_id,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "mask_threshold": self.mask_threshold,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
            "chunk_size_frames": self.chunk_size_frames,
            "chunk_count": chunk_count,
            "device": str(self._device or self.requested_device),
            "detection_count": detection_count,
        }


def _normalized_prompt(value: Any) -> str:
    prompt = str(value).strip().lower() or DEFAULT_ROBOT_PROMPT
    return prompt if prompt.endswith(".") else f"{prompt}."


def _deduplicate_detections(
    boxes: list[list[float]],
    scores: list[float],
    *,
    iou_threshold: float,
) -> list[list[float]]:
    """Keep every distinct detection while suppressing near-duplicate boxes."""

    ranked = sorted(
        zip(scores, boxes),
        key=lambda item: item[0],
        reverse=True,
    )
    kept: list[list[float]] = []
    for _, box in ranked:
        if all(_box_iou(box, previous) <= iou_threshold for previous in kept):
            kept.append(box)
    return kept


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_rgb_image(image_path: Path) -> Any:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


def _empty_masks(
    *,
    count: int,
    height: int,
    width: int,
) -> list[Any]:
    import numpy as np  # type: ignore[import-not-found]

    return [
        np.zeros((height, width), dtype=bool)
        for _ in range(count)
    ]


def _union_sam2_masks(
    processor: Any,
    pred_masks: Any,
    original_sizes: Any,
    mask_threshold: float,
) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    processed = processor.post_process_masks(
        [pred_masks],
        original_sizes=original_sizes,
        binarize=False,
    )[0]
    array = processed.detach().cpu().numpy()
    reduction_axes = tuple(range(max(0, array.ndim - 2)))
    return np.any(array > mask_threshold, axis=reduction_axes).astype(bool)
