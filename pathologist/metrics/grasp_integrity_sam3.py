"""Lazy SAM 3.1 runtime for bimanual grasp-integrity metrics."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


SAM3_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM3_CHECKPOINT_REPO = "facebook/sam3.1"
SAM3_CHECKPOINT_NAME = "sam3.1_multiplex.pt"


@dataclass(frozen=True, slots=True)
class Sam3Mask:
    """One mask and confidence returned by SAM 3."""

    mask: np.ndarray
    score: float
    object_id: int


@dataclass(frozen=True, slots=True)
class Sam3TrackedFrame:
    """One propagated object observation."""

    frame_index: int
    mask: np.ndarray | None
    score: float


class Sam3Runtime:
    """Own one lazily constructed official SAM 3.1 predictor."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        probability_threshold: float = 0.5,
        version: str = "sam3.1",
    ) -> None:
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        )
        self.probability_threshold = float(probability_threshold)
        self.version = str(version)
        self._predictor: Any | None = None

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            try:
                from sam3.model_builder import build_sam3_predictor
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "SAM 3 is required; install requirements-sam3.txt"
                ) from exc
            kwargs: dict[str, Any] = dict(
                checkpoint_path=self.checkpoint_path,
                version=self.version,
                compile=False,
                warm_up=False,
                max_num_objects=16,
                use_fa3=False,
                async_loading_frames=False,
            )
            if self.version == "sam3.1":
                kwargs["multiplex_count"] = 16
            self._predictor = build_sam3_predictor(**kwargs)
        return self._predictor

    def prompt(
        self,
        frame_paths: Sequence[str | Path],
        *,
        frame_index: int,
        text: str | None = None,
        points_xy: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        object_id: int | None = None,
    ) -> list[Sam3Mask]:
        """Run one text or point prompt and close the temporary session."""

        with _ordered_frame_folder(frame_paths) as folder:
            response = _start_compatible_session(self.predictor, str(folder))
            session_id = str(response["session_id"])
            try:
                prompt_response = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(frame_index),
                        "text": text,
                        "points": (
                            [list(map(float, point)) for point in points_xy]
                            if points_xy is not None
                            else None
                        ),
                        "point_labels": (
                            [int(label) for label in point_labels]
                            if point_labels is not None
                            else None
                        ),
                        "obj_id": object_id,
                        "rel_coordinates": False,
                        "output_prob_thresh": self.probability_threshold,
                    }
                )
                return _masks_from_output(prompt_response.get("outputs", {}))
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )

    def open_session(
        self,
        frame_paths: Sequence[str | Path],
    ) -> Sam3Session:
        """Create a reusable session context for several initialization prompts."""

        return Sam3Session(self, frame_paths)

    def track(
        self,
        frame_paths: Sequence[str | Path],
        *,
        positive_points_xy: Sequence[Sequence[float]],
        negative_points_xy: Sequence[Sequence[float]],
        object_id: int = 0,
    ) -> list[Sam3TrackedFrame]:
        """Track one point-prompted object forward over an ordered frame folder."""

        points = [
            *[list(map(float, point)) for point in positive_points_xy],
            *[list(map(float, point)) for point in negative_points_xy],
        ]
        labels = [
            *([1] * len(positive_points_xy)),
            *([0] * len(negative_points_xy)),
        ]
        with _ordered_frame_folder(frame_paths) as folder:
            response = _start_compatible_session(self.predictor, str(folder))
            session_id = str(response["session_id"])
            observations: dict[int, Sam3TrackedFrame] = {}
            try:
                initial = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "points": points,
                        "point_labels": labels,
                        "obj_id": int(object_id),
                        "rel_coordinates": False,
                        "output_prob_thresh": self.probability_threshold,
                    }
                )
                observations[0] = _tracked_frame(
                    0,
                    initial.get("outputs", {}),
                    object_id=object_id,
                )
                request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": max(0, len(frame_paths) - 1),
                    "output_prob_thresh": self.probability_threshold,
                }
                for item in self.predictor.handle_stream_request(request):
                    frame_index = int(item["frame_index"])
                    observations[frame_index] = _tracked_frame(
                        frame_index,
                        item.get("outputs", {}),
                        object_id=object_id,
                    )
            finally:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
        return [
            observations.get(
                frame_index,
                Sam3TrackedFrame(frame_index, None, 0.0),
            )
            for frame_index in range(len(frame_paths))
        ]


class Sam3ImageRuntime:
    """SAM 3 concept and interactive image paths on one shared image model."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        probability_threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        )
        self.probability_threshold = float(probability_threshold)
        self._model: Any | None = None
        self._processor: Any | None = None

    def concept_masks(
        self,
        image: np.ndarray,
        text: str,
    ) -> list[Sam3Mask]:
        return self.analyze_image(
            image,
            text_prompts=[text],
            point_prompts=[],
        )[0][text]

    def analyze_image(
        self,
        image: np.ndarray,
        *,
        text_prompts: Sequence[str],
        point_prompts: Sequence[
            tuple[Sequence[Sequence[float]], Sequence[int]]
        ],
    ) -> tuple[dict[str, list[Sam3Mask]], list[list[Sam3Mask]]]:
        """Run many concept and point prompts with one image backbone pass."""

        from PIL import Image

        state = self.processor.set_image(
            Image.fromarray(_rgb_image(image))
        )
        text_output: dict[str, list[Sam3Mask]] = {}
        for text in text_prompts:
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(text, state)
            text_output[text] = _concept_masks_from_state(state)
        point_output = [
            self._point_masks_from_state(
                state,
                points_xy=points,
                point_labels=labels,
            )
            for points, labels in point_prompts
        ]
        return text_output, point_output

    def _point_masks_from_state(
        self,
        state: dict[str, Any],
        *,
        points_xy: Sequence[Sequence[float]],
        point_labels: Sequence[int],
    ) -> list[Sam3Mask]:
        masks, scores, _ = self.model.predict_inst(
            state,
            point_coords=np.asarray(points_xy, dtype=np.float32),
            point_labels=np.asarray(point_labels, dtype=np.int32),
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )
        binary = np.asarray(masks, dtype=bool)
        quality = np.asarray(scores, dtype=float).reshape(-1)
        return [
            Sam3Mask(
                mask=np.asarray(binary[index], dtype=bool),
                score=float(quality[index]),
                object_id=index,
            )
            for index in range(min(len(binary), len(quality)))
        ]

    def point_masks(
        self,
        image: np.ndarray,
        *,
        points_xy: Sequence[Sequence[float]],
        point_labels: Sequence[int],
    ) -> list[Sam3Mask]:
        return self.analyze_image(
            image,
            text_prompts=[],
            point_prompts=[(points_xy, point_labels)],
        )[1][0]

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sam3.model_builder import build_sam3_image_model
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "SAM 3 is required; install requirements-sam3.txt"
                ) from exc
            self._model = build_sam3_image_model(
                device="cuda",
                eval_mode=True,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=self.checkpoint_path is None,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            from sam3.model.sam3_image_processor import Sam3Processor

            self._processor = Sam3Processor(
                self.model,
                device="cuda",
                confidence_threshold=self.probability_threshold,
            )
        return self._processor


def _concept_masks_from_state(state: dict[str, Any]) -> list[Sam3Mask]:
    masks = np.asarray(_numpy(state.get("masks", [])), dtype=bool)
    scores = np.asarray(
        _numpy(state.get("scores", [])),
        dtype=float,
    ).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    count = min(len(masks), len(scores))
    return [
        Sam3Mask(
            mask=np.asarray(masks[index], dtype=bool),
            score=float(scores[index]),
            object_id=index,
        )
        for index in range(count)
    ]


class Sam3Session:
    """Reusable SAM 3 session over one ordered hold-window frame sequence."""

    def __init__(
        self,
        runtime: Sam3Runtime,
        frame_paths: Sequence[str | Path],
    ) -> None:
        self.runtime = runtime
        self.frame_paths = list(frame_paths)
        self._folder_context: _ordered_frame_folder | None = None
        self.session_id: str | None = None

    def __enter__(self) -> Sam3Session:
        self._folder_context = _ordered_frame_folder(self.frame_paths)
        folder = self._folder_context.__enter__()
        response = _start_compatible_session(
            self.runtime.predictor,
            str(folder),
        )
        self.session_id = str(response["session_id"])
        return self

    def __exit__(self, *_: object) -> None:
        if self.session_id is not None:
            self.runtime.predictor.handle_request(
                {"type": "close_session", "session_id": self.session_id}
            )
        if self._folder_context is not None:
            self._folder_context.__exit__()
        self.session_id = None
        self._folder_context = None

    def reset(self) -> None:
        self.runtime.predictor.handle_request(
            {"type": "reset_session", "session_id": self._require_id()}
        )

    def prompt(
        self,
        *,
        frame_index: int,
        text: str | None = None,
        points_xy: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        object_id: int | None = None,
        reset: bool = True,
    ) -> list[Sam3Mask]:
        if reset:
            self.reset()
        response = self.runtime.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self._require_id(),
                "frame_index": int(frame_index),
                "text": text,
                "points": (
                    [list(map(float, point)) for point in points_xy]
                    if points_xy is not None
                    else None
                ),
                "point_labels": (
                    [int(label) for label in point_labels]
                    if point_labels is not None
                    else None
                ),
                "obj_id": object_id,
                "rel_coordinates": False,
                "output_prob_thresh": self.runtime.probability_threshold,
            }
        )
        return _masks_from_output(response.get("outputs", {}))

    def propagate(self, *, object_id: int = 0) -> list[Sam3TrackedFrame]:
        request = {
            "type": "propagate_in_video",
            "session_id": self._require_id(),
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": max(0, len(self.frame_paths) - 1),
            "output_prob_thresh": self.runtime.probability_threshold,
        }
        observations: dict[int, Sam3TrackedFrame] = {}
        for item in self.runtime.predictor.handle_stream_request(request):
            frame_index = int(item["frame_index"])
            observations[frame_index] = _tracked_frame(
                frame_index,
                item.get("outputs", {}),
                object_id=object_id,
            )
        return [
            observations.get(
                frame_index,
                Sam3TrackedFrame(frame_index, None, 0.0),
            )
            for frame_index in range(len(self.frame_paths))
        ]

    def prompt_mask(
        self,
        mask: np.ndarray,
        *,
        frame_index: int = 0,
        object_id: int = 0,
    ) -> None:
        """Replace an initialized object's point mask with a known binary mask."""

        import torch

        session_id = self._require_id()
        predictor_session = self.runtime.predictor._all_inference_states[session_id]
        inference_state = predictor_session["state"]
        model = self.runtime.predictor.model
        if hasattr(model, "_get_tracker_inference_states_by_obj_ids"):
            tracker_states = model._get_tracker_inference_states_by_obj_ids(
                inference_state,
                [int(object_id)],
            )
        else:
            tracker_states = model._get_sam2_inference_states_by_obj_ids(
                inference_state,
                [int(object_id)],
            )
        if len(tracker_states) != 1:
            raise RuntimeError(
                f"SAM 3 object {object_id} was not initialized before mask prompting"
            )
        tensor = torch.as_tensor(np.asarray(mask, dtype=bool))
        if hasattr(model.tracker, "add_new_mask"):
            model.tracker.add_new_mask(
                tracker_states[0],
                int(frame_index),
                int(object_id),
                tensor,
            )
        else:
            model.tracker.add_new_masks(
                tracker_states[0],
                int(frame_index),
                [int(object_id)],
                tensor[None],
            )

    def propagate_tracker(
        self,
        *,
        object_id: int = 0,
    ) -> list[Sam3TrackedFrame]:
        """Propagate one interactive object without concept-detector suppression."""

        session_id = self._require_id()
        predictor_session = self.runtime.predictor._all_inference_states[session_id]
        inference_state = predictor_session["state"]
        model = self.runtime.predictor.model
        tracker_states = model._get_sam2_inference_states_by_obj_ids(
            inference_state,
            [int(object_id)],
        )
        if len(tracker_states) != 1:
            raise RuntimeError(f"SAM 3 object {object_id} is not initialized")
        observations: dict[int, Sam3TrackedFrame] = {}
        iterator = model.tracker.propagate_in_video(
            inference_state=tracker_states[0],
            start_frame_idx=0,
            max_frame_num_to_track=len(self.frame_paths),
            reverse=False,
        )
        for item in iterator:
            frame_index = int(item[0])
            masks = np.asarray(_numpy(item[3]))
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            scores = (
                np.asarray(_numpy(item[4]), dtype=float).reshape(-1)
                if len(item) > 4
                else np.ones(len(masks), dtype=float)
            )
            score = float(scores[0]) if len(scores) else 0.0
            if not 0.0 <= score <= 1.0:
                score = float(1.0 / (1.0 + np.exp(-score)))
            mask = np.asarray(masks[0] > 0.0, dtype=bool) if len(masks) else None
            observations[frame_index] = Sam3TrackedFrame(
                frame_index=frame_index,
                mask=mask,
                score=score,
            )
        return [
            observations.get(
                frame_index,
                Sam3TrackedFrame(frame_index, None, 0.0),
            )
            for frame_index in range(len(self.frame_paths))
        ]

    def _require_id(self) -> str:
        if self.session_id is None:
            raise RuntimeError("SAM 3 session is not open")
        return self.session_id


class Sam3MaskCache:
    """NPZ cache keyed by episode/window/prompt/checkpoint identity."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {
                **payload,
                "sam3_commit": SAM3_COMMIT,
                "checkpoint_repo": SAM3_CHECKPOINT_REPO,
                "checkpoint_name": SAM3_CHECKPOINT_NAME,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        path = self.root / f"{key}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            return (
                np.asarray(payload["masks"], dtype=bool),
                np.asarray(payload["scores"], dtype=float),
            )

    def save(
        self,
        key: str,
        *,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> Path:
        path = self.root / f"{key}.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            masks=np.asarray(masks, dtype=bool),
            scores=np.asarray(scores, dtype=float),
        )
        temporary.replace(path)
        return path


class _ordered_frame_folder:
    """Temporary lexically ordered links to an existing JPEG sequence."""

    def __init__(self, frame_paths: Sequence[str | Path]) -> None:
        self.frame_paths = [Path(path).expanduser().resolve() for path in frame_paths]
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if not self.frame_paths:
            raise ValueError("SAM 3 requires at least one frame")
        self._temporary = tempfile.TemporaryDirectory(prefix="prism_sam3_")
        root = Path(self._temporary.name)
        for index, source in enumerate(self.frame_paths):
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = root / f"{index:06d}{source.suffix.lower() or '.jpg'}"
            os.symlink(source, destination)
        return root

    def __exit__(self, *_: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def _masks_from_output(output: Any) -> list[Sam3Mask]:
    if not isinstance(output, dict):
        return []
    masks = np.asarray(output.get("out_binary_masks", []), dtype=bool)
    scores = np.asarray(output.get("out_probs", []), dtype=float).reshape(-1)
    object_ids = np.asarray(output.get("out_obj_ids", []), dtype=int).reshape(-1)
    if masks.ndim == 2:
        masks = masks[None, ...]
    count = min(len(masks), len(scores), len(object_ids))
    return [
        Sam3Mask(
            mask=np.asarray(masks[index], dtype=bool),
            score=float(scores[index]),
            object_id=int(object_ids[index]),
        )
        for index in range(count)
    ]


def _tracked_frame(
    frame_index: int,
    output: Any,
    *,
    object_id: int,
) -> Sam3TrackedFrame:
    candidates = _masks_from_output(output)
    for candidate in candidates:
        if candidate.object_id == object_id:
            return Sam3TrackedFrame(
                frame_index=frame_index,
                mask=candidate.mask,
                score=candidate.score,
            )
    if len(candidates) == 1:
        candidate = candidates[0]
        return Sam3TrackedFrame(
            frame_index=frame_index,
            mask=candidate.mask,
            score=candidate.score,
        )
    return Sam3TrackedFrame(frame_index=frame_index, mask=None, score=0.0)


def _start_compatible_session(predictor: Any, resource_path: str) -> dict[str, str]:
    """Start a session while tolerating SAM 3.1's narrower init signature."""

    model = predictor.model
    signature = inspect.signature(model.init_state)
    candidates = {
        "resource_path": resource_path,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "async_loading_frames": getattr(
            predictor,
            "async_loading_frames",
            False,
        ),
        "video_loader_type": getattr(predictor, "video_loader_type", "cv2"),
        "use_cv2": getattr(predictor, "video_loader_type", "cv2") == "cv2",
    }
    kwargs = {
        name: value
        for name, value in candidates.items()
        if name in signature.parameters
    }
    state = model.init_state(**kwargs)
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    return {"session_id": session_id}


def _rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("SAM 3 image input must have shape H x W x 3")
    # OpenCV is the image source throughout this project.
    return np.ascontiguousarray(array[..., ::-1])


def _numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value
