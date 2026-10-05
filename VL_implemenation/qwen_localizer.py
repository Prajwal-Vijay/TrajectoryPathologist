"""Deterministic Qwen3-VL K=1 temporal localization."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping


EVENT_TYPES = {"insertion", "removal"}


def build_localization_prompt(instruction: str) -> str:
    normalized_instruction = instruction.casefold()
    explicit_event_task = any(
        phrase in normalized_instruction
        for phrase in ("insert", "remove", "plug in", "take out")
    )
    existence_rule = (
        "The task instruction establishes that an insertion or removal occurs. "
        "Do not decide whether it exists: return exactly one best interval."
        if explicit_event_task
        else (
            'If neither event occurs, return {"events":[]}. Return an event only '
            "when it is supported by the video."
        )
    )
    return f"""
You are temporally annotating a three-view robot-manipulation video.
The task is: {instruction}.

Find only the first chronological interval where an insertion or removal is
actively occurring. Insertion means actively putting, pressing, plugging, or
fitting an object into or onto its destination. Removal means actively taking,
unplugging, or pulling an object out of or off its holder or destination.
Exclude reaching, grasping, transport, waiting, resets, and recovery unless the
object is actively being inserted or removed.

Use the task instruction as context. Identify the best matching visible
interval even if the action is brief or partially occluded. {existence_rule}

Return JSON only, with an "events" list containing at most one object. The
object must have "type" ("insertion" or "removal"), "start_sec", and "end_sec".
The two times must be the actual observed timestamps, not placeholders or
example values.
Times are seconds from the beginning of this video. Do not include confidence,
explanation, or Markdown.
""".strip()


def build_window_prompt(instruction: str) -> str:
    return f"""
You are temporally annotating one short window from a three-view
robot-manipulation video. The task is: {instruction}.

This window has already been positively identified as containing insertion or
removal. Localize that event. Insertion is the contact phase where an object
enters or is fitted into its destination. Removal is the contact phase where an
object is pulled out of or detached from its holder. Do not choose reaching,
grasping, transport, ordinary placement, folding, waiting, reset, or recovery.

Return only the first event in this window. Return JSON only, with an "events"
list containing exactly one object. The object must have "type"
("insertion" or "removal"), "start_sec", and "end_sec". Times must be actual
observed seconds measured from the beginning of this supplied window, never
placeholders. Do not return an empty list. Do not include confidence,
explanation, or Markdown.
""".strip()


def parse_localization_answer(
    answer: str,
    *,
    video_duration_sec: float,
    allowed_event_types: set[str] | None = None,
) -> dict[str, Any]:
    allowed_types = EVENT_TYPES if allowed_event_types is None else allowed_event_types
    stripped = answer.strip()
    raw = re.sub(r"^\s*```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "parse_status": "unparseable_json",
            "parse_error": str(exc),
            "parsed_payload": None,
            "event": None,
        }
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return {
            "parse_status": "invalid_schema",
            "parse_error": "Expected an object with an events list.",
            "parsed_payload": payload,
            "event": None,
        }
    events = payload["events"]
    if not events:
        return {
            "parse_status": "valid_no_event",
            "parse_error": None,
            "parsed_payload": payload,
            "event": None,
            "returned_event_count": 0,
        }
    first = events[0]
    if not isinstance(first, dict):
        return {
            "parse_status": "invalid_event",
            "parse_error": "The first event is not an object.",
            "parsed_payload": payload,
            "event": None,
            "returned_event_count": len(events),
        }
    event_type = str(first.get("type", "")).strip().casefold()
    if event_type not in allowed_types:
        return {
            "parse_status": "invalid_event",
            "parse_error": f"Unsupported event type: {event_type!r}.",
            "parsed_payload": payload,
            "event": None,
            "returned_event_count": len(events),
        }
    try:
        start_sec = float(first["start_sec"])
        end_sec = float(first["end_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "parse_status": "invalid_event",
            "parse_error": f"Invalid interval bounds: {exc}",
            "parsed_payload": payload,
            "event": None,
            "returned_event_count": len(events),
        }
    if not math.isfinite(start_sec) or not math.isfinite(end_sec):
        return {
            "parse_status": "invalid_event",
            "parse_error": "Interval bounds must be finite.",
            "parsed_payload": payload,
            "event": None,
            "returned_event_count": len(events),
        }
    low, high = sorted((start_sec, end_sec))
    low = max(0.0, min(low, video_duration_sec))
    high = max(0.0, min(high, video_duration_sec))
    event = {
        "type": event_type,
        "start_sec": low,
        "end_sec": high,
        "midpoint_sec": (low + high) / 2.0,
    }
    return {
        "parse_status": "valid_event",
        "parse_error": None,
        "parsed_payload": payload,
        "event": event,
        "returned_event_count": len(events),
        "extra_events_ignored": max(0, len(events) - 1),
    }


class QwenLocalizer:
    def __init__(
        self,
        *,
        model_path: str | Path,
        model_sample_fps: float,
        min_pixels: int,
        max_pixels: int,
        total_pixels: int,
        scan_total_pixels: int,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")

        self.torch = torch
        self.model_sample_fps = model_sample_fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.scan_total_pixels = scan_total_pixels
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.model_path = Path(model_path).expanduser().resolve()
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        self.runtime = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "video_reader": os.environ["FORCE_QWENVL_VIDEO_READER"],
        }

    def localize(
        self,
        *,
        video_path: Path,
        instruction: str,
        video_duration_sec: float,
    ) -> dict[str, Any]:
        return self._run_prompt(
            video_path=video_path,
            prompt=build_localization_prompt(instruction),
            video_duration_sec=video_duration_sec,
            allowed_event_types=EVENT_TYPES,
        )

    def localize_gripper_close(
        self,
        *,
        video_path: Path,
        video_duration_sec: float,
    ) -> dict[str, Any]:
        prompt = """
You are temporally annotating a synchronized three-view robot video.
Find the first moment when either robot gripper visibly closes around an
object, ignoring any static closed pose before motion begins. Return JSON only,
with an "events" list containing at most one object. That object must have type
"gripper_close", plus "start_sec" and "end_sec" set to actual observed
timestamps rather than placeholder values.
If no gripper visibly closes around an object, return an empty "events" list.
Times are seconds from the beginning of this video. Do not include confidence,
explanation, or Markdown.
""".strip()
        return self._run_prompt(
            video_path=video_path,
            prompt=prompt,
            video_duration_sec=video_duration_sec,
            allowed_event_types={"gripper_close"},
        )

    def localize_window(
        self,
        *,
        video_path: Path,
        instruction: str,
        window_start_sec: float,
        window_end_sec: float,
    ) -> dict[str, Any]:
        if window_start_sec < 0 or window_end_sec <= window_start_sec:
            raise ValueError("Invalid temporal window.")
        return self._run_prompt(
            video_path=video_path,
            prompt=build_window_prompt(instruction),
            video_duration_sec=window_end_sec - window_start_sec,
            allowed_event_types=EVENT_TYPES,
            video_start_sec=window_start_sec,
            video_end_sec=window_end_sec,
            rebase_video_metadata=True,
        )

    def score_window(
        self,
        *,
        video_path: Path,
        instruction: str,
        window_start_sec: float,
        window_end_sec: float,
    ) -> dict[str, Any]:
        prompt = f"""
You are checking one short window from a three-view robot video for the task:
{instruction}.

Does this window visibly contain the active contact phase of inserting an
object into its destination or removing an object from its holder? Ordinary
placement, folding, reaching, grasping, transport, waiting, reset, and recovery
are not insertion/removal. Answer exactly Yes or No.
""".strip()
        inputs, input_log, decoded_sec, started = self._prepare_inputs(
            video_path=video_path,
            prompt=prompt,
            video_start_sec=window_start_sec,
            video_end_sec=window_end_sec,
            rebase_video_metadata=True,
            total_pixels=self.scan_total_pixels,
        )
        inference_started = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[0, -1]
        yes_ids = self.processor.tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = self.processor.tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError("Yes/No must each tokenize to one token.")
        choice_logits = self.torch.stack([logits[yes_ids[0]], logits[no_ids[0]]])
        probabilities = self.torch.softmax(choice_logits.float(), dim=0)
        inference_sec = time.perf_counter() - inference_started
        return {
            "yes_probability": float(probabilities[0].item()),
            "no_probability": float(probabilities[1].item()),
            "yes_logit": float(choice_logits[0].item()),
            "no_logit": float(choice_logits[1].item()),
            "prompt": prompt,
            "model_input": input_log,
            "runtime": self.runtime,
            "decode_and_preprocess_sec": decoded_sec,
            "inference_sec": inference_sec,
            "total_model_sec": time.perf_counter() - started,
        }

    def _run_prompt(
        self,
        *,
        video_path: Path,
        prompt: str,
        video_duration_sec: float,
        allowed_event_types: set[str],
        video_start_sec: float | None = None,
        video_end_sec: float | None = None,
        rebase_video_metadata: bool = False,
    ) -> dict[str, Any]:
        inputs, input_log, decoded_sec, started = self._prepare_inputs(
            video_path=video_path,
            prompt=prompt,
            video_start_sec=video_start_sec,
            video_end_sec=video_end_sec,
            rebase_video_metadata=rebase_video_metadata,
            total_pixels=self.total_pixels,
        )
        inference_started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        inference_sec = time.perf_counter() - inference_started
        answer_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        answer = self.processor.batch_decode(
            answer_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_localization_answer(
            answer,
            video_duration_sec=video_duration_sec,
            allowed_event_types=allowed_event_types,
        )
        return {
            "raw_answer": answer,
            "prompt": prompt,
            **parsed,
            "model_input": input_log,
            "runtime": self.runtime,
            "decode_and_preprocess_sec": decoded_sec,
            "inference_sec": inference_sec,
            "total_model_sec": time.perf_counter() - started,
        }

    def _prepare_inputs(
        self,
        *,
        video_path: Path,
        prompt: str,
        video_start_sec: float | None,
        video_end_sec: float | None,
        rebase_video_metadata: bool,
        total_pixels: int,
    ) -> tuple[Any, dict[str, Any], float, float]:
        from qwen_vl_utils import process_vision_info

        video_content: dict[str, Any] = {
            "type": "video",
            "video": video_path.resolve().as_uri(),
            "fps": self.model_sample_fps,
            "min_pixels": self.min_pixels,
            "total_pixels": total_pixels,
        }
        if video_start_sec is not None:
            video_content["video_start"] = video_start_sec
        if video_end_sec is not None:
            video_content["video_end"] = video_end_sec
        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        started = time.perf_counter()
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos_with_metadata, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not videos_with_metadata:
            raise RuntimeError(f"Qwen video loader returned no video for {video_path}")
        videos, metadata = zip(*videos_with_metadata)
        videos = list(videos)
        metadata = list(metadata)
        original_metadata = [_metadata_to_dict(item) for item in metadata]
        if rebase_video_metadata:
            metadata = [_rebased_metadata(item) for item in metadata]
        decoded_sec = time.perf_counter() - started

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=metadata,
            return_tensors="pt",
            do_resize=False,
            cap_pixels_per_frame=True,
            **video_kwargs,
        )
        input_log = {
            "video_metadata": [_metadata_to_dict(item) for item in metadata],
            "original_video_metadata": original_metadata,
            "video_kwargs": _jsonable(video_kwargs),
            "decoded_video_shapes": [
                list(getattr(video, "shape", [])) for video in videos
            ],
            "video_grid_thw": _jsonable(inputs.get("video_grid_thw")),
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "model_sample_fps_requested": self.model_sample_fps,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_pixels_passed_to_loader": False,
            "total_pixels": total_pixels,
            "video_start_sec": video_start_sec,
            "video_end_sec": video_end_sec,
            "metadata_rebased_to_window": rebase_video_metadata,
        }
        inputs = inputs.to(self.model.device)
        return inputs, input_log, decoded_sec, started


def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, Mapping):
        return {str(key): _jsonable(value) for key, value in metadata.items()}
    fields = {}
    for name in (
        "fps",
        "frames_indices",
        "total_num_frames",
        "video_backend",
        "duration",
    ):
        if hasattr(metadata, name):
            fields[name] = _jsonable(getattr(metadata, name))
    return fields or {"repr": repr(metadata)}


def _rebased_metadata(metadata: Any) -> dict[str, Any]:
    output = _metadata_to_dict(metadata)
    indices = [int(value) for value in output.get("frames_indices", [])]
    if not indices:
        return output
    first = indices[0]
    output["frames_indices"] = [value - first for value in indices]
    output["total_num_frames"] = max(output["frames_indices"]) + 1
    return output


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)
