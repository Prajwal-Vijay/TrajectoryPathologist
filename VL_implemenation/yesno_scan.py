"""Fast cached Qwen3-VL yes/no scan for ABC insert/remove localization."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from VL_implemenation.abc_data import (
    Episode,
    camera_cache_directory,
    camera_timeline,
    discover_episodes,
)
from VL_implemenation.mosaic import CAMERA_NAMES, build_mosaic_timeline


PROMPT = (
    "This clip shows a robot manipulation task from three cameras (top, left "
    "wrist, right wrist). In this clip, is the robot actively inserting an "
    "object into, or pulling an object out of, a hole, slot, socket, or holder "
    "(the contact phase, not reaching or carrying)? Answer Yes or No."
)
FRAME_RATE = 5.0
PANEL_SIZE = (320, 256)


@dataclass(frozen=True, slots=True)
class Window:
    """One fixed-length frame window and its trajectory-time coordinates."""

    indices: tuple[int, ...]
    start_sec: float
    end_sec: float
    center_sec: float


@dataclass(slots=True)
class LoadedEpisode:
    """Pre-resized in-memory mosaic frames for one episode."""

    episode: Episode
    frames: np.ndarray
    times: np.ndarray
    windows: list[Window]
    frame_loading_sec: float


@dataclass(slots=True)
class ScoreResult:
    """Scores and timing returned by one episode scan."""

    margin: np.ndarray
    p_yes: np.ndarray
    yes_no_mass: np.ndarray
    preprocessing_sec: float
    gpu_sec: float
    wall_sec: float
    batch_count: int
    batch_size: int
    top_tokens: list[dict[str, Any]]


def build_windows(
    times: Sequence[float],
    *,
    window_frames: int,
    stride_frames: int,
) -> list[Window]:
    """Build fully covering, fixed-size windows, padding very short clips."""

    if window_frames <= 0 or window_frames % 2:
        raise ValueError("window_frames must be a positive even number")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    if not times:
        return []

    frame_count = len(times)
    if frame_count <= window_frames:
        indices = list(range(frame_count))
        indices.extend([frame_count - 1] * (window_frames - frame_count))
        starts = [0]
        index_sets = [indices]
    else:
        starts = list(range(0, frame_count - window_frames + 1, stride_frames))
        final_start = frame_count - window_frames
        if starts[-1] != final_start:
            starts.append(final_start)
        index_sets = [
            list(range(start, start + window_frames))
            for start in starts
        ]

    output = []
    for indices in index_sets:
        frame_times = [float(times[index]) for index in indices]
        output.append(
            Window(
                indices=tuple(indices),
                start_sec=frame_times[0],
                end_sec=frame_times[-1],
                center_sec=float(np.mean(frame_times)),
            )
        )
    return output


def load_episode_frames(
    episode: Episode,
    *,
    frame_cache_root: Path,
    frame_rate: float,
    window_frames: int,
    stride_frames: int,
    workers: int,
) -> LoadedEpisode:
    """Read synchronized cached JPEGs and construct RGB mosaics in memory."""

    started = time.perf_counter()
    timelines = {
        camera: camera_timeline(
            camera_cache_directory(frame_cache_root, episode, camera, frame_rate)
        )
        for camera in CAMERA_NAMES
    }
    records = build_mosaic_timeline(timelines, output_fps=frame_rate)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        mosaics = list(pool.map(_read_mosaic_record, records))
    frames = np.stack(mosaics).astype(np.uint8, copy=False)
    times = np.asarray(
        [float(record["trajectory_sec"]) for record in records],
        dtype=np.float64,
    )
    windows = build_windows(
        times.tolist(),
        window_frames=window_frames,
        stride_frames=stride_frames,
    )
    return LoadedEpisode(
        episode=episode,
        frames=frames,
        times=times,
        windows=windows,
        frame_loading_sec=time.perf_counter() - started,
    )


def _read_mosaic_record(record: dict[str, Any]) -> np.ndarray:
    import cv2

    panels = []
    for camera in CAMERA_NAMES:
        frame = cv2.imread(str(record[f"{camera}_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read {record[f'{camera}_path']}")
        panels.append(cv2.resize(frame, PANEL_SIZE, interpolation=cv2.INTER_AREA))
    bgr = cv2.hconcat(panels)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class QwenYesNoScanner:
    """Batched last-token scorer with asynchronous CPU preprocessing."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_id: str,
        prompt: str = PROMPT,
        frame_rate: float,
        window_frames: int,
        batch_size: int,
        queue_depth: int = 2,
        compile_model: bool = True,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen yes/no scanning")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("high")

        self.torch = torch
        self.model_id = model_id
        self.model_path = model_path.expanduser().resolve()
        self.frame_rate = frame_rate
        self.window_frames = window_frames
        self.batch_size = batch_size
        self.queue_depth = queue_depth
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.processor.tokenizer.padding_side = "left"
        self.yes_ids = self.processor.tokenizer.encode(
            "Yes", add_special_tokens=False
        )
        self.no_ids = self.processor.tokenizer.encode(
            "No", add_special_tokens=False
        )
        if len(self.yes_ids) != 1 or len(self.no_ids) != 1:
            raise RuntimeError(
                f"Yes/No must each be one token: {self.yes_ids=} {self.no_ids=}"
            )
        self.yes_id = int(self.yes_ids[0])
        self.no_id = int(self.no_ids[0])
        self.prompt = ""
        self.prompt_text = ""
        self.set_prompt(prompt)

        attention_backend = (
            "flash_attention_2"
            if importlib.util.find_spec("flash_attn") is not None
            else "sdpa"
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()},
            attn_implementation=attention_backend,
            low_cpu_mem_usage=True,
        ).eval()
        self.device = self.model.device
        self.choice_weight = (
            self.model.lm_head.weight[[self.yes_id, self.no_id]]
            .detach()
            .float()
            .contiguous()
        )
        if compile_model:
            self.model.model.visual = torch.compile(
                self.model.model.visual,
                mode="reduce-overhead",
                fullgraph=False,
            )
        self.runtime = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "attention_backend": attention_backend,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "yes_token_id": self.yes_id,
            "no_token_id": self.no_id,
            "chat_template": self.prompt_text,
            "torch_compile": compile_model,
        }

    def set_prompt(self, prompt: str) -> None:
        """Change only the frozen binary question without reloading the model."""

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("The yes/no scan prompt cannot be empty")
        prompt_text = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "<think>" in prompt_text.casefold():
            raise RuntimeError("The Instruct chat template unexpectedly added thinking")
        self.prompt = prompt
        self.prompt_text = prompt_text
        if hasattr(self, "runtime"):
            self.runtime["chat_template"] = prompt_text

    def score(
        self,
        loaded: LoadedEpisode,
        *,
        max_batches: int | None = None,
        top_token_limit: int = 20,
    ) -> ScoreResult:
        """Score all windows, retrying the episode at half batch size on OOM."""

        batch_size = self.batch_size
        while True:
            try:
                result = self._score_once(
                    loaded,
                    batch_size=batch_size,
                    max_batches=max_batches,
                    top_token_limit=top_token_limit,
                )
                self.batch_size = batch_size
                return result
            except self.torch.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"CUDA OOM; retrying with batch_size={batch_size}", flush=True)

    def _score_once(
        self,
        loaded: LoadedEpisode,
        *,
        batch_size: int,
        max_batches: int | None,
        top_token_limit: int,
    ) -> ScoreResult:
        started = time.perf_counter()
        work_queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_depth)
        stop = object()
        producer_errors: list[BaseException] = []
        limit = len(loaded.windows)
        if max_batches is not None:
            limit = min(limit, max_batches * batch_size)

        def produce() -> None:
            try:
                for first in range(0, limit, batch_size):
                    last = min(limit, first + batch_size)
                    preprocess_started = time.perf_counter()
                    windows = [
                        loaded.frames[list(window.indices)]
                        for window in loaded.windows[first:last]
                    ]
                    metadata = [
                        {
                            "fps": self.frame_rate,
                            "frames_indices": list(range(self.window_frames)),
                            "total_num_frames": self.window_frames,
                            "duration": self.window_frames / self.frame_rate,
                        }
                        for _ in windows
                    ]
                    inputs = self.processor(
                        text=[self.prompt_text] * len(windows),
                        videos=windows,
                        video_metadata=metadata,
                        do_resize=False,
                        do_sample_frames=False,
                        cap_pixels_per_frame=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    for key, value in list(inputs.items()):
                        if hasattr(value, "pin_memory"):
                            inputs[key] = value.pin_memory()
                    work_queue.put(
                        (
                            first,
                            inputs,
                            time.perf_counter() - preprocess_started,
                        )
                    )
            except BaseException as exc:
                producer_errors.append(exc)
            finally:
                work_queue.put(stop)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        margins: list[float] = []
        probabilities: list[float] = []
        masses: list[float] = []
        top_tokens: list[dict[str, Any]] = []
        preprocessing_sec = 0.0
        gpu_sec = 0.0
        batch_count = 0

        while True:
            item = work_queue.get()
            if item is stop:
                break
            first, inputs, batch_preprocess_sec = item
            preprocessing_sec += float(batch_preprocess_sec)
            inputs = inputs.to(self.device, non_blocking=True)
            self.torch.cuda.synchronize()
            gpu_started = time.perf_counter()
            with self.torch.inference_mode():
                outputs = self.model.model(
                    **inputs,
                    use_cache=False,
                )
                last_hidden = outputs.last_hidden_state[:, -1]
                choice_logits = last_hidden.float() @ self.choice_weight.T
                # The complete vocabulary is materialized only for one row per
                # sequence, solely to measure the Yes/No probability mass.
                logits = self.model.lm_head(last_hidden).float()
            self.torch.cuda.synchronize()
            gpu_sec += time.perf_counter() - gpu_started
            batch_count += 1

            margin = choice_logits[:, 0] - choice_logits[:, 1]
            p_yes = self.torch.sigmoid(margin)
            yes_no_logits = logits[:, [self.yes_id, self.no_id]]
            yes_no_mass = self.torch.exp(
                self.torch.logsumexp(yes_no_logits, dim=-1)
                - self.torch.logsumexp(logits, dim=-1)
            )
            margins.extend(margin.detach().cpu().tolist())
            probabilities.extend(p_yes.detach().cpu().tolist())
            masses.extend(yes_no_mass.detach().cpu().tolist())

            remaining = max(0, top_token_limit - len(top_tokens))
            if remaining:
                count = min(remaining, logits.shape[0])
                values, ids = self.torch.topk(logits[:count], k=5, dim=-1)
                for row in range(count):
                    token_ids = ids[row].detach().cpu().tolist()
                    top_tokens.append(
                        {
                            "window_index": first + row,
                            "tokens": [
                                {
                                    "id": int(token_id),
                                    "text": self.processor.tokenizer.decode(
                                        [token_id]
                                    ),
                                    "logit": float(value),
                                }
                                for token_id, value in zip(
                                    token_ids,
                                    values[row].detach().cpu().tolist(),
                                )
                            ],
                        }
                    )
            del outputs, logits, choice_logits, last_hidden, inputs

        producer.join()
        if producer_errors:
            raise producer_errors[0]
        return ScoreResult(
            margin=np.asarray(margins, dtype=np.float32),
            p_yes=np.asarray(probabilities, dtype=np.float32),
            yes_no_mass=np.asarray(masses, dtype=np.float32),
            preprocessing_sec=preprocessing_sec,
            gpu_sec=gpu_sec,
            wall_sec=time.perf_counter() - started,
            batch_count=batch_count,
            batch_size=batch_size,
            top_tokens=top_tokens,
        )


def cache_paths(cache_root: Path, episode: Episode) -> tuple[Path, Path]:
    directory = cache_root / episode.task_name
    return directory / f"{episode.bundle_id}.npz", directory / f"{episode.bundle_id}.json"


def write_cache(
    loaded: LoadedEpisode,
    scores: ScoreResult,
    *,
    cache_root: Path,
    model_id: str,
    window_sec: float,
    stride_sec: float,
    git_commit: str,
    runtime: dict[str, Any],
    prompt: str = PROMPT,
    provenance: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Atomically save compact arrays and a human-readable provenance sidecar."""

    npz_path, json_path = cache_paths(cache_root, loaded.episode)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = npz_path.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            window_start_sec=np.asarray(
                [window.start_sec for window in loaded.windows], dtype=np.float64
            ),
            window_end_sec=np.asarray(
                [window.end_sec for window in loaded.windows], dtype=np.float64
            ),
            center_sec=np.asarray(
                [window.center_sec for window in loaded.windows], dtype=np.float64
            ),
            margin=scores.margin,
            p_yes=scores.p_yes,
            yes_no_mass=scores.yes_no_mass,
        )
    temporary_npz.replace(npz_path)
    payload = {
        "status": "complete",
        "model_id": model_id,
        "prompt": prompt,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "frame_rate": FRAME_RATE,
        "resolution": {
            "per_view_width": PANEL_SIZE[0],
            "per_view_height": PANEL_SIZE[1],
            "mosaic_width": PANEL_SIZE[0] * len(CAMERA_NAMES),
            "mosaic_height": PANEL_SIZE[1],
        },
        "batch_size": scores.batch_size,
        "git_commit": git_commit,
        "bundle_id": loaded.episode.bundle_id,
        "task_name": loaded.episode.task_name,
        "window_count": len(loaded.windows),
        "low_yes_no_mass_count": int(np.sum(scores.yes_no_mass < 0.5)),
        "timing": {
            "frame_loading_sec": loaded.frame_loading_sec,
            "preprocessing_sec": scores.preprocessing_sec,
            "gpu_sec": scores.gpu_sec,
            "scoring_wall_sec": scores.wall_sec,
        },
        "top_tokens_first_20_windows": scores.top_tokens,
        "runtime": runtime,
    }
    if provenance:
        payload["provenance"] = provenance
    _write_json_atomic(json_path, payload)
    return npz_path, json_path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def ground_duration_summary(episodes: Sequence[Episode]) -> dict[str, float | int]:
    durations = sorted(
        interval.end_sec - interval.start_sec
        for episode in episodes
        for interval in episode.ground_intervals
    )
    if not durations:
        return {"count": 0}
    quartiles = np.percentile(np.asarray(durations), [25, 50, 75])
    return {
        "count": len(durations),
        "median_sec": float(quartiles[1]),
        "q1_sec": float(quartiles[0]),
        "q3_sec": float(quartiles[2]),
        "iqr_sec": float(quartiles[2] - quartiles[0]),
        "min_sec": float(durations[0]),
        "max_sec": float(durations[-1]),
    }


def expected_window_count(
    episode: Episode,
    *,
    window_frames: int,
    stride_frames: int,
    frame_rate: float,
) -> int:
    estimated_frames = max(1, int(math.floor(episode.duration_sec * frame_rate)) + 1)
    if estimated_frames <= window_frames:
        return 1
    regular = 1 + (estimated_frames - window_frames) // stride_frames
    final_regular_start = (regular - 1) * stride_frames
    return regular + int(final_regular_start != estimated_frames - window_frames)


def _selected_episodes(
    episodes: Sequence[Episode],
    *,
    shard: int,
    num_shards: int,
    limit: int | None,
    event_only: bool,
) -> list[Episode]:
    if num_shards <= 0 or shard < 0 or shard >= num_shards:
        raise ValueError("Require 0 <= shard < num_shards")
    selected = [
        episode
        for index, episode in enumerate(episodes)
        if index % num_shards == shard and (not event_only or episode.ground_intervals)
    ]
    return selected[:limit] if limit is not None else selected


def _print_episode_log(loaded: LoadedEpisode, scores: ScoreResult) -> None:
    low_mass = int(np.sum(scores.yes_no_mass < 0.5))
    print(
        f"{loaded.episode.task_name}/{loaded.episode.bundle_id}: "
        f"windows={len(scores.margin)} wall={scores.wall_sec:.2f}s "
        f"frames={loaded.frame_loading_sec:.2f}s "
        f"preprocess={scores.preprocessing_sec:.2f}s gpu={scores.gpu_sec:.2f}s "
        f"batch={scores.batch_size} yes_no_mass<0.5={low_mass}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("VL_implemenation/config.json"))
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jpeg-workers", type=int, default=12)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-batches", type=int, default=20)
    parser.add_argument("--event-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parent.parent
    episodes = discover_episodes(config["data_root"])
    population = _selected_episodes(
        episodes,
        shard=args.shard,
        num_shards=args.num_shards,
        limit=None,
        event_only=args.event_only,
    )
    selected = _selected_episodes(
        episodes,
        shard=args.shard,
        num_shards=args.num_shards,
        limit=args.limit,
        event_only=args.event_only,
    )
    window_frames = int(round(args.window_sec * FRAME_RATE))
    stride_frames = int(round(args.stride_sec * FRAME_RATE))
    if window_frames <= 0 or window_frames % 2:
        raise ValueError(
            f"--window-sec must produce a positive even frame count at {FRAME_RATE:g} Hz"
        )
    if stride_frames <= 0:
        raise ValueError("--stride-sec is too short for the 5 Hz frame cache")
    cache_root = Path(config["output_root"]).expanduser().resolve() / "vlm_scores"
    resume_skipped = 0
    if not args.dry_run:
        pending = []
        for episode in selected:
            npz_path, _ = cache_paths(cache_root, episode)
            if npz_path.is_file():
                resume_skipped += 1
                print(
                    f"resume skip {episode.task_name}/{episode.bundle_id}",
                    flush=True,
                )
            else:
                pending.append(episode)
        selected = pending
    total_windows = sum(
        expected_window_count(
            episode,
            window_frames=window_frames,
            stride_frames=stride_frames,
            frame_rate=FRAME_RATE,
        )
        for episode in population
    )
    print(
        json.dumps(
            {
                "episode_count": len(selected),
                "resume_skipped_episode_count": resume_skipped,
                "projected_population_episode_count": len(population),
                "estimated_total_windows": total_windows,
                "ground_duration_statistics": ground_duration_summary(episodes),
                "window_sec": args.window_sec,
                "window_frames": window_frames,
                "stride_sec": args.stride_sec,
                "stride_frames": stride_frames,
                "effective_stride_sec": stride_frames / FRAME_RATE,
                "cache_root": str(cache_root),
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not selected:
        return

    scanner = QwenYesNoScanner(
        model_path=Path(config["model_path"]),
        model_id=str(config["model_id"]),
        frame_rate=FRAME_RATE,
        window_frames=window_frames,
        batch_size=args.batch_size,
        queue_depth=args.queue_depth,
        compile_model=not args.no_compile,
    )
    commit = _git_commit(repo_root)
    completed_windows = 0
    completed_batches = 0
    total_gpu_sec = 0.0
    total_preprocess_sec = 0.0
    total_frame_sec = 0.0
    mass_values: list[float] = []
    top_token_logs: list[dict[str, Any]] = []
    scan_started: float | None = None
    warmup_sec = 0.0

    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        future: Future[LoadedEpisode] | None = prefetch_pool.submit(
            load_episode_frames,
            selected[0],
            frame_cache_root=Path(config["frame_cache_root"]),
            frame_rate=FRAME_RATE,
            window_frames=window_frames,
            stride_frames=stride_frames,
            workers=args.jpeg_workers,
        )
        for index, episode in enumerate(selected):
            loaded = future.result()
            future = (
                prefetch_pool.submit(
                    load_episode_frames,
                    selected[index + 1],
                    frame_cache_root=Path(config["frame_cache_root"]),
                    frame_rate=FRAME_RATE,
                    window_frames=window_frames,
                    stride_frames=stride_frames,
                    workers=args.jpeg_workers,
                )
                if index + 1 < len(selected)
                else None
            )
            if index == 0 and not args.no_compile:
                warmup_started = time.perf_counter()
                scanner.score(
                    loaded,
                    max_batches=1,
                    top_token_limit=0,
                )
                warmup_sec = time.perf_counter() - warmup_started
                print(
                    f"compile warm-up: {warmup_sec:.2f}s "
                    f"(batch_size={scanner.batch_size})",
                    flush=True,
                )
                scan_started = time.perf_counter()
            elif scan_started is None:
                scan_started = time.perf_counter()
            remaining_batches = None
            if args.dry_run:
                remaining_batches = args.dry_run_batches - completed_batches
                if remaining_batches <= 0:
                    break
            scores = scanner.score(
                loaded,
                max_batches=remaining_batches,
                top_token_limit=max(0, 20 - completed_windows),
            )
            _print_episode_log(loaded, scores)
            completed_windows += len(scores.margin)
            completed_batches += scores.batch_count
            total_gpu_sec += scores.gpu_sec
            total_preprocess_sec += scores.preprocessing_sec
            total_frame_sec += loaded.frame_loading_sec
            mass_values.extend(scores.yes_no_mass.tolist())
            top_token_logs.extend(scores.top_tokens)
            if not args.dry_run:
                write_cache(
                    loaded,
                    scores,
                    cache_root=cache_root,
                    model_id=str(config["model_id"]),
                    window_sec=args.window_sec,
                    stride_sec=args.stride_sec,
                    git_commit=commit,
                    runtime=scanner.runtime,
                    prompt=scanner.prompt,
                )
            if args.dry_run and completed_batches >= args.dry_run_batches:
                break

    if scan_started is None:
        raise RuntimeError("No scan timing start was recorded")
    elapsed = time.perf_counter() - scan_started
    windows_per_sec = completed_windows / elapsed if elapsed else 0.0
    mass_array = np.asarray(mass_values, dtype=np.float64)
    summary = {
        "completed_windows": completed_windows,
        "completed_batches": completed_batches,
        "elapsed_sec": elapsed,
        "compile_warmup_sec_excluded": warmup_sec,
        "windows_per_sec": windows_per_sec,
        "frame_loading_sec_sum": total_frame_sec,
        "preprocessing_sec_sum": total_preprocess_sec,
        "gpu_sec_sum": total_gpu_sec,
        "projected_total_sec": (
            total_windows / windows_per_sec if windows_per_sec else None
        ),
        "projected_total_hours": (
            total_windows / windows_per_sec / 3600 if windows_per_sec else None
        ),
        "yes_no_mass": (
            {
                "min": float(np.min(mass_array)),
                "q1": float(np.percentile(mass_array, 25)),
                "median": float(np.median(mass_array)),
                "q3": float(np.percentile(mass_array, 75)),
                "max": float(np.max(mass_array)),
                "below_0_5_count": int(np.sum(mass_array < 0.5)),
                "below_0_5_fraction": float(np.mean(mass_array < 0.5)),
            }
            if mass_array.size
            else None
        ),
        "top_tokens_first_20_windows": top_token_logs[:20],
        "runtime": scanner.runtime,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
