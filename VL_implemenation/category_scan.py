"""Run resumable Qwen yes/no scans for the seven frozen ABC categories."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from VL_implemenation.abc_data import Episode, discover_episodes
from VL_implemenation.category_protocol import (
    DEFAULT_EXPERIMENT_CONFIG,
    load_experiment_config,
    render_category_prompt,
    selected_categories,
)
from VL_implemenation.yesno_scan import (
    FRAME_RATE,
    LoadedEpisode,
    QwenYesNoScanner,
    cache_paths,
    expected_window_count,
    load_episode_frames,
    write_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("VL_implemenation/config.json"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG,
    )
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--jpeg-workers", type=int)
    parser.add_argument("--queue-depth", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-batches", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = load_experiment_config(args.experiment_config)
    categories = selected_categories(experiment, args.categories)
    scan_config = experiment["scan"]
    frame_rate = float(scan_config["frame_rate"])
    if frame_rate != FRAME_RATE:
        raise ValueError(
            f"Category experiment requires the existing {FRAME_RATE:g} Hz cache"
        )
    window_sec = float(scan_config["window_sec"])
    stride_sec = float(scan_config["stride_sec"])
    window_frames = int(round(window_sec * frame_rate))
    stride_frames = int(round(stride_sec * frame_rate))
    if window_frames <= 0 or window_frames % 2:
        raise ValueError("Category scan window must contain a positive even frame count")
    if stride_frames <= 0:
        raise ValueError("Category scan stride must contain at least one frame")
    episodes = _sharded_episodes(
        discover_episodes(base_config["data_root"]),
        shard=args.shard,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        print("No episodes selected", flush=True)
        return

    batch_size = int(args.batch_size or scan_config["batch_size"])
    jpeg_workers = int(args.jpeg_workers or scan_config["jpeg_workers"])
    queue_depth = int(args.queue_depth or scan_config["queue_depth"])
    output_root = Path(base_config["output_root"]).expanduser().resolve()
    commit = _git_commit(Path(__file__).resolve().parent.parent)
    scanner: QwenYesNoScanner | None = None

    for category in categories:
        prompt = render_category_prompt(experiment, category)
        if scanner is None:
            scanner = QwenYesNoScanner(
                model_path=Path(base_config["model_path"]),
                model_id=str(base_config["model_id"]),
                prompt=prompt,
                frame_rate=frame_rate,
                window_frames=window_frames,
                batch_size=batch_size,
                queue_depth=queue_depth,
                compile_model=not args.no_compile,
            )
        else:
            scanner.set_prompt(prompt)
        summary = scan_category(
            category,
            prompt=prompt,
            episodes=episodes,
            scanner=scanner,
            frame_cache_root=Path(base_config["frame_cache_root"]),
            output_root=output_root,
            window_sec=window_sec,
            stride_sec=stride_sec,
            window_frames=window_frames,
            stride_frames=stride_frames,
            jpeg_workers=jpeg_workers,
            dry_run=args.dry_run,
            dry_run_batches=args.dry_run_batches,
            model_id=str(base_config["model_id"]),
            git_commit=commit,
            provenance={
                "category": category,
                "experiment_config_path": experiment["_config_path"],
                "experiment_config_sha256": experiment["_config_sha256"],
                "taxonomy_sha256": experiment["_taxonomy_sha256"],
            },
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if not args.dry_run:
            summary_path = (
                output_root
                / "category_scan_summaries"
                / f"{category}.json"
            )
            _write_json_atomic(summary_path, summary)


def scan_category(
    category: str,
    *,
    prompt: str,
    episodes: Sequence[Episode],
    scanner: QwenYesNoScanner,
    frame_cache_root: Path,
    output_root: Path,
    window_sec: float,
    stride_sec: float,
    window_frames: int,
    stride_frames: int,
    jpeg_workers: int,
    dry_run: bool,
    dry_run_batches: int,
    model_id: str,
    git_commit: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Scan one complete category and return timing/quality diagnostics."""

    cache_root = output_root / "vlm_scores_by_category" / category
    pending: list[Episode] = []
    skipped = 0
    for episode in episodes:
        npz_path, _ = cache_paths(cache_root, episode)
        if not dry_run and npz_path.is_file():
            skipped += 1
        else:
            pending.append(episode)
    total_windows = sum(
        expected_window_count(
            episode,
            window_frames=window_frames,
            stride_frames=stride_frames,
            frame_rate=FRAME_RATE,
        )
        for episode in episodes
    )
    if not pending:
        return {
            "status": "complete",
            "category": category,
            "episode_count": len(episodes),
            "resume_skipped_episode_count": skipped,
            "completed_episode_count": 0,
            "estimated_total_windows": total_windows,
            "cache_root": str(cache_root),
            "prompt": prompt,
        }

    started = time.perf_counter()
    completed_windows = 0
    completed_batches = 0
    completed_episodes = 0
    frame_sec = 0.0
    preprocessing_sec = 0.0
    gpu_sec = 0.0
    mass_values: list[float] = []
    top_tokens: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future[LoadedEpisode] | None = _submit_load(
            pool,
            pending[0],
            frame_cache_root=frame_cache_root,
            window_frames=window_frames,
            stride_frames=stride_frames,
            jpeg_workers=jpeg_workers,
        )
        for index, episode in enumerate(pending):
            if future is None:
                break
            loaded = future.result()
            future = (
                _submit_load(
                    pool,
                    pending[index + 1],
                    frame_cache_root=frame_cache_root,
                    window_frames=window_frames,
                    stride_frames=stride_frames,
                    jpeg_workers=jpeg_workers,
                )
                if index + 1 < len(pending)
                else None
            )
            max_batches = None
            if dry_run:
                remaining = dry_run_batches - completed_batches
                if remaining <= 0:
                    break
                max_batches = remaining
            scores = scanner.score(
                loaded,
                max_batches=max_batches,
                top_token_limit=max(0, 20 - completed_windows),
            )
            completed_episodes += 1
            completed_windows += len(scores.margin)
            completed_batches += scores.batch_count
            frame_sec += loaded.frame_loading_sec
            preprocessing_sec += scores.preprocessing_sec
            gpu_sec += scores.gpu_sec
            mass_values.extend(scores.yes_no_mass.tolist())
            top_tokens.extend(scores.top_tokens)
            print(
                f"[{category}] {index + 1}/{len(pending)} "
                f"{episode.task_name}/{episode.bundle_id}: "
                f"windows={len(scores.margin)} gpu={scores.gpu_sec:.2f}s",
                flush=True,
            )
            if not dry_run:
                write_cache(
                    loaded,
                    scores,
                    cache_root=cache_root,
                    model_id=model_id,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    git_commit=git_commit,
                    runtime=scanner.runtime,
                    prompt=prompt,
                    provenance=provenance,
                )
            if dry_run and completed_batches >= dry_run_batches:
                break
    elapsed = time.perf_counter() - started
    windows_per_sec = completed_windows / elapsed if elapsed > 0 else 0.0
    masses = np.asarray(mass_values, dtype=np.float64)
    return {
        "status": "dry_run" if dry_run else "complete",
        "category": category,
        "prompt": prompt,
        "episode_count": len(episodes),
        "pending_episode_count": len(pending),
        "resume_skipped_episode_count": skipped,
        "completed_episode_count": completed_episodes,
        "completed_windows": completed_windows,
        "completed_batches": completed_batches,
        "estimated_total_windows": total_windows,
        "elapsed_sec": elapsed,
        "windows_per_sec": windows_per_sec,
        "projected_total_hours": (
            total_windows / windows_per_sec / 3600
            if windows_per_sec > 0
            else None
        ),
        "frame_loading_sec_sum": frame_sec,
        "preprocessing_sec_sum": preprocessing_sec,
        "gpu_sec_sum": gpu_sec,
        "cache_root": str(cache_root),
        "yes_no_mass": (
            {
                "min": float(np.min(masses)),
                "median": float(np.median(masses)),
                "max": float(np.max(masses)),
                "below_0_5_count": int(np.sum(masses < 0.5)),
                "below_0_5_fraction": float(np.mean(masses < 0.5)),
            }
            if masses.size
            else None
        ),
        "top_tokens_first_20_windows": top_tokens[:20],
        "provenance": provenance,
        "runtime": scanner.runtime,
    }


def _submit_load(
    pool: ThreadPoolExecutor,
    episode: Episode,
    *,
    frame_cache_root: Path,
    window_frames: int,
    stride_frames: int,
    jpeg_workers: int,
) -> Future[LoadedEpisode]:
    return pool.submit(
        load_episode_frames,
        episode,
        frame_cache_root=frame_cache_root,
        frame_rate=FRAME_RATE,
        window_frames=window_frames,
        stride_frames=stride_frames,
        workers=jpeg_workers,
    )


def _sharded_episodes(
    episodes: Sequence[Episode],
    *,
    shard: int,
    num_shards: int,
) -> list[Episode]:
    if num_shards <= 0 or shard < 0 or shard >= num_shards:
        raise ValueError("Require 0 <= shard < num_shards")
    return [
        episode
        for index, episode in enumerate(episodes)
        if index % num_shards == shard
    ]


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
