"""FV-Action-style coarse-to-fine localization using cached coarse Qwen scores.

The existing 1 Hz, 2 s Qwen scan is reused as the coarse curve. This script
performs score-ranked temporal NMS, runs Qwen only on localized fine probes,
combines coarse and fine probabilities with the paper's geometric mean, and
returns one final point per episode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
from VL_implemenation.yesno_scan import (
    FRAME_RATE,
    LoadedEpisode,
    PROMPT,
    QwenYesNoScanner,
    Window,
    _read_mosaic_record,
    cache_paths,
)


DEFAULT_RADIUS_SEC = 8.0
DEFAULT_CANDIDATE_COUNT = 3
DEFAULT_FINE_COUNT = 8
DEFAULT_ALPHA = 0.5
DEFAULT_WINDOW_SEC = 2.0


@dataclass(frozen=True, slots=True)
class CoarsePlan:
    indices: np.ndarray
    centers: np.ndarray
    margins: np.ndarray
    probabilities: np.ndarray
    fine_centers: np.ndarray


@dataclass(slots=True)
class LoadedFineEpisode:
    loaded: LoadedEpisode
    plan: CoarsePlan


def greedy_temporal_nms(
    times: Sequence[float],
    scores: Sequence[float],
    *,
    radius_sec: float,
    max_candidates: int,
) -> list[int]:
    """Greedily select highest scores separated by more than the NMS radius."""

    if radius_sec < 0:
        raise ValueError("radius_sec must be non-negative")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    time_array = np.asarray(times, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if time_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("times and scores must be one-dimensional")
    if len(time_array) != len(score_array):
        raise ValueError("times and scores must have equal lengths")
    if not len(time_array):
        return []
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(score_array)):
        raise ValueError("times and scores must be finite")

    ranked = sorted(
        range(len(time_array)),
        key=lambda index: (-score_array[index], time_array[index], index),
    )
    selected: list[int] = []
    for index in ranked:
        if all(
            abs(float(time_array[index] - time_array[previous])) > radius_sec
            for previous in selected
        ):
            selected.append(index)
            if len(selected) == max_candidates:
                break
    return selected


def build_coarse_plan(
    centers: Sequence[float],
    margins: Sequence[float],
    *,
    duration_sec: float,
    radius_sec: float,
    candidate_count: int,
    fine_count: int,
) -> CoarsePlan:
    """Select coarse candidates and place uniform fine probes around each."""

    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if fine_count <= 0:
        raise ValueError("fine_count must be positive")
    center_array = np.asarray(centers, dtype=np.float64)
    margin_array = np.asarray(margins, dtype=np.float64)
    probabilities = _sigmoid(margin_array)
    indices = np.asarray(
        greedy_temporal_nms(
            center_array,
            probabilities,
            radius_sec=radius_sec,
            max_candidates=candidate_count,
        ),
        dtype=np.int64,
    )
    candidate_centers = center_array[indices]
    candidate_margins = margin_array[indices]
    candidate_probabilities = probabilities[indices]
    fine_centers = np.stack(
        [
            np.linspace(
                max(0.0, float(center) - radius_sec),
                min(duration_sec, float(center) + radius_sec),
                fine_count,
                dtype=np.float64,
            )
            for center in candidate_centers
        ]
    )
    return CoarsePlan(
        indices=indices,
        centers=candidate_centers,
        margins=candidate_margins,
        probabilities=candidate_probabilities,
        fine_centers=fine_centers,
    )


def select_probe_frame_indices(
    times: Sequence[float],
    *,
    center_sec: float,
    frame_count: int,
) -> tuple[int, ...]:
    """Choose a contiguous fixed-size probe whose mean time is nearest center."""

    if frame_count <= 0 or frame_count % 2:
        raise ValueError("frame_count must be a positive even number")
    time_array = np.asarray(times, dtype=np.float64)
    if time_array.ndim != 1 or not len(time_array):
        raise ValueError("times must be a non-empty one-dimensional sequence")
    if len(time_array) <= frame_count:
        indices = list(range(len(time_array)))
        indices.extend([len(time_array) - 1] * (frame_count - len(time_array)))
        return tuple(indices)

    insertion = int(np.searchsorted(time_array, center_sec, side="left"))
    approximate = insertion - frame_count // 2
    candidates = {
        min(max(0, approximate + offset), len(time_array) - frame_count)
        for offset in range(-2, 3)
    }
    start = min(
        candidates,
        key=lambda value: (
            abs(float(np.mean(time_array[value : value + frame_count])) - center_sec),
            value,
        ),
    )
    return tuple(range(start, start + frame_count))


def load_fine_episode(
    episode: Episode,
    *,
    plan: CoarsePlan,
    frame_cache_root: Path,
    frame_rate: float,
    window_frames: int,
    workers: int,
) -> LoadedFineEpisode:
    """Read only JPEGs needed by the localized fine probes."""

    started = time.perf_counter()
    timelines = {
        camera: camera_timeline(
            camera_cache_directory(frame_cache_root, episode, camera, frame_rate)
        )
        for camera in CAMERA_NAMES
    }
    records = build_mosaic_timeline(timelines, output_fps=frame_rate)
    times = np.asarray(
        [float(record["trajectory_sec"]) for record in records],
        dtype=np.float64,
    )
    source_windows = [
        select_probe_frame_indices(
            times,
            center_sec=float(center),
            frame_count=window_frames,
        )
        for center in plan.fine_centers.reshape(-1)
    ]
    needed_indices = sorted({index for window in source_windows for index in window})
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        mosaics = list(
            pool.map(
                _read_mosaic_record,
                [records[index] for index in needed_indices],
            )
        )
    local_index = {
        source_index: index for index, source_index in enumerate(needed_indices)
    }
    frames = np.stack(mosaics).astype(np.uint8, copy=False)
    windows = []
    for source_indices in source_windows:
        frame_times = times[list(source_indices)]
        windows.append(
            Window(
                indices=tuple(local_index[index] for index in source_indices),
                start_sec=float(frame_times[0]),
                end_sec=float(frame_times[-1]),
                center_sec=float(np.mean(frame_times)),
            )
        )
    return LoadedFineEpisode(
        loaded=LoadedEpisode(
            episode=episode,
            frames=frames,
            times=times[needed_indices],
            windows=windows,
            frame_loading_sec=time.perf_counter() - started,
        ),
        plan=plan,
    )


def select_final_peak(
    plan: CoarsePlan,
    fine_margins: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Apply geometric joint ranking and discrete fine-curve argmax."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    margins = np.asarray(fine_margins, dtype=np.float64)
    if margins.shape != plan.fine_centers.shape:
        raise ValueError("fine margin shape does not match the coarse plan")
    fine_probabilities = _sigmoid(margins)
    fine_argmax = np.argmax(fine_probabilities, axis=1)
    fine_max = fine_probabilities[
        np.arange(len(fine_probabilities)),
        fine_argmax,
    ]
    joint = np.power(plan.probabilities, alpha) * np.power(
        fine_max,
        1.0 - alpha,
    )
    winner = int(np.argmax(joint))
    winner_fine_index = int(fine_argmax[winner])
    return {
        "candidate_joint_scores": joint,
        "candidate_fine_max_probabilities": fine_max,
        "candidate_fine_argmax": fine_argmax,
        "winning_candidate_index": winner,
        "winning_fine_index": winner_fine_index,
        "prediction_sec": float(
            plan.fine_centers[winner, winner_fine_index]
        ),
    }


def fine_cache_paths(
    output_root: Path,
    episode: Episode,
    *,
    subdirectory: str | None = "fine_scores",
) -> tuple[Path, Path]:
    directory = output_root
    if subdirectory:
        directory = directory / subdirectory
    directory = directory / episode.task_name
    return directory / f"{episode.bundle_id}.npz", directory / f"{episode.bundle_id}.json"


def run_scan(
    episodes: Sequence[Episode],
    *,
    coarse_cache_root: Path,
    frame_cache_root: Path,
    output_root: Path,
    model_path: Path,
    model_id: str,
    radius_sec: float,
    candidate_count: int,
    fine_count: int,
    alpha: float,
    window_sec: float,
    batch_size: int,
    workers: int,
    queue_depth: int,
    compile_model: bool,
    resume: bool,
    prompt: str = PROMPT,
    scanner: QwenYesNoScanner | None = None,
    output_subdirectory: str | None = "fine_scores",
    provenance: dict[str, Any] | None = None,
) -> QwenYesNoScanner | None:
    """Run localized fine probes, loading the model once."""

    window_frames = int(round(window_sec * FRAME_RATE))
    if window_frames <= 0 or window_frames % 2:
        raise ValueError("window_sec must produce a positive even frame count")
    pending = []
    for episode in episodes:
        output_path, _ = fine_cache_paths(
            output_root,
            episode,
            subdirectory=output_subdirectory,
        )
        if resume and output_path.is_file():
            continue
        pending.append(episode)
    if not pending:
        print("All requested fine-score caches already exist.", flush=True)
        return scanner

    if scanner is None:
        scanner = QwenYesNoScanner(
            model_path=model_path,
            model_id=model_id,
            prompt=prompt,
            frame_rate=FRAME_RATE,
            window_frames=window_frames,
            batch_size=batch_size,
            queue_depth=queue_depth,
            compile_model=compile_model,
        )
    else:
        scanner.set_prompt(prompt)
    git_commit = _git_commit(Path(__file__).resolve().parent.parent)
    for episode_index, episode in enumerate(pending, start=1):
        coarse_path, _ = cache_paths(coarse_cache_root, episode)
        with np.load(coarse_path, allow_pickle=False) as cached:
            coarse_centers = np.asarray(cached["center_sec"], dtype=np.float64)
            coarse_margins = np.asarray(cached["margin"], dtype=np.float64)
        plan = build_coarse_plan(
            coarse_centers,
            coarse_margins,
            duration_sec=episode.duration_sec,
            radius_sec=radius_sec,
            candidate_count=candidate_count,
            fine_count=fine_count,
        )
        fine_loaded = load_fine_episode(
            episode,
            plan=plan,
            frame_cache_root=frame_cache_root,
            frame_rate=FRAME_RATE,
            window_frames=window_frames,
            workers=workers,
        )
        scores = scanner.score(
            fine_loaded.loaded,
            top_token_limit=0,
        )
        fine_shape = plan.fine_centers.shape
        fine_margins = scores.margin.reshape(fine_shape)
        fine_masses = scores.yes_no_mass.reshape(fine_shape)
        selection = select_final_peak(plan, fine_margins, alpha=alpha)
        _write_fine_cache(
            episode,
            output_root=output_root,
            plan=plan,
            loaded=fine_loaded.loaded,
            fine_margins=fine_margins,
            fine_masses=fine_masses,
            selection=selection,
            radius_sec=radius_sec,
            candidate_count=candidate_count,
            fine_count=fine_count,
            alpha=alpha,
            window_sec=window_sec,
            model_id=model_id,
            git_commit=git_commit,
            scanner=scanner,
            scores=scores,
            prompt=prompt,
            provenance=provenance,
            output_subdirectory=output_subdirectory,
        )
        print(
            f"[{episode_index}/{len(pending)}] "
            f"{episode.task_name}/{episode.bundle_id}: "
            f"prediction={selection['prediction_sec']:.3f}s "
            f"frames={fine_loaded.loaded.frame_loading_sec:.2f}s "
            f"gpu={scores.gpu_sec:.2f}s",
            flush=True,
        )
    return scanner


def _write_fine_cache(
    episode: Episode,
    *,
    output_root: Path,
    plan: CoarsePlan,
    loaded: LoadedEpisode,
    fine_margins: np.ndarray,
    fine_masses: np.ndarray,
    selection: dict[str, Any],
    radius_sec: float,
    candidate_count: int,
    fine_count: int,
    alpha: float,
    window_sec: float,
    model_id: str,
    git_commit: str,
    scanner: QwenYesNoScanner,
    scores: Any,
    prompt: str = PROMPT,
    provenance: dict[str, Any] | None = None,
    output_subdirectory: str | None = "fine_scores",
) -> None:
    npz_path, json_path = fine_cache_paths(
        output_root,
        episode,
        subdirectory=output_subdirectory,
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    observed_centers = np.asarray(
        [window.center_sec for window in loaded.windows],
        dtype=np.float64,
    ).reshape(plan.fine_centers.shape)
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            coarse_candidate_index=plan.indices,
            coarse_candidate_sec=plan.centers,
            coarse_candidate_margin=plan.margins,
            coarse_candidate_p_yes=plan.probabilities,
            fine_center_sec=plan.fine_centers,
            fine_observed_center_sec=observed_centers,
            fine_margin=fine_margins,
            fine_p_yes=_sigmoid(fine_margins),
            fine_yes_no_mass=fine_masses,
            candidate_joint_score=selection["candidate_joint_scores"],
            candidate_fine_argmax=selection["candidate_fine_argmax"],
            prediction_sec=np.asarray(selection["prediction_sec"]),
            winning_candidate_index=np.asarray(
                selection["winning_candidate_index"]
            ),
            winning_fine_index=np.asarray(selection["winning_fine_index"]),
        )
    temporary.replace(npz_path)
    _write_json_atomic(
        json_path,
        {
            "status": "complete",
            "bundle_id": episode.bundle_id,
            "task_name": episode.task_name,
            "model_id": model_id,
            "git_commit": git_commit,
            "prompt": prompt,
            "method": {
                "coarse_source": "cached 1 Hz Qwen Yes/No curve",
                "nms": "greedy score-ranked temporal NMS",
                "radius_sec": radius_sec,
                "candidate_count": candidate_count,
                "fine_count": fine_count,
                "fine_interval": "[candidate-radius, candidate+radius], clipped",
                "alpha": alpha,
                "joint_score": "coarse_p_yes^alpha * fine_max_p_yes^(1-alpha)",
                "peak_readout": "discrete fine-curve argmax",
                "output_count": 1,
                "window_sec": window_sec,
                "fine_probe_adaptation": (
                    "same 10-frame 960x256 three-camera mosaic as the cached "
                    "coarse measurement layer"
                ),
            },
            "prediction_sec": selection["prediction_sec"],
            "timing": {
                "frame_loading_sec": loaded.frame_loading_sec,
                "preprocessing_sec": scores.preprocessing_sec,
                "gpu_sec": scores.gpu_sec,
                "wall_sec": scores.wall_sec,
            },
            "runtime": scanner.runtime,
            "provenance": provenance or {},
        },
    )


def evaluate(
    episodes: Sequence[Episode],
    *,
    output_root: Path,
    bootstrap_samples: int,
    null_draws: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the one final point on event-bearing ABC episodes."""

    rows = []
    missing = 0
    for episode in episodes:
        npz_path, _ = fine_cache_paths(output_root, episode)
        if not npz_path.is_file():
            missing += 1
            continue
        with np.load(npz_path, allow_pickle=False) as cached:
            prediction = float(cached["prediction_sec"])
            coarse_prediction = float(cached["coarse_candidate_sec"][0])
        intervals = [interval.to_dict() for interval in episode.ground_intervals]
        rows.append(
            {
                "task_name": episode.task_name,
                "bundle_id": episode.bundle_id,
                "duration_sec": episode.duration_sec,
                "ground_intervals": intervals,
                "coarse_prediction_sec": coarse_prediction,
                "prediction_sec": prediction,
                "hit": _contains_any(intervals, prediction),
                "coarse_hit": _contains_any(intervals, coarse_prediction),
            }
        )
    event_rows = [row for row in rows if row["ground_intervals"]]
    observed = _hit_rate(event_rows, "prediction_sec")
    coarse = _hit_rate(event_rows, "coarse_prediction_sec")
    bootstrap = _bootstrap_hit_rates(
        event_rows,
        samples=bootstrap_samples,
        seed=seed,
    )
    null = _chance_null(
        event_rows,
        draws=null_draws,
        seed=seed,
    )
    per_type = {}
    for event_type in ("insertion", "removal"):
        typed_rows = [
            row
            for row in event_rows
            if any(
                interval["event_type"] == event_type
                for interval in row["ground_intervals"]
            )
        ]
        per_type[event_type] = {
            "trajectory_count": len(typed_rows),
            "hit_at_1": (
                sum(
                    any(
                        interval["event_type"] == event_type
                        and _contains(interval, row["prediction_sec"])
                        for interval in row["ground_intervals"]
                    )
                    for row in typed_rows
                )
                / len(typed_rows)
                if typed_rows
                else None
            ),
        }
    summary = {
        "protocol": {
            "metric": "trajectory Hit@1",
            "interval_semantics": "half-open [start_sec, end_sec)",
            "population": "episodes with at least one insert/remove interval",
            "bootstrap_episode_resamples": bootstrap_samples,
            "chance_draws_per_episode": null_draws,
            "seed": seed,
        },
        "episode_cache_count": len(rows),
        "missing_cache_count": missing,
        "event_episode_count": len(event_rows),
        "coarse_hit_at_1": coarse,
        "coarse_to_fine_hit_at_1": observed,
        "coarse_to_fine_bootstrap_95ci": bootstrap["coarse_to_fine"],
        "paired_difference_vs_coarse": bootstrap["difference"],
        "chance_hit_at_1": null,
        "per_type": per_type,
    }
    _write_json_atomic(output_root / "evaluation.json", summary)
    _write_jsonl_atomic(output_root / "predictions.jsonl", rows)
    return summary


def _hit_rate(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(
        _contains_any(row["ground_intervals"], float(row[key]))
        for row in rows
    ) / len(rows)


def _bootstrap_hit_rates(
    rows: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0 or not rows:
        return {"coarse_to_fine": [], "difference": {"value": None, "ci": []}}
    rng = np.random.default_rng(seed)
    fine_values = []
    differences = []
    observed_difference = (
        float(_hit_rate(rows, "prediction_sec"))
        - float(_hit_rate(rows, "coarse_prediction_sec"))
    )
    for _ in range(samples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        fine = float(_hit_rate(sampled, "prediction_sec"))
        coarse = float(_hit_rate(sampled, "coarse_prediction_sec"))
        fine_values.append(fine)
        differences.append(fine - coarse)
    return {
        "coarse_to_fine": _percentile_interval(fine_values),
        "difference": {
            "value": observed_difference,
            "ci": _percentile_interval(differences),
        },
    }


def _chance_null(
    rows: Sequence[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if draws <= 0 or not rows:
        return {"mean": None, "simulation_95ci": []}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        hits = [
            _contains_any(
                row["ground_intervals"],
                float(rng.uniform(0.0, row["duration_sec"])),
            )
            for row in rows
        ]
        values.append(float(np.mean(hits)))
    return {
        "mean": float(np.mean(values)),
        "simulation_95ci": _percentile_interval(values),
    }


def _contains(interval: dict[str, Any], point: float) -> bool:
    return float(interval["start_sec"]) <= point < float(interval["end_sec"])


def _contains_any(intervals: Sequence[dict[str, Any]], point: float) -> bool:
    return any(_contains(interval, point) for interval in intervals)


def _sigmoid(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0)))


def _percentile_interval(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
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


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["scan", "evaluate", "full"],
        default="full",
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("VL_implemenation/config.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--radius-sec", type=float, default=DEFAULT_RADIUS_SEC)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
    )
    parser.add_argument("--fine-count", type=int, default=DEFAULT_FINE_COUNT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--jpeg-workers", type=int, default=12)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    external_output = Path(config["output_root"]).expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else external_output / "coarse_to_fine"
    )
    episodes = discover_episodes(config["data_root"])
    if args.limit is not None:
        episodes = episodes[: args.limit]
    coarse_cache_root = external_output / "vlm_scores"
    missing_coarse = [
        episode
        for episode in episodes
        if not cache_paths(coarse_cache_root, episode)[0].is_file()
    ]
    if missing_coarse:
        raise FileNotFoundError(
            f"Missing {len(missing_coarse)} coarse VLM caches"
        )
    if args.command in {"scan", "full"}:
        run_scan(
            episodes,
            coarse_cache_root=coarse_cache_root,
            frame_cache_root=Path(config["frame_cache_root"]),
            output_root=output_root,
            model_path=Path(config["model_path"]),
            model_id=str(config["model_id"]),
            radius_sec=args.radius_sec,
            candidate_count=args.candidate_count,
            fine_count=args.fine_count,
            alpha=args.alpha,
            window_sec=args.window_sec,
            batch_size=args.batch_size,
            workers=args.jpeg_workers,
            queue_depth=args.queue_depth,
            compile_model=not args.no_compile,
            resume=not args.no_resume,
        )
    if args.command in {"evaluate", "full"}:
        summary = evaluate(
            episodes,
            output_root=output_root,
            bootstrap_samples=args.bootstrap_samples,
            null_draws=args.null_draws,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
