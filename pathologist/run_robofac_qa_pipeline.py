"""Run RoboFAC QA evaluation with comparable keyframe selectors."""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.adapters.robofac_bundle import RoboFACBundleAdapter
from pathologist.config import load_pipeline_config
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac import (
    DEFAULT_ROBOFAC_MCQ_TYPES,
    RoboFACPrediction,
    aggregate_predictions,
    build_qa_prompt,
    load_robofac_qa_items,
    score_failure_detection_answer,
    score_mcq_answer,
)
from pathologist.qa.robofac_client import OpenAIRoboFACQAClient, RoboFACQAClient, StaticRoboFACQAClient
from pathologist.qa.robofac_selectors import select_keyframes


ROBOFAC_HF_SIMULATION_BASE_URL = (
    "https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset/resolve/main/simulation_data"
)


class RemoteFileUnavailableError(RuntimeError):
    """Raised when a selectively fetched RoboFAC file is unavailable."""


def run_robofac_qa_pipeline(
    annotation_path: str | Path,
    h5_path: str | Path | None = None,
    traj_json_path: str | Path | None = None,
    video_root: str | Path | None = None,
    data_root: str | Path | None = None,
    dataset_root: str | Path | None = None,
    video_path: str | Path | None = None,
    output_dir: str | Path = "robofac_outputs",
    question_types: list[str] | None = None,
    selectors: list[str] | None = None,
    frame_budget: int = 5,
    max_videos: int | None = None,
    shuffle_videos: bool = False,
    video_sample_seed: int | None = None,
    config_path: str | Path | None = None,
    qa_client: RoboFACQAClient | None = None,
    openai_model: str | None = None,
    skip_vlm: bool = False,
    download_missing: bool = False,
    hf_base_url: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run RoboFAC QA for available local samples."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_question_types = question_types or list(DEFAULT_ROBOFAC_MCQ_TYPES)
    resolved_selectors = selectors or ["uniform", "kite_flow", "metric"]
    config = load_pipeline_config(config_path=config_path)
    annotation_paths = _resolve_annotation_paths(annotation_path)
    items = [
        item
        for current_annotation_path in annotation_paths
        for item in load_robofac_qa_items(
            current_annotation_path,
            question_types=set(resolved_question_types),
        )
    ]
    _log(
        verbose,
        (
            f"Loaded {len(items)} QA item(s) from {len(annotation_paths)} annotation file(s) "
            f"for question types {resolved_question_types}."
        ),
    )
    items_by_video: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        items_by_video[item.video_id].append(item)
    video_work_items = _video_items_for_sampling(
        items_by_video,
        shuffle_videos=shuffle_videos,
        video_sample_seed=video_sample_seed,
    )

    adapter = RoboFACBundleAdapter()
    client = qa_client
    if client is None:
        client = StaticRoboFACQAClient("") if skip_vlm else OpenAIRoboFACQAClient(model=openai_model)

    predictions: list[RoboFACPrediction] = []
    manifest: list[dict[str, Any]] = []
    alignment_debug: list[dict[str, Any]] = []
    processed_video_count = 0
    explicit_video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
    resolved_dataset_root = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else _infer_dataset_root(annotation_paths)
    )
    resolved_data_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else resolved_dataset_root
    )
    resolved_video_root = (
        Path(video_root).expanduser().resolve()
        if video_root is not None
        else resolved_data_root
    )

    for video_id, video_items in video_work_items:
        if max_videos is not None and processed_video_count >= max_videos:
            break
        _log(
            verbose,
            f"[{video_id}] Resolving files for {len(video_items)} QA item(s).",
        )
        try:
            candidate_video_path = _resolve_video_path(
                video_id=video_id,
                video_rel_path=video_items[0].video_rel_path,
                video_root=resolved_video_root,
                explicit_video_path=explicit_video_path,
                download_missing=download_missing,
                hf_base_url=hf_base_url,
                download_dir=output_path / "downloaded_videos",
                verbose=verbose,
            )
        except RemoteFileUnavailableError as exc:
            _log(verbose, f"[{video_id}] Skipped: video download failed ({exc}).")
            alignment_debug.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": "video_download_failed",
                    "message": str(exc),
                    "video_rel_path": video_items[0].video_rel_path,
                }
            )
            continue
        if candidate_video_path is None:
            _log(verbose, f"[{video_id}] Skipped: video not found.")
            alignment_debug.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": "video_not_found",
                    "video_rel_path": video_items[0].video_rel_path,
                }
            )
            continue
        _log(verbose, f"[{video_id}] Video: {candidate_video_path}")
        try:
            trajectory_files = _resolve_trajectory_files(
                video_rel_path=video_items[0].video_rel_path,
                explicit_h5_path=Path(h5_path).expanduser().resolve() if h5_path is not None else None,
                explicit_traj_json_path=(
                    Path(traj_json_path).expanduser().resolve()
                    if traj_json_path is not None
                    else None
                ),
                data_root=resolved_data_root,
                download_missing=download_missing,
                hf_base_url=hf_base_url,
                download_dir=output_path / "downloaded_data",
                verbose=verbose,
                allow_missing=True,
            )
        except Exception as exc:
            _log(verbose, f"[{video_id}] Skipped: trajectory files unavailable ({type(exc).__name__}: {exc}).")
            alignment_debug.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "video_path": str(candidate_video_path),
                    "video_rel_path": video_items[0].video_rel_path,
                }
            )
            continue
        candidate_h5_path = None
        candidate_traj_json_path = None
        if trajectory_files is not None:
            candidate_h5_path, candidate_traj_json_path = trajectory_files
            _log(verbose, f"[{video_id}] H5: {candidate_h5_path}")
            _log(verbose, f"[{video_id}] Trajectory JSON: {candidate_traj_json_path}")
        else:
            _log(verbose, f"[{video_id}] Trajectory files not found; using video-only selectors.")
        try:
            if candidate_h5_path is not None and candidate_traj_json_path is not None:
                try:
                    bundle = adapter.load(
                        video_path=candidate_video_path,
                        h5_path=candidate_h5_path,
                        traj_json_path=candidate_traj_json_path,
                        frame_cache_dir=output_path / "frame_cache" / video_id,
                        video_id=video_id,
                    )
                except ModuleNotFoundError as exc:
                    if not _has_video_only_selector(resolved_selectors):
                        raise
                    _log(
                        verbose,
                        (
                            f"[{video_id}] Telemetry adapter unavailable ({exc}); "
                            "falling back to video-only selectors."
                        ),
                    )
                    bundle = adapter.load_video_only(
                        video_path=candidate_video_path,
                        frame_cache_dir=output_path / "frame_cache" / video_id,
                        video_id=video_id,
                        task_name=video_items[0].task,
                    )
            else:
                bundle = adapter.load_video_only(
                    video_path=candidate_video_path,
                    frame_cache_dir=output_path / "frame_cache" / video_id,
                    video_id=video_id,
                    task_name=video_items[0].task,
                )
        except Exception as exc:
            _log(verbose, f"[{video_id}] Skipped: adapter failed ({type(exc).__name__}: {exc}).")
            alignment_debug.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "video_path": str(candidate_video_path),
                }
            )
            continue

        video_selectors = _selectors_available_for_bundle(resolved_selectors, bundle.raw_metadata)
        if not video_selectors:
            _log(verbose, f"[{video_id}] Skipped: no requested selector is available.")
            alignment_debug.append(
                {
                    "video_id": video_id,
                    "status": "skipped",
                    "reason": "selectors_unavailable",
                    "video_path": str(candidate_video_path),
                    "requested_selectors": resolved_selectors,
                    "video_only": bool(bundle.raw_metadata.get("video_only")),
                }
            )
            continue

        processed_video_count += 1
        _log(
            verbose,
            (
                f"[{video_id}] Mapped to episode {bundle.raw_metadata.get('episode_id')} "
                f"/ {bundle.raw_metadata.get('traj_key')} | frames={bundle.raw_metadata.get('frame_count')} "
                f"actions={bundle.raw_metadata.get('action_count')} fps={bundle.raw_metadata.get('fps')}"
            ),
        )
        manifest.append(
            {
                "video_id": video_id,
                "task": video_items[0].task,
                "annotation_types": [item.question_type for item in video_items],
                "video_path": str(candidate_video_path),
                "h5_path": str(candidate_h5_path) if candidate_h5_path is not None else None,
                "traj_json_path": str(candidate_traj_json_path) if candidate_traj_json_path is not None else None,
                "video_only": bool(bundle.raw_metadata.get("video_only")),
                "selectors": video_selectors,
                "omitted_selectors": [
                    selector for selector in resolved_selectors if selector not in video_selectors
                ],
                "matched_episode_id": bundle.raw_metadata.get("episode_id"),
                "matched_traj_key": bundle.raw_metadata.get("traj_key"),
                "frame_count": bundle.raw_metadata.get("frame_count"),
                "action_count": bundle.raw_metadata.get("action_count"),
                "state_count": bundle.raw_metadata.get("state_count"),
                "fps": bundle.raw_metadata.get("fps"),
                "alignment_validation_notes": bundle.raw_metadata.get("alignment_validation_notes", []),
            }
        )
        metric_profile = None
        if "metric" in video_selectors:
            _log(verbose, f"[{video_id}] Computing metric profile for metric selector.")
            metric_profile = MetricEngine(config=config).compute(bundle)
        selected_frames_by_selector = {
            selector_name: select_keyframes(
                selector_name=selector_name,
                bundle=bundle,
                frame_budget=frame_budget,
                metric_profile=metric_profile,
                config=config,
            )
            for selector_name in video_selectors
        }
        for selector_name, selected_frames in selected_frames_by_selector.items():
            _log(
                verbose,
                (
                    f"[{video_id}] Selector={selector_name} chose "
                    f"{[frame.frame_index for frame in selected_frames]} "
                    f"at {[round(frame.timestamp_sec, 3) for frame in selected_frames]} sec."
                ),
            )
        _write_selector_debug(output_path / "selected_frames" / video_id, selected_frames_by_selector)
        for item in video_items:
            for selector_name, selected_frames in selected_frames_by_selector.items():
                _log(verbose, f"[{video_id}] QA {item.question_type} with selector={selector_name}.")
                prompt = build_qa_prompt(item, selected_frames)
                raw_prediction = client.answer(prompt, selected_frames)
                correct = None
                parsed = None
                score = None
                if item.question_type in DEFAULT_ROBOFAC_MCQ_TYPES and raw_prediction:
                    if item.question_type == "Failure detection":
                        correct, parsed, score = score_failure_detection_answer(
                            prediction=raw_prediction,
                            reference=item.reference_answer,
                        )
                    else:
                        correct, parsed, score = score_mcq_answer(
                            prediction=raw_prediction,
                            reference=item.reference_answer,
                            options=item.options,
                        )
                predictions.append(
                    RoboFACPrediction(
                        video_id=item.video_id,
                        question_type=item.question_type,
                        selector_name=selector_name,
                        question=item.question,
                        reference_answer=item.reference_answer,
                        raw_prediction=raw_prediction,
                        parsed_prediction=parsed,
                        correct=correct,
                        score=score,
                        selected_frame_indices=[frame.frame_index for frame in selected_frames],
                        selected_timestamps_sec=[frame.timestamp_sec for frame in selected_frames],
                        metadata={
                            "frame_budget": frame_budget,
                            "prompt_contains_metric_values": False,
                            "prompt_contains_bev": False,
                            "prompt_contains_object_state": False,
                        },
                    )
                )

    stats = aggregate_predictions(predictions)
    manifest_path = output_path / "robofac_manifest.json"
    predictions_path = output_path / "robofac_predictions.jsonl"
    stats_path = output_path / "robofac_stats.json"
    alignment_path = output_path / "robofac_alignment_debug.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    predictions_path.write_text(
        "\n".join(json.dumps(prediction.to_dict()) for prediction in predictions),
        encoding="utf-8",
    )
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    alignment_path.write_text(json.dumps(alignment_debug, indent=2), encoding="utf-8")
    summary = {
        "annotation_path": str(Path(annotation_path).expanduser().resolve()),
        "annotation_paths": [str(path) for path in annotation_paths],
        "dataset_root": str(resolved_dataset_root) if resolved_dataset_root is not None else None,
        "data_root": str(resolved_data_root) if resolved_data_root is not None else None,
        "output_dir": str(output_path),
        "question_types": resolved_question_types,
        "selectors": resolved_selectors,
        "frame_budget": frame_budget,
        "shuffle_videos": shuffle_videos,
        "video_sample_seed": video_sample_seed,
        "qa_item_count": len(items),
        "processed_video_count": processed_video_count,
        "prediction_count": len(predictions),
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "stats_path": str(stats_path),
        "alignment_debug_path": str(alignment_path),
    }
    (output_path / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(
        verbose,
        (
            f"Finished RoboFAC QA run: processed_videos={processed_video_count}, "
            f"predictions={len(predictions)}, output_dir={output_path}"
        ),
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoboFAC keyframe-selection QA evaluation.")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to one annos_per_video*.json file, or a directory containing them.",
    )
    parser.add_argument("--h5-path", help="Optional explicit RoboFAC/ManiSkill trajectory .h5 file.")
    parser.add_argument("--traj-json-path", help="Optional explicit matching trajectory metadata JSON.")
    parser.add_argument("--video-root", help="Root containing videos in annotation-relative paths.")
    parser.add_argument(
        "--data-root",
        help=(
            "Root containing RoboFAC demo folders. May be the full RoboFAC-dataset root, "
            "simulation_data, or realworld_data."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="RoboFAC-dataset",
        help=(
            "Full local RoboFAC-dataset root used to resolve simulation_data/ and realworld_data/. "
            "Defaults to ./RoboFAC-dataset when run from the project root."
        ),
    )
    parser.add_argument("--video-path", help="Single local video path for smoke tests.")
    parser.add_argument("--output-dir", default="robofac_outputs", help="Output directory.")
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=list(DEFAULT_ROBOFAC_MCQ_TYPES),
        help="RoboFAC question types to evaluate.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=["uniform", "kite_flow", "metric"],
        choices=["uniform", "flow", "kite_flow", "metric"],
        help="Frame selectors to compare.",
    )
    parser.add_argument("--frame-budget", type=int, default=5, help="Number of keyframes per selector.")
    parser.add_argument("--max-videos", type=int, help="Optional maximum number of videos to process.")
    parser.add_argument(
        "--shuffle-videos",
        action="store_true",
        help="Shuffle video processing order before applying --max-videos.",
    )
    parser.add_argument(
        "--video-sample-seed",
        type=int,
        help=(
            "Seed for reproducible shuffled video sampling. Providing this also enables "
            "shuffling even if --shuffle-videos is omitted."
        ),
    )
    parser.add_argument("--config-path", help="Optional pathologist config override.")
    parser.add_argument("--openai-model", help="OpenAI model for QA calls.")
    parser.add_argument("--skip-vlm", action="store_true", help="Build manifest/selected frames without VLM calls.")
    parser.add_argument("--verbose", action="store_true", help="Print progress/debug information while running.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing videos from --hf-base-url using annotation-relative video paths.",
    )
    parser.add_argument(
        "--hf-base-url",
        default=ROBOFAC_HF_SIMULATION_BASE_URL,
        help=(
            "Base URL for selective fetching of annotation-relative RoboFAC simulation files. "
            "Defaults to the RoboFAC Hugging Face simulation_data resolve URL."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_robofac_qa_pipeline(
        annotation_path=args.annotations,
        h5_path=args.h5_path,
        traj_json_path=args.traj_json_path,
        video_root=args.video_root,
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        video_path=args.video_path,
        output_dir=args.output_dir,
        question_types=args.question_types,
        selectors=args.selectors,
        frame_budget=args.frame_budget,
        max_videos=args.max_videos,
        shuffle_videos=args.shuffle_videos,
        video_sample_seed=args.video_sample_seed,
        config_path=args.config_path,
        openai_model=args.openai_model,
        skip_vlm=args.skip_vlm,
        download_missing=args.download_missing,
        hf_base_url=args.hf_base_url,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


def _resolve_video_path(
    video_id: str,
    video_rel_path: str,
    video_root: Path | None,
    explicit_video_path: Path | None,
    download_missing: bool = False,
    hf_base_url: str | None = None,
    download_dir: Path | None = None,
    verbose: bool = False,
) -> Path | None:
    if explicit_video_path is not None and explicit_video_path.stem == video_id and explicit_video_path.exists():
        return explicit_video_path
    if video_root is not None:
        for root in _candidate_roots(video_root):
            candidates = [
                root / video_rel_path,
                root / Path(video_rel_path).name,
                root / f"{video_id}.mp4",
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
        for root in _candidate_roots(video_root):
            matches = sorted(root.rglob(f"{video_id}.mp4"), key=_natural_path_sort_key)
            if matches:
                return matches[0].resolve()
    if download_missing and hf_base_url and video_rel_path:
        if download_dir is None:
            return None
        download_path = download_dir / video_rel_path
        download_path.parent.mkdir(parents=True, exist_ok=True)
        if not download_path.exists():
            url = hf_base_url.rstrip("/") + "/" + video_rel_path.lstrip("/")
            _download_if_missing(url=url, destination=download_path, verbose=verbose, label="video")
        else:
            _log(verbose, f"Using cached downloaded video: {download_path}")
        return download_path.resolve() if download_path.exists() else None
    return None


def _candidate_roots(root: Path | None) -> list[Path]:
    if root is None:
        return []
    candidates = [root]
    for child_name in ("simulation_data", "realworld_data"):
        child = root / child_name
        if child.exists() and child.is_dir():
            candidates.append(child)
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[robofac] {message}", flush=True)


def _resolve_annotation_paths(annotation_path: str | Path) -> list[Path]:
    path = Path(annotation_path).expanduser()
    if path.is_file():
        return [path.resolve()]
    if not path.exists():
        raise FileNotFoundError(f"RoboFAC annotation path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"RoboFAC annotation path must be a file or directory: {path}")
    matches = sorted(
        [candidate.resolve() for candidate in path.rglob("annos_per_video*.json") if candidate.is_file()],
        key=_natural_path_sort_key,
    )
    if not matches:
        raise FileNotFoundError(f"No annos_per_video*.json files found under: {path}")
    return matches


def _infer_dataset_root(annotation_paths: list[Path]) -> Path | None:
    for annotation_path in annotation_paths:
        for parent in annotation_path.parents:
            if parent.name == "RoboFAC-dataset":
                return parent.resolve()
        if annotation_path.parent.name in {"test_qa_sim", "test_qa_realworld"}:
            return annotation_path.parent.parent.resolve()
    local_dataset_root = Path("RoboFAC-dataset")
    if local_dataset_root.exists() and local_dataset_root.is_dir():
        return local_dataset_root.resolve()
    return None


def _natural_path_sort_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def _selectors_available_for_bundle(
    requested_selectors: list[str],
    raw_metadata: dict[str, Any],
) -> list[str]:
    if not raw_metadata.get("video_only"):
        return requested_selectors
    return [
        selector
        for selector in requested_selectors
        if selector.lower() not in {"metric"}
    ]


def _has_video_only_selector(requested_selectors: list[str]) -> bool:
    return any(selector.lower() != "metric" for selector in requested_selectors)


def _video_items_for_sampling(
    items_by_video: dict[str, list[Any]],
    shuffle_videos: bool,
    video_sample_seed: int | None,
) -> list[tuple[str, list[Any]]]:
    video_items = list(items_by_video.items())
    if shuffle_videos or video_sample_seed is not None:
        rng = random.Random(video_sample_seed)
        rng.shuffle(video_items)
    return video_items


def _resolve_trajectory_files(
    video_rel_path: str,
    explicit_h5_path: Path | None,
    explicit_traj_json_path: Path | None,
    data_root: Path | None,
    download_missing: bool,
    hf_base_url: str | None,
    download_dir: Path,
    verbose: bool = False,
    allow_missing: bool = False,
) -> tuple[Path, Path] | None:
    if explicit_h5_path is not None and explicit_traj_json_path is not None:
        if not explicit_h5_path.exists():
            raise FileNotFoundError(f"Explicit H5 path does not exist: {explicit_h5_path}")
        if not explicit_traj_json_path.exists():
            raise FileNotFoundError(f"Explicit trajectory JSON path does not exist: {explicit_traj_json_path}")
        return explicit_h5_path, explicit_traj_json_path

    folder_rel = str(Path(video_rel_path).parent)
    local_folders = [
        root / folder_rel
        for root in _candidate_roots(data_root)
    ]
    h5_candidates: list[Path] = []
    json_candidates: list[Path] = []
    for local_folder in local_folders:
        h5_candidates.extend(_trajectory_h5_candidates(local_folder))
        json_candidates.extend(_trajectory_json_candidates(local_folder))
    if explicit_h5_path is not None:
        h5_candidates.insert(0, explicit_h5_path)
    if explicit_traj_json_path is not None:
        json_candidates.insert(0, explicit_traj_json_path)

    h5_path, json_path = _first_existing_trajectory_pair(h5_candidates, json_candidates)
    if h5_path is not None and json_path is not None:
        return h5_path.resolve(), json_path.resolve()

    if not download_missing or not hf_base_url:
        if allow_missing:
            return None
        missing = []
        if h5_path is None:
            missing.append("*.h5")
        if json_path is None:
            missing.append("*.json")
        raise FileNotFoundError(
            f"Missing RoboFAC trajectory file(s) in {folder_rel}: {', '.join(missing)}. "
            "Pass --data-root with local RoboFAC folders or use --download-missing."
        )

    download_folder = download_dir / folder_rel
    download_folder.mkdir(parents=True, exist_ok=True)
    if h5_path is None:
        h5_path = _download_if_missing(
            url=hf_base_url.rstrip("/") + "/" + folder_rel + "/ToolsTask_traj.h5",
            destination=download_folder / "ToolsTask_traj.h5",
            verbose=verbose,
        )
    if json_path is None:
        json_path = _download_if_missing(
            url=hf_base_url.rstrip("/") + "/" + folder_rel + "/ToolsTask_traj.json",
            destination=download_folder / "ToolsTask_traj.json",
            verbose=verbose,
        )
    return h5_path.resolve(), json_path.resolve()


def _trajectory_h5_candidates(folder: Path) -> list[Path]:
    return _unique_paths(
        [
            folder / "ToolsTask_traj.h5",
            folder / "ToolsTask_traj.hdf5",
            folder / f"{folder.name}.h5",
            folder / f"{folder.name}.hdf5",
            *sorted(folder.glob("*.h5"), key=_natural_path_sort_key),
            *sorted(folder.glob("*.hdf5"), key=_natural_path_sort_key),
        ]
    )


def _trajectory_json_candidates(folder: Path) -> list[Path]:
    return _unique_paths(
        [
            folder / "ToolsTask_traj.json",
            folder / f"{folder.name}.json",
            *sorted(folder.glob("*.json"), key=_natural_path_sort_key),
        ]
    )


def _first_existing_trajectory_pair(
    h5_candidates: list[Path],
    json_candidates: list[Path],
) -> tuple[Path | None, Path | None]:
    existing_h5 = [candidate for candidate in _unique_paths(h5_candidates) if candidate.exists()]
    existing_json = [candidate for candidate in _unique_paths(json_candidates) if candidate.exists()]
    if not existing_h5 or not existing_json:
        return (
            existing_h5[0] if existing_h5 else None,
            existing_json[0] if existing_json else None,
        )
    for h5_path in existing_h5:
        matching_json = next(
            (json_path for json_path in existing_json if json_path.stem == h5_path.stem),
            None,
        )
        if matching_json is not None:
            return h5_path, matching_json
    return existing_h5[0], existing_json[0]


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique


def _download_if_missing(
    url: str,
    destination: Path,
    verbose: bool = False,
    label: str = "file",
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        _log(verbose, f"Downloading {label} {url} -> {destination}")
        try:
            urllib.request.urlretrieve(url, destination)
        except urllib.error.HTTPError as exc:
            _remove_partial_download(destination)
            raise RemoteFileUnavailableError(
                f"{label} unavailable at {url}: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            _remove_partial_download(destination)
            raise RemoteFileUnavailableError(
                f"{label} unavailable at {url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            _remove_partial_download(destination)
            raise RemoteFileUnavailableError(
                f"{label} unavailable at {url}: {exc}"
            ) from exc
    else:
        _log(verbose, f"Using cached downloaded file: {destination}")
    return destination


def _remove_partial_download(destination: Path) -> None:
    try:
        if destination.exists() and destination.is_file():
            destination.unlink()
    except OSError:
        pass


def _write_selector_debug(
    output_dir: Path,
    selected_frames_by_selector: dict[str, list[Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        selector_name: [
            {
                "camera_name": frame.camera_name,
                "timestamp_sec": frame.timestamp_sec,
                "frame_index": frame.frame_index,
                "image_path": str(frame.image_path),
            }
            for frame in frames
        ]
        for selector_name, frames in selected_frames_by_selector.items()
    }
    (output_dir / "selected_frames.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
