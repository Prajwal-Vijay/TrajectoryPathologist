"""Run RoboFAC QA evaluation with comparable keyframe selectors."""

from __future__ import annotations

import argparse
import json
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
    video_path: str | Path | None = None,
    output_dir: str | Path = "robofac_outputs",
    question_types: list[str] | None = None,
    selectors: list[str] | None = None,
    frame_budget: int = 5,
    max_videos: int | None = None,
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
    items = load_robofac_qa_items(annotation_path, question_types=set(resolved_question_types))
    _log(
        verbose,
        (
            f"Loaded {len(items)} QA item(s) from {annotation_path} "
            f"for question types {resolved_question_types}."
        ),
    )
    items_by_video: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        items_by_video[item.video_id].append(item)

    adapter = RoboFACBundleAdapter()
    client = qa_client
    if client is None:
        client = StaticRoboFACQAClient("") if skip_vlm else OpenAIRoboFACQAClient(model=openai_model)

    predictions: list[RoboFACPrediction] = []
    manifest: list[dict[str, Any]] = []
    alignment_debug: list[dict[str, Any]] = []
    processed_video_count = 0
    explicit_video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
    resolved_data_root = Path(data_root).expanduser().resolve() if data_root is not None else None
    resolved_video_root = (
        Path(video_root).expanduser().resolve()
        if video_root is not None
        else resolved_data_root
    )

    for video_id, video_items in items_by_video.items():
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
            candidate_h5_path, candidate_traj_json_path = _resolve_trajectory_files(
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
        _log(verbose, f"[{video_id}] H5: {candidate_h5_path}")
        _log(verbose, f"[{video_id}] Trajectory JSON: {candidate_traj_json_path}")
        try:
            bundle = adapter.load(
                video_path=candidate_video_path,
                h5_path=candidate_h5_path,
                traj_json_path=candidate_traj_json_path,
                frame_cache_dir=output_path / "frame_cache" / video_id,
                video_id=video_id,
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
                "h5_path": str(candidate_h5_path),
                "traj_json_path": str(candidate_traj_json_path),
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
        if "metric" in resolved_selectors:
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
            for selector_name in resolved_selectors
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
        "output_dir": str(output_path),
        "question_types": resolved_question_types,
        "selectors": resolved_selectors,
        "frame_budget": frame_budget,
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
    parser.add_argument("--annotations", required=True, help="Path to annos_per_video_split*.json.")
    parser.add_argument("--h5-path", help="Optional explicit RoboFAC/ManiSkill trajectory .h5 file.")
    parser.add_argument("--traj-json-path", help="Optional explicit matching trajectory metadata JSON.")
    parser.add_argument("--video-root", help="Root containing videos in annotation-relative paths.")
    parser.add_argument(
        "--data-root",
        help=(
            "Root containing RoboFAC simulation_data-style folders. Used to resolve videos "
            "and per-folder ToolsTask_traj.h5/json when explicit paths are omitted."
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
        video_path=args.video_path,
        output_dir=args.output_dir,
        question_types=args.question_types,
        selectors=args.selectors,
        frame_budget=args.frame_budget,
        max_videos=args.max_videos,
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
        candidates = [
            video_root / video_rel_path,
            video_root / Path(video_rel_path).name,
            video_root / f"{video_id}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        matches = list(video_root.rglob(f"{video_id}.mp4"))
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


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[robofac] {message}", flush=True)


def _resolve_trajectory_files(
    video_rel_path: str,
    explicit_h5_path: Path | None,
    explicit_traj_json_path: Path | None,
    data_root: Path | None,
    download_missing: bool,
    hf_base_url: str | None,
    download_dir: Path,
    verbose: bool = False,
) -> tuple[Path, Path]:
    if explicit_h5_path is not None and explicit_traj_json_path is not None:
        if not explicit_h5_path.exists():
            raise FileNotFoundError(f"Explicit H5 path does not exist: {explicit_h5_path}")
        if not explicit_traj_json_path.exists():
            raise FileNotFoundError(f"Explicit trajectory JSON path does not exist: {explicit_traj_json_path}")
        return explicit_h5_path, explicit_traj_json_path

    folder_rel = str(Path(video_rel_path).parent)
    local_folder = data_root / folder_rel if data_root is not None else None
    h5_candidates = []
    json_candidates = []
    if local_folder is not None:
        h5_candidates.extend(
            [
                local_folder / "ToolsTask_traj.h5",
                local_folder / "ToolsTask_traj.hdf5",
            ]
        )
        json_candidates.append(local_folder / "ToolsTask_traj.json")
    if explicit_h5_path is not None:
        h5_candidates.insert(0, explicit_h5_path)
    if explicit_traj_json_path is not None:
        json_candidates.insert(0, explicit_traj_json_path)

    h5_path = next((candidate for candidate in h5_candidates if candidate.exists()), None)
    json_path = next((candidate for candidate in json_candidates if candidate.exists()), None)
    if h5_path is not None and json_path is not None:
        return h5_path.resolve(), json_path.resolve()

    if not download_missing or not hf_base_url:
        missing = []
        if h5_path is None:
            missing.append("ToolsTask_traj.h5")
        if json_path is None:
            missing.append("ToolsTask_traj.json")
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
