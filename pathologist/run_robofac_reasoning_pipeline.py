"""Evaluate RoboFAC explanation and planning with comparable frame selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pathologist.adapters.robofac_bundle import RoboFACBundleAdapter
from pathologist.config import load_pipeline_config
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac import RoboFACQAItem, load_robofac_qa_items
from pathologist.qa.robofac_client import OpenAIRoboFACQAClient, RoboFACQAClient
from pathologist.qa.robofac_reasoning import (
    JUDGE_CRITERIA,
    SUPPORTED_REASONING_QUESTION_TYPES,
    OpenAIRoboFACJudgeClient,
    RoboFACJudgeClient,
    build_reasoning_prompt,
    build_robofac_judge_prompt,
    parse_robofac_judge_response,
    validate_reasoning_question_types,
)
from pathologist.qa.robofac_selectors import select_keyframes
from pathologist.run_robofac_qa_pipeline import (
    ROBOFAC_HF_SIMULATION_BASE_URL,
    _infer_dataset_root,
    _resolve_annotation_paths,
    _resolve_trajectory_files,
    _resolve_video_path,
    _write_json_atomic,
    _write_selector_debug,
)


DEFAULT_SELECTORS = ("uniform", "kite_flow", "metric")


def run_robofac_reasoning_pipeline(
    annotation_path: str | Path,
    *,
    dataset_root: str | Path | None = "RoboFAC-dataset",
    data_root: str | Path | None = None,
    video_root: str | Path | None = None,
    output_dir: str | Path = "robofac_reasoning_outputs",
    question_types: list[str] | None = None,
    selectors: list[str] | None = None,
    frame_budget: int = 5,
    metric_ranking: str = "area",
    max_videos: int | None = None,
    max_questions_per_type: int | None = None,
    video_sample_seed: int | None = None,
    config_path: str | Path | None = None,
    qa_client: RoboFACQAClient | None = None,
    judge_client: RoboFACJudgeClient | None = None,
    openai_model: str = "gpt-4o",
    judge_model: str = "gpt-4o",
    openai_temperature: float = 0.1,
    judge_temperature: float = 0.2,
    judge_top_p: float = 1.0,
    openai_timeout_sec: float | None = None,
    openai_max_retries: int | None = None,
    openai_retry_backoff_sec: float | None = None,
    fail_on_vlm_error: bool = False,
    resume: bool = False,
    download_missing: bool = False,
    hf_base_url: str = ROBOFAC_HF_SIMULATION_BASE_URL,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run simulation-only open-ended RoboFAC evaluation and LLM judging."""

    resolved_question_types = validate_reasoning_question_types(
        list(SUPPORTED_REASONING_QUESTION_TYPES)
        if question_types is None
        else question_types
    )
    resolved_selectors = _validate_selectors(
        list(DEFAULT_SELECTORS) if selectors is None else selectors
    )
    resolved_metric_ranking = _validate_metric_ranking(metric_ranking)
    _validate_limits(frame_budget, max_videos, max_questions_per_type)
    _validate_request_parameters(
        openai_temperature=openai_temperature,
        judge_temperature=judge_temperature,
        judge_top_p=judge_top_p,
        openai_max_retries=openai_max_retries,
        openai_retry_backoff_sec=openai_retry_backoff_sec,
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "robofac_manifest.json"
    predictions_path = output_path / "robofac_predictions.jsonl"
    judgments_path = output_path / "robofac_judgments.jsonl"
    stats_path = output_path / "robofac_stats.json"
    summary_path = output_path / "run_summary.json"
    alignment_path = output_path / "robofac_alignment_debug.json"

    annotation_paths = _resolve_annotation_paths(annotation_path)
    _validate_simulation_annotations(annotation_paths)
    items = [
        item
        for current_path in annotation_paths
        for item in load_robofac_qa_items(
            current_path,
            question_types=set(resolved_question_types),
        )
    ]
    if any("realworld_data" in Path(item.video_rel_path).parts for item in items):
        raise ValueError("RoboFAC reasoning evaluation supports simulation annotations only.")
    video_work_items, eligible_question_counts = select_reasoning_work_items(
        items,
        question_types=resolved_question_types,
        video_sample_seed=video_sample_seed,
        max_videos=None,
        max_questions_per_type=None,
    )
    _log(
        verbose,
        (
            f"Loaded {len(items)} supported QA items across {len(video_work_items)} "
            f"eligible videos with question counts {eligible_question_counts}."
        ),
    )

    config = load_pipeline_config(config_path=config_path)
    _set_metric_ranking(config, resolved_metric_ranking)
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

    answer_client = qa_client or OpenAIRoboFACQAClient(
        model=openai_model,
        temperature=openai_temperature,
        include_frame_metadata=False,
        timeout_sec=openai_timeout_sec,
        max_retries=openai_max_retries,
        retry_backoff_sec=openai_retry_backoff_sec,
    )
    resolved_judge_client = judge_client or OpenAIRoboFACJudgeClient(
        model=judge_model,
        temperature=judge_temperature,
        top_p=judge_top_p,
        timeout_sec=openai_timeout_sec,
        max_retries=openai_max_retries,
        retry_backoff_sec=openai_retry_backoff_sec,
    )

    settings = {
        "question_types": resolved_question_types,
        "selectors": resolved_selectors,
        "frame_budget": frame_budget,
        "metric_ranking": resolved_metric_ranking,
        "max_videos": max_videos,
        "max_questions_per_type": max_questions_per_type,
        "video_sample_seed": video_sample_seed,
        "openai_model": openai_model,
        "judge_model": judge_model,
        "openai_temperature": openai_temperature,
        "judge_temperature": judge_temperature,
        "judge_top_p": judge_top_p,
    }
    if resume:
        existing_summary = _load_json_object(summary_path)
        _validate_resume_settings(existing_summary, settings)
        manifest = _load_json_list(manifest_path)
        alignment_debug = _load_json_list(alignment_path)
        prediction_rows = _load_jsonl(predictions_path, verbose=verbose)
        judgment_rows = _load_jsonl(judgments_path, verbose=verbose)
    else:
        manifest = []
        alignment_debug = []
        prediction_rows = []
        judgment_rows = []
        _write_json_atomic(manifest_path, [])
        predictions_path.write_text("", encoding="utf-8")
        judgments_path.write_text("", encoding="utf-8")
        _write_json_atomic(stats_path, {})
        _write_json_atomic(alignment_path, [])

    predictions_by_id = _latest_rows_by_id(prediction_rows)
    judgments_by_id = _latest_rows_by_id(judgment_rows)
    manifest_by_video = {
        str(row["video_id"]): row
        for row in manifest
        if isinstance(row, dict) and row.get("video_id")
    }
    selected_question_counts = _manifest_question_counts(
        manifest_by_video.values(),
        resolved_question_types,
    )
    summary_base = {
        "status": "running",
        "dataset": "RoboFAC",
        "split": "simulation",
        "annotation_path": str(Path(annotation_path).expanduser().resolve()),
        "annotation_paths": [str(path) for path in annotation_paths],
        "dataset_root": str(resolved_dataset_root) if resolved_dataset_root else None,
        "data_root": str(resolved_data_root) if resolved_data_root else None,
        "output_dir": str(output_path),
        **settings,
        "eligible_question_count": len(items),
        "candidate_video_count": len(video_work_items),
        "selected_video_count": len(manifest_by_video),
        "selected_question_counts": selected_question_counts,
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "judgments_path": str(judgments_path),
        "stats_path": str(stats_path),
        "alignment_debug_path": str(alignment_path),
        "resume": resume,
    }
    adapter = RoboFACBundleAdapter()
    invalid_judge_retries = 2 if openai_max_retries is None else openai_max_retries

    with predictions_path.open("a", encoding="utf-8") as predictions_file, judgments_path.open(
        "a", encoding="utf-8"
    ) as judgments_file:
        for candidate_index, (video_id, available_video_items) in enumerate(
            video_work_items,
            start=1,
        ):
            existing_manifest_entry = manifest_by_video.get(video_id)
            is_new_video = existing_manifest_entry is None
            if is_new_video:
                if max_videos is not None and len(manifest_by_video) >= max_videos:
                    break
                if _question_limits_reached(
                    selected_question_counts,
                    resolved_question_types,
                    max_questions_per_type,
                ):
                    break
                video_items = _items_with_remaining_question_capacity(
                    available_video_items,
                    selected_question_counts,
                    max_questions_per_type,
                )
            else:
                video_items = _items_recorded_in_manifest(
                    available_video_items,
                    existing_manifest_entry,
                )
            if not video_items:
                continue

            expected_ids = {
                prediction_id(item, selector_name)
                for item in video_items
                for selector_name in resolved_selectors
            }
            if expected_ids and all(
                _has_successful_prediction(predictions_by_id.get(row_id))
                and _has_valid_judgment(judgments_by_id.get(row_id))
                for row_id in expected_ids
            ):
                _log(verbose, f"[{video_id}] Resume: all predictions and judgments already exist.")
                continue

            needs_generation = is_new_video or any(
                not _has_successful_prediction(predictions_by_id.get(row_id))
                for row_id in expected_ids
            )
            selected_frames_by_selector: dict[str, list[Any]] = {}
            if needs_generation:
                target_video_count = max_videos or len(video_work_items)
                _log(
                    verbose,
                    f"[{len(manifest_by_video) + (1 if is_new_video else 0)}/"
                    f"{target_video_count}] Loading {video_id} for {len(video_items)} "
                    f"question(s) (candidate {candidate_index}/{len(video_work_items)}).",
                )
                try:
                    video_path = _resolve_video_path(
                        video_id=video_id,
                        video_rel_path=video_items[0].video_rel_path,
                        video_root=resolved_video_root,
                        explicit_video_path=None,
                        download_missing=download_missing,
                        hf_base_url=hf_base_url,
                        download_dir=output_path / "downloaded_videos",
                        verbose=verbose,
                    )
                    if video_path is None:
                        raise FileNotFoundError(f"Video not found for RoboFAC sample {video_id}.")
                    h5_path, traj_json_path = _resolve_trajectory_files(
                        video_rel_path=video_items[0].video_rel_path,
                        explicit_h5_path=None,
                        explicit_traj_json_path=None,
                        data_root=resolved_data_root,
                        download_missing=download_missing,
                        hf_base_url=hf_base_url,
                        download_dir=output_path / "downloaded_data",
                        verbose=verbose,
                        allow_missing=False,
                    )
                    bundle = adapter.load(
                        video_path=video_path,
                        h5_path=h5_path,
                        traj_json_path=traj_json_path,
                        frame_cache_dir=output_path / "frame_cache" / video_id,
                        video_id=video_id,
                    )
                    metric_profile = (
                        MetricEngine(config=config).compute(bundle)
                        if "metric" in resolved_selectors
                        else None
                    )
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
                    if set(selected_frames_by_selector) != set(resolved_selectors):
                        raise RuntimeError("Not all requested selectors produced a frame set.")
                    invalid_frame_counts = {
                        selector_name: len(frames)
                        for selector_name, frames in selected_frames_by_selector.items()
                        if len(frames) != frame_budget
                    }
                    if invalid_frame_counts:
                        raise RuntimeError(
                            f"Every selector must produce exactly {frame_budget} frames; "
                            f"received {invalid_frame_counts}."
                        )
                except Exception as exc:
                    _record_alignment_error(alignment_debug, video_id, video_items, exc)
                    _log(True, f"[{video_id}] Skipped: {type(exc).__name__}: {exc}")
                    _persist_state(
                        manifest,
                        alignment_debug,
                        predictions_by_id,
                        judgments_by_id,
                        resolved_selectors,
                        resolved_question_types,
                        summary_base,
                        manifest_path,
                        alignment_path,
                        stats_path,
                        summary_path,
                    )
                    continue

                _write_selector_debug(
                    output_path / "selected_frames" / video_id,
                    selected_frames_by_selector,
                )
                manifest_entry = {
                    "video_id": video_id,
                    "task": video_items[0].task,
                    "question_types": [item.question_type for item in video_items],
                    "video_path": str(video_path),
                    "h5_path": str(h5_path),
                    "traj_json_path": str(traj_json_path),
                    "selectors": resolved_selectors,
                    "frame_budget": frame_budget,
                    "metric_ranking": resolved_metric_ranking,
                    "matched_episode_id": bundle.raw_metadata.get("episode_id"),
                    "matched_traj_key": bundle.raw_metadata.get("traj_key"),
                    "frame_count": bundle.raw_metadata.get("frame_count"),
                    "fps": bundle.raw_metadata.get("fps"),
                }
                if video_id in manifest_by_video:
                    manifest_by_video[video_id].update(manifest_entry)
                else:
                    manifest.append(manifest_entry)
                    manifest_by_video[video_id] = manifest_entry
                    for item in video_items:
                        selected_question_counts[item.question_type] += 1
                    summary_base["selected_video_count"] = len(manifest_by_video)
                    summary_base["selected_question_counts"] = dict(
                        selected_question_counts
                    )
            else:
                _log(verbose, f"[{video_id}] Resume: generation complete; judging only.")

            for item in video_items:
                prompt = build_reasoning_prompt(item.question)
                for selector_name in resolved_selectors:
                    row_id = prediction_id(item, selector_name)
                    prediction_row = predictions_by_id.get(row_id)
                    if not _has_successful_prediction(prediction_row):
                        frames = selected_frames_by_selector[selector_name]
                        prediction_row = _generate_prediction_row(
                            row_id=row_id,
                            item=item,
                            selector_name=selector_name,
                            frames=frames,
                            prompt=prompt,
                            client=answer_client,
                            model=openai_model,
                            temperature=openai_temperature,
                            frame_budget=frame_budget,
                            fail_on_error=fail_on_vlm_error,
                        )
                        _append_jsonl(predictions_file, prediction_row)
                        predictions_by_id[row_id] = prediction_row
                        _persist_state(
                            manifest,
                            alignment_debug,
                            predictions_by_id,
                            judgments_by_id,
                            resolved_selectors,
                            resolved_question_types,
                            summary_base,
                            manifest_path,
                            alignment_path,
                            stats_path,
                            summary_path,
                        )
                        generation_error = prediction_row.get("generation_error")
                        if isinstance(generation_error, dict):
                            status_code = generation_error.get("status_code")
                            _log(
                                True,
                                f"[{video_id}][{selector_name}] Generation failed"
                                f"{f' (HTTP {status_code})' if status_code else ''}: "
                                f"{generation_error.get('message', 'unknown error')}",
                            )
                            if status_code in {401, 403}:
                                raise RuntimeError(
                                    "OpenAI authentication failed. Export a valid "
                                    "OPENAI_API_KEY, then rerun this command with --resume."
                                )
                    if not _has_successful_prediction(prediction_row):
                        continue
                    if _has_valid_judgment(judgments_by_id.get(row_id)):
                        continue

                    judgment_row = _judge_prediction_row(
                        prediction_row=prediction_row,
                        client=resolved_judge_client,
                        judge_model=judge_model,
                        temperature=judge_temperature,
                        top_p=judge_top_p,
                        invalid_response_retries=invalid_judge_retries,
                        fail_on_error=fail_on_vlm_error,
                    )
                    _append_jsonl(judgments_file, judgment_row)
                    judgments_by_id[row_id] = judgment_row
                    _persist_state(
                        manifest,
                        alignment_debug,
                        predictions_by_id,
                        judgments_by_id,
                        resolved_selectors,
                        resolved_question_types,
                        summary_base,
                        manifest_path,
                        alignment_path,
                        stats_path,
                        summary_path,
                    )
                    _print_live_progress(
                        video_id=video_id,
                        predictions_by_id=predictions_by_id,
                        judgments_by_id=judgments_by_id,
                        selectors=resolved_selectors,
                        question_types=resolved_question_types,
                    )

    final_summary = _persist_state(
        manifest,
        alignment_debug,
        predictions_by_id,
        judgments_by_id,
        resolved_selectors,
        resolved_question_types,
        {**summary_base, "status": "complete"},
        manifest_path,
        alignment_path,
        stats_path,
        summary_path,
    )
    return final_summary


def select_reasoning_work_items(
    items: list[RoboFACQAItem],
    *,
    question_types: list[str],
    video_sample_seed: int | None,
    max_videos: int | None,
    max_questions_per_type: int | None,
) -> tuple[list[tuple[str, list[RoboFACQAItem]]], dict[str, int]]:
    """Filter, shuffle, and cap QA items without splitting selector comparisons."""

    resolved_types = validate_reasoning_question_types(question_types)
    type_order = {question_type: index for index, question_type in enumerate(resolved_types)}
    grouped: dict[str, list[RoboFACQAItem]] = defaultdict(list)
    for item in items:
        if item.question_type in type_order:
            grouped[item.video_id].append(item)
    video_items = list(grouped.items())
    if video_sample_seed is not None:
        random.Random(video_sample_seed).shuffle(video_items)

    counts = {question_type: 0 for question_type in resolved_types}
    selected: list[tuple[str, list[RoboFACQAItem]]] = []
    for video_id, current_items in video_items:
        if max_videos is not None and len(selected) >= max_videos:
            break
        ordered_items = sorted(current_items, key=lambda item: type_order[item.question_type])
        accepted = []
        for item in ordered_items:
            if max_questions_per_type is not None and counts[item.question_type] >= max_questions_per_type:
                continue
            accepted.append(item)
            counts[item.question_type] += 1
        if accepted:
            selected.append((video_id, accepted))
        if max_questions_per_type is not None and all(
            count >= max_questions_per_type for count in counts.values()
        ):
            break
    return selected, counts


def _manifest_question_counts(
    manifest_rows: Any,
    question_types: list[str],
) -> dict[str, int]:
    counts = {question_type: 0 for question_type in question_types}
    for row in manifest_rows:
        recorded_types = row.get("question_types") if isinstance(row, dict) else None
        if not isinstance(recorded_types, list):
            continue
        for question_type in recorded_types:
            if question_type in counts:
                counts[question_type] += 1
    return counts


def _items_recorded_in_manifest(
    available_items: list[RoboFACQAItem],
    manifest_entry: dict[str, Any],
) -> list[RoboFACQAItem]:
    recorded_types = manifest_entry.get("question_types")
    if not isinstance(recorded_types, list):
        return []
    remaining = Counter(str(question_type) for question_type in recorded_types)
    selected = []
    for item in available_items:
        if remaining[item.question_type] <= 0:
            continue
        selected.append(item)
        remaining[item.question_type] -= 1
    return selected


def _items_with_remaining_question_capacity(
    available_items: list[RoboFACQAItem],
    selected_question_counts: dict[str, int],
    max_questions_per_type: int | None,
) -> list[RoboFACQAItem]:
    accepted_counts: Counter[str] = Counter()
    selected = []
    for item in available_items:
        if max_questions_per_type is not None:
            current_count = selected_question_counts.get(item.question_type, 0)
            if current_count + accepted_counts[item.question_type] >= max_questions_per_type:
                continue
        selected.append(item)
        accepted_counts[item.question_type] += 1
    return selected


def _question_limits_reached(
    selected_question_counts: dict[str, int],
    question_types: list[str],
    max_questions_per_type: int | None,
) -> bool:
    return bool(
        max_questions_per_type is not None
        and all(
            selected_question_counts.get(question_type, 0) >= max_questions_per_type
            for question_type in question_types
        )
    )


def prediction_id(item: RoboFACQAItem, selector_name: str) -> str:
    payload = [
        item.video_id,
        item.question_type,
        item.question,
        item.reference_answer,
        selector_name,
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_reasoning_stats(
    predictions_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    selectors: list[str],
    question_types: list[str],
) -> dict[str, Any]:
    """Aggregate latest valid normalized judge scores by selector and type."""

    output: dict[str, Any] = {}
    for selector_name in selectors:
        output[selector_name] = {}
        overall_rows = []
        overall_predictions = []
        for question_type in question_types:
            prediction_rows = [
                row
                for row in predictions_by_id.values()
                if row.get("selector_name") == selector_name
                and row.get("question_type") == question_type
            ]
            judgment_rows = [
                judgments_by_id[row["prediction_id"]]
                for row in prediction_rows
                if str(row.get("prediction_id")) in judgments_by_id
            ]
            output[selector_name][question_type] = _stats_bucket(prediction_rows, judgment_rows)
            overall_predictions.extend(prediction_rows)
            overall_rows.extend(judgment_rows)
        output[selector_name]["overall"] = _stats_bucket(overall_predictions, overall_rows)
    return output


def _generate_prediction_row(
    *,
    row_id: str,
    item: RoboFACQAItem,
    selector_name: str,
    frames: list[Any],
    prompt: str,
    client: RoboFACQAClient,
    model: str,
    temperature: float,
    frame_budget: int,
    fail_on_error: bool,
) -> dict[str, Any]:
    raw_prediction = ""
    error = None
    try:
        raw_prediction = client.answer(prompt, frames)
    except Exception as exc:
        if fail_on_error:
            raise
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "status_code": getattr(exc, "status_code", None),
        }
    return {
        "prediction_id": row_id,
        "video_id": item.video_id,
        "task": item.task,
        "question_type": item.question_type,
        "selector_name": selector_name,
        "question": item.question.replace("<image>", "").strip(),
        "reference_answer": item.reference_answer,
        "prompt": prompt,
        "raw_prediction": raw_prediction,
        "selected_frame_indices": [frame.frame_index for frame in frames],
        "selected_timestamps_sec": [frame.timestamp_sec for frame in frames],
        "frame_budget": frame_budget,
        "model": model,
        "temperature": temperature,
        "generation_error": error,
    }


def _judge_prediction_row(
    *,
    prediction_row: dict[str, Any],
    client: RoboFACJudgeClient,
    judge_model: str,
    temperature: float,
    top_p: float,
    invalid_response_retries: int,
    fail_on_error: bool,
) -> dict[str, Any]:
    prompt = build_robofac_judge_prompt(
        question=str(prediction_row["question"]),
        reference=str(prediction_row["reference_answer"]),
        prediction=str(prediction_row["raw_prediction"]),
    )
    raw_response = ""
    result = None
    error = None
    for attempt in range(max(0, invalid_response_retries) + 1):
        try:
            raw_response = client.judge(prompt)
            result = parse_robofac_judge_response(raw_response)
            break
        except ValueError as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if attempt >= max(0, invalid_response_retries):
                if fail_on_error:
                    raise
                break
        except Exception as exc:
            if fail_on_error:
                raise
            error = {"type": type(exc).__name__, "message": str(exc)}
            break
    row = {
        "prediction_id": prediction_row["prediction_id"],
        "video_id": prediction_row["video_id"],
        "question_type": prediction_row["question_type"],
        "selector_name": prediction_row["selector_name"],
        "judge_model": judge_model,
        "temperature": temperature,
        "top_p": top_p,
        "prompt": prompt,
        "raw_judgment": raw_response,
        "judge_error": error if result is None else None,
    }
    if result is not None:
        row.update(result.to_dict())
    return row


def _stats_bucket(
    prediction_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_judgment_rows = [row for row in judgment_rows if _has_valid_judgment(row)]
    normalized_scores = [float(row["normalized_score"]) for row in valid_judgment_rows]
    criteria = {}
    for criterion in JUDGE_CRITERIA:
        scores = [
            float(row["criteria"][criterion]["score"]) * 20.0
            for row in valid_judgment_rows
        ]
        criteria[criterion] = {
            "mean_score": sum(scores) / len(scores) if scores else None,
        }
    successful_prediction_ids = {
        str(row["prediction_id"])
        for row in prediction_rows
        if _has_successful_prediction(row)
    }
    judge_attempt_ids = {str(row["prediction_id"]) for row in judgment_rows}
    valid_judgment_ids = {
        str(row["prediction_id"]) for row in valid_judgment_rows
    }
    return {
        "attempted_count": len(prediction_rows),
        "generated_count": len(successful_prediction_ids),
        "judge_attempt_count": len(judgment_rows),
        "judged_count": len(valid_judgment_rows),
        "generation_error_count": sum(
            1 for row in prediction_rows if not _has_successful_prediction(row)
        ),
        "judge_error_count": sum(
            1 for row in judgment_rows if not _has_valid_judgment(row)
        ),
        "missing_judgment_count": len(successful_prediction_ids - judge_attempt_ids),
        "pending_judgment_count": len(
            successful_prediction_ids - valid_judgment_ids
        ),
        "mean_score": (
            sum(normalized_scores) / len(normalized_scores)
            if normalized_scores
            else None
        ),
        "criteria": criteria,
    }


def _persist_state(
    manifest: list[dict[str, Any]],
    alignment_debug: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    selectors: list[str],
    question_types: list[str],
    summary_base: dict[str, Any],
    manifest_path: Path,
    alignment_path: Path,
    stats_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    stats = aggregate_reasoning_stats(
        predictions_by_id,
        judgments_by_id,
        selectors,
        question_types,
    )
    valid_judgment_count = sum(
        1 for row in judgments_by_id.values() if _has_valid_judgment(row)
    )
    summary = {
        **summary_base,
        "processed_video_count": len(manifest),
        "prediction_count": len(predictions_by_id),
        "successful_prediction_count": sum(
            1 for row in predictions_by_id.values() if _has_successful_prediction(row)
        ),
        "valid_judgment_count": valid_judgment_count,
        "generation_error_count": sum(
            1 for row in predictions_by_id.values() if not _has_successful_prediction(row)
        ),
        "judge_error_count": sum(
            1
            for row in judgments_by_id.values()
            if not _has_valid_judgment(row) and row.get("judge_error")
        ),
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(alignment_path, alignment_debug)
    _write_json_atomic(stats_path, stats)
    _write_json_atomic(summary_path, summary)
    return summary


def _print_live_progress(
    *,
    video_id: str,
    predictions_by_id: dict[str, dict[str, Any]],
    judgments_by_id: dict[str, dict[str, Any]],
    selectors: list[str],
    question_types: list[str],
) -> None:
    stats = aggregate_reasoning_stats(
        predictions_by_id,
        judgments_by_id,
        selectors,
        question_types,
    )
    parts = []
    for selector_name in selectors:
        bucket = stats[selector_name]["overall"]
        score = bucket["mean_score"]
        parts.append(
            f"{selector_name}={'n/a' if score is None else f'{score:.2f}'}"
            f"(n={bucket['judged_count']})"
        )
    print(f"[robofac-reasoning] last_video={video_id} | " + " | ".join(parts), flush=True)


def _validate_selectors(selectors: list[str]) -> list[str]:
    aliases = {"flow": "kite_flow"}
    resolved = [aliases.get(selector.lower(), selector.lower()) for selector in selectors]
    unsupported = sorted(set(resolved) - set(DEFAULT_SELECTORS))
    if unsupported:
        raise ValueError(
            f"Unsupported RoboFAC reasoning selector(s): {unsupported}. "
            f"Allowed values: {list(DEFAULT_SELECTORS)}."
        )
    if not resolved:
        raise ValueError("At least one selector is required.")
    return list(dict.fromkeys(resolved))


def _validate_metric_ranking(metric_ranking: str) -> str:
    aliases = {
        "area": "area",
        "area_above_base": "area",
        "persistence": "persistence",
    }
    resolved = aliases.get(metric_ranking.strip().lower())
    if resolved is None:
        raise ValueError("metric_ranking must be 'area' or 'persistence'.")
    return resolved


def _set_metric_ranking(config: dict[str, Any], metric_ranking: str) -> None:
    selector_config = config.setdefault("selector", {})
    if not isinstance(selector_config, dict):
        raise ValueError("Pipeline config selector must be an object.")
    surprise_flood_config = selector_config.setdefault("surprise_flood", {})
    if not isinstance(surprise_flood_config, dict):
        raise ValueError("Pipeline config selector.surprise_flood must be an object.")
    surprise_flood_config["ranking"] = metric_ranking


def _validate_limits(
    frame_budget: int,
    max_videos: int | None,
    max_questions_per_type: int | None,
) -> None:
    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive.")
    if max_videos is not None and max_videos < 0:
        raise ValueError("max_videos must be non-negative.")
    if max_questions_per_type is not None and max_questions_per_type < 0:
        raise ValueError("max_questions_per_type must be non-negative.")


def _validate_simulation_annotations(annotation_paths: list[Path]) -> None:
    invalid_paths = [path for path in annotation_paths if "test_qa_sim" not in path.parts]
    if invalid_paths:
        raise ValueError(
            "RoboFAC reasoning evaluation is simulation-only and requires annotations "
            f"from test_qa_sim. Invalid path(s): {[str(path) for path in invalid_paths]}."
        )


def _validate_request_parameters(
    *,
    openai_temperature: float,
    judge_temperature: float,
    judge_top_p: float,
    openai_max_retries: int | None,
    openai_retry_backoff_sec: float | None,
) -> None:
    for name, value in (
        ("openai_temperature", openai_temperature),
        ("judge_temperature", judge_temperature),
    ):
        if not 0 <= value <= 2:
            raise ValueError(f"{name} must be between 0 and 2.")
    if not 0 <= judge_top_p <= 1:
        raise ValueError("judge_top_p must be between 0 and 1.")
    if openai_max_retries is not None and openai_max_retries < 0:
        raise ValueError("openai_max_retries must be non-negative.")
    if openai_retry_backoff_sec is not None and openai_retry_backoff_sec < 0:
        raise ValueError("openai_retry_backoff_sec must be non-negative.")


def _validate_resume_settings(existing_summary: dict[str, Any], settings: dict[str, Any]) -> None:
    if not existing_summary:
        raise ValueError("Cannot resume: run_summary.json is missing or invalid.")
    for key, value in settings.items():
        if key in existing_summary and existing_summary[key] != value:
            raise ValueError(
                f"Cannot resume with different {key}. Saved value is "
                f"{existing_summary[key]!r}, requested value is {value!r}."
            )


def _record_alignment_error(
    alignment_debug: list[dict[str, Any]],
    video_id: str,
    items: list[RoboFACQAItem],
    exc: Exception,
) -> None:
    alignment_debug.append(
        {
            "video_id": video_id,
            "status": "skipped",
            "reason": type(exc).__name__,
            "message": str(exc),
            "video_rel_path": items[0].video_rel_path,
            "question_types": [item.question_type for item in items],
        }
    )


def _has_successful_prediction(row: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(row, dict)
        and isinstance(row.get("raw_prediction"), str)
        and row["raw_prediction"].strip()
        and not row.get("generation_error")
    )


def _has_valid_judgment(row: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(row, dict)
        and isinstance(row.get("normalized_score"), (int, float))
        and not isinstance(row.get("normalized_score"), bool)
        and isinstance(row.get("criteria"), dict)
        and not row.get("judge_error")
    )


def _latest_rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["prediction_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("prediction_id")
    }


def _append_jsonl(file: Any, row: dict[str, Any]) -> None:
    file.write(json.dumps(row, ensure_ascii=True) + "\n")
    file.flush()


def _load_jsonl(path: Path, *, verbose: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(verbose, f"Ignoring malformed JSONL row {path}:{line_number}: {exc}")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[robofac-reasoning] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RoboFAC failure explanation and task planning with LLM judging."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Simulation annos_per_video*.json file or test_qa_sim directory.",
    )
    parser.add_argument("--dataset-root", default="RoboFAC-dataset")
    parser.add_argument("--data-root")
    parser.add_argument("--video-root")
    parser.add_argument("--output-dir", default="robofac_reasoning_outputs")
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=list(SUPPORTED_REASONING_QUESTION_TYPES),
        default=list(SUPPORTED_REASONING_QUESTION_TYPES),
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=["uniform", "flow", "kite_flow", "metric"],
        default=list(DEFAULT_SELECTORS),
    )
    parser.add_argument("--frame-budget", type=int, default=5)
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default="area",
        help="Rank metric peaks by area first or persistence first.",
    )
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-questions-per-type", type=int)
    parser.add_argument("--video-sample-seed", type=int)
    parser.add_argument("--config-path")
    parser.add_argument("--openai-model", default="gpt-4o")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--openai-temperature", type=float, default=0.1)
    parser.add_argument("--judge-temperature", type=float, default=0.2)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--openai-timeout-sec", type=float)
    parser.add_argument("--openai-max-retries", type=int)
    parser.add_argument("--openai-retry-backoff-sec", type=float)
    parser.add_argument("--fail-on-vlm-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--hf-base-url", default=ROBOFAC_HF_SIMULATION_BASE_URL)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_robofac_reasoning_pipeline(
        annotation_path=args.annotations,
        dataset_root=args.dataset_root,
        data_root=args.data_root,
        video_root=args.video_root,
        output_dir=args.output_dir,
        question_types=args.question_types,
        selectors=args.selectors,
        frame_budget=args.frame_budget,
        metric_ranking=args.metric_ranking,
        max_videos=args.max_videos,
        max_questions_per_type=args.max_questions_per_type,
        video_sample_seed=args.video_sample_seed,
        config_path=args.config_path,
        openai_model=args.openai_model,
        judge_model=args.judge_model,
        openai_temperature=args.openai_temperature,
        judge_temperature=args.judge_temperature,
        judge_top_p=args.judge_top_p,
        openai_timeout_sec=args.openai_timeout_sec,
        openai_max_retries=args.openai_max_retries,
        openai_retry_backoff_sec=args.openai_retry_backoff_sec,
        fail_on_vlm_error=args.fail_on_vlm_error,
        resume=args.resume,
        download_missing=args.download_missing,
        hf_base_url=args.hf_base_url,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
