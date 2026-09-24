"""Run RoboFine-Bench VQA with uniform / kite_flow / metric keyframe selectors."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Any

from pathologist.adapters.robofine_lerobot import (
    SURPRISE_SOURCE_METRICS,
    adapter_for_dataset,
    load_jsonl_by_sample_id,
    resolve_sample_paths,
)
from pathologist.config import load_pipeline_config
from pathologist.events import ecod_aggregate_output_paths
from pathologist.metrics import MetricEngine
from pathologist.qa.robofac_client import (
    OpenAIRoboFACQAClient,
    OpenAIRoboFACQAError,
    RoboFACQAClient,
    StaticRoboFACQAClient,
)
from pathologist.qa.robofac_selectors import select_keyframes
from pathologist.qa.robofine import (
    RoboFinePrediction,
    RoboFineQAItem,
    aggregate_robofine_stats,
    build_vqa_prompt,
    load_robofine_eval_samples,
    load_robofine_qa_items,
    parse_vqa_answers,
    score_robofine_answer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATA = REPO_ROOT / "RoboFine-Bench" / "EvalData"
DEFAULT_DATASETS = ("RT-1", "BridgeDataV2", "BC-Z")
DEFAULT_SELECTORS = ("uniform", "kite_flow", "metric")
SELECTOR_ALIASES = {"flow": "kite_flow", "kite": "kite_flow", "ecod": "metric", "prism": "metric"}
DEFAULT_METRIC_RANKING = "persistence"
DEFAULT_OPENAI_MODEL = "gpt-5.4"


def run_robofine_qa_pipeline(
    eval_data: str | Path = DEFAULT_EVAL_DATA,
    output_dir: str | Path = "robofine_outputs",
    datasets: list[str] | None = None,
    selectors: list[str] | None = None,
    frame_budget: int = 5,
    max_samples: int | None = None,
    config_path: str | Path | None = None,
    metric_ranking: str | None = None,
    smoothing_window: int | None = None,
    plot_ecod_aggregates: bool = True,
    qa_client: RoboFACQAClient | None = None,
    openai_model: str | None = None,
    openai_temperature: float | None = None,
    openai_timeout_sec: float | None = None,
    openai_max_retries: int | None = None,
    openai_retry_backoff_sec: float | None = None,
    fail_on_vlm_error: bool = False,
    resume: bool = False,
    skip_vlm: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate verified RoboFine telemetry samples with K-frame selectors."""

    if frame_budget <= 0:
        raise ValueError("frame_budget must be positive.")
    eval_data_path = Path(eval_data).expanduser().resolve()
    bench_dir = eval_data_path.parent
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "robofine_manifest.json"
    predictions_path = output_path / "robofine_predictions.jsonl"
    stats_path = output_path / "robofine_stats.json"
    alignment_path = output_path / "robofine_alignment_debug.json"
    summary_path = output_path / "run_summary.json"
    metric_availability_path = output_path / "metric_availability.jsonl"
    ecod_aggregate_dir = output_path / "ecod_aggregates"

    resolved_datasets = list(datasets or DEFAULT_DATASETS)
    resolved_selectors = resolve_selector_names(selectors)
    resolved_metric_ranking = _resolve_metric_ranking(
        metric_ranking if metric_ranking is not None else DEFAULT_METRIC_RANKING
    )
    config = load_pipeline_config(config_path=config_path)
    _set_metric_ranking(config, resolved_metric_ranking)
    _set_max_proposals(config, frame_budget)
    if smoothing_window is not None:
        _set_smoothing_window(config, smoothing_window)
    resolved_smoothing_window = int(_surprise_flood_config(config).get("smoothing_window", 1))

    existing_summary = _load_json_object(summary_path) if resume else {}
    if resume:
        _validate_resume_settings(
            existing_summary=existing_summary,
            datasets=resolved_datasets,
            selectors=resolved_selectors,
            frame_budget=frame_budget,
            metric_ranking=resolved_metric_ranking,
            smoothing_window=resolved_smoothing_window,
            skip_vlm=skip_vlm,
        )

    telemetry_manifest = load_jsonl_by_sample_id(eval_data_path / "telemetry" / "telemetry_manifest.jsonl")
    verification = load_jsonl_by_sample_id(eval_data_path / "telemetry" / "alignment_verification.jsonl")
    eval_samples = load_robofine_eval_samples(eval_data_path / "EvalSets.json")
    allowed_ids = {
        sample_id
        for sample_id, entry in telemetry_manifest.items()
        if entry.get("status") == "ok"
        and entry.get("length_matches_video") is True
        and str(entry.get("dataset")) in resolved_datasets
        and sample_id in eval_samples
    }
    qa_items = load_robofine_qa_items(
        eval_data_path / "QAEvalSets.json",
        eval_samples,
        allowed_sample_ids=allowed_ids,
    )
    items_by_sample: dict[str, list[RoboFineQAItem]] = defaultdict(list)
    for item in qa_items:
        items_by_sample[item.sample_id].append(item)
    sample_ids = sorted(allowed_ids)
    if not skip_vlm:
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in items_by_sample]
    if max_samples is not None:
        sample_ids = sample_ids[: max(0, max_samples)]
    _log(
        verbose,
        (
            f"Loaded {len(qa_items)} questions across {len(items_by_sample)} samples "
            f"({len(allowed_ids)} telemetry-verified); evaluating {len(sample_ids)}."
        ),
    )

    client = qa_client
    if client is None:
        client = (
            StaticRoboFACQAClient("")
            if skip_vlm
            else OpenAIRoboFACQAClient(
                model=openai_model or DEFAULT_OPENAI_MODEL,
                temperature=openai_temperature,
                timeout_sec=openai_timeout_sec,
                max_retries=openai_max_retries,
                retry_backoff_sec=openai_retry_backoff_sec,
                max_output_tokens=2048,
                include_frame_timing_metadata=False,
            )
        )

    if resume:
        existing_predictions, manifest, alignment_debug = _load_resume_state(
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            alignment_path=alignment_path,
            verbose=verbose,
        )
        existing_prediction_keys = {
            key for row in existing_predictions if (key := _prediction_key_from_row(row)) is not None
        }
        metric_availability_rows = _unique_metric_availability_rows(_load_jsonl(metric_availability_path))
    else:
        existing_predictions = []
        existing_prediction_keys = set()
        manifest = []
        alignment_debug = []
        metric_availability_rows = []
        _write_json_atomic(manifest_path, [])
        predictions_path.write_text("", encoding="utf-8")
        _write_json_atomic(stats_path, {})
        _write_json_atomic(alignment_path, [])
        metric_availability_path.write_text("", encoding="utf-8")

    manifest_by_sample = {
        str(entry.get("sample_id")): entry
        for entry in manifest
        if isinstance(entry, dict) and entry.get("sample_id")
    }
    processed_sample_count = len(manifest_by_sample)
    prediction_count = len(existing_predictions)
    vlm_error_count = sum(
        1
        for row in existing_predictions
        if isinstance(row.get("metadata"), dict) and row["metadata"].get("vlm_error")
    )
    padded_sample_ids = [
        str(entry["sample_id"])
        for entry in manifest
        if isinstance(entry, dict)
        and entry.get("sample_id")
        and isinstance(entry.get("metric_selection"), dict)
        and entry["metric_selection"].get("padded")
    ]
    summary_base = _build_run_summary(
        eval_data_path=eval_data_path,
        output_path=output_path,
        datasets=resolved_datasets,
        selectors=resolved_selectors,
        frame_budget=frame_budget,
        metric_ranking=resolved_metric_ranking,
        smoothing_window=resolved_smoothing_window,
        skip_vlm=skip_vlm,
        resume=resume,
        sample_count=len(sample_ids),
        question_count=sum(len(items_by_sample[sample_id]) for sample_id in sample_ids),
        telemetry_sample_count=len(allowed_ids),
        openai_model=None if skip_vlm else getattr(client, "model", openai_model),
    )
    _write_json_atomic(summary_path, summary_base)

    for sample_id in sample_ids:
        sample_items = items_by_sample[sample_id]
        sample = eval_samples[sample_id]
        dataset = str(sample["dataset"])
        question_ids = [item.question_id for item in sample_items]
        already_complete = sample_id in manifest_by_sample and (
            skip_vlm
            or all(
                _selector_complete(sample_id, selector, question_ids, existing_prediction_keys)
                for selector in resolved_selectors
            )
        )
        if resume and already_complete:
            _log(verbose, f"[{sample_id}] Resume: already complete; skipping.")
            continue

        try:
            bundle, video_path, parquet_path, camera_name, manifest_entry = _load_sample_bundle(
                sample=sample,
                telemetry_manifest=telemetry_manifest,
                verification=verification,
                eval_data_path=eval_data_path,
                bench_dir=bench_dir,
                frame_cache_dir=output_path / "frame_cache" / sample_id,
            )
        except Exception as exc:
            alignment_debug.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "status": "skipped",
                    "reason": "adapter_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            _log(verbose, f"[{sample_id}] Skipped: {type(exc).__name__}: {exc}")
            _persist_run_state(
                manifest=manifest,
                alignment_debug=alignment_debug,
                predictions=existing_predictions,
                output_path=output_path,
                summary_base=summary_base,
                processed_sample_count=processed_sample_count,
                prediction_count=prediction_count,
                vlm_error_count=vlm_error_count,
                metric_availability_rows=metric_availability_rows,
                padded_sample_ids=padded_sample_ids,
            )
            continue

        sample_config = copy.deepcopy(config)
        _set_visual_flow_camera_names(sample_config, [camera_name])
        metric_profile = None
        metric_availability: dict[str, Any] = {}
        if "metric" in resolved_selectors:
            _log(verbose, f"[{sample_id}] Computing metric profile (camera={camera_name}).")
            metric_profile = MetricEngine(config=sample_config).compute(bundle)
            metric_availability = _metric_availability(metric_profile)
        ecod_aggregate_path = ecod_aggregate_dir / sample_id if plot_ecod_aggregates else None
        metric_selection_stats: dict[str, Any] = {}
        selected_frames_by_selector = {
            selector_name: select_keyframes(
                selector_name=selector_name,
                bundle=bundle,
                frame_budget=frame_budget,
                metric_profile=metric_profile,
                config=sample_config,
                ecod_aggregate_path=ecod_aggregate_path,
                selection_stats=metric_selection_stats if selector_name == "metric" else None,
            )
            for selector_name in resolved_selectors
        }
        kite_frame_count = len(selected_frames_by_selector.get("kite_flow", []))
        if "kite_flow" in selected_frames_by_selector and kite_frame_count < frame_budget:
            _log(
                verbose,
                (
                    f"[{sample_id}] Official KITE motion NMS selected "
                    f"{kite_frame_count}/{frame_budget} frame(s); K is an upper bound and "
                    "the official motion strategy does not add uniform padding."
                ),
            )
        if metric_selection_stats.get("padded"):
            padded_sample_ids.append(sample_id)
            _warn(
                f"[{sample_id}] metric selector found only "
                f"{metric_selection_stats['proposal_count']} surprise peak(s) and used "
                f"{metric_selection_stats['uniform_padded_frame_count']}/{frame_budget} uniform "
                f"frame(s) as padding. Lower selector.surprise_flood.smoothing_window "
                f"(currently {resolved_smoothing_window}) to keep this comparison clean."
            )
        _write_selector_debug(output_path / "selected_frames" / sample_id, selected_frames_by_selector)
        if ecod_aggregate_path is not None and "metric" in resolved_selectors:
            plot_error = _write_selector_overlay_plot(
                ecod_aggregate_path=ecod_aggregate_path,
                sample_id=sample_id,
                selected_frames_by_selector=selected_frames_by_selector,
                metric_ranking=resolved_metric_ranking,
            )
            if plot_error is not None:
                _log(verbose, f"[{sample_id}] Selector overlay plot skipped: {plot_error}")

        if sample_id not in manifest_by_sample:
            processed_sample_count += 1
            entry = {
                "sample_id": sample_id,
                "dataset": dataset,
                "video_path": str(video_path),
                "telemetry_path": str(parquet_path),
                "camera_name": camera_name,
                "fps": bundle.raw_metadata.get("fps"),
                "frame_count": bundle.raw_metadata.get("frame_count"),
                "telemetry_rows": bundle.raw_metadata.get("telemetry_rows"),
                "episode_index": bundle.raw_metadata.get("episode_index"),
                "pixel_corr": bundle.raw_metadata.get("pixel_corr"),
                "selectors": list(resolved_selectors),
                "selected_frame_counts": {
                    name: len(frames) for name, frames in selected_frames_by_selector.items()
                },
                "metric_selection": dict(metric_selection_stats),
                "question_ids": question_ids,
                "alignment_validation_notes": bundle.raw_metadata.get("alignment_validation_notes", []),
                "metric_availability": {
                    name: info.get("available")
                    for name, info in metric_availability.items()
                    if name in {*SURPRISE_SOURCE_METRICS, "ecod_aggregate_score"}
                },
            }
            manifest.append(entry)
            manifest_by_sample[sample_id] = entry
        if metric_availability:
            metric_availability_rows = [
                row for row in metric_availability_rows if row.get("sample_id") != sample_id
            ]
            metric_availability_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "metrics": metric_availability,
                }
            )

        if skip_vlm:
            _persist_run_state(
                manifest=manifest,
                alignment_debug=alignment_debug,
                predictions=existing_predictions,
                output_path=output_path,
                summary_base=summary_base,
                processed_sample_count=processed_sample_count,
                prediction_count=prediction_count,
                vlm_error_count=vlm_error_count,
                metric_availability_rows=metric_availability_rows,
                padded_sample_ids=padded_sample_ids,
            )
            _print_live_progress(
                sample_id=sample_id,
                processed_sample_count=processed_sample_count,
                max_samples=max_samples,
                prediction_count=prediction_count,
                vlm_error_count=vlm_error_count,
                stats={},
                skip_vlm=True,
            )
            continue

        for selector_name, selected_frames in selected_frames_by_selector.items():
            if _selector_complete(sample_id, selector_name, question_ids, existing_prediction_keys):
                _log(verbose, f"[{sample_id}] Resume: selector={selector_name} already complete.")
                continue
            system_prompt, user_prompt, shuffled = build_vqa_prompt(sample_items)
            raw_prediction = ""
            vlm_error = None
            vlm_error_type = None
            try:
                raw_prediction = client.answer(
                    user_prompt,
                    selected_frames,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                vlm_error = f"{type(exc).__name__}: {exc}"
                vlm_error_type = type(exc).__name__
                vlm_error_count += 1
                if fail_on_vlm_error or not _can_continue_after_vlm_error(exc):
                    _persist_run_state(
                        manifest=manifest,
                        alignment_debug=alignment_debug,
                        predictions=existing_predictions,
                        output_path=output_path,
                        summary_base=summary_base,
                        processed_sample_count=processed_sample_count,
                        prediction_count=prediction_count,
                        vlm_error_count=vlm_error_count,
                        metric_availability_rows=metric_availability_rows,
                        padded_sample_ids=padded_sample_ids,
                    )
                    raise
                _log(verbose, f"[{sample_id}] VLM error for selector={selector_name}: {vlm_error}")

            parsed_answers = parse_vqa_answers(raw_prediction) if raw_prediction else {}
            for item in sample_items:
                parsed_value = parsed_answers.get(item.question_id)
                correct, parsed, score = score_robofine_answer(
                    item,
                    parsed_value,
                    shuffled.get(item.question_id),
                )
                if vlm_error and not raw_prediction:
                    correct, parsed, score = None, None, None
                prediction = RoboFinePrediction(
                    sample_id=sample_id,
                    dataset=dataset,
                    question_id=item.question_id,
                    selector_name=selector_name,
                    capability=item.capability,
                    answer_type=item.answer_type,
                    question=item.question,
                    reference_answer=item.answer,
                    raw_prediction=raw_prediction,
                    parsed_prediction=parsed,
                    correct=correct,
                    score=score,
                    selected_frame_indices=[frame.frame_index for frame in selected_frames],
                    selected_timestamps_sec=[frame.timestamp_sec for frame in selected_frames],
                    metadata={
                        "frame_budget": frame_budget,
                        "camera_name": camera_name,
                        "mode": item.mode,
                        "source_question_id": item.source_question_id,
                        "shuffled_correct_letter": (
                            shuffled[item.question_id].correct_letter
                            if item.question_id in shuffled
                            else None
                        ),
                        "vlm_error": vlm_error,
                        "vlm_error_type": vlm_error_type,
                    },
                )
                existing_predictions.append(prediction.to_dict())
                existing_prediction_keys.add((sample_id, item.question_id, selector_name))
                prediction_count += 1
                with predictions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(prediction.to_dict()) + "\n")

        _persist_run_state(
            manifest=manifest,
            alignment_debug=alignment_debug,
            predictions=existing_predictions,
            output_path=output_path,
            summary_base=summary_base,
            processed_sample_count=processed_sample_count,
            prediction_count=prediction_count,
            vlm_error_count=vlm_error_count,
            metric_availability_rows=metric_availability_rows,
            padded_sample_ids=padded_sample_ids,
        )
        _print_live_progress(
            sample_id=sample_id,
            processed_sample_count=processed_sample_count,
            max_samples=max_samples,
            prediction_count=prediction_count,
            vlm_error_count=vlm_error_count,
            stats=aggregate_robofine_stats(existing_predictions),
            skip_vlm=False,
        )

    stats = aggregate_robofine_stats(existing_predictions)
    metric_counts = _summarize_metric_availability(metric_availability_rows)
    unique_padded = sorted(set(padded_sample_ids))
    if unique_padded:
        _warn(
            f"{len(unique_padded)}/{processed_sample_count} sample(s) had metric frames padded "
            f"with uniform frames at smoothing_window={resolved_smoothing_window}. "
            "Those samples weaken the metric-vs-uniform contrast; see "
            "metric_uniform_padded_sample_ids in run_summary.json."
        )
    summary = dict(summary_base)
    summary.update(
        {
            "status": "complete",
            "processed_sample_count": processed_sample_count,
            "prediction_count": prediction_count,
            "vlm_error_count": vlm_error_count,
            "skipped_count": len(alignment_debug),
            "metric_availability_counts": metric_counts,
            "metric_uniform_padded_sample_count": len(unique_padded),
            "metric_uniform_padded_sample_ids": unique_padded,
            "stats": stats,
        }
    )
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(stats_path, stats)
    _write_json_atomic(alignment_path, alignment_debug)
    _write_json_atomic(summary_path, summary)
    _write_jsonl(metric_availability_path, metric_availability_rows)
    _log(
        verbose,
        (
            f"Finished RoboFine QA: samples={processed_sample_count}, "
            f"predictions={prediction_count}, output_dir={output_path}"
        ),
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RoboFine-Bench K-frame VQA evaluation.")
    parser.add_argument(
        "--eval-data",
        default=str(DEFAULT_EVAL_DATA),
        help="RoboFine-Bench EvalData directory.",
    )
    parser.add_argument("--output-dir", default="robofine_outputs", help="Output directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Benchmark datasets to evaluate. Defaults to RT-1, BridgeDataV2, BC-Z.",
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        metavar="NAME",
        default=list(DEFAULT_SELECTORS),
        help=(
            "Frame selectors to compare, space- or comma-separated. "
            f"Choose from all, {', '.join(DEFAULT_SELECTORS)} "
            f"(aliases: {', '.join(sorted(SELECTOR_ALIASES))}). Defaults to all."
        ),
    )
    parser.add_argument(
        "--top-k",
        "--frame-budget",
        dest="frame_budget",
        type=int,
        default=5,
        help="Number of keyframes each selector may pick. Also caps metric flood proposals.",
    )
    parser.add_argument("--max-samples", type=int, help="Optional cap on samples.")
    parser.add_argument("--config-path", help="Optional pathologist config override.")
    parser.add_argument(
        "--metric-ranking",
        choices=["area", "persistence"],
        default=DEFAULT_METRIC_RANKING,
        help="Rank metric proposals by area or persistence. Defaults to persistence.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help=(
            "Savitzky-Golay window (in samples) for the aggregate surprise curve. "
            "Even values round up to the next odd value, and values below 4 are a no-op. "
            "Defaults to the config value. Large windows erase peaks on low-rate datasets "
            "and force uniform padding."
        ),
    )
    parser.add_argument(
        "--plot",
        dest="plot_ecod_aggregates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write ECOD aggregate curves, flood maxima, and per-selector peak overlays under ecod_aggregates/.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for QA calls. Defaults to the paper's {DEFAULT_OPENAI_MODEL} baseline.",
    )
    parser.add_argument("--openai-temperature", type=float, help="OpenAI sampling temperature.")
    parser.add_argument("--openai-timeout-sec", type=float, help="Per-request OpenAI timeout seconds.")
    parser.add_argument("--openai-max-retries", type=int, help="Retries after the initial OpenAI request.")
    parser.add_argument("--openai-retry-backoff-sec", type=float, help="Initial retry backoff seconds.")
    parser.add_argument(
        "--fail-on-vlm-error",
        action="store_true",
        help="Stop if a retryable VLM request still fails.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run in --output-dir.")
    parser.add_argument("--skip-vlm", action="store_true", help="Load bundles and select frames without VLM calls.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        selectors = resolve_selector_names(args.selectors)
    except ValueError as exc:
        parser.error(str(exc))
    summary = run_robofine_qa_pipeline(
        eval_data=args.eval_data,
        output_dir=args.output_dir,
        datasets=args.datasets,
        selectors=selectors,
        frame_budget=args.frame_budget,
        max_samples=args.max_samples,
        config_path=args.config_path,
        metric_ranking=args.metric_ranking,
        smoothing_window=args.smoothing_window,
        plot_ecod_aggregates=args.plot_ecod_aggregates,
        openai_model=args.openai_model,
        openai_temperature=args.openai_temperature,
        openai_timeout_sec=args.openai_timeout_sec,
        openai_max_retries=args.openai_max_retries,
        openai_retry_backoff_sec=args.openai_retry_backoff_sec,
        fail_on_vlm_error=args.fail_on_vlm_error,
        resume=args.resume,
        skip_vlm=args.skip_vlm,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, default=str))


def _load_sample_bundle(
    *,
    sample: dict[str, Any],
    telemetry_manifest: dict[str, dict[str, Any]],
    verification: dict[str, dict[str, Any]],
    eval_data_path: Path,
    bench_dir: Path,
    frame_cache_dir: Path,
):
    sample_id = str(sample["sample_id"])
    manifest_entry = telemetry_manifest.get(sample_id)
    if manifest_entry is None:
        raise FileNotFoundError(f"{sample_id} is missing from telemetry_manifest.jsonl.")
    video_path, parquet_path, camera_name = resolve_sample_paths(
        sample,
        manifest_entry,
        eval_data=eval_data_path,
        bench_dir=bench_dir,
    )
    adapter = adapter_for_dataset(str(sample["dataset"]))
    bundle = adapter.load(
        sample_id=sample_id,
        video_path=video_path,
        parquet_path=parquet_path,
        camera_name=camera_name,
        frame_cache_dir=frame_cache_dir,
        manifest_entry=manifest_entry,
        verification_entry=verification.get(sample_id),
        eval_sample=sample,
    )
    return bundle, video_path, parquet_path, camera_name, manifest_entry


def _set_visual_flow_camera_names(config: dict[str, Any], camera_names: list[str]) -> None:
    metrics_config = config.setdefault("metrics", {})
    if not isinstance(metrics_config, dict):
        return
    for metric_name in (
        "visual_flow",
        "visual_patch_novelty",
        "object_interaction_motion",
    ):
        visual_config = metrics_config.setdefault(metric_name, {})
        if isinstance(visual_config, dict):
            visual_config["camera_names"] = list(camera_names)


def _metric_availability(profile: Any) -> dict[str, Any]:
    availability: dict[str, Any] = {}
    for metric in getattr(profile, "metrics", []):
        metadata = metric.metadata if isinstance(getattr(metric, "metadata", None), dict) else {}
        n_samples = len(getattr(metric, "samples", None) or [])
        available = metadata.get("available")
        if available is None:
            available = n_samples >= 2
        availability[metric.name] = {
            "available": bool(available),
            "unavailable_reason": metadata.get("unavailable_reason"),
            "n_samples": n_samples,
        }
    return availability


def _summarize_metric_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for name, info in metrics.items():
            bucket = counts.setdefault(name, {"available": 0, "unavailable": 0})
            if isinstance(info, dict) and info.get("available"):
                bucket["available"] += 1
            else:
                bucket["unavailable"] += 1
    return counts


def _selector_complete(
    sample_id: str,
    selector_name: str,
    question_ids: list[str],
    existing_keys: set[tuple[str, str, str]],
) -> bool:
    return all((sample_id, question_id, selector_name) in existing_keys for question_id in question_ids)


def _prediction_key_from_row(row: dict[str, Any]) -> tuple[str, str, str] | None:
    sample_id = row.get("sample_id")
    question_id = row.get("question_id")
    selector_name = row.get("selector_name")
    if isinstance(sample_id, str) and isinstance(question_id, str) and isinstance(selector_name, str):
        return sample_id, question_id, selector_name
    return None


def resolve_selector_names(values: Any) -> list[str]:
    """Normalize requested selectors into a deduped list in canonical order.

    Accepts ``None`` or ``"all"`` for every selector, space- or comma-separated tokens,
    and the aliases in ``SELECTOR_ALIASES``. Canonical ordering keeps the per-selector
    columns in stats, manifests, and overlay plots comparable across runs.
    """

    if values is None:
        return list(DEFAULT_SELECTORS)
    if isinstance(values, str):
        values = [values]
    requested: set[str] = set()
    for value in values:
        for token in str(value).replace(",", " ").split():
            name = token.strip().lower()
            if not name:
                continue
            if name == "all":
                requested.update(DEFAULT_SELECTORS)
                continue
            name = SELECTOR_ALIASES.get(name, name)
            if name not in DEFAULT_SELECTORS:
                allowed = ", ".join(["all", *DEFAULT_SELECTORS, *sorted(SELECTOR_ALIASES)])
                raise ValueError(f"Unknown selector {token!r}. Choose from: {allowed}.")
            requested.add(name)
    if not requested:
        raise ValueError("At least one selector is required.")
    return [name for name in DEFAULT_SELECTORS if name in requested]


def _resolve_metric_ranking(value: Any) -> str:
    """Normalize a metric ranking name to 'area' or 'persistence'."""

    aliases = {
        "area": "area",
        "area_above_base": "area",
        "persistence": "persistence",
    }
    ranking = aliases.get(str(value).strip().lower())
    if ranking is None:
        raise ValueError("metric_ranking must be 'area' or 'persistence'.")
    return ranking


def _surprise_flood_config(config: dict[str, Any]) -> dict[str, Any]:
    selector_config = config.setdefault("selector", {})
    if not isinstance(selector_config, dict):
        raise ValueError("The selector config must be an object.")
    surprise_flood_config = selector_config.setdefault("surprise_flood", {})
    if not isinstance(surprise_flood_config, dict):
        raise ValueError("The selector.surprise_flood config must be an object.")
    return surprise_flood_config


def _set_metric_ranking(config: dict[str, Any], ranking: str) -> None:
    """Write the resolved ranking into selector.surprise_flood."""

    _surprise_flood_config(config)["ranking"] = _resolve_metric_ranking(ranking)


def _set_max_proposals(config: dict[str, Any], top_k: int) -> None:
    """Keep metric flood proposals aligned with the requested keyframe budget."""

    _surprise_flood_config(config)["max_proposals"] = int(top_k)


def _set_smoothing_window(config: dict[str, Any], smoothing_window: int) -> None:
    """Override the Savitzky-Golay window applied to the aggregate surprise curve."""

    window = int(smoothing_window)
    if window < 1:
        raise ValueError("smoothing_window must be at least 1.")
    _surprise_flood_config(config)["smoothing_window"] = window


def _validate_resume_settings(
    *,
    existing_summary: dict[str, Any],
    datasets: list[str],
    selectors: list[str],
    frame_budget: int,
    metric_ranking: str,
    smoothing_window: int,
    skip_vlm: bool,
) -> None:
    if not existing_summary:
        return
    expected = {
        "datasets": datasets,
        "selectors": selectors,
        "frame_budget": frame_budget,
        "metric_ranking": metric_ranking,
        "smoothing_window": smoothing_window,
    }
    for key, value in expected.items():
        existing = existing_summary.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"Resume settings mismatch for {key}: existing={existing} requested={value}")
    existing_skip = existing_summary.get("skip_vlm")
    if existing_skip is not None and existing_skip != skip_vlm:
        if existing_skip is True and skip_vlm is False:
            return
        if existing_skip is False and skip_vlm is True and not existing_summary.get("prediction_count"):
            return
        raise ValueError(
            f"Resume settings mismatch for skip_vlm: existing={existing_skip} requested={skip_vlm}"
        )


def _build_run_summary(
    *,
    eval_data_path: Path,
    output_path: Path,
    datasets: list[str],
    selectors: list[str],
    frame_budget: int,
    metric_ranking: str,
    smoothing_window: int,
    skip_vlm: bool,
    resume: bool,
    sample_count: int,
    question_count: int,
    telemetry_sample_count: int,
    openai_model: str | None,
) -> dict[str, Any]:
    return {
        "eval_data": str(eval_data_path),
        "output_dir": str(output_path),
        "datasets": datasets,
        "selectors": selectors,
        "frame_budget": frame_budget,
        "metric_ranking": metric_ranking,
        "smoothing_window": smoothing_window,
        "skip_vlm": skip_vlm,
        "resume": resume,
        "sample_count": sample_count,
        "question_count": question_count,
        "telemetry_sample_count": telemetry_sample_count,
        "openai_model": openai_model,
        "status": "running",
        "manifest_path": str(output_path / "robofine_manifest.json"),
        "predictions_path": str(output_path / "robofine_predictions.jsonl"),
        "stats_path": str(output_path / "robofine_stats.json"),
    }


def _persist_run_state(
    *,
    manifest: list[dict[str, Any]],
    alignment_debug: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    summary_base: dict[str, Any],
    processed_sample_count: int,
    prediction_count: int,
    vlm_error_count: int,
    metric_availability_rows: list[dict[str, Any]],
    padded_sample_ids: list[str] | None = None,
) -> None:
    stats = aggregate_robofine_stats(predictions)
    padded = sorted(set(padded_sample_ids or []))
    summary = dict(summary_base)
    summary.update(
        {
            "processed_sample_count": processed_sample_count,
            "prediction_count": prediction_count,
            "vlm_error_count": vlm_error_count,
            "metric_availability_counts": _summarize_metric_availability(metric_availability_rows),
            "metric_uniform_padded_sample_count": len(padded),
            "metric_uniform_padded_sample_ids": padded,
            "stats": stats,
        }
    )
    _write_json_atomic(output_path / "robofine_manifest.json", manifest)
    _write_json_atomic(output_path / "robofine_stats.json", stats)
    _write_json_atomic(output_path / "robofine_alignment_debug.json", alignment_debug)
    _write_json_atomic(output_path / "run_summary.json", summary)
    _write_jsonl(output_path / "metric_availability.jsonl", _unique_metric_availability_rows(metric_availability_rows))


def _load_resume_state(
    *,
    predictions_path: Path,
    manifest_path: Path,
    alignment_path: Path,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = _load_jsonl(predictions_path)
    manifest = _load_json_list(manifest_path)
    alignment_debug = _load_json_list(alignment_path)
    _log(
        verbose,
        (
            f"Resume loaded {len(predictions)} prediction row(s), "
            f"{len(manifest)} manifest entrie(s)."
        ),
    )
    return predictions, manifest, alignment_debug


def _write_selector_debug(output_dir: Path, selected_frames_by_selector: dict[str, list[Any]]) -> None:
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


SELECTOR_PLOT_STYLE = {
    "uniform": ("#2ca02c", "o"),
    "kite_flow": ("#ff7f0e", "s"),
    "flow": ("#ff7f0e", "s"),
    "metric": ("#d62728", "*"),
}


def _write_selector_overlay_plot(
    *,
    ecod_aggregate_path: Path,
    sample_id: str,
    selected_frames_by_selector: dict[str, list[Any]],
    metric_ranking: str,
) -> str | None:
    """Overlay every selector's chosen peaks on the smoothed ECOD aggregate curve."""

    data_path = ecod_aggregate_output_paths(ecod_aggregate_path)["data"]
    if not data_path.exists():
        return f"missing {data_path.name}"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if not payload.get("available"):
        return str(payload.get("reason") or "aggregate metric unavailable")

    timestamps = payload.get("timestamps_sec") or []
    raw_scores = payload.get("raw_scores") or []
    smoothed_scores = payload.get("smoothed_scores") or []
    if not timestamps or not smoothed_scores:
        return "aggregate curve has no samples"

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/pathologist_matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    def score_at(timestamp_sec: float) -> float:
        nearest = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - timestamp_sec))
        return float(smoothed_scores[nearest])

    plot_path = ecod_aggregate_path.with_name(f"{ecod_aggregate_path.name}_selectors.png")
    try:
        figure, axis = plt.subplots(figsize=(11, 4.5))
        if raw_scores:
            axis.plot(timestamps, raw_scores, color="#9a9a9a", linewidth=0.8, alpha=0.45, label="raw")
        axis.plot(timestamps, smoothed_scores, color="#111111", linewidth=1.4, label="smoothed")

        for event in payload.get("local_maxima") or []:
            axis.axvline(
                float(event["peak_sec"]),
                color="#1f77b4",
                linewidth=0.7,
                alpha=0.35,
                zorder=1,
            )

        for selector_name, frames in selected_frames_by_selector.items():
            color, marker = SELECTOR_PLOT_STYLE.get(selector_name, ("#7f7f7f", "D"))
            picked = [float(frame.timestamp_sec) for frame in frames]
            if not picked:
                continue
            axis.scatter(
                picked,
                [score_at(timestamp) for timestamp in picked],
                color=color,
                marker=marker,
                s=130 if marker == "*" else 55,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.6,
                label=f"{selector_name} (K={len(picked)})",
                zorder=5,
            )
            for timestamp in picked:
                axis.axvline(timestamp, color=color, linewidth=0.9, alpha=0.35, linestyle="--", zorder=2)

        axis.set_xlabel("timestamp_sec")
        axis.set_ylabel("Aggregate ECOD score")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        figure.suptitle(f"{sample_id}: selector keyframes on ECOD aggregate (ranking={metric_ranking})")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _print_live_progress(
    *,
    sample_id: str,
    processed_sample_count: int,
    max_samples: int | None,
    prediction_count: int,
    vlm_error_count: int,
    stats: dict[str, Any],
    skip_vlm: bool,
) -> None:
    target = str(max_samples) if max_samples is not None else "all"
    if skip_vlm:
        metrics = "skip-vlm"
    else:
        parts = []
        for selector, selector_stats in stats.items():
            overall = selector_stats.get("overall", {})
            accuracy = overall.get("accuracy")
            count = overall.get("count", 0)
            if accuracy is None:
                parts.append(f"{selector}=n/a(n={count})")
            else:
                parts.append(f"{selector}={accuracy:.3f}(n={count})")
        metrics = " | ".join(parts) if parts else "no scored predictions yet"
    print(
        (
            f"[robofine] progress {processed_sample_count}/{target} samples "
            f"| predictions={prediction_count} | vlm_errors={vlm_error_count} "
            f"| last_sample={sample_id} | {metrics}"
        ),
        flush=True,
    )


def _can_continue_after_vlm_error(exc: Exception) -> bool:
    if isinstance(exc, OpenAIRoboFACQAError):
        return exc.retryable
    return isinstance(exc, (TimeoutError, urllib.error.URLError, OSError))


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _unique_metric_availability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            unique[sample_id] = row
    return [unique[sample_id] for sample_id in sorted(unique)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _warn(message: str) -> None:
    """Print a warning regardless of --verbose, since silent fallbacks skew the comparison."""

    print(f"[robofine][warn] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
