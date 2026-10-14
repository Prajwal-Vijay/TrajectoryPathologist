"""Gate existing telemetry peaks with cached Qwen insert/remove margins.

This stage is CPU-only: it reads the already-ranked telemetry peaks and the
already-computed VLM window scores. It does not decode videos, run Qwen, or
recompute telemetry metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from VL_implemenation.prism_ablation import (
    _bootstrap_metrics,
    _score_rows,
)


DEFAULT_MARGIN_THRESHOLDS = (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
GATED_METHOD_NAME = "telemetry_vlm_gate"


def build_gated_rows(
    rows: Sequence[dict[str, Any]],
    *,
    cache_root: Path,
    thresholds: Sequence[float],
    source_method: str = "telemetry",
    selection_mode: str = "persistence",
    nms_radius_sec: float = 8.0,
) -> list[dict[str, Any]]:
    """Attach gated peak lists for every threshold, loading each cache once."""

    unique_thresholds = _unique_thresholds(thresholds)
    output: list[dict[str, Any]] = []
    for row in rows:
        task_name = str(row["task_name"])
        bundle_id = str(row["bundle_id"])
        cache_path = cache_root / task_name / f"{bundle_id}.npz"
        all_peaks = row.get("all_peaks")
        if not isinstance(all_peaks, dict) or source_method not in all_peaks:
            raise ValueError(
                f"{task_name}/{bundle_id} does not contain all {source_method} "
                "peaks; rerun prism_ablation.py run --no-resume"
            )
        source_peaks = list(all_peaks[source_method])
        try:
            centers, margins, masses = _load_cache(cache_path)
            gated = {
                threshold_key(threshold): gate_peaks(
                    source_peaks,
                    centers=centers,
                    margins=margins,
                    yes_no_masses=masses,
                    threshold=threshold,
                    max_peaks=5,
                    selection_mode=selection_mode,
                    nms_radius_sec=nms_radius_sec,
                )
                for threshold in unique_thresholds
            }
            available = True
            unavailable_reason = None
        except (OSError, KeyError, ValueError) as exc:
            gated = {
                threshold_key(threshold): [] for threshold in unique_thresholds
            }
            available = False
            unavailable_reason = f"{type(exc).__name__}: {exc}"
        output.append(
            {
                **row,
                "gate_source_method": source_method,
                "gate_cache_path": str(cache_path),
                "gate_cache_available": available,
                "gate_unavailable_reason": unavailable_reason,
                "gate_source_peak_count": len(source_peaks),
                "gate_selection_mode": selection_mode,
                "gate_nms_radius_sec": (
                    nms_radius_sec if selection_mode == "nms" else None
                ),
                "gated_peaks": gated,
            }
        )
    return output


def gate_peaks(
    peaks: Sequence[dict[str, Any]],
    *,
    centers: np.ndarray,
    margins: np.ndarray,
    yes_no_masses: np.ndarray,
    threshold: float,
    max_peaks: int | None = None,
    selection_mode: str = "persistence",
    nms_radius_sec: float = 8.0,
) -> list[dict[str, Any]]:
    """Gate all source peaks, then rank by persistence or peak-height NMS."""

    if max_peaks is not None and max_peaks <= 0:
        raise ValueError("max_peaks must be positive or None")
    if selection_mode not in {"persistence", "nms"}:
        raise ValueError("selection_mode must be 'persistence' or 'nms'")
    if nms_radius_sec < 0:
        raise ValueError("nms_radius_sec must be non-negative")
    eligible: list[dict[str, Any]] = []
    for peak in peaks:
        peak_sec = float(peak["time_sec"])
        index = nearest_time_index(centers, peak_sec)
        margin = float(margins[index])
        if margin <= threshold:
            continue
        eligible.append(
            {
                **peak,
                "source_rank": peak.get("rank"),
                "vlm_center_sec": float(centers[index]),
                "vlm_time_offset_sec": peak_sec - float(centers[index]),
                "vlm_margin": margin,
                "vlm_p_yes": _sigmoid(margin),
                "vlm_yes_no_mass": float(yes_no_masses[index]),
                "vlm_margin_threshold": float(threshold),
            }
        )
    selected = _select_gated_peaks(
        eligible,
        selection_mode=selection_mode,
        nms_radius_sec=nms_radius_sec,
        max_peaks=max_peaks,
    )
    return [
        {
            **peak,
            "rank": rank,
            "gate_selection_mode": selection_mode,
            "nms_radius_sec": (
                float(nms_radius_sec) if selection_mode == "nms" else None
            ),
        }
        for rank, peak in enumerate(selected, start=1)
    ]


def _select_gated_peaks(
    peaks: Sequence[dict[str, Any]],
    *,
    selection_mode: str,
    nms_radius_sec: float,
    max_peaks: int | None,
) -> list[dict[str, Any]]:
    if selection_mode == "persistence":
        ranked = list(peaks)
    else:
        ranked = sorted(
            peaks,
            key=lambda peak: (
                -float(peak["peak_score"]),
                -float(peak.get("persistence", 0.0)),
                float(peak["time_sec"]),
            ),
        )
    selected: list[dict[str, Any]] = []
    for peak in ranked:
        if selection_mode == "nms" and any(
            abs(float(peak["time_sec"]) - float(previous["time_sec"]))
            <= nms_radius_sec
            for previous in selected
        ):
            continue
        selected.append(peak)
        if max_peaks is not None and len(selected) >= max_peaks:
            break
    return selected


def nearest_time_index(times: np.ndarray, timestamp_sec: float) -> int:
    """Return nearest sorted timestamp index, choosing the lower index on ties."""

    if times.ndim != 1 or not len(times):
        raise ValueError("times must be a non-empty one-dimensional array")
    right = int(np.searchsorted(times, timestamp_sec, side="left"))
    if right == 0:
        return 0
    if right == len(times):
        return len(times) - 1
    left = right - 1
    if timestamp_sec - float(times[left]) <= float(times[right]) - timestamp_sec:
        return left
    return right


def evaluate_gates(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    bootstrap_samples: int,
    seed: int,
    source_method: str = "telemetry",
    selection_mode: str = "persistence",
    nms_radius_sec: float = 8.0,
) -> dict[str, Any]:
    """Evaluate every fixed threshold without selecting one on test results."""

    event_rows = [row for row in rows if row["ground_intervals"]]
    no_event_rows = [row for row in rows if not row["ground_intervals"]]
    baseline = {
        method: _score_rows(event_rows, method_name=method)
        for method in (source_method, "vlm_only")
    }
    sweep = []
    for threshold in _unique_thresholds(thresholds):
        key = threshold_key(threshold)
        scoring_rows = _scoring_rows(rows, key)
        scoring_event_rows = [
            row for row in scoring_rows if row["ground_intervals"]
        ]
        scoring_no_event_rows = [
            row for row in scoring_rows if not row["ground_intervals"]
        ]
        metrics = _score_rows(
            scoring_event_rows,
            method_name=GATED_METHOD_NAME,
        )
        metrics["bootstrap_95ci"] = _bootstrap_metrics(
            scoring_event_rows,
            method_name=GATED_METHOD_NAME,
            samples=bootstrap_samples,
            seed=seed,
        )
        sweep.append(
            {
                "threshold_margin": threshold,
                "equivalent_conditional_p_yes": _sigmoid(threshold),
                "metrics": metrics,
                "paired_vs_source": _paired_bootstrap_safe(
                    scoring_event_rows,
                    left=GATED_METHOD_NAME,
                    right=source_method,
                    samples=bootstrap_samples,
                    seed=seed,
                ),
                "event_retention": _retention_summary(
                    scoring_event_rows,
                    source_method=source_method,
                ),
                "no_event_retention": _retention_summary(
                    scoring_no_event_rows,
                    source_method=source_method,
                ),
            }
        )
    return {
        "protocol": {
            "source_method": source_method,
            "source_peak_limit": None,
            "gate_score": "cached Qwen Yes-minus-No logit margin",
            "lookup": "nearest VLM window center; lower center wins exact ties",
            "comparison": "strict margin > threshold",
            "selection_order": (
                "extract all telemetry local maxima, gate every candidate, then "
                + (
                    "rank by persistence and take top 5"
                    if selection_mode == "persistence"
                    else "rank by ECOD peak height with temporal NMS and take top 5"
                )
            ),
            "selection_mode": selection_mode,
            "nms_radius_sec": (
                nms_radius_sec if selection_mode == "nms" else None
            ),
            "gate_before_top_k": True,
            "threshold_selection": (
                "fixed sweep; choosing a threshold from these test results would "
                "require separate validation"
            ),
            "bootstrap_episode_resamples": bootstrap_samples,
            "seed": seed,
        },
        "episode_count": len(rows),
        "event_episode_count": len(event_rows),
        "no_event_episode_count": len(no_event_rows),
        "cache_available_count": sum(
            bool(row["gate_cache_available"]) for row in rows
        ),
        "cache_missing_count": sum(
            not bool(row["gate_cache_available"]) for row in rows
        ),
        "baseline": baseline,
        "threshold_sweep": sweep,
    }


def threshold_key(threshold: float) -> str:
    return f"margin_gt_{float(threshold):g}"


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as cached:
        centers = np.asarray(cached["center_sec"], dtype=np.float64)
        margins = np.asarray(cached["margin"], dtype=np.float64)
        masses = np.asarray(cached["yes_no_mass"], dtype=np.float64)
    if any(array.ndim != 1 for array in (centers, margins, masses)):
        raise ValueError("cache arrays must be one-dimensional")
    if not (len(centers) == len(margins) == len(masses)):
        raise ValueError("cache arrays have unequal lengths")
    if not len(centers):
        raise ValueError("cache contains no windows")
    if not all(np.all(np.isfinite(array)) for array in (centers, margins, masses)):
        raise ValueError("cache contains non-finite values")
    if np.any(np.diff(centers) < 0):
        raise ValueError("cache window centers are not sorted")
    return centers, margins, masses


def _scoring_rows(
    rows: Sequence[dict[str, Any]],
    threshold_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "peaks": {
                **row.get("peaks", {}),
                GATED_METHOD_NAME: row["gated_peaks"][threshold_name],
            },
        }
        for row in rows
    ]


def _retention_summary(
    rows: Sequence[dict[str, Any]],
    *,
    source_method: str,
) -> dict[str, Any]:
    source_count = sum(
        len(row.get("all_peaks", {}).get(source_method, [])) for row in rows
    )
    retained_count = sum(
        len(row.get("peaks", {}).get(GATED_METHOD_NAME, [])) for row in rows
    )
    episode_count = len(rows)
    episodes_with_retained_peak = sum(
        bool(row.get("peaks", {}).get(GATED_METHOD_NAME, [])) for row in rows
    )
    return {
        "source_peak_count": source_count,
        "retained_peak_count": retained_count,
        "retained_peak_fraction": (
            retained_count / source_count if source_count else None
        ),
        "episodes_with_retained_peak": episodes_with_retained_peak,
        "episode_any_peak_rate": (
            episodes_with_retained_peak / episode_count if episode_count else None
        ),
    }


def _paired_bootstrap_safe(
    rows: Sequence[dict[str, Any]],
    *,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap that permits undefined precision after strict gating."""

    metric_names = (
        "trajectory_hit_at_1",
        "trajectory_hit_at_3",
        "trajectory_hit_at_5",
        "segment_recall_at_5",
        "precision_at_5",
    )
    observed_left = _score_rows(rows, method_name=left)
    observed_right = _score_rows(rows, method_name=right)
    output = {}
    for name in metric_names:
        left_value = observed_left[name]
        right_value = observed_right[name]
        output[name] = {
            "difference": (
                left_value - right_value
                if left_value is not None and right_value is not None
                else None
            ),
            "bootstrap_95ci": [],
            "valid_bootstrap_samples": 0,
        }
    if samples <= 0 or not rows:
        return output

    rng = np.random.default_rng(seed)
    differences = {name: [] for name in metric_names}
    for _ in range(samples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        left_metrics = _score_rows(sampled, method_name=left)
        right_metrics = _score_rows(sampled, method_name=right)
        for name in metric_names:
            left_value = left_metrics[name]
            right_value = right_metrics[name]
            if left_value is not None and right_value is not None:
                differences[name].append(left_value - right_value)
    for name, values in differences.items():
        output[name]["valid_bootstrap_samples"] = len(values)
        if values:
            output[name]["bootstrap_95ci"] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
    return output


def _unique_thresholds(thresholds: Sequence[float]) -> list[float]:
    values = sorted({float(value) for value in thresholds})
    if not values or not all(np.isfinite(value) for value in values):
        raise ValueError("at least one finite threshold is required")
    return values


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -80.0, 80.0))))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("VL_implemenation/config.json"),
    )
    parser.add_argument("--ranked-peaks", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARGIN_THRESHOLDS),
    )
    parser.add_argument("--source-method", default="telemetry")
    parser.add_argument(
        "--selection",
        choices=["persistence", "nms"],
        default="persistence",
    )
    parser.add_argument("--nms-radius-sec", type=float, default=8.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    external_output = Path(config["output_root"]).expanduser().resolve()
    ranked_peaks = (
        args.ranked_peaks.expanduser().resolve()
        if args.ranked_peaks is not None
        else external_output / "prism_ablation" / "ranked_peaks.jsonl"
    )
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else external_output / "vlm_scores"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else external_output / (
            "vlm_peak_gate_nms"
            if args.selection == "nms"
            else "vlm_peak_gate"
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = build_gated_rows(
        _read_jsonl(ranked_peaks),
        cache_root=cache_root,
        thresholds=args.thresholds,
        source_method=args.source_method,
        selection_mode=args.selection,
        nms_radius_sec=args.nms_radius_sec,
    )
    _write_jsonl_atomic(output_root / "gated_peaks.jsonl", rows)
    summary = evaluate_gates(
        rows,
        thresholds=args.thresholds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        source_method=args.source_method,
        selection_mode=args.selection,
        nms_radius_sec=args.nms_radius_sec,
    )
    _write_json_atomic(output_root / "evaluation.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
