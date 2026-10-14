"""Compare persistence and peak-height NMS on cached PRISM local maxima."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from VL_implemenation.prism_ablation import _bootstrap_metrics, _score_rows
from VL_implemenation.vlm_peak_gate import _paired_bootstrap_safe


DEFAULT_METHODS = ("vlm_only", "grouped")


def select_nms_peaks(
    peaks: Sequence[dict[str, Any]],
    *,
    radius_sec: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank local maxima by peak height and greedily suppress nearby maxima."""

    if radius_sec < 0:
        raise ValueError("radius_sec must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
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
        if any(
            abs(float(peak["time_sec"]) - float(previous["time_sec"]))
            <= radius_sec
            for previous in selected
        ):
            continue
        selected.append(
            {
                **peak,
                "source_persistence_rank": peak.get("rank"),
                "rank": len(selected) + 1,
                "selection_mode": "peak_height_nms",
                "nms_radius_sec": float(radius_sec),
            }
        )
        if len(selected) >= top_k:
            break
    return selected


def build_nms_rows(
    rows: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    radius_sec: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Attach NMS-selected peaks without modifying persistence selections."""

    output = []
    for row in rows:
        all_peaks = row.get("all_peaks")
        if not isinstance(all_peaks, dict):
            raise ValueError(
                f"{row.get('bundle_id')} has no all_peaks; rerun prism_ablation"
            )
        nms_peaks = {}
        for method in methods:
            if method not in all_peaks:
                raise ValueError(
                    f"{row.get('bundle_id')} has no complete peaks for {method}; "
                    "rerun prism_ablation.py run --no-resume"
                )
            nms_peaks[f"{method}_nms"] = select_nms_peaks(
                all_peaks[method],
                radius_sec=radius_sec,
                top_k=top_k,
            )
        output.append(
            {
                **row,
                "peaks": {**row.get("peaks", {}), **nms_peaks},
            }
        )
    return output


def evaluate_nms(
    rows: Sequence[dict[str, Any]],
    *,
    methods: Sequence[str],
    radius_sec: float,
    top_k: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Score NMS selections and paired differences from persistence."""

    event_rows = [row for row in rows if row["ground_intervals"]]
    table = {}
    comparisons = {}
    for method in methods:
        nms_method = f"{method}_nms"
        for evaluated_method in (method, nms_method):
            metrics = _score_rows(event_rows, method_name=evaluated_method)
            metrics["bootstrap_95ci"] = _bootstrap_metrics(
                event_rows,
                method_name=evaluated_method,
                samples=bootstrap_samples,
                seed=seed,
            )
            table[evaluated_method] = metrics
        comparisons[f"{nms_method}_minus_{method}"] = _paired_bootstrap_safe(
            event_rows,
            left=nms_method,
            right=method,
            samples=bootstrap_samples,
            seed=seed,
        )
    return {
        "protocol": {
            "candidate_source": "all interior local maxima of smoothed ECOD aggregate",
            "nms_ranking": "aggregate ECOD peak_score descending",
            "nms_radius_sec": radius_sec,
            "top_k": top_k,
            "persistence_baseline": "same candidates ranked by persistence",
            "bootstrap_episode_resamples": bootstrap_samples,
            "seed": seed,
        },
        "episode_count": len(rows),
        "event_episode_count": len(event_rows),
        "table": table,
        "paired_differences": comparisons,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--radius-sec", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=5)
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
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else external_output / "nms_ablation"
    )
    rows = build_nms_rows(
        _read_jsonl(ranked_peaks),
        methods=args.methods,
        radius_sec=args.radius_sec,
        top_k=args.top_k,
    )
    _write_jsonl_atomic(output_root / "ranked_peaks.jsonl", rows)
    summary = evaluate_nms(
        rows,
        methods=args.methods,
        radius_sec=args.radius_sec,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_json_atomic(output_root / "evaluation.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
