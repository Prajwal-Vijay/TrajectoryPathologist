"""Plot cached Qwen yes/no margins against ABC insert/remove intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from VL_implemenation.abc_data import Episode, discover_episodes
from VL_implemenation.yesno_scan import cache_paths


EVENT_COLORS = {
    "insertion": "#2ca02c",
    "removal": "#d62728",
}


def plot_episode(
    episode: Episode,
    *,
    cache_root: Path,
    plot_root: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path, _ = cache_paths(cache_root, episode)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing cached scores: {npz_path}")
    with np.load(npz_path) as cached:
        centers = np.asarray(cached["center_sec"], dtype=np.float64)
        margins = np.asarray(cached["margin"], dtype=np.float64)
        masses = np.asarray(cached["yes_no_mass"], dtype=np.float64)
    p_yes = 1.0 / (1.0 + np.exp(-np.clip(margins, -80.0, 80.0)))

    figure, (margin_axis, probability_axis) = plt.subplots(
        2,
        1,
        figsize=(13, 6),
        sharex=True,
        constrained_layout=True,
    )
    margin_axis.plot(centers, margins, color="#1f77b4", linewidth=1.2)
    margin_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    margin_axis.set_ylabel("Yes − No logit")
    probability_axis.plot(centers, p_yes, color="#9467bd", linewidth=1.2)
    probability_axis.axhline(0.5, color="black", linewidth=0.7, alpha=0.6)
    probability_axis.set_ylabel("sigmoid(margin)")
    probability_axis.set_xlabel("trajectory time (s)")
    probability_axis.set_ylim(-0.02, 1.02)

    labels_seen: set[str] = set()
    for interval in episode.ground_intervals:
        color = EVENT_COLORS[interval.event_type]
        label = interval.event_type if interval.event_type not in labels_seen else None
        labels_seen.add(interval.event_type)
        for axis in (margin_axis, probability_axis):
            axis.axvspan(
                interval.start_sec,
                interval.end_sec,
                color=color,
                alpha=0.18,
                label=label,
            )
    margin_axis.legend(loc="upper right")
    margin_axis.set_title(f"{episode.task_name} / {episode.bundle_id}")

    output_path = plot_root / episode.task_name / f"{episode.bundle_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    inside = np.asarray(
        [
            any(
                interval.start_sec <= center < interval.end_sec
                for interval in episode.ground_intervals
            )
            for center in centers
        ],
        dtype=bool,
    )
    return {
        "task_name": episode.task_name,
        "bundle_id": episode.bundle_id,
        "plot_path": str(output_path),
        "window_count": int(len(centers)),
        "margin_std": float(np.std(margins)),
        "inside_window_count": int(np.sum(inside)),
        "inside_margin_median": (
            float(np.median(margins[inside])) if np.any(inside) else None
        ),
        "outside_margin_median": (
            float(np.median(margins[~inside])) if np.any(~inside) else None
        ),
        "maximum_margin": float(np.max(margins)),
        "maximum_margin_time_sec": float(centers[int(np.argmax(margins))]),
        "maximum_inside_ground": bool(inside[int(np.argmax(margins))]),
        "yes_no_mass_below_0_5_count": int(np.sum(masses < 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("VL_implemenation/config.json"),
    )
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    episodes = [
        episode
        for episode in discover_episodes(config["data_root"])
        if episode.ground_intervals
    ][: args.count]
    output_root = Path(config["output_root"]).expanduser().resolve()
    cache_root = output_root / "vlm_scores"
    plot_root = output_root / "yesno_sanity_plots"
    rows = [
        plot_episode(
            episode,
            cache_root=cache_root,
            plot_root=plot_root,
        )
        for episode in episodes
    ]
    summary = {
        "episode_count": len(rows),
        "plot_root": str(plot_root),
        "flat_curve_count_std_below_0_1": sum(
            row["margin_std"] < 0.1 for row in rows
        ),
        "maximum_inside_ground_count": sum(
            row["maximum_inside_ground"] for row in rows
        ),
        "median_inside_minus_outside_margin": float(
            np.median(
                [
                    row["inside_margin_median"] - row["outside_margin_median"]
                    for row in rows
                    if row["inside_margin_median"] is not None
                    and row["outside_margin_median"] is not None
                ]
            )
        ),
        "episodes": rows,
    }
    summary_path = output_root / "yesno_sanity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
