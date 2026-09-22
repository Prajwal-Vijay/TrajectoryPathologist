"""Create publication figures for the unpadded ABC-130K K=5 evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "abc130k"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#767676",
    "teal": "#42949E",
}

METHODS = ["Uniform", "KITE", "RONAR-K", "PRISM"]
METHOD_COLORS = {
    "Uniform": PALETTE["neutral"],
    "KITE": PALETTE["red_strong"],
    "RONAR-K": PALETTE["teal"],
    "PRISM": PALETTE["blue_main"],
}
METHOD_STYLES = {
    "Uniform": ("--", "o"),
    "KITE": ("-.", "s"),
    "RONAR-K": (":", "D"),
    "PRISM": ("-", "o"),
}

TASKS = [
    "Pick-and-\nPlace",
    "Fine\nP&P",
    "Folding",
    "Insertion /\nEjection",
    "Tool\nUse",
    "Sorting",
    "Tying /\nUntying",
]

# Values recomputed from ABC130K_failure_eval_k5_expanded after removing all
# uniform_padding selections. PRISM returns 1,317 native peaks over 280
# trajectories (4.70 per trajectory); all other methods return exactly 1,400.
TASK_RESULTS = {
    "Peak precision": {
        "Uniform": [8.0, 7.0, 4.5, 13.0, 10.5, 6.5, 8.0],
        "KITE": [9.0, 6.5, 6.5, 14.5, 14.5, 18.0, 10.0],
        "RONAR-K": [13.0, 5.0, 4.5, 12.0, 8.0, 10.5, 5.5],
        "PRISM": [18.0, 8.5, 7.0, 17.5, 14.0, 16.2, 11.6],
    },
    "Segment recall": {
        "Uniform": [13.5, 16.7, 14.8, 15.2, 17.1, 10.0, 22.2],
        "KITE": [12.6, 11.9, 16.4, 11.7, 19.0, 19.2, 20.8],
        "RONAR-K": [22.5, 11.9, 13.1, 14.0, 13.3, 15.0, 15.3],
        "PRISM": [27.9, 19.0, 13.1, 18.7, 23.8, 20.8, 29.2],
    },
    "Trajectory recall": {
        "Uniform": [32.5, 32.5, 20.0, 45.0, 32.5, 25.0, 35.0],
        "KITE": [30.0, 22.5, 22.5, 42.5, 42.5, 45.0, 37.5],
        "RONAR-K": [47.5, 22.5, 17.5, 50.0, 35.0, 37.5, 27.5],
        "PRISM": [60.0, 35.0, 20.0, 50.0, 50.0, 47.5, 45.0],
    },
}

OVERALL = {
    "Peak precision": [8.2, 11.3, 8.4, 13.5],
    "Segment recall": [15.2, 15.5, 15.2, 21.8],
    "Trajectory recall": [31.8, 34.6, 33.9, 43.9],
}
OVERALL_CI = {
    "Peak precision": ([6.6, 9.1, 6.9, 11.4], [9.9, 13.5, 9.9, 15.7]),
    "Segment recall": ([12.6, 12.8, 12.6, 18.8], [18.0, 18.4, 18.1, 25.1]),
    "Trajectory recall": ([26.4, 28.9, 28.6, 38.2], [37.5, 40.4, 39.6, 49.6]),
}


def apply_publication_style(font_size: int = 15) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.linewidth": 2.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"{basename}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.10,
        )


def make_radar_figure() -> None:
    apply_publication_style(font_size=14)
    theta = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False)
    closed_theta = np.r_[theta, theta[0]]
    panel_specs = [
        ("Peak precision", 20, [5, 10, 15, 20]),
        ("Segment recall", 35, [10, 20, 30]),
        ("Trajectory recall", 65, [20, 40, 60]),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.5, 7.8),
        subplot_kw={"projection": "polar"},
    )
    for panel, (metric, radial_max, radial_ticks) in zip(axes, panel_specs):
        panel.set_theta_offset(np.pi / 2)
        panel.set_theta_direction(-1)
        panel.set_xticks(theta)
        panel.set_xticklabels(TASKS, fontsize=12)
        panel.tick_params(axis="x", pad=14)
        panel.set_ylim(0, radial_max)
        panel.set_yticks(radial_ticks)
        panel.set_yticklabels([f"{value}%" for value in radial_ticks], fontsize=10)
        panel.set_rlabel_position(18)
        panel.grid(color="#CFCECE", linewidth=0.8, alpha=0.85)
        panel.spines["polar"].set_color("#4D4D4D")
        panel.spines["polar"].set_linewidth(1.2)
        panel.set_title(metric, fontsize=17, fontweight="bold", pad=27)

        for method in METHODS:
            values = np.asarray(TASK_RESULTS[metric][method], dtype=float)
            closed_values = np.r_[values, values[0]]
            linestyle, marker = METHOD_STYLES[method]
            linewidth = 3.0 if method == "PRISM" else 2.0
            zorder = 5 if method == "PRISM" else 3
            panel.plot(
                closed_theta,
                closed_values,
                color=METHOD_COLORS[method],
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=5.5,
                label=method,
                zorder=zorder,
            )
            if method == "PRISM":
                panel.fill(
                    closed_theta,
                    closed_values,
                    color=METHOD_COLORS[method],
                    alpha=0.10,
                    zorder=1,
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        fontsize=14,
        handlelength=3.2,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Task-wise failure localization on ABC-130K",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        "Unpadded selection, K≤5. Each task category contains 40 trajectories. Values are percentages.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0.145, 1, 0.93), pad=1.8, w_pad=1.6)
    save_all(fig, "taskwise_radar")
    plt.close(fig)


def make_overall_bar_figure() -> None:
    apply_publication_style(font_size=16)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    x = np.arange(len(METHODS))
    width = 0.23
    metric_styles = [
        ("Peak precision", PALETTE["red_2"], "///"),
        ("Segment recall", PALETTE["green_3"], "\\\\\\"),
        ("Trajectory recall", PALETTE["blue_secondary"], "..."),
    ]

    for index, (metric, color, hatch) in enumerate(metric_styles):
        values = np.asarray(OVERALL[metric], dtype=float)
        lower, upper = OVERALL_CI[metric]
        errors = np.vstack(
            [
                values - np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float) - values,
            ]
        )
        offset = (index - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=5,
            color=color,
            edgecolor="black",
            linewidth=1.4,
            hatch=hatch,
            error_kw={"elinewidth": 1.4, "capthick": 1.4},
            label=metric,
            zorder=3,
        )
        for bar, value, high in zip(bars, values, upper):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(high) + 1.25,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(x)
    tick_labels = ax.set_xticklabels(METHODS, fontsize=15)
    tick_labels[-1].set_fontweight("bold")
    tick_labels[-1].set_color(PALETTE["blue_main"])
    ax.set_ylabel("Score (%)", fontsize=17)
    ax.set_ylim(0, 56)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.grid(axis="y", color="#CFCECE", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        ncol=3,
        fontsize=13,
        handlelength=2.0,
        columnspacing=1.5,
    )
    ax.set_title(
        "Overall failure localization on ABC-130K",
        fontsize=21,
        fontweight="bold",
        pad=20,
    )
    fig.text(
        0.5,
        0.005,
        "Unpadded selection, K≤5; error bars show 95% trajectory-bootstrap confidence intervals. "
        "PRISM returns 4.70 events/trajectory; other methods return 5.00.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=1.7)
    save_all(fig, "overall_grouped_bars")
    plt.close(fig)


if __name__ == "__main__":
    make_radar_figure()
    make_overall_bar_figure()
