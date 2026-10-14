"""Inspect selected ABC-130K interaction peaks alongside cached frames and masks."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("/media/helen/28TB_VOL4/ABC130K_global_object_interaction_k5")
DEFAULT_FRAME_CACHE = Path(
    "/home/helen/Documents/TrajectoryPathologist/ABC130K_CVPR_all_methods_k5/frame_cache"
)
FRAME_TIME_RE = re.compile(r"_t(?P<seconds>\d+(?:\.\d+)?)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
OBJECT_PROMPT = "object held by robot. manipulated object. tool. component. part."
GRIPPER_PROMPT = "robot gripper."
OBJECT_SELECTION_DIGEST = "82fcc406982b485a"


@dataclass(frozen=True)
class Frame:
    path: Path
    seconds: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review ten ABC-130K trajectories from the CoTracker object-interaction run."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--selection",
        choices=["balanced", "first"],
        default="balanced",
        help="balanced selects five high-precision and five zero-precision trajectories.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Write the selected trajectories, peaks, and nearest top-camera frames to JSON.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected trajectories and peak/frame locations without opening the GUI.",
    )
    return parser


def _load_eval_rows(eval_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("selector_name") == "metric" and row.get("selected_peaks"):
                rows.append(row)
    if not rows:
        raise ValueError(f"No metric-selector rows found in {eval_path}")
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    count: int,
    selection: str,
) -> list[dict[str, Any]]:
    """Choose a deterministic, diagnostically useful subset."""

    count = max(1, min(int(count), len(rows)))
    if selection == "first":
        return rows[:count]

    def precision(row: dict[str, Any]) -> float:
        value = row.get("point_precision_at_k")
        if value is None:
            value = row.get("point_precision")
        return float(value or 0.0)

    by_quality = sorted(
        rows,
        key=lambda row: (
            -precision(row),
            str(row.get("task_name", "")),
            str(row.get("bundle_id", "")),
        ),
    )
    high_target = (count + 1) // 2
    chosen = by_quality[:high_target]
    chosen_ids = {str(row["bundle_id"]) for row in chosen}
    low_quality = sorted(
        rows,
        key=lambda row: (
            precision(row),
            str(row.get("task_name", "")),
            str(row.get("bundle_id", "")),
        ),
    )
    for row in low_quality:
        if str(row["bundle_id"]) in chosen_ids:
            continue
        chosen.append(row)
        chosen_ids.add(str(row["bundle_id"]))
        if len(chosen) == count:
            break
    return chosen


def _frame_seconds(path: Path) -> float | None:
    match = FRAME_TIME_RE.search(path.name)
    return float(match.group("seconds")) if match else None


def load_frames(frame_dir: Path) -> list[Frame]:
    frames: list[Frame] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for path in frame_dir.glob(pattern):
            seconds = _frame_seconds(path)
            if seconds is not None:
                frames.append(Frame(path=path, seconds=seconds))
    return sorted(frames, key=lambda frame: frame.seconds)


def camera_directories(
    frame_cache: Path,
    row: dict[str, Any],
) -> dict[str, Path]:
    episode_dir = frame_cache / str(row["task_name"]) / str(row["bundle_id"])
    directories: dict[str, Path] = {}
    for camera in ("top", "left_wrist", "right_wrist"):
        matches = sorted(episode_dir.glob(f"{camera}_*hz"))
        if matches:
            directories[camera] = matches[0]
    return directories


def nearest_frame(frames: list[Frame], seconds: float) -> Frame | None:
    if not frames:
        return None
    times = [frame.seconds for frame in frames]
    index = bisect.bisect_left(times, seconds)
    candidates = [max(0, index - 1), min(len(frames) - 1, index)]
    return min((frames[item] for item in candidates), key=lambda frame: abs(frame.seconds - seconds))


def build_manifest(
    rows: list[dict[str, Any]],
    frame_cache: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    for row in rows:
        directories = camera_directories(frame_cache, row)
        top_frames = load_frames(directories["top"]) if "top" in directories else []
        peaks: list[dict[str, Any]] = []
        for peak in row.get("selected_peaks", []):
            peak_copy = dict(peak)
            frame = nearest_frame(top_frames, float(peak["peak_sec"]))
            peak_copy["top_frame"] = str(frame.path) if frame else None
            peaks.append(peak_copy)
        trajectories.append(
            {
                "bundle_id": row["bundle_id"],
                "task_name": row["task_name"],
                "duration_sec": row.get("duration_sec"),
                "point_precision_at_k": row.get(
                    "point_precision_at_k", row.get("point_precision")
                ),
                "failed_segments": row.get("failed_segments", []),
                "selected_peaks": peaks,
                "camera_directories": {
                    name: str(path) for name, path in directories.items()
                },
            }
        )
    return {
        "source_eval": str(output_dir / "abc130k_eval.jsonl"),
        "frame_cache": str(frame_cache),
        "interaction_mask_cache": str(output_dir / "interaction_mask_cache"),
        "score_note": (
            "rank_score is the exact 0-D persistence of each selected peak. "
            "The historical run did not persist the full ECOD aggregate time series."
        ),
        "trajectories": trajectories,
    }


def print_manifest(manifest: dict[str, Any]) -> None:
    for index, trajectory in enumerate(manifest["trajectories"], start=1):
        print(
            f"{index:2d}. {trajectory['task_name']} / {trajectory['bundle_id']} "
            f"(precision={float(trajectory.get('point_precision_at_k') or 0.0):.2f})"
        )
        print(f"    frames: {trajectory['camera_directories'].get('top', 'missing')}")
        for peak in trajectory["selected_peaks"]:
            print(
                f"    peak {int(peak['rank'])}: {float(peak['peak_sec']):8.3f}s "
                f"persistence={float(peak.get('rank_score') or 0.0):7.3f} "
                f"frame={peak.get('top_frame') or 'missing'}"
            )


def _mask_cache_path(
    image_path: Path,
    cache_dir: Path,
    prompt: str,
) -> Path:
    stat = image_path.stat()
    payload = {
        "cache_version": 2,
        "path": str(image_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "prompt": prompt if prompt.endswith(".") else f"{prompt}.",
        "dino_model_id": "IDEA-Research/grounding-dino-tiny",
        "sam2_model_id": "facebook/sam2.1-hiera-tiny",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "mask_threshold": 0.0,
        "duplicate_iou_threshold": 0.8,
        "chunk_size_frames": 32,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / f"{digest}.png"


class InteractionViewer:
    def __init__(self, manifest: dict[str, Any]) -> None:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.manifest = manifest
        self.trajectories = manifest["trajectories"]
        self.mask_root = Path(manifest["interaction_mask_cache"])
        self.trajectory_index = 0
        self.camera_name = "top"
        self.frame_index = 0
        self.frames: list[Frame] = []
        self.photo: Any = None

        self.root = tk.Tk()
        self.root.title("ABC-130K Object Interaction Peak Viewer")
        self.root.geometry("1450x900")

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Previous trajectory [P]", command=self.previous_trajectory).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Next trajectory [N]", command=self.next_trajectory).pack(
            side=tk.LEFT, padx=(4, 12)
        )
        ttk.Button(controls, text="Previous peak [↑]", command=self.previous_peak).pack(
            side=tk.LEFT
        )
        ttk.Button(controls, text="Next peak [↓]", command=self.next_peak).pack(
            side=tk.LEFT, padx=(4, 12)
        )
        ttk.Button(controls, text="Camera [C]", command=self.next_camera).pack(side=tk.LEFT)

        self.object_mask = tk.BooleanVar(value=False)
        self.gripper_mask = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Object mask [M]",
            variable=self.object_mask,
            command=self.render_frame,
        ).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Checkbutton(
            controls,
            text="Gripper mask [G]",
            variable=self.gripper_mask,
            command=self.render_frame,
        ).pack(side=tk.LEFT)

        self.title_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.title_var, padding=(8, 2)).pack(fill=tk.X)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        image_panel = ttk.Frame(body)
        plot_panel = ttk.Frame(body)
        body.add(image_panel, weight=3)
        body.add(plot_panel, weight=2)

        self.image_label = ttk.Label(image_panel, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.frame_var = tk.StringVar()
        ttk.Label(image_panel, textvariable=self.frame_var, anchor=tk.CENTER).pack(fill=tk.X)

        self.figure = Figure(figsize=(6, 7), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.help_var = tk.StringVar(
            value=(
                "←/→ frame   ↑/↓ peak   N/P trajectory   C camera   "
                "M object mask   G gripper mask"
            )
        )
        ttk.Label(self.root, textvariable=self.help_var, padding=6).pack(fill=tk.X)

        self.root.bind("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind("<Right>", lambda _event: self.step_frame(1))
        self.root.bind("<Up>", lambda _event: self.previous_peak())
        self.root.bind("<Down>", lambda _event: self.next_peak())
        self.root.bind("n", lambda _event: self.next_trajectory())
        self.root.bind("p", lambda _event: self.previous_trajectory())
        self.root.bind("c", lambda _event: self.next_camera())
        self.root.bind("m", lambda _event: self._toggle(self.object_mask))
        self.root.bind("g", lambda _event: self._toggle(self.gripper_mask))
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        self.load_trajectory(0)

    @property
    def trajectory(self) -> dict[str, Any]:
        return self.trajectories[self.trajectory_index]

    def run(self) -> None:
        self.root.mainloop()

    def _toggle(self, variable: Any) -> None:
        variable.set(not variable.get())
        self.render_frame()

    def load_trajectory(self, index: int) -> None:
        self.trajectory_index = index % len(self.trajectories)
        available = self.trajectory["camera_directories"]
        if self.camera_name not in available:
            self.camera_name = next(iter(available), "top")
        frame_dir = Path(available[self.camera_name]) if self.camera_name in available else Path()
        self.frames = load_frames(frame_dir) if frame_dir.is_dir() else []
        self.frame_index = 0
        peaks = self.trajectory["selected_peaks"]
        if peaks and self.frames:
            frame = nearest_frame(self.frames, float(peaks[0]["peak_sec"]))
            if frame:
                self.frame_index = self.frames.index(frame)
        self.render_plot()
        self.render_frame()

    def step_frame(self, delta: int) -> None:
        if not self.frames:
            return
        self.frame_index = max(0, min(len(self.frames) - 1, self.frame_index + delta))
        self.render_frame()
        self.render_plot()

    def _jump_peak(self, delta: int) -> None:
        if not self.frames:
            return
        peaks = sorted(
            self.trajectory["selected_peaks"], key=lambda peak: float(peak["peak_sec"])
        )
        now = self.frames[self.frame_index].seconds
        if delta > 0:
            candidates = [peak for peak in peaks if float(peak["peak_sec"]) > now + 1e-6]
            peak = candidates[0] if candidates else peaks[0]
        else:
            candidates = [peak for peak in peaks if float(peak["peak_sec"]) < now - 1e-6]
            peak = candidates[-1] if candidates else peaks[-1]
        frame = nearest_frame(self.frames, float(peak["peak_sec"]))
        if frame:
            self.frame_index = self.frames.index(frame)
        self.render_frame()
        self.render_plot()

    def previous_peak(self) -> None:
        self._jump_peak(-1)

    def next_peak(self) -> None:
        self._jump_peak(1)

    def previous_trajectory(self) -> None:
        self.load_trajectory(self.trajectory_index - 1)

    def next_trajectory(self) -> None:
        self.load_trajectory(self.trajectory_index + 1)

    def next_camera(self) -> None:
        available = list(self.trajectory["camera_directories"])
        if not available:
            return
        current = available.index(self.camera_name) if self.camera_name in available else -1
        self.camera_name = available[(current + 1) % len(available)]
        self.load_trajectory(self.trajectory_index)

    def _on_plot_click(self, event: Any) -> None:
        if event.xdata is None or not self.frames:
            return
        frame = nearest_frame(self.frames, float(event.xdata))
        if frame:
            self.frame_index = self.frames.index(frame)
            self.render_frame()
            self.render_plot()

    def _overlay_masks(self, image: Any, frame_path: Path) -> tuple[Any, list[str]]:
        from PIL import Image

        missing: list[str] = []
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        requests = []
        if self.object_mask.get():
            requests.append(
                (
                    "object",
                    self.mask_root / "object" / OBJECT_SELECTION_DIGEST,
                    OBJECT_PROMPT,
                    (0, 255, 80, 105),
                )
            )
        if self.gripper_mask.get():
            requests.append(
                ("gripper", self.mask_root / "gripper", GRIPPER_PROMPT, (255, 40, 190, 105))
            )
        for name, cache_dir, prompt, color in requests:
            mask_path = _mask_cache_path(frame_path, cache_dir, prompt)
            if not mask_path.is_file():
                missing.append(name)
                continue
            mask = Image.open(mask_path).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            color_layer = Image.new("RGBA", image.size, color)
            overlay.alpha_composite(Image.composite(color_layer, Image.new("RGBA", image.size), mask))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), missing

    def render_frame(self) -> None:
        from PIL import Image, ImageTk

        trajectory = self.trajectory
        precision = float(trajectory.get("point_precision_at_k") or 0.0)
        self.title_var.set(
            f"{self.trajectory_index + 1}/{len(self.trajectories)}  "
            f"{trajectory['task_name']}  |  {trajectory['bundle_id']}  |  "
            f"Precision@5={precision:.2f}  |  camera={self.camera_name}"
        )
        if not self.frames:
            self.image_label.configure(image="", text="No cached frames found")
            self.frame_var.set("")
            return
        frame = self.frames[self.frame_index]
        image = Image.open(frame.path).convert("RGB")
        image, missing = self._overlay_masks(image, frame.path)
        image.thumbnail((850, 760), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")
        suffix = f" | mask unavailable: {', '.join(missing)}" if missing else ""
        self.frame_var.set(
            f"frame {self.frame_index + 1}/{len(self.frames)} | "
            f"t={frame.seconds:.3f}s | {frame.path.name}{suffix}"
        )

    def render_plot(self) -> None:
        trajectory = self.trajectory
        self.axis.clear()
        peaks = trajectory["selected_peaks"]
        peak_times = [float(peak["peak_sec"]) for peak in peaks]
        scores = [float(peak.get("rank_score") or 0.0) for peak in peaks]
        self.axis.vlines(peak_times, 0.0, scores, color="#175cd3", linewidth=2.5)
        self.axis.scatter(peak_times, scores, color="#175cd3", zorder=3)
        for peak, x_value, y_value in zip(peaks, peak_times, scores):
            self.axis.annotate(
                f"#{int(peak['rank'])}",
                (x_value, y_value),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=8,
            )
        for index, segment in enumerate(trajectory.get("failed_segments", [])):
            self.axis.axvspan(
                float(segment["start_sec"]),
                float(segment["end_sec"]),
                color="#f04438",
                alpha=0.16,
                label="failure interval" if index == 0 else None,
            )
        if self.frames:
            self.axis.axvline(
                self.frames[self.frame_index].seconds,
                color="#101828",
                linestyle="--",
                linewidth=1.2,
                label="current frame",
            )
        self.axis.set_xlim(0.0, float(trajectory.get("duration_sec") or 1.0))
        self.axis.set_ylim(bottom=0.0)
        self.axis.set_xlabel("Trajectory time (s)")
        self.axis.set_ylabel("0-D persistence of ECOD aggregate peak")
        self.axis.set_title("Selected ECOD peaks (click graph to seek)")
        self.axis.grid(alpha=0.2)
        self.axis.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    frame_cache = args.frame_cache.expanduser().resolve()
    rows = _load_eval_rows(output_dir / "abc130k_eval.jsonl")
    selected = select_rows(rows, args.count, args.selection)
    manifest = build_manifest(selected, frame_cache, output_dir)

    if args.write_manifest:
        destination = args.write_manifest.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {destination}")
    if args.list:
        print_manifest(manifest)
        return

    InteractionViewer(manifest).run()


if __name__ == "__main__":
    main()
