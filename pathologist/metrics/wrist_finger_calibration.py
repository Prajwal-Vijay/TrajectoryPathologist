"""Wrist-camera finger masks calibrated from data, per aperture.

A wrist camera is bolted to the gripper, so the fingers' image region depends
only on the aperture: it is the same in every episode, whatever the scene.
Everything else (background, objects, the other arm) changes from episode to
episode. So, across many episodes, the finger pixels are the pixels whose
appearance barely varies at a given aperture.

The calibrator bins frames by measured aperture, keeps at most a few frames per
episode and bin (so one long, static episode cannot dominate), and accumulates
the per-pixel mean and variance of the blurred grey image. For each bin with
enough episodes, the finger mask is the low-standard-deviation region (Otsu
threshold on the std map, clipped to a sane range), restricted to components
that touch the image edge the fingers enter from, with holes filled.

Frames inside hold windows (plus a margin) are excluded: a held object is also
rigid with the camera, and if one task holds the same object in every episode it
would otherwise be mistaken for part of the gripper.

This replaces text-prompted detection ("robot gripper.") for the fingers, which
in wrist views routinely returns the held object, the other arm, or most of the
image, because the box around the fingers also contains whatever they hold.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CALIBRATION_VERSION = 1

__all__ = [
    "FingerCalibration",
    "FingerCalibrator",
    "fill_holes",
    "finger_mask_from_std",
    "keep_edge_components",
]


@dataclass
class FingerCalibration:
    """Finger masks at work resolution, one per aperture bin."""

    camera: str
    centers: Any  # (n_bins,) float, aperture bin centres
    masks: Any  # (n_bins, H, W) bool
    episode_counts: list[int]
    std_thresholds: list[float]
    bin_width: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def work_shape(self) -> tuple[int, int]:
        return int(self.masks.shape[1]), int(self.masks.shape[2])

    def mask_for(self, aperture: float) -> tuple[Any, float, float]:
        """Mask of the bin nearest ``aperture``: (mask, bin centre, |aperture - centre|)."""

        import numpy as np

        distances = np.abs(np.asarray(self.centers, dtype=float) - float(aperture))
        index = int(np.argmin(distances))
        return self.masks[index].astype(bool), float(self.centers[index]), float(distances[index])

    def save(self, path: str | Path) -> Path:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": CALIBRATION_VERSION,
            "camera": self.camera,
            "episode_counts": [int(c) for c in self.episode_counts],
            "std_thresholds": [float(v) for v in self.std_thresholds],
            "bin_width": float(self.bin_width),
            "metadata": self.metadata,
        }
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                centers=np.asarray(self.centers, dtype=float),
                masks=np.asarray(self.masks, dtype=np.uint8),
                meta=np.asarray(json.dumps(meta, default=str)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FingerCalibration:
        import numpy as np

        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if int(meta.get("version", 0)) != CALIBRATION_VERSION:
                raise ValueError(f"finger calibration {path} has version {meta.get('version')}")
            return cls(
                camera=str(meta["camera"]),
                centers=np.asarray(data["centers"], dtype=float),
                masks=np.asarray(data["masks"]).astype(bool),
                episode_counts=list(meta["episode_counts"]),
                std_thresholds=list(meta["std_thresholds"]),
                bin_width=float(meta["bin_width"]),
                metadata=dict(meta.get("metadata") or {}),
            )

    def digest(self) -> str:
        import numpy as np

        hasher = hashlib.sha256()
        hasher.update(np.asarray(self.centers, dtype=float).tobytes())
        hasher.update(np.packbits(np.asarray(self.masks, dtype=bool)).tobytes())
        return hasher.hexdigest()[:16]


@dataclass
class _BinStats:
    total: Any
    total_sq: Any
    count: int = 0
    episodes: int = 0


class FingerCalibrator:
    """Accumulates per-aperture pixel statistics over episodes of one wrist camera."""

    def __init__(
        self,
        *,
        work_width: int = 320,
        bin_width: float = 0.1,
        frames_per_episode_bin: int = 6,
        blur_sigma: float = 1.0,
        entry_edge: str = "bottom",
        edge_fraction: float = 0.05,
    ) -> None:
        self.work_width = int(work_width)
        self.bin_width = float(bin_width)
        self.frames_per_episode_bin = int(frames_per_episode_bin)
        self.blur_sigma = float(blur_sigma)
        self.entry_edge = entry_edge
        self.edge_fraction = float(edge_fraction)
        self.work_shape: tuple[int, int] | None = None
        self._bins: dict[int, _BinStats] = {}
        self.n_episodes = 0

    def _bin(self, aperture: float) -> int:
        return round(float(aperture) / self.bin_width)

    def _load(self, path: Path) -> Any | None:
        import cv2
        import numpy as np

        image = cv2.imread(str(path))
        if image is None:
            return None
        full_h, full_w = image.shape[:2]
        if self.work_shape is None:
            work_w = min(self.work_width, full_w)
            self.work_shape = (max(2, round(full_h * work_w / full_w)), work_w)
        height, width = self.work_shape
        gray = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY).astype(np.float64)
        if self.blur_sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.blur_sigma)
        return gray

    def add_episode(
        self,
        frame_paths: list[Path],
        frame_times: list[float],
        aperture_times: Any,
        apertures: Any,
        exclude: Iterable[tuple[float, float]] = (),
    ) -> int:
        """Add one episode's frames; returns how many frames were used."""

        import numpy as np

        if not frame_paths:
            return 0
        times = np.asarray(frame_times, dtype=float)
        values = np.interp(times, np.asarray(aperture_times, dtype=float), np.asarray(apertures, dtype=float))
        keep = np.ones(len(times), dtype=bool)
        for start, end in exclude:
            keep &= ~((times >= start) & (times < end))
        by_bin: dict[int, list[int]] = {}
        for index in np.nonzero(keep)[0]:
            by_bin.setdefault(self._bin(values[index]), []).append(int(index))
        used = 0
        for bin_index, indices in by_bin.items():
            picks = np.unique(
                np.linspace(0, len(indices) - 1, num=min(self.frames_per_episode_bin, len(indices)))
                .round()
                .astype(int)
            )
            added = 0
            for pick in picks:
                gray = self._load(Path(frame_paths[indices[pick]]))
                if gray is None:
                    continue
                stats = self._bins.get(bin_index)
                if stats is None:
                    stats = _BinStats(np.zeros_like(gray), np.zeros_like(gray))
                    self._bins[bin_index] = stats
                stats.total += gray
                stats.total_sq += gray * gray
                stats.count += 1
                added += 1
            if added:
                self._bins[bin_index].episodes += 1
                used += added
        if used:
            self.n_episodes += 1
        return used

    def bin_summary(self) -> dict[float, dict[str, int]]:
        return {
            round(index * self.bin_width, 6): {"frames": stats.count, "episodes": stats.episodes}
            for index, stats in sorted(self._bins.items())
        }

    def finish(
        self,
        camera: str,
        *,
        min_episodes: int = 8,
        std_bounds: tuple[float, float] = (6.0, 30.0),
        max_area_frac: float = 0.7,
        min_area_frac: float = 0.005,
    ) -> FingerCalibration:
        import numpy as np

        centers, masks, counts, thresholds, skipped = [], [], [], [], {}
        for index, stats in sorted(self._bins.items()):
            center = round(index * self.bin_width, 6)
            if stats.episodes < min_episodes:
                skipped[center] = f"only {stats.episodes} episodes"
                continue
            mean = stats.total / stats.count
            std = np.sqrt(np.maximum(stats.total_sq / stats.count - mean * mean, 0.0))
            mask, threshold = finger_mask_from_std(
                std, self.entry_edge, self.edge_fraction, std_bounds
            )
            area = float(mask.mean())
            if not (min_area_frac <= area <= max_area_frac):
                skipped[center] = f"mask area {area:.3f} outside [{min_area_frac}, {max_area_frac}]"
                continue
            centers.append(center)
            masks.append(mask)
            counts.append(stats.episodes)
            thresholds.append(threshold)
        if not masks:
            raise ValueError(
                f"no aperture bin produced a finger mask for {camera!r}: "
                f"{skipped or 'no frames were added'}"
            )
        return FingerCalibration(
            camera=camera,
            centers=np.asarray(centers, dtype=float),
            masks=np.stack(masks).astype(bool),
            episode_counts=counts,
            std_thresholds=thresholds,
            bin_width=self.bin_width,
            metadata={
                "n_episodes": self.n_episodes,
                "skipped_bins": skipped,
                "entry_edge": self.entry_edge,
                "work_shape": list(self.work_shape or ()),
            },
        )


def finger_mask_from_std(
    std: Any,
    entry_edge: str,
    edge_fraction: float,
    std_bounds: tuple[float, float] = (6.0, 30.0),
) -> tuple[Any, float]:
    """Low-variance, edge-connected, hole-filled region of a per-pixel std map."""

    import cv2
    import numpy as np

    std = np.asarray(std, dtype=np.float64)
    std8 = np.clip(std, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(std8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu, std_bounds[0], std_bounds[1]))
    mask = (std <= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = keep_edge_components(mask.astype(bool), entry_edge, edge_fraction)
    return fill_holes(mask), threshold


def keep_edge_components(mask: Any, entry_edge: str, edge_fraction: float) -> Any:
    """Connected components of ``mask`` that reach the strip along ``entry_edge``."""

    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    strip = np.zeros_like(mask)
    depth_y = max(1, round(edge_fraction * height))
    depth_x = max(1, round(edge_fraction * width))
    if entry_edge == "top":
        strip[:depth_y, :] = True
    elif entry_edge == "left":
        strip[:, :depth_x] = True
    elif entry_edge == "right":
        strip[:, width - depth_x :] = True
    else:
        strip[height - depth_y :, :] = True
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    out = np.zeros_like(mask)
    for label in range(1, count):
        component = labels == label
        if (component & strip).any():
            out |= component
    return out


def fill_holes(mask: Any) -> Any:
    """Fill background regions not connected to the image border."""

    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    # Pad with background so every border-connected background pixel is reached
    # from one corner; what the flood does not reach is a hole.
    padded = np.pad((~mask).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(padded, None, (0, 0), 0)
    return mask | padded[1:-1, 1:-1].astype(bool)
