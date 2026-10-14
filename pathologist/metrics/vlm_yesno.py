"""Cached Qwen yes/no margin metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pathologist.metrics.base import MetricComputationContext
from pathologist.metrics.helpers import summarize
from pathologist.schemas.common import Modality
from pathologist.schemas.metrics import MetricResult, MetricSample


class CachedVLMYesNoMetric:
    """Load trajectory-time Qwen margins produced by the offline GPU scan."""

    name = "vlm_insert_remove_margin"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = str(name)

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = context.config.get("vlm_yesno")
        cache_root = config.get("cache_root") if isinstance(config, dict) else None
        category = (
            str(config.get("category", "")).strip()
            if isinstance(config, dict)
            else ""
        )
        task_name = str(context.bundle.raw_metadata.get("task_name", "")).strip()
        cache_path = (
            Path(str(cache_root)).expanduser().resolve()
            / task_name
            / f"{context.bundle.bundle_id}.npz"
            if cache_root and task_name
            else None
        )
        base_metadata: dict[str, Any] = {
            "available": False,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "task_name": task_name or None,
            "bundle_id": context.bundle.bundle_id,
        }
        if cache_path is None:
            return self._unavailable(
                base_metadata,
                "vlm_yesno.cache_root and bundle task_name are required",
            )
        if not cache_path.is_file():
            return self._unavailable(base_metadata, "cache file is missing")

        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                centers = np.asarray(cached["center_sec"], dtype=np.float64)
                margins = np.asarray(cached["margin"], dtype=np.float64)
                masses = np.asarray(cached["yes_no_mass"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            return self._unavailable(
                base_metadata,
                f"invalid cache file: {type(exc).__name__}: {exc}",
            )
        if centers.ndim != 1 or margins.ndim != 1 or masses.ndim != 1:
            return self._unavailable(base_metadata, "cache arrays must be one-dimensional")
        if not (len(centers) == len(margins) == len(masses)):
            return self._unavailable(base_metadata, "cache arrays have unequal lengths")
        if len(centers) < 1:
            return self._unavailable(base_metadata, "cache contains no windows")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(margins)):
            return self._unavailable(base_metadata, "cache contains non-finite times or margins")

        sidecar_path = cache_path.with_suffix(".json")
        sidecar: dict[str, Any] = {}
        if sidecar_path.is_file():
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
                sidecar = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                sidecar = {}
        sidecar_provenance = sidecar.get("provenance")
        cached_category = (
            str(sidecar_provenance.get("category", "")).strip()
            if isinstance(sidecar_provenance, dict)
            else ""
        )
        if category and cached_category and category != cached_category:
            return self._unavailable(
                base_metadata,
                (
                    f"cache category mismatch: expected {category!r}, "
                    f"found {cached_category!r}"
                ),
            )
        values = margins.tolist()
        description_target = category.replace("_", " ") if category else "insertion/removal"
        return MetricResult(
            name=self.name,
            description=(
                "Cached Qwen3-VL Yes-minus-No logit margin for "
                f"{description_target}."
            ),
            source_modalities=[Modality.RGB, Modality.LANGUAGE],
            unit="logit_margin",
            samples=[
                MetricSample(timestamp_sec=float(timestamp), value=float(value))
                for timestamp, value in zip(centers, margins)
            ],
            summary=summarize(values),
            thresholds=[],
            events=[],
            display_hint="vlm_yesno_timeseries",
            metadata={
                **base_metadata,
                "available": True,
                "window_count": len(values),
                "yes_no_mass_below_0_5_count": int(np.sum(masses < 0.5)),
                "yes_no_mass_min": float(np.min(masses)),
                "category": category or cached_category or None,
                "sidecar_path": str(sidecar_path) if sidecar_path.is_file() else None,
                "scan_provenance": {
                    key: sidecar.get(key)
                    for key in (
                        "model_id",
                        "prompt",
                        "window_sec",
                        "stride_sec",
                        "resolution",
                        "git_commit",
                    )
                    if key in sidecar
                },
                "cache_provenance": (
                    sidecar_provenance
                    if isinstance(sidecar_provenance, dict)
                    else {}
                ),
            },
        )

    def _unavailable(
        self,
        metadata: dict[str, Any],
        reason: str,
    ) -> MetricResult:
        return MetricResult(
            name=self.name,
            description=(
                "Cached Qwen3-VL Yes-minus-No logit margin."
            ),
            source_modalities=[Modality.RGB, Modality.LANGUAGE],
            unit="logit_margin",
            samples=[],
            thresholds=[],
            events=[],
            display_hint="vlm_yesno_timeseries",
            metadata={
                **metadata,
                "available": False,
                "unavailable_reason": reason,
            },
        )
