"""Adapter for loading RoboEval-style trajectory folders into TrajectoryBundle."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pathologist.schemas import (
    BundleFormat,
    CameraFrameRef,
    Modality,
    ModalityStream,
    TaskContract,
    TrajectoryBundle,
)


class RoboEvalBundleAdapter:
    """Load a RoboEval/VLM bundle directory into the internal trajectory schema."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
    METADATA_FILENAMES = (
        "metadata.json",
        "bundle_metadata.json",
        "manifest.json",
        "info.json",
    )

    def load(
        self,
        bundle_dir: str | Path,
        task_contract: TaskContract | None = None,
        bundle_id: str | None = None,
    ) -> TrajectoryBundle:
        """Load one trajectory directory."""

        bundle_path = Path(bundle_dir).expanduser().resolve()
        telemetry_path = bundle_path / "telemetry.jsonl"
        if not telemetry_path.exists():
            raise FileNotFoundError(f"Could not find telemetry.jsonl in {bundle_path}")

        records = self._read_jsonl(telemetry_path)
        if not records:
            raise ValueError(f"No telemetry records found in {telemetry_path}")

        timestamps = self._extract_timestamps(records)
        raw_metadata = self._load_metadata(bundle_path)
        raw_metadata["telemetry_record_count"] = len(records)
        if "task_success" in records[-1]:
            raw_metadata["final_task_success"] = records[-1]["task_success"]

        streams = self._build_streams(records, timestamps)
        camera_frames = self._load_camera_frames(bundle_path, timestamps)
        duration_sec = timestamps[-1] if timestamps else None

        return TrajectoryBundle(
            bundle_id=bundle_id or bundle_path.name,
            source_format=BundleFormat.ROBOEVAL_VLM_BUNDLE,
            root_path=bundle_path,
            task_contract=task_contract,
            duration_sec=duration_sec,
            streams=streams,
            camera_frames=camera_frames,
            raw_metadata=raw_metadata,
        )

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
        return records

    def _extract_timestamps(self, records: list[dict[str, Any]]) -> list[float]:
        timestamps: list[float] = []
        for index, record in enumerate(records):
            if isinstance(record.get("timestamp_sec"), (int, float)):
                timestamps.append(float(record["timestamp_sec"]))
            elif isinstance(record.get("sim_time_sec"), (int, float)):
                timestamps.append(float(record["sim_time_sec"]))
            else:
                timestamps.append(float(index))
        return timestamps

    def _build_streams(
        self,
        records: list[dict[str, Any]],
        timestamps: list[float],
    ) -> list[ModalityStream]:
        streams = [
            ModalityStream(
                name="telemetry",
                modality=Modality.TELEMETRY,
                timestamps_sec=timestamps,
                records=records,
            )
        ]
        if self._has_any(records, ("action", "actions")):
            streams.append(
                ModalityStream(
                    name="action",
                    modality=Modality.ACTION,
                    timestamps_sec=timestamps,
                    records=records,
                )
            )
        if self._has_any(records, ("proprioception", "state", "joint_positions")):
            streams.append(
                ModalityStream(
                    name="proprioception",
                    modality=Modality.PROPRIOCEPTION,
                    timestamps_sec=timestamps,
                    records=records,
                )
            )
        if self._has_any(records, ("proprioception_grippers", "gripper", "gripper_state")):
            streams.append(
                ModalityStream(
                    name="gripper",
                    modality=Modality.GRIPPER,
                    timestamps_sec=timestamps,
                    records=records,
                )
            )
        return streams

    def _load_metadata(self, bundle_path: Path) -> dict[str, Any]:
        for filename in self.METADATA_FILENAMES:
            path = bundle_path / filename
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                return value
        return {}

    def _load_camera_frames(
        self,
        bundle_path: Path,
        timestamps: list[float],
    ) -> list[CameraFrameRef]:
        grouped_images: dict[str, list[Path]] = {}
        for image_path in bundle_path.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in self.IMAGE_SUFFIXES:
                continue
            if "pathologist_outputs" in image_path.parts:
                continue
            camera_name = image_path.parent.name
            grouped_images.setdefault(camera_name, []).append(image_path)

        frames: list[CameraFrameRef] = []
        for camera_name, paths in sorted(grouped_images.items()):
            paths.sort(key=self._natural_sort_key)
            for frame_index, image_path in enumerate(paths):
                timestamp = self._timestamp_for_image(
                    image_path=image_path,
                    frame_index=frame_index,
                    total_frames=len(paths),
                    timestamps=timestamps,
                )
                frames.append(
                    CameraFrameRef(
                        camera_name=camera_name,
                        timestamp_sec=timestamp,
                        image_path=image_path,
                        frame_index=frame_index,
                    )
                )
        frames.sort(key=lambda frame: (frame.timestamp_sec, frame.camera_name, frame.frame_index or -1))
        return frames

    def _timestamp_for_image(
        self,
        image_path: Path,
        frame_index: int,
        total_frames: int,
        timestamps: list[float],
    ) -> float:
        stem_numbers = re.findall(r"\d+(?:\.\d+)?", image_path.stem)
        if stem_numbers and any("." in token for token in stem_numbers):
            try:
                return float(stem_numbers[-1])
            except ValueError:
                pass
        if timestamps:
            if total_frames == len(timestamps):
                return timestamps[frame_index]
            mapped_index = round(frame_index * (len(timestamps) - 1) / max(total_frames - 1, 1))
            return timestamps[mapped_index]
        return float(frame_index)

    def _has_any(self, records: list[dict[str, Any]], keys: tuple[str, ...]) -> bool:
        for record in records:
            for key in keys:
                if record.get(key) is not None:
                    return True
        return False

    def _natural_sort_key(self, path: Path) -> list[Any]:
        parts = re.split(r"(\d+)", path.stem)
        key: list[Any] = []
        for part in parts:
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part)
        key.append(path.suffix.lower())
        return key
