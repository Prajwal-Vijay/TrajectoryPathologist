"""Training-free audio event metrics."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pathologist.metrics.base import MetricComputationContext, metric_config
from pathologist.metrics.helpers import summarize
from pathologist.schemas.common import Modality
from pathologist.schemas.metrics import MetricResult, MetricSample


class AudioSpectralFluxMetric:
    """Positive log-mel spectral flux across available microphones."""

    name = "audio_spectral_flux"

    def compute(self, context: MetricComputationContext) -> MetricResult:
        config = metric_config(context, self.name)
        sample_rate_hz = max(1000, int(config.get("sample_rate_hz", 16000)))
        window_ms = max(1.0, float(config.get("window_ms", 25.0)))
        hop_ms = max(1.0, float(config.get("hop_ms", 10.0)))
        mel_bins = max(2, int(config.get("mel_bins", 64)))
        min_frequency_hz = max(0.0, float(config.get("min_frequency_hz", 50.0)))
        max_frequency_hz = min(
            sample_rate_hz / 2.0,
            float(config.get("max_frequency_hz", sample_rate_hz / 2.0)),
        )
        audio_streams = sorted(
            (
                stream
                for stream in context.bundle.streams
                if stream.modality == Modality.AUDIO and stream.data_path is not None
            ),
            key=lambda stream: stream.name,
        )
        if not audio_streams:
            return _unavailable_result(
                reason="requires at least one decodable audio stream",
                sample_rate_hz=sample_rate_hz,
                window_ms=window_ms,
                hop_ms=hop_ms,
                mel_bins=mel_bins,
            )

        component_results: dict[str, tuple[list[float], list[float]]] = {}
        decode_errors: dict[str, str] = {}
        for stream in audio_streams:
            microphone_name = str(
                stream.metadata.get("microphone_name", stream.name)
            )
            try:
                waveform = _audio_segment(
                    Path(stream.data_path),
                    sample_rate_hz=sample_rate_hz,
                    start_sec=float(
                        stream.metadata.get("audio_source_start_sec", 0.0)
                    ),
                    duration_sec=float(
                        stream.metadata.get(
                            "audio_segment_duration_sec",
                            context.bundle.duration_sec or 0.0,
                        )
                    ),
                )
                timestamps, flux = _log_mel_spectral_flux(
                    waveform,
                    sample_rate_hz=sample_rate_hz,
                    window_ms=window_ms,
                    hop_ms=hop_ms,
                    mel_bins=mel_bins,
                    min_frequency_hz=min_frequency_hz,
                    max_frequency_hz=max_frequency_hz,
                )
            except Exception as exc:
                decode_errors[microphone_name] = f"{type(exc).__name__}: {exc}"
                continue
            if flux:
                component_results[microphone_name] = (timestamps, flux)

        if not component_results:
            reason = "audio streams could not produce spectral-flux samples"
            if decode_errors:
                reason = "audio decoding or spectral-flux computation failed"
            result = _unavailable_result(
                reason=reason,
                sample_rate_hz=sample_rate_hz,
                window_ms=window_ms,
                hop_ms=hop_ms,
                mel_bins=mel_bins,
            )
            result.metadata["decode_errors"] = decode_errors
            return result

        common_count = min(len(values) for _, values in component_results.values())
        reference_name = max(
            component_results,
            key=lambda name: len(component_results[name][1]),
        )
        reference_timestamps = component_results[reference_name][0][:common_count]
        component_values = {
            name: values[:common_count]
            for name, (_, values) in component_results.items()
        }
        aggregate_values = [
            max(values[index] for values in component_values.values())
            for index in range(common_count)
        ]

        return MetricResult(
            name=self.name,
            description=(
                "Positive log-mel spectral flux; microphones are treated as "
                "components and combined by a per-tail maximum after ECOD scoring."
            ),
            source_modalities=[Modality.AUDIO],
            unit="log_mel_positive_spectral_flux",
            samples=[
                MetricSample(timestamp_sec=timestamp, value=value)
                for timestamp, value in zip(reference_timestamps, aggregate_values)
            ],
            summary=summarize(aggregate_values),
            thresholds=[],
            events=[],
            display_hint="audio_spectral_flux_timeseries",
            metadata={
                "available": True,
                "microphone_names": list(component_values),
                "microphone_count": len(component_values),
                "sample_rate_hz": sample_rate_hz,
                "window_ms": window_ms,
                "hop_ms": hop_ms,
                "mel_bins": mel_bins,
                "min_frequency_hz": min_frequency_hz,
                "max_frequency_hz": max_frequency_hz,
                "microphone_reduction": "per_tail_max_after_ecod",
                "surprise_tail_interpretation": "right",
                "ecod_component_values": component_values,
                "decode_errors": decode_errors,
            },
        )


def _audio_segment(
    audio_path: Path,
    *,
    sample_rate_hz: int,
    start_sec: float,
    duration_sec: float,
) -> Any:
    waveform = _decode_audio_file(str(audio_path.resolve()), sample_rate_hz)
    start_index = max(0, round(max(0.0, start_sec) * sample_rate_hz))
    if duration_sec <= 0.0:
        return waveform[start_index:]
    end_index = min(
        len(waveform),
        start_index + round(duration_sec * sample_rate_hz),
    )
    return waveform[start_index:end_index]


@lru_cache(maxsize=3)
def _decode_audio_file(audio_path: str, sample_rate_hz: int) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            audio_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate_hz),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"ffmpeg exited with {completed.returncode}")
    waveform = np.frombuffer(completed.stdout, dtype="<f4")
    if waveform.size == 0:
        raise ValueError("decoded audio stream is empty")
    return waveform


def _log_mel_spectral_flux(
    waveform: Any,
    *,
    sample_rate_hz: int,
    window_ms: float,
    hop_ms: float,
    mel_bins: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> tuple[list[float], list[float]]:
    import numpy as np  # type: ignore[import-not-found]

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window_size = max(2, round(sample_rate_hz * window_ms / 1000.0))
    hop_size = max(1, round(sample_rate_hz * hop_ms / 1000.0))
    if values.size < window_size + hop_size:
        return [], []
    frame_count = 1 + (values.size - window_size) // hop_size
    frame_starts = np.arange(frame_count) * hop_size
    frames = np.lib.stride_tricks.sliding_window_view(values, window_size)[
        frame_starts
    ]
    window = np.hanning(window_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2
    mel_filters = _mel_filterbank(
        sample_rate_hz=sample_rate_hz,
        fft_size=window_size,
        mel_bins=mel_bins,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
    )
    log_mel = np.log1p(power @ mel_filters.T)
    positive_differences = np.maximum(log_mel[1:] - log_mel[:-1], 0.0)
    flux = positive_differences.sum(axis=1)
    timestamps = (
        (frame_starts[1:] + window_size / 2.0) / float(sample_rate_hz)
    )
    return timestamps.astype(float).tolist(), flux.astype(float).tolist()


def _mel_filterbank(
    *,
    sample_rate_hz: int,
    fft_size: int,
    mel_bins: int,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> Any:
    import numpy as np  # type: ignore[import-not-found]

    def hz_to_mel(frequency: Any) -> Any:
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: Any) -> Any:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    frequencies = np.linspace(0.0, sample_rate_hz / 2.0, fft_size // 2 + 1)
    mel_edges = np.linspace(
        hz_to_mel(min_frequency_hz),
        hz_to_mel(max_frequency_hz),
        mel_bins + 2,
    )
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((mel_bins, len(frequencies)), dtype=np.float32)
    for index in range(mel_bins):
        left, center, right = hz_edges[index:index + 3]
        if center > left:
            filters[index] = np.maximum(
                filters[index],
                (frequencies - left) / (center - left),
            )
        if right > center:
            filters[index] = np.minimum(
                filters[index],
                (right - frequencies) / (right - center),
            )
        filters[index] = np.maximum(filters[index], 0.0)
        total = float(filters[index].sum())
        if total > 0.0:
            filters[index] /= total
    return filters


def _unavailable_result(
    *,
    reason: str,
    sample_rate_hz: int,
    window_ms: float,
    hop_ms: float,
    mel_bins: int,
) -> MetricResult:
    return MetricResult(
        name=AudioSpectralFluxMetric.name,
        description="Positive log-mel spectral flux across available microphones.",
        source_modalities=[Modality.AUDIO],
        samples=[],
        metadata={
            "available": False,
            "unavailable_reason": reason,
            "sample_rate_hz": sample_rate_hz,
            "window_ms": window_ms,
            "hop_ms": hop_ms,
            "mel_bins": mel_bins,
        },
    )
