"""Tests for training-free audio spectral flux."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathologist.metrics.audio import (
    AudioSpectralFluxMetric,
    _log_mel_spectral_flux,
)
from pathologist.metrics.base import MetricComputationContext
from pathologist.schemas import BundleFormat, Modality, ModalityStream, TrajectoryBundle


np = pytest.importorskip("numpy")


def test_log_mel_spectral_flux_responds_to_audio_onset() -> None:
    sample_rate_hz = 16000
    waveform = np.zeros(sample_rate_hz, dtype=np.float32)
    onset_index = sample_rate_hz // 2
    times = np.arange(sample_rate_hz - onset_index) / sample_rate_hz
    waveform[onset_index:] = np.sin(2.0 * np.pi * 1000.0 * times)

    timestamps, flux = _log_mel_spectral_flux(
        waveform,
        sample_rate_hz=sample_rate_hz,
        window_ms=25.0,
        hop_ms=10.0,
        mel_bins=32,
        min_frequency_hz=50.0,
        max_frequency_hz=8000.0,
    )

    peak_timestamp = timestamps[flux.index(max(flux))]
    assert peak_timestamp == pytest.approx(0.5, abs=0.03)
    assert max(flux) > 0.0


def test_audio_metric_keeps_microphones_as_ecod_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_rate_hz = 1000
    waveforms = {}
    for index, microphone_name in enumerate(("hama1", "hama2", "hand")):
        path = (tmp_path / f"{microphone_name}.mp3").resolve()
        waveform = np.zeros(sample_rate_hz, dtype=np.float32)
        waveform[200 + index * 200:] = np.sin(
            2.0 * np.pi * (80.0 + index * 20.0)
            * np.arange(sample_rate_hz - 200 - index * 200)
            / sample_rate_hz
        )
        waveforms[str(path)] = waveform

    monkeypatch.setattr(
        "pathologist.metrics.audio._decode_audio_file",
        lambda path, _: waveforms[path],
    )
    streams = [
        ModalityStream(
            name=f"{microphone_name}_audio",
            modality=Modality.AUDIO,
            timestamps_sec=[0.0, 1.0],
            data_path=tmp_path / f"{microphone_name}.mp3",
            metadata={
                "microphone_name": microphone_name,
                "audio_source_start_sec": 0.0,
                "audio_segment_duration_sec": 1.0,
            },
        )
        for microphone_name in ("hama1", "hama2", "hand")
    ]
    bundle = TrajectoryBundle(
        bundle_id="audio-demo",
        source_format=BundleFormat.CUSTOM,
        duration_sec=1.0,
        streams=streams,
    )

    result = AudioSpectralFluxMetric().compute(
        MetricComputationContext(
            bundle=bundle,
            config={
                "metrics": {
                    "audio_spectral_flux": {
                        "sample_rate_hz": sample_rate_hz,
                        "window_ms": 10.0,
                        "hop_ms": 5.0,
                        "mel_bins": 8,
                        "min_frequency_hz": 20.0,
                        "max_frequency_hz": 500.0,
                    }
                }
            },
        )
    )

    components = result.metadata["ecod_component_values"]
    assert set(components) == {"hama1", "hama2", "hand"}
    assert result.metadata["microphone_reduction"] == "per_tail_max_after_ecod"
    assert [sample.value for sample in result.samples] == pytest.approx(
        [
            max(component[index] for component in components.values())
            for index in range(len(result.samples))
        ]
    )
