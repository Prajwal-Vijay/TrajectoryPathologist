"""Tests for RoboFAC pipeline download failure handling."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from pathologist.run_robofac_qa_pipeline import (
    RemoteFileUnavailableError,
    _download_if_missing,
)


def test_download_if_missing_removes_partial_file_and_raises_remote_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "missing.mp4"

    def fail_download(_url: str, path: Path) -> None:
        Path(path).write_text("partial", encoding="utf-8")
        raise urllib.error.HTTPError(
            url="https://example.com/missing.mp4",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlretrieve", fail_download)

    with pytest.raises(RemoteFileUnavailableError, match="HTTP 404"):
        _download_if_missing(
            url="https://example.com/missing.mp4",
            destination=destination,
            label="video",
        )

    assert not destination.exists()
