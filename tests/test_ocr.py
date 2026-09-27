from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from app.config import Settings
from app.ocr import OcrScanner
from app.profanity import ProfanityFilter
from tests.make_media import build_clip


@pytest.fixture(scope="module")
def words() -> ProfanityFilter:
    return ProfanityFilter.load()


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ocr") / "clip.mp4"
    build_clip(
        path,
        duration=8.0,
        subtitles=[
            (2.0, 5.0, "what the fuck"),
            (6.0, 7.0, "this class is great"),
        ],
    )
    return path


def test_finds_subtitle_and_ignores_benign(clip: Path, words: ProfanityFilter) -> None:
    settings = Settings()
    scanner = OcrScanner(settings, words)
    try:
        hits = scanner.scan(clip, 8.0)
    finally:
        scanner.release()
    assert len(hits) == 1, f"expected exactly one hit, got {[h.text for h in hits]}"
    hit = hits[0]
    assert "fuck" in hit.text
    assert hit.confidence > 0.6
    step = 1.0 / Settings().ocr_fps
    assert 2.0 - step <= hit.start <= 2.0 + 0.1
    assert hit.end >= 5.0 - 2 * step
    assert hit.end <= 5.0 + 0.1
    x0, y0, x1, y1 = hit.box
    assert 0.0 <= x0 < x1 <= 1.0
    assert y0 > 0.5, "subtitle box should sit in the lower half"
    assert y1 <= 1.0


def test_scan_stops_when_cancelled(clip: Path, words: ProfanityFilter) -> None:
    settings = Settings()
    scanner = OcrScanner(settings, words)
    try:
        assert scanner.scan(clip, 8.0, cancelled=lambda: True) == []
    finally:
        scanner.release()


def test_scan_rejects_non_positive_duration(tmp_path: Path, words: ProfanityFilter) -> None:
    path = tmp_path / "tiny.mp4"
    build_clip(path, duration=1.0)
    scanner = OcrScanner(Settings(), words)
    try:
        assert scanner.scan(path, 0.0) == []
    finally:
        scanner.release()
