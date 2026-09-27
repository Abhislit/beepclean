from __future__ import annotations

import warnings
from pathlib import Path

import av
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from app import pipeline
from app.config import Settings
from app.intervals import Interval
from app.video import probe
from tests.make_media import build_clip

SIZE = (640, 360)


DIRTY = [
    ("this", 0.2, 0.5, 0.99),
    ("class", 0.5, 0.8, 0.98),
    ("is", 0.8, 0.9, 0.97),
    ("fucking", 1.0, 1.6, 0.95),
    ("great", 1.6, 1.9, 0.96),
]

CLEAN = [
    ("this", 0.2, 0.5, 0.99),
    ("class", 0.5, 0.8, 0.98),
    ("is", 0.8, 0.9, 0.97),
    ("lovely", 1.0, 1.6, 0.95),
    ("great", 1.6, 1.9, 0.96),
]


class StubTranscriber:
    """Stands in for faster-whisper so tests never download a model."""

    spoken = DIRTY
    calls: list[Path] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def words(self, path, progress=None, cancelled=None):
        StubTranscriber.calls.append(path)
        if progress:
            progress(0.0, "Listening for words")
        return list(self.spoken)

    def release(self) -> None:
        return None


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    StubTranscriber.calls = []
    StubTranscriber.spoken = DIRTY
    monkeypatch.setattr(pipeline, "Transcriber", StubTranscriber)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pipe") / "clip.mp4"
    build_clip(
        path,
        duration=8.0,
        size=SIZE,
        subtitles=[(3.0, 6.0, "you are such a bitch")],
        tones=[(1.0, 1.6)],
    )
    return path


def base_settings(**overrides) -> Settings:
    settings = Settings()
    settings.model = "tiny.en"
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def read_audio_peak(path: Path, at: float, rate: int = 48000) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp", layout=stream.layout.name, rate=stream.rate
        )
        target = int(at * rate)
        seen = 0
        while True:
            frame = next(container.decode(audio=0), None)
            if frame is None:
                return 0.0
            for converted in resampler.resample(frame):
                block = converted.to_ndarray()[0]
                count = len(block)
                if seen <= target < seen + count:
                    return float(np.abs(block[target - seen : target - seen + 2048]).max())
                seen += count


def test_spoken_hits_map_tokens_to_timestamps() -> None:
    from app.profanity import ProfanityFilter

    words = ProfanityFilter.load()
    settings = Settings()
    tokens = [("this", 0.2, 0.5, 0.9), ("class", 0.5, 0.8, 0.9), ("fucking", 1.0, 1.6, 0.9)]
    hits = pipeline._spoken_hits(tokens, words, settings)
    assert [h.word for h in hits] == ["fucking"]
    assert hits[0].start == pytest.approx(1.0, abs=0.01)
    assert hits[0].end == pytest.approx(1.6, abs=0.01)


def test_min_confidence_filters_weak_words() -> None:
    from app.profanity import ProfanityFilter

    words = ProfanityFilter.load()
    settings = Settings()
    settings.min_confidence = 0.5
    hits = pipeline._spoken_hits([("fuck", 1.0, 1.4, 0.2)], words, settings)
    assert hits == []


def test_full_run_beeps_and_blurs(clip: Path, tmp_path: Path) -> None:
    settings = base_settings(ocr_fps=3.0)
    destination = tmp_path / "out.mp4"
    report = pipeline.run(clip, destination, settings)

    assert destination.exists()
    assert report.beep_count >= 1
    assert report.blur_count >= 1
    assert report.reencoded_audio and report.reencoded_video
    assert any(w["word"] == "fucking" for w in report.words)
    assert any("bitch" in h["text"] for h in report.text_hits)

    info = probe(destination)
    assert info["has_video"] and info["has_audio"]
    assert info["duration"] == pytest.approx(8.0, abs=0.3)
    assert info["width"] == SIZE[0]


def test_beep_actually_present_in_output(clip: Path, tmp_path: Path) -> None:
    settings = base_settings(ocr_enabled=False)
    destination = tmp_path / "beep.mp4"
    pipeline.run(clip, destination, settings)
    assert read_audio_peak(destination, 1.25) > 0.4
    assert read_audio_peak(destination, 0.6) < 0.4


def test_clean_video_is_copied_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "clean.mp4"
    build_clip(path, duration=4.0, size=SIZE, subtitles=[(1.0, 3.0, "a lovely class")])
    destination = tmp_path / "copy.mp4"
    StubTranscriber.spoken = CLEAN
    report = pipeline.run(path, destination, base_settings())
    assert report.beep_count == 0 and report.blur_count == 0
    assert not report.reencoded_audio and not report.reencoded_video
    assert any("copied unchanged" in note for note in report.notes)
    assert destination.exists()


def test_report_only_makes_no_edits(clip: Path, tmp_path: Path) -> None:
    destination = tmp_path / "report.mp4"
    report = pipeline.run(clip, destination, base_settings(report_only=True))
    assert report.beep_count >= 1
    assert not report.reencoded_audio and not report.reencoded_video
    assert any("Report-only" in note for note in report.notes)


def test_min_confidence_end_to_end(clip: Path, tmp_path: Path) -> None:
    settings = base_settings(ocr_enabled=False, min_confidence=0.99)
    destination = tmp_path / "strict.mp4"
    report = pipeline.run(clip, destination, settings)
    assert report.beep_count == 0


def test_video_without_audio_uses_ocr_only(tmp_path: Path) -> None:
    path = tmp_path / "silent.mp4"
    build_clip(
        path,
        duration=6.0,
        size=SIZE,
        subtitles=[(1.0, 4.0, "what the fuck")],
        with_audio=False,
    )
    destination = tmp_path / "silent_out.mp4"
    report = pipeline.run(path, destination, base_settings())
    assert report.has_audio is False
    assert report.beep_count == 0
    assert report.blur_count >= 1
    assert any("No audio track" in note for note in report.notes)


def test_cancellation_raises(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    build_clip(path, duration=3.0, size=SIZE)
    with pytest.raises(InterruptedError):
        pipeline.run(path, tmp_path / "x.mp4", base_settings(), cancelled=lambda: True)


def test_missing_video_track_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "audio.m4a"
    build_clip(path, duration=2.0, with_audio=True, with_video=False)
    with pytest.raises(ValueError, match="no video track"):
        pipeline.run(path, tmp_path / "y.mp4", base_settings())
