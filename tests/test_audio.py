from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np
import pytest

from app.audio import build_plan, make_beep, splice_audio
from app.config import Settings
from app.intervals import Interval, merge
from tests.make_media import build_clip


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp", layout=stream.layout.name, rate=stream.rate
        )
        chunks = []
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray()[0])
    return (np.concatenate(chunks) if chunks else np.zeros(0)), stream.rate


def dominant_hz(signal: np.ndarray, rate: int) -> float:
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / rate)
    return float(freqs[int(np.argmax(spectrum))])


def segment(x: np.ndarray, rate: int, start: float, length: float = 0.12) -> np.ndarray:
    return x[int(start * rate) : int(start * rate) + int(length * rate)]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "clip.mp4"
    build_clip(path, duration=6.0, tones=[(2.0, 2.5)])
    return path


def test_beep_shape_and_amplitude() -> None:
    settings = Settings()
    wave = make_beep(48000, 4800, settings)
    assert len(wave) == 4800
    assert abs(float(np.abs(wave).max()) - settings.beep_amplitude) < 0.01
    assert abs(float(wave[0])) < 0.01
    assert abs(float(wave[-1])) < 0.01
    assert dominant_hz(wave, 48000) == pytest.approx(settings.beep_freq, rel=0.02)


def test_plan_respects_min_and_max_duration() -> None:
    settings = Settings()
    settings.beep_min_ms = 120
    settings.beep_max_ms = 600
    settings.edge_pad_ms = 0
    plan = build_plan([Interval(1.0, 1.01, "x")], 48000, settings)
    assert plan[0][1] - plan[0][0] == pytest.approx(5760, rel=0.01)
    plan = build_plan([Interval(1.0, 3.0, "x")], 48000, settings)
    assert plan[0][1] - plan[0][0] == pytest.approx(28800, rel=0.01)


def test_merge_joins_overlapping_intervals() -> None:
    merged = merge([Interval(0.0, 1.0, "a"), Interval(0.9, 2.0, "b"), Interval(5.0, 6.0, "c")])
    assert len(merged) == 2
    assert merged[0].start == 0.0 and merged[0].end == 2.0
    assert merged[1].start == 5.0


def test_splice_replaces_tone_with_beep_and_keeps_length(clip: Path, tmp_path: Path) -> None:
    settings = Settings()
    settings.edge_pad_ms = 0
    destination = tmp_path / "out.m4a"
    assert splice_audio(clip, destination, [Interval(2.0, 2.5, "fuck")], settings) is True

    original, rate = read_audio(clip)
    patched, out_rate = read_audio(destination)
    assert out_rate == rate
    assert abs(len(patched) / out_rate - len(original) / rate) < 0.05

    assert dominant_hz(segment(patched, out_rate, 0.5), out_rate) == pytest.approx(330, rel=0.05)
    during = segment(patched, out_rate, 2.2)
    assert dominant_hz(during, out_rate) == pytest.approx(settings.beep_freq, rel=0.05)
    assert float(np.abs(during).max()) == pytest.approx(settings.beep_amplitude, rel=0.1)
    assert dominant_hz(segment(patched, out_rate, 4.5), out_rate) == pytest.approx(330, rel=0.05)


def test_splice_leaves_audio_untouched_outside_intervals(clip: Path, tmp_path: Path) -> None:
    settings = Settings()
    destination = tmp_path / "quiet.m4a"
    splice_audio(clip, destination, [Interval(2.0, 2.5, "fuck")], settings)
    original, rate = read_audio(clip)
    patched, _ = read_audio(destination)
    before_a = segment(original, rate, 0.2, 0.4)
    before_b = segment(patched, rate, 0.2, 0.4)
    assert np.corrcoef(before_a, before_b)[0, 1] > 0.99


def test_splice_has_no_clicks(clip: Path, tmp_path: Path) -> None:
    settings = Settings()
    destination = tmp_path / "clean.m4a"
    splice_audio(clip, destination, [Interval(2.0, 2.5, "fuck")], settings)
    patched, rate = read_audio(destination)
    assert float(np.abs(np.diff(patched)).max()) < 0.25


def test_splice_without_audio_track_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "silent.mp4"
    build_clip(path, duration=2.0, with_audio=False)
    assert splice_audio(path, tmp_path / "x.m4a", [Interval(0, 1, "fuck")], Settings()) is False
