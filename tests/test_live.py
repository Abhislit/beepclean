from __future__ import annotations

import warnings
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

warnings.filterwarnings("ignore")

from app import live as live_module
from app.config import Settings
from app.intervals import Interval
from app.live import LiveFilter, LiveOptions
from app.profanity import ProfanityFilter
from tests.make_media import build_clip, subtitle_box

SIZE = (640, 360)
PHRASE = "what the fuck"
CLEAN = "a lovely class"


def read_text(crop: Image.Image) -> str:
    from rapidocr import RapidOCR

    engine = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 640})
    result = engine(np.asarray(crop.convert("RGB")))
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return ""
    return " ".join(result.txts)


def crop_at(path: Path, box, at: float, size=SIZE) -> Image.Image:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(
            int(at / av.time_base), stream=stream, backward=True, any_frame=False
        )
        best = None
        for frame in container.decode(video=0):
            stamp = float(frame.pts * frame.time_base)
            if best is None or abs(stamp - at) < abs(best[0] - at):
                best = (stamp, frame)
            if stamp > at + 0.5:
                break
    assert best is not None
    image = best[1].to_image()
    width, height = image.size
    return image.crop(
        (
            int(box[0] * width),
            int(box[1] * height),
            int(box[2] * width),
            int(box[3] * height),
        )
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("live") / "clip.mp4"
    build_clip(
        path,
        duration=12.0,
        size=SIZE,
        subtitles=[(2.0, 9.0, PHRASE), (10.0, 11.0, CLEAN)],
        tones=[(1.0, 2.0)],
    )
    return path


@pytest.fixture
def no_speech():
    return LiveOptions(pace=False, speech_enabled=False, visual_enabled=True)


def test_live_run_preserves_duration(clip: Path, tmp_path: Path, no_speech) -> None:
    out = tmp_path / "live.mp4"
    report = LiveFilter(Settings(), no_speech).run(str(clip), str(out))
    assert out.exists()
    assert report.windows >= 2
    assert report.media_seconds == pytest.approx(12.0, abs=0.5)
    with av.open(str(out)) as container:
        frames = sum(1 for _ in container.decode(video=0))
        assert container.streams.audio, "audio stream must survive"
    assert frames > 0


def test_live_blurs_profanity_and_leaves_clean_text(clip: Path, tmp_path: Path, no_speech) -> None:
    out = tmp_path / "blur.mp4"
    LiveFilter(Settings(), no_speech).run(str(clip), str(out))
    box = subtitle_box(PHRASE, *SIZE)
    assert read_text(crop_at(clip, box, 4.0)).lower().find("fuck") >= 0
    assert read_text(crop_at(out, box, 4.0)) == "", "profane caption should be unreadable"
    clean_box = subtitle_box(CLEAN, *SIZE)
    assert read_text(crop_at(out, clean_box, 10.4)).lower().find("class") >= 0


def test_live_finds_a_caption_spanning_a_window_boundary(
    clip: Path, tmp_path: Path, no_speech
) -> None:
    """A short caption must not be lost just because it straddles two windows."""
    options = LiveOptions(pace=False, speech_enabled=False, visual_enabled=True, window=2.0)
    out = tmp_path / "boundary.mp4"
    report = LiveFilter(Settings(), options).run(str(clip), str(out))
    assert report.blurs, "a caption crossing a window edge should still be found"


def test_live_report_lists_hits(clip: Path, tmp_path: Path, no_speech) -> None:
    out = tmp_path / "report.mp4"
    report = LiveFilter(Settings(), no_speech).run(str(clip), str(out))
    assert report.blurs
    hit = report.blurs[0]
    assert hit["word"] == "fuck"
    assert 1.0 < hit["start"] < 4.0
    assert hit["confidence"] > 0.5


def test_live_reports_latency(clip: Path, tmp_path: Path, no_speech) -> None:
    out = tmp_path / "latency.mp4"
    report = LiveFilter(Settings(), no_speech).run(str(clip), str(out))
    assert report.window_latencies
    assert report.average_latency > 0
    assert report.peak_latency >= report.average_latency


def test_live_speech_path_inserts_beep(clip: Path, tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_spoken(audio, offset, start, end, words, transcriber, settings, cancelled):
        calls.append((start, end))
        if start <= 4.0 < end:
            return [Interval(4.0, 4.4, "fuck", "audio", 0.9)]
        return []

    monkeypatch.setattr(live_module, "_spoken", fake_spoken)
    options = LiveOptions(pace=False, speech_enabled=True, visual_enabled=False)
    out = tmp_path / "beep.mp4"
    report = LiveFilter(Settings(), options).run(str(clip), str(out))
    assert calls, "speech analysis should have run"
    assert report.beeps
    assert report.beeps[0]["word"] == "fuck"


def test_live_degrades_when_it_cannot_keep_up(clip: Path, tmp_path: Path, no_speech) -> None:
    options = LiveOptions(
        pace=False,
        speech_enabled=False,
        visual_enabled=True,
        window=1.0,
        max_backlog=0.0,
        degrade_when_behind=True,
    )
    out = tmp_path / "degraded.mp4"
    report = LiveFilter(Settings(), options).run(str(clip), str(out))
    assert out.exists()
    assert report.degraded >= 0


def test_live_handles_missing_audio(tmp_path: Path) -> None:
    path = tmp_path / "silent.mp4"
    build_clip(
        path,
        duration=8.0,
        size=SIZE,
        subtitles=[(1.0, 6.0, PHRASE)],
        with_audio=False,
    )
    out = tmp_path / "silent_out.mp4"
    options = LiveOptions(pace=False, speech_enabled=False, visual_enabled=True)
    report = LiveFilter(Settings(), options).run(str(path), str(out))
    assert out.exists()
    assert report.blurs


def test_live_rejects_source_without_video(tmp_path: Path) -> None:
    path = tmp_path / "audio.m4a"
    build_clip(path, duration=3.0, with_audio=True, with_video=False)
    with pytest.raises(ValueError, match="no video stream"):
        LiveFilter(Settings(), LiveOptions(pace=False)).run(
            str(path), str(tmp_path / "x.mp4")
        )
