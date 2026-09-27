from __future__ import annotations

import warnings
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

warnings.filterwarnings("ignore")

from app.config import Settings
from app.intervals import Interval
from app.video import render_video
from tests.make_media import build_clip, subtitle_box

SIZE = (640, 360)
PHRASE = "what the fuck"


def crop_at(path: Path, box, at: float, size=SIZE):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(at / av.time_base), stream=stream, backward=True, any_frame=False)
        frame = None
        for candidate in container.decode(video=0):
            if float(candidate.pts * candidate.time_base) >= at - 0.02:
                frame = candidate
                break
    assert frame is not None
    image = frame.to_image()
    width, height = image.size
    return image.crop(
        (
            int(box[0] * width),
            int(box[1] * height),
            int(box[2] * width),
            int(box[3] * height),
        )
    )


def sharpness(crop: Image.Image) -> float:
    grey = np.asarray(crop.convert("L")).astype(np.float32)
    return float(np.abs(np.diff(grey, axis=0)).mean())


def read_text(crop: Image.Image) -> str:
    from rapidocr import RapidOCR

    engine = RapidOCR(params={"Det.limit_type": "max", "Det.limit_side_len": 960})
    result = engine(np.asarray(crop.convert("RGB")))
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return ""
    return " ".join(result.txts)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("defeat") / "clip.mp4"
    build_clip(path, duration=6.0, size=SIZE, subtitles=[(1.0, 5.0, PHRASE)])
    return path


def test_source_caption_is_readable(clip: Path) -> None:
    """Guards the test itself: if OCR cannot read the original, the test is worthless."""
    box = subtitle_box(PHRASE, *SIZE)
    crop = crop_at(clip, box, 3.0)
    assert read_text(crop).lower().find("fuck") >= 0


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_effect_makes_caption_unreadable(clip: Path, tmp_path: Path, mode: str) -> None:
    settings = Settings()
    settings.blur_mode = mode
    box = subtitle_box(PHRASE, *SIZE)
    track = Interval(1.5, 4.5, "fuck", "visual", 1.0, box, PHRASE)
    destination = tmp_path / f"{mode}.mp4"
    render_video(clip, destination, [track], settings)

    before = crop_at(clip, box, 3.0)
    after = crop_at(destination, box, 3.0)
    assert sharpness(after) < sharpness(before) * 0.6
    assert read_text(after) == "", f"{mode} left the caption readable: {read_text(after)!r}"


def test_blackout_removes_caption(clip: Path, tmp_path: Path) -> None:
    settings = Settings()
    settings.blur_mode = "blackout"
    box = subtitle_box(PHRASE, *SIZE)
    destination = tmp_path / "blackout.mp4"
    render_video(clip, destination, [Interval(1.5, 4.5, "fuck", "visual", 1.0, box)], settings)
    assert read_text(crop_at(destination, box, 3.0)) == ""


def test_effect_scales_with_caption_size(tmp_path: Path) -> None:
    """The old bug: an absolute pixel cap left big captions legible at 1080p."""
    for size in ((640, 360), (1080, 1920)):
        path = tmp_path / f"clip_{size[1]}.mp4"
        build_clip(path, duration=4.0, size=size, subtitles=[(0.5, 3.5, PHRASE)])
        box = subtitle_box(PHRASE, *size)
        settings = Settings()
        destination = tmp_path / f"out_{size[1]}.mp4"
        render_video(path, destination, [Interval(0.8, 3.2, "fuck", "visual", 1.0, box)], settings)
        crop = crop_at(destination, box, 2.0, size)
        assert read_text(crop) == "", f"caption survived at {size}"
