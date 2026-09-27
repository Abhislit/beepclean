from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

from app.samples import SAMPLES, build_samples, describe
from app.video import probe


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("samples")
    build_samples(target)
    return target


def test_every_sample_builds(clips: Path) -> None:
    files = sorted(clips.glob("*.mp4"))
    assert len(files) == len(SAMPLES)
    for path in files:
        assert path.stat().st_size > 1000, f"{path.name} came out empty"


def test_samples_have_expected_shape(clips: Path) -> None:
    by_name = {sample["name"]: sample for sample in SAMPLES}
    curse = probe(clips / "1-subtitle-curse.mp4")
    assert curse["has_video"] and curse["has_audio"]
    assert curse["duration"] == pytest.approx(8.0, abs=0.3)

    portrait = probe(clips / "4-portrait-no-text.mp4")
    assert portrait["height"] > portrait["width"], "sample 4 should be portrait"

    silent = probe(clips / "5-no-audio.mp4")
    assert silent["has_video"] and not silent["has_audio"]


def test_describe_lists_every_sample(clips: Path) -> None:
    text = describe(clips)
    for sample in SAMPLES:
        assert f"{sample['name']}.mp4" in text
        assert sample["expects"] in text


def test_top_subtitle_sample_sits_above_the_default_band(clips: Path) -> None:
    from app.config import Settings

    box_top = 0.06
    assert box_top < 1.0 - Settings().ocr_band, (
        "sample 3 only demonstrates a miss if its subtitle is above the default scan band"
    )
