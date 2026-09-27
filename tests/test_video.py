from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from app.config import Settings
from app.intervals import Interval
from app.video import probe, remux, render_video
from tests.make_media import build_clip, subtitle_box

SIZE = (640, 360)


def luma(path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            plane = frame.planes[0]
            array = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
            frames.append(array[: frame.height, : frame.width].copy())
    return frames


def edge_energy(region: np.ndarray) -> float:
    return float(np.abs(np.diff(region.astype(np.float32), axis=0)).mean())


def box_region(region_frame: np.ndarray, box: tuple[float, float, float, float], w: int, h: int):
    x0, y0, x1, y1 = box
    return region_frame[
        int(y0 * h) : int(y1 * h),
        int(x0 * w) : int(x1 * w),
    ]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    build_clip(path, duration=6.0, size=SIZE, subtitles=[(2.0, 5.0, "what the fuck")])
    return path


@pytest.fixture(params=["blur", "pixelate", "blackout"])
def mode(request) -> str:
    return request.param


def test_probe_reports_streams(clip: Path) -> None:
    info = probe(clip)
    assert info["has_video"] and info["has_audio"]
    assert info["width"] == SIZE[0] and info["height"] == SIZE[1]
    assert info["fps"] == pytest.approx(30.0)
    assert info["duration"] == pytest.approx(6.0, abs=0.2)


def test_blur_hides_text_only_inside_interval(clip: Path, tmp_path: Path, mode: str) -> None:
    settings = Settings()
    settings.blur_mode = mode
    box = subtitle_box("what the fuck", *SIZE)
    track = Interval(2.5, 4.5, "fuck", "visual", 1.0, box, "what the fuck")
    destination = tmp_path / f"out_{mode}.mp4"
    assert render_video(clip, destination, [track], settings) is True

    before, after = luma(clip), luma(destination)
    assert len(after) == len(before)
    inside_before = edge_energy(box_region(before[90], box, *SIZE))
    inside_after = edge_energy(box_region(after[90], box, *SIZE))
    assert inside_after < inside_before * 0.6, f"{mode} did not reduce detail"

    outside_before = edge_energy(box_region(before[10], box, *SIZE))
    outside_after = edge_energy(box_region(after[10], box, *SIZE))
    assert abs(outside_after - outside_before) < max(3.0, outside_before * 0.35)


def test_render_without_tracks_returns_false(clip: Path, tmp_path: Path) -> None:
    assert render_video(clip, tmp_path / "none.mp4", [], Settings()) is False


def test_render_scales_down_large_video(tmp_path: Path) -> None:
    path = tmp_path / "big.mp4"
    build_clip(path, duration=2.0, size=(1920, 1080), subtitles=[(0.5, 1.5, "fuck")])
    settings = Settings()
    settings.max_resolution = 480
    box = subtitle_box("fuck", 1920, 1080)
    destination = tmp_path / "small.mp4"
    render_video(path, destination, [Interval(0.5, 1.5, "fuck", "visual", 1.0, box)], settings)
    info = probe(destination)
    assert info["height"] == 480
    assert info["width"] % 2 == 0 and info["height"] % 2 == 0


def test_remux_preserves_both_streams(clip: Path, tmp_path: Path) -> None:
    destination = tmp_path / "copy.mp4"
    assert remux(clip, destination) is True
    info = probe(destination)
    assert info["has_video"] and info["has_audio"]
    assert info["duration"] == pytest.approx(6.0, abs=0.2)
    assert info["width"] == SIZE[0]


def test_remux_swaps_in_replacement_video(clip: Path, tmp_path: Path) -> None:
    settings = Settings()
    settings.blur_mode = "blackout"
    box = subtitle_box("what the fuck", *SIZE)
    video_only = tmp_path / "v.mp4"
    render_video(clip, video_only, [Interval(2.5, 4.5, "fuck", "visual", 1.0, box)], settings)
    destination = tmp_path / "swapped.mp4"
    assert remux(clip, destination, video_path=video_only) is True
    info = probe(destination)
    assert info["has_audio"], "audio must survive when only video is replaced"
    assert info["duration"] == pytest.approx(6.0, abs=0.2)


def test_remux_rejects_nothing_and_keeps_duration(clip: Path, tmp_path: Path) -> None:
    destination = tmp_path / "again.mp4"
    remux(clip, destination)
    first, second = probe(clip)["duration"], probe(destination)["duration"]
    assert abs(first - second) < 0.2
