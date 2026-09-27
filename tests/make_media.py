from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def build_clip(
    path: Path,
    duration: float = 6.0,
    fps: int = 30,
    size: tuple[int, int] = (640, 360),
    subtitles: list[tuple[float, float, str]] | None = None,
    tones: list[tuple[float, float]] | None = None,
    with_audio: bool = True,
    with_video: bool = True,
) -> dict:
    subtitles = subtitles or []
    tones = tones or []
    width, height = size
    rate = 48000
    font = load_font(max(18, height // 12))
    with av.open(str(path), "w") as out:
        video = None
        if with_video:
            video = out.add_stream("libx264", rate=fps)
            video.width, video.height = width, height
            video.pix_fmt = "yuv420p"
            video.options = {"crf": "20", "preset": "veryfast"}
        audio = None
        if with_audio:
            audio = out.add_stream("aac", rate=rate)
            audio.layout = "stereo"
        frames = int(duration * fps)
        for index in range(frames):
            now = index / fps
            background = Image.new("RGB", (width, height), (24, 32, 48))
            painter = ImageDraw.Draw(background)
            for row in range(0, height, 4):
                shade = int(20 + 40 * (row / height))
                painter.line([(0, row), (width, row)], fill=(shade, shade + 8, shade + 24))
            painter.rectangle([8, 8, width - 8, height // 3], outline=(90, 140, 200), width=2)
            for entry in subtitles:
                start, stop, text = entry[0], entry[1], entry[2]
                position = entry[3] if len(entry) > 3 else 0.78
                if start <= now < stop:
                    left, top, right, bottom = _text_box(
                        painter, font, text, width, height, position
                    )
                    painter.rectangle([left - 4, top - 4, right + 4, bottom + 4], fill=(0, 0, 0))
                    painter.text((left, top), text, font=font, fill=(255, 255, 255))
            if video is None:
                continue
            frame = av.VideoFrame.from_image(background).reformat(format="yuv420p")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame):
                out.mux(packet)
        if video is not None:
            for packet in video.encode():
                out.mux(packet)
        if audio is not None:
            total = int(duration * rate)
            stamps = np.arange(total, dtype=np.float32) / rate
            signal = (0.25 * np.sin(2.0 * math.pi * 330.0 * stamps)).astype(np.float32)
            for start, stop in tones:
                mask = (stamps >= start) & (stamps < stop)
                signal[mask] = 0.5 * np.sin(2.0 * math.pi * 330.0 * stamps[mask])
            block = np.empty((2, total), dtype=np.float32)
            block[0] = signal
            block[1] = signal
            frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = 0
            for packet in audio.encode(frame):
                out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return {"path": path, "duration": duration, "fps": fps, "size": size}


def _text_box(painter, font, text: str, width: int, height: int, position: float = 0.78):
    left, top, right, bottom = painter.textbbox((0, 0), text, font=font)
    offset_x = max(12, (width - (right - left)) // 2)
    offset_y = max(4, int(height * position))
    offset_y = min(offset_y, height - (bottom - top) - 4)
    return offset_x, offset_y, offset_x + (right - left), offset_y + (bottom - top)


def subtitle_box(text: str, width: int, height: int, pad: int = 6) -> tuple[float, float, float, float]:
    font = load_font(max(18, height // 12))
    scratch = Image.new("RGB", (8, 8))
    left, top, right, bottom = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    text_width, text_height = right - left, bottom - top
    x0 = max(12, (width - text_width) // 2) - pad
    y0 = int(height * 0.78) - pad
    return (
        x0 / width,
        y0 / height,
        min(1.0, (x0 + text_width + 2 * pad) / width),
        min(1.0, (y0 + text_height + 2 * pad) / height),
    )
