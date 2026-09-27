from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence

import av
import numpy as np
from PIL import Image, ImageFilter

from .config import Settings
from .intervals import Box, Interval

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

LUMA_BLACK = 16
CHROMA_NEUTRAL = 128


def probe(path: Path) -> dict:
    with av.open(str(path)) as container:
        video = container.streams.video[0] if container.streams.video else None
        audio = container.streams.audio[0] if container.streams.audio else None
        duration = 0.0
        if container.duration:
            duration = float(container.duration) / float(av.time_base)
        elif video is not None and video.duration and video.time_base:
            duration = float(video.duration * video.time_base)
        rate = 0.0
        if video is not None and video.average_rate:
            rate = float(video.average_rate)
        return {
            "duration": duration,
            "width": int(video.codec_context.width) if video else 0,
            "height": int(video.codec_context.height) if video else 0,
            "fps": rate,
            "frames": int(video.frames) if video else 0,
            "has_video": video is not None,
            "has_audio": audio is not None,
            "sample_rate": int(audio.rate) if audio else 0,
            "channels": len(audio.layout.channels) if audio else 0,
        }


def _plane_views(frame: av.VideoFrame) -> list[tuple[av.VideoPlane, np.ndarray]]:
    views = []
    for plane in frame.planes:
        buffer = np.frombuffer(plane, dtype=np.uint8)
        views.append((plane, buffer.reshape(plane.height, plane.line_size)))
    return views


def _blackout(region: np.ndarray, luma: bool) -> None:
    region[:] = LUMA_BLACK if luma else CHROMA_NEUTRAL


def _pixelate(region: np.ndarray, factor: int) -> None:
    height, width = region.shape
    step = max(2, factor)
    if height < step or width < step:
        return
    rows = height // step
    columns = width // step
    small = (
        region[: rows * step, : columns * step]
        .reshape(rows, step, columns, step)
        .mean(axis=(1, 3))
        .astype(np.uint8)
    )
    blocky = np.repeat(np.repeat(small, step, axis=0), step, axis=1)
    if blocky.shape[0] < height or blocky.shape[1] < width:
        blocky = np.pad(
            blocky,
            (
                (0, max(0, height - blocky.shape[0])),
                (0, max(0, width - blocky.shape[1])),
            ),
            mode="edge",
        )
    region[:] = blocky[:height, :width]


def _blur(region: np.ndarray, radius: int) -> None:
    if radius < 1 or min(region.shape) < 2:
        return
    source = Image.fromarray(np.ascontiguousarray(region))
    region[:] = np.asarray(source.filter(ImageFilter.BoxBlur(radius)))


def _apply_effects(
    frame: av.VideoFrame, tracks: Sequence[Interval], settings: Settings
) -> None:
    width, height = frame.width, frame.height
    views = _plane_views(frame)
    for track in tracks:
        box: Box = track.box or (0.0, 0.0, 1.0, 1.0)
        x0 = max(0, min(width, int(box[0] * width)))
        y0 = max(0, min(height, int(box[1] * height)))
        x1 = max(x0 + 1, min(width, int(round(box[2] * width))))
        y1 = max(y0 + 1, min(height, int(round(box[3] * height))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        box_width, box_height = x1 - x0, y1 - y0
        for index, (_, array) in enumerate(views):
            plane_width, plane_height = array.shape[1], array.shape[0]
            px0 = int(x0 * plane_width / width)
            py0 = int(y0 * plane_height / height)
            px1 = max(px0 + 1, int(round(x1 * plane_width / width)))
            py1 = max(py0 + 1, int(round(y1 * plane_height / height)))
            px1 = min(plane_width, px1)
            py1 = min(plane_height, py1)
            if px1 <= px0 or py1 <= py0:
                continue
            region = array[py0:py1, px0:px1]
            if settings.blur_mode == "blackout":
                _blackout(region, luma=index == 0)
            elif settings.blur_mode == "pixelate":
                _pixelate(region, max(4, min(200, int(box_height * 0.25))))
            else:
                _blur(region, max(3, min(200, int(box_height * 0.35))))


def render_video(
    src: Path,
    dst: Path,
    tracks: Sequence[Interval],
    settings: Settings,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> bool:
    tracks = sorted(tracks, key=lambda item: item.start)
    if not tracks:
        return False
    with av.open(str(src)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(30, 1)
        source_width = int(stream.codec_context.width)
        source_height = int(stream.codec_context.height)
        scale = min(
            1.0, settings.max_resolution / min(source_width, source_height)
        )
        out_width = max(2, int(source_width * scale) & ~1)
        out_height = max(2, int(source_height * scale) & ~1)
        expected = int(stream.frames or 0)
        seen = 0
        with av.open(str(dst), "w") as out:
            target = out.add_stream("libx264", rate=rate)
            target.width = out_width
            target.height = out_height
            target.pix_fmt = "yuv420p"
            target.options = {"crf": str(settings.crf), "preset": settings.preset}
            for frame in container.decode(video=0):
                if cancelled and cancelled():
                    return False
                stamp = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base
                    else seen / float(rate)
                )
                active = [track for track in tracks if track.start <= stamp <= track.end]
                if out_width != frame.width or out_height != frame.height:
                    frame = frame.reformat(width=out_width, height=out_height)
                if active:
                    _apply_effects(frame, active, settings)
                for packet in target.encode(frame):
                    out.mux(packet)
                seen += 1
                if progress:
                    fraction = seen / expected if expected else min(0.99, seen / 300.0)
                    progress(fraction, "Rendering video")
            for packet in target.encode():
                out.mux(packet)
    return True


def _packet_time(packet: av.Packet) -> float:
    base = packet.time_base
    if base is None:
        return 0.0
    stamp = packet.dts if packet.dts is not None else packet.pts
    if stamp is None:
        return 0.0
    return max(0.0, float(stamp * base))


def remux(
    src: Path,
    dst: Path,
    video_path: Path | None = None,
    audio_path: Path | None = None,
) -> bool:
    original = av.open(str(src))
    try:
        replacement_video = av.open(str(video_path)) if video_path else None
        replacement_audio = av.open(str(audio_path)) if audio_path else None
        try:
            out = av.open(str(dst), "w")
            try:
                plan: list[tuple[float, int, av.Packet, object]] = []
                order = 0
                sources: list[tuple[object, list, list]] = []

                new_video = (
                    replacement_video.streams.video[0]
                    if replacement_video and replacement_video.streams.video
                    else None
                )
                new_audio = (
                    replacement_audio.streams.audio[0]
                    if replacement_audio and replacement_audio.streams.audio
                    else None
                )

                def outputs_for(container, streams) -> list:
                    return [out.add_stream_from_template(stream) for stream in streams]

                kept = [
                    stream
                    for stream in original.streams
                    if not (
                        (stream.type == "video" and new_video is not None)
                        or (stream.type == "audio" and new_audio is not None)
                    )
                ]
                if kept:
                    sources.append((original, kept, outputs_for(original, kept)))
                if new_video is not None:
                    sources.append(
                        (replacement_video, [new_video], outputs_for(replacement_video, [new_video]))
                    )
                if new_audio is not None:
                    sources.append(
                        (replacement_audio, [new_audio], outputs_for(replacement_audio, [new_audio]))
                    )

                for container, ins, outs in sources:
                    mapping = dict(zip(ins, outs))
                    for packet in container.demux(*ins):
                        if packet.pts is None and packet.dts is None:
                            continue
                        plan.append((_packet_time(packet), order, packet, mapping[packet.stream]))
                        order += 1

                plan.sort(key=lambda item: (item[0], item[1]))
                for _, _, packet, target in plan:
                    packet.stream = target
                    out.mux_one(packet)
            finally:
                out.close()
        finally:
            if replacement_video is not None:
                replacement_video.close()
            if replacement_audio is not None:
                replacement_audio.close()
    finally:
        original.close()
    return True
