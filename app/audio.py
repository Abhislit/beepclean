from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import av
import numpy as np

from .config import Settings
from .intervals import Interval, merge

Progress = Callable[[float, str], None]


def make_beep(rate: int, samples: int, settings: Settings) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / float(rate)
    wave = np.sin(2.0 * math.pi * settings.beep_freq * t).astype(np.float32)
    fade = min(int(rate * settings.fade_ms / 1000.0), samples // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave * float(settings.beep_amplitude)


def build_plan(
    intervals: Sequence[Interval], rate: int, settings: Settings
) -> list[tuple[int, int, str]]:
    pad = settings.edge_pad_ms / 1000.0
    minimum = settings.beep_min_ms / 1000.0
    maximum = settings.beep_max_ms / 1000.0
    plan: list[tuple[int, int, str]] = []
    for interval in intervals:
        start = max(0.0, interval.start - pad)
        end = interval.end + pad
        duration = min(maximum, max(minimum, end - start))
        first = int(round(start * rate))
        last = int(round((start + duration) * rate))
        if last <= first:
            last = first + max(1, int(rate * 0.02))
        plan.append((first, last, interval.word))
    plan.sort()
    return plan


def has_audio(path: Path) -> bool:
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except av.FFmpegError:
        return False


class BeepPlan:
    """Applies beeps to decoded planar-float audio, frame by frame.

    Shared by the batch pipeline and the live filter so both produce identical
    beeps. Beep waveforms are pre-rendered per interval and sliced by absolute
    sample index, so a beep stays phase continuous even when it spans several
    audio frames.
    """

    def __init__(self, intervals: Sequence[Interval], rate: int, settings: Settings) -> None:
        self.rate = rate
        self.entries: list[tuple[int, int, np.ndarray]] = [
            (first, last, make_beep(rate, last - first, settings))
            for first, last, _ in build_plan(merge(intervals), rate, settings)
        ]
        self.total = sum(last - first for first, last, _ in self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def apply(self, array: np.ndarray, base: int) -> bool:
        """Patch array in place. base is the absolute sample index of array[0]."""
        if not self.entries or array.size == 0:
            return False
        count = array.shape[-1]
        touched = False
        for first, last, beep in self.entries:
            low = max(base, first)
            high = min(base + count, last)
            if high <= low:
                continue
            patch = beep[low - first : high - first]
            if array.ndim == 1:
                array[low - base : high - base] = patch
            else:
                array[:, low - base : high - base] = patch.reshape(1, -1)
            touched = True
        return touched


def splice_audio(
    src: Path,
    dst: Path,
    intervals: Sequence[Interval],
    settings: Settings,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    with av.open(str(src)) as container:
        if not container.streams.audio:
            return False
        source = container.streams.audio[0]
        rate = int(source.rate)
        layout = source.layout.name
        plan = BeepPlan(intervals, rate, settings)
        cursor = 0

        with av.open(str(dst), "w") as out:
            target = out.add_stream("aac", rate=rate)
            target.layout = layout
            target.bit_rate = int(settings.audio_bitrate.rstrip("k")) * 1000
            reader = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            writer = av.AudioResampler(format="fltp", layout=layout, rate=rate)

            def emit(frames) -> None:
                nonlocal cursor
                for frame in frames:
                    if frame is None:
                        continue
                    array = frame.to_ndarray()
                    if array.ndim == 1:
                        array = array.reshape(1, -1)
                    base = cursor
                    cursor += array.shape[-1]
                    if plan.apply(array, base):
                        replacement = av.AudioFrame.from_ndarray(
                            np.ascontiguousarray(array), format="fltp", layout=layout
                        )
                        replacement.sample_rate = rate
                        replacement.pts = frame.pts
                        replacement.time_base = frame.time_base
                        frame = replacement
                    for converted in writer.resample(frame):
                        for packet in target.encode(converted):
                            out.mux(packet)

            for frame in container.decode(audio=0):
                if cancelled and cancelled():
                    return False
                emit(reader.resample(frame))
                if progress:
                    progress(0.0, "Beeping audio")
            emit(reader.resample(None))
            for converted in writer.resample(None):
                for packet in target.encode(converted):
                    out.mux(packet)
            for packet in target.encode(None):
                out.mux(packet)
    return True
