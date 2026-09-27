from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import av
import numpy as np

from .config import Settings
from .intervals import Box, Interval, merge_tracks
from .profanity import ProfanityFilter, normalize

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

MIN_IOU = 0.30
MAX_BOX_HEIGHT = 0.25


@dataclass
class _Track:
    box: Box
    text: str
    confidence: float
    start: float
    last: float
    best_confidence: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.last - self.start)


def _iou(first: Box, second: Box) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    left = max(ax0, bx0)
    top = max(ay0, by0)
    right = min(ax1, bx1)
    bottom = min(ay1, by1)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _similar(first: str, second: str) -> bool:
    if first == second:
        return True
    if not first or not second:
        return False
    return first in second or second in first


class OcrScanner:
    def __init__(self, settings: Settings, words: ProfanityFilter) -> None:
        self.settings = settings
        self.words = words
        self._engine = None

    def engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR(
                params={
                    "Det.limit_type": "max",
                    "Det.limit_side_len": int(self.settings.ocr_max_width),
                }
            )
        return self._engine

    def release(self) -> None:
        self._engine = None
        gc.collect()

    def _detect(self, image: np.ndarray) -> list[tuple[Box, str, float]]:
        result = self.engine()(image)
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is None or len(boxes) == 0:
            return []
        found: list[tuple[Box, str, float]] = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            if score < self.settings.ocr_min_conf or not text or not text.strip():
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            found.append(
                ((min(xs), min(ys), max(xs), max(ys)), text.strip(), float(score))
            )
        return found

    def scan(
        self,
        path: Path,
        duration: float,
        progress: Progress | None = None,
        cancelled: Cancelled | None = None,
    ) -> list[Interval]:
        if duration <= 0:
            return []
        step = 1.0 / max(0.5, self.settings.ocr_fps)
        tracks: list[_Track] = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            if not rate:
                return []
            frame_duration = 1.0 / float(rate)
            band = self.settings.ocr_band
            samples = max(1, int(duration / step))
            reader = None
            position = -1.0
            index = 0
            while index <= samples:
                if cancelled and cancelled():
                    return []
                target = min(duration - 1e-3, index * step)
                index += 1
                if reader is None or target < position or target > position + 0.5:
                    try:
                        container.seek(
                            int(target / av.time_base),
                            stream=stream,
                            backward=True,
                            any_frame=False,
                        )
                    except av.FFmpegError:
                        break
                    reader = container.decode(video=0)
                    position = target
                frame = None
                found_time = target
                for candidate in reader:
                    if candidate.pts is None or not candidate.time_base:
                        continue
                    found_time = float(candidate.pts * candidate.time_base)
                    frame = candidate
                    if found_time >= target - frame_duration * 0.5:
                        break
                position = found_time
                if frame is None:
                    reader = None
                    continue
                image = frame.to_image()
                width, height = image.size
                top = int(height * (1.0 - band))
                crop = image.crop((0, top, width, height))
                scale = min(1.0, self.settings.ocr_max_width / max(1, crop.width))
                if scale < 1.0:
                    crop = crop.resize(
                        (max(32, int(crop.width * scale)), max(8, int(crop.height * scale)))
                    )
                detections = self._detect(np.asarray(crop.convert("RGB")))
                normalized: list[tuple[Box, str, float]] = []
                for box, text, score in detections:
                    x0, y0, x1, y1 = box
                    full_y0 = top + (y0 / max(1, crop.height)) * (height - top)
                    full_y1 = top + (y1 / max(1, crop.height)) * (height - top)
                    normalized_box = (
                        max(0.0, min(1.0, x0 / max(1, crop.width))),
                        max(0.0, min(1.0, full_y0 / height)),
                        max(0.0, min(1.0, x1 / max(1, crop.width))),
                        max(0.0, min(1.0, full_y1 / height)),
                    )
                    if normalized_box[3] - normalized_box[1] > MAX_BOX_HEIGHT:
                        continue
                    normalized.append((normalized_box, text, score))
                self._update(tracks, normalized, found_time)
                if progress:
                    progress(min(0.99, found_time / max(0.001, duration)), "Reading on-screen text")
        return self._finish(tracks)

    def _update(
        self, tracks: list[_Track], detections: Sequence[tuple[Box, str, float]], now: float
    ) -> None:
        gap = self.settings.ocr_merge_gap
        for box, text, score in detections:
            clean = normalize(text)
            best: _Track | None = None
            best_score = 0.0
            for track in tracks:
                if now - track.last > gap:
                    continue
                overlap = _iou(track.box, box)
                if overlap < MIN_IOU:
                    continue
                if _similar(track.text, clean):
                    quality = overlap + 0.5
                elif track.duration < 0.5:
                    quality = overlap
                else:
                    continue
                if quality > best_score:
                    best, best_score = track, quality
            if best is not None:
                best.last = now
                if score > best.best_confidence:
                    best.best_confidence = score
                    best.box = box
                    best.text = clean
            else:
                tracks.append(_Track(box, clean, score, now, now, score))

    def _finish(self, tracks: Sequence[_Track]) -> list[Interval]:
        results: list[Interval] = []
        for track in tracks:
            if track.duration < self.settings.ocr_min_duration:
                continue
            hits = self.words.find(track.text)
            if not hits:
                continue
            pad = 0.08
            start = max(0.0, track.start - pad)
            end = track.last + pad
            results.append(
                Interval(
                    start=start,
                    end=end,
                    word=hits[0],
                    kind="visual",
                    confidence=track.best_confidence,
                    box=track.box,
                    text=track.text,
                )
            )
        return merge_tracks(results, self.settings.ocr_merge_gap)
