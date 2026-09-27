from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import audio as audio_engine
from . import video as video_engine
from .config import Settings
from .intervals import Interval, merge
from .ocr import OcrScanner
from .profanity import ProfanityFilter
from .video import probe

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

STAGES = (
    ("probe", "Inspecting video"),
    ("transcribe", "Listening for words"),
    ("text", "Reading on-screen text"),
    ("render", "Replacing words"),
    ("finish", "Finishing up"),
)


@dataclass
class Report:
    words: list[dict] = field(default_factory=list)
    text_hits: list[dict] = field(default_factory=list)
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    model: str = ""
    elapsed: float = 0.0
    beep_count: int = 0
    blur_count: int = 0
    reencoded_video: bool = False
    reencoded_audio: bool = False
    notes: list[str] = field(default_factory=list)


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.model,
                device="cpu",
                compute_type=self.settings.compute_type,
                cpu_threads=self.settings.cpu_threads,
            )
        return self._model

    def release(self) -> None:
        self._model = None
        gc.collect()

    def words(self, path: Path, progress: Progress | None, cancelled: Cancelled | None):
        segments, _ = self.model().transcribe(
            str(path),
            language="en",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 400,
                "threshold": self.settings.vad_threshold,
            },
            condition_on_previous_text=self.settings.condition_on_previous_text,
            beam_size=self.settings.beam_size,
            temperature=0.0,
        )
        collected: list[tuple[str, float, float, float]] = []
        for segment in segments:
            if cancelled and cancelled():
                return collected
            for word in segment.words or ():
                text = (word.word or "").strip()
                if text:
                    collected.append(
                        (text, float(word.start), float(word.end), float(word.probability or 0.0))
                    )
            if progress:
                end = float(segment.end or 0.0)
                progress(min(0.99, end), "Listening for words")
        return collected


def _spoken_hits(
    words: list[tuple[str, float, float, float]],
    filter_: ProfanityFilter,
    settings: Settings,
) -> list[Interval]:
    from .profanity import tokenize

    tokens: list[str] = []
    spans: list[tuple[float, float, float]] = []
    for text, start, end, probability in words:
        parts = tokenize(text)
        if not parts:
            continue
        tokens.extend(parts)
        share = (end - start) / len(parts)
        for offset in range(len(parts)):
            spans.append((start + offset * share, start + (offset + 1) * share, probability))
    hits: list[Interval] = []
    for match in filter_.scan(tokens):
        chosen = spans[match.first] if match.first < len(spans) else None
        if chosen is None:
            continue
        start, end, probability = chosen
        if probability < settings.min_confidence:
            continue
        last = min(match.last, len(spans) - 1)
        if last >= match.first:
            end = spans[last][1]
            probability = min(probability, spans[last][2])
        hits.append(
            Interval(
                start=start,
                end=end,
                word=match.text,
                kind="audio",
                confidence=probability,
            )
        )
    return merge(hits, gap=0.02)


def run(
    source: Path,
    destination: Path,
    settings: Settings,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> Report:
    started = time.time()
    report = Report(model=settings.model)
    work = destination.parent

    def stage(name: str) -> None:
        if progress:
            progress(0.0, dict(STAGES).get(name, name))

    stage("probe")
    info = probe(source)
    report.duration = info["duration"]
    report.width = info["width"]
    report.height = info["height"]
    report.fps = info["fps"]
    report.has_audio = info["has_audio"]
    if not info["has_video"]:
        raise ValueError("This file has no video track.")
    if not info["has_audio"] and settings.ocr_enabled:
        report.notes.append("No audio track found, so only on-screen text was checked.")

    words_filter = ProfanityFilter.load()
    spoken: list[Interval] = []
    if info["has_audio"]:
        stage("transcribe")
        transcriber = Transcriber(settings)

        def on_transcribed(seconds: float, message: str) -> None:
            if not progress:
                return
            fraction = max(0.0, min(1.0, seconds / max(0.001, info["duration"])))
            progress(0.03 + 0.52 * fraction, message)

        try:
            tokens = transcriber.words(source, on_transcribed, cancelled)
        finally:
            transcriber.release()
        spoken = _spoken_hits(tokens, words_filter, settings)
        report.words = [
            {
                "word": hit.word,
                "start": round(hit.start, 3),
                "end": round(hit.end, 3),
                "confidence": round(hit.confidence, 3),
            }
            for hit in spoken
        ]
    else:
        report.notes.append("No audio track found, so no words were beeped.")

    visual: list[Interval] = []
    if settings.ocr_enabled:
        stage("text")
        scanner = OcrScanner(settings, words_filter)
        try:
            visual = scanner.scan(
                source,
                info["duration"],
                lambda fraction, message: progress(0.55 + 0.25 * fraction, message)
                if progress
                else None,
                cancelled,
            )
        finally:
            scanner.release()
        report.text_hits = [
            {
                "word": hit.word,
                "text": hit.text,
                "start": round(hit.start, 3),
                "end": round(hit.end, 3),
                "confidence": round(hit.confidence, 3),
                "box": [round(value, 4) for value in (hit.box or ())],
            }
            for hit in visual
        ]

    report.beep_count = len(spoken)
    report.blur_count = len(visual)

    if cancelled and cancelled():
        raise InterruptedError("Cancelled")

    if not spoken and not visual:
        stage("render")
        video_engine.remux(source, destination)
        report.notes.append("No curse words were found, so the video was copied unchanged.")
    elif settings.report_only:
        stage("render")
        video_engine.remux(source, destination)
        report.notes.append("Report-only mode: the file was copied without changes.")
    else:
        stage("render")
        video_tmp: Path | None = None
        audio_tmp: Path | None = None
        if visual:
            video_tmp = work / "video_track.mp4"
            video_engine.render_video(
                source,
                video_tmp,
                visual,
                settings,
                lambda fraction, message: progress(0.8 + 0.18 * fraction, message)
                if progress
                else None,
                cancelled,
            )
            report.reencoded_video = True
        if spoken:
            audio_tmp = work / "audio_track.m4a"
            audio_engine.splice_audio(
                source,
                audio_tmp,
                spoken,
                settings,
                lambda fraction, message: progress(0.8 + 0.18 * fraction, message)
                if progress
                else None,
                cancelled,
            )
            report.reencoded_audio = True
        video_engine.remux(
            source, destination, video_path=video_tmp, audio_path=audio_tmp
        )
        for temporary in (video_tmp, audio_tmp):
            if temporary is not None and temporary.exists():
                temporary.unlink()

    stage("finish")
    gc.collect()
    report.elapsed = time.time() - started
    return report


def write_report(path: Path, report: Report) -> None:
    path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
