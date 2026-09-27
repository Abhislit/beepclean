from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Sequence

import av
import numpy as np

from .audio import BeepPlan
from .config import Settings
from .intervals import Interval, merge, merge_tracks
from .ocr import OcrScanner
from .profanity import ProfanityFilter, tokenize

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

WHISPER_RATE = 16000


@dataclass
class LiveOptions:
    window: float = 6.0
    overlap: float = 1.5
    ocr_interval: float = 0.75
    ocr_max_width: int = 480
    ocr_band: float | None = None
    ocr_pad: float = 0.7
    max_height: int = 0
    preset: str = "ultrafast"
    crf: int = 23
    audio_bitrate: str = "128k"
    pace: bool | None = None
    realtime_fallback: bool = False
    drop_when_behind: bool = True
    degrade_when_behind: bool = True
    max_backlog: float = 15.0
    speech_enabled: bool = True
    visual_enabled: bool = True


@dataclass
class LiveReport:
    source: str = ""
    destination: str = ""
    windows: int = 0
    beeps: list[dict] = field(default_factory=list)
    blurs: list[dict] = field(default_factory=list)
    media_seconds: float = 0.0
    wall_seconds: float = 0.0
    window_latencies: list[float] = field(default_factory=list)
    dropped: bool = False
    degraded: int = 0
    backlog: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def average_latency(self) -> float:
        if not self.window_latencies:
            return 0.0
        return sum(self.window_latencies) / len(self.window_latencies)

    @property
    def peak_latency(self) -> float:
        return max(self.window_latencies) if self.window_latencies else 0.0

    @property
    def speed(self) -> float:
        return self.media_seconds / self.wall_seconds if self.wall_seconds else 0.0


def _spoken(
    audio: np.ndarray,
    offset: float,
    window_start: float,
    window_end: float,
    words: ProfanityFilter,
    transcriber,
    settings: Settings,
    cancelled: Cancelled | None,
) -> list[Interval]:
    if audio.size == 0 or not settings.model:
        return []
    try:
        tokens = transcriber.words_array(
            audio, offset, lambda f, m: None, cancelled
        )
    except Exception:  # noqa: BLE001
        return []
    if not tokens:
        return []
    hits: list[Interval] = []
    pieces: list[str] = []
    spans: list[tuple[float, float, float]] = []
    for text, start, end, probability in tokens:
        parts = tokenize(text)
        if not parts:
            continue
        pieces.extend(parts)
        share = (end - start) / len(parts)
        for step in range(len(parts)):
            spans.append(
                (start + step * share, start + (step + 1) * share, probability)
            )
    for match in words.scan(pieces):
        if match.first >= len(spans):
            continue
        start, end, probability = spans[match.first]
        last = min(match.last, len(spans) - 1)
        if last >= match.first:
            end = spans[last][1]
            probability = min(probability, spans[last][2])
        if probability < settings.min_confidence:
            continue
        if end <= window_start or start >= window_end:
            continue
        hits.append(
            Interval(
                start=max(start, window_start),
                end=min(end, window_end),
                word=match.text,
                kind="audio",
                confidence=probability,
            )
        )
    return merge(hits, gap=0.02)


class LiveFilter:
    def __init__(self, settings: Settings, options: LiveOptions | None = None) -> None:
        self.settings = settings
        self.options = options or LiveOptions()
        self.words = ProfanityFilter.load()
        self._scanner: OcrScanner | None = None
        self._transcriber = None
        self._records: list = []

    def _ocr(self) -> OcrScanner:
        if self._scanner is None:
            self._scanner = OcrScanner(self.settings, self.words)
        return self._scanner

    def _whisper(self):
        if self._transcriber is None:
            from .pipeline import Transcriber

            self._transcriber = Transcriber(self.settings)
        return self._transcriber

    def release(self) -> None:
        if self._scanner is not None:
            self._scanner.release()
            self._scanner = None
        if self._transcriber is not None:
            self._transcriber.release()
            self._transcriber = None
        gc.collect()

    def run(
        self,
        source: str,
        destination: str,
        progress: Progress | None = None,
        cancelled: Cancelled | None = None,
    ) -> LiveReport:
        options = self.options
        report = LiveReport(source=source, destination=destination)
        started = time.time()
        window = max(1.0, options.window)

        container = av.open(source, timeout=10_000_000 if "://" in source else None)
        try:
            if not container.streams.video:
                raise ValueError("Live source has no video stream.")
            video_in = container.streams.video[0]
            audio_in = container.streams.audio[0] if container.streams.audio else None
            rate = video_in.average_rate or Fraction(30, 1)
            source_width = int(video_in.codec_context.width)
            source_height = int(video_in.codec_context.height)
            if options.max_height and options.max_height < max(
                source_width, source_height
            ):
                scale = options.max_height / max(source_width, source_height)
                out_width = max(2, int(source_width * scale) & ~1)
                out_height = max(2, int(source_height * scale) & ~1)
            else:
                out_width = source_width
                out_height = source_height

            out = av.open(destination, "w")
            try:
                video_out = out.add_stream("libx264", rate=rate)
                video_out.width = out_width
                video_out.height = out_height
                video_out.pix_fmt = "yuv420p"
                video_out.options = {
                    "crf": str(options.crf),
                    "preset": options.preset,
                    "tune": "zerolatency",
                }
                audio_out = None
                audio_rate = 0
                audio_layout = "stereo"
                if audio_in is not None:
                    audio_rate = int(audio_in.rate)
                    audio_layout = audio_in.layout.name
                    audio_out = out.add_stream("aac", rate=audio_rate)
                    audio_out.layout = audio_layout
                    audio_out.bit_rate = int(options.audio_bitrate.rstrip("k")) * 1000

                wanted = []
                if video_in is not None:
                    wanted.append(video_in)
                if audio_in is not None:
                    wanted.append(audio_in)

                demux = container.demux(*wanted)
                ingest = _Ingest(
                    container=container,
                    video_in=video_in,
                    audio_in=audio_in,
                    audio_rate=audio_rate,
                    audio_layout=audio_layout,
                    out_width=out_width,
                    out_height=out_height,
                    window=window,
                    options=options,
                    words=self.words,
                    scanner=self._ocr() if options.visual_enabled else None,
                    cancelled=cancelled,
                    on_overflow=None,
                )
                def _roll(media_time: float = 0.0) -> None:
                    behind = (wall_origin + ingest.end) - time.time()
                    report.backlog = max(report.backlog, behind)
                    skip = (
                        ("://" in source or options.pace)
                        and options.degrade_when_behind
                        and behind > options.max_backlog
                    )
                    self._close_window(ingest, emit, report, progress, skip)

                ingest.on_overflow = _roll
                emit = _Emit(
                    out=out,
                    video_out=video_out,
                    audio_out=audio_out,
                    out_width=out_width,
                    out_height=out_height,
                    source_width=source_width,
                    source_height=source_height,
                    rate=rate,
                    audio_rate=audio_rate,
                    audio_layout=audio_layout,
                    settings=self.settings,
                    options=options,
                )
                wall_origin = time.time()
                last_media = 0.0
                for packet in demux:
                    if cancelled and cancelled():
                        break
                    if packet.pts is None and packet.dts is None:
                        continue
                    media_time = (
                        float(packet.pts * packet.time_base)
                        if packet.pts is not None and packet.time_base
                        else last_media
                    )
                    last_media = max(last_media, media_time)
                    if media_time < 0:
                        continue
                    paced = ("://" in source) if options.pace is None else options.pace
                    if paced:
                        gap = wall_origin + media_time - time.time()
                        if gap > 0.02:
                            time.sleep(gap)
                        elif gap < -options.max_backlog:
                            if options.drop_when_behind:
                                report.dropped = True
                            report.errors.append(
                                f"fell behind by {abs(gap):.1f}s at t={media_time:.1f}"
                            )
                    ingest.push(packet, media_time, report)
                    if progress:
                        done = min(0.99, media_time / max(window * 4.0, 1.0))
                        progress(
                            done,
                            f"live {media_time:.0f}s in, {report.average_latency:.1f}s behind",
                        )
                ingest.flush()
                while ingest.has_pending():
                    self._close_window(ingest, emit, report, progress)
                emit.finish()
                report.media_seconds = last_media
            finally:
                out.close()
        finally:
            container.close()
        report.wall_seconds = time.time() - started
        self._records = []
        self.release()
        return report

    def _close_window(
        self,
        ingest: "_Ingest",
        emit: "_Emit",
        report: LiveReport,
        progress: Progress | None,
        skip_analysis: bool = False,
    ) -> None:
        start, end = ingest.bounds
        opened = time.time()
        video_frames, audio_frames, ocr_records = ingest.take()
        audio_base = ingest.audio_cursor - sum(
            frame.to_ndarray().shape[-1] for frame, _ in audio_frames
        )
        beeps: list[Interval] = []
        blurs: list[Interval] = []
        if skip_analysis:
            report.degraded += 1
        else:
            if self.options.speech_enabled and ingest.speech_audio().size:
                beeps = _spoken(
                    ingest.speech_audio(),
                    ingest.speech_offset,
                    start,
                    end,
                    self.words,
                    self._whisper(),
                    self.settings,
                    ingest.cancelled,
                )
            if self.options.visual_enabled and self._scanner is not None:
                self._records.extend(ocr_records)
                history = self.options.window * 6.0
                if self._records and start - self._records[0][0] > history:
                    self._records = [r for r in self._records if r[0] >= start - history]
                for interval in self._scanner.tracks_to_intervals(
                    self._records, min_duration=0.0, pad=self.options.ocr_pad
                ):
                    if interval.end > start and interval.start < end:
                        blurs.append(interval)
        emit.process(
            start=start,
            end=end,
            video_frames=video_frames,
            audio_frames=audio_frames,
            blurs=blurs,
            beeps=beeps,
            audio_base=audio_base,
            audio_rate=ingest.audio_rate,
            audio_layout=ingest.audio_layout,
            settings=self.settings,
        )
        report.windows += 1
        report.window_latencies.append(max(0.0, time.time() - opened))
        for hit in beeps:
            report.beeps.append(
                {
                    "word": hit.word,
                    "start": round(hit.start, 3),
                    "end": round(hit.end, 3),
                    "confidence": round(hit.confidence, 3),
                }
            )
        for hit in blurs:
            report.blurs.append(
                {
                    "word": hit.word,
                    "text": hit.text,
                    "start": round(hit.start, 3),
                    "end": round(hit.end, 3),
                    "confidence": round(hit.confidence, 3),
                }
            )
        ingest.roll()
        if progress:
            progress(0.0, f"emitted window ending {end:.1f}s")


class _Ingest:
    def __init__(
        self,
        container,
        video_in,
        audio_in,
        audio_rate: int,
        audio_layout: str,
        out_width: int,
        out_height: int,
        window: float,
        options: LiveOptions,
        words: ProfanityFilter,
        scanner: OcrScanner | None,
        cancelled: Cancelled | None,
        on_overflow: Callable[[], None] | None = None,
    ) -> None:
        self.on_overflow = on_overflow
        self.container = container
        self.video_in = video_in
        self.audio_in = audio_in
        self.audio_rate = audio_rate
        self.audio_layout = audio_layout
        self.out_width = out_width
        self.out_height = out_height
        self.window = window
        self.options = options
        self.words = words
        self.scanner = scanner
        self.cancelled = cancelled
        self.index = 0
        self.bounds = (0.0, window)
        self.video_frames: list[tuple[av.VideoFrame, float]] = []
        self.audio_frames: list[tuple[av.AudioFrame, float]] = []
        self.ocr_records: list = []
        self.speech_offset = 0.0
        self._speech: list[np.ndarray] = []
        self._tail = np.zeros(0, dtype=np.float32)
        self._next_sample = 0.0
        self._speech_resampler = (
            av.AudioResampler(format="fltp", layout="mono", rate=WHISPER_RATE)
            if audio_in is not None
            else None
        )
        self._audio_reader = (
            av.AudioResampler(
                format="fltp", layout=audio_layout, rate=audio_rate
            )
            if audio_in is not None
            else None
        )
        self._video_decoder = video_in.codec_context if video_in is not None else None
        self._audio_decoder = audio_in.codec_context if audio_in is not None else None
        self._audio_cursor = 0

    @property
    def start(self) -> float:
        return self.bounds[0]

    @property
    def end(self) -> float:
        return self.bounds[1]

    @property
    def audio_cursor(self) -> int:
        return self._audio_cursor

    def has_pending(self) -> bool:
        return bool(self.video_frames or self.audio_frames)

    def speech_audio(self) -> np.ndarray:
        parts = ([self._tail] if self._tail.size else []) + self._speech
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([p.reshape(-1) for p in parts])

    def push(self, packet, media_time: float, report: LiveReport) -> None:
        if packet.stream is self.video_in or packet.stream.type == "video":
            self._push_video(packet, media_time)
        else:
            self._push_audio(packet, media_time)

    def _push_video(self, packet, media_time: float) -> None:
        if self._video_decoder is None:
            return
        try:
            frames = self._video_decoder.decode(packet)
        except av.FFmpegError:
            return
        for frame in frames:
            if frame.pts is None or not frame.time_base:
                continue
            stamp = float(frame.pts * frame.time_base)
            while stamp >= self.end and self.has_pending():
                if self.on_overflow is None:
                    break
                self.on_overflow(stamp)
            if stamp < self.start or stamp >= self.end:
                continue
            if frame.width != self.out_width or frame.height != self.out_height:
                frame = frame.reformat(
                    width=self.out_width, height=self.out_height
                )
            if self.scanner is not None and stamp >= self._next_sample:
                self.scanner.inspect(frame, stamp, self.ocr_records, self.options)
                self._next_sample = stamp + max(0.05, self.options.ocr_interval)
            self.video_frames.append((frame, stamp))

    def _push_audio(self, packet, media_time: float) -> None:
        if self._audio_decoder is None or self._audio_reader is None:
            return
        try:
            frames = self._audio_decoder.decode(packet)
        except av.FFmpegError:
            return
        for frame in frames:
            if self._speech_resampler is not None:
                for mono in self._speech_resampler.resample(frame):
                    self._speech.append(mono.to_ndarray()[0])
            for converted in self._audio_reader.resample(frame):
                count = converted.to_ndarray().shape[-1]
                if media_time >= self.start:
                    self.audio_frames.append((converted, media_time))
                    self._audio_cursor += count

    def take(self) -> tuple[list, list, list]:
        video, self.video_frames = self.video_frames, []
        audio, self.audio_frames = self.audio_frames, []
        records, self.ocr_records = self.ocr_records, []
        return video, audio, records

    def roll(self) -> None:
        finished_end = self.end
        self.index += 1
        self.bounds = (
            self.start + self.window,
            self.start + self.window + self.window,
        )
        keep = max(0.0, finished_end - self.start)
        self._next_sample = self.bounds[0]
        keep_samples = int(keep * WHISPER_RATE)
        if self._speech:
            joined = np.concatenate([p.reshape(-1) for p in self._speech])
            self._tail = (
                joined[-keep_samples:]
                if keep_samples
                else np.zeros(0, dtype=np.float32)
            )
        else:
            self._tail = np.zeros(0, dtype=np.float32)
        self._speech = []
        self.speech_offset = self.bounds[0]

    def flush(self) -> None:
        """Drain frames still held inside the decoders at end of stream."""
        if self._video_decoder is not None:
            try:
                for frame in self._video_decoder.decode(None):
                    if frame.pts is None or not frame.time_base:
                        continue
                    stamp = float(frame.pts * frame.time_base)
                    if stamp < self.start or stamp >= self.end:
                        continue
                    if frame.width != self.out_width or frame.height != self.out_height:
                        frame = frame.reformat(
                            width=self.out_width, height=self.out_height
                        )
                    self.video_frames.append((frame, stamp))
            except av.FFmpegError:
                pass
        if self._audio_decoder is not None and self._audio_reader is not None:
            try:
                for frame in self._audio_decoder.decode(None):
                    if self._speech_resampler is not None:
                        for mono in self._speech_resampler.resample(frame):
                            self._speech.append(mono.to_ndarray()[0])
                    for converted in self._audio_reader.resample(frame):
                        self._audio_cursor += converted.to_ndarray().shape[-1]
                        self.audio_frames.append((converted, self.end - 0.001))
            except av.FFmpegError:
                pass

    def reset(self) -> None:
        self.video_frames = []
        self.audio_frames = []
        self.ocr_records = []


class _Emit:
    def __init__(
        self,
        out,
        video_out,
        audio_out,
        out_width: int,
        out_height: int,
        source_width: int,
        source_height: int,
        rate: Fraction,
        audio_rate: int,
        audio_layout: str,
        settings: Settings,
        options: LiveOptions,
    ) -> None:
        self.out = out
        self.video_out = video_out
        self.audio_out = audio_out
        self.out_width = out_width
        self.out_height = out_height
        self.source_width = source_width
        self.source_height = source_height
        self.rate = rate
        self.audio_rate = audio_rate
        self.audio_layout = audio_layout
        self.settings = settings
        self.options = options
        self._video_pts = 0
        self._audio_writer = (
            av.AudioResampler(
                format="fltp", layout=audio_layout, rate=audio_rate
            )
            if audio_out is not None
            else None
        )

    def process(
        self,
        start: float,
        end: float,
        video_frames: list,
        audio_frames: list,
        blurs: list[Interval],
        beeps: list[Interval],
        audio_base: int,
        audio_rate: int,
        audio_layout: str,
        settings: Settings,
    ) -> None:
        if video_frames:
            self._video(video_frames, blurs, start)
        if audio_frames and self.audio_out is not None:
            if beeps:
                self._audio(
                    audio_frames, beeps, audio_base, audio_rate, audio_layout, settings
                )
            else:
                self._audio_plain(audio_frames)

    def _audio_plain(self, frames: list) -> None:
        for frame, _ in frames:
            for converted in self._audio_writer.resample(frame):
                for encoded in self.audio_out.encode(converted):
                    self.out.mux_one(encoded)

    def _video(self, frames: list, blurs: list[Interval], start: float) -> None:
        from .video import apply_blur_to_frame

        rate = int(self.rate)
        for frame, stamp in frames:
            if stamp < start:
                continue
            active = [track for track in blurs if track.start <= stamp <= track.end]
            if active:
                apply_blur_to_frame(frame, active, self.settings.blur_mode)
            frame.pts = self._video_pts
            frame.time_base = Fraction(1, rate)
            self._video_pts += 1
            for encoded in self.video_out.encode(frame):
                self.out.mux_one(encoded)

    def _audio(
        self,
        frames: list,
        beeps: list[Interval],
        base: int,
        rate: int,
        layout: str,
        settings: Settings,
    ) -> None:
        plan = BeepPlan(beeps, rate, settings)
        cursor = base
        for frame, _ in frames:
            array = frame.to_ndarray()
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if plan and plan.apply(array, cursor):
                replacement = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(array), format="fltp", layout=layout
                )
                replacement.sample_rate = rate
                frame = replacement
            cursor += array.shape[-1]
            for converted in self._audio_writer.resample(frame):
                for encoded in self.audio_out.encode(converted):
                    self.out.mux_one(encoded)

    def finish(self) -> None:
        for encoded in self.video_out.encode(None):
            self.out.mux_one(encoded)
        if self.audio_out is not None:
            for converted in self._audio_writer.resample(None):
                for encoded in self.audio_out.encode(converted):
                    self.out.mux_one(encoded)
            for encoded in self.audio_out.encode(None):
                self.out.mux_one(encoded)
