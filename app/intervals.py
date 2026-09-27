from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    word: str = ""
    kind: str = "audio"
    confidence: float = 0.0
    box: Box | None = None
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def clipped(self, limit: float) -> "Interval | None":
        start = max(0.0, min(self.start, limit))
        end = max(0.0, min(self.end, limit))
        if end - start < 0.01:
            return None
        return Interval(start, end, self.word, self.kind, self.confidence, self.box, self.text)


def merge(items: Iterable[Interval], gap: float = 0.01) -> list[Interval]:
    ordered = sorted(items, key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for item in ordered:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        if item.start <= previous.end + gap:
            labels = [label for label in (previous.word, item.word) if label]
            boxes = [previous.box, item.box]
            kind = previous.kind if previous.kind == item.kind else item.kind
            merged[-1] = Interval(
                start=previous.start,
                end=max(previous.end, item.end),
                word=labels[0] if labels else "",
                kind=kind,
                confidence=max(previous.confidence, item.confidence),
                box=boxes[0] if boxes[0] is not None else boxes[1],
                text=previous.text or item.text,
            )
        else:
            merged.append(item)
    return merged


def merge_tracks(tracks: Sequence[Interval], gap: float) -> list[Interval]:
    grouped: dict[Box, list[Interval]] = {}
    for track in tracks:
        grouped.setdefault(track.box or (0.0, 0.0, 0.0, 0.0), []).append(track)
    result: list[Interval] = []
    for items in grouped.values():
        result.extend(merge(items, gap=gap))
    return sorted(result, key=lambda item: item.start)
