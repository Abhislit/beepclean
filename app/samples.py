from __future__ import annotations

from pathlib import Path

from .video import probe
from tests.make_media import build_clip

SAMPLES = (
    {
        "name": "1-subtitle-curse",
        "title": "curse word in a subtitle, plus a clean subtitle",
        "expects": "1 blurred, 0 beeped",
        "notes": "The audio is a tone, not speech, so nothing should be beeped.",
        "build": dict(
            duration=8.0,
            subtitles=[(2.0, 5.0, "what the fuck"), (6.0, 7.0, "a lovely class")],
            tones=[(1.0, 1.5)],
        ),
    },
    {
        "name": "2-clean",
        "title": "nothing to flag anywhere",
        "expects": "0 blurred, 0 beeped",
        "notes": "Confirms a clean file is copied untouched instead of re-encoded.",
        "build": dict(
            duration=6.0,
            subtitles=[(1.0, 4.0, "a lovely class"), (1.0, 4.0, "what a great day")],
            tones=[(0.5, 1.0)],
        ),
    },
    {
        "name": "3-subtitle-near-top",
        "title": "curse word in a subtitle near the top of the frame",
        "expects": "0 blurred with the default scan band",
        "notes": "Subtitles near the top are outside the default bottom 30% band. "
        "Use --ocr-band 0.9 to catch these.",
        "build": dict(
            duration=8.0,
            subtitles=[(2.0, 5.0, "what the fuck", 0.06)],
            tones=[(1.0, 1.5)],
        ),
    },
    {
        "name": "4-portrait-no-text",
        "title": "portrait video, 720x1280, no text at all",
        "expects": "0 blurred, 0 beeped",
        "notes": "Shaped like a phone recording, to check portrait handling.",
        "build": dict(
            duration=6.0,
            size=(720, 1280),
            subtitles=[],
            tones=[(1.0, 2.0)],
        ),
    },
    {
        "name": "5-no-audio",
        "title": "video with a curse word in the subtitle but no audio track",
        "expects": "1 blurred, 0 beeped",
        "notes": "Checks the app does not fall over when there is nothing to listen to.",
        "build": dict(
            duration=8.0,
            subtitles=[(2.0, 5.0, "what the fuck")],
            with_audio=False,
        ),
    },
)


def build_samples(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sample in SAMPLES:
        path = destination / f"{sample['name']}.mp4"
        build_clip(path, **sample["build"])
        written.append(path)
    return written


def describe(destination: Path) -> str:
    lines = [f"Sample clips in {destination}", ""]
    for sample in SAMPLES:
        path = destination / f"{sample['name']}.mp4"
        if not path.exists():
            continue
        info = probe(path)
        audio = "yes" if info["has_audio"] else "no"
        lines.append(f"  {path.name}")
        lines.append(f"    {sample['title']}")
        lines.append(
            f"    {info['duration']:.1f}s  {info['width']}x{info['height']}"
            f"  audio={audio}  {path.stat().st_size // 1024} KB"
        )
        lines.append(f"    expect: {sample['expects']}")
        if sample["notes"]:
            lines.append(f"    {sample['notes']}")
        lines.append("")
    return "\n".join(lines)
