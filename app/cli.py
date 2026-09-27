from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from . import pipeline
from .config import load_settings
from .video import probe


def fmt(seconds: float) -> str:
    seconds = max(0.0, seconds or 0.0)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}.{int((seconds % 1) * 10)}"


def report_lines(report, name: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {name}")
    print(f"{'=' * 62}")
    print(
        f"  {fmt(report.duration)} long  {report.width}x{report.height}"
        f"  {report.fps:.0f}fps  audio={'yes' if report.has_audio else 'no'}"
    )
    print(f"  model {report.model}   finished in {fmt(report.elapsed)}")
    print(f"  {report.beep_count} beeped   {report.blur_count} blurred")
    if report.reencoded_audio:
        print("  audio re-encoded (beeps inserted)")
    if report.reencoded_video:
        print("  video re-encoded (text blurred)")
    if not report.reencoded_audio and not report.reencoded_video:
        print("  no stream re-encoded, original quality kept")

    if report.words:
        print("\n  SPOKEN (beeped):")
        for hit in report.words:
            print(
                f"    {fmt(hit['start'])} -> {fmt(hit['end'])}  {hit['word']:<18}"
                f" confidence {hit['confidence']:.2f}"
            )
    if report.text_hits:
        print("\n  ON-SCREEN TEXT (blurred):")
        for hit in report.text_hits:
            box = hit.get("box") or []
            where = ""
            if len(box) == 4:
                where = f"  box {box[0]:.2f},{box[1]:.2f} {box[2]:.2f},{box[3]:.2f}"
            print(
                f"    {fmt(hit['start'])} -> {fmt(hit['end'])}  {hit['word']:<18}"
                f" read as {hit['text']!r:<28} confidence {hit['confidence']:.2f}{where}"
            )
    if not report.words and not report.text_hits:
        print("\n  nothing was flagged")
    for note in report.notes:
        print(f"\n  note: {note}")


def do_demo(directory: Path) -> int:
    from tests.make_media import build_clip

    print("Building a test clip with a known subtitle...")
    clip = directory / "demo_input.mp4"
    build_clip(
        clip,
        duration=8.0,
        subtitles=[(2.0, 5.0, "what the fuck"), (6.0, 7.0, "a lovely class")],
        tones=[(1.0, 1.5)],
    )
    print(f"  {clip}  ({clip.stat().st_size // 1024} KB)")
    print("  it contains a subtitle saying 'what the fuck' from 2.0s to 5.0s,")
    print("  and a clean subtitle 'a lovely class' from 6.0s to 7.0s that must NOT be flagged.")
    print("  The audio is a tone, not speech, so expect 0 spoken hits.")

    settings = load_settings()
    destination = directory / "demo_output.mp4"
    report = pipeline.run(clip, destination, settings)
    report_lines(report, "demo result")

    print("\n  EXPECTED: 1 blurred hit reading 'what the fuck' around 2.0s-5.0s,")
    print("            0 spoken hits, and 'a lovely class' left alone.")

    ok = (
        report.blur_count == 1
        and report.beep_count == 0
        and destination.exists()
    )
    print(f"\n  RESULT: {'PASS' if ok else 'CHECK IT'}")
    if ok:
        info = probe(destination)
        print(f"  output {destination}  {info['width']}x{info['height']}  {fmt(info['duration'])} long")
        print("  open it and confirm the subtitle is unreadable between 2s and 5s:")
        print(f"    xdg-open {destination}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beepclean", description="Beep out curse words in a video."
    )
    parser.add_argument("file", nargs="?", help="video to clean")
    parser.add_argument("-o", "--out", help="where to write the result")
    parser.add_argument("--no-ocr", action="store_true", help="skip on-screen text, faster")
    parser.add_argument("--report-only", action="store_true", help="list hits, change nothing")
    parser.add_argument("--model", choices=["tiny.en", "base.en", "small.en"])
    parser.add_argument(
        "--blur", choices=["blur", "pixelate", "blackout"], help="on-screen text style"
    )
    parser.add_argument("--out-dir", default="out", help="where results are written")
    parser.add_argument("--samples", default="samples", help="where sample clips live")
    parser.add_argument("--ocr-band", type=float, help="how much of the frame to scan, 0.05-1.0")
    parser.add_argument("--make-samples", action="store_true", help="write sample clips and exit")
    parser.add_argument("--list-samples", action="store_true", help="show the sample clips")
    parser.add_argument("--demo", action="store_true", help="build a test clip and check it")
    args = parser.parse_args(argv)

    directory = Path(args.out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    if args.make_samples:
        from .samples import build_samples, describe

        target = Path(args.samples).expanduser()
        written = build_samples(target)
        print(f"Wrote {len(written)} clips.")
        print(describe(target))
        return 0

    if args.list_samples:
        from .samples import describe

        print(describe(Path(args.samples).expanduser()))
        return 0

    if args.demo:
        return do_demo(directory)

    if not args.file:
        parser.error("give a file, or use --demo")
    source = Path(args.file).expanduser()
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2

    settings = load_settings()
    if args.no_ocr:
        settings.ocr_enabled = False
    if args.report_only:
        settings.report_only = True
    if args.model:
        settings.model = args.model
    if args.blur:
        settings.blur_mode = args.blur
    if args.ocr_band:
        settings.ocr_band = args.ocr_band

    destination = (
        Path(args.out).expanduser()
        if args.out
        else directory / f"{source.stem}-beeped.mp4"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {source.name} ...")
    started = time.time()
    last = [0.0]

    def on_progress(fraction: float, message: str) -> None:
        if fraction - last[0] >= 0.04 or fraction >= 0.999:
            last[0] = fraction
            print(f"  {fraction * 100:5.1f}%  {message}", flush=True)

    try:
        report = pipeline.run(source, destination, settings, on_progress)
    except Exception as error:  # noqa: BLE001
        print(f"\nFAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    report_lines(report, f"{source.name} -> {destination}")
    print(f"\n  total {fmt(time.time() - started)} wall clock")
    print(f"  play or open it:  xdg-open {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
