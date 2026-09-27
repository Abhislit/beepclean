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


def do_live(args, settings) -> int:
    from .live import LiveFilter, LiveOptions

    directory = Path(args.out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    if args.rtmp_in:
        source = args.rtmp_in
        destination = args.rtmp_out or ""
        if not destination:
            print("Live input needs --out with an rtmp:// destination.", file=sys.stderr)
            return 2
    elif args.simulate:
        origin = Path(args.simulate).expanduser()
        if not origin.exists():
            print(f"no such file: {origin}", file=sys.stderr)
            return 2
        source = str(origin)
        destination = str(
            directory / (args.out or f"{origin.stem}-live.mp4")
        )
    else:
        print("give --rtmp-in, or --simulate FILE to try it without a stream.", file=sys.stderr)
        return 2

    options = LiveOptions(
        window=args.window,
        ocr_interval=args.ocr_interval,
        max_height=args.max_height,
        preset=args.preset,
        pace=True if args.paced else (False if args.fast else None),
        speech_enabled=not args.no_speech,
        visual_enabled=not args.no_visual,
    )
    if args.ocr_band:
        settings.ocr_band = args.ocr_band

    live = LiveFilter(settings, options)
    print(f"source:      {source}")
    print(f"destination: {destination}")
    print(
        f"window {options.window}s, OCR every {options.ocr_interval}s,"
        f" output {'source size' if not options.max_height else str(options.max_height) + 'px'},"
        f" preset {options.preset}"
    )
    pace_label = (
        "real time, like a live stream"
        if options.pace
        else ("as fast as possible" if options.pace is False else "as fast as possible")
    )
    print(f"pacing:      {pace_label}")
    print("\nCtrl+C to stop.\n")

    try:
        report = live.run(
            source,
            destination,
            progress=lambda fraction, message: print(
                f"  {message}", flush=True
            )
            if message.startswith(("live", "emitted"))
            else None,
        )
    except KeyboardInterrupt:
        print("\nstopped.")
        return 130
    except Exception as error:  # noqa: BLE001
        print(f"\nFAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 62}")
    print("  live run finished")
    print(f"{'=' * 62}")
    print(f"  windows processed   {report.windows}")
    print(f"  media consumed      {report.media_seconds:.1f}s")
    print(f"  wall clock          {report.wall_seconds:.1f}s  ({report.speed:.2f}x realtime)")
    print(f"  average latency     {report.average_latency:.2f}s per window")
    print(f"  peak latency        {report.peak_latency:.2f}s")
    print(f"  beeps inserted      {len(report.beeps)}")
    print(f"  regions blurred     {len(report.blurs)}")
    for hit in report.beeps[:40]:
        print(
            f"    {fmt(hit['start'])} -> {fmt(hit['end'])}  {hit['word']:<18}"
            f" confidence {hit['confidence']:.2f}"
        )
    for hit in report.blurs[:40]:
        print(
            f"    {fmt(hit['start'])} -> {fmt(hit['end'])}  {hit['word']:<18}"
            f" read as {hit['text']!r}"
        )
    if report.dropped:
        print("  WARNING: could not keep up, some input was skipped.")
    for note in report.errors[-5:]:
        print(f"  note: {note}")
    if destination and "://" not in destination:
        print(f"\n  play it: xdg-open {destination}")
    return 0


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
    parser.add_argument("--rtmp-in", help="live source, e.g. rtmp://live.twitch.tv/app/KEY")
    parser.add_argument("--rtmp-out", help="live destination, e.g. rtmp://a.relay/KEY")
    parser.add_argument("--simulate", metavar="FILE", help="treat a local file as a live feed")
    parser.add_argument("--fast", action="store_true", help="run as fast as possible, no waiting")
    parser.add_argument("--paced", action="store_true", help="play a local file at wall-clock speed, like a real stream")
    parser.add_argument("--window", type=float, default=6.0, help="live analysis window, seconds")
    parser.add_argument("--ocr-interval", type=float, default=0.75, help="seconds between OCR samples")
    parser.add_argument("--max-height", type=int, default=0, help="cap output height for live, 0 keeps source size")
    parser.add_argument("--preset", default="ultrafast", help="x264 preset for live")
    parser.add_argument("--no-speech", action="store_true", help="live: skip the beep path")
    parser.add_argument("--no-visual", action="store_true", help="live: skip the blur path")
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

    settings = load_settings()
    if args.ocr_band:
        settings.ocr_band = args.ocr_band

    if args.rtmp_in or args.simulate:
        return do_live(args, settings)

    if not args.file:
        parser.error("give a file, or use --demo")
    source = Path(args.file).expanduser()
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2

    if args.no_ocr:
        settings.ocr_enabled = False
    if args.report_only:
        settings.report_only = True
    if args.model:
        settings.model = args.model
    if args.blur:
        settings.blur_mode = args.blur

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
