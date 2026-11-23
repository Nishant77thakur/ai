"""Command line: turn one or many script files into finished Shorts.

    python -m shorts_editor make scripts/before_bed.yaml
    python -m shorts_editor make scripts/*.yaml --out-dir videos
    python -m shorts_editor preview scripts/before_bed.yaml --at 0 5 12
    python -m shorts_editor voices --lang en
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import script as script_mod


def _bar(i: int, total: int) -> None:
    pct = int(100 * i / max(1, total))
    sys.stdout.write(f"\r  rendering {pct:3d}%")
    sys.stdout.flush()


def cmd_make(args) -> int:
    from .pipeline import make

    failed = 0
    for path in args.scripts:
        started = time.time()
        try:
            short = script_mod.load(path)
            out = Path(args.out_dir) / (Path(path).stem + ".mp4")
            print(f"{path}: {short.title}")
            res = make(short, out, voice=args.voice, fps=args.fps, progress=_bar)
            print(f"\r  done in {time.time() - started:.0f}s -> {res.video} ({res.duration:.1f}s)")
            print(f"  captions: {res.captions}\n  cover:    {res.cover}")
        except Exception as exc:  # keep going with the other scripts in a batch
            failed += 1
            print(f"\n  FAILED: {exc}", file=sys.stderr)
    return 1 if failed else 0


def cmd_preview(args) -> int:
    from .render import Renderer
    from .tts import narrate

    short = script_mod.load(args.script)
    nar = narrate(short.hook, short.lines, args.voice or short.voice, short.rate)
    r = Renderer(short, nar)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t in args.at:
        p = out / f"{Path(args.script).stem}-{t:05.1f}s.png"
        r.frame(t).save(p)
        print(p)
    r.bg.close()
    print(f"video length: {r.plan.duration:.1f}s")
    return 0


def cmd_voices(args) -> int:
    import asyncio

    import edge_tts

    voices = asyncio.run(edge_tts.list_voices())
    for v in voices:
        if v["Locale"].lower().startswith(args.lang.lower()):
            print(f'{v["ShortName"]:32} {v["Gender"]:7} {v["Locale"]}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="shorts_editor", description="Script in, finished YouTube Short out.")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="render one or more scripts to MP4")
    m.add_argument("scripts", nargs="+")
    m.add_argument("--out-dir", default="output")
    m.add_argument("--voice", help="override the script's voice (e.g. en-IN-PrabhatNeural, or 'silent')")
    m.add_argument("--fps", type=int, default=30)
    m.set_defaults(func=cmd_make)

    pv = sub.add_parser("preview", help="save still frames to check the look quickly")
    pv.add_argument("script")
    pv.add_argument("--at", type=float, nargs="+", default=[0.0, 4.0, 10.0])
    pv.add_argument("--out-dir", default="output/preview")
    pv.add_argument("--voice")
    pv.set_defaults(func=cmd_preview)

    v = sub.add_parser("voices", help="list the AI voices for a language")
    v.add_argument("--lang", default="en")
    v.set_defaults(func=cmd_voices)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
