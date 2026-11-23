"""Script in, finished Short out: voice, music, motion graphics, captions file and cover."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .render import Renderer
from .script import Short, resolve
from .tts import Narration, narrate, polish


@dataclass
class Result:
    video: Path
    captions: Path
    cover: Path
    duration: float


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(short: Short, narration: Narration, path: Path) -> None:
    """One subtitle per spoken sentence, for uploading to YouTube alongside the video."""
    blocks = []
    for n, (li, line) in enumerate([(-1, short.hook)] + list(enumerate(short.lines)), start=1):
        s, e = narration.line_spans[li]
        blocks.append(f"{n}\n{srt_time(s)} --> {srt_time(e)}\n{line.spoken}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def make(short: Short, out: Path, voice: str | None = None, fps: int = 30, progress=None) -> Result:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    narration = narrate(short.hook, short.lines, voice or short.voice, short.rate, short.pitch, short.pause)
    narration.audio = polish(narration.audio, short.voice_fx)
    renderer = Renderer(short, narration, fps=fps)
    duration = renderer.plan.duration

    if short.music == "none":
        music = None
    elif short.music == "auto":
        music = audio.ambient_pad(duration)
    else:
        music = audio.load_music(resolve(short, short.music), duration)
    spans = [narration.line_spans[k] for k in sorted(narration.line_spans)]
    sfx = [renderer.plan.hook_end, renderer.plan.end_start]
    mixed = audio.mix(narration.audio, music, duration, spans, short.music_volume, sfx)

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "mix.wav"
        audio.write_wav(wav, mixed)
        renderer.encode(wav, out, progress)

    captions = out.with_suffix(".srt")
    write_srt(short, narration, captions)
    cover = out.with_name(out.stem + "-cover.png")
    Renderer(short, narration, fps=fps).frame(0.0).save(cover)  # first frame = the hook
    return Result(out, captions, cover, duration)
