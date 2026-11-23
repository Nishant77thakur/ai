"""AI voice-over with per-word timings, so captions land exactly on each spoken word.

The default voice uses Microsoft Edge's free online text-to-speech (the `edge-tts`
package). It needs an internet connection. The `silent` voice makes no sound and
estimates timings; it exists for tests and for previewing a layout offline.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .script import Line

SAMPLE_RATE = 44100
GAP_AFTER_LINE = 0.35  # seconds of breath between sentences


@dataclass
class TimedWord:
    line: int  # -1 = hook, 0..n-1 = script lines
    index: int  # position of the word inside its line
    start: float
    end: float


@dataclass
class Narration:
    audio: np.ndarray  # mono float32, SAMPLE_RATE
    words: list[TimedWord]
    line_spans: dict[int, tuple[float, float]]  # line -> (start, end) seconds

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def decode(path: Path) -> np.ndarray:
    """Any audio file -> mono float32 at SAMPLE_RATE, via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def align(n_words: int, boundaries: list[tuple[float, float]], duration: float,
          weights: list[int]) -> list[tuple[float, float]]:
    """Give each displayed word a (start, end).

    Uses the voice engine's word boundaries when they match the displayed words one to
    one; otherwise spreads the spoken time over the words by their length, so captions
    stay in sync even when the engine splits or joins words (numbers, contractions).
    """
    if len(boundaries) == n_words and n_words:
        return boundaries
    if boundaries:
        t0, t1 = boundaries[0][0], boundaries[-1][1]
    else:
        t0, t1 = 0.0, duration
    total = max(1, sum(weights))
    spans, t = [], t0
    for w in weights:
        d = (t1 - t0) * w / total
        spans.append((t, t + d))
        t += d
    return spans


async def _edge(text: str, voice: str, rate: str, mp3: Path) -> list[tuple[float, float]]:
    import edge_tts  # imported here so the silent voice works without it

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    kwargs = {"rate": rate, "boundary": "WordBoundary"}
    if proxy:
        kwargs["proxy"] = proxy
    comm = edge_tts.Communicate(text, voice, **kwargs)
    bounds = []
    with open(mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7  # 100-nanosecond units
                bounds.append((start, start + chunk["duration"] / 1e7))
    return bounds


def _speak_line(line: Line, voice: str, rate: str, tmp: Path, i: int):
    weights = [max(1, len(w.text)) for w in line.words]
    if voice == "silent":
        # Roughly 2.6 words a second, the pace of calm narration.
        dur = 0.3 + len(line.words) / 2.6
        audio = np.zeros(int(dur * SAMPLE_RATE), dtype=np.float32)
        return audio, align(len(line.words), [], dur, weights)
    mp3 = tmp / f"line{i}.mp3"
    bounds = asyncio.run(_edge(line.spoken, voice, rate, mp3))
    if not mp3.exists() or mp3.stat().st_size == 0:
        raise RuntimeError("The voice service returned no audio. Check your internet connection.")
    audio = decode(mp3)
    return audio, align(len(line.words), bounds, len(audio) / SAMPLE_RATE, weights)


def narrate(hook: Line, lines: list[Line], voice: str, rate: str = "-5%") -> Narration:
    """Speak the hook then every line, one after another, and time every word."""
    pieces, words, spans = [], [], {}
    t = 0.0
    with tempfile.TemporaryDirectory() as d:
        for li, line in [(-1, hook)] + list(enumerate(lines)):
            audio, timing = _speak_line(line, voice, rate, Path(d), li + 1)
            for wi, (s, e) in enumerate(timing):
                words.append(TimedWord(li, wi, t + s, t + e))
            dur = len(audio) / SAMPLE_RATE
            spans[li] = (t, t + dur)
            gap = np.zeros(int(GAP_AFTER_LINE * SAMPLE_RATE), dtype=np.float32)
            pieces += [audio, gap]
            t += dur + GAP_AFTER_LINE
    return Narration(np.concatenate(pieces) if pieces else np.zeros(1, np.float32), words, spans)
