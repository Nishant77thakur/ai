"""Use your own recorded voice instead of an AI voice.

Record the hook and every line in order, pausing about a second between lines. The
editor then:
  1. trims the silence at the start and end,
  2. finds the line breaks (the longest pauses, one fewer than the number of lines),
  3. shortens every pause to the script's `pause` length, so the video moves briskly,
  4. times each line's words across its stretch of speech, so captions follow your voice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .script import Line
from .tts import SAMPLE_RATE, Narration, TimedWord, align, decode

FRAME = 0.02  # seconds per loudness measurement
MIN_GAP = 0.25  # shorter dips are breaths inside a sentence, not line breaks


def speech_mask(audio: np.ndarray) -> np.ndarray:
    """True for each 20 ms frame that holds speech, judged against the recording's own levels."""
    n = int(FRAME * SAMPLE_RATE)
    frames = len(audio) // n
    if frames == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(np.mean(audio[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    loud = np.percentile(db, 95)
    floor = np.percentile(db, 10)
    threshold = max(floor + 6, loud - 36)  # above the room noise, within 36 dB of speech
    mask = db > threshold
    # Close tiny holes (consonants, plosives) so words are not split apart.
    hole = int(0.08 / FRAME)
    i = 0
    while i < len(mask):
        if not mask[i]:
            j = i
            while j < len(mask) and not mask[j]:
                j += 1
            if 0 < i and j < len(mask) and j - i <= hole:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def find_segments(audio: np.ndarray, count: int) -> list[tuple[float, float]]:
    """Split the recording into `count` stretches of speech, at its longest pauses."""
    mask = speech_mask(audio)
    if not mask.any():
        raise ValueError("No speech found in the recording. Is it silent, or very quiet?")
    gaps = []  # (length, start_frame, end_frame) of each silent run inside the speech
    first, last = int(np.argmax(mask)), len(mask) - int(np.argmax(mask[::-1]))
    i = first
    while i < last:
        if not mask[i]:
            j = i
            while j < last and not mask[j]:
                j += 1
            if (j - i) * FRAME >= MIN_GAP:
                gaps.append((j - i, i, j))
            i = j
        else:
            i += 1
    if len(gaps) < count - 1:
        raise ValueError(
            f"Found {len(gaps) + 1} parts in the recording but the script has {count} lines "
            f"(hook included). Pause for about a second between lines and record again, "
            f"or check that every line was read.")
    breaks = sorted(sorted(gaps, reverse=True)[: count - 1], key=lambda g: g[1])
    bounds, start = [], first
    for _, gs, ge in breaks:
        bounds.append((start * FRAME, gs * FRAME))
        start = ge
    bounds.append((start * FRAME, last * FRAME))
    return bounds


def narrate_recording(path: str | Path, hook: Line, lines: list[Line], pause: float = 0.35) -> Narration:
    raw = decode(Path(path))
    every = [(-1, hook)] + list(enumerate(lines))
    segments = find_segments(raw, len(every))
    pieces, words, spans, t = [], [], {}, 0.0
    lead = 0.12  # keep a little before and after so soft syllables are not clipped
    for (li, line), (s, e) in zip(every, segments):
        a = int(max(0.0, s - lead) * SAMPLE_RATE)
        b = int(min(len(raw) / SAMPLE_RATE, e + lead) * SAMPLE_RATE)
        chunk = raw[a:b]
        dur = len(chunk) / SAMPLE_RATE
        weights = [max(1, len(w.text)) for w in line.words]
        for wi, (ws, we) in enumerate(align(len(line.words), [(lead, max(lead, dur - lead))], dur, weights)):
            words.append(TimedWord(li, wi, t + ws, t + we))
        spans[li] = (t, t + dur)
        # Fade the edges of each cut so there are no clicks.
        f = min(len(chunk) // 4, int(0.01 * SAMPLE_RATE))
        if f > 0:
            chunk = chunk.copy()
            chunk[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            chunk[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        pieces += [chunk, np.zeros(int(pause * SAMPLE_RATE), np.float32)]
        t += dur + pause
    return Narration(np.concatenate(pieces), words, spans)
