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


def _norm(text: str) -> str:
    """Compare words loosely: drop punctuation and the marks speech recognition often gets wrong."""
    drop = set("।,.!?;:'\"“”‘’()-–—…") | {"़", "ँ", "ं", "्"}  # nukta, chandrabindu, anusvara, virama
    t = "".join(ch for ch in text.lower() if ch not in drop)
    # Long and short vowel signs are often confused: treat them as the same.
    return t.translate(str.maketrans({"ी": "ि", "ू": "ु", "आ": "अ", "ई": "इ", "ऊ": "उ"}))


def match_words(script_words: list[str], heard: list[str]) -> list[int | None]:
    """For each script word, the index of the heard word it matches (or None).

    A global alignment that lets extra heard words (spoken line numbers, ad-libs) be
    skipped for free, and scores pairs by how alike they are, so recognition slips like
    'अर्जून' for 'अर्जुन' still match.
    """
    from difflib import SequenceMatcher

    a, b = [_norm(w) for w in script_words], [_norm(w) for w in heard]
    n, m = len(a), len(b)
    skip_script = -0.4
    score = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 = match, 2 = skip script word, 3 = skip heard word
    for i in range(1, n + 1):
        score[i, 0], move[i, 0] = score[i - 1, 0] + skip_script, 2
    for j in range(1, m + 1):
        move[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = SequenceMatcher(None, a[i - 1], b[j - 1]).ratio() if a[i - 1] and b[j - 1] else 0.0
            options = (score[i - 1, j - 1] + (sim - 0.45) * 2, score[i - 1, j] + skip_script, score[i, j - 1])
            k = int(np.argmax(options))
            score[i, j], move[i, j] = options[k], k + 1
    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        mv = move[i, j]
        if mv == 1:
            if SequenceMatcher(None, a[i - 1], b[j - 1]).ratio() >= 0.45:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    return out


def transcribe(raw: np.ndarray, prompt: str, language: str | None) -> list[tuple[str, float, float]]:
    """Heard words with (start, end) seconds, using faster-whisper when it is installed."""
    from faster_whisper import WhisperModel  # optional dependency

    import os

    model = WhisperModel(os.environ.get("SHORTS_WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    step = SAMPLE_RATE // 16000 if SAMPLE_RATE % 16000 == 0 else None
    audio16 = raw[::step] if step else np.interp(
        np.arange(0, len(raw), SAMPLE_RATE / 16000), np.arange(len(raw)), raw).astype(np.float32)
    segments, _ = model.transcribe(audio16, language=language, word_timestamps=True,
                                   initial_prompt=prompt[:600], beam_size=5)
    return [(w.word.strip(), w.start, w.end) for s in segments for w in s.words]


def _timed_lines(every, heard):
    """Per line: its word timings in the recording, from the matched heard words."""
    flat = [(li, wi, w.text) for li, line in every for wi, w in enumerate(line.words)]
    match = match_words([x[2] for x in flat], [h[0] for h in heard])
    if sum(m is not None for m in match) < 0.5 * len(flat):
        raise ValueError("The recording does not seem to match the script's words.")
    times = [(heard[m][1], heard[m][2]) if m is not None else None for m in match]
    # Fill words the recogniser missed by interpolating between their neighbours.
    for k, tv in enumerate(times):
        if tv is None:
            prev = next((times[p][1] for p in range(k - 1, -1, -1) if times[p]), 0.0)
            nxt = next((times[q][0] for q in range(k + 1, len(times)) if times[q]), prev + 0.4)
            times[k] = (prev, max(prev, nxt))
    out, k = [], 0
    for li, line in every:
        out.append(times[k:k + len(line.words)])
        k += len(line.words)
    # Heard-word boundaries let a cut stop before anything spoken that is not in the script.
    used = sorted({m for m in match if m is not None})
    return out, match, used


def narrate_recording(path: str | Path, hook: Line, lines: list[Line], pause: float = 0.35,
                      language: str | None = None, use_transcript: bool = True) -> Narration:
    raw = decode(Path(path))
    every = [(-1, hook)] + list(enumerate(lines))
    lead = 0.12  # keep a little before and after so soft syllables are not clipped
    word_times = None
    if use_transcript:
        try:
            prompt = " ".join(line.spoken for _, line in every)
            heard = transcribe(raw, prompt, language)
            word_times, _, _ = _timed_lines(every, heard)
        except (ImportError, ValueError):
            # Speech recognition not installed, or the words did not match: use the pauses.
            word_times = None
    if word_times is not None:
        total = len(raw) / SAMPLE_RATE
        segments = []
        for wt in word_times:
            s, e = wt[0][0], wt[-1][1]
            # Never reach into a neighbouring line or into unscripted words (like a spoken number).
            prev_end = max((h[2] for h in heard if h[2] <= s + 0.01), default=0.0)
            next_start = min((h[1] for h in heard if h[1] >= e - 0.01), default=total)
            segments.append((max(prev_end, s - lead), min(next_start, e + lead)))
    else:
        segments = [(s - lead, e + lead) for s, e in find_segments(raw, len(every))]
    pieces, words, spans, t = [], [], {}, 0.0
    for k, ((li, line), (s, e)) in enumerate(zip(every, segments)):
        a = int(max(0.0, s) * SAMPLE_RATE)
        b = int(min(len(raw) / SAMPLE_RATE, e) * SAMPLE_RATE)
        chunk = raw[a:b]
        dur = len(chunk) / SAMPLE_RATE
        if word_times is not None:
            timing = [(max(0.0, ws - s), max(0.0, we - s)) for ws, we in word_times[k]]
        else:
            weights = [max(1, len(w.text)) for w in line.words]
            timing = align(len(line.words), [(lead, max(lead, dur - lead))], dur, weights)
        for wi, (ws, we) in enumerate(timing):
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
