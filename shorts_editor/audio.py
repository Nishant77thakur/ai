"""Background music and the final mix: voice on top, music ducked underneath."""
from __future__ import annotations

import numpy as np

from .tts import SAMPLE_RATE, decode


def ambient_pad(seconds: float, seed: int = 7) -> np.ndarray:
    """A calm, royalty-free synth pad generated on the spot (no licence worries).

    Four slow chords, each a stack of soft sine voices with a gentle swell, plus a
    quiet low drone. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    # A minor-ish, sleepy progression: Am9 - Fmaj7 - Cmaj7 - Em7 (Hz)
    chords = [
        [110.0, 220.0, 261.63, 329.63, 493.88],
        [87.31, 174.61, 220.0, 261.63, 329.63],
        [65.41, 130.81, 196.0, 246.94, 329.63],
        [82.41, 164.81, 196.0, 246.94, 293.66],
    ]
    bar = 6.0  # seconds per chord
    out = np.zeros(n, dtype=np.float64)
    for k in range(int(np.ceil(seconds / bar)) + 1):
        chord = chords[k % len(chords)]
        start = k * bar - 1.0  # overlap so chords cross-fade
        seg = (t >= start) & (t < start + bar + 2.0)
        if not seg.any():
            continue
        local = t[seg] - start
        env = np.sin(np.clip(local / (bar + 2.0), 0, 1) * np.pi) ** 2
        tone = np.zeros(local.shape)
        for j, f in enumerate(chord):
            detune = 1 + rng.uniform(-0.002, 0.002)
            tone += np.sin(2 * np.pi * f * detune * local + rng.uniform(0, 6.28)) / (1.6 + j * 0.6)
        out[seg] += tone * env
    drone = 0.25 * np.sin(2 * np.pi * 55.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * t))
    out += drone
    out /= max(1e-9, np.abs(out).max())
    return (out * 0.8).astype(np.float32)


def whoosh(seconds: float = 0.5, seed: int = 3) -> np.ndarray:
    """A soft filtered-noise swish for scene changes."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    noise = rng.standard_normal(n)
    # Simple one-pole low-pass whose cutoff sweeps up then down.
    sweep = np.sin(np.linspace(0, np.pi, n))
    out = np.zeros(n)
    y = 0.0
    for i in range(n):
        a = 0.02 + 0.25 * sweep[i]
        y += a * (noise[i] - y)
        out[i] = y
    out *= sweep
    out /= max(1e-9, np.abs(out).max())
    return (out * 0.35).astype(np.float32)


def load_music(path, seconds: float) -> np.ndarray:
    """Decode a music file and loop it to cover the video."""
    music = decode(path)
    if len(music) == 0:
        raise ValueError(f"No audio found in {path}")
    need = int(seconds * SAMPLE_RATE)
    reps = int(np.ceil(need / len(music)))
    return np.tile(music, reps)[:need]


def mix(voice: np.ndarray, music: np.ndarray | None, seconds: float, speech_spans,
        music_volume: float = 0.22, sfx_times=()) -> np.ndarray:
    """Voice at full level, music ducked to a third while someone is speaking."""
    n = int(seconds * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    v = voice[:n]
    out[: len(v)] += v
    if music is not None and len(music):
        m = np.zeros(n, dtype=np.float32)
        m[: min(n, len(music))] = music[:n]
        gain = np.full(n, 1.0, dtype=np.float32)
        for s, e in speech_spans:
            a, b = int(max(0, s - 0.15) * SAMPLE_RATE), int(min(seconds, e + 0.25) * SAMPLE_RATE)
            gain[a:b] = 0.35
        # Smooth the ducking so the music breathes instead of jumping.
        k = int(0.25 * SAMPLE_RATE)
        gain = np.convolve(gain, np.ones(k) / k, mode="same").astype(np.float32)
        fade = int(min(2.0, seconds / 4) * SAMPLE_RATE)
        env = np.ones(n, dtype=np.float32)
        env[:int(0.3 * SAMPLE_RATE)] = np.linspace(0, 1, int(0.3 * SAMPLE_RATE))
        env[-fade:] = np.linspace(1, 0, fade)
        out += m * gain * env * music_volume
    sw = whoosh()
    for t in sfx_times:
        a = int(t * SAMPLE_RATE)
        b = min(n, a + len(sw))
        if 0 <= a < n:
            out[a:b] += sw[: b - a] * 0.5
    peak = float(np.abs(out).max())
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def write_wav(path, audio: np.ndarray) -> None:
    import wave

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
