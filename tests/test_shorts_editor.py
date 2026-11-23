import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shorts_editor import audio, script
from shorts_editor.pipeline import make, srt_time
from shorts_editor.render import CAPTION_FONT, Renderer, W, layout, plan
from shorts_editor.tts import SAMPLE_RATE, align, narrate


def write_script(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(
        "hook: Do this *before bed*\n"
        "lines:\n"
        "  - Your brain decides tomorrow... *tonight.*\n"
        "  - It keeps what *matters*.\n"
        "end_card:\n  title: Sleep with *Impact*\n" + extra,
        encoding="utf-8",
    )
    return p


def test_highlighted_words_are_parsed():
    line = script.parse_line("So the last *ten minutes* before bed")
    assert [w.text for w in line.words] == ["So", "the", "last", "ten", "minutes", "before", "bed"]
    assert [w.accent for w in line.words] == [False, False, False, True, True, False, False]
    with pytest.raises(ValueError):
        script.parse_line("an *unclosed highlight")


def test_lone_punctuation_joins_the_word_before():
    line = script.parse_line("यही है महाभारत की *सीख*।")
    assert [w.text for w in line.words][-1] == "सीख।" and line.words[-1].accent


def test_hindi_script_renders_with_chakra(tmp_path):
    p = tmp_path / "hi.yaml"
    p.write_text("hook: महाभारत का *सबसे बड़ा* सबक\nfonts: hindi\nlogo: chakra\n"
                 "lines:\n  - तब *श्रीकृष्ण* ने ज्ञान दिया।\n", encoding="utf-8")
    s = script.load(p)
    r = Renderer(s, narrate(s.hook, s.lines, "silent"), fps=10)
    assert "Mukta" in r.f_head
    assert np.asarray(r.frame(0.0)).shape == (1920, 1080, 3)
    r.bg.close()
    bad = tmp_path / "bad.yaml"
    bad.write_text("hook: x\nlogo: star\nlines: [y]\n")
    with pytest.raises(ValueError, match="logo"):
        script.load(bad)


def test_load_requires_hook_and_lines(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("lines: [hello]\n")
    with pytest.raises(ValueError, match="hook"):
        script.load(p)
    s = script.load(write_script(tmp_path))
    assert s.title == "Do this before bed" and len(s.lines) == 2
    assert s.brand["accent"] == "#FF5A1F"


def test_align_uses_engine_timings_or_spreads_by_length():
    exact = [(0.0, 0.3), (0.3, 0.9)]
    assert align(2, exact, 1.0, [2, 5]) == exact
    spread = align(3, [(0.0, 0.5), (0.5, 1.0)], 1.0, [1, 1, 2])  # engine split differently
    assert spread[0] == (0.0, 0.25) and spread[-1][1] == pytest.approx(1.0)


def test_silent_narration_times_every_word_in_order():
    hook = script.parse_line("Do this before bed")
    lines = [script.parse_line("One two three"), script.parse_line("Four five")]
    nar = narrate(hook, lines, "silent")
    starts = [w.start for w in nar.words]
    assert starts == sorted(starts) and len(nar.words) == 4 + 3 + 2
    assert nar.line_spans[-1][0] == 0 and nar.line_spans[1][1] <= nar.duration


def test_layout_keeps_words_inside_the_frame():
    line = script.parse_line("Write the first thing you'll do tomorrow before you sleep tonight")
    placed = layout(line, str(CAPTION_FONT), 104, 920, 960)
    assert len(placed) == len(line.words)
    assert len({round(p.cy) for p in placed}) > 1  # wrapped onto several rows
    assert all(60 < p.cx < W - 60 for p in placed)


def test_lines_never_overlap_and_end_card_follows(tmp_path):
    s = script.load(write_script(tmp_path))
    nar = narrate(s.hook, s.lines, "silent")
    pl = plan(s, nar)
    for (a0, a1), (b0, b1) in zip(pl.line_windows, pl.line_windows[1:]):
        assert a1 <= b0
    assert pl.line_windows[0][0] >= pl.hook_end
    assert pl.end_start >= pl.line_windows[-1][1] and pl.duration > pl.end_start


def test_first_frame_shows_the_hook(tmp_path):
    s = script.load(write_script(tmp_path))
    nar = narrate(s.hook, s.lines, "silent")
    r = Renderer(s, nar, fps=10)
    first = np.asarray(r.frame(0.0)).astype(int)
    # The accent-coloured headline is on screen from the very first frame.
    accent = (abs(first[..., 0] - 255) < 30) & (abs(first[..., 1] - 90) < 40) & (first[..., 2] < 80)
    assert accent[600:1100].sum() > 5000
    r.bg.close()


def test_music_ducks_under_the_voice():
    sec = 4.0
    voice = np.zeros(int(sec * SAMPLE_RATE), np.float32)
    music = np.ones(int(sec * SAMPLE_RATE), np.float32) * 0.5
    out = audio.mix(voice, music, sec, [(1.0, 2.0)], music_volume=1.0)
    speaking = np.abs(out[int(1.4 * SAMPLE_RATE):int(1.6 * SAMPLE_RATE)]).mean()
    quiet = np.abs(out[int(0.6 * SAMPLE_RATE):int(0.7 * SAMPLE_RATE)]).mean()
    assert speaking < quiet * 0.6


def test_ambient_pad_is_deterministic_and_bounded():
    a, b = audio.ambient_pad(3.0), audio.ambient_pad(3.0)
    assert np.array_equal(a, b) and np.abs(a).max() <= 1.0 and len(a) == 3 * SAMPLE_RATE


def test_srt_time_format():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(83.4567) == "00:01:23,457"


def test_full_render_produces_a_vertical_video_with_sound(tmp_path):
    s = script.load(write_script(tmp_path))
    res = make(s, tmp_path / "out.mp4", voice="silent", fps=8)
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(res.video)],
        check=True, capture_output=True).stdout)
    kinds = {st["codec_type"]: st for st in info["streams"]}
    assert kinds["video"]["width"] == 1080 and kinds["video"]["height"] == 1920
    assert "audio" in kinds
    assert float(info["format"]["duration"]) == pytest.approx(res.duration, abs=0.6)
    assert "Do this before bed" in res.captions.read_text()
    assert res.cover.exists()


def test_image_background(tmp_path):
    from PIL import Image

    Image.new("RGB", (640, 480), (40, 80, 120)).save(tmp_path / "bg.jpg")
    s = script.load(write_script(tmp_path, "background: bg.jpg\nmusic: none\n"))
    r = Renderer(s, narrate(s.hook, s.lines, "silent"), fps=10)
    frame = np.asarray(r.frame(5.0))
    assert frame.shape == (1920, 1080, 3)
    assert frame[1700:1800, :, 2].mean() > frame[1700:1800, :, 0].mean()  # the blue photo shows through


def test_voice_polish_keeps_length_and_changes_the_sound():
    from shorts_editor.tts import polish

    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = polish(tone, "warm")
    assert len(out) == len(tone) and not np.allclose(out, tone)
    assert polish(tone, "none") is tone
    with pytest.raises(ValueError):
        polish(tone, "robot")


def _fake_recording(path, parts, gaps, noise=0.002):
    """Tones standing in for spoken lines, separated by pauses, over faint room noise."""
    rng = np.random.default_rng(0)
    out = [np.zeros(int(0.8 * SAMPLE_RATE), np.float32)]
    for i, dur in enumerate(parts):
        t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
        # Short dips inside a "sentence", like breaths between words.
        env = (np.sin(2 * np.pi * 3 * t) > -0.95).astype(np.float32)
        out.append((0.3 * np.sin(2 * np.pi * 200 * t) * env).astype(np.float32))
        out.append(np.zeros(int((gaps[i] if i < len(gaps) else 1.0) * SAMPLE_RATE), np.float32))
    a = np.concatenate(out)
    a += (noise * rng.standard_normal(len(a))).astype(np.float32)
    audio.write_wav(path, a)


def test_own_recording_is_split_into_lines_and_pauses_shortened(tmp_path):
    from shorts_editor.recording import narrate_recording

    rec = tmp_path / "me.wav"
    # hook + 2 lines; one long breath inside line 1 must not count as a line break
    _fake_recording(rec, [1.5, 3.0, 2.0], [1.4, 1.2])
    hook = script.parse_line("Do this before bed")
    lines = [script.parse_line("One two three four"), script.parse_line("Five six")]
    nar = narrate_recording(rec, hook, lines, pause=0.3, use_transcript=False)
    spans = [nar.line_spans[k] for k in (-1, 0, 1)]
    assert [round(e - s, 1) for s, e in spans] == pytest.approx([1.7, 3.2, 2.2], abs=0.15)
    # The 0.8 s lead-in and the long pauses are gone: 1.5+3+2 of speech plus short pauses.
    assert nar.duration == pytest.approx(7.1 + 3 * 0.3, abs=0.3)
    assert len(nar.words) == 4 + 4 + 2


def test_recording_with_a_missing_line_explains_what_to_do(tmp_path):
    from shorts_editor.recording import narrate_recording

    rec = tmp_path / "me.wav"
    _fake_recording(rec, [1.5, 2.0], [1.2])
    lines = [script.parse_line("a b"), script.parse_line("c d")]
    with pytest.raises(ValueError, match="Pause for about a second"):
        narrate_recording(rec, script.parse_line("hook"), lines, use_transcript=False)


def test_script_with_narration_defaults_to_clean_voice(tmp_path):
    s = script.load(write_script(tmp_path, "narration: me.m4a\n"))
    assert s.narration == "me.m4a" and s.voice_fx == "clean"


def test_elevenlabs_character_timings_become_word_timings():
    from shorts_editor.tts import words_from_characters

    text = "तब श्रीकृष्ण ने"
    starts = [i * 0.1 for i in range(len(text))]
    ends = [s + 0.1 for s in starts]
    words = words_from_characters(text, starts, ends)
    assert len(words) == 3
    assert words[0] == (0.0, pytest.approx(0.2)) and words[2][1] == pytest.approx(ends[-1])


def test_elevenlabs_voice_is_used_with_a_key(tmp_path, monkeypatch):
    import base64
    import io
    import urllib.request

    from shorts_editor import tts

    tone = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=1", str(tone)], check=True)
    sent = {}

    def fake_urlopen(req, timeout=0):
        sent["url"], sent["key"] = req.full_url, req.headers.get("Xi-api-key")
        text = json.loads(req.data)["text"]
        n = len(text)
        payload = {"audio_base64": base64.b64encode(tone.read_bytes()).decode(),
                   "alignment": {"characters": list(text),
                                 "character_start_times_seconds": [i / n for i in range(n)],
                                 "character_end_times_seconds": [(i + 1) / n for i in range(n)]}}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    nar = tts.narrate(script.parse_line("Do this"), [script.parse_line("one two three")], "elevenlabs:abc123")
    assert sent["url"].endswith("/abc123/with-timestamps") and sent["key"] == "test-key"
    assert len(nar.words) == 5 and nar.duration > 2.0
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        tts.narrate(script.parse_line("x"), [], "elevenlabs:abc123")


def test_word_matching_skips_spoken_line_numbers():
    from shorts_editor.recording import match_words

    script_words = ["सामने", "उसके", "अपने", "ही", "गुरु"]
    heard = ["तीन,", "सामने", "उसके", "अपने", "ही", "गुरू"]
    assert match_words(script_words, heard) == [1, 2, 3, 4, 5]
