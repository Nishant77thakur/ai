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
