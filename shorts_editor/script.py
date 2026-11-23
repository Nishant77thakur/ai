"""Loading a Short's script file (YAML) into plain data the editor works from.

A script looks like:

    title: Your brain decides tomorrow, tonight
    hook: Do this *before bed*
    lines:
      - Your brain decides tomorrow... *tonight.*
      - While you sleep, your brain *sorts* through your day.
    end_card:
      title: Sleep with *Impact*
      subtitle: Follow for one idea every night

Words wrapped in *asterisks* are highlighted in the accent colour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_BRAND = {
    "name": "Sleep With Impact",
    "background": "#07080D",
    "text": "#F4EDE4",
    "accent": "#FF5A1F",
    "muted": "#A39C94",
    "sky_mid": "#0B0E1C",
    "sky_bottom": "#1B100B",
    "particles": "#F4EDE4",
}


@dataclass
class Word:
    text: str
    accent: bool = False


@dataclass
class Line:
    """One sentence of the script: what is spoken and how it is shown."""

    words: list[Word]

    @property
    def spoken(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Short:
    title: str
    hook: Line
    lines: list[Line]
    end_title: Line
    end_subtitle: str
    voice: str = "en-US-AndrewNeural"
    rate: str = "-5%"
    pitch: str = "+0Hz"  # e.g. "-8Hz" for a deeper voice
    background: str = "stars"
    music: str = "auto"
    music_volume: float = 0.22
    fonts: str = "latin"  # "latin" or "hindi" (Devanagari)
    logo: str = "moon"  # "moon", "chakra" or "none"
    brand: dict = field(default_factory=lambda: dict(DEFAULT_BRAND))
    base_dir: Path = Path(".")


def parse_line(text: str) -> Line:
    """Split a line into words, marking *highlighted* phrases as accent words."""
    text = str(text).strip()
    if not text:
        raise ValueError("A script line cannot be empty")
    words: list[Word] = []
    # Alternate between plain and *accent* chunks.
    for i, chunk in enumerate(re.split(r"\*", text)):
        for token in chunk.split():
            if words and all(not ch.isalnum() for ch in token):
                # Punctuation on its own (like "।" after *word*) belongs to the word before.
                words[-1].text += token
                continue
            words.append(Word(token, accent=(i % 2 == 1)))
    if text.count("*") % 2:
        raise ValueError(f"Unbalanced * in line: {text!r}")
    return Line(words)


def load(path: str | Path) -> Short:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the script must be a mapping of keys like hook:, lines:")
    for key in ("hook", "lines"):
        if not data.get(key):
            raise ValueError(f"{path}: missing '{key}'")
    if not isinstance(data["lines"], list):
        raise ValueError(f"{path}: 'lines' must be a list")
    brand = dict(DEFAULT_BRAND)
    brand.update(data.get("brand") or {})
    end = data.get("end_card") or {}
    return Short(
        title=str(data.get("title") or data["hook"]).replace("*", ""),
        hook=parse_line(data["hook"]),
        lines=[parse_line(x) for x in data["lines"]],
        end_title=parse_line(end.get("title") or brand["name"]),
        end_subtitle=str(end.get("subtitle", "Follow for more")),
        voice=str(data.get("voice", "en-US-AndrewNeural")),
        rate=str(data.get("rate", "-5%")),
        pitch=str(data.get("pitch", "+0Hz")),
        background=str(data.get("background", "stars")),
        music=str(data.get("music", "auto")),
        music_volume=float(data.get("music_volume", 0.22)),
        fonts=str(data.get("fonts", "latin")),
        logo=_choice(data.get("logo", "moon"), ("moon", "chakra", "none"), "logo"),
        brand=brand,
        base_dir=path.parent,
    )


def _choice(value, options, key):
    value = str(value)
    if value not in options:
        raise ValueError(f"{key} must be one of {', '.join(options)}, got {value!r}")
    return value


def resolve(short: Short, value: str) -> Path:
    """A media path from the script, relative to the script file."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (short.base_dir / p)
