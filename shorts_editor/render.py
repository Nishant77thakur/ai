"""Draws every frame of the Short and encodes it with the mixed audio.

Layers, bottom to top:
  background  - generated night sky with twinkling stars, or your image/video with a slow zoom
  shade       - a dark gradient so text always reads
  logo        - the crescent moon or a turning chakra, drifting; it moves to the centre for the end card
  hook        - big headline on the very first frame, while the hook is spoken
  captions    - each line's words pop in exactly when they are spoken, the spoken word lit up
  end card    - channel name and a follow line
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .script import Line, Short, resolve
from .tts import Narration

W, H = 1080, 1920
FONTS = Path(__file__).parent / "assets" / "fonts"
HEADLINE_FONT = FONTS / "Anton-Regular.ttf"
CAPTION_FONT = FONTS / "Inter-ExtraBold.ttf"
BODY_FONT = FONTS / "Inter-Medium.ttf"
# headline, caption, body font and headline size for each writing system
FONT_SETS = {
    "latin": (HEADLINE_FONT, CAPTION_FONT, BODY_FONT, 190),
    "hindi": (FONTS / "Mukta-ExtraBold.ttf", FONTS / "Mukta-ExtraBold.ttf", FONTS / "Mukta-SemiBold.ttf", 172),
}
END_CARD_SECONDS = 3.2
HOOK_MIN_SECONDS = 1.6
CAPTION_SIZE = 104


def hex_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    if len(v) != 6:
        raise ValueError(f"Colours must look like #RRGGBB, got {value!r}")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ease_back(x: float) -> float:
    """Overshoots slightly past 1 then settles: the 'pop' of a word appearing."""
    x = min(1.0, max(0.0, x))
    c = 1.9
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


@lru_cache(maxsize=16)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=2048)
def word_sprite(text: str, path: str, size: int, rgb: tuple[int, int, int]) -> Image.Image:
    """A word drawn once on a transparent tile, with a soft shadow for contrast."""
    f = font(path, size)
    l, t, r, b = f.getbbox(text)
    pad = max(8, size // 6)
    w, h = r - l + pad * 2, b - t + pad * 2
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((pad - l, pad - t + size // 30), text, font=f, fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(size / 18))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.alpha_composite(shadow)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=f, fill=rgb + (255,))
    return img


def with_alpha(img: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return img
    a = img.getchannel("A").point([int(i * max(0.0, alpha)) for i in range(256)])
    out = img.copy()
    out.putalpha(a)
    return out


def paste_centered(canvas: Image.Image, sprite: Image.Image, cx: float, cy: float,
                   scale: float = 1.0, alpha: float = 1.0) -> None:
    if alpha <= 0.01 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.005:
        sprite = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))),
                               Image.BILINEAR)
    sprite = with_alpha(sprite, alpha)
    canvas.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))


@dataclass
class Placed:
    index: int
    cx: float
    cy: float
    text: str
    accent: bool


def layout(line: Line, path: str, size: int, max_width: int, center_y: float,
           upper: bool = False) -> list[Placed]:
    """Wrap a line's words into centred rows; returns each word's centre point."""
    f = font(path, size)
    space = size * 0.28
    rows, row, row_w = [], [], 0.0
    for i, w in enumerate(line.words):
        text = w.text.upper() if upper else w.text
        ww = f.getlength(text)
        add = ww if not row else row_w + space + ww
        if row and add > max_width:
            rows.append((row, row_w))
            row, row_w = [], 0.0
            add = ww
        row.append((i, text, ww, w.accent))
        row_w = add
    if row:
        rows.append((row, row_w))
    line_h = size * 1.18
    top = center_y - line_h * len(rows) / 2 + line_h / 2
    placed = []
    for r, (items, total) in enumerate(rows):
        x = W / 2 - total / 2
        for i, text, ww, acc in items:
            placed.append(Placed(i, x + ww / 2, top + r * line_h, text, acc))
            x += ww + space
    return placed


def moon_sprite(diameter: int, rgb) -> Image.Image:
    s = diameter * 4  # draw large, shrink for smooth edges
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    mask = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((s * 0.05, s * 0.05, s * 0.95, s * 0.95), fill=255)
    d.ellipse((s * 0.30, s * -0.05, s * 1.10, s * 0.75), fill=0)  # the bite that makes a crescent
    img.paste(Image.new("RGBA", (s, s), rgb + (255,)), (0, 0), mask)
    return img.resize((diameter, diameter), Image.LANCZOS)


def chakra_sprite(diameter: int, rgb) -> Image.Image:
    """A golden spoked wheel with a toothed rim, drawn large then shrunk for smooth edges."""
    s = diameter * 4
    c = s / 2
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = rgb + (255,)
    teeth = 24
    pts = []
    for k in range(teeth * 2):
        a = math.pi * k / teeth
        r = c * (0.98 if k % 2 == 0 else 0.86)
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=col)
    d.ellipse((c - c * 0.74, c - c * 0.74, c + c * 0.74, c + c * 0.74), fill=(0, 0, 0, 0))
    for k in range(12):
        a = math.tau * k / 12
        d.line((c, c, c + c * 0.75 * math.cos(a), c + c * 0.75 * math.sin(a)), fill=col, width=int(s * 0.035))
    d.ellipse((c - c * 0.2, c - c * 0.2, c + c * 0.2, c + c * 0.2), fill=col)
    d.ellipse((c - c * 0.08, c - c * 0.08, c + c * 0.08, c + c * 0.08), fill=(0, 0, 0, 0))
    return img.resize((diameter, diameter), Image.LANCZOS)


def glow_sprite(diameter: int, rgb, strength: float = 0.28) -> Image.Image:
    yy, xx = np.mgrid[0:diameter, 0:diameter]
    r = np.hypot(xx - diameter / 2, yy - diameter / 2) / (diameter / 2)
    a = np.clip(1 - r, 0, 1) ** 2 * 255 * strength
    arr = np.zeros((diameter, diameter, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[..., 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


class Background:
    """Night sky with stars, or the script's own image or video."""

    def __init__(self, short: Short, fps: int):
        self.fps = fps
        self.kind = "stars"
        self.proc = None
        bg = short.background
        base = hex_rgb(short.brand["background"])
        if bg and bg != "stars":
            path = resolve(short, bg)
            if not path.exists():
                raise FileNotFoundError(f"Background not found: {path}")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                self.kind = "image"
                self.image = self._cover(Image.open(path).convert("RGB"), 1.12)
            else:
                self.kind = "video"
                self.video_path = path
        # Vertical gradient: brand background at the top, a warm glow at the bottom.
        top = np.array(base, dtype=np.float32)
        mid = np.array(hex_rgb(short.brand["sky_mid"]), dtype=np.float32)
        bottom = np.array(hex_rgb(short.brand["sky_bottom"]), dtype=np.float32)
        self.particle_rgb = np.array(hex_rgb(short.brand["particles"]), dtype=np.float32) / 255
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        grad = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (bottom - mid) * ((y - 0.55) / 0.45))
        self.sky = Image.fromarray(np.repeat(grad[:, None, :], W, axis=1).reshape(H, W, 3).astype(np.uint8), "RGB")
        rng = np.random.default_rng(11)
        self.stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.2, 3.4),
                       rng.uniform(4, 14), rng.uniform(0.5, 2.0), rng.uniform(0, 6.28)) for _ in range(140)]
        shade = np.zeros((H, W, 4), dtype=np.uint8)
        a = np.concatenate([np.linspace(150, 70, H // 2), np.linspace(70, 190, H - H // 2)])
        shade[..., 3] = a[:, None].astype(np.uint8)
        self.shade = Image.fromarray(shade, "RGBA")

    @staticmethod
    def _cover(img: Image.Image, extra: float = 1.0) -> Image.Image:
        s = max(W / img.width, H / img.height) * extra
        img = img.resize((int(img.width * s) + 1, int(img.height * s) + 1), Image.LANCZOS)
        return img

    def _video_frame(self):
        if self.proc is None:
            vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={self.fps}"
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-i", str(self.video_path), "-vf", vf, "-an",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        raw = self.proc.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:  # clip ended: loop it
            self.proc.wait()
            self.proc = None
            return self._video_frame()
        return Image.frombytes("RGB", (W, H), raw)

    def frame(self, t: float, zoom: float) -> Image.Image:
        if self.kind == "image":
            img = self.image
            z = zoom * (1 + 0.04 * t / 30)
            cw, ch = W / z * (img.width / (W * 1.12)), H / z * (img.height / (H * 1.12))
            cx, cy = img.width / 2, img.height / 2
            canvas = img.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((W, H))
        elif self.kind == "video":
            canvas = self._video_frame()
            if zoom > 1.001:
                cw, ch = W / zoom, H / zoom
                canvas = canvas.crop((int((W - cw) / 2), int((H - ch) / 2), int((W + cw) / 2), int((H + ch) / 2))).resize((W, H))
        else:
            canvas = self.sky.copy()
            d = ImageDraw.Draw(canvas)
            for x, y0, size, speed, tw, ph in self.stars:
                y = (y0 - t * speed) % H
                b = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(t * tw + ph))
                c = (90 + 165 * b) * self.particle_rgb
                r = size * (0.7 + 0.3 * b) / 2
                d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in c))
        canvas = canvas.convert("RGBA")
        canvas.alpha_composite(self.shade)
        return canvas

    def close(self):
        if self.proc:
            self.proc.kill()
            self.proc = None


@dataclass
class Plan:
    duration: float
    hook_end: float
    end_start: float
    line_windows: list[tuple[float, float]]


def plan(short: Short, narration: Narration) -> Plan:
    hs, he = narration.line_spans[-1]
    hook_end = max(he, HOOK_MIN_SECONDS)
    # Each line owns the screen from just before it is spoken until the next line
    # starts, so two sentences never overlap.
    starts = [max(hook_end, narration.line_spans[i][0] - 0.1) for i in range(len(short.lines))]
    last_end = narration.line_spans[len(short.lines) - 1][1] + 0.5 if short.lines else hook_end
    ends = starts[1:] + [last_end]
    windows = [(s, max(s + 0.3, e)) for s, e in zip(starts, ends)]
    end_start = windows[-1][1] if windows else hook_end
    return Plan(end_start + END_CARD_SECONDS, hook_end, end_start, windows)


class Renderer:
    def __init__(self, short: Short, narration: Narration, fps: int = 30):
        self.short, self.nar, self.fps = short, narration, fps
        b = short.brand
        self.text_rgb, self.accent_rgb, self.muted_rgb = hex_rgb(b["text"]), hex_rgb(b["accent"]), hex_rgb(b["muted"])
        self.plan = plan(short, narration)
        self.bg = Background(short, fps)
        if short.fonts not in FONT_SETS:
            raise ValueError(f"fonts must be one of {sorted(FONT_SETS)}, got {short.fonts!r}")
        head, cap, body, self.head_size = FONT_SETS[short.fonts]
        self.f_head, self.f_cap, self.f_body = str(head), str(cap), str(body)
        upper = short.fonts == "latin"
        self.logo = {"moon": lambda: moon_sprite(170, self.accent_rgb),
                     "chakra": lambda: chakra_sprite(190, self.accent_rgb),
                     "none": lambda: None}[short.logo]()
        self.glow = glow_sprite(520, self.accent_rgb)
        self.hook_layout = layout(short.hook, self.f_head, self.head_size, 900, H * 0.44, upper=upper)
        self.layouts = [layout(line, self.f_cap, CAPTION_SIZE, 920, H * 0.5) for line in short.lines]
        self.end_size = int(self.head_size * 0.79)
        self.end_layout = layout(short.end_title, self.f_head, self.end_size, 920, H * 0.56, upper=upper)
        self.word_times = {}
        for tw in narration.words:
            self.word_times[(tw.line, tw.index)] = (tw.start, tw.end)

    # -- layers ---------------------------------------------------------------------
    def _zoom(self, t: float) -> float:
        """A slow push-in on lines that carry a highlighted word, eased in and out."""
        for i, (s, e) in enumerate(self.plan.line_windows):
            if s <= t < e and any(w.accent for w in self.short.lines[i].words):
                p = (t - s) / max(0.1, e - s)
                return 1 + 0.05 * math.sin(math.pi * p)
        return 1.0

    def _moon(self, canvas, t: float) -> None:
        p = ease_out((t - self.plan.end_start + 0.2) / 1.0)
        x = (W - 190 + math.sin(t * 0.3) * 10) * (1 - p) + (W / 2) * p
        y = (230 + t * 1.8) * (1 - p) + (H * 0.37) * p
        s = 1 + 0.45 * p
        if self.logo is None:
            return
        paste_centered(canvas, self.glow, x, y, s)
        sprite = self.logo
        if self.short.logo == "chakra":  # the wheel turns slowly, faster as it arrives at the end
            sprite = sprite.rotate(-(t * 25 + p * 180), resample=Image.BICUBIC)
        paste_centered(canvas, sprite, x, y, s)

    def _hook(self, canvas, t: float) -> None:
        end = self.plan.hook_end
        if t >= end:
            return
        out = ease_out((t - (end - 0.2)) / 0.2)
        breathe = 1 + 0.03 * min(1.0, t / max(0.1, end))
        for p in self.hook_layout:
            rgb = self.accent_rgb if p.accent else self.text_rgb
            sprite = word_sprite(p.text, self.f_head, self.head_size, rgb)
            cx = W / 2 + (p.cx - W / 2) * breathe
            cy = H * 0.44 + (p.cy - H * 0.44) * breathe - out * 80
            paste_centered(canvas, sprite, cx, cy, breathe * (1 + 0.15 * out), 1 - out)

    def _captions(self, canvas, t: float) -> None:
        for li, (s, e) in enumerate(self.plan.line_windows):
            if not (s <= t < e):
                continue
            leave = ease_out((t - (e - 0.16)) / 0.16)
            for p in self.layouts[li]:
                ws, we = self.word_times.get((li, p.index), (s, e))
                if t < ws - 0.04:
                    # Not spoken yet: shown faintly so the sentence keeps its shape.
                    ghost = 0.22 * ease_out((t - s) / 0.2) * (1 - leave)
                    sprite = word_sprite(p.text, self.f_cap, CAPTION_SIZE, self.text_rgb)
                    paste_centered(canvas, sprite, p.cx, p.cy - leave * 50, 1.0, ghost)
                    continue
                pop = ease_back((t - ws + 0.04) / 0.22)
                speaking = ws - 0.04 <= t <= we + 0.08
                rgb = self.accent_rgb if (p.accent or speaking) else self.text_rgb
                sprite = word_sprite(p.text, self.f_cap, CAPTION_SIZE, rgb)
                scale = (0.55 + 0.45 * pop) * (1.07 if speaking else 1.0)
                paste_centered(canvas, sprite, p.cx, p.cy - leave * 50, scale, min(1.0, pop * 1.4) * (1 - leave))

    def _end_card(self, canvas, t: float) -> None:
        if t < self.plan.end_start:
            return
        lt = t - self.plan.end_start
        for k, p in enumerate(self.end_layout):
            pop = ease_back((lt - 0.25 - k * 0.12) / 0.3)
            if pop <= 0:
                continue
            rgb = self.accent_rgb if p.accent else self.text_rgb
            paste_centered(canvas, word_sprite(p.text, self.f_head, self.end_size, rgb), p.cx, p.cy,
                           0.6 + 0.4 * pop, min(1.0, pop * 1.3))
        sub = ease_out((lt - 0.9) / 0.5)
        if sub > 0:
            last_y = max(p.cy for p in self.end_layout)
            sprite = word_sprite(self.short.end_subtitle, self.f_body, 50, self.muted_rgb)
            paste_centered(canvas, sprite, W / 2, last_y + 130 + (1 - sub) * 30, 1.0, sub)

    # -- output ---------------------------------------------------------------------
    def frame(self, t: float) -> Image.Image:
        canvas = self.bg.frame(t, self._zoom(t))
        self._moon(canvas, t)
        self._hook(canvas, t)
        self._captions(canvas, t)
        self._end_card(canvas, t)
        return canvas.convert("RGB")

    def encode(self, audio_wav: Path, out: Path, progress=None) -> None:
        total = int(math.ceil(self.plan.duration * self.fps))
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(self.fps), "-i", "-",
               "-i", str(audio_wav),
               "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
               # Loudness YouTube expects (-14 LUFS), so the Short isn't quieter than the feed.
               "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "44100",
               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(total):
                proc.stdin.write(self.frame(i / self.fps).tobytes())
                if progress and i % self.fps == 0:
                    progress(i, total)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg could not encode the video")
        finally:
            self.bg.close()
            if proc.poll() is None:
                proc.kill()
