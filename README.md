# Shorts editor

Turns a short text script into a finished vertical YouTube Short, with no editing by hand:

- **AI voice-over** read from your script (free Microsoft Edge voices, 300+ voices in many languages, including Hindi and Indian English)
- **Captions synced to the voice**: each word lights up as it is spoken, and highlighted words stay in your accent colour
- **Motion graphics**: a big hook headline on the first frame, word pop-in animation, slow push-ins on key lines, a drifting moon logo and an animated end card
- **Background music** ducked under the voice: a generated, royalty-free calm pad by default, or your own track
- **Background**: an animated starry night sky, or your own photo or video with a slow zoom
- Output: `1080×1920` MP4 at YouTube loudness, plus an `.srt` captions file and a cover image

## Setup

You need Python 3.10 or newer and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Make a Short

```bash
python -m shorts_editor make scripts/before_bed.yaml
```

The video lands in `output/before_bed.mp4`, with `before_bed.srt` and `before_bed-cover.png` beside it.
The AI voice needs an internet connection.

Make many at once (your autopilot batch):

```bash
python -m shorts_editor make scripts/*.yaml --out-dir output
```

Check the look in seconds before a full render (saves still frames):

```bash
python -m shorts_editor preview scripts/before_bed.yaml --at 0 5 12
```

Pick a voice:

```bash
python -m shorts_editor voices --lang en      # or --lang hi for Hindi
```

## Writing a script

Copy `scripts/before_bed.yaml` and change the words. Wrap words in `*asterisks*` to highlight them.

```yaml
title: Your brain decides tomorrow, tonight
hook: Do this *before bed*            # shown big on the first frame, and spoken first
voice: en-US-AndrewNeural             # e.g. en-IN-PrabhatNeural, hi-IN-MadhurNeural
rate: "-5%"                           # speaking speed
pitch: "+0Hz"                         # e.g. "-8Hz" for a deeper voice
background: stars                     # or my_photo.jpg / my_clip.mp4 (path relative to this file)
music: auto                           # auto, none, or my_track.mp3
music_volume: 0.22
fonts: latin                          # hindi for Devanagari text
logo: moon                            # moon, chakra or none
lines:
  - Your brain decides tomorrow... *tonight.*
  - While you sleep, your brain *sorts* through your day.
end_card:
  title: Sleep with *Impact*
  subtitle: Follow for one idea every night
brand:                                # optional; these are the defaults
  accent: "#FF5A1F"
  text: "#F4EDE4"
  sky_mid: "#0B0E1C"                  # background gradient
  sky_bottom: "#1B100B"
  particles: "#F4EDE4"                # stars (or embers, in a warm colour)
```

### Hindi

`scripts/mahabharat_karma.yaml` is a full Hindi example: an educational Short on the
Bhagavad Gita's lesson on karma, with a Hindi voice (`hi-IN-MadhurNeural`), Devanagari
captions (`fonts: hindi`), a turning golden chakra (`logo: chakra`) and rising embers.

Tips that make Shorts perform better:

- Keep the hook to 2–4 words; it is the first thing viewers see.
- Aim for 25–45 seconds: 8–10 short lines.
- Use only music you have the rights to. `music: auto` is always safe.

## Tests

```bash
python -m pytest tests
```

## Project layout

```
shorts_editor/
  script.py     reads the YAML script
  tts.py        AI voice and word timings
  audio.py      generated music, sound effects, ducking and mixing
  render.py     motion graphics and video encoding
  pipeline.py   script in, video + captions + cover out
  cli.py        the make / preview / voices commands
  assets/fonts  Anton, Inter and Mukta (SIL Open Font License)
scripts/        your Short scripts
tests/
```
