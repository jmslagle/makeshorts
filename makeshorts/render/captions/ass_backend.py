"""Captions as an Advanced SubStation Alpha file, for the libass path.

Dormant on this machine -- the installed ffmpeg has no libass, so `caps.py`
routes to the Pillow backend and nothing here runs. It exists because that is
a property of one homebrew build, not of the design, and the day someone runs
this on a box with a full ffmpeg the cheap path should already be correct and
tested rather than written in a hurry.

When it does activate the win is real: one filter, one file, no rasterizing
hundreds of PNGs, and text that stays crisp because libass renders at the
output resolution during encode.

**Karaoke semantics differ from the Pillow backend, deliberately.** `\\k` is
progressive-fill karaoke: a word switches from SecondaryColour to
PrimaryColour when its turn arrives and *stays* there for the rest of the cue.
The Pillow backend lights exactly one word at a time. Both are conventional
looks; `\\k` is the one ASS can express in a single event, and expressing the
single-word version would mean one Dialogue line per word and giving up the
tag the format is built around. The difference is per-cue and invisible in
practice, but it is a real difference and worth knowing before comparing
output across machines.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from makeshorts.render.captions.base import (
    CaptionPosition,
    SubtitleAssets,
)
from makeshorts.render.captions.cues import Cue
from makeshorts.render.captions.style import CaptionStyle, RGBA

__all__ = ["AssCaptionBackend", "build_ass", "ass_color", "ass_time"]

# ASS alignment codes (numpad layout). Only the centered column is used --
# captions that drift left or right across cues read as a mistake.
_ALIGN = {"lower_third": 2, "center": 5, "upper_third": 8}


def ass_color(rgba: RGBA) -> str:
    """(r,g,b,a) -> `&HAABBGGRR&`.

    Two traps in one format: the byte order is reversed, and the alpha channel
    is *inverted* -- 00 is opaque and FF is invisible, the opposite of every
    other colour notation in this project.
    """
    r, g, b, a = rgba
    return f"&H{255 - a:02X}{b:02X}{g:02X}{r:02X}&"


def ass_time(seconds: float) -> str:
    """Seconds -> `H:MM:SS.cc`. ASS resolution is centiseconds, not ms."""
    seconds = max(seconds, 0.0)
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """Neutralize the three characters that mean something to the parser."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _style_line(style: CaptionStyle, output_size: tuple[int, int], position: CaptionPosition) -> str:
    out_w, out_h = output_size
    safe_top, safe_right, safe_bottom, safe_left = style.safe_area.insets(out_w, out_h)
    margin_v = safe_top if position == "upper_third" else safe_bottom

    # `\k` fills from SecondaryColour into PrimaryColour, so Primary is the
    # *sung* colour -- the highlight -- and Secondary is the resting fill.
    # Without karaoke the two are the same and only Primary is ever seen.
    primary = style.highlight_rgba if style.karaoke else style.fill_rgba
    secondary = style.fill_rgba

    pill = style.pill_rgba
    if pill[3] > 0:
        # BorderStyle 4 draws an opaque box behind the whole line. It is a
        # rectangle -- libass has no rounded corners -- so a `pill_radius` in
        # the style is silently ignored on this path.
        border_style = 4
        outline = max(style.pill_padding_px(out_w, out_h)[1] // 2, 1)
        back = pill
    else:
        border_style = 1
        outline = max(style.stroke_px(out_h), 1)
        back = style.shadow_rgba

    shadow = style.shadow_px(out_h)
    bold = -1 if "bold" in style.font_family.lower() else 0

    fields = [
        "Default",
        style.font_family,
        str(style.font_px(out_h)),
        ass_color(primary),
        ass_color(secondary),
        ass_color(style.stroke_rgba),
        ass_color(back),
        str(bold),
        "0",  # Italic
        "0",  # Underline
        "0",  # StrikeOut
        "100",  # ScaleX
        "100",  # ScaleY
        "0",  # Spacing
        "0",  # Angle
        str(border_style),
        str(outline),
        str(shadow),
        str(_ALIGN.get(position, 2)),
        str(safe_left),
        str(safe_right),
        str(margin_v),
        "1",  # Encoding: default
    ]
    return "Style: " + ",".join(fields)


def _cue_text(cue: Cue, style: CaptionStyle) -> str:
    """One Dialogue payload: `\\k`-tagged words with `\\N` between lines.

    Durations come from `karaoke_states`, so a pause between words extends the
    preceding word's fill rather than leaving the highlight stranded.
    """
    durations: dict[int, float] = {}
    if style.karaoke:
        for idx, start, end in cue.karaoke_states():
            durations[idx] = max(end - start, 0.0)

    parts: list[str] = []
    for li, line in enumerate(cue.lines):
        if li:
            parts.append("\\N")
        chunks: list[str] = []
        for k, widx in enumerate(line):
            text = cue.words[widx].text
            if style.uppercase:
                text = text.upper()
            if k:
                text = " " + text
            if style.karaoke:
                cs = max(int(round(durations.get(widx, 0.0) * 100)), 1)
                chunks.append(f"{{\\k{cs}}}{_escape(text)}")
            else:
                chunks.append(_escape(text))
        parts.append("".join(chunks))
    return "".join(parts)


def build_ass(
    cues: list[Cue],
    style: CaptionStyle,
    output_size: tuple[int, int],
    position: CaptionPosition = "lower_third",
) -> str:
    """The complete `.ass` document, as text.

    `PlayResX`/`PlayResY` are set to the output size so every pixel value in
    the style means an actual output pixel and libass does no scaling of its
    own -- the same numbers the Pillow backend uses.
    """
    out_w, out_h = output_size
    lines = [
        "[Script Info]",
        "; Generated by makeshorts. Do not edit -- regenerate from clips.json.",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {out_w}",
        f"PlayResY: {out_h}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        _style_line(style, output_size, position),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        if not cue.words:
            continue
        lines.append(
            "Dialogue: 0,"
            f"{ass_time(cue.start)},{ass_time(cue.end)},Default,,0,0,0,,"
            f"{_cue_text(cue, style)}"
        )
    return "\n".join(lines) + "\n"


class AssCaptionBackend:
    """Write cues to an `.ass` file for a libass-capable renderer.

    `is_available` takes the probed capabilities rather than deciding for
    itself, so the one source of truth about the toolchain stays `caps.py`.
    """

    name = "ass"

    def __init__(self, caps: object | None = None) -> None:
        self._caps = caps

    def is_available(self) -> bool:
        if self._caps is None:
            from makeshorts.render.caps import probe_caps

            self._caps = probe_caps()
        return bool(getattr(self._caps, "has_libass", False))

    def render(
        self,
        cues: list[Cue],
        style: CaptionStyle,
        output_size: tuple[int, int],
        workdir: Path,
        position: CaptionPosition = "lower_third",
    ) -> SubtitleAssets:
        text = build_ass(cues, style, output_size, position)
        out_dir = Path(workdir) / "captions"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Content-addressed like the PNGs, for the same reason: re-rendering a
        # clip must not churn files the receipt refers to.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
        path = out_dir / f"cues-{digest}.ass"
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        return SubtitleAssets(path=str(path), format="ass")
