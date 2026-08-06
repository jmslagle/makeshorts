"""Captions rendered in Python, composited as images.

This is the backend that works on this machine. The installed ffmpeg has no
libass and no libfreetype, so it cannot draw a glyph at all -- but it has
`overlay`, which is timeline-enabled, and Pillow has a full freetype. So the
text is rasterized here and ffmpeg only has to put pictures on top of video,
which it will do on any build ever made.

**Karaoke without motion.** Each cue becomes one PNG per highlighted word. The
pill and the text layout are measured from the *whole* cue, once, so every
state of a cue is exactly the same size with every word in exactly the same
place -- only the colour of one word changes between frames. A layout measured
per-state would reflow as the highlight moved and the caption would visibly
shimmer.

**Dedupe.** A PNG's filename is a hash of everything that determines its
pixels: the style, the size, the line texts, and which word is lit. Two cues
that render identically (a repeated phrase, or the same word lit in the same
line) therefore share one file on disk, and a re-render of the same clip
rewrites nothing. This is also what makes the backend idempotent, which the
engine needs for `ms render` to be re-runnable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from makeshorts.render.captions.base import (
    CaptionOverlay,
    CaptionPosition,
    CaptionRenderError,
    OverlayAssets,
    anchor_y,
)
from makeshorts.render.captions.cues import Cue
from makeshorts.render.captions.style import CaptionStyle, RGBA, resolve_font_path

__all__ = ["PillowCaptionBackend", "render_cue_image", "CueLayout"]

CACHE_DIRNAME = "captions"

# Font size is reduced by this factor until the widest line fits the safe
# width. Small steps keep the size close to what the style asked for.
_SHRINK_STEP = 0.94
_MIN_FONT_PX = 12


class CueLayout:
    """Measured geometry for one cue: font, lines, word boxes, image size.

    Computed once per cue and reused for every karaoke state, which is the
    whole reason word positions stay put as the highlight moves.
    """

    def __init__(
        self,
        cue: Cue,
        style: CaptionStyle,
        output_size: tuple[int, int],
    ) -> None:
        out_w, out_h = output_size
        self.style = style
        self.output_size = output_size

        safe_top, safe_right, safe_bottom, safe_left = style.safe_area.insets(out_w, out_h)
        self.safe_top = safe_top
        self.safe_bottom = safe_bottom
        pad_x, pad_y = style.pill_padding_px(out_w, out_h)
        self.pad_x = pad_x
        self.pad_y = pad_y

        avail = max(out_w - safe_left - safe_right - 2 * pad_x, 64)
        self.texts = [
            [
                (w.text.upper() if style.uppercase else w.text)
                for w in (cue.words[i] for i in line)
            ]
            for line in cue.lines
        ]
        self.word_indices = [list(line) for line in cue.lines]

        font_px = style.font_px(out_h)
        font_path = resolve_font_path(style.font_file)
        self.font_path = font_path
        # `cues.py` wraps by character count, which is only an estimate for a
        # proportional face. Shrink until the real measurement fits rather
        # than letting a wide line run off the frame.
        while font_px > _MIN_FONT_PX:
            font = _load_font(font_path, font_px)
            extra = style.word_space_px(font_px)
            widths = [_line_width(font, words, extra) for words in self.texts]
            if not widths or max(widths) <= avail:
                break
            font_px = int(font_px * _SHRINK_STEP)
        self.font_px = max(font_px, _MIN_FONT_PX)
        self.font = _load_font(font_path, self.font_px)
        self.word_space = self.font.getlength(" ") + style.word_space_px(self.font_px)

        self.stroke = style.stroke_px(out_h)
        self.shadow = style.shadow_px(out_h)
        self.line_gap = style.line_spacing_px(out_h)

        # Use the font's own metrics rather than per-line ink extents, so
        # lines with no descender do not sit at a different height.
        ascent, descent = self.font.getmetrics()
        self.line_height = ascent + descent
        self.ascent = ascent

        self.line_widths = [
            _line_width(self.font, words, style.word_space_px(self.font_px))
            for words in self.texts
        ]
        self.text_w = max(self.line_widths) if self.line_widths else 0
        n = len(self.texts)
        self.text_h = n * self.line_height + max(n - 1, 0) * self.line_gap

        margin = self.stroke + self.shadow + 2  # room for outline and shadow ink
        self.width = _even(self.text_w + 2 * pad_x + 2 * margin)
        self.height = _even(self.text_h + 2 * pad_y + 2 * margin)
        self.margin = margin

        self.pill_box = (
            margin,
            margin,
            self.width - margin,
            self.height - margin,
        )

        # (line_index, word_index_in_cue, x, y) for every word, in image space.
        self.word_boxes: list[tuple[int, int, int, int, int]] = []
        y = margin + pad_y
        for li, words in enumerate(self.texts):
            line_w = self.line_widths[li]
            inner_left = margin + pad_x
            inner_w = self.width - 2 * (margin + pad_x)
            if style.align == "left":
                x = inner_left
            elif style.align == "right":
                x = inner_left + inner_w - line_w
            else:
                x = inner_left + (inner_w - line_w) // 2
            for k, text in enumerate(words):
                w = int(round(self.font.getlength(text)))
                self.word_boxes.append((li, self.word_indices[li][k], int(x), y, w))
                x += w + self.word_space
            y += self.line_height + self.line_gap

    def fingerprint(self, active_word: int | None) -> str:
        """Content hash of everything that determines this image's pixels."""
        payload = {
            "v": 2,
            "font": str(self.font_path),
            "font_px": self.font_px,
            "size": [self.width, self.height],
            "lines": self.texts,
            "active": active_word,
            "align": self.style.align,
            "style": {
                k: getattr(self.style, k)
                for k in (
                    "fill",
                    "stroke",
                    "stroke_width_pct",
                    "shadow_color",
                    "shadow_offset_pct",
                    "highlight_fill",
                    "highlight_pill",
                    "pill_color",
                    "pill_radius_pct",
                    "pill_padding_x_pct",
                    "pill_padding_y_pct",
                    "uppercase",
                    "karaoke",
                    "word_space_pct",
                )
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _even(v: float) -> int:
    return int(v) + (int(v) % 2)


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError as exc:
        raise CaptionRenderError(f"cannot load font {path}: {exc}") from exc


def _line_width(font: ImageFont.FreeTypeFont, words: list[str], extra_space: float = 0.0) -> int:
    """Width of a line laid out word-by-word.

    Measured as the sum of word widths plus the inter-word gaps, matching
    exactly how the words are then drawn -- measuring the joined string
    instead would include kerning across the space and drift from the drawn
    positions.
    """
    if not words:
        return 0
    space = font.getlength(" ") + extra_space
    return int(round(sum(font.getlength(w) for w in words) + space * (len(words) - 1)))


def render_cue_image(layout: CueLayout, active_word: int | None) -> Image.Image:
    """One RGBA image of a cue with `active_word` highlighted (None = none)."""
    style = layout.style
    img = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pill: RGBA = style.pill_rgba
    if pill[3] > 0:
        radius = min(
            style.pill_radius_px(layout.output_size[1]),
            (layout.pill_box[3] - layout.pill_box[1]) // 2,
            (layout.pill_box[2] - layout.pill_box[0]) // 2,
        )
        draw.rounded_rectangle(layout.pill_box, radius=max(radius, 0), fill=pill)

    hl_pill: RGBA = style.highlight_pill_rgba
    if hl_pill[3] > 0 and active_word is not None:
        for _li, widx, x, y, w in layout.word_boxes:
            if widx != active_word:
                continue
            pad = max(layout.font_px // 8, 2)
            draw.rounded_rectangle(
                (x - pad, y - pad // 2, x + w + pad, y + layout.line_height + pad // 2),
                radius=max(pad, 2),
                fill=hl_pill,
            )

    shadow: RGBA = style.shadow_rgba
    fill: RGBA = style.fill_rgba
    stroke: RGBA = style.stroke_rgba
    highlight: RGBA = style.highlight_rgba

    for _li, widx, x, y, _w in layout.word_boxes:
        if shadow[3] > 0 and layout.shadow > 0:
            draw.text(
                (x + layout.shadow, y + layout.shadow),
                _word_text(layout, widx),
                font=layout.font,
                fill=shadow,
            )

    for _li, widx, x, y, _w in layout.word_boxes:
        active = style.karaoke and active_word is not None and widx == active_word
        draw.text(
            (x, y),
            _word_text(layout, widx),
            font=layout.font,
            fill=highlight if active else fill,
            stroke_width=layout.stroke,
            stroke_fill=stroke if layout.stroke else None,
        )
    return img


def _word_text(layout: CueLayout, cue_word_index: int) -> str:
    for li, indices in enumerate(layout.word_indices):
        if cue_word_index in indices:
            return layout.texts[li][indices.index(cue_word_index)]
    raise CaptionRenderError(f"word {cue_word_index} is not in this cue's layout")


class PillowCaptionBackend:
    """Rasterize cues to RGBA PNGs for the engine to composite.

    Always available: it depends on Pillow and a font file, neither of which
    is a property of the video toolchain.
    """

    name = "pillow"

    def __init__(self, cache_dirname: str = CACHE_DIRNAME) -> None:
        self.cache_dirname = cache_dirname

    def is_available(self) -> bool:
        try:
            resolve_font_path(None)
        except Exception:
            return False
        return True

    def render(
        self,
        cues: list[Cue],
        style: CaptionStyle,
        output_size: tuple[int, int],
        workdir: Path,
        position: CaptionPosition = "lower_third",
    ) -> OverlayAssets:
        out_w, out_h = output_size
        cache = Path(workdir) / self.cache_dirname
        cache.mkdir(parents=True, exist_ok=True)

        overlays: list[CaptionOverlay] = []
        written: set[str] = set()

        for ci, cue in enumerate(cues):
            if not cue.words:
                continue
            layout = CueLayout(cue, style, output_size)
            y = anchor_y(position, layout.height, out_h, layout.safe_top, layout.safe_bottom)
            x = max((out_w - layout.width) // 2, 0)

            states: list[tuple[int | None, float, float]]
            if style.karaoke:
                states = [(i, s, e) for i, s, e in cue.karaoke_states()]
            else:
                states = [(None, cue.start, cue.end)]

            for active, start, end in states:
                if end <= start:
                    continue
                key = layout.fingerprint(active)
                png = cache / f"cue-{key}.png"
                if key not in written and not png.exists():
                    render_cue_image(layout, active).save(png, "PNG", optimize=True)
                written.add(key)
                overlays.append(
                    CaptionOverlay(
                        png_path=str(png),
                        x=x,
                        y=y,
                        width=layout.width,
                        height=layout.height,
                        start=start,
                        end=end,
                        cue_index=ci,
                        word_index=active,
                    )
                )

        return OverlayAssets(
            overlays=overlays,
            unique_images=len({o.png_path for o in overlays}),
        )
