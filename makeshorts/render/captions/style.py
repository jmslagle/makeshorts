"""Caption styles — the tunable surface of what captions look like.

`clips.json` says `{"style": "pill-karaoke"}` and nothing more. Everything
concrete -- font file, size, colours, padding -- resolves here at render time,
from `config/styles.yaml`, so restyling every clip is a YAML edit and not a
re-render decision baked into the edit list.

Sizes are fractions of the **output height** rather than absolute points, so a
style survives a change of output resolution unchanged. A 1080x1920 vertical
frame with `font_size_pct: 0.042` gets 80px text; the same style at 720x1280
gets 54px and looks identical.

Colours are `#RGB`, `#RRGGBB`, or `#RRGGBBAA` strings. Both backends need them
in different byte orders, so they are parsed once here into RGBA tuples and
each backend formats from that.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from makeshorts.artifacts import Strict

__all__ = [
    "RGBA",
    "SafeArea",
    "CaptionStyle",
    "DEFAULT_STYLES",
    "DEFAULT_STYLE_NAME",
    "StyleError",
    "parse_color",
    "load_styles",
    "get_style",
    "default_style_name",
    "resolve_font_path",
    "styles_yaml_template",
]

RGBA = tuple[int, int, int, int]

DEFAULT_STYLE_NAME = "pill-karaoke"


class StyleError(ValueError):
    """An unknown style name, or a styles.yaml that will not parse."""


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

_HEX = re.compile(r"^#?(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

_NAMED: dict[str, str] = {
    "white": "#FFFFFFFF",
    "black": "#000000FF",
    "transparent": "#00000000",
    "yellow": "#FFD400FF",
    "cyan": "#22D3EEFF",
    "lime": "#A3E635FF",
}


def parse_color(value: str) -> RGBA:
    """`#RGB` / `#RGBA` / `#RRGGBB` / `#RRGGBBAA` / a few names -> (r,g,b,a).

    Alpha defaults to fully opaque when omitted, which is what anyone writing
    `#FFFFFF` in a YAML file means.
    """
    raw = _NAMED.get(value.strip().lower(), value.strip())
    if not _HEX.match(raw):
        raise StyleError(f"not a colour: {value!r} (want #RRGGBB or #RRGGBBAA)")
    h = raw.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        h += "FF"
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))


# --------------------------------------------------------------------------
# Fonts
#
# ffmpeg on this machine has no font handling at all, so the font file is
# Pillow's business. The ASS backend needs a *family name* instead, which is
# why the style carries both.
# --------------------------------------------------------------------------

FONT_SEARCH_PATHS: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def resolve_font_path(font_file: str | None) -> Path:
    """The style's font if it exists, else the first font that does.

    A missing font is a bad reason to fail a render that is otherwise fine, so
    this degrades rather than raising -- but only over a known-good list, never
    to Pillow's unmetricated default bitmap font, which would silently produce
    tiny unreadable captions.
    """
    if font_file:
        p = Path(font_file).expanduser()
        if p.is_file():
            return p
    for candidate in FONT_SEARCH_PATHS:
        p = Path(candidate)
        if p.is_file():
            return p
    raise StyleError(
        f"no usable font: {font_file!r} does not exist and none of the "
        f"fallbacks are present ({', '.join(FONT_SEARCH_PATHS)})"
    )


# --------------------------------------------------------------------------
# The style model
# --------------------------------------------------------------------------


class SafeArea(Strict):
    """Keep-out margins as fractions of the output dimensions.

    The bottom inset is the load-bearing one: platform UI (captions toggle,
    handle, progress bar) covers roughly the lower 12% of a vertical feed
    video, and text under it is text nobody reads.
    """

    top_pct: float = Field(default=0.08, ge=0.0, le=0.45)
    bottom_pct: float = Field(default=0.14, ge=0.0, le=0.45)
    left_pct: float = Field(default=0.06, ge=0.0, le=0.45)
    right_pct: float = Field(default=0.06, ge=0.0, le=0.45)

    def insets(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(top, right, bottom, left) in pixels, CSS order."""
        return (
            int(round(self.top_pct * height)),
            int(round(self.right_pct * width)),
            int(round(self.bottom_pct * height)),
            int(round(self.left_pct * width)),
        )


class CaptionStyle(Strict):
    """One named caption look.

    Both backends read this. Where a backend cannot express something (libass
    has no rounded corners; Pillow has no font-family lookup) it approximates
    and says so in its own docstring.
    """

    name: str = DEFAULT_STYLE_NAME

    # ---- Type ----
    # Absolute path for Pillow. Falls back through FONT_SEARCH_PATHS.
    font_file: str = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    # Family name for libass, which resolves through fontconfig, not paths.
    font_family: str = "Arial Bold"
    font_size_pct: float = Field(default=0.040, gt=0.0, le=0.25)
    line_spacing_pct: float = Field(default=0.012, ge=0.0, le=0.1)
    # Extra space between words, on top of the font's own space glyph, as a
    # fraction of the font size. Condensed faces (Impact) set a narrow space
    # that leaves karaoke-highlighted words looking joined together.
    word_space_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    uppercase: bool = False
    align: Literal["left", "center", "right"] = "center"

    # ---- Colours ----
    fill: str = "#FFFFFF"
    stroke: str = "#000000"
    stroke_width_pct: float = Field(default=0.0035, ge=0.0, le=0.03)
    shadow_color: str = "#00000099"
    shadow_offset_pct: float = Field(default=0.0025, ge=0.0, le=0.03)

    # ---- Karaoke ----
    karaoke: bool = True
    highlight_fill: str = "#FFD400"
    # A filled box behind the active word. "#00000000" disables it.
    highlight_pill: str = "#00000000"

    # ---- Pill (the rounded slab behind the whole cue) ----
    pill_color: str = "#000000B3"
    pill_radius_pct: float = Field(default=0.014, ge=0.0, le=0.1)
    pill_padding_x_pct: float = Field(default=0.022, ge=0.0, le=0.2)
    pill_padding_y_pct: float = Field(default=0.010, ge=0.0, le=0.2)

    # ---- Grouping (consumed by cues.py, not by any renderer) ----
    max_chars_per_line: int = Field(default=26, ge=6, le=120)
    max_lines: int = Field(default=2, ge=1, le=6)

    safe_area: SafeArea = Field(default_factory=SafeArea)

    # ---- Parsed colours -------------------------------------------------

    @property
    def fill_rgba(self) -> RGBA:
        return parse_color(self.fill)

    @property
    def stroke_rgba(self) -> RGBA:
        return parse_color(self.stroke)

    @property
    def shadow_rgba(self) -> RGBA:
        return parse_color(self.shadow_color)

    @property
    def highlight_rgba(self) -> RGBA:
        return parse_color(self.highlight_fill)

    @property
    def highlight_pill_rgba(self) -> RGBA:
        return parse_color(self.highlight_pill)

    @property
    def pill_rgba(self) -> RGBA:
        return parse_color(self.pill_color)

    # ---- Pixel resolution ------------------------------------------------

    def font_px(self, output_height: int) -> int:
        return max(8, int(round(self.font_size_pct * output_height)))

    def line_spacing_px(self, output_height: int) -> int:
        return int(round(self.line_spacing_pct * output_height))

    def word_space_px(self, font_px: int) -> float:
        return self.word_space_pct * font_px

    def stroke_px(self, output_height: int) -> int:
        return int(round(self.stroke_width_pct * output_height))

    def shadow_px(self, output_height: int) -> int:
        return int(round(self.shadow_offset_pct * output_height))

    def pill_radius_px(self, output_height: int) -> int:
        return int(round(self.pill_radius_pct * output_height))

    def pill_padding_px(self, output_width: int, output_height: int) -> tuple[int, int]:
        return (
            int(round(self.pill_padding_x_pct * output_width)),
            int(round(self.pill_padding_y_pct * output_height)),
        )

    @field_validator(
        "fill",
        "stroke",
        "shadow_color",
        "highlight_fill",
        "highlight_pill",
        "pill_color",
    )
    @classmethod
    def _valid_color(cls, v: str) -> str:
        parse_color(v)  # raises StyleError on anything unparseable
        return v


# --------------------------------------------------------------------------
# Defaults
#
# These exist so the render path works before anyone writes config/styles.yaml
# and so tests do not depend on a config file. The YAML, when present,
# overrides field-by-field.
# --------------------------------------------------------------------------

DEFAULT_STYLES: dict[str, CaptionStyle] = {
    # The house style: white bold on a translucent dark slab, active word in
    # amber. Legible over both a bright slide and a dim webcam.
    "pill-karaoke": CaptionStyle(
        name="pill-karaoke",
        font_file="/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        font_family="Arial Bold",
        font_size_pct=0.040,
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width_pct=0.0030,
        karaoke=True,
        highlight_fill="#FFD400",
        pill_color="#000000B3",
        pill_radius_pct=0.014,
        max_chars_per_line=26,
        max_lines=2,
    ),
    # No slab; relies on stroke and shadow. Reads as less "produced" and keeps
    # more of the frame visible.
    "clean-bold": CaptionStyle(
        name="clean-bold",
        font_file="/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        font_family="Arial Bold",
        font_size_pct=0.038,
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width_pct=0.0045,
        shadow_offset_pct=0.0030,
        karaoke=True,
        highlight_fill="#7DD3FC",
        pill_color="#00000000",
        pill_padding_x_pct=0.012,
        pill_padding_y_pct=0.006,
        max_chars_per_line=24,
        max_lines=2,
    ),
    # Loud. One line at a time, uppercase, the look people expect from a
    # sports or hype clip.
    "impact-punch": CaptionStyle(
        name="impact-punch",
        font_file="/System/Library/Fonts/Supplemental/Impact.ttf",
        font_family="Impact",
        font_size_pct=0.052,
        # Impact's space glyph is very narrow; without extra tracking the
        # highlight box makes adjacent words read as one.
        word_space_pct=0.18,
        uppercase=True,
        fill="#FFFFFF",
        stroke="#000000",
        stroke_width_pct=0.0060,
        karaoke=True,
        highlight_fill="#22D3EE",
        highlight_pill="#000000CC",
        pill_color="#00000000",
        pill_padding_x_pct=0.014,
        pill_padding_y_pct=0.008,
        max_chars_per_line=18,
        max_lines=1,
    ),
    # Whole cue at once, no word highlighting. For talks where the karaoke
    # effect would be a distraction.
    "static-block": CaptionStyle(
        name="static-block",
        font_file="/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        font_family="Arial Bold",
        font_size_pct=0.036,
        karaoke=False,
        pill_color="#000000CC",
        max_chars_per_line=30,
        max_lines=3,
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

DEFAULT_STYLES_PATH = Path("config/styles.yaml")


def _coerce_mapping(data: Any) -> dict[str, Any]:
    """Accept either `{styles: {name: {...}}}` or a bare `{name: {...}}`.

    config/styles.yaml belongs to another part of the project; tolerating both
    shapes is cheaper than coupling to whichever one it lands on.
    """
    if not isinstance(data, dict):
        raise StyleError(f"styles.yaml must be a mapping, got {type(data).__name__}")
    if "styles" in data and isinstance(data["styles"], dict):
        return dict(data["styles"])
    return {k: v for k, v in data.items() if k not in ("version", "default")}


def load_styles(path: str | Path | None = None) -> dict[str, CaptionStyle]:
    """Built-in styles, overlaid with `config/styles.yaml` if it exists.

    Overlay is per-field: a YAML entry naming an existing style need only set
    the fields it changes. An entry with a new name must be complete enough to
    validate on its own (every field has a default, so in practice `{}` works).

    A missing file is not an error -- the defaults are a complete, usable set.
    A malformed file *is* an error, because silently rendering the wrong look
    is worse than failing.
    """
    p = Path(path) if path is not None else DEFAULT_STYLES_PATH
    styles = {k: v.model_copy(deep=True) for k, v in DEFAULT_STYLES.items()}
    if not p.is_file():
        return styles

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a hard dep
        raise StyleError("pyyaml is required to read styles.yaml") from exc

    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except Exception as exc:
        raise StyleError(f"{p}: {exc}") from exc

    for name, fields in _coerce_mapping(raw).items():
        if fields is None:
            fields = {}
        if not isinstance(fields, dict):
            raise StyleError(f"{p}: style {name!r} must be a mapping")
        base = styles.get(name)
        merged: dict[str, Any] = (
            base.model_dump() if base is not None else {}
        ) | dict(fields)
        merged["name"] = name
        # Nested safe_area merges too, so a YAML style can nudge one inset.
        if base is not None and isinstance(fields.get("safe_area"), dict):
            merged["safe_area"] = base.safe_area.model_dump() | fields["safe_area"]
        try:
            styles[name] = CaptionStyle.model_validate(merged)
        except Exception as exc:
            raise StyleError(f"{p}: style {name!r}: {exc}") from exc
    return styles


def default_style_name(path: str | Path | None = None) -> str:
    """The `default:` style named by styles.yaml, or the built-in default.

    Used when neither the clip nor render.yaml names one. A `default:` naming
    a style that does not exist falls back rather than failing, since the
    caller is about to look the name up and will produce the better message.
    """
    p = Path(path) if path is not None else DEFAULT_STYLES_PATH
    if not p.is_file():
        return DEFAULT_STYLE_NAME
    try:
        import yaml

        raw = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return DEFAULT_STYLE_NAME
    name = raw.get("default") if isinstance(raw, dict) else None
    return name if isinstance(name, str) and name else DEFAULT_STYLE_NAME


def get_style(name: str, path: str | Path | None = None) -> CaptionStyle:
    """One style by name, or a `StyleError` listing what does exist."""
    styles = load_styles(path)
    if name not in styles:
        raise StyleError(f"unknown caption style {name!r}; available: {sorted(styles)}")
    return styles[name]


def styles_yaml_template() -> str:
    """The built-in styles as YAML, in the shape `load_styles` reads.

    Whoever owns `config/styles.yaml` can start from this rather than
    reverse-engineering the field names from the model.
    """
    import yaml

    payload = {
        "styles": {
            name: style.model_dump(exclude={"name"}) for name, style in DEFAULT_STYLES.items()
        }
    }
    return yaml.safe_dump(payload, sort_keys=False, width=100)
