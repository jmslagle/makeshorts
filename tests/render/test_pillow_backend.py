"""Pillow caption backend tests.

Text rendering cannot be asserted pixel-for-pixel without pinning a font
version, so these check the properties that actually matter: the image is the
right size, it has ink in the right place, consecutive karaoke frames differ
in exactly the region where the highlight moved, and identical states share
one file on disk.

A handful of real PNGs are written to `tests/render/_samples/` for a human to
look at, because "there is ink somewhere" is not the same as "this looks
right".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from makeshorts.render.captions.base import anchor_y
from makeshorts.render.captions.cues import CueOptions, build_cues
from makeshorts.render.captions.pillow_backend import (
    CueLayout,
    PillowCaptionBackend,
    render_cue_image,
)
from makeshorts.render.captions.style import DEFAULT_STYLES, get_style

from tests.render.conftest import evenly_spaced, words

SAMPLES = Path(__file__).parent / "_samples"
OUT = (1080, 1920)

LINE = "Eighteen months is the number that kills companies."


@pytest.fixture
def cues():
    return build_cues(evenly_spaced(LINE, per_word=0.4), 0.0, 20.0, CueOptions())


@pytest.fixture
def style():
    return DEFAULT_STYLES["pill-karaoke"]


def ink_bbox(img: Image.Image):
    """Bounding box of everything non-transparent."""
    return img.getchannel("A").getbbox()


# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------


def test_render_produces_one_overlay_per_word(tmp_path, cues, style):
    backend = PillowCaptionBackend()
    assets = backend.render(cues, style, OUT, tmp_path)
    assert assets.kind == "overlays"
    assert len(assets.overlays) == sum(len(c.words) for c in cues)
    for o in assets.overlays:
        assert Path(o.png_path).is_file()
        assert o.end > o.start


def test_overlay_images_are_rgba_and_match_their_declared_size(tmp_path, cues, style):
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path)
    for o in assets.overlays[:6]:
        with Image.open(o.png_path) as img:
            assert img.mode == "RGBA"
            assert img.size == (o.width, o.height)


def test_every_state_of_a_cue_is_the_same_size_and_position(tmp_path, cues, style):
    """A layout measured per-state would reflow and the caption would shimmer."""
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path)
    by_cue: dict[int, list] = {}
    for o in assets.overlays:
        by_cue.setdefault(o.cue_index, []).append(o)
    for group in by_cue.values():
        assert len({(o.x, o.y, o.width, o.height) for o in group}) == 1


def test_overlays_fit_inside_the_output_frame(tmp_path, cues, style):
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path)
    for o in assets.overlays:
        assert 0 <= o.x and o.x + o.width <= OUT[0]
        assert 0 <= o.y and o.y + o.height <= OUT[1]


def test_lower_third_sits_above_the_safe_area(tmp_path, cues, style):
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path, "lower_third")
    safe_bottom = int(round(style.safe_area.bottom_pct * OUT[1]))
    for o in assets.overlays:
        assert o.y + o.height <= OUT[1] - safe_bottom + 1


@pytest.mark.parametrize("position", ["lower_third", "center", "upper_third"])
def test_each_position_lands_in_its_third(tmp_path, cues, style, position):
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path, position)
    o = assets.overlays[0]
    mid = o.y + o.height // 2
    if position == "lower_third":
        assert mid > OUT[1] * 0.6
    elif position == "center":
        assert OUT[1] * 0.4 < mid < OUT[1] * 0.6
    else:
        assert mid < OUT[1] * 0.4


def test_lower_third_baseline_does_not_move_between_one_and_two_line_cues(style):
    """The bottom edge is the anchor, so a taller cue grows upward."""
    safe_top, _, safe_bottom, _ = style.safe_area.insets(*OUT)
    one = anchor_y("lower_third", 120, OUT[1], safe_top, safe_bottom)
    two = anchor_y("lower_third", 220, OUT[1], safe_top, safe_bottom)
    assert one + 120 == two + 220


def test_the_pill_is_wider_than_the_text_it_wraps(cues, style):
    layout = CueLayout(cues[0], style, OUT)
    assert layout.width > layout.text_w
    assert layout.height > layout.text_h


def test_a_long_line_shrinks_the_font_to_fit(style):
    """cues.py wraps by character count; a wide face can still overrun."""
    wide = "MMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM"
    doc = evenly_spaced(wide, per_word=0.4)
    cue = build_cues(doc, 0.0, 10.0, CueOptions(max_chars_per_line=80, max_lines=1))[0]
    layout = CueLayout(cue, style, OUT)
    assert layout.font_px < style.font_px(OUT[1]), "should have shrunk"
    assert layout.width <= OUT[0]


# --------------------------------------------------------------------------
# Ink
# --------------------------------------------------------------------------


def test_the_image_actually_has_ink(cues, style):
    img = render_cue_image(CueLayout(cues[0], style, OUT), 0)
    bbox = ink_bbox(img)
    assert bbox is not None, "a fully transparent caption is a rendering failure"
    x0, y0, x1, y1 = bbox
    assert x1 - x0 > img.width * 0.5
    assert y1 - y0 > img.height * 0.3


def test_no_pill_style_still_renders_ink():
    style = DEFAULT_STYLES["clean-bold"]
    doc = evenly_spaced("no slab behind this text", per_word=0.4)
    cue = build_cues(doc, 0.0, 5.0, CueOptions())[0]
    img = render_cue_image(CueLayout(cue, style, OUT), 1)
    assert ink_bbox(img) is not None
    corner = img.getpixel((1, 1))
    assert corner[3] == 0, "without a pill the corners must stay transparent"


def test_consecutive_karaoke_frames_differ(cues, style):
    layout = CueLayout(cues[0], style, OUT)
    frames = [render_cue_image(layout, i).tobytes() for i in range(len(cues[0].words))]
    for a, b in zip(frames, frames[1:], strict=False):
        assert a != b, "the highlight did not move between states"
    assert len(set(frames)) == len(frames), "every state must be visually distinct"


def test_the_highlight_changes_only_the_active_word_region(cues, style):
    """Everything except the two words involved must be byte-identical."""
    layout = CueLayout(cues[0], style, OUT)
    a = render_cue_image(layout, 0)
    b = render_cue_image(layout, 1)
    from PIL import ImageChops

    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox()
    assert diff is not None
    # The changed area is the first two words, not the whole pill.
    boxes = {widx: (x, w) for _li, widx, x, _y, w in layout.word_boxes}
    x_lo = min(boxes[0][0], boxes[1][0])
    x_hi = max(boxes[0][0] + boxes[0][1], boxes[1][0] + boxes[1][1])
    pad = layout.font_px  # stroke, shadow and the highlight pill spill a little
    assert diff[0] >= x_lo - pad
    assert diff[2] <= x_hi + pad


def test_uppercase_style_renders_upper_case(style):
    upper = DEFAULT_STYLES["impact-punch"]
    doc = evenly_spaced("quiet words", per_word=0.4)
    cue = build_cues(doc, 0.0, 5.0, CueOptions(max_chars_per_line=26, max_lines=2))[0]
    layout = CueLayout(cue, upper, OUT)
    assert layout.texts == [["QUIET", "WORDS"]]


def test_non_karaoke_style_emits_one_image_per_cue(tmp_path, cues):
    style = DEFAULT_STYLES["static-block"]
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path)
    assert len(assets.overlays) == len(cues)
    assert all(o.word_index is None for o in assets.overlays)


# --------------------------------------------------------------------------
# Caching and dedupe
# --------------------------------------------------------------------------


def test_identical_states_share_one_png(tmp_path, style):
    """The same phrase twice must not rasterize twice."""
    doc = words(
        [
            ("same", 0.0, 0.4),
            ("words", 0.4, 0.8),
            ("here.", 0.8, 1.2),
            ("same", 3.0, 3.4),
            ("words", 3.4, 3.8),
            ("here.", 3.8, 4.2),
        ]
    )
    cues = build_cues(doc, 0.0, 6.0, CueOptions())
    assert len(cues) == 2, "fixture must produce two identical cues"
    assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path / "job")
    assert len(assets.overlays) == 6
    assert assets.unique_images == 3, "three states, rendered once each"
    assert len(list((tmp_path / "job" / "captions").glob("*.png"))) == 3


def test_rendering_twice_writes_nothing_new(tmp_path, cues, style):
    backend = PillowCaptionBackend()
    first = backend.render(cues, style, OUT, tmp_path)
    stamps = {
        p: p.stat().st_mtime_ns for p in (tmp_path / "captions").glob("*.png")
    }
    second = backend.render(cues, style, OUT, tmp_path)
    assert [o.png_path for o in first.overlays] == [o.png_path for o in second.overlays]
    assert {
        p: p.stat().st_mtime_ns for p in (tmp_path / "captions").glob("*.png")
    } == stamps


def test_a_style_change_changes_the_filenames(tmp_path, cues, style):
    a = PillowCaptionBackend().render(cues, style, OUT, tmp_path / "a")
    other = style.model_copy(update={"highlight_fill": "#FF00FF"})
    b = PillowCaptionBackend().render(cues, other, OUT, tmp_path / "b")
    assert {Path(o.png_path).name for o in a.overlays} != {
        Path(o.png_path).name for o in b.overlays
    }


def test_backend_is_available():
    assert PillowCaptionBackend().is_available()


# --------------------------------------------------------------------------
# Samples for a human to eyeball
# --------------------------------------------------------------------------


def test_write_samples():
    """Not really an assertion -- it produces `tests/render/_samples/*.png`.

    Composited over a mid-grey card at the real 1080x1920 so the safe area and
    the caption's position in the frame can be judged, not just the pill.
    """
    SAMPLES.mkdir(parents=True, exist_ok=True)
    doc = evenly_spaced(LINE, per_word=0.42)
    written = []

    for style_name, position in (
        ("pill-karaoke", "lower_third"),
        ("clean-bold", "lower_third"),
        ("impact-punch", "center"),
        ("static-block", "lower_third"),
    ):
        style = get_style(style_name)
        cues = build_cues(doc, 0.0, 20.0, CueOptions(
            max_chars_per_line=style.max_chars_per_line, max_lines=style.max_lines
        ))
        cue = cues[0]
        layout = CueLayout(cue, style, OUT)
        safe_top, _, safe_bottom, _ = style.safe_area.insets(*OUT)
        y = anchor_y(position, layout.height, OUT[1], safe_top, safe_bottom)
        x = (OUT[0] - layout.width) // 2

        for state in (0, min(2, len(cue.words) - 1)):
            frame = Image.new("RGBA", OUT, (72, 76, 84, 255))
            # Mark the caption safe area so it is obvious whether text clears it.
            band = Image.new("RGBA", (OUT[0], safe_bottom), (200, 60, 60, 40))
            frame.alpha_composite(band, (0, OUT[1] - safe_bottom))
            frame.alpha_composite(render_cue_image(layout, state), (x, y))
            path = SAMPLES / f"{style_name}--{position}--w{state}.png"
            frame.convert("RGB").save(path, "PNG")
            written.append(path)

    # And one bare cue image on transparency, which is what ffmpeg overlays.
    style = get_style("pill-karaoke")
    cue = build_cues(doc, 0.0, 20.0, CueOptions())[0]
    bare = SAMPLES / "pill-karaoke--transparent.png"
    render_cue_image(CueLayout(cue, style, OUT), 1).save(bare, "PNG")
    written.append(bare)

    assert all(p.is_file() and p.stat().st_size > 0 for p in written)
