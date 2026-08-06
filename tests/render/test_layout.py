"""Geometry tests.

`layout.py` is shared by every engine and has no way to fail loudly at render
time -- a wrong crop produces a video that is merely *wrong*, not one that
errors. So the invariants are asserted directly and for every combination,
rather than trusting a visual check on one sample.
"""

from __future__ import annotations

import itertools

import pytest

from makeshorts.artifacts import Region
from makeshorts.render.layout import (
    LayoutError,
    LayoutOptions,
    PixelRect,
    compute_placements,
    plan_clip,
    region_pixel_rect,
    regions_by_id,
    span_times,
)
from makeshorts.select.schema import (
    Clip,
    ClipsDoc,
    CriteriaRef,
    CriterionScore,
    FocusLayout,
    HeroInsetLayout,
    OutputSpec,
    SourceSpec,
    StackLayout,
    Why,
)

from tests.render.conftest import OUTPUT_VERTICAL, SOURCE_1080P, region

VERTICAL = OUTPUT_VERTICAL


# --------------------------------------------------------------------------
# Invariants that must hold for every plan, whatever the mode
# --------------------------------------------------------------------------


def assert_sane(plan, source_size=SOURCE_1080P, output_size=VERTICAL) -> None:
    src_w, src_h = source_size
    out_w, out_h = output_size
    assert plan.placements, "a plan with no placements renders a black frame"
    for i, p in enumerate(plan.placements):
        for name, r, bw, bh in (
            ("source", p.source_rect, src_w, src_h),
            ("dest", p.dest_rect, out_w, out_h),
        ):
            assert r.x % 2 == 0 and r.y % 2 == 0, f"{name}[{i}] offset odd: {r}"
            assert r.w % 2 == 0 and r.h % 2 == 0, f"{name}[{i}] size odd: {r}"
            assert r.x >= 0 and r.y >= 0, f"{name}[{i}] negative origin: {r}"
            assert r.right <= bw, f"{name}[{i}] overflows right: {r} vs {bw}"
            assert r.bottom <= bh, f"{name}[{i}] overflows bottom: {r} vs {bh}"
            assert r.w >= 2 and r.h >= 2, f"{name}[{i}] degenerate: {r}"


def assert_aspect_preserved(plan, tol=0.02) -> None:
    """Scaling source->dest must not visibly stretch.

    Tolerance covers even-pixel rounding only; anything larger means the fit
    maths dropped a term.
    """
    for i, p in enumerate(plan.placements):
        ratio = p.source_rect.aspect / p.dest_rect.aspect
        assert abs(ratio - 1.0) < tol, (
            f"placement[{i}] region={p.region_id} distorts: "
            f"src {p.source_rect} dest {p.dest_rect} ratio {ratio:.4f}"
        )


# --------------------------------------------------------------------------
# focus
# --------------------------------------------------------------------------


@pytest.mark.parametrize("region_id", ["frame", "cam_a", "slides"])
@pytest.mark.parametrize("fit", ["cover", "contain_blur"])
def test_focus_every_region_and_fit(two_up, region_id, fit):
    plan = compute_placements(
        FocusLayout(region=region_id, fit=fit), two_up, SOURCE_1080P, VERTICAL
    )
    assert_sane(plan)
    assert_aspect_preserved(plan)
    assert plan.mode == "focus"
    assert all(p.region_id == region_id for p in plan.placements)


def test_focus_cover_fills_the_output_exactly(two_up):
    plan = compute_placements(FocusLayout(region="cam_a"), two_up, SOURCE_1080P, VERTICAL)
    assert len(plan.placements) == 1
    p = plan.placements[0]
    assert p.dest_rect.as_tuple() == (0, 0, 1080, 1920)
    assert not p.blur_backdrop


def test_focus_cover_crops_the_overflowing_axis_only(two_up):
    """cam_a is 960x1080 (8:9); the target is 9:16, i.e. narrower.

    So width must be cropped and the full height kept -- cropping height too
    would throw away picture for no reason.
    """
    plan = compute_placements(FocusLayout(region="cam_a"), two_up, SOURCE_1080P, VERTICAL)
    src = plan.placements[0].source_rect
    assert src.h == 1080, "full height should survive"
    assert src.w == pytest.approx(1080 * 1080 / 1920, abs=2)
    # Centered within the region.
    assert src.x == pytest.approx((960 - src.w) / 2, abs=2)


def test_focus_cover_on_a_wide_frame_crops_hard(two_up):
    """The whole 16:9 frame into 9:16 keeps only the middle ~32% of the width."""
    plan = compute_placements(FocusLayout(region="frame"), two_up, SOURCE_1080P, VERTICAL)
    src = plan.placements[0].source_rect
    assert src.h == 1080
    assert src.w == pytest.approx(607, abs=2)
    assert src.x == pytest.approx((1920 - src.w) / 2, abs=2)


def test_focus_contain_blur_emits_backdrop_then_foreground(two_up):
    plan = compute_placements(
        FocusLayout(region="frame", fit="contain_blur"), two_up, SOURCE_1080P, VERTICAL
    )
    assert len(plan.placements) == 2
    back, front = plan.placements
    assert back.blur_backdrop and back.role == "backdrop"
    assert back.blur_radius_px > 0
    assert back.dest_rect.as_tuple() == (0, 0, 1080, 1920), "backdrop must fill the frame"
    assert not front.blur_backdrop and front.blur_radius_px == 0
    # The whole region, uncropped, is what "contain" means.
    assert front.source_rect.as_tuple() == (0, 0, 1920, 1080)
    # Letterboxed: full width, centered vertically.
    assert front.dest_rect.w == 1080
    assert front.dest_rect.h == pytest.approx(1080 * 1080 / 1920, abs=2)
    assert front.dest_rect.y == pytest.approx((1920 - front.dest_rect.h) / 2, abs=2)
    assert plan.needs_blur


def test_focus_contain_blur_on_a_tall_region_pillarboxes(two_up):
    """A region narrower than 9:16 fits by height, leaving blur left and right."""
    regions = dict(two_up, narrow=region("narrow", "speaker", (0.4, 0.0, 0.1, 1.0)))
    plan = compute_placements(
        FocusLayout(region="narrow", fit="contain_blur"), regions, SOURCE_1080P, VERTICAL
    )
    front = plan.placements[1]
    assert front.dest_rect.h == 1920, "fits by height"
    assert front.dest_rect.w < 1080
    assert front.dest_rect.x > 0
    assert_aspect_preserved(plan)


def test_frame_is_available_even_when_not_declared():
    """`{"region": "frame"}` must work on a source with no detected regions."""
    plan = compute_placements(FocusLayout(region="frame"), {}, SOURCE_1080P, VERTICAL)
    assert_sane(plan)
    assert plan.placements[0].region_id == "frame"


def test_regions_by_id_injects_frame():
    resolved = regions_by_id([region("cam_a", "speaker", (0.0, 0.0, 0.5, 1.0))])
    assert set(resolved) == {"frame", "cam_a"}
    assert resolved["frame"].rect == (0.0, 0.0, 1.0, 1.0)


def test_unknown_region_is_a_layout_error(two_up):
    with pytest.raises(LayoutError, match="unknown region 'ghost'"):
        compute_placements(FocusLayout(region="ghost"), two_up, SOURCE_1080P, VERTICAL)


# --------------------------------------------------------------------------
# hero_inset
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corner", ["top_left", "top_right", "bottom_left", "bottom_right"]
)
@pytest.mark.parametrize("hero_fit", ["cover", "contain_blur"])
def test_hero_inset_every_corner_and_fit(two_up, corner, hero_fit):
    plan = compute_placements(
        HeroInsetLayout(
            hero="slides", inset="cam_a", inset_corner=corner, hero_fit=hero_fit
        ),
        two_up,
        SOURCE_1080P,
        VERTICAL,
    )
    assert_sane(plan)
    assert_aspect_preserved(plan)
    inset = plan.placements[-1]
    assert inset.role == "inset" and inset.region_id == "cam_a"
    left = "left" in corner
    if left:
        assert inset.dest_rect.x < 1080 // 2
    else:
        assert inset.dest_rect.right > 1080 // 2
    if corner.startswith("top"):
        assert inset.dest_rect.y < 1920 // 2
    else:
        assert inset.dest_rect.bottom > 1920 // 2


def test_hero_inset_width_follows_inset_scale(two_up):
    plan = compute_placements(
        HeroInsetLayout(hero="slides", inset="cam_a", inset_scale=0.30),
        two_up,
        SOURCE_1080P,
        VERTICAL,
    )
    inset = plan.placements[-1].dest_rect
    assert inset.w == pytest.approx(0.30 * 1080, abs=2)
    # The inset keeps cam_a's own 960:1080 aspect rather than being re-cropped.
    assert inset.aspect == pytest.approx(960 / 1080, abs=0.02)
    assert plan.placements[-1].source_rect.as_tuple() == (0, 0, 960, 1080)


def test_hero_inset_clears_the_caption_safe_area(two_up):
    safe = 260
    opts = LayoutOptions(caption_safe_bottom_px=safe, inset_margin_pct=0.03)
    plan = compute_placements(
        HeroInsetLayout(hero="slides", inset="cam_a", inset_corner="bottom_right"),
        two_up,
        SOURCE_1080P,
        VERTICAL,
        opts,
    )
    inset = plan.placements[-1].dest_rect
    assert inset.bottom <= 1920 - safe, "inset sits under the captions"


def test_hero_inset_respects_the_top_safe_area(two_up):
    opts = LayoutOptions(caption_safe_top_px=200)
    plan = compute_placements(
        HeroInsetLayout(hero="slides", inset="cam_a", inset_corner="top_left"),
        two_up,
        SOURCE_1080P,
        VERTICAL,
        opts,
    )
    assert plan.placements[-1].dest_rect.y >= 200


def test_hero_inset_is_height_capped(two_up):
    """A tall narrow inset at a big scale must not take over the frame."""
    regions = dict(two_up, tall=region("tall", "speaker", (0.0, 0.0, 0.06, 1.0)))
    plan = compute_placements(
        HeroInsetLayout(hero="slides", inset="tall", inset_scale=0.6),
        regions,
        SOURCE_1080P,
        VERTICAL,
        LayoutOptions(inset_max_height_pct=0.45),
    )
    inset = plan.placements[-1].dest_rect
    assert inset.h <= int(0.45 * 1920) + 2
    assert_aspect_preserved(plan)


def test_hero_inset_never_leaves_the_frame_even_when_over_constrained(two_up):
    """Huge inset + deep safe area + big margin: slide it back, do not clip."""
    opts = LayoutOptions(
        caption_safe_bottom_px=900, inset_margin_pct=0.2, inset_max_height_pct=1.0
    )
    plan = compute_placements(
        HeroInsetLayout(
            hero="slides", inset="cam_a", inset_corner="bottom_right", inset_scale=0.6
        ),
        two_up,
        SOURCE_1080P,
        VERTICAL,
        opts,
    )
    assert_sane(plan)


def test_hero_inset_ordering_is_back_to_front(two_up):
    """The inset must be last, or the hero would paint over it."""
    plan = compute_placements(
        HeroInsetLayout(hero="frame", inset="cam_a", hero_fit="contain_blur"),
        two_up,
        SOURCE_1080P,
        VERTICAL,
    )
    roles = [p.role for p in plan.placements]
    assert roles == ["backdrop", "primary", "inset"]


# --------------------------------------------------------------------------
# stack
# --------------------------------------------------------------------------


def _band_boxes(plan) -> list:
    """The destination *box* of each band.

    Under `cover` the band placement fills its box; under `contain_blur` the
    band is letterboxed and the backdrop is what fills it. Either way the
    first placement of each band is the one that owns the box.
    """
    boxes = []
    for p in plan.placements:
        if p.role == "backdrop":
            boxes.append(p.dest_rect)
        elif p.role == "band" and (not boxes or boxes[-1].bottom <= p.dest_rect.y):
            boxes.append(p.dest_rect)
    return boxes


@pytest.mark.parametrize("fit", ["cover", "contain_blur"])
def test_stack_two_bands_tile_the_output_exactly(two_up, fit):
    plan = compute_placements(
        StackLayout(regions=["cam_a", "slides"], fit=fit), two_up, SOURCE_1080P, VERTICAL
    )
    assert_sane(plan)
    assert_aspect_preserved(plan)
    boxes = _band_boxes(plan)
    assert [b.y for b in boxes] == [0, 960]
    assert all(b.h == 960 and b.w == 1080 for b in boxes)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_stack_of_n_tiles_without_seams(n):
    """Awkward divisions (1920/7) must still tile edge to edge and stay even."""
    regions = {
        f"r{i}": region(f"r{i}", "speaker", (i / n, 0.0, 1.0 / n, 1.0)) for i in range(n)
    }
    plan = compute_placements(
        StackLayout(regions=list(regions)), regions, SOURCE_1080P, VERTICAL
    )
    assert_sane(plan)
    boxes = _band_boxes(plan)
    assert len(boxes) == n
    assert boxes[0].y == 0
    assert boxes[-1].bottom == 1920
    for a, b in itertools.pairwise(boxes):
        assert a.bottom == b.y, f"seam between {a} and {b}"


def test_stack_gap_still_spans_the_full_height(two_up):
    plan = compute_placements(
        StackLayout(regions=["cam_a", "slides"]),
        two_up,
        SOURCE_1080P,
        VERTICAL,
        LayoutOptions(stack_gap_px=20),
    )
    bands = [p.dest_rect for p in plan.placements if p.role == "band"]
    assert bands[0].y == 0 and bands[-1].bottom == 1920
    assert bands[1].y - bands[0].bottom == 20


def test_stack_contain_blur_gives_each_band_its_own_backdrop(two_up):
    plan = compute_placements(
        StackLayout(regions=["cam_a", "slides"], fit="contain_blur"),
        two_up,
        SOURCE_1080P,
        VERTICAL,
    )
    assert [p.role for p in plan.placements] == ["backdrop", "band", "backdrop", "band"]
    for backdrop in (plan.placements[0], plan.placements[2]):
        band = plan.placements[plan.placements.index(backdrop) + 1]
        assert backdrop.dest_rect.h == 960
        assert band.dest_rect.h <= backdrop.dest_rect.h


def test_stack_of_too_many_regions_for_the_frame():
    n = 1200  # 1920/1200 = 1.6px a band; two edges round to the same value
    regions = {f"r{i}": region(f"r{i}", "speaker", (0.0, 0.0, 1.0, 1.0)) for i in range(n)}
    with pytest.raises(LayoutError, match="band"):
        compute_placements(
            StackLayout(regions=list(regions)), regions, SOURCE_1080P, VERTICAL
        )


# --------------------------------------------------------------------------
# Clamping and rounding
# --------------------------------------------------------------------------


def test_region_overflowing_the_source_is_clamped_not_rejected():
    """A hand-edited rect running past the right edge renders the visible part."""
    regions = {"wide": region("wide", "speaker", (0.6, 0.0, 0.5, 1.0))}
    r = region_pixel_rect("wide", regions, SOURCE_1080P)
    assert r.x == 1152 and r.right == 1920
    plan = compute_placements(FocusLayout(region="wide"), regions, SOURCE_1080P, VERTICAL)
    assert_sane(plan)


@pytest.mark.parametrize(
    "rect",
    [
        (-0.3, -0.3, 1.6, 1.6),  # overflows on all four sides
        (0.0, 0.9, 1.0, 0.5),  # runs off the bottom
        (0.95, 0.0, 0.5, 1.0),  # only a sliver is on screen
    ],
)
def test_clamping_keeps_every_rect_inside_the_source(rect):
    regions = {"r": region("r", "unknown", rect)}
    plan = compute_placements(FocusLayout(region="r"), regions, SOURCE_1080P, VERTICAL)
    assert_sane(plan)


def test_region_entirely_outside_the_source_is_an_error():
    regions = {"gone": region("gone", "speaker", (1.5, 0.0, 0.2, 1.0))}
    with pytest.raises(LayoutError, match="nothing to render"):
        compute_placements(FocusLayout(region="gone"), regions, SOURCE_1080P, VERTICAL)


@pytest.mark.parametrize(
    "source_size",
    [(1920, 1080), (1919, 1079), (1280, 720), (3840, 2160), (720, 1280), (640, 480)],
)
@pytest.mark.parametrize("output_size", [(1080, 1920), (720, 1280), (1081, 1921)])
def test_even_dimensions_across_odd_sources_and_outputs(two_up, source_size, output_size):
    """Odd source or output sizes must still yield even rects inside bounds."""
    for layout in (
        FocusLayout(region="cam_a"),
        FocusLayout(region="frame", fit="contain_blur"),
        HeroInsetLayout(hero="slides", inset="cam_a"),
        StackLayout(regions=["cam_a", "slides"]),
    ):
        plan = compute_placements(layout, two_up, source_size, output_size)
        assert_sane(plan, source_size, output_size)


@pytest.mark.parametrize(
    "source_size",
    [
        (1920, 1080),  # 16:9, wider than the target
        (1080, 1920),  # 9:16, exactly the target
        (1080, 1350),  # 4:5, narrower than 16:9 but still wider than target
        (600, 1600),  # 3:8, narrower than the target
        (1000, 1000),  # square
    ],
)
def test_frame_focus_across_source_aspect_ratios(source_size):
    regions = {}
    cover = compute_placements(
        FocusLayout(region="frame"), regions, source_size, VERTICAL
    )
    assert_sane(cover, source_size)
    assert_aspect_preserved(cover)
    assert cover.placements[0].dest_rect.as_tuple() == (0, 0, 1080, 1920)

    contained = compute_placements(
        FocusLayout(region="frame", fit="contain_blur"), regions, source_size, VERTICAL
    )
    assert_sane(contained, source_size)
    assert_aspect_preserved(contained)
    front = contained.placements[1].dest_rect
    assert front.w <= 1080 and front.h <= 1920
    assert front.w == 1080 or front.h == 1920, "contain must touch one pair of edges"


def test_source_matching_the_target_aspect_is_a_no_op_cover():
    plan = compute_placements(FocusLayout(region="frame"), {}, (1080, 1920), VERTICAL)
    p = plan.placements[0]
    assert p.source_rect.as_tuple() == (0, 0, 1080, 1920)
    assert p.dest_rect.as_tuple() == (0, 0, 1080, 1920)


def test_tiny_source_is_rejected():
    with pytest.raises(LayoutError, match="too small"):
        compute_placements(FocusLayout(region="frame"), {}, (1, 1), VERTICAL)


def test_placement_rects_reject_zero_extent():
    with pytest.raises(Exception):
        PixelRect(x=0, y=0, w=0, h=10)


# --------------------------------------------------------------------------
# Multi-span clips
# --------------------------------------------------------------------------


def _clip(layout) -> Clip:
    return Clip(
        id="01-test",
        title="t",
        start=100.0,
        end=140.0,
        source_text="x",
        why=Why(
            one_line="l",
            theme="t",
            scores={"hook_strength": CriterionScore(score=5, evidence="e")},
            weighted_score=5.0,
        ),
        layout=layout,
    )


def test_span_times_close_the_last_span_against_the_clip_end():
    clip = _clip(
        [
            FocusLayout(at=0.0, region="cam_a"),
            HeroInsetLayout(at=11.7, hero="slides", inset="cam_a"),
        ]
    )
    spans = span_times(clip)
    assert [(s, e) for s, e, _ in spans] == [(0.0, 11.7), (11.7, 40.0)]


def test_span_times_on_a_single_layout_clip():
    spans = span_times(_clip(FocusLayout(region="frame")))
    assert len(spans) == 1
    assert spans[0][:2] == (0.0, 40.0)


def test_plan_clip_covers_the_whole_duration_without_gaps(two_up):
    clip = _clip(
        [
            FocusLayout(at=0.0, region="cam_a"),
            StackLayout(at=8.0, regions=["cam_a", "slides"]),
            FocusLayout(at=25.0, region="slides", fit="contain_blur"),
        ]
    )
    plans = plan_clip(clip, two_up, SOURCE_1080P, VERTICAL)
    assert [sp.start for sp in plans] == [0.0, 8.0, 25.0]
    assert plans[-1].end == clip.duration
    assert sum(sp.duration for sp in plans) == pytest.approx(clip.duration)
    for sp in plans:
        assert_sane(sp.plan)
        assert_aspect_preserved(sp.plan)


# --------------------------------------------------------------------------
# Several frame-aligned source files
#
# The Zoom case: sharp slides in one export, the only usable face in another,
# and ONE clip composites both. Each region must be cropped in the pixel space
# of its own file. Getting this wrong yields a plausible-looking rect that is
# simply the wrong part of the picture, so these tests assert exact geometry
# and not merely that everything landed inside some frame.
# --------------------------------------------------------------------------

SLIDES_SIZE = (2378, 1410)  # a screenshare render; ~1.686:1
CAM_SIZE = (1280, 720)  # the active-speaker render; 1.778:1
SOURCE_SIZES = {"slides": SLIDES_SIZE, "cam": CAM_SIZE}


def sregion(rid: str, kind: str, rect, source: str) -> Region:
    return Region(id=rid, kind=kind, rect=rect, source=source)


@pytest.fixture
def two_file_regions() -> dict[str, Region]:
    """What `ClipsDoc.resolved_regions()` yields for a two-file job."""
    return {
        "frame": sregion("frame", "unknown", (0.0, 0.0, 1.0, 1.0), "slides"),
        "slides_frame": sregion("slides_frame", "unknown", (0.0, 0.0, 1.0, 1.0), "slides"),
        "cam_frame": sregion("cam_frame", "unknown", (0.0, 0.0, 1.0, 1.0), "cam"),
        "deck": sregion("deck", "slide", (0.05, 0.10, 0.90, 0.80), "slides"),
        "face": sregion("face", "speaker", (0.30, 0.05, 0.40, 0.90), "cam"),
    }


def _two_file_doc(layout) -> ClipsDoc:
    """A real two-source ClipsDoc, so the ClipsDoc path is exercised too."""
    return ClipsDoc(
        job="zoom-export",
        sources={
            "slides": SourceSpec(
                path="screenshare.mp4",
                duration=3600.0,
                resolution=SLIDES_SIZE,
                regions=[Region(id="deck", kind="slide", rect=(0.05, 0.10, 0.90, 0.80))],
            ),
            "cam": SourceSpec(
                path="speaker.mp4",
                duration=3600.0,
                resolution=CAM_SIZE,
                regions=[Region(id="face", kind="speaker", rect=(0.30, 0.05, 0.40, 0.90))],
            ),
        },
        output=OutputSpec(width=1080, height=1920),
        criteria_ref=CriteriaRef(file="config/criteria.yaml", version="1", sha256="0" * 64),
        clips=[_clip(layout)],
    )


def assert_within_its_own_source(plan) -> None:
    """Every source_rect inside the frame of the file it names -- not inside
    whichever source happened to be first."""
    for i, p in enumerate(plan.placements):
        assert p.source in plan.source_sizes, f"placement[{i}] names an unknown source"
        w, h = plan.size_of(p)
        r = p.source_rect
        assert r.right <= w and r.bottom <= h, (
            f"placement[{i}] region={p.region_id} source={p.source} rect {r} "
            f"escapes its {w}x{h} file"
        )
        assert r.x % 2 == 0 and r.y % 2 == 0 and r.w % 2 == 0 and r.h % 2 == 0


@pytest.mark.parametrize("via", ["doc", "mapping"])
def test_hero_inset_crops_each_source_in_its_own_pixel_space(two_file_regions, via):
    """The whole point: a 2378x1410 hero and a 1280x720 inset in one frame.

    Both files are 16:9-ish but neither the size nor the aspect matches, so a
    crop computed against the wrong one lands inside the frame and is still
    the wrong pixels. Hence exact expected rects.
    """
    layout = HeroInsetLayout(hero="slides_frame", inset="cam_frame")
    if via == "doc":
        doc = _two_file_doc(layout)
        plan = compute_placements(layout, doc.resolved_regions(), doc, VERTICAL)
    else:
        plan = compute_placements(layout, two_file_regions, SOURCE_SIZES, VERTICAL)

    assert_within_its_own_source(plan)
    assert_aspect_preserved(plan)

    hero, inset = plan.placements
    assert (hero.source, inset.source) == ("slides", "cam")

    # Cover of a 2378x1410 frame into 9:16: full height, centered width of
    # 1410 * 1080/1920. Measured in the *cam* file this would be ~404 wide.
    assert hero.source_rect.as_tuple() == (792, 0, 794, 1410)
    assert hero.dest_rect.as_tuple() == (0, 0, 1080, 1920)

    # The inset is the whole cam frame -- 1280x720, not 2378x1410.
    assert inset.source_rect.as_tuple() == (0, 0, 1280, 720)
    assert inset.dest_rect.aspect == pytest.approx(1280 / 720, abs=0.02)

    assert plan.source_sizes == {"slides": SLIDES_SIZE, "cam": CAM_SIZE}
    assert plan.source_names == ["slides", "cam"]


def test_hero_inset_on_declared_regions_from_two_files(two_file_regions):
    """Same, for regions that are sub-rects rather than whole frames."""
    layout = HeroInsetLayout(hero="deck", inset="face", inset_scale=0.30)
    plan = compute_placements(layout, two_file_regions, SOURCE_SIZES, VERTICAL)
    assert_within_its_own_source(plan)
    assert_aspect_preserved(plan)

    hero, inset = plan.placements
    # deck is 0.90x0.80 of 2378x1410 = 2140.2 x 1128, at (118.9, 141).
    assert hero.source == "slides"
    assert hero.source_rect.h == 1128
    assert hero.source_rect.w == pytest.approx(1128 * 1080 / 1920, abs=2)
    # face is 0.40x0.90 of 1280x720 = 512 x 648, at (384, 36).
    assert inset.source == "cam"
    assert inset.source_rect.as_tuple() == (384, 36, 512, 648)
    assert inset.dest_rect.aspect == pytest.approx(512 / 648, abs=0.02)


def test_hero_inset_swapped_sources_is_not_symmetric(two_file_regions):
    """Guards against a crop that is right only because both files agree."""
    plan = compute_placements(
        HeroInsetLayout(hero="cam_frame", inset="slides_frame"),
        two_file_regions,
        SOURCE_SIZES,
        VERTICAL,
    )
    assert_within_its_own_source(plan)
    hero, inset = plan.placements
    assert (hero.source, inset.source) == ("cam", "slides")
    # Cover of 1280x720 into 9:16: full height, 720 * 1080/1920 = 405px wide,
    # centered at x=437.5 and evened to (438, 404).
    assert hero.source_rect.as_tuple() == (438, 0, 404, 720)
    assert inset.source_rect.as_tuple() == (0, 0, 2378, 1410)


@pytest.mark.parametrize("fit", ["cover", "contain_blur"])
def test_stack_across_two_sources(two_file_regions, fit):
    plan = compute_placements(
        StackLayout(regions=["slides_frame", "face"], fit=fit),
        two_file_regions,
        SOURCE_SIZES,
        VERTICAL,
    )
    assert_within_its_own_source(plan)
    assert_aspect_preserved(plan)
    by_region = {p.region_id: p for p in plan.placements}
    assert by_region["slides_frame"].source == "slides"
    assert by_region["face"].source == "cam"
    assert by_region["face"].source_rect.right <= 1280
    assert by_region["slides_frame"].source_rect.right <= 2378
    boxes = _band_boxes(plan)
    assert [b.y for b in boxes] == [0, 960]


def test_focus_on_the_second_source_uses_only_that_file(two_file_regions):
    plan = compute_placements(
        FocusLayout(region="face"), two_file_regions, SOURCE_SIZES, VERTICAL
    )
    assert plan.source_names == ["cam"]
    assert plan.source_sizes == {"cam": CAM_SIZE}
    # No decode is asked of the slides file when nothing reads from it.
    assert "slides" not in plan.source_sizes
    assert_within_its_own_source(plan)


def test_primary_source_dims_are_the_first_source(two_file_regions):
    plan = compute_placements(
        HeroInsetLayout(hero="slides_frame", inset="cam_frame"),
        two_file_regions,
        SOURCE_SIZES,
        VERTICAL,
    )
    assert (plan.source_width, plan.source_height) == SLIDES_SIZE


def test_plan_clip_takes_a_clips_doc_and_switches_sources_per_span(two_file_regions):
    clip = _clip(
        [
            FocusLayout(at=0.0, region="slides_frame"),
            HeroInsetLayout(at=10.0, hero="slides_frame", inset="face"),
            FocusLayout(at=30.0, region="cam_frame"),
        ]
    )
    doc = _two_file_doc(clip.layout)
    plans = plan_clip(clip, doc.resolved_regions(), doc, VERTICAL)
    assert [sp.plan.source_names for sp in plans] == [
        ["slides"],
        ["slides", "cam"],
        ["cam"],
    ]
    for sp in plans:
        assert_within_its_own_source(sp.plan)
        assert_aspect_preserved(sp.plan)
    # A span reading one file lists only that file.
    assert plans[2].plan.source_sizes == {"cam": CAM_SIZE}


def test_region_pixel_rect_honours_the_regions_own_source(two_file_regions):
    assert region_pixel_rect("face", two_file_regions, SOURCE_SIZES).as_tuple() == (
        384,
        36,
        512,
        648,
    )
    assert region_pixel_rect("deck", two_file_regions, SOURCE_SIZES).as_tuple() == (
        118,
        142,
        2142,
        1128,
    )


# -- single-source jobs are unchanged ---------------------------------------


def test_single_source_placements_say_main(two_up):
    """Every placement of a one-file job names `main`, so an engine can read
    `p.source` unconditionally."""
    for layout in (
        FocusLayout(region="cam_a"),
        FocusLayout(region="frame", fit="contain_blur"),
        HeroInsetLayout(hero="slides", inset="cam_a"),
        StackLayout(regions=["cam_a", "slides"], fit="contain_blur"),
    ):
        plan = compute_placements(layout, two_up, SOURCE_1080P, VERTICAL)
        assert plan.placements
        assert all(p.source == "main" for p in plan.placements), layout.mode
        assert plan.source_sizes == {"main": SOURCE_1080P}
        assert plan.source_names == ["main"]
        assert (plan.source_width, plan.source_height) == SOURCE_1080P


def test_single_source_clips_doc_still_says_main():
    doc = ClipsDoc(
        job="one-file",
        source=SourceSpec(
            path="source.mp4",
            duration=600.0,
            resolution=SOURCE_1080P,
            regions=[Region(id="cam_a", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0))],
        ),
        criteria_ref=CriteriaRef(file="config/criteria.yaml", version="1", sha256="0" * 64),
        clips=[_clip(FocusLayout(region="cam_a"))],
    )
    plan = compute_placements(
        FocusLayout(region="cam_a"), doc.resolved_regions(), doc, VERTICAL
    )
    assert [p.source for p in plan.placements] == ["main"]
    # Identical geometry to the bare-tuple call it replaces.
    legacy = compute_placements(
        FocusLayout(region="cam_a"), doc.resolved_regions(), SOURCE_1080P, VERTICAL
    )
    assert plan.placements == legacy.placements


def test_source_size_keyword_still_works(two_up):
    """The old spelling of the argument, kept for existing callers."""
    a = compute_placements(FocusLayout(region="cam_a"), two_up, source_size=SOURCE_1080P)
    b = compute_placements(FocusLayout(region="cam_a"), two_up, SOURCE_1080P, VERTICAL)
    assert a.placements == b.placements
    assert plan_clip(_clip(FocusLayout(region="cam_a")), two_up, source_size=SOURCE_1080P)


# -- refusing to guess ------------------------------------------------------


def test_one_resolution_for_two_sources_is_refused(two_file_regions):
    """Silently cropping the cam region in slides pixels is the bug this whole
    change exists to prevent; a single resolution must not be stretched over
    two files."""
    with pytest.raises(LayoutError, match="different sources"):
        compute_placements(
            HeroInsetLayout(hero="slides_frame", inset="cam_frame"),
            two_file_regions,
            SLIDES_SIZE,
            VERTICAL,
        )


def test_region_naming_an_unknown_source_is_an_error(two_file_regions):
    regions = dict(two_file_regions, ghost=sregion("ghost", "slide", (0, 0, 1, 1), "gone"))
    with pytest.raises(LayoutError, match="names source 'gone'"):
        compute_placements(FocusLayout(region="ghost"), regions, SOURCE_SIZES, VERTICAL)


def test_missing_and_doubled_source_arguments(two_up):
    with pytest.raises(LayoutError, match="required"):
        compute_placements(FocusLayout(region="frame"), two_up)
    with pytest.raises(LayoutError, match="not both"):
        compute_placements(
            FocusLayout(region="frame"), two_up, SOURCE_1080P, source_size=SOURCE_1080P
        )
    with pytest.raises(LayoutError, match="no source resolutions"):
        compute_placements(FocusLayout(region="frame"), two_up, {}, VERTICAL)


def test_a_tiny_source_is_rejected_by_name():
    with pytest.raises(LayoutError, match="'cam' 1x1 is too small"):
        compute_placements(
            FocusLayout(region="frame"), {}, {"cam": (1, 1)}, VERTICAL
        )


# --------------------------------------------------------------------------
# Purity — this module is shared by every engine and must stay dependency-free
# --------------------------------------------------------------------------


def test_layout_module_has_no_engine_vocabulary():
    from pathlib import Path

    import makeshorts.render.layout as mod

    text = Path(mod.__file__).read_text().lower()
    for token in ("ffmpeg", "filter_complex", "-vf", "libx264", "boxblur", "overlay="):
        assert token not in text, f"layout.py mentions {token!r}"


def test_layout_module_imports_nothing_heavy():
    import ast
    from pathlib import Path

    import makeshorts.render.layout as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "math", "collections", "typing", "pydantic", "makeshorts"}
