"""Pure geometry: semantic layout + normalized regions -> concrete pixel rects.

This module is deliberately NOT part of any engine. It has no I/O, no
subprocesses, and no vocabulary from any particular renderer. It answers one
question -- *which pixels of the source go where in the output frame* -- and
returns plain data. Translating that data into a filtergraph, a React
component, or a Resolve timeline is the engine's job.

The whole contract is:

    compute_placements(layout, regions, sources, output_size) -> LayoutPlan

`LayoutPlan.placements` is an ordered list, **back to front**. An engine
composites them in that order: paint placement 0, then 1 on top, and so on.

`sources` is where the multi-file case lives. A job may have several
frame-aligned input files -- a Zoom export gives sharp slides in one render
and the only usable face in another -- and **a single layout may mix them**.
So a region is never measured against "the" source resolution; it is measured
against the resolution of *its own* file, named by `Region.source`. Pass the
`ClipsDoc` (or a name -> resolution mapping) and each region is cropped in the
right pixel space. A bare `(w, h)` tuple still works for single-source jobs.
Every `Placement` says which source it reads from, in `Placement.source`.

Three invariants the engine may rely on:

1. `source_rect` and `dest_rect` have (near-)equal aspect ratios, so scaling
   one to the other never distorts. "Near" because both are rounded to even
   pixel boundaries at the end; the residual is under a pixel.
2. Every `source_rect` lies strictly inside the frame of *its own* source
   (`LayoutPlan.source_sizes[placement.source]`) and every `dest_rect` lies
   strictly inside the output frame. Nothing needs re-clamping downstream.
3. `placement.source` is always a key of `LayoutPlan.source_sizes`.

Clamping: a region rect that extends past the source frame (a hand-edited
`clips.json` with ``[0.6, 0, 0.5, 1]``) is silently clamped to the frame
bounds rather than rejected -- the linter is the place to complain about a bad
rect, and a renderer that dies on a 1% overhang helps nobody. A region that
clamps away to nothing raises `LayoutError`.

Even dimensions: every x, y, w, h returned is an even integer. Chroma-
subsampled encoders require even frame dimensions, and even offsets keep
chroma planes aligned. This is the one encoder-shaped fact this module admits,
and it is a property of video in general rather than of any single tool.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from makeshorts.artifacts import IMPLICIT_FRAME_REGION_ID, Rect, Region, Strict
from makeshorts.select.schema import (
    PRIMARY_SOURCE_NAME,
    Clip,
    ClipsDoc,
    FocusLayout,
    HeroInsetLayout,
    Layout,
    StackLayout,
)

__all__ = [
    "LayoutError",
    "PixelRect",
    "Placement",
    "PlacementRole",
    "LayoutPlan",
    "LayoutOptions",
    "SourceSizes",
    "Sources",
    "SpanPlan",
    "compute_placements",
    "plan_clip",
    "span_times",
    "region_pixel_rect",
    "regions_by_id",
]

# Source name -> (width, height) in pixels.
SourceSizes = Mapping[str, tuple[int, int]]

# What every entry point will accept for "how big are the source files":
# the edit list itself (preferred -- it knows the names), a name -> resolution
# mapping, or a single `(w, h)` for the single-file case.
Sources = ClipsDoc | SourceSizes | tuple[int, int]


class LayoutError(ValueError):
    """A layout that cannot be turned into geometry.

    Unknown region id, or a region that clamps to zero area. Both are things
    `ms lint` should have caught; this is the backstop.
    """


# --------------------------------------------------------------------------
# Rounding helpers
#
# All geometry is computed in float and quantized exactly once, at the end.
# Quantizing intermediate results compounds error across a hero_inset's four
# nested rects.
# --------------------------------------------------------------------------


def _even(value: float) -> int:
    """Round to the nearest even integer, halves away from zero.

    Plain `round()` is banker's rounding, which makes test expectations
    surprising for no benefit here.
    """
    return int(math.floor(value / 2.0 + 0.5)) * 2


def _even_floor(value: int) -> int:
    """Largest even integer <= value. Used to keep a rect inside an odd bound."""
    return value - (value % 2)


class PixelRect(Strict):
    """An axis-aligned rectangle in integer pixels. All fields are even."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class _FRect:
    """Internal float rectangle. Never escapes this module."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h > 0 else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_FRect({self.x:.2f}, {self.y:.2f}, {self.w:.2f}, {self.h:.2f})"


def _quantize(r: _FRect, bound_w: int, bound_h: int, min_extent: int = 2) -> PixelRect:
    """Snap a float rect to even pixels, fully inside (0,0,bound_w,bound_h).

    Edges are rounded independently and the width derived from them, so a rect
    that abuts a neighbour (stack bands) tiles exactly with no seam.
    """
    max_x = _even_floor(bound_w)
    max_y = _even_floor(bound_h)
    if max_x < min_extent or max_y < min_extent:
        raise LayoutError(f"frame {bound_w}x{bound_h} is too small to place anything in")

    x0 = min(max(_even(r.x), 0), max_x - min_extent)
    x1 = min(max(_even(r.x + r.w), x0 + min_extent), max_x)
    if x1 - x0 < min_extent:  # ran out of room at the right edge; back off left
        x0 = x1 - min_extent

    y0 = min(max(_even(r.y), 0), max_y - min_extent)
    y1 = min(max(_even(r.y + r.h), y0 + min_extent), max_y)
    if y1 - y0 < min_extent:
        y0 = y1 - min_extent

    return PixelRect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


# --------------------------------------------------------------------------
# Fit maths
# --------------------------------------------------------------------------


def _cover_crop(src: _FRect, dest_aspect: float) -> _FRect:
    """Sub-rect of `src` with `dest_aspect`, centered. Overflow is discarded.

    This is the "fill the box, crop what doesn't fit" half of `fit: "cover"`.
    """
    if src.h <= 0 or src.w <= 0 or dest_aspect <= 0:
        raise LayoutError(f"cannot cover-crop a degenerate rect: {src!r}")
    if src.aspect > dest_aspect:
        w = src.h * dest_aspect
        return _FRect(src.x + (src.w - w) / 2.0, src.y, w, src.h)
    h = src.w / dest_aspect
    return _FRect(src.x, src.y + (src.h - h) / 2.0, src.w, h)


def _contain_box(src_aspect: float, box: _FRect) -> _FRect:
    """Largest rect of `src_aspect` fitting inside `box`, centered.

    The "letterbox it whole" half of `fit: "contain_blur"`.
    """
    if src_aspect <= 0:
        raise LayoutError("cannot contain a degenerate aspect ratio")
    if src_aspect > box.aspect:
        w = box.w
        h = box.w / src_aspect
    else:
        h = box.h
        w = box.h * src_aspect
    return _FRect(box.x + (box.w - w) / 2.0, box.y + (box.h - h) / 2.0, w, h)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

# What a placement is *for*. Engines mostly do not need to branch on this, but
# it makes plans readable in a receipt and lets an engine skip, say, backdrops
# when a cheap preview is wanted.
PlacementRole = Literal["backdrop", "primary", "inset", "band"]


class Placement(Strict):
    """One source rectangle scaled into one destination rectangle.

    Composited in list order, back to front. `source_rect` is in the pixels of
    the file named by `source`, `dest_rect` in *output* pixels.
    """

    region_id: str
    # Which input file this reads from -- a key of `LayoutPlan.source_sizes`,
    # and of `ClipsDoc.resolved_sources()`. Single-source jobs see `"main"`
    # here and can ignore it; a two-file layout is exactly why it exists, and
    # `source_rect` means nothing without it.
    source: str
    source_rect: PixelRect
    dest_rect: PixelRect
    # True when this layer is the scaled-and-blurred fill behind a
    # `contain_blur` letterbox. The engine blurs it; how is its business.
    blur_backdrop: bool = False
    # Suggested blur strength in output pixels. 0 when blur_backdrop is False.
    blur_radius_px: int = 0
    role: PlacementRole = "primary"


class LayoutOptions(Strict):
    """Knobs that are presentation policy rather than schema.

    These come from `config/render.yaml` or from the caption style, never from
    `clips.json` -- the edit list says *hero_inset, bottom_right, 0.28*, and
    this decides what margin that implies in pixels.
    """

    # Inset margin from the frame edge, as a fraction of output width.
    inset_margin_pct: float = Field(default=0.03, ge=0.0, le=0.25)
    # An inset is never allowed to be taller than this fraction of the output;
    # a tall narrow region would otherwise produce an inset that dominates.
    inset_max_height_pct: float = Field(default=0.45, gt=0.0, le=1.0)
    # Keep-out zones so a bottom_* inset does not sit under the captions.
    caption_safe_bottom_px: int = Field(default=0, ge=0)
    caption_safe_top_px: int = Field(default=0, ge=0)
    # Blur strength for contain_blur backdrops, as a fraction of output width.
    blur_radius_pct: float = Field(default=0.02, ge=0.0, le=0.25)
    # Gap between stack bands, in output pixels. Even values only; odd is
    # rounded down. Bands still tile the full height exactly.
    stack_gap_px: int = Field(default=0, ge=0)


class LayoutPlan(Strict):
    """Everything an engine needs to build one frame's composition."""

    mode: Literal["focus", "hero_inset", "stack"]
    output_width: int
    output_height: int
    # Resolution of every source this plan touches, by name. This is the
    # authoritative one: `placements[i].source_rect` is in the pixels of
    # `source_sizes[placements[i].source]`, and mixing two files in one layout
    # means there is no single "the source size" to speak of.
    source_sizes: dict[str, tuple[int, int]] = Field(default_factory=dict)
    # The *primary* source's dimensions -- the first entry of `sources`, or the
    # only file on a single-source job. Kept because a single-source engine
    # (and every existing caller) reasonably wants one number, and because a
    # receipt reads better with it. Do NOT crop against these: with two
    # sources in one layout they are right for some placements and silently
    # wrong for the rest. Use `source_sizes[p.source]`.
    source_width: int
    source_height: int
    placements: list[Placement]

    @property
    def needs_blur(self) -> bool:
        return any(p.blur_backdrop for p in self.placements)

    @property
    def source_names(self) -> list[str]:
        """Every distinct source used, in back-to-front first-use order.

        An engine wires one input per name; a single-source plan gets a
        one-element list."""
        seen: dict[str, None] = {}
        for p in self.placements:
            seen.setdefault(p.source, None)
        return list(seen)

    def size_of(self, placement: Placement) -> tuple[int, int]:
        """The frame the placement's `source_rect` is measured in."""
        return self.source_sizes[placement.source]


class SpanPlan(Strict):
    """A layout plan plus the clip-relative window it applies to."""

    start: float  # seconds from clip start
    end: float  # seconds from clip start
    plan: LayoutPlan

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# Region resolution
# --------------------------------------------------------------------------


def _lookup(region_id: str, regions: Mapping[str, Region]) -> Rect:
    if region_id in regions:
        return regions[region_id].rect
    if region_id == IMPLICIT_FRAME_REGION_ID:
        # `frame` works even when a caller passed a bare dict without it.
        return (0.0, 0.0, 1.0, 1.0)
    raise LayoutError(
        f"unknown region {region_id!r}; known: {sorted(regions) or ['<none>']}"
    )


class _Src:
    """A region resolved against the file it is actually measured in.

    Carrying the source name and its dimensions alongside the rect is what
    keeps a two-file layout honest: every quantize downstream is bounded by
    *this* file rather than by whichever resolution happened to be in scope.
    """

    __slots__ = ("name", "width", "height", "rect")

    def __init__(self, name: str, width: int, height: int, rect: _FRect) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.rect = rect

    @property
    def aspect(self) -> float:
        return self.rect.aspect


class _SourceTable:
    """Name -> resolution, plus the rule for which name a region belongs to.

    Built from a `ClipsDoc`, a plain mapping, or -- the single-file spelling
    every existing caller uses -- one `(w, h)` tuple.
    """

    __slots__ = ("sizes", "primary", "single", "used")

    def __init__(
        self, sizes: dict[str, tuple[int, int]], primary: str, single: bool
    ) -> None:
        if not sizes:
            raise LayoutError("no source resolutions given")
        for name, (w, h) in sizes.items():
            if w < 2 or h < 2:
                raise LayoutError(f"source {name!r} {w}x{h} is too small")
        self.sizes = sizes
        self.primary = primary
        self.single = single
        self.used: dict[str, tuple[int, int]] = {}

    def resolve(self, region_id: str, regions: Mapping[str, Region]) -> _Src:
        """The region's rect in its own file's pixels, clamped to that frame."""
        rect = _lookup(region_id, regions)
        region = regions.get(region_id)
        name = (region.source if region is not None else None) or self.primary

        if name in self.sizes:
            src_w, src_h = self.sizes[name]
        elif self.single:
            # One resolution was handed in, so it stands for whatever the
            # region calls its source -- but only as long as there is only
            # one. Two names and one resolution means at least one crop would
            # be computed in the wrong pixel space, which produces a wrong
            # video rather than an error. Refuse instead.
            src_w, src_h = self.sizes[self.primary]
        else:
            raise LayoutError(
                f"region {region_id!r} names source {name!r}, which has no "
                f"resolution; known sources: {sorted(self.sizes)}"
            )

        self.used[name] = (src_w, src_h)
        if self.single and len(self.used) > 1:
            raise LayoutError(
                f"regions in this layout come from different sources "
                f"({sorted(self.used)}) but only one resolution was given; pass "
                f"the ClipsDoc (or a name -> resolution mapping) so each region "
                f"is cropped against its own file"
            )

        x, y, w, h = rect
        x0 = min(max(x, 0.0), 1.0)
        y0 = min(max(y, 0.0), 1.0)
        x1 = min(max(x + w, 0.0), 1.0)
        y1 = min(max(y + h, 0.0), 1.0)
        fr = _FRect(x0 * src_w, y0 * src_h, (x1 - x0) * src_w, (y1 - y0) * src_h)
        if fr.w < 2 or fr.h < 2:
            raise LayoutError(
                f"region {region_id!r} rect {(x, y, w, h)} clamps to "
                f"{fr.w:.1f}x{fr.h:.1f}px in a {src_w}x{src_h} source -- nothing to render"
            )
        return _Src(name, src_w, src_h, fr)


def _source_table(sources: Sources) -> _SourceTable:
    """Normalize whatever the caller passed into one lookup table."""
    if isinstance(sources, ClipsDoc):
        specs = sources.resolved_sources()
        sizes = {name: (int(s.resolution[0]), int(s.resolution[1]))
                 for name, s in specs.items()}
        return _SourceTable(sizes, sources.primary_source_name, single=False)

    if isinstance(sources, Mapping):
        sizes = {name: (int(wh[0]), int(wh[1])) for name, wh in sources.items()}
        if not sizes:
            raise LayoutError("no source resolutions given")
        # Insertion order decides the primary, matching ClipsDoc.
        return _SourceTable(sizes, next(iter(sizes)), single=len(sizes) == 1)

    try:
        w, h = sources  # type: ignore[misc]
    except (TypeError, ValueError):  # pragma: no cover - caller error
        raise LayoutError(
            f"cannot read source resolutions from {type(sources).__name__}; pass a "
            f"ClipsDoc, a name -> (w, h) mapping, or a single (w, h)"
        ) from None
    return _SourceTable(
        {PRIMARY_SOURCE_NAME: (int(w), int(h))}, PRIMARY_SOURCE_NAME, single=True
    )


def _resolve_sources(sources: Sources | None, source_size: Sources | None) -> _SourceTable:
    """`source_size=` is the old keyword for the same argument."""
    if sources is None and source_size is None:
        raise LayoutError("source resolutions are required")
    if sources is not None and source_size is not None:
        raise LayoutError("pass `sources` or `source_size`, not both")
    chosen = sources if sources is not None else source_size
    assert chosen is not None
    return _source_table(chosen)


def region_pixel_rect(
    region_id: str,
    regions: Mapping[str, Region],
    sources: Sources | None = None,
    *,
    source_size: Sources | None = None,
) -> PixelRect:
    """A region's own rect in its source's pixels, clamped and evened.

    Exposed because callers outside layout (thumbnailers, debug overlays) want
    it without going through a full plan.
    """
    src = _resolve_sources(sources, source_size).resolve(region_id, regions)
    return _quantize(src.rect, src.width, src.height)


# --------------------------------------------------------------------------
# The three modes
# --------------------------------------------------------------------------


def _fill_box(
    region_id: str,
    src: _Src,
    box: _FRect,
    fit: str,
    out_w: int,
    out_h: int,
    opts: LayoutOptions,
    role: PlacementRole,
) -> list[Placement]:
    """Put one region into one destination box under the given fit.

    `cover` -> a single placement whose source is the box-aspect center crop.
    `contain_blur` -> a blurred cover placement filling the box, then the whole
    region letterboxed on top of it.

    This is the primitive; focus, each stack band, and hero all reduce to it.
    Every crop is bounded by `src`'s own file, never by a sibling region's.
    """
    src_w, src_h = src.width, src.height
    if fit == "cover":
        crop = _cover_crop(src.rect, box.aspect)
        return [
            Placement(
                region_id=region_id,
                source=src.name,
                source_rect=_quantize(crop, src_w, src_h),
                dest_rect=_quantize(box, out_w, out_h),
                role=role,
            )
        ]

    if fit != "contain_blur":  # pragma: no cover - schema restricts Fit
        raise LayoutError(f"unknown fit {fit!r}")

    inner = _contain_box(src.aspect, box)
    blur_px = _even(opts.blur_radius_pct * out_w)
    placements = [
        Placement(
            region_id=region_id,
            source=src.name,
            source_rect=_quantize(_cover_crop(src.rect, box.aspect), src_w, src_h),
            dest_rect=_quantize(box, out_w, out_h),
            blur_backdrop=True,
            blur_radius_px=max(blur_px, 2),
            role="backdrop",
        )
    ]
    # When the region already matches the box aspect the backdrop is invisible
    # -- the foreground covers it exactly. Emit it anyway: an engine that drops
    # a no-op layer is doing an optimization, and this module does not
    # second-guess it. (It is one extra scale on a handful of frames.)
    placements.append(
        Placement(
            region_id=region_id,
            source=src.name,
            source_rect=_quantize(src.rect, src_w, src_h),
            dest_rect=_quantize(inner, out_w, out_h),
            role=role,
        )
    )
    return placements


def _plan_focus(
    layout: FocusLayout,
    regions: Mapping[str, Region],
    table: _SourceTable,
    out_w: int,
    out_h: int,
    opts: LayoutOptions,
) -> list[Placement]:
    src = table.resolve(layout.region, regions)
    box = _FRect(0.0, 0.0, float(out_w), float(out_h))
    return _fill_box(layout.region, src, box, layout.fit, out_w, out_h, opts, "primary")


def _plan_hero_inset(
    layout: HeroInsetLayout,
    regions: Mapping[str, Region],
    table: _SourceTable,
    out_w: int,
    out_h: int,
    opts: LayoutOptions,
) -> list[Placement]:
    # The whole point of the multi-source work: these two may live in files of
    # different resolution *and* different aspect, so each is resolved against
    # its own.
    hero_src = table.resolve(layout.hero, regions)
    inset_src = table.resolve(layout.inset, regions)

    placements = _fill_box(
        layout.hero,
        hero_src,
        _FRect(0.0, 0.0, float(out_w), float(out_h)),
        layout.hero_fit,
        out_w,
        out_h,
        opts,
        "primary",
    )

    # The inset keeps its own aspect ratio -- cropping it to some arbitrary box
    # shape would be a second editorial decision the edit list never made.
    ins_w = layout.inset_scale * out_w
    ins_h = ins_w / inset_src.aspect
    max_h = opts.inset_max_height_pct * out_h
    if ins_h > max_h:
        ins_h = max_h
        ins_w = ins_h * inset_src.aspect

    margin = opts.inset_margin_pct * out_w
    left = "left" in layout.inset_corner
    top = layout.inset_corner.startswith("top")
    x = margin if left else out_w - margin - ins_w
    if top:
        y = margin + opts.caption_safe_top_px
    else:
        y = out_h - opts.caption_safe_bottom_px - margin - ins_h

    # A large inset plus a deep caption safe area can push the box off-frame.
    # Sliding it back beats clipping it.
    x = min(max(x, 0.0), max(out_w - ins_w, 0.0))
    y = min(max(y, 0.0), max(out_h - ins_h, 0.0))

    inset_box = _FRect(x, y, ins_w, ins_h)
    placements.append(
        Placement(
            region_id=layout.inset,
            source=inset_src.name,
            source_rect=_quantize(inset_src.rect, inset_src.width, inset_src.height),
            dest_rect=_quantize(inset_box, out_w, out_h),
            role="inset",
        )
    )
    return placements


def _plan_stack(
    layout: StackLayout,
    regions: Mapping[str, Region],
    table: _SourceTable,
    out_w: int,
    out_h: int,
    opts: LayoutOptions,
) -> list[Placement]:
    n = len(layout.regions)
    gap = _even_floor(opts.stack_gap_px)
    # Band edges are computed on the ideal split and evened independently, so
    # consecutive bands share an edge exactly. No seam, no off-by-one, and the
    # bands always sum to the full output height.
    edges = [_even(i * out_h / n) for i in range(n + 1)]
    edges[0] = 0
    edges[-1] = _even_floor(out_h)

    placements: list[Placement] = []
    for i, region_id in enumerate(layout.regions):
        top = edges[i] + (gap // 2 if i > 0 else 0)
        bottom = edges[i + 1] - (gap // 2 if i < n - 1 else 0)
        if bottom - top < 2:
            raise LayoutError(
                f"stack of {n} regions leaves band {i} only {bottom - top}px tall "
                f"in a {out_w}x{out_h} output"
            )
        src = table.resolve(region_id, regions)
        box = _FRect(0.0, float(top), float(out_w), float(bottom - top))
        placements.extend(
            _fill_box(region_id, src, box, layout.fit, out_w, out_h, opts, "band")
        )
    return placements


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def compute_placements(
    layout: Layout,
    regions: Mapping[str, Region],
    sources: Sources | None = None,
    output_size: tuple[int, int] = (1080, 1920),
    options: LayoutOptions | None = None,
    *,
    source_size: Sources | None = None,
) -> LayoutPlan:
    """Turn one semantic layout into concrete pixel placements.

    `regions` is normally `ClipsDoc.resolved_regions()`, which already contains
    the implicit `frame` (and a `<name>_frame` per source when there are
    several). A bare dict works too -- `frame` is synthesized if absent.

    `sources` says how big each input file is. Pass the `ClipsDoc` and every
    region is measured against the file `Region.source` names it lives in;
    pass a `{name: (w, h)}` mapping for the same effect without the document.
    A single `(w, h)` is the single-file spelling and is what the old
    `source_size` argument meant -- that keyword still works.
    """
    opts = options or LayoutOptions()
    table = _resolve_sources(sources, source_size)
    out_w, out_h = output_size
    if out_w < 2 or out_h < 2:
        raise LayoutError(f"output {out_w}x{out_h} is too small")

    if isinstance(layout, FocusLayout):
        placements = _plan_focus(layout, regions, table, out_w, out_h, opts)
    elif isinstance(layout, HeroInsetLayout):
        placements = _plan_hero_inset(layout, regions, table, out_w, out_h, opts)
    elif isinstance(layout, StackLayout):
        placements = _plan_stack(layout, regions, table, out_w, out_h, opts)
    else:  # pragma: no cover - the discriminated union admits nothing else
        raise LayoutError(f"unsupported layout {type(layout).__name__}")

    # Only the sources this layout actually reads. An engine wires one input
    # per entry, so listing a job's other five files would cost five decodes.
    used = dict(table.used)
    primary_w, primary_h = used.get(table.primary) or next(iter(used.values()))
    return LayoutPlan(
        mode=layout.mode,
        output_width=out_w,
        output_height=out_h,
        source_sizes=used,
        source_width=primary_w,
        source_height=primary_h,
        placements=placements,
    )


def span_times(clip: Clip) -> list[tuple[float, float, Layout]]:
    """Layout spans as (start, end, layout), clip-relative, gap-free.

    The schema guarantees the first span is at 0.0 and that they are strictly
    ordered, so this is just pairing each `at` with the next one and closing
    the last against the clip duration.
    """
    spans = clip.spans
    duration = clip.duration
    out: list[tuple[float, float, Layout]] = []
    for i, span in enumerate(spans):
        end = spans[i + 1].at if i + 1 < len(spans) else duration
        out.append((span.at, end, span))
    return out


def plan_clip(
    clip: Clip,
    regions: Mapping[str, Region],
    sources: Sources | None = None,
    output_size: tuple[int, int] = (1080, 1920),
    options: LayoutOptions | None = None,
    *,
    source_size: Sources | None = None,
) -> list[SpanPlan]:
    """Every layout span of a clip, planned. One entry for a single-layout clip.

    An engine renders one video segment per SpanPlan and concatenates them.
    `sources` is as in `compute_placements`; each span is planned
    independently, so `sp.plan.source_sizes` lists only what that span reads.
    """
    # Normalized once here so a bad `sources` fails before any span is planned;
    # each span then gets its own table, since `source_sizes` is per span.
    sizes = _resolve_sources(sources, source_size).sizes
    return [
        SpanPlan(
            start=start,
            end=end,
            plan=compute_placements(layout, regions, sizes, output_size, options),
        )
        for start, end, layout in span_times(clip)
    ]


def regions_by_id(regions: Sequence[Region]) -> dict[str, Region]:
    """Convenience for callers holding a list rather than a ClipsDoc."""
    out = {
        IMPLICIT_FRAME_REGION_ID: Region(
            id=IMPLICIT_FRAME_REGION_ID,
            kind="unknown",
            rect=(0.0, 0.0, 1.0, 1.0),
            label="whole source frame",
        )
    }
    for r in regions:
        out[r.id] = r
    return out
