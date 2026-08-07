"""clips.json — the edit list. THE contract of this project.

This file is engine-agnostic by construction. Times are absolute seconds in the
source. Geometry is normalized 0..1. Styles are *names* resolved at render time.

Nothing ffmpeg-shaped may ever appear here: no codec, bitrate, preset, filter
string, pixel crop, font path, or output directory. Those live in
config/render.yaml and are the render engine's business alone. `tests/
test_engine_agnostic.py` enforces this mechanically -- if you find yourself
wanting to add such a field, the design has sprung a leak.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from makeshorts.artifacts import IMPLICIT_FRAME_REGION_ID, Rect, Region, Strict

SCHEMA_VERSION = "1.0"

# Drives the output filename, so it must be filesystem-safe and sortable.
CLIP_ID_PATTERN = r"^\d{2}-[a-z0-9][a-z0-9-]*$"


# --------------------------------------------------------------------------
# Layout — composes over named regions, so speaker count is irrelevant here.
# --------------------------------------------------------------------------

# "cover" fills the output frame, cropping overflow.
# "contain_blur" fits the whole region in and fills the remainder with a
# blurred scaled copy. This is what makes a separate `full` layout mode
# unnecessary: {"mode": "focus", "region": "frame", "fit": "contain_blur"}.
Fit = Literal["cover", "contain_blur"]

InsetCorner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]


class LayoutBase(Strict):
    # Seconds relative to the START OF THE CLIP, not the source. A single-layout
    # clip omits it. The first span must be at 0.0 (enforced by the linter).
    at: float = 0.0

    # How this span is entered: a transition preset NAME from render.yaml, or
    # omitted for a hard cut. A name rather than a filter or a duration for the
    # same reason a caption style is a name -- "dissolve" is a decision about
    # this cut, while what a dissolve looks like is presentation config every
    # engine resolves for itself. Meaningless on the first span, which has
    # nothing to transition from; the linter says so.
    transition: str | None = None


class FocusLayout(LayoutBase):
    """One region fills the output frame.

    Covers active-speaker crop, single-speaker sources, and slides-only alike.
    """

    mode: Literal["focus"] = "focus"
    region: str
    fit: Fit = "cover"


class HeroInsetLayout(LayoutBase):
    """One region fills the frame with a second inset over it.

    Slide + talking head, or speaker + reaction shot.
    """

    mode: Literal["hero_inset"] = "hero_inset"
    hero: str
    inset: str
    inset_corner: InsetCorner = "bottom_right"
    inset_scale: float = Field(default=0.28, gt=0.05, le=0.6)
    hero_fit: Fit = "cover"

    @model_validator(mode="after")
    def _distinct(self) -> HeroInsetLayout:
        if self.hero == self.inset:
            raise ValueError("hero and inset must be different regions")
        return self


class StackLayout(LayoutBase):
    """Regions tiled vertically, each fit to its own band."""

    mode: Literal["stack"] = "stack"
    regions: list[str] = Field(min_length=2)
    fit: Fit = "cover"

    @model_validator(mode="after")
    def _distinct(self) -> StackLayout:
        if len(set(self.regions)) != len(self.regions):
            raise ValueError("stack regions must be distinct")
        return self


Layout = Annotated[
    FocusLayout | HeroInsetLayout | StackLayout,
    Field(discriminator="mode"),
]


def layout_region_ids(layout: Layout) -> list[str]:
    """Every region id a layout references. Used by the linter for referential
    integrity and for the visual-dependency rule."""
    if isinstance(layout, FocusLayout):
        return [layout.region]
    if isinstance(layout, HeroInsetLayout):
        return [layout.hero, layout.inset]
    return list(layout.regions)


# --------------------------------------------------------------------------
# Why — the reviewable record of editorial judgment
# --------------------------------------------------------------------------


class CriterionScore(Strict):
    """A score against one rubric criterion, plus the evidence for it.

    `evidence` is what makes this file worth reading. A score with no evidence
    is an assertion; a score with a quote is an argument you can disagree with.
    """

    score: int = Field(ge=1, le=5)
    evidence: str = Field(min_length=1)


class RejectedAlternative(Strict):
    start: float
    end: float
    reason: str


class Why(Strict):
    one_line: str
    theme: str
    # Keys must exactly match the rubric ids in config/criteria.yaml. The
    # linter enforces this -- the model cannot skip an inconvenient criterion.
    scores: dict[str, CriterionScore]
    # Recomputed and compared by the linter; never trusted as written.
    weighted_score: float
    rejected_alternatives: list[RejectedAlternative] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Clip
# --------------------------------------------------------------------------


class Captions(Strict):
    # A style NAME. Resolves against config/styles.yaml at render time.
    style: str = "pill-karaoke"
    position: Literal["lower_third", "center", "upper_third"] = "lower_third"
    enabled: bool = True


class Audio(Strict):
    normalize: bool = True


class Clip(Strict):
    id: str = Field(pattern=CLIP_ID_PATTERN)
    title: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    # Verbatim transcript of the span. The linter checks this against
    # words.json -- it is the primary defence against invented timestamps and
    # invented quotes, which is the failure mode that actually occurs.
    source_text: str

    why: Why

    # True when the clip's words lean on what is on screen ("as you can see
    # here"). The linter then requires the layout to include a slide region.
    # This is the payoff of the region model: an editorial claim about meaning
    # becomes a mechanically checkable constraint.
    visual_dependency: bool = False

    # Which region is talking. Editorial, overridable by hand.
    speaker: str | None = None

    # Either a single layout or an ordered list of timed spans.
    layout: Layout | list[Layout]

    captions: Captions = Field(default_factory=Captions)
    audio: Audio = Field(default_factory=Audio)

    # A branding preset NAME defined in config/render.yaml, "none" to suppress
    # a watermark this once, or omitted to take the configured default. A path
    # would be wrong here for the same reason a font path would be: which
    # image, at what size, in which corner is presentation config that every
    # clip shares, not an editorial decision about this clip.
    branding: str | None = None

    # An outro preset NAME, "none" to omit one on this clip, or omitted to take
    # the configured default. Same rule as branding: the edit list never
    # carries the file.
    outro: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def spans(self) -> list[Layout]:
        """Layout normalized to a list. Engines should use this and never
        branch on the single-vs-list distinction themselves."""
        return self.layout if isinstance(self.layout, list) else [self.layout]

    @model_validator(mode="after")
    def _ordered(self) -> Clip:
        if self.end <= self.start:
            raise ValueError(f"clip {self.id}: end must be after start")
        ats = [s.at for s in self.spans]
        if ats and ats[0] != 0.0:
            raise ValueError(f"clip {self.id}: first layout span must be at 0.0")
        if ats != sorted(ats) or len(set(ats)) != len(ats):
            raise ValueError(f"clip {self.id}: layout spans must be strictly ordered by `at`")
        if any(a >= self.duration for a in ats):
            raise ValueError(f"clip {self.id}: a layout span starts at or past the clip end")
        return self


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


class SourceSpec(Strict):
    """One input file.

    A job usually has one. It may have several when a recording was exported
    as multiple frame-aligned views of the same timeline -- Zoom's
    active-speaker, gallery and shared-screen renders, for instance, where the
    sharp slides and the usable face are in different files. Multiple sources
    are only meaningful when they share a clock: a timestamp must mean the
    same instant in every one of them, and `ms lint` checks that each covers
    the clip's range.
    """

    path: str
    duration: float
    resolution: tuple[int, int]
    # Confirmed from regions.json by the editorial step. `frame` is implicit
    # and need not be listed; see resolved_regions().
    regions: list[Region] = Field(default_factory=list)


class CriteriaRef(Strict):
    """Which rubric produced this edit list.

    Recorded so a clips.json reviewed months later can be read against the
    criteria that actually generated it, and so the linter can warn when the
    rubric has changed underneath an existing edit list.
    """

    file: str
    version: str
    sha256: str


class OutputSpec(Strict):
    width: int = 1080
    height: int = 1920
    fps: int = 30


PRIMARY_SOURCE_NAME = "main"


class ClipsDoc(Strict):
    # `source` (one file) and `sources` (several named, frame-aligned files)
    # are two spellings of the same thing. Single-file jobs keep using
    # `source` and never see the difference; everything downstream reads
    # `resolved_sources()`, which always returns a dict.
    schema_version: str = SCHEMA_VERSION
    job: str
    source: SourceSpec | None = None
    sources: dict[str, SourceSpec] | None = None
    output: OutputSpec = Field(default_factory=OutputSpec)
    criteria_ref: CriteriaRef
    clips: list[Clip]

    @model_validator(mode="after")
    def _exactly_one_source_form(self) -> ClipsDoc:
        if (self.source is None) == (self.sources is None):
            raise ValueError(
                "provide exactly one of `source` (a single file) or `sources` "
                "(a mapping of name -> file)"
            )
        if self.sources is not None and not self.sources:
            raise ValueError("`sources` must not be empty")
        return self

    def resolved_sources(self) -> dict[str, SourceSpec]:
        """Every input file by name. Single-source jobs get one entry, `main`."""
        if self.sources is not None:
            return dict(self.sources)
        assert self.source is not None
        return {PRIMARY_SOURCE_NAME: self.source}

    @property
    def primary_source_name(self) -> str:
        """The source a region falls back to when it names none, and the one
        the bare `frame` region refers to. For `sources`, insertion order
        decides -- so write the file you think of as primary first."""
        return next(iter(self.resolved_sources()))

    def resolved_regions(self) -> dict[str, Region]:
        """Declared regions plus the implicit whole-frame regions.

        Always use this rather than reaching into a SourceSpec, so `frame`
        works everywhere without every caller special-casing it. Each source
        also gets a `<name>_frame` region, which is what makes a multi-source
        layout expressible without declaring anything by hand:

            {"mode": "hero_inset", "hero": "slides_frame", "inset": "cam_frame"}

        Every returned region has `source` filled in -- callers never have to
        decide what None meant.
        """
        srcs = self.resolved_sources()
        primary = self.primary_source_name
        out: dict[str, Region] = {
            IMPLICIT_FRAME_REGION_ID: Region(
                id=IMPLICIT_FRAME_REGION_ID,
                kind="unknown",
                rect=(0.0, 0.0, 1.0, 1.0),
                source=primary,
                label="whole source frame",
            )
        }
        # Only worth naming per-source frames when there is a choice to make.
        # On a single-source job `frame` already means this, and offering
        # `main_frame` alongside it would be noise in every error message that
        # lists the known regions.
        if len(srcs) > 1:
            for name in srcs:
                out[f"{name}_frame"] = Region(
                    id=f"{name}_frame",
                    kind="unknown",
                    rect=(0.0, 0.0, 1.0, 1.0),
                    source=name,
                    label=f"whole frame of source {name!r}",
                )
        for name, spec in srcs.items():
            for r in spec.regions:
                out[r.id] = r if r.source else r.model_copy(update={"source": name})
        return out

    def source_of(self, region_id: str) -> SourceSpec:
        """The file a region is measured against. Raises for unknown ids, which
        the linter reports as a referential-integrity error."""
        region = self.resolved_regions()[region_id]
        srcs = self.resolved_sources()
        return srcs[region.source or self.primary_source_name]

    @model_validator(mode="after")
    def _unique_ids(self) -> ClipsDoc:
        ids = [c.id for c in self.clips]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate clip ids: {sorted(dupes)}")
        return self


__all__ = [
    "SCHEMA_VERSION",
    "CLIP_ID_PATTERN",
    "Rect",
    "Fit",
    "InsetCorner",
    "FocusLayout",
    "HeroInsetLayout",
    "StackLayout",
    "Layout",
    "layout_region_ids",
    "CriterionScore",
    "RejectedAlternative",
    "Why",
    "Captions",
    "Audio",
    "Clip",
    "SourceSpec",
    "CriteriaRef",
    "OutputSpec",
    "ClipsDoc",
]
