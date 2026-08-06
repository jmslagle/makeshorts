"""The caption backend seam.

Two backends, one interface. Which one runs is decided by `render/caps.py`
probing the installed ffmpeg, not by configuration, because it is a fact about
the machine rather than a preference: a build without libass simply cannot run
the `subtitles` filter, and a build with it should not be paying to rasterize
PNGs in Python.

The interface deliberately does not return filter strings. A backend returns
either **overlay images with positions and times** or **a subtitle file**, and
the engine decides how to feed that to whatever it drives. That is what lets a
non-ffmpeg engine reuse the Pillow backend unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field

from makeshorts.artifacts import Strict
from makeshorts.render.captions.cues import Cue
from makeshorts.render.captions.style import CaptionStyle

__all__ = [
    "CaptionPosition",
    "CaptionOverlay",
    "OverlayAssets",
    "SubtitleAssets",
    "CaptionAssets",
    "CaptionBackend",
    "CaptionRenderError",
    "anchor_y",
]

# Matches `Captions.position` in the edit list.
CaptionPosition = Literal["lower_third", "center", "upper_third"]


class CaptionRenderError(RuntimeError):
    """A backend could not produce captions for these cues."""


class CaptionOverlay(Strict):
    """One RGBA image to composite at a fixed spot for a fixed window.

    `start`/`end` are seconds from the start of the *clip*. Sizes are in
    output pixels. The engine composites `png_path` at (`x`, `y`) while
    `start <= t < end` and does nothing else with it.
    """

    png_path: str
    x: int
    y: int
    width: int
    height: int
    start: float
    end: float
    # Which cue this state belongs to, and which word is lit. Not needed to
    # composite; kept because it makes a receipt or a debug dump legible.
    cue_index: int = 0
    word_index: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple[str, int, int, float, float]:
        """(png_path, dest_x, dest_y, start_time, end_time)."""
        return (self.png_path, self.x, self.y, self.start, self.end)


class OverlayAssets(Strict):
    """What the Pillow backend produces: a stack of timed images."""

    kind: Literal["overlays"] = "overlays"
    overlays: list[CaptionOverlay] = Field(default_factory=list)
    # Distinct PNGs actually on disk. Lower than len(overlays) whenever
    # dedupe hit; useful in a receipt and as a cheap regression signal.
    unique_images: int = 0

    def as_tuples(self) -> list[tuple[str, int, int, float, float]]:
        return [o.as_tuple() for o in self.overlays]


class SubtitleAssets(Strict):
    """What the libass backend produces: one file the renderer burns in."""

    kind: Literal["subtitle_file"] = "subtitle_file"
    path: str
    format: Literal["ass", "srt"] = "ass"
    # Where the fonts named by the file live, when they are not installed
    # system-wide. None means "let fontconfig find them".
    fonts_dir: str | None = None


CaptionAssets = Annotated[OverlayAssets | SubtitleAssets, Field(discriminator="kind")]


@runtime_checkable
class CaptionBackend(Protocol):
    """Cues + style + output size -> something compositable.

    `workdir` is the job directory the backend may write into. It must be safe
    to call `render` twice with the same arguments and get the same result
    without re-doing the work -- caption assets are content-addressed, and the
    engine relies on that for re-render idempotency.
    """

    name: str

    def is_available(self) -> bool:
        """Whether this backend can run against the installed toolchain."""
        ...

    def render(
        self,
        cues: list[Cue],
        style: CaptionStyle,
        output_size: tuple[int, int],
        workdir: Path,
        position: CaptionPosition = "lower_third",
    ) -> CaptionAssets:
        ...


def anchor_y(
    position: CaptionPosition,
    block_height: int,
    output_height: int,
    safe_top: int,
    safe_bottom: int,
) -> int:
    """Top edge of a caption block of `block_height`, in output pixels.

    Shared by both backends so `lower_third` means the same distance from the
    bottom whichever one is running. `lower_third` anchors the *bottom* of the
    block against the safe area rather than the top, so a two-line cue grows
    upward and the baseline stays put between cues of different heights --
    captions that jump vertically as the line count changes are the single
    most distracting thing a caption renderer can do.
    """
    if position == "center":
        y = (output_height - block_height) // 2
    elif position == "upper_third":
        y = safe_top
    else:
        y = output_height - safe_bottom - block_height
    return max(0, min(y, max(output_height - block_height, 0)))
