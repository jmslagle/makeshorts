"""Shapes of the mechanical-stage artifacts.

These are the contract between `prepare/` (which writes them) and `select/` +
`render/` (which read them). Nothing here knows about ffmpeg or about the AI.

Every file in this module is written by deterministic code. If a value in here
required judgment to produce, it belongs in clips.json instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# A normalized rectangle within the source frame: [x, y, w, h], each 0..1.
# Normalized rather than pixels so the edit list survives a re-encode at a
# different resolution, and so no engine-specific crop geometry leaks upward.
Rect = tuple[float, float, float, float]

FULL_FRAME: Rect = (0.0, 0.0, 1.0, 1.0)

# Always available as a layout target, even when region detection finds nothing.
# `{"mode": "focus", "region": "frame"}` is the degenerate whole-frame layout.
IMPLICIT_FRAME_REGION_ID = "frame"


class Strict(BaseModel):
    """Reject unknown keys everywhere. A typo in a hand-edited artifact should
    fail loudly rather than being silently dropped."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# media.json
# --------------------------------------------------------------------------


class MediaDoc(Strict):
    path: str
    duration: float
    resolution: tuple[int, int]
    fps: float
    has_audio: bool
    audio_channels: int = 0
    sha256: str


# --------------------------------------------------------------------------
# words.json  — the source of truth for every timestamp in the system
# --------------------------------------------------------------------------


class Word(Strict):
    """One token with its own timing.

    `sentence_start` / `sentence_end` are computed mechanically from
    punctuation during transcription. The linter uses them to enforce the
    `start_on_sentence_start` / `end_on_sentence_end` gates, so they must be
    present even when the gates are disabled.
    """

    text: str
    start: float
    end: float
    probability: float | None = None
    sentence_start: bool = False
    sentence_end: bool = False


class WordsDoc(Strict):
    model: str
    language: str
    words: list[Word]


# --------------------------------------------------------------------------
# silence.json
# --------------------------------------------------------------------------


class SilenceSpan(Strict):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SilenceDoc(Strict):
    threshold_db: float
    min_duration: float
    spans: list[SilenceSpan]


# --------------------------------------------------------------------------
# regions.json — a *proposal*, not a decision
# --------------------------------------------------------------------------

RegionKind = Literal["speaker", "slide", "unknown"]


class Region(Strict):
    """A named area of the source frame.

    Detection proposes these; the editorial step confirms or corrects them into
    clips.json. `confidence` exists so a human reviewing regions.json knows
    which ones to look at.
    """

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: RegionKind
    rect: Rect
    # Which entry of `sources` this rect is measured against. None means the
    # job's single/primary source, which is what every single-file job uses.
    # Zoom-style exports are the motivating case: several frame-aligned views
    # of one meeting, where the sharp slides and the large face live in
    # different files.
    source: str | None = None
    label: str | None = None
    motion: float | None = None
    change_points: list[float] = Field(default_factory=list)
    confidence: float | None = None


class RegionsDoc(Strict):
    grid: str
    frames_sampled: int
    regions: list[Region]
