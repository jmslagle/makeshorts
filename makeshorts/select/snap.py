"""Boundary snapping — pure arithmetic over words.json.

No I/O and no config objects: every threshold arrives as an argument, so the
same functions serve the measuring path (`ms lint`) and the moving path
(`ms lint --fix`), and so this module stays trivially testable.

Two ideas do all the work here.

*Boundaries.* A boundary is a real edge in words.json — the start of a word,
the end of a word, or either of those on a word that also opens or closes a
sentence. A clip time is only ever legitimate in relation to one of these. A
time that sits near none of them was not read out of words.json; it was
invented, and that is the failure mode this whole module exists to catch.

*Padding.* `pad_in` seconds of air before the first word and `pad_out` after
the last are a deliberate, bounded departure from a boundary — and never
allowed to run over the neighbouring word. Because padding is legitimate, the
tolerance for a start is asymmetric: early by up to `tolerance + pad_in`, late
by only `tolerance`. The reverse holds for an end. Without that asymmetry a
file that `--fix` has just repaired would fail the very check `--fix` exists to
satisfy.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from makeshorts.artifacts import Word, WordsDoc

__all__ = [
    "EPS",
    "PRECISION",
    "Side",
    "BoundaryKind",
    "Boundary",
    "SnapResult",
    "BoundaryCheck",
    "ClipSnap",
    "boundaries",
    "has_sentence_flags",
    "nearest_boundary",
    "snap",
    "snap_start",
    "snap_end",
    "pad_range",
    "snap_clip",
    "words_in_span",
    "span_text",
    "check_boundary",
]

# Float noise guard. Word timings are written with millisecond precision at
# most, so anything below this is not a real difference.
EPS = 1e-9

# Decimal places kept when a time is written back to clips.json. Rounding
# happens inside snap_clip() rather than at the write site so that `--fix`
# reaches a fixed point on the first run and the second run is a no-op.
PRECISION = 3

Side = Literal["start", "end"]

BoundaryKind = Literal["word_start", "word_end", "sentence_start", "sentence_end"]

_SIDE_KIND: dict[Side, BoundaryKind] = {
    "start": "word_start",
    "end": "word_end",
}


@dataclass(frozen=True)
class Boundary:
    """A real edge in words.json, with the word it belongs to.

    Carrying the word makes every lint message able to quote the text at the
    boundary, which is the difference between "start is 0.9s off" and "start is
    0.9s before the word 'eighteen'".
    """

    time: float
    index: int
    kind: BoundaryKind
    word: Word

    @property
    def text(self) -> str:
        return self.word.text.strip()


@dataclass(frozen=True)
class SnapResult:
    """Where a time moved to, and how far. The distance is what lets the caller
    decide whether a miss is rounding jitter or a fabrication."""

    original: float
    time: float
    boundary: Boundary | None

    @property
    def distance(self) -> float:
        return abs(self.time - self.original)

    @property
    def moved(self) -> bool:
        return self.distance > EPS

    @property
    def snapped(self) -> bool:
        return self.boundary is not None


@dataclass(frozen=True)
class BoundaryCheck:
    """Does a time sit where a clip edge is allowed to sit?

    `offset` is signed — negative means the time is earlier than the boundary
    it is nearest to, which for a start is the padded (legitimate) direction.
    """

    side: Side
    time: float
    nearest: Boundary | None
    inside_word: Word | None
    tolerance: float
    pad: float

    @property
    def offset(self) -> float | None:
        if self.nearest is None:
            return None
        return self.time - self.nearest.time

    @property
    def distance(self) -> float | None:
        off = self.offset
        return None if off is None else abs(off)

    @property
    def ok(self) -> bool:
        off = self.offset
        if off is None:
            return False
        if self.side == "start":
            return -(self.tolerance + self.pad) - EPS <= off <= self.tolerance + EPS
        return -self.tolerance - EPS <= off <= self.tolerance + self.pad + EPS


@dataclass(frozen=True)
class ClipSnap:
    """The full result of snapping and padding one clip range."""

    start: float
    end: float
    start_snap: SnapResult
    end_snap: SnapResult
    pad_in_applied: float
    pad_out_applied: float

    @property
    def start_moved(self) -> float:
        """Total distance the start moved, snapping and padding together."""
        return abs(self.start - self.start_snap.original)

    @property
    def end_moved(self) -> float:
        return abs(self.end - self.end_snap.original)

    @property
    def changed(self) -> bool:
        return self.start_moved > EPS or self.end_moved > EPS


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------


def _as_words(source: WordsDoc | Sequence[Word]) -> Sequence[Word]:
    return source.words if isinstance(source, WordsDoc) else source


def boundaries(source: WordsDoc | Sequence[Word], kind: BoundaryKind) -> list[Boundary]:
    """Every boundary of one kind, in ascending time order."""
    words = _as_words(source)
    out: list[Boundary] = []
    for i, w in enumerate(words):
        if kind == "word_start":
            out.append(Boundary(w.start, i, kind, w))
        elif kind == "word_end":
            out.append(Boundary(w.end, i, kind, w))
        elif kind == "sentence_start":
            if w.sentence_start:
                out.append(Boundary(w.start, i, kind, w))
        elif kind == "sentence_end":
            if w.sentence_end:
                out.append(Boundary(w.end, i, kind, w))
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown boundary kind {kind!r}")
    out.sort(key=lambda b: b.time)
    return out


def has_sentence_flags(source: WordsDoc | Sequence[Word]) -> bool:
    """True when the transcript carries any sentence segmentation at all.

    A transcript with no flags anywhere means the segmentation step did not
    run, not that every clip starts mid-sentence — callers should say so rather
    than failing every clip.
    """
    return any(w.sentence_start or w.sentence_end for w in _as_words(source))


def nearest_boundary(
    source: WordsDoc | Sequence[Word],
    t: float,
    kind: BoundaryKind,
    *,
    prefer: Literal["earlier", "later"] = "earlier",
) -> Boundary | None:
    """The boundary of `kind` closest to `t`, or None if there are none.

    Ties are broken by `prefer`, which the callers set so that a start opens
    early and an end closes late — erring towards keeping a whole word rather
    than shaving one.
    """
    bs = boundaries(source, kind)
    if not bs:
        return None
    times = [b.time for b in bs]
    i = bisect_left(times, t)
    candidates = [b for b in (bs[i - 1] if i > 0 else None, bs[i] if i < len(bs) else None) if b]
    best = candidates[0]
    best_d = abs(best.time - t)
    for c in candidates[1:]:
        d = abs(c.time - t)
        if d < best_d - EPS:
            best, best_d = c, d
        elif abs(d - best_d) <= EPS and prefer == "later" and c.time > best.time:
            best, best_d = c, d
    return best


# --------------------------------------------------------------------------
# Snapping
# --------------------------------------------------------------------------


def snap(
    source: WordsDoc | Sequence[Word],
    t: float,
    kind: BoundaryKind,
    *,
    prefer: Literal["earlier", "later"] = "earlier",
) -> SnapResult:
    b = nearest_boundary(source, t, kind, prefer=prefer)
    return SnapResult(original=t, time=t if b is None else b.time, boundary=b)


def snap_start(
    source: WordsDoc | Sequence[Word], t: float, *, to_sentence: bool = False
) -> SnapResult:
    """Snap a clip start to the nearest word start, or sentence start.

    `to_sentence` falls back to word starts when the transcript carries no
    sentence flags, so a transcript without segmentation still snaps somewhere
    real instead of not snapping at all.
    """
    kind: BoundaryKind = "word_start"
    if to_sentence and boundaries(source, "sentence_start"):
        kind = "sentence_start"
    return snap(source, t, kind, prefer="earlier")


def snap_end(
    source: WordsDoc | Sequence[Word], t: float, *, to_sentence: bool = False
) -> SnapResult:
    kind: BoundaryKind = "word_end"
    if to_sentence and boundaries(source, "sentence_end"):
        kind = "sentence_end"
    return snap(source, t, kind, prefer="later")


def pad_range(
    source: WordsDoc | Sequence[Word],
    start: float,
    end: float,
    *,
    pad_in: float,
    pad_out: float,
    source_duration: float | None = None,
) -> tuple[float, float, float, float]:
    """Widen [start, end] by the pads without running over a neighbouring word.

    Returns (start, end, pad_in_applied, pad_out_applied). The applied pads are
    what actually fitted, which is less than the requested pad whenever the
    adjacent word is closer than the pad — the case that would otherwise clip a
    syllable off the word next door.
    """
    words = _as_words(source)

    floor = 0.0
    for w in words:
        if w.end <= start + EPS and w.end > floor:
            floor = w.end
    ceiling = source_duration if source_duration is not None else float("inf")
    for w in words:
        if w.start >= end - EPS:
            ceiling = min(ceiling, w.start)
            break

    new_start = max(start - max(pad_in, 0.0), floor, 0.0)
    new_end = min(end + max(pad_out, 0.0), ceiling)
    if source_duration is not None:
        new_end = min(new_end, source_duration)
    # Padding only ever widens; a pad that computed backwards means the
    # neighbour is already touching, so keep the original edge.
    new_start = min(new_start, start)
    new_end = max(new_end, end)
    return new_start, new_end, start - new_start, new_end - end


def snap_clip(
    source: WordsDoc | Sequence[Word],
    start: float,
    end: float,
    *,
    to_sentence_start: bool = True,
    to_sentence_end: bool = True,
    pad_in: float = 0.0,
    pad_out: float = 0.0,
    source_duration: float | None = None,
) -> ClipSnap:
    """Snap both edges of a clip and apply padding. The whole of `--fix`.

    Rounding to `PRECISION` happens here, and the clamps are applied to the
    *rounded* floor and ceiling, so that running this on its own output is a
    no-op — `--fix` must be idempotent or nobody will trust it.
    """
    s = snap_start(source, start, to_sentence=to_sentence_start)
    e = snap_end(source, end, to_sentence=to_sentence_end)

    snapped_start, snapped_end = s.time, e.time
    if snapped_end <= snapped_start:
        # Degenerate input (or a transcript so sparse both edges snapped to the
        # same place). Leave the range alone rather than inventing one.
        snapped_start, snapped_end = start, end

    padded_start, padded_end, applied_in, applied_out = pad_range(
        source,
        snapped_start,
        snapped_end,
        pad_in=pad_in,
        pad_out=pad_out,
        source_duration=source_duration,
    )
    return ClipSnap(
        start=round(padded_start, PRECISION),
        end=round(padded_end, PRECISION),
        start_snap=s,
        end_snap=e,
        pad_in_applied=round(applied_in, PRECISION),
        pad_out_applied=round(applied_out, PRECISION),
    )


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def words_in_span(
    source: WordsDoc | Sequence[Word], start: float, end: float
) -> list[Word]:
    """The words a clip actually contains.

    Membership is decided by the word's midpoint. A start edge that sits a
    little inside the first word (rounding) still keeps that word, and padding
    into the silence on either side never drags in a neighbour — both of which
    a naive containment test gets wrong.
    """
    return [w for w in _as_words(source) if start - EPS <= (w.start + w.end) / 2 <= end + EPS]


def span_text(words: Sequence[Word]) -> str:
    """Join words back into the text as spoken. Whisper emits leading spaces on
    most tokens, so strip before joining rather than concatenating raw."""
    return " ".join(w.text.strip() for w in words if w.text.strip())


def check_boundary(
    source: WordsDoc | Sequence[Word],
    t: float,
    side: Side,
    *,
    tolerance: float,
    pad: float = 0.0,
) -> BoundaryCheck:
    """Measure a clip edge against the nearest real word edge.

    This is the measurement behind the `time.*` rules. It reports rather than
    decides: `ok` is the plain in-tolerance answer, `distance` lets the caller
    separate jitter from fabrication, and `inside_word` flags the specific case
    of a cut landing in the middle of a spoken word.
    """
    kind = _SIDE_KIND[side]
    nearest = nearest_boundary(source, t, kind, prefer="earlier" if side == "start" else "later")
    inside = next(
        (w for w in _as_words(source) if w.start + EPS < t < w.end - EPS),
        None,
    )
    return BoundaryCheck(
        side=side,
        time=t,
        nearest=nearest,
        inside_word=inside,
        tolerance=tolerance,
        pad=pad,
    )
