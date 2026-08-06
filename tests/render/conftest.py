"""Shared fixtures for the render tests."""

from __future__ import annotations

import pytest

from makeshorts.artifacts import Region, Word, WordsDoc

# A two-up webinar frame: speaker on the left half, slides on the right.
# 1920x1080 because that is what every recording tool produces.
SOURCE_1080P = (1920, 1080)
OUTPUT_VERTICAL = (1080, 1920)


def region(rid: str, kind: str, rect: tuple[float, float, float, float]) -> Region:
    return Region(id=rid, kind=kind, rect=rect)


@pytest.fixture
def two_up() -> dict[str, Region]:
    """The canonical speaker + slides source, plus the implicit `frame`."""
    return {
        "frame": region("frame", "unknown", (0.0, 0.0, 1.0, 1.0)),
        "cam_a": region("cam_a", "speaker", (0.0, 0.0, 0.5, 1.0)),
        "slides": region("slides", "slide", (0.5, 0.0, 0.5, 1.0)),
    }


def words(
    spec: list[tuple[str, float, float]], *, sentence_ends: set[int] | None = None
) -> WordsDoc:
    """Build a WordsDoc from (text, start, end) triples."""
    ends = sentence_ends or set()
    return WordsDoc(
        model="test",
        language="en",
        words=[
            Word(text=t, start=s, end=e, sentence_end=(i in ends))
            for i, (t, s, e) in enumerate(spec)
        ],
    )


def evenly_spaced(text: str, start: float = 0.0, per_word: float = 0.4) -> WordsDoc:
    """One word every `per_word` seconds, no gaps. Keeps timing out of the way
    when a test is about grouping rather than pacing."""
    toks = text.split()
    return words(
        [(t, start + i * per_word, start + (i + 1) * per_word) for i, t in enumerate(toks)]
    )


@pytest.fixture
def sample_words() -> WordsDoc:
    return evenly_spaced(
        "Eighteen months is the number that kills companies. "
        "Nobody checks it, and that is the whole problem here.",
        start=10.0,
        per_word=0.4,
    )
