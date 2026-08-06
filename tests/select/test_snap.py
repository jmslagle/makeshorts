"""Unit tests for boundary snapping.

snap.py is pure arithmetic, so it gets exhaustive small cases rather than
fixture-driven ones. The tests that matter are the awkward ones: ties, the
edges of the word list, padding that has nowhere to go, and idempotency --
`--fix` runs on files people keep, so a second run must be a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeshorts.artifacts import Word, WordsDoc
from makeshorts.select import snap

FIXTURES = Path(__file__).parent / "fixtures"


def w(
    text: str,
    start: float,
    end: float,
    *,
    ss: bool = False,
    se: bool = False,
) -> Word:
    return Word(text=text, start=start, end=end, sentence_start=ss, sentence_end=se)


def doc(*words: Word) -> WordsDoc:
    return WordsDoc(model="test", language="en", words=list(words))


# Two sentences with a 1.0s gap between them, and a 0.1s gap inside each.
#   "Eighteen months."            10.0 – 11.4
#   "That is the number."         12.4 – 15.0
SIMPLE = doc(
    w("Eighteen", 10.0, 10.6, ss=True),
    w("months.", 10.7, 11.4, se=True),
    w("That", 12.4, 12.8, ss=True),
    w("is", 12.9, 13.1),
    w("the", 13.2, 13.4),
    w("number.", 13.5, 15.0, se=True),
)


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------


def test_boundaries_of_each_kind() -> None:
    assert [b.time for b in snap.boundaries(SIMPLE, "word_start")] == [
        10.0, 10.7, 12.4, 12.9, 13.2, 13.5
    ]
    assert [b.time for b in snap.boundaries(SIMPLE, "word_end")] == [
        10.6, 11.4, 12.8, 13.1, 13.4, 15.0
    ]
    assert [b.time for b in snap.boundaries(SIMPLE, "sentence_start")] == [10.0, 12.4]
    assert [b.time for b in snap.boundaries(SIMPLE, "sentence_end")] == [11.4, 15.0]


def test_boundaries_carry_their_word() -> None:
    b = snap.boundaries(SIMPLE, "sentence_start")[1]
    assert b.text == "That"
    assert b.index == 2
    assert b.kind == "sentence_start"


def test_boundaries_of_empty_transcript() -> None:
    assert snap.boundaries(doc(), "word_start") == []
    assert not snap.has_sentence_flags(doc())


def test_has_sentence_flags() -> None:
    assert snap.has_sentence_flags(SIMPLE)
    assert not snap.has_sentence_flags(doc(w("a", 0.0, 1.0), w("b", 1.0, 2.0)))


def test_boundaries_accept_a_bare_word_list() -> None:
    assert snap.boundaries(SIMPLE.words, "word_start") == snap.boundaries(SIMPLE, "word_start")


# --------------------------------------------------------------------------
# nearest_boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (10.0, 10.0),   # exact hit
        (0.0, 10.0),    # before the first word
        (99.0, 13.5),   # after the last word
        (12.3, 12.4),   # just before
        (11.9, 12.4),   # nearer the next sentence than the previous word
        (11.4, 10.7),   # nearer the previous word start
    ],
)
def test_nearest_word_start(t: float, expected: float) -> None:
    b = snap.nearest_boundary(SIMPLE, t, "word_start")
    assert b is not None and b.time == expected


def test_nearest_boundary_of_empty_kind_is_none() -> None:
    flat = doc(w("a", 0.0, 1.0))
    assert snap.nearest_boundary(flat, 0.5, "sentence_start") is None
    assert snap.nearest_boundary(doc(), 0.5, "word_start") is None


def test_ties_break_by_preference() -> None:
    """Equidistant boundaries: a start opens early, an end closes late, so a
    tie never shaves a word."""
    two = doc(w("a", 0.0, 1.0), w("b", 2.0, 3.0))
    # word starts are 0.0 and 2.0; word ends are 1.0 and 3.0.
    assert snap.nearest_boundary(two, 1.0, "word_start", prefer="earlier").time == 0.0
    assert snap.nearest_boundary(two, 1.0, "word_start", prefer="later").time == 2.0
    assert snap.nearest_boundary(two, 2.0, "word_end", prefer="earlier").time == 1.0
    assert snap.nearest_boundary(two, 2.0, "word_end", prefer="later").time == 3.0


# --------------------------------------------------------------------------
# snapping
# --------------------------------------------------------------------------


def test_snap_start_to_word_and_to_sentence() -> None:
    assert snap.snap_start(SIMPLE, 12.85, to_sentence=False).time == 12.9
    assert snap.snap_start(SIMPLE, 12.85, to_sentence=True).time == 12.4


def test_snap_end_to_word_and_to_sentence() -> None:
    assert snap.snap_end(SIMPLE, 13.15, to_sentence=False).time == 13.1
    assert snap.snap_end(SIMPLE, 13.15, to_sentence=True).time == 11.4


def test_snap_reports_distance_moved() -> None:
    r = snap.snap_start(SIMPLE, 12.85, to_sentence=True)
    assert r.original == 12.85
    assert r.time == 12.4
    assert r.distance == pytest.approx(0.45)
    assert r.moved and r.snapped


def test_snap_to_an_exact_boundary_does_not_move() -> None:
    r = snap.snap_start(SIMPLE, 12.4, to_sentence=True)
    assert r.time == 12.4 and not r.moved and r.distance == 0.0


def test_sentence_snapping_falls_back_to_words_without_flags() -> None:
    """A transcript with no segmentation must still snap somewhere real."""
    flat = doc(w("a", 0.0, 1.0), w("b", 2.0, 3.0))
    r = snap.snap_start(flat, 1.9, to_sentence=True)
    assert r.time == 2.0 and r.boundary is not None
    assert r.boundary.kind == "word_start"


def test_snap_with_no_words_leaves_the_time_alone() -> None:
    r = snap.snap_start(doc(), 5.0, to_sentence=True)
    assert r.time == 5.0 and not r.snapped and not r.moved


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def test_padding_applies_in_full_when_there_is_room() -> None:
    start, end, pin, pout = snap.pad_range(SIMPLE, 12.4, 15.0, pad_in=0.3, pad_out=0.4)
    assert (start, end) == (12.1, 15.4)
    assert (pin, pout) == pytest.approx((0.3, 0.4))


def test_padding_never_runs_over_the_neighbouring_word() -> None:
    """The gap before "That" is 1.0s, but the gap before "is" is 0.1s."""
    start, _, pin, _ = snap.pad_range(SIMPLE, 12.9, 13.1, pad_in=0.5, pad_out=0.0)
    assert start == pytest.approx(12.8)  # the end of "That", not 12.4
    assert pin == pytest.approx(0.1)

    _, end, _, pout = snap.pad_range(SIMPLE, 10.0, 10.6, pad_in=0.0, pad_out=0.5)
    assert end == pytest.approx(10.7)  # the start of "months."
    assert pout == pytest.approx(0.1)


def test_padding_clamps_at_zero_and_at_the_source_duration() -> None:
    tight = doc(w("a", 0.05, 1.0, ss=True, se=True))
    start, end, _, _ = snap.pad_range(
        tight, 0.05, 1.0, pad_in=0.5, pad_out=0.5, source_duration=1.2
    )
    assert start == 0.0
    assert end == 1.2


def test_padding_with_a_touching_neighbour_does_nothing() -> None:
    touching = doc(w("a", 0.0, 1.0), w("b", 1.0, 2.0), w("c", 2.0, 3.0))
    start, end, pin, pout = snap.pad_range(touching, 1.0, 2.0, pad_in=0.3, pad_out=0.3)
    assert (start, end) == (1.0, 2.0)
    assert (pin, pout) == (0.0, 0.0)


def test_padding_runs_freely_past_the_last_word() -> None:
    """Nothing to collide with after the final word except the source itself."""
    two = doc(w("a", 0.0, 1.0), w("b", 1.0, 2.0))
    _, end, _, pout = snap.pad_range(two, 1.0, 2.0, pad_in=0.0, pad_out=0.3)
    assert end == pytest.approx(2.3)
    assert pout == pytest.approx(0.3)


# --------------------------------------------------------------------------
# snap_clip -- the whole of --fix
# --------------------------------------------------------------------------


def test_snap_clip_snaps_both_edges_and_pads() -> None:
    r = snap.snap_clip(
        SIMPLE, 10.3, 14.8, to_sentence_start=True, to_sentence_end=True,
        pad_in=0.12, pad_out=0.25,
    )
    assert r.start == 9.88   # sentence start 10.0, padded back
    assert r.end == 15.25    # sentence end 15.0, padded forward
    assert r.start_snap.distance == pytest.approx(0.3)
    assert r.end_snap.distance == pytest.approx(0.2)
    assert r.start_moved == pytest.approx(0.42)
    assert r.end_moved == pytest.approx(0.45)
    assert r.changed


def test_snap_clip_is_idempotent() -> None:
    kwargs = dict(to_sentence_start=True, to_sentence_end=True, pad_in=0.12, pad_out=0.25)
    once = snap.snap_clip(SIMPLE, 10.31, 14.79, **kwargs)
    twice = snap.snap_clip(SIMPLE, once.start, once.end, **kwargs)
    assert (twice.start, twice.end) == (once.start, once.end)
    assert not twice.changed


def test_snap_clip_rounds_to_millisecond_precision() -> None:
    odd = doc(w("a", 1.00049, 2.0, ss=True, se=True))
    r = snap.snap_clip(odd, 1.0, 2.0, pad_in=0.0, pad_out=0.0)
    assert r.start == 1.0
    assert len(str(r.start).split(".")[-1]) <= snap.PRECISION


def test_snap_clip_leaves_a_degenerate_range_alone() -> None:
    """Both edges snapping to the same place would produce a zero-length clip;
    better to hand the caller back what it gave us and let lint complain."""
    # A range stranded in a long gap: the start is nearest the *later* word
    # and the end is nearest the *earlier* one, so snapping would inverted the
    # range.
    far = doc(w("a", 0.0, 1.0, ss=True, se=True), w("b", 10.0, 11.0, ss=True, se=True))
    r = snap.snap_clip(far, 5.6, 5.7, to_sentence_start=True, to_sentence_end=True)
    assert r.start_snap.time == 10.0 and r.end_snap.time == 1.0
    assert r.start == 5.6 and r.end == 5.7


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------


def test_words_in_span_uses_the_midpoint_rule() -> None:
    # Starting a little inside "Eighteen" keeps it; starting past its midpoint
    # does not.
    assert [x.text for x in snap.words_in_span(SIMPLE, 10.2, 11.4)] == ["Eighteen", "months."]
    assert [x.text for x in snap.words_in_span(SIMPLE, 10.4, 11.4)] == ["months."]


def test_padding_into_silence_does_not_drag_in_a_neighbour() -> None:
    assert [x.text for x in snap.words_in_span(SIMPLE, 12.28, 15.25)] == [
        "That", "is", "the", "number."
    ]


def test_words_in_span_of_an_empty_range() -> None:
    assert snap.words_in_span(SIMPLE, 11.5, 12.3) == []


def test_span_text_strips_leading_spaces() -> None:
    """Whisper emits most tokens with a leading space; concatenating raw would
    double them."""
    spaced = doc(w("Eighteen", 0.0, 1.0), w(" months.", 1.0, 2.0))
    assert snap.span_text(spaced.words) == "Eighteen months."


# --------------------------------------------------------------------------
# check_boundary -- the measurement behind the time.* rules
# --------------------------------------------------------------------------


def test_boundary_check_accepts_an_exact_edge() -> None:
    c = snap.check_boundary(SIMPLE, 12.4, "start", tolerance=0.08, pad=0.12)
    assert c.ok and c.offset == pytest.approx(0.0)


def test_boundary_tolerance_is_asymmetric_for_a_start() -> None:
    """Early by up to tolerance + pad_in is the padded case and legal; late by
    more than the tolerance means the clip has already cut into the word."""
    early = snap.check_boundary(SIMPLE, 12.4 - 0.19, "start", tolerance=0.08, pad=0.12)
    too_early = snap.check_boundary(SIMPLE, 12.4 - 0.25, "start", tolerance=0.08, pad=0.12)
    late = snap.check_boundary(SIMPLE, 12.4 + 0.05, "start", tolerance=0.08, pad=0.12)
    too_late = snap.check_boundary(SIMPLE, 12.4 + 0.15, "start", tolerance=0.08, pad=0.12)
    assert early.ok and late.ok
    assert not too_early.ok and not too_late.ok


def test_boundary_tolerance_is_asymmetric_for_an_end() -> None:
    late = snap.check_boundary(SIMPLE, 11.4 + 0.3, "end", tolerance=0.08, pad=0.25)
    too_late = snap.check_boundary(SIMPLE, 11.4 + 0.4, "end", tolerance=0.08, pad=0.25)
    too_early = snap.check_boundary(SIMPLE, 11.4 - 0.2, "end", tolerance=0.08, pad=0.25)
    assert late.ok
    assert not too_late.ok and not too_early.ok


def test_boundary_check_reports_the_word_a_cut_lands_inside() -> None:
    c = snap.check_boundary(SIMPLE, 14.0, "end", tolerance=0.08, pad=0.25)
    assert not c.ok
    assert c.inside_word is not None and c.inside_word.text == "number."
    assert c.nearest is not None and c.nearest.time == 13.4


def test_boundary_check_in_silence_reports_no_inside_word() -> None:
    c = snap.check_boundary(SIMPLE, 11.9, "start", tolerance=0.08, pad=0.12)
    assert not c.ok and c.inside_word is None
    assert c.distance == pytest.approx(0.5)


def test_boundary_check_without_words() -> None:
    c = snap.check_boundary(doc(), 1.0, "start", tolerance=0.08)
    assert not c.ok and c.nearest is None and c.offset is None and c.distance is None


# --------------------------------------------------------------------------
# against the real fixture transcript
# --------------------------------------------------------------------------


def test_every_fixture_clip_edge_sits_on_a_real_boundary() -> None:
    words = WordsDoc.model_validate(json.loads((FIXTURES / "words.json").read_text()))
    clips = json.loads((FIXTURES / "clips.valid.json").read_text())["clips"]
    for clip in clips:
        s = snap.check_boundary(words, clip["start"], "start", tolerance=0.08, pad=0.12)
        e = snap.check_boundary(words, clip["end"], "end", tolerance=0.08, pad=0.25)
        assert s.ok, f"{clip['id']} start"
        assert e.ok, f"{clip['id']} end"
        assert s.nearest is not None and s.nearest.word.sentence_start
        assert e.nearest is not None and e.nearest.word.sentence_end


def test_fixture_clip_text_round_trips_through_span_text() -> None:
    words = WordsDoc.model_validate(json.loads((FIXTURES / "words.json").read_text()))
    clips = json.loads((FIXTURES / "clips.valid.json").read_text())["clips"]
    for clip in clips:
        span = snap.words_in_span(words, clip["start"], clip["end"])
        assert snap.span_text(span) == clip["source_text"]
