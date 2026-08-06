"""Cue grouping tests.

Grouping is where captions become readable or not, and the failure modes are
all quiet ones: a cue that overruns the frame, a cue that lags the audio, a
word dropped at a clip boundary. None of them raise.
"""

from __future__ import annotations

import pytest

from makeshorts.artifacts import Word, WordsDoc
from makeshorts.render.captions.cues import Cue, CueOptions, build_cues, cues_for_clip
from makeshorts.render.captions.style import DEFAULT_STYLES

from tests.render.conftest import evenly_spaced, words

OPTS = CueOptions(max_chars_per_line=26, max_lines=2)


def assert_well_formed(cues: list[Cue], opts: CueOptions = OPTS) -> None:
    for i, cue in enumerate(cues):
        assert cue.words, "an empty cue would render a bare pill"
        assert cue.lines, "every cue must have at least one line"
        assert len(cue.lines) <= opts.max_lines
        flat = [w for line in cue.lines for w in line]
        assert flat == list(range(len(cue.words))), "lines must cover the words in order"
        assert cue.start <= cue.words[0].start + 1e-6
        assert cue.end >= cue.words[-1].end - 1e-6
        assert cue.start >= 0.0
        for a, b in zip(cue.words, cue.words[1:], strict=False):
            assert a.start <= b.start
        if i:
            assert cue.start >= cues[i - 1].start
            assert cues[i - 1].end <= cue.start + 1e-6, "cues must not overlap"


# --------------------------------------------------------------------------
# Times are clip-relative
# --------------------------------------------------------------------------


def test_times_are_relative_to_the_clip_not_the_source(sample_words):
    cues = build_cues(sample_words, 10.0, 18.0, OPTS)
    assert cues
    assert cues[0].start == pytest.approx(0.0, abs=0.01)
    assert all(w.start >= 0 for c in cues for w in c.words)
    assert all(w.end <= 8.0 + 1e-6 for c in cues for w in c.words)
    assert_well_formed(cues)


def test_words_outside_the_range_are_excluded():
    doc = evenly_spaced("one two three four five six", start=0.0, per_word=1.0)
    cues = build_cues(doc, 2.0, 4.0, OPTS)
    assert [w.text for c in cues for w in c.words] == ["three", "four"]


def test_a_word_straddling_the_boundary_is_kept_and_clamped():
    """Half a word is still audible in the clip, so it must be captioned."""
    doc = words([("straddler", 4.5, 5.5), ("inside", 5.5, 6.0)])
    cues = build_cues(doc, 5.0, 6.0, OPTS)
    texts = [w.text for c in cues for w in c.words]
    assert texts == ["straddler", "inside"]
    first = cues[0].words[0]
    assert first.start == 0.0, "clamped to the clip start, not negative"
    assert first.end == pytest.approx(0.5)


def test_empty_range_yields_no_cues(sample_words):
    assert build_cues(sample_words, 10.0, 10.0, OPTS) == []
    assert build_cues(sample_words, 20.0, 10.0, OPTS) == []


def test_range_with_no_speech_yields_no_cues(sample_words):
    """A silent clip renders without captions rather than failing."""
    assert build_cues(sample_words, 500.0, 540.0, OPTS) == []


def test_empty_words_doc():
    assert build_cues(WordsDoc(model="m", language="en", words=[]), 0.0, 10.0, OPTS) == []


# --------------------------------------------------------------------------
# Line and cue sizing
# --------------------------------------------------------------------------


def test_no_line_exceeds_the_character_budget():
    doc = evenly_spaced(
        "the quick brown fox jumps over the lazy dog and keeps on running "
        "well past the end of the sentence",
        per_word=0.3,
    )
    cues = build_cues(doc, 0.0, 60.0, OPTS)
    assert_well_formed(cues)
    for cue in cues:
        for text in cue.line_texts():
            assert len(text) <= OPTS.max_chars_per_line


@pytest.mark.parametrize("max_lines", [1, 2, 3])
@pytest.mark.parametrize("max_chars", [12, 26, 48])
def test_budgets_are_respected_across_settings(max_chars, max_lines):
    opts = CueOptions(max_chars_per_line=max_chars, max_lines=max_lines)
    doc = evenly_spaced(
        "we shipped it on a Tuesday and nobody noticed until the following "
        "Thursday when support lit up like a christmas tree",
        per_word=0.25,
    )
    cues = build_cues(doc, 0.0, 60.0, opts)
    assert_well_formed(cues, opts)
    for cue in cues:
        assert all(len(t) <= max_chars for t in cue.line_texts())


def test_a_single_very_long_word_gets_its_own_cue_and_does_not_hang():
    monster = "Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz"
    doc = words([("okay", 0.0, 0.4), (monster, 0.4, 2.0), ("right", 2.0, 2.4)])
    cues = build_cues(doc, 0.0, 3.0, OPTS)
    assert_well_formed(cues)
    all_text = " ".join(c.text for c in cues)
    assert monster in all_text, "an unsplittable word must not be dropped"
    # It overruns the budget by necessity; the renderer shrinks to fit.
    assert any(len(t) > OPTS.max_chars_per_line for c in cues for t in c.line_texts())


def test_text_with_no_punctuation_still_groups():
    doc = evenly_spaced(" ".join(["word"] * 40), per_word=0.3)
    cues = build_cues(doc, 0.0, 20.0, OPTS)
    assert len(cues) > 1
    assert_well_formed(cues)
    assert " ".join(c.text for c in cues) == " ".join(["word"] * 40)


def test_every_word_survives_grouping(sample_words):
    cues = build_cues(sample_words, 0.0, 100.0, OPTS)
    got = [w.text for c in cues for w in c.words]
    assert got == [w.text for w in sample_words.words]


# --------------------------------------------------------------------------
# Boundary preference
# --------------------------------------------------------------------------


def test_cues_prefer_to_break_at_a_sentence_end():
    doc = evenly_spaced(
        "Eighteen months. That is the number that kills companies quietly.",
        per_word=0.35,
    )
    cues = build_cues(doc, 0.0, 30.0, OPTS)
    assert cues[0].text == "Eighteen months."
    assert_well_formed(cues)


def test_sentence_end_flags_are_honoured_without_punctuation():
    """A transcript that marks sentences but drops the period still breaks."""
    doc = words(
        [
            ("eighteen", 0.0, 0.4),
            ("months", 0.4, 0.8),
            ("that", 0.8, 1.2),
            ("is", 1.2, 1.6),
            ("the", 1.6, 2.0),
            ("number", 2.0, 2.4),
        ],
        sentence_ends={1},
    )
    cues = build_cues(doc, 0.0, 5.0, CueOptions(max_chars_per_line=60, max_lines=2))
    assert [c.text for c in cues] == ["eighteen months", "that is the number"]


def test_clause_boundary_beats_an_arbitrary_split():
    doc = evenly_spaced(
        "nobody checks it, and that is the whole problem here today", per_word=0.3
    )
    cues = build_cues(doc, 0.0, 30.0, CueOptions(max_chars_per_line=20, max_lines=1))
    assert cues[0].text.endswith(",")


def test_a_long_pause_starts_a_new_cue():
    doc = words(
        [("before", 0.0, 0.5), ("pause", 0.5, 1.0), ("after", 4.0, 4.5)],
    )
    cues = build_cues(doc, 0.0, 6.0, OPTS)
    assert [c.text for c in cues] == ["before pause", "after"]


def test_a_cue_never_outstays_max_cue_duration():
    opts = CueOptions(max_chars_per_line=120, max_lines=6, max_cue_duration=3.0)
    doc = evenly_spaced(" ".join(f"w{i}" for i in range(30)), per_word=0.5)
    cues = build_cues(doc, 0.0, 20.0, opts)
    assert all(c.words[-1].end - c.words[0].start <= 3.0 + 1e-6 for c in cues)


# --------------------------------------------------------------------------
# Gap filling and karaoke states
# --------------------------------------------------------------------------


def test_short_gaps_are_held_rather_than_blinking():
    doc = words([("aaaaaa", 0.0, 0.4), ("bbbbbb", 1.0, 1.4)])
    cues = build_cues(doc, 0.0, 3.0, CueOptions(max_chars_per_line=6, max_lines=1))
    assert len(cues) == 2
    # The first cue is held toward the next rather than ending at 0.4.
    assert cues[0].end > cues[0].words[-1].end
    assert cues[0].end <= cues[1].start + 1e-6


def test_gap_fill_never_runs_past_the_clip():
    doc = words([("last", 0.0, 0.4)])
    cues = build_cues(doc, 0.0, 0.5, CueOptions(gap_fill=5.0))
    assert cues[0].end <= 0.5


def test_karaoke_states_tile_the_cue_with_no_holes():
    doc = evenly_spaced("one two three four", per_word=0.5)
    cue = build_cues(doc, 0.0, 4.0, OPTS)[0]
    states = cue.karaoke_states()
    assert [i for i, _, _ in states] == [0, 1, 2, 3]
    assert states[0][1] == pytest.approx(cue.start)
    assert states[-1][2] == pytest.approx(cue.end)
    for (_, _, e), (_, s2, _) in zip(states, states[1:], strict=False):
        assert e == pytest.approx(s2), "a hole here means an unlit frame"


def test_karaoke_states_absorb_an_internal_pause():
    """Between words the *preceding* word stays lit, which is how a pause reads."""
    doc = words([("hold", 0.0, 0.3), ("on", 1.0, 1.3)])
    cue = build_cues(doc, 0.0, 2.0, OPTS)[0]
    states = cue.karaoke_states()
    assert states[0][2] == pytest.approx(1.0), "first word holds through the gap"


def test_single_word_cue_has_one_state():
    doc = words([("solo", 0.0, 0.5)])
    cue = build_cues(doc, 0.0, 1.0, OPTS)[0]
    assert len(cue.karaoke_states()) == 1


# --------------------------------------------------------------------------
# Cue helpers
# --------------------------------------------------------------------------


def test_line_of_locates_a_word():
    doc = evenly_spaced("alpha bravo charlie delta echo foxtrot golf", per_word=0.3)
    cue = build_cues(doc, 0.0, 10.0, CueOptions(max_chars_per_line=14, max_lines=2))[0]
    assert cue.line_of(0) == 0
    assert cue.line_of(len(cue.words) - 1) == len(cue.lines) - 1
    with pytest.raises(IndexError):
        cue.line_of(999)


def test_cues_for_clip_takes_limits_from_the_style():
    doc = evenly_spaced(" ".join(["word"] * 30), per_word=0.3)
    style = DEFAULT_STYLES["impact-punch"]  # 18 chars, 1 line
    cues = cues_for_clip(doc, 0.0, 20.0, style)
    for cue in cues:
        assert len(cue.lines) == 1
        assert all(len(t) <= 18 for t in cue.line_texts())


def test_out_of_order_input_is_sorted_not_trusted():
    doc = WordsDoc(
        model="m",
        language="en",
        words=[
            Word(text="second", start=1.0, end=1.4),
            Word(text="first", start=0.0, end=0.4),
        ],
    )
    cues = build_cues(doc, 0.0, 3.0, OPTS)
    assert [w.text for c in cues for w in c.words] == ["first", "second"]


def test_blank_tokens_are_dropped():
    doc = words([("real", 0.0, 0.4), ("   ", 0.4, 0.5), ("words", 0.5, 0.9)])
    cues = build_cues(doc, 0.0, 2.0, OPTS)
    assert [w.text for c in cues for w in c.words] == ["real", "words"]


# --------------------------------------------------------------------------
# Stress: the budget invariant must hold for arbitrary text
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_grouping_never_exceeds_the_budget_on_random_text(seed):
    """Random word lengths and punctuation, checked against the invariant.

    The greedy break-at-a-boundary path leaves a remainder that must itself
    still fit, which is easy to get subtly wrong and impossible to notice from
    a handful of hand-written sentences.
    """
    import random

    rng = random.Random(seed)
    toks = []
    for _ in range(rng.randint(5, 80)):
        word = "x" * rng.randint(1, 14)
        if rng.random() < 0.15:
            word += rng.choice([",", ".", ";", "?"])
        toks.append(word)
    doc = evenly_spaced(" ".join(toks), per_word=rng.choice([0.2, 0.4, 0.9]))
    opts = CueOptions(
        max_chars_per_line=rng.choice([8, 16, 26, 40]),
        max_lines=rng.choice([1, 2, 3]),
    )
    cues = build_cues(doc, 0.0, 200.0, opts)
    assert_well_formed(cues, opts)
    for cue in cues:
        for text in cue.line_texts():
            # A line may only overrun when one unsplittable word is that long.
            longest = max(len(w.text) for w in cue.words)
            assert len(text) <= max(opts.max_chars_per_line, longest)
    assert " ".join(c.text for c in cues) == " ".join(toks), "no word may be lost"
