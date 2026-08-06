"""Sentence segmentation, tested without the model.

`sentence_start` / `sentence_end` are what the `start_on_sentence_start` and
`end_on_sentence_end` gates are enforced from, so these flags are load-bearing
in a way the rest of words.json is not. The segmenter is a pure function over
`Word` objects precisely so it can be tested exhaustively here, at zero cost,
instead of behind a 1.5GB model download.
"""

from __future__ import annotations

import importlib.util

import pytest

from makeshorts.artifacts import Word, WordsDoc
from makeshorts.prepare.transcribe import (
    is_sentence_end,
    mark_sentence_boundaries,
    words_from_segments,
)


def flags(text: str) -> list[tuple[str, bool, bool]]:
    """`(token, sentence_start, sentence_end)` for a whitespace-split sentence."""
    words = [
        Word(text=t, start=float(i), end=float(i) + 0.5) for i, t in enumerate(text.split())
    ]
    return [(w.text, w.sentence_start, w.sentence_end) for w in mark_sentence_boundaries(words)]


def ends(text: str) -> list[bool]:
    return [e for _, _, e in flags(text)]


def starts(text: str) -> list[bool]:
    return [s for s, in ((s,) for _, s, _ in flags(text))]


# -- is_sentence_end, token by token ---------------------------------------


@pytest.mark.parametrize(
    "token",
    ["done.", "really?", "stop!", 'said."', "wait!)", "over.'", "yes!”", "(right?)"],
)
def test_terminal_punctuation_ends_a_sentence(token: str) -> None:
    """Closing quotes and brackets sit *after* the terminator."""
    assert is_sentence_end(token, "Next") is True


@pytest.mark.parametrize("token", ["hello", "eighteen", "months,", "well-", '"quote'])
def test_non_terminal_tokens_do_not_end_a_sentence(token: str) -> None:
    assert is_sentence_end(token, "next") is False


@pytest.mark.parametrize("token", ["Mr.", "Dr.", "Mrs.", "vs.", "etc.", "Inc.", "Sept.", "No."])
def test_abbreviations_do_not_end_a_sentence(token: str) -> None:
    assert is_sentence_end(token, "Reyes") is False


@pytest.mark.parametrize("token", ["e.g.", "i.e.", "U.S.", "a.m.", "J."])
def test_dotted_acronyms_and_initials_do_not_end_a_sentence(token: str) -> None:
    assert is_sentence_end(token, "Something") is False


@pytest.mark.parametrize("token", [".S.", ".g.", ".m."])
def test_an_acronym_fragment_with_a_leading_dot_does_not_end_a_sentence(token: str) -> None:
    """Whisper splits `U.S.` into `U` + `.S.`, so the fragment carrying the
    trailing period carries a leading one too. Observed on real output."""
    assert is_sentence_end(token, "Market") is False


def test_a_leading_dot_number_fragment_does_not_end_a_sentence() -> None:
    """The same split happens to decimals: `3.5` arrives as `3` + `.5`."""
    assert is_sentence_end("3", ".5") is False
    assert is_sentence_end(".5", "years") is False


def test_a_decimal_split_across_tokens_does_not_end_a_sentence() -> None:
    assert is_sentence_end("3.", "5") is False


def test_a_decimal_inside_one_token_never_looked_terminal() -> None:
    assert is_sentence_end("3.5", "million") is False
    assert is_sentence_end("4.2x", "better") is False


def test_a_year_followed_by_a_capital_does_end_a_sentence() -> None:
    assert is_sentence_end("1999.", "Then") is True


def test_a_lowercase_follower_means_an_unlisted_abbreviation() -> None:
    """Speech-to-text capitalises sentence starts reliably, so a lowercase next
    token means the period belonged to something we did not list."""
    assert is_sentence_end("approx.", "eighteen") is False
    assert is_sentence_end("Ph.D.", "candidates") is False


def test_a_question_mark_ends_a_sentence_even_before_a_lowercase_word() -> None:
    """The lowercase heuristic applies only to the ambiguous period."""
    assert is_sentence_end("really?", "yeah") is True


@pytest.mark.parametrize("token", ["so...", "well…", "hmm.."])
def test_an_ellipsis_is_a_trail_off_not_a_full_stop(token: str) -> None:
    assert is_sentence_end(token, "Then") is False


def test_a_bare_terminator_with_no_word_is_not_a_sentence_end() -> None:
    assert is_sentence_end("", "Next") is False
    assert is_sentence_end('"', "Next") is False


def test_the_last_token_needs_no_follower() -> None:
    assert is_sentence_end("finished.", None) is True


# -- mark_sentence_boundaries over a whole transcript ----------------------


def test_flags_across_two_sentences() -> None:
    assert flags("Eighteen months. That kills companies.") == [
        ("Eighteen", True, False),
        ("months.", False, True),
        ("That", True, False),
        ("kills", False, False),
        ("companies.", False, True),
    ]


def test_an_abbreviation_does_not_open_a_new_sentence_mid_stream() -> None:
    result = flags("Mr. Reyes ran the numbers.")
    assert result[0] == ("Mr.", True, False)
    assert result[1] == ("Reyes", False, False)
    assert result[-1][2] is True


def test_a_decimal_does_not_split_a_sentence() -> None:
    assert ends("Payback was 3.5 months on average.") == [False] * 5 + [True]


def test_the_final_word_always_closes_a_sentence() -> None:
    """A transcript ending without punctuation would otherwise leave the last
    sentence open, and `end_on_sentence_end` would reject every clip that runs
    to the end of the recording."""
    result = flags("and then we just sort of trailed off")
    assert result[-1][2] is True


def test_the_first_word_always_starts_a_sentence() -> None:
    assert flags("mid-thought already in progress")[0][1] is True


def test_every_sentence_start_follows_a_sentence_end() -> None:
    marked = mark_sentence_boundaries(
        [
            Word(text=t, start=float(i), end=float(i) + 0.5)
            for i, t in enumerate("One. Two? Three! Four.".split())
        ]
    )
    for previous, current in zip(marked, marked[1:]):
        assert current.sentence_start == previous.sentence_end


def test_empty_input_is_handled() -> None:
    assert mark_sentence_boundaries([]) == []


def test_marking_does_not_mutate_the_input() -> None:
    original = [Word(text="Done.", start=0.0, end=1.0)]
    marked = mark_sentence_boundaries(original)
    assert marked[0].sentence_end is True
    assert original[0].sentence_end is False


# -- shape conversion ------------------------------------------------------


class FakeWord:
    def __init__(self, word: str, start: float, end: float, probability: float | None = 0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class FakeSegment:
    def __init__(self, words):
        self.words = words
        self.end = words[-1].end if words else 0.0


def test_words_from_segments_strips_whisper_leading_spaces() -> None:
    words = words_from_segments([FakeSegment([FakeWord(" Hello", 0.0, 0.4)])])
    assert words[0].text == "Hello"
    assert words[0].probability == 0.9


def test_words_from_segments_drops_empty_tokens() -> None:
    words = words_from_segments([FakeSegment([FakeWord("  ", 0.0, 0.4), FakeWord(" a", 0.5, 0.8)])])
    assert [w.text for w in words] == ["a"]


def test_words_from_segments_forces_monotonic_non_zero_spans() -> None:
    """Whisper emits overlapping and zero-length words around silence; snap.py
    and cues.py both assume monotonicity."""
    words = words_from_segments(
        [FakeSegment([FakeWord(" a", 0.0, 1.0), FakeWord(" b", 0.5, 0.5), FakeWord(" c", -1.0, 3.0)])]
    )
    assert [w.start for w in words] == [0.0, 1.0, 1.01]
    for word in words:
        assert word.end > word.start
    for previous, current in zip(words, words[1:]):
        assert current.start >= previous.end


def test_words_from_segments_spans_several_segments() -> None:
    words = words_from_segments(
        [
            FakeSegment([FakeWord(" One.", 0.0, 0.5)]),
            FakeSegment([FakeWord(" Two.", 1.0, 1.5)]),
        ]
    )
    assert [w.text for w in words] == ["One.", "Two."]


def test_a_words_doc_built_this_way_validates() -> None:
    doc = WordsDoc(
        model="distil-large-v3",
        language="en",
        words=mark_sentence_boundaries(
            words_from_segments([FakeSegment([FakeWord(" Hi.", 0.0, 0.4)])])
        ),
    )
    assert WordsDoc.model_validate(doc.model_dump(mode="json")) == doc


# -- the live model --------------------------------------------------------


def _cached_model() -> str | None:
    """The name of an already-downloaded whisper model, if there is one.

    Returns the *cached* name rather than a hardcoded one so that running this
    test never triggers a model download -- a test that fetches 1.5GB on a cold
    machine is not a test anyone will keep running.
    """
    if importlib.util.find_spec("faster_whisper") is None:
        return None
    from pathlib import Path

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.is_dir():
        return None
    for entry in sorted(cache.glob("models--*--*whisper*")):
        if not any(entry.glob("snapshots/*/model.bin")):
            continue
        _, org, name = entry.name.split("--", 2)
        return f"{org}/{name}"
    return None


CACHED_MODEL = _cached_model()


@pytest.mark.skipif(CACHED_MODEL is None, reason="no whisper model downloaded")
def test_live_transcription_produces_flagged_words(av_file) -> None:
    """Smoke test only. The fixture is a sine wave, so there is nothing to
    recognise -- what matters is that the call succeeds and the contract holds."""
    from makeshorts.prepare.transcribe import transcribe

    doc = transcribe(av_file, model=CACHED_MODEL, language="en")
    assert isinstance(doc, WordsDoc)
    assert doc.language == "en"
    for word in doc.words:
        assert word.end > word.start
    for previous, current in zip(doc.words, doc.words[1:]):
        assert current.start >= previous.end
    if doc.words:
        assert doc.words[0].sentence_start is True
        assert doc.words[-1].sentence_end is True
