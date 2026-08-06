from __future__ import annotations

import re

import pytest

from makeshorts.prepare.text_outputs import (
    PARAGRAPH_GAP,
    format_clock,
    format_srt_timestamp,
    group_cues,
    render_srt,
    render_transcript,
    wrap_cue_text,
)

from .conftest import evenly_timed, make_words

SRT_TIMING = re.compile(
    r"^(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})$"
)


# -- timestamp formatting --------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00"),
        (59.9, "00:00:59"),
        (60.0, "00:01:00"),
        (3599.0, "00:59:59"),
        (3600.0, "01:00:00"),
        (3612.44, "01:00:12"),
        (-5.0, "00:00:00"),
    ],
)
def test_format_clock(seconds, expected) -> None:
    assert format_clock(seconds) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (0.001, "00:00:00,001"),
        (61.25, "00:01:01,250"),
        (3661.007, "01:01:01,007"),
        (3612.4449, "01:00:12,445"),
        (-1.0, "00:00:00,000"),
    ],
)
def test_format_srt_timestamp(seconds, expected) -> None:
    assert format_srt_timestamp(seconds) == expected


def test_srt_timestamp_never_produces_a_1000_millisecond_field() -> None:
    """Rounding up must carry into the seconds field, not print `,1000`."""
    assert format_srt_timestamp(1.9999) == "00:00:02,000"


# -- transcript.txt --------------------------------------------------------


def test_transcript_lines_all_carry_a_timestamp() -> None:
    doc = evenly_timed(
        "Eighteen months is the number that kills companies and nobody ever checks it "
        "before the board meeting where it finally matters."
    )
    text = render_transcript(doc)
    for line in text.splitlines():
        if line:
            assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] ", line), line


def test_transcript_wraps_to_the_requested_width() -> None:
    doc = evenly_timed(" ".join(["word"] * 80))
    for line in render_transcript(doc, width=60).splitlines():
        assert len(line) <= 60


def test_transcript_timestamp_matches_the_first_word_of_its_line() -> None:
    doc = evenly_timed(" ".join(["word"] * 40), start=125.0, per_word=1.0)
    first = render_transcript(doc, width=40).splitlines()[0]
    assert first.startswith("[00:02:05] ")


def test_a_long_pause_becomes_a_paragraph_break() -> None:
    doc = make_words(
        ("First", 0.0, 0.4),
        ("thought.", 0.5, 0.9),
        ("Much", 0.9 + PARAGRAPH_GAP + 1.0, 0.9 + PARAGRAPH_GAP + 1.4),
        ("later.", 0.9 + PARAGRAPH_GAP + 1.5, 0.9 + PARAGRAPH_GAP + 1.9),
    )
    lines = render_transcript(doc).splitlines()
    assert "" in lines
    assert lines.index("") == 1


def test_transcript_ends_with_exactly_one_newline_and_no_trailing_blank() -> None:
    doc = evenly_timed("A short line.")
    text = render_transcript(doc)
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_transcript_preserves_every_word_in_order() -> None:
    sentence = "Eighteen months. That is the number that kills companies. Nobody checks it."
    doc = evenly_timed(sentence)
    body = re.sub(r"\[\d{2}:\d{2}:\d{2}\] ", "", render_transcript(doc, width=44))
    assert body.split() == sentence.split()


def test_empty_transcript_is_empty() -> None:
    assert render_transcript(make_words()) == ""


# -- cue grouping ----------------------------------------------------------


def test_cues_break_at_sentence_ends() -> None:
    doc = evenly_timed("One two. Three four.")
    cues = group_cues(doc.words)
    assert [[w.text for w in cue] for cue in cues] == [["One", "two."], ["Three", "four."]]


def test_cues_break_at_the_character_budget() -> None:
    doc = evenly_timed(" ".join(["word"] * 40))
    for cue in group_cues(doc.words, max_chars=20):
        assert len(" ".join(w.text for w in cue)) <= 20


def test_cues_break_at_the_duration_budget() -> None:
    doc = evenly_timed(" ".join(["word"] * 40), per_word=1.0)
    for cue in group_cues(doc.words, max_duration=3.0, max_chars=1000):
        assert cue[-1].end - cue[0].start <= 3.0 + 1.0


def test_cues_break_after_a_pause() -> None:
    doc = make_words(("one", 0.0, 0.4), ("two", 5.0, 5.4))
    assert len(group_cues(doc.words, max_chars=1000)) == 2


def test_grouping_loses_no_words() -> None:
    doc = evenly_timed("Eighteen months is the number. Nobody ever checks it before the board.")
    flat = [w.text for cue in group_cues(doc.words, max_chars=25) for w in cue]
    assert flat == [w.text for w in doc.words]


# -- raw.srt ---------------------------------------------------------------


def parse_srt(text: str) -> list[tuple[int, str, str, str]]:
    blocks = [b for b in text.split("\n\n") if b.strip()]
    parsed = []
    for block in blocks:
        lines = block.strip().split("\n")
        match = SRT_TIMING.match(lines[1])
        assert match, f"bad timing line: {lines[1]!r}"
        parsed.append((int(lines[0]), match.group(1), match.group(2), "\n".join(lines[2:])))
    return parsed


def test_srt_is_structurally_valid() -> None:
    doc = evenly_timed(
        "Eighteen months is the number that kills companies. Nobody ever checks it. "
        "The board only asks about it once."
    )
    entries = parse_srt(render_srt(doc))
    assert [e[0] for e in entries] == list(range(1, len(entries) + 1))
    for _, start, end, body in entries:
        assert start < end
        assert body.strip()


def test_srt_cues_never_overlap() -> None:
    doc = evenly_timed(" ".join(["word"] * 60), per_word=0.2)
    entries = parse_srt(render_srt(doc, max_chars=20))
    for previous, current in zip(entries, entries[1:]):
        assert previous[2] <= current[1], f"{previous[2]} overlaps {current[1]}"


def test_a_very_short_cue_is_extended_but_not_past_the_next_one() -> None:
    doc = make_words(("Hi.", 0.0, 0.05), ("Then.", 0.3, 0.9))
    entries = parse_srt(render_srt(doc))
    assert entries[0][2] == "00:00:00,300"


def test_srt_cue_times_come_from_the_words() -> None:
    doc = make_words(("Hello", 12.0, 12.5), ("there.", 12.6, 13.4))
    entries = parse_srt(render_srt(doc))
    assert entries[0][1] == "00:00:12,000"
    assert entries[0][2] == "00:00:13,400"


def test_srt_wraps_a_long_cue_onto_two_balanced_lines() -> None:
    doc = evenly_timed(
        "Eighteen months is the number that quietly kills perfectly good companies"
    )
    body = parse_srt(render_srt(doc))[0][3]
    lines = body.split("\n")
    assert len(lines) == 2
    # Balanced, not greedy: 40/8 would be ugly and is free to avoid.
    assert abs(len(lines[0]) - len(lines[1])) < 20


def test_wrap_cue_text_of_a_single_word_is_one_line() -> None:
    doc = make_words(("Right.", 0.0, 0.5))
    assert "\n" not in wrap_cue_text(doc.words)


def test_empty_srt_is_empty() -> None:
    assert render_srt(make_words()) == ""


def test_writers_create_files(tmp_path) -> None:
    from makeshorts.prepare.text_outputs import write_srt, write_transcript

    doc = evenly_timed("Eighteen months. Nobody checks it.")
    transcript = write_transcript(doc, tmp_path / "sub" / "transcript.txt")
    srt = write_srt(doc, tmp_path / "sub" / "raw.srt")
    assert transcript.read_text().startswith("[00:00:00]")
    assert srt.read_text().startswith("1\n")
