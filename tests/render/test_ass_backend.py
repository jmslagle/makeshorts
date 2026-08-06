"""ASS backend tests.

Nothing on this machine will render one of these files -- ffmpeg here has no
libass. That is exactly why the format has to be checked structurally: there
is no visual feedback loop to catch a mistake, so the parts a parser is strict
about (colour byte order, timecode format, the Style field count) are asserted
directly.
"""

from __future__ import annotations

import re

import pytest

from makeshorts.render.captions.ass_backend import (
    AssCaptionBackend,
    ass_color,
    ass_time,
    build_ass,
)
from makeshorts.render.captions.cues import CueOptions, build_cues
from makeshorts.render.captions.style import DEFAULT_STYLES

from tests.render.conftest import evenly_spaced

OUT = (1080, 1920)
STYLE_FORMAT_FIELDS = 23  # the count in the V4+ Format line


@pytest.fixture
def cues():
    doc = evenly_spaced(
        "Eighteen months is the number that kills companies quietly.", per_word=0.4
    )
    return build_cues(doc, 0.0, 20.0, CueOptions())


@pytest.fixture
def doc(cues):
    return build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_ass_color_reverses_bytes_and_inverts_alpha():
    assert ass_color((255, 212, 0, 255)) == "&H0000D4FF&"  # opaque amber
    assert ass_color((0, 0, 0, 179)) == "&H4C000000&"
    assert ass_color((0, 0, 0, 0)) == "&HFF000000&", "alpha 0 is FF in ASS"


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0:00:00.00"),
        (1.5, "0:00:01.50"),
        (61.234, "0:01:01.23"),
        (3661.0, "1:01:01.00"),
        (-3.0, "0:00:00.00"),
    ],
)
def test_ass_time(seconds, expected):
    assert ass_time(seconds) == expected


# --------------------------------------------------------------------------
# Document structure
# --------------------------------------------------------------------------


def test_the_required_sections_are_present(doc):
    for section in ("[Script Info]", "[V4+ Styles]", "[Events]"):
        assert section in doc


def test_playres_matches_the_output_so_libass_does_not_rescale(doc):
    assert "PlayResX: 1080" in doc
    assert "PlayResY: 1920" in doc


def test_the_style_line_has_exactly_the_declared_field_count(doc):
    fmt = next(l for l in doc.splitlines() if l.startswith("Format: Name,"))
    style = next(l for l in doc.splitlines() if l.startswith("Style:"))
    assert len(fmt.split("Format:")[1].split(",")) == STYLE_FORMAT_FIELDS
    assert len(style.split("Style:")[1].split(",")) == STYLE_FORMAT_FIELDS


def test_one_dialogue_line_per_cue(cues, doc):
    events = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == len(cues)


def test_dialogue_times_match_the_cues(cues, doc):
    events = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    for cue, line in zip(cues, events, strict=True):
        _, start, end, *_ = line.split(",")
        assert start == ass_time(cue.start)
        assert end == ass_time(cue.end)


# --------------------------------------------------------------------------
# Karaoke
# --------------------------------------------------------------------------


def test_every_word_gets_a_k_tag(cues, doc):
    events = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    for cue, line in zip(cues, events, strict=True):
        assert len(re.findall(r"\{\\k\d+\}", line)) == len(cue.words)


def test_k_durations_sum_to_the_cue_duration(cues, doc):
    events = [l for l in doc.splitlines() if l.startswith("Dialogue:")]
    for cue, line in zip(cues, events, strict=True):
        total_cs = sum(int(m) for m in re.findall(r"\{\\k(\d+)\}", line))
        assert total_cs == pytest.approx(cue.duration * 100, abs=len(cue.words) + 1)


def test_primary_is_the_highlight_because_k_fills_into_it(cues):
    """`\\k` moves a word from SecondaryColour to PrimaryColour as it is sung."""
    style = DEFAULT_STYLES["pill-karaoke"]
    line = next(
        l for l in build_ass(cues, style, OUT).splitlines() if l.startswith("Style:")
    )
    fields = line.split("Style:")[1].split(",")
    assert fields[3].strip() == ass_color(style.highlight_rgba)  # PrimaryColour
    assert fields[4].strip() == ass_color(style.fill_rgba)  # SecondaryColour


def test_a_non_karaoke_style_emits_no_k_tags(cues):
    text = build_ass(cues, DEFAULT_STYLES["static-block"], OUT)
    assert "\\k" not in text
    line = next(l for l in text.splitlines() if l.startswith("Style:"))
    fields = line.split("Style:")[1].split(",")
    assert fields[3].strip() == fields[4].strip(), "no fill means both colours match"


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_line_breaks_use_the_ass_escape(cues, doc):
    multi = [c for c in cues if len(c.lines) > 1]
    assert multi, "fixture must produce at least one two-line cue"
    assert "\\N" in doc


@pytest.mark.parametrize(
    "position,alignment", [("lower_third", "2"), ("center", "5"), ("upper_third", "8")]
)
def test_alignment_follows_the_position(cues, position, alignment):
    text = build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT, position)
    assert _field(text, 19) == alignment  # Alignment is the 19th Format field


def test_the_pill_becomes_an_opaque_box(cues):
    """libass has no rounded corners; BorderStyle 4 is the closest it gets."""
    with_pill = build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT)
    without = build_ass(cues, DEFAULT_STYLES["clean-bold"], OUT)
    assert _field(with_pill, 16) == "4"
    assert _field(without, 16) == "1"


def test_margin_v_clears_the_safe_area(cues):
    style = DEFAULT_STYLES["pill-karaoke"]
    text = build_ass(cues, style, OUT, "lower_third")
    assert int(_field(text, 22)) == round(style.safe_area.bottom_pct * OUT[1])


def _field(text: str, index: int) -> str:
    line = next(l for l in text.splitlines() if l.startswith("Style:"))
    return line.split("Style:")[1].split(",")[index - 1].strip()


# --------------------------------------------------------------------------
# Escaping and uppercase
# --------------------------------------------------------------------------


def test_braces_in_the_transcript_are_escaped():
    doc = evenly_spaced("use {curly} braces carefully", per_word=0.4)
    cues = build_cues(doc, 0.0, 5.0, CueOptions())
    text = build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT)
    assert "\\{curly\\}" in text
    # The override blocks the parser cares about are still intact.
    assert re.search(r"\{\\k\d+\}", text)


def test_uppercase_style_uppercases_the_events():
    doc = evenly_spaced("quiet words here", per_word=0.4)
    cues = build_cues(doc, 0.0, 5.0, CueOptions())
    text = build_ass(cues, DEFAULT_STYLES["impact-punch"], OUT)
    assert "QUIET" in text and "quiet" not in text


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


def test_backend_writes_a_content_addressed_file(tmp_path, cues):
    backend = AssCaptionBackend(caps=object())
    a = backend.render(cues, DEFAULT_STYLES["pill-karaoke"], OUT, tmp_path)
    assert a.kind == "subtitle_file" and a.format == "ass"
    from pathlib import Path

    assert Path(a.path).is_file()
    b = backend.render(cues, DEFAULT_STYLES["pill-karaoke"], OUT, tmp_path)
    assert a.path == b.path, "the same cues must not produce a second file"
    c = backend.render(cues, DEFAULT_STYLES["clean-bold"], OUT, tmp_path)
    assert c.path != a.path


def test_backend_is_dormant_without_libass():
    class NoAss:
        has_libass = False

    class HasAss:
        has_libass = True

    assert not AssCaptionBackend(caps=NoAss()).is_available()
    assert AssCaptionBackend(caps=HasAss()).is_available()


def test_empty_cues_still_produce_a_valid_document():
    text = build_ass([], DEFAULT_STYLES["pill-karaoke"], OUT)
    assert "[Events]" in text
    assert not [l for l in text.splitlines() if l.startswith("Dialogue:")]
