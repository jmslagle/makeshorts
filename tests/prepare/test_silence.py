"""silencedetect parsing.

Most of this tests captured stderr rather than a live encode. The edge cases
that matter -- a trailing silence ffmpeg never closes, a negative start, a
truncated log -- are ones we cannot reliably provoke from a real file on this
ffmpeg build, and pinning them to strings is how they stay fixed.
"""

from __future__ import annotations

import pytest

from makeshorts.artifacts import SilenceDoc
from makeshorts.prepare.silence import (
    DEFAULT_MIN_DURATION,
    DEFAULT_THRESHOLD_DB,
    detect_silence,
    parse_silencedetect,
    silencedetect_command,
)

from .conftest import requires_ffmpeg

# Real output from ffmpeg 8.1.2, banner stripped.
PAIRED = """\
[Parsed_silencedetect_0 @ 0x946c29c80] silence_start: 2.020136
[Parsed_silencedetect_0 @ 0x946c29c80] silence_end: 4.017143 | silence_duration: 1.997007
"""

MULTIPLE = """\
[Parsed_silencedetect_0 @ 0x14f704080] silence_start: 5.5
[Parsed_silencedetect_0 @ 0x14f704080] silence_end: 8.25 | silence_duration: 2.75
size=N/A time=00:00:12.00 bitrate=N/A speed=  48x
[Parsed_silencedetect_0 @ 0x14f704080] silence_start: 30.125
[Parsed_silencedetect_0 @ 0x14f704080] silence_end: 31.5 | silence_duration: 1.375
"""

# The case the parser exists for: the recording ends while still silent and the
# filter is flushed at EOF without printing a silence_end.
UNTERMINATED = """\
[Parsed_silencedetect_0 @ 0x14f704080] silence_start: 10.0
[Parsed_silencedetect_0 @ 0x14f704080] silence_end: 12.0 | silence_duration: 2.0
[Parsed_silencedetect_0 @ 0x14f704080] silence_start: 57.44
frame= 1800 fps=0.0 q=-1.0 Lsize=N/A time=00:01:00.00 bitrate=N/A speed= 120x
"""


def test_parses_a_paired_span() -> None:
    spans = parse_silencedetect(PAIRED, duration=6.0)
    assert [(s.start, s.end) for s in spans] == [(2.02, 4.017)]


def test_parses_several_spans_interleaved_with_progress_lines() -> None:
    spans = parse_silencedetect(MULTIPLE, duration=40.0)
    assert [(s.start, s.end) for s in spans] == [(5.5, 8.25), (30.125, 31.5)]


def test_unterminated_trailing_silence_is_closed_at_the_media_duration() -> None:
    spans = parse_silencedetect(UNTERMINATED, duration=60.0)
    assert [(s.start, s.end) for s in spans] == [(10.0, 12.0), (57.44, 60.0)]
    assert spans[-1].duration == pytest.approx(2.56)


def test_unterminated_trailing_silence_is_dropped_when_duration_is_unknown() -> None:
    """Better a missing span than one with an invented end."""
    spans = parse_silencedetect(UNTERMINATED, duration=None)
    assert [(s.start, s.end) for s in spans] == [(10.0, 12.0)]


def test_unterminated_span_shorter_than_min_duration_is_dropped() -> None:
    """The duration from media.json can land just before the reported start
    when the container rounds; that is not a 0.05s pause."""
    spans = parse_silencedetect(UNTERMINATED, duration=57.5, min_duration=0.5)
    assert [(s.start, s.end) for s in spans] == [(10.0, 12.0)]


def test_duration_before_an_unterminated_start_produces_no_span() -> None:
    assert parse_silencedetect("silence_start: 57.44\n", duration=50.0) == []


def test_negative_start_is_clamped_to_zero() -> None:
    """A file that opens in silence can report a small negative start."""
    stderr = "silence_start: -0.012\nsilence_end: 3.0 | silence_duration: 3.012\n"
    spans = parse_silencedetect(stderr, duration=10.0)
    assert [(s.start, s.end) for s in spans] == [(0.0, 3.0)]


def test_end_without_a_start_is_ignored() -> None:
    stderr = "silence_end: 4.0 | silence_duration: 2.0\nsilence_start: 9.0\n"
    assert parse_silencedetect(stderr, duration=None) == []


def test_second_start_before_an_end_supersedes_the_first() -> None:
    stderr = "silence_start: 1.0\nsilence_start: 4.0\nsilence_end: 6.0\n"
    spans = parse_silencedetect(stderr, duration=None)
    assert [(s.start, s.end) for s in spans] == [(4.0, 6.0)]


def test_end_line_without_a_duration_field_still_parses() -> None:
    """silence_duration is redundant, and not every build prints it."""
    stderr = "silence_start: 1.0\nsilence_end: 2.5\n"
    spans = parse_silencedetect(stderr, duration=None)
    assert [(s.start, s.end) for s in spans] == [(1.0, 2.5)]


def test_zero_length_span_is_discarded() -> None:
    stderr = "silence_start: 3.0\nsilence_end: 3.0 | silence_duration: 0.0\n"
    assert parse_silencedetect(stderr, duration=None) == []


def test_empty_output_means_no_silence() -> None:
    assert parse_silencedetect("", duration=10.0) == []


def test_scientific_notation_start_parses() -> None:
    """silencedetect prints very small numbers in exponent form."""
    stderr = "silence_start: 1.5e-05\nsilence_end: 2.0\n"
    spans = parse_silencedetect(stderr, duration=None)
    assert spans[0].start == 0.0


def test_command_uses_an_argument_list_with_the_expected_filter() -> None:
    from pathlib import Path

    cmd = silencedetect_command(Path("/tmp/a b.mp4"), threshold_db=-35.0, min_duration=0.75)
    assert isinstance(cmd, list)
    assert "silencedetect=n=-35.0dB:d=0.75" in cmd
    # An un-escaped path proves nothing goes through a shell.
    assert "/tmp/a b.mp4" in cmd


@requires_ffmpeg
def test_detect_silence_finds_the_synthesised_gap(av_file) -> None:
    """The fixture is silent from 2.0s to 4.0s by construction."""
    doc = detect_silence(av_file, duration=6.0)
    assert isinstance(doc, SilenceDoc)
    assert doc.threshold_db == DEFAULT_THRESHOLD_DB
    assert doc.min_duration == DEFAULT_MIN_DURATION
    assert len(doc.spans) == 1
    span = doc.spans[0]
    assert span.start == pytest.approx(2.0, abs=0.15)
    assert span.end == pytest.approx(4.0, abs=0.15)


@requires_ffmpeg
def test_silence_doc_round_trips_through_json(av_file) -> None:
    doc = detect_silence(av_file, duration=6.0)
    assert SilenceDoc.model_validate(doc.model_dump(mode="json")) == doc
