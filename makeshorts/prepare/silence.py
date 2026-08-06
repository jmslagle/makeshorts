"""ffmpeg silencedetect -> silence.json.

Dead air is what tells the linter that a candidate clip has a hole in it, and
what tells `--fix` where a cut can land without chopping a word in half. The
filter reports to stderr as it decodes; this module runs it and parses that.

The parser is deliberately separate from the subprocess call: silencedetect's
output has awkward edge cases (see `parse_silencedetect`) and they are far
easier to pin down against captured stderr than against a live encode.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from makeshorts.artifacts import SilenceDoc, SilenceSpan
from makeshorts.prepare.probe import find_tool

# Quiet enough to be a pause, not so quiet that room tone reads as speech.
# Webinar audio is usually normalized and compressed, so -30 is generous.
DEFAULT_THRESHOLD_DB = -30.0

# Shorter than this is the gap between words, not a place you can cut.
DEFAULT_MIN_DURATION = 0.5

# `silence_start: 12.34` -- may be negative when the file opens in silence, as
# the filter reports the start relative to a lookback window.
_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

# `silence_end: 15.0 | silence_duration: 2.66` -- duration is present in every
# ffmpeg we care about but is treated as optional, since it is redundant.
_END_RE = re.compile(
    r"silence_end:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s*\|\s*silence_duration:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?"
)


class SilenceError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    return find_tool("ffmpeg")


def parse_silencedetect(
    stderr: str,
    *,
    duration: float | None = None,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> list[SilenceSpan]:
    """Parse silencedetect's stderr into spans.

    The edge cases this exists to handle:

    * **Unterminated trailing silence.** If a file ends while still silent,
      some ffmpeg builds never print the matching `silence_end` -- the filter is
      flushed at EOF without reporting. The span is real and is usually the
      most interesting one in the file (it is the outro). It is closed at
      `duration`; if the duration is unknown the span is dropped, because a
      span with an invented end is worse than a missing one.
    * **Negative start.** A file that opens in silence can report a small
      negative start. Clamped to 0.
    * **An end with no start**, from interleaved or truncated logs: ignored.
    * **A second start before the first has ended**: the later start wins, on
      the theory that the log line closest to the end is the accurate one.
    * Lines are matched anywhere in the text, so the surrounding ffmpeg banner,
      progress lines, and the `[Parsed_silencedetect_0 @ 0x...]` prefix are all
      irrelevant.
    """
    spans: list[SilenceSpan] = []
    open_start: float | None = None

    for line in stderr.splitlines():
        # Order matters: "silence_end" lines never carry a start, but checking
        # end first keeps a malformed line from being read as both.
        end_match = _END_RE.search(line)
        if end_match is not None:
            if open_start is None:
                continue
            end = float(end_match.group(1))
            span = _make_span(open_start, end)
            if span is not None:
                spans.append(span)
            open_start = None
            continue

        start_match = _START_RE.search(line)
        if start_match is not None:
            open_start = max(0.0, float(start_match.group(1)))

    if open_start is not None and duration is not None:
        span = _make_span(open_start, duration)
        # An unterminated span shorter than the threshold is an artifact of a
        # duration estimate that disagrees with the decoder, not a real pause.
        if span is not None and span.duration + 1e-6 >= min_duration:
            spans.append(span)

    return spans


def _make_span(start: float, end: float) -> SilenceSpan | None:
    start = max(0.0, start)
    if end <= start:
        return None
    return SilenceSpan(start=round(start, 3), end=round(end, 3))


def silencedetect_command(
    src: Path,
    *,
    threshold_db: float,
    min_duration: float,
) -> list[str]:
    """The argument list, exposed so tests and receipts can see it."""
    return [
        ffmpeg_path(),
        "-nostdin",
        "-hide_banner",
        "-nostats",
        "-i",
        str(src),
        "-map",
        "0:a:0",
        "-af",
        f"silencedetect=n={threshold_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]


def detect_silence(
    path: str | Path,
    *,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_duration: float = DEFAULT_MIN_DURATION,
    duration: float | None = None,
) -> SilenceDoc:
    """Run silencedetect over `path`.

    `duration` should be `media.json`'s -- it is what closes a trailing silence
    that ffmpeg never terminates.
    """
    src = Path(path)
    if not src.exists():
        raise SilenceError(f"no such file: {src}")

    cmd = silencedetect_command(src, threshold_db=threshold_db, min_duration=min_duration)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SilenceError(f"ffmpeg silencedetect failed on {src}:\n{proc.stderr.strip()}")

    spans = parse_silencedetect(proc.stderr, duration=duration, min_duration=min_duration)
    return SilenceDoc(threshold_db=threshold_db, min_duration=min_duration, spans=spans)


def silence_doc_for_no_audio(
    *,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> SilenceDoc:
    """A file with no audio stream has no detectable silence -- not one long
    silent span. Returning empty keeps the linter's `max_internal_silence`
    check from failing every clip on a silent source."""
    return SilenceDoc(threshold_db=threshold_db, min_duration=min_duration, spans=[])


__all__ = [
    "DEFAULT_MIN_DURATION",
    "DEFAULT_THRESHOLD_DB",
    "SilenceError",
    "detect_silence",
    "ffmpeg_path",
    "parse_silencedetect",
    "silence_doc_for_no_audio",
    "silencedetect_command",
]
