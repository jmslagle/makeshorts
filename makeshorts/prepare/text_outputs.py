"""words.json -> transcript.txt and raw.srt.

Both are derived views. Neither is read back by the pipeline -- `words.json` is
the machine-readable truth -- so these exist purely to be read by a human or
pasted into a model's context.

`transcript.txt` is timestamped on every line on purpose: the editorial step's
whole job is to name start and end times, and a transcript you have to count
paragraphs through is useless for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from makeshorts.artifacts import Word, WordsDoc

# Wide enough to read, narrow enough to survive being pasted into a chat.
DEFAULT_WIDTH = 88

# A break longer than this is a topic change, not a breath. Rendered as a blank
# line so the transcript has visible structure.
PARAGRAPH_GAP = 2.0

# SRT cue shaping. Two lines of ~42 characters is the readable maximum for
# burned-in captions and the conventional maximum for broadcast subtitles.
SRT_MAX_CHARS = 84
SRT_MAX_LINES = 2
SRT_MAX_DURATION = 6.0
SRT_MAX_GAP = 0.8
SRT_MIN_DURATION = 0.6


def format_clock(seconds: float) -> str:
    """`[HH:MM:SS]`-style clock, without the brackets."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """`HH:MM:SS,mmm`. Milliseconds are truncated toward zero after rounding to
    the nearest millisecond, so 1.0005 -> `00:00:01,001` and never `,1000`."""
    ms_total = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# --------------------------------------------------------------------------
# transcript.txt
# --------------------------------------------------------------------------


def render_transcript(doc: WordsDoc, *, width: int = DEFAULT_WIDTH) -> str:
    """Timestamped, wrapped, paragraph-broken plain text.

    Lines break at a sentence end when one is available and the line is already
    half full, and at the wrap width otherwise. Sentence-aligned lines matter
    because the times a reader copies out of this file are the times they will
    put in clips.json, and the gates require sentence boundaries.
    """
    words = doc.words
    if not words:
        return ""

    prefix_len = len(format_clock(0)) + 3  # "[HH:MM:SS] "
    text_width = max(20, width - prefix_len)
    soft_break = text_width // 2

    lines: list[str] = []
    current: list[str] = []
    current_start = words[0].start
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            lines.append(f"[{format_clock(current_start)}] {' '.join(current)}")
            current = []
            current_len = 0

    for i, word in enumerate(words):
        token = word.text
        addition = len(token) + (1 if current else 0)
        if current and current_len + addition > text_width:
            flush()
        if not current:
            current_start = word.start
            addition = len(token)
        current.append(token)
        current_len += addition

        next_word = words[i + 1] if i + 1 < len(words) else None
        gap = (next_word.start - word.end) if next_word else 0.0

        if next_word is None:
            flush()
        elif gap >= PARAGRAPH_GAP:
            flush()
            lines.append("")
        elif word.sentence_end and current_len >= soft_break:
            flush()

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def write_transcript(doc: WordsDoc, path: str | Path, *, width: int = DEFAULT_WIDTH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_transcript(doc, width=width), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# raw.srt
# --------------------------------------------------------------------------


def group_cues(
    words: Sequence[Word],
    *,
    max_chars: int = SRT_MAX_CHARS,
    max_duration: float = SRT_MAX_DURATION,
    max_gap: float = SRT_MAX_GAP,
) -> list[list[Word]]:
    """Group words into subtitle cues.

    A cue breaks at a sentence end, at a pause, at the character budget, or at
    the duration budget -- whichever comes first. This is the naive grouping
    that `raw.srt` wants; the caption backend has its own, style-aware one.
    """
    cues: list[list[Word]] = []
    current: list[Word] = []
    length = 0

    for word in words:
        addition = len(word.text) + (1 if current else 0)
        too_long = current and length + addition > max_chars
        too_slow = current and (word.end - current[0].start) > max_duration
        after_pause = current and (word.start - current[-1].end) > max_gap
        if too_long or too_slow or after_pause:
            cues.append(current)
            current = []
            length = 0
            addition = len(word.text)

        current.append(word)
        length += addition

        if word.sentence_end:
            cues.append(current)
            current = []
            length = 0

    if current:
        cues.append(current)
    return cues


def wrap_cue_text(words: Sequence[Word], *, max_lines: int = SRT_MAX_LINES) -> str:
    """Balance a cue's words across at most `max_lines` lines.

    Balanced rather than greedy: a two-line cue reading 40/8 characters looks
    broken, and the fix is free.
    """
    tokens = [w.text for w in words]
    if not tokens:
        return ""
    if max_lines <= 1 or len(tokens) == 1:
        return " ".join(tokens)

    total = len(" ".join(tokens))
    target = total / max_lines

    lines: list[str] = []
    remaining = list(tokens)
    for line_index in range(max_lines - 1):
        if not remaining:
            break
        budget = target * (line_index + 1)
        taken: list[str] = [remaining.pop(0)]
        # Keep taking while the line is still short of its share and doing so
        # gets us closer to the target than stopping would.
        while remaining:
            current_len = len(" ".join(lines + [" ".join(taken)]))
            with_next = current_len + 1 + len(remaining[0])
            if abs(with_next - budget) < abs(current_len - budget):
                taken.append(remaining.pop(0))
            else:
                break
        lines.append(" ".join(taken))
    if remaining:
        lines.append(" ".join(remaining))
    return "\n".join(line for line in lines if line)


def render_srt(
    doc: WordsDoc,
    *,
    max_chars: int = SRT_MAX_CHARS,
    max_duration: float = SRT_MAX_DURATION,
    max_gap: float = SRT_MAX_GAP,
) -> str:
    """A valid SRT of the whole transcript.

    Cue times come straight from the words. A cue is extended to
    `SRT_MIN_DURATION` when the words are shorter than that, but never past the
    next cue's start -- overlapping cues are the one thing players actually
    render wrongly.
    """
    cues = group_cues(doc.words, max_chars=max_chars, max_duration=max_duration, max_gap=max_gap)
    blocks: list[str] = []

    for index, cue in enumerate(cues, start=1):
        start = cue[0].start
        end = max(cue[-1].end, start + SRT_MIN_DURATION)
        next_cue = cues[index] if index < len(cues) else None
        if next_cue is not None:
            end = min(end, max(next_cue[0].start, cue[-1].end))
        if end <= start:
            end = start + 0.001
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n"
            f"{wrap_cue_text(cue)}\n"
        )

    return "\n".join(blocks)


def write_srt(doc: WordsDoc, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_srt(doc), encoding="utf-8")
    return out


__all__ = [
    "DEFAULT_WIDTH",
    "PARAGRAPH_GAP",
    "SRT_MAX_CHARS",
    "SRT_MAX_DURATION",
    "SRT_MAX_GAP",
    "SRT_MAX_LINES",
    "SRT_MIN_DURATION",
    "format_clock",
    "format_srt_timestamp",
    "group_cues",
    "render_srt",
    "render_transcript",
    "wrap_cue_text",
    "write_srt",
    "write_transcript",
]
