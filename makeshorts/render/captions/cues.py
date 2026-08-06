"""words.json + a clip's time range -> caption cues.

Backend-agnostic on purpose. Grouping words into readable chunks is a
typography and pacing problem, not a rendering one, and both caption backends
want exactly the same answer. Getting it right once here is why the libass
path and the Pillow path can look like the same product.

Three things this must preserve:

* **Word-level timings.** Karaoke highlighting is per word, so a cue that
  collapsed to a single string with one start and one end would have thrown
  away the only data that makes the effect possible.
* **Clip-relative times.** Everything downstream measures from the start of
  the clip, because that is where the rendered video starts. Absolute source
  times exist only in `words.json` and `clips.json`.
* **Sentence shape.** A cue that breaks mid-clause reads badly even when every
  word is on screen at the right moment, so breaks are placed at the strongest
  boundary available within the size budget.

Line wrapping here is by *character count*, which is an approximation for a
proportional font. The Pillow backend measures for real and shrinks the font
if a line overruns; this pass only has to be close.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import Field

from makeshorts.artifacts import Strict, Word, WordsDoc

__all__ = [
    "CueWord",
    "Cue",
    "CueOptions",
    "build_cues",
    "cues_for_clip",
]

# Punctuation that ends a sentence. `words.json` carries `sentence_end` flags
# computed at transcription time, but a hand-edited or third-party words.json
# may not, so the text is checked as well.
_SENTENCE_END = re.compile(r"[.!?…]['\"”’)\]]*$")
# Weaker breaks: still better than splitting mid-phrase.
_CLAUSE_END = re.compile(r"[,;:—–-]['\"”’)\]]*$")


class CueWord(Strict):
    """One word inside a cue, timed relative to the start of the clip."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class Cue(Strict):
    """One on-screen caption: a few words, wrapped into lines.

    `lines` holds indices into `words` rather than strings, so a backend can
    lay out the line *and* still know which word is active — the two things
    karaoke needs at once.
    """

    start: float
    end: float
    words: list[CueWord]
    lines: list[list[int]] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def line_text(self, index: int) -> str:
        return " ".join(self.words[i].text for i in self.lines[index])

    def line_texts(self) -> list[str]:
        return [self.line_text(i) for i in range(len(self.lines))]

    def line_of(self, word_index: int) -> int:
        for li, line in enumerate(self.lines):
            if word_index in line:
                return li
        raise IndexError(f"word {word_index} is in no line of this cue")

    def karaoke_states(self) -> list[tuple[int, float, float]]:
        """(active_word_index, start, end) — one entry per visual state.

        The first state is stretched back to the cue start and the last
        forward to the cue end, so the cue is never on screen with nothing
        highlighted. Gaps between words are absorbed by the *preceding* word,
        which matches how a pause reads: the last word spoken stays lit.
        """
        states: list[tuple[int, float, float]] = []
        n = len(self.words)
        for i, w in enumerate(self.words):
            s = self.start if i == 0 else w.start
            e = self.words[i + 1].start if i + 1 < n else self.end
            e = max(e, s)  # defend against out-of-order input
            states.append((i, s, e))
        return [st for st in states if st[2] > st[1]] or (
            [(0, self.start, self.end)] if n else []
        )


class CueOptions(Strict):
    """Pacing knobs. Sizing comes from the caption style; these are timing."""

    max_chars_per_line: int = Field(default=26, ge=6, le=120)
    max_lines: int = Field(default=2, ge=1, le=6)
    # A pause longer than this starts a new cue regardless of size. Speech
    # resuming after most of a second is a new thought.
    max_word_gap: float = Field(default=0.75, gt=0.0)
    # No cue may sit on screen longer than this even if it is short. Beyond
    # about 6s a static caption stops tracking the audio.
    max_cue_duration: float = Field(default=6.0, gt=0.0)
    # A cue this short is merged forward rather than flashed.
    min_cue_duration: float = Field(default=0.35, ge=0.0)
    # Hold a cue on screen through a short silence rather than blinking off.
    gap_fill: float = Field(default=0.30, ge=0.0)

    @property
    def max_chars(self) -> int:
        return self.max_chars_per_line * self.max_lines


# --------------------------------------------------------------------------
# Wrapping
# --------------------------------------------------------------------------


def _wrap(texts: Sequence[str], max_chars: int, max_lines: int) -> list[list[int]] | None:
    """Greedy wrap of word indices into lines, or None if it does not fit.

    A single word longer than `max_chars` gets a line to itself rather than
    failing -- a URL or a compound German noun should still be captioned, just
    with an overrun the renderer will shrink to fit.
    """
    lines: list[list[int]] = []
    current: list[int] = []
    width = 0
    for i, t in enumerate(texts):
        add = len(t) if not current else len(t) + 1
        if current and width + add > max_chars:
            lines.append(current)
            current = [i]
            width = len(t)
        else:
            current.append(i)
            width += add
        if len(lines) > max_lines:
            return None
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        return None
    return lines


def _fits(texts: Sequence[str], opts: CueOptions) -> bool:
    return _wrap(texts, opts.max_chars_per_line, opts.max_lines) is not None


# --------------------------------------------------------------------------
# Boundary strength
# --------------------------------------------------------------------------


def _is_sentence_end(w: Word) -> bool:
    return bool(w.sentence_end) or bool(_SENTENCE_END.search(w.text.strip()))


def _is_clause_end(w: Word) -> bool:
    return bool(_CLAUSE_END.search(w.text.strip()))


def _break_score(words: Sequence[Word], i: int) -> int:
    """How good a break *after* word i is. Higher is better.

    Sentence ends beat clause ends beat nothing. This is what stops a cue from
    ending on "the" when it could have ended on a full stop two words earlier.
    """
    w = words[i]
    if _is_sentence_end(w):
        return 3
    if _is_clause_end(w):
        return 2
    return 1


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def _select_words(
    words: Iterable[Word], clip_start: float, clip_end: float
) -> list[CueWord]:
    """Words overlapping the clip, retimed to clip-relative and clamped.

    Overlap rather than containment: a word straddling the clip boundary is
    audible in the clip, so dropping it would caption less than was said.
    """
    duration = clip_end - clip_start
    out: list[CueWord] = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        if not w.text.strip():
            continue
        s = min(max(w.start - clip_start, 0.0), duration)
        e = min(max(w.end - clip_start, 0.0), duration)
        if e <= s:
            e = min(s + 0.04, duration)  # a clamped-to-nothing word still needs width
        if e <= s:
            continue
        out.append(CueWord(text=w.text.strip(), start=s, end=e))
    return out


def _group(words: Sequence[Word], opts: CueOptions) -> list[list[int]]:
    """Indices of `words` grouped into cues.

    Greedy: extend the current cue while it still fits and still tracks the
    audio, then break at the strongest boundary seen inside the group. Falling
    back to the size limit when there is no punctuation at all is the common
    case for conversational speech, and it is fine -- the gap and duration
    rules keep those cues in sync even without sentence structure.
    """
    groups: list[list[int]] = []
    current: list[int] = []

    def flush(upto: int | None = None) -> None:
        nonlocal current
        if not current:
            return
        if upto is None or upto >= len(current) - 1:
            groups.append(current)
            current = []
        else:
            groups.append(current[: upto + 1])
            current = current[upto + 1 :]

    for i, w in enumerate(words):
        if current:
            prev = words[current[-1]]
            gap = w.start - prev.end
            span = w.end - words[current[0]].start
            candidate = [words[j].text for j in current] + [w.text]
            too_wide = not _fits(candidate, opts)
            too_long = span > opts.max_cue_duration
            if gap > opts.max_word_gap or too_long:
                flush()
            elif too_wide:
                # Prefer the strongest boundary inside what we already have,
                # but only if it does not throw away most of the cue.
                best = None
                best_score = 1
                floor = max(0, len(current) // 2 - 1)
                for k in range(len(current) - 1, floor - 1, -1):
                    score = _break_score(words, current[k])
                    if score > best_score:
                        best_score = score
                        best = k
                        if score == 3:
                            break
                flush(best)
                # Breaking at a boundary leaves a remainder, and the remainder
                # plus this word may still not fit. Without this the cue would
                # silently overrun max_lines.
                if current and not _fits(
                    [words[j].text for j in current] + [w.text], opts
                ):
                    flush()
        current.append(i)
        # A sentence ended and the next word would start a new one: break here
        # rather than letting the size limit choose a worse spot later.
        if _is_sentence_end(words[i]) and len(current) >= 2:
            flush()

    flush()
    return [g for g in groups if g]


def _merge_runts(groups: list[list[int]], words: Sequence[Word], opts: CueOptions) -> list[list[int]]:
    """Fold away cues too short to read, when the merge still fits."""
    if opts.min_cue_duration <= 0:
        return groups
    out: list[list[int]] = []
    for g in groups:
        dur = words[g[-1]].end - words[g[0]].start
        if out and dur < opts.min_cue_duration:
            merged = out[-1] + g
            if _fits([words[j].text for j in merged], opts):
                out[-1] = merged
                continue
        out.append(g)
    # A leading runt has nothing before it to merge into; try the one after.
    if len(out) >= 2:
        first_dur = words[out[0][-1]].end - words[out[0][0]].start
        if first_dur < opts.min_cue_duration:
            merged = out[0] + out[1]
            if _fits([words[j].text for j in merged], opts):
                out = [merged] + out[2:]
    return out


def build_cues(
    words: WordsDoc | Sequence[Word],
    clip_start: float,
    clip_end: float,
    options: CueOptions | None = None,
) -> list[Cue]:
    """Caption cues for `[clip_start, clip_end)`, timed from the clip start.

    An empty or reversed range, or a range containing no speech, yields `[]` --
    a clip of pure silence is legitimate and should render without captions
    rather than raising.
    """
    opts = options or CueOptions()
    seq = words.words if isinstance(words, WordsDoc) else list(words)
    if clip_end <= clip_start:
        return []

    picked = _select_words(seq, clip_start, clip_end)
    if not picked:
        return []
    picked.sort(key=lambda w: (w.start, w.end))

    # Grouping works on Word (it needs the sentence flags), so pair the
    # retimed CueWords back up with the flags from the source words.
    flagged = [
        Word(
            text=cw.text,
            start=cw.start,
            end=cw.end,
            sentence_start=False,
            sentence_end=_SENTENCE_END.search(cw.text) is not None,
        )
        for cw in picked
    ]
    # Carry through explicit flags from the source where they exist; a
    # transcript that marks sentence ends without punctuation still gets them.
    by_text_time = {(w.text.strip(), round(w.start, 3)): w for w in seq}
    for f, cw in zip(flagged, picked, strict=True):
        src = by_text_time.get((cw.text, round(cw.start + clip_start, 3)))
        if src is not None and src.sentence_end:
            f.sentence_end = True

    groups = _merge_runts(_group(flagged, opts), flagged, opts)

    cues: list[Cue] = []
    for g in groups:
        cue_words = [picked[i] for i in g]
        texts = [w.text for w in cue_words]
        lines = _wrap(texts, opts.max_chars_per_line, opts.max_lines)
        if lines is None:
            # Reachable only when a single word is wider than a whole line.
            # Widen the line rather than dropping the word; the Pillow backend
            # shrinks the font until the result actually fits the frame.
            widest = max(opts.max_chars_per_line, max(len(t) for t in texts))
            lines = _wrap(texts, widest, opts.max_lines) or _wrap(
                texts, widest, len(texts)
            )
        assert lines is not None, "wrapping one word per line always succeeds"
        cues.append(
            Cue(
                start=cue_words[0].start,
                end=cue_words[-1].end,
                words=cue_words,
                lines=lines,
            )
        )

    _fill_gaps(cues, opts, clip_end - clip_start)
    return cues


def _fill_gaps(cues: list[Cue], opts: CueOptions, duration: float) -> None:
    """Extend each cue over a short following silence, in place.

    Without this a natural pause between sentences makes the caption blink
    off and back on, which reads as a glitch rather than as a pause.
    """
    if opts.gap_fill <= 0:
        return
    for i, cue in enumerate(cues):
        limit = cues[i + 1].start if i + 1 < len(cues) else duration
        cue.end = min(max(cue.end, min(cue.end + opts.gap_fill, limit)), duration)


def cues_for_clip(
    words: WordsDoc | Sequence[Word],
    clip_start: float,
    clip_end: float,
    style: "object | None" = None,
) -> list[Cue]:
    """`build_cues` with the grouping limits taken from a CaptionStyle.

    Typed loosely so `cues.py` stays free of any import from `style.py`; the
    only thing it reads are the two grouping fields.
    """
    opts = CueOptions()
    if style is not None:
        opts = CueOptions(
            max_chars_per_line=getattr(style, "max_chars_per_line", opts.max_chars_per_line),
            max_lines=getattr(style, "max_lines", opts.max_lines),
        )
    return build_cues(words, clip_start, clip_end, opts)
