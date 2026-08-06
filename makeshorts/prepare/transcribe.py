"""faster-whisper -> words.json.

`words.json` is the source of truth for every timestamp in the system: the
linter checks the AI's start/end against it, `--fix` snaps to it, and the
caption backend groups it into cues. Nothing downstream ever re-reads the audio.

The interesting work here is not the model call -- it is
`mark_sentence_boundaries`, which turns a flat list of timed tokens into
sentences. Whisper does not tell you where a sentence ends; it only puts a
period on a word. The `start_on_sentence_start` / `end_on_sentence_end` gates
are enforced entirely from those two booleans, so getting them wrong makes the
gate either useless or impossible to satisfy. It is a pure function over
`Word` objects for exactly that reason -- it is tested without the model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from makeshorts.artifacts import Word, WordsDoc

# distil-large-v3 is ~6x faster than large-v3 at close to the same WER on
# clean speech, which is what a webinar is. CTranslate2 has no Metal backend,
# so Apple Silicon runs this on CPU regardless of what device is requested.
DEFAULT_MODEL = "distil-large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 5

# Punctuation that can sit *after* a terminator: `she said "no."` and
# `(see below!)` both end sentences.
_CLOSERS = "\"')]}»”’›"
_OPENERS = "\"'([{«“‘¿¡"

_TERMINATORS = ".?!"

# Abbreviations whose trailing period is not a sentence end. Kept deliberately
# short: this list only needs the ones that actually occur mid-sentence in
# spoken business English. Anything missed is caught by the lowercase-follower
# rule below.
_ABBREVIATIONS = frozenset(
    """
    mr. mrs. ms. dr. prof. sr. jr. st. rev. hon. gen. sgt. capt. lt.
    vs. etc. approx. est. dept. fig. no. nos. vol. ch. pp. ed. eds.
    inc. ltd. co. corp. llc. plc. univ.
    jan. feb. mar. apr. jun. jul. aug. sep. sept. oct. nov. dec.
    mon. tue. tues. wed. thu. thur. thurs. fri. sat. sun.
    """.split()
)

# `e.g.` `i.e.` `U.S.` `a.m.` -- any run of single letters each followed by a
# dot. Also matches a bare initial like `J.`, which is why `{1,}` not `{2,}`.
#
# The leading `\.?` is not hypothetical: whisper's tokeniser splits `U.S.` into
# `U` + `.S.`, so the fragment that carries the trailing period also carries a
# leading one. Observed on real output, not imagined.
_DOTTED_ACRONYM = re.compile(r"^\.?(?:[a-z]\.){1,}$")

# `...` and `…` are a trail-off, not a full stop. Ending a clip on one produces
# a clip that sounds unfinished, so they are deliberately not sentence ends.
_ELLIPSIS = re.compile(r"(?:\.{2,}|…)$")


def _core(token: str) -> str:
    """The token with surrounding quotes and brackets removed, so that
    `."` and `!)` are seen as terminators."""
    return token.strip().rstrip(_CLOSERS).strip()


def _lead_stripped(token: str) -> str:
    return token.lstrip(_OPENERS)


def is_sentence_end(token: str, next_token: str | None = None) -> bool:
    """Does `token` end a sentence?

    `next_token` is consulted only to resolve genuinely ambiguous periods; the
    decision is otherwise local. Rules, in order:

    1. Nothing but punctuation, or a trailing ellipsis -> no.
    2. `?` or `!` (possibly behind a closing quote) -> yes, unconditionally.
    3. A trailing `.` is a sentence end *unless* the token is a known
       abbreviation, a dotted acronym or initial (`e.g.` `U.S.` `J.`), a
       number whose decimal point got split across tokens (`3.` + `5`), or the
       next token starts lowercase -- speech-to-text reliably capitalises the
       start of a sentence, so a lowercase follower means the period belonged
       to an abbreviation this module has not listed.
    """
    core = _core(token)
    if not core:
        return False
    if _ELLIPSIS.search(core):
        return False

    last = core[-1]
    if last not in _TERMINATORS:
        return False
    if last in "?!":
        return True

    word = _lead_stripped(core).lower()
    if word in _ABBREVIATIONS:
        return False
    if _DOTTED_ACRONYM.match(word):
        return False

    following = _core(next_token or "").lstrip(_OPENERS)

    # `3.` followed by `5` is one decimal number that the tokeniser split.
    if len(core) >= 2 and core[-2].isdigit() and following[:1].isdigit():
        return False

    if following[:1].isalpha() and following[:1].islower():
        return False

    return True


def mark_sentence_boundaries(words: Sequence[Word]) -> list[Word]:
    """Return copies of `words` with `sentence_start`/`sentence_end` set.

    The final word always closes a sentence: a transcript that ends without
    punctuation would otherwise leave the last sentence open, and the
    `end_on_sentence_end` gate would reject every clip that runs to the end of
    the recording.
    """
    marked = [w.model_copy(deep=True) for w in words]
    if not marked:
        return marked

    starts_sentence = True
    for i, word in enumerate(marked):
        next_text = marked[i + 1].text if i + 1 < len(marked) else None
        word.sentence_start = starts_sentence
        word.sentence_end = is_sentence_end(word.text, next_text)
        starts_sentence = word.sentence_end

    marked[-1].sentence_end = True
    return marked


def words_from_segments(segments: Iterable[object]) -> list[Word]:
    """faster-whisper segments -> `Word`s, unflagged.

    Kept separate from `transcribe` so the shape conversion can be tested with
    plain stand-in objects instead of a downloaded model.

    Whisper emits each word with a leading space and can emit zero-length or
    out-of-order words around silence; those are repaired here rather than
    left for the linter to trip over.
    """
    words: list[Word] = []
    previous_end = 0.0
    for segment in segments:
        for raw in getattr(segment, "words", None) or []:
            text = (getattr(raw, "word", None) or "").strip()
            if not text:
                continue
            start = float(getattr(raw, "start", 0.0) or 0.0)
            end = float(getattr(raw, "end", 0.0) or 0.0)
            start = max(start, 0.0)
            # Monotonic by construction -- snap.py and cues.py both assume it.
            start = max(start, previous_end)
            if end <= start:
                end = start + 0.01
            probability = getattr(raw, "probability", None)
            words.append(
                Word(
                    text=text,
                    start=round(start, 3),
                    end=round(end, 3),
                    probability=round(float(probability), 4)
                    if probability is not None
                    else None,
                )
            )
            previous_end = end
    return words


def transcribe(
    path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    beam_size: int = DEFAULT_BEAM_SIZE,
    vad_filter: bool = True,
    download_root: str | None = None,
    progress: object | None = None,
) -> WordsDoc:
    """Transcribe `path` with word-level timestamps.

    `faster_whisper` is imported here rather than at module scope: it drags in
    CTranslate2 and (on first use) downloads a model, and the rest of this
    package -- including the linter and the renderer -- must work on a machine
    where it was never installed.

    `progress` may be any callable taking `(word_count, seconds_done)`; it
    exists because a two-hour webinar takes long enough that a silent process
    looks hung.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"no such file: {src}")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "faster-whisper is not installed. `uv pip install faster-whisper`, "
            "or run the other prepare stages with --skip transcribe."
        ) from exc

    whisper = WhisperModel(
        model,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )
    segments, info = whisper.transcribe(
        str(src),
        word_timestamps=True,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    # `segments` is a generator -- decoding happens as it is consumed, which is
    # what makes progress reporting possible at all.
    collected = []
    for segment in segments:
        collected.append(segment)
        if callable(progress):
            progress(len(collected), float(getattr(segment, "end", 0.0) or 0.0))

    words = mark_sentence_boundaries(words_from_segments(collected))
    return WordsDoc(
        model=model,
        language=getattr(info, "language", None) or language or "unknown",
        words=words,
    )


__all__ = [
    "DEFAULT_BEAM_SIZE",
    "DEFAULT_COMPUTE_TYPE",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL",
    "is_sentence_end",
    "mark_sentence_boundaries",
    "transcribe",
    "words_from_segments",
]
