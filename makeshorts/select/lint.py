"""`ms lint` — the gate between the editorial step and the render step.

Nothing renders until this passes. Everything in here is a mechanical check of
an AI-authored (or hand-edited) `clips.json` against artifacts that were
produced deterministically: `words.json`, `silence.json`, and
`config/criteria.yaml`. The linter never adjudicates a *score* — that is
editorial, and disagreeing with one means editing the file by hand. It
adjudicates everything that can be checked without judgment, and it does so
loudly.

The seven rule groups, in the order they run:

1. ``time.*`` / ``text.*`` — no hallucinated time. The highest-value check,
   because it is the failure mode that actually happens: a model that has read
   a transcript will happily write a plausible timestamp it never verified, and
   a plausible quote to go with it. Both are caught against words.json.
2. ``--fix`` — see :func:`fix_text`. Snaps to real boundaries and pads.
3. ``gate.*`` — the mechanical gates from criteria.yaml, against words.json and
   silence.json.
4. ``rubric.*`` — every criterion scored, the arithmetic recomputed rather than
   trusted, vetoes and thresholds applied.
5. ``visual.*`` — ``visual_dependency: true`` requires a slide region on screen.
6. ``ref.*`` — referential integrity of ids, regions, and spans.
7. ``source.*`` — when a job has several inputs, that they are what a
   multi-source job assumes them to be: frame-aligned views of one recording,
   each long enough to cover the clips cut from it.

Findings are structured, not a bool: every one carries a stable ``rule_id`` so
a caller can suppress, count, or test for one specific rule. ERROR blocks the
render; WARN does not.
"""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from makeshorts.artifacts import SilenceDoc, WordsDoc
from makeshorts.select import snap
from makeshorts.select.criteria import Criteria, Gates, load_criteria
from makeshorts.select.schema import CLIP_ID_PATTERN, Clip, ClipsDoc, layout_region_ids

__all__ = [
    "Severity",
    "Finding",
    "LintReport",
    "RULES",
    "FIXABLE_RULES",
    "FixChange",
    "FixResult",
    "lint_doc",
    "lint_job",
    "lint_paths",
    "fix_text",
    "fix_file",
    "normalize_text",
    "normalize_tokens",
]

# How far a clip edge may miss a word edge before we stop calling it jitter.
# Inside this window the number is wrong but recognisably derived from the
# transcript, and `--fix` will repair it -> WARN. Outside it, no word edge is
# anywhere near, which means the value was not read from words.json at all
# -> ERROR. This is the one number in the module not taken from criteria.yaml,
# because it is not a policy choice: it separates "off" from "invented".
HALLUCINATION_WINDOW = 0.5

# weighted_score is written to two decimals by convention, so anything beyond
# this is real arithmetic disagreement rather than rounding.
SCORE_TOLERANCE = 0.01

# How far two sources' durations may disagree before we stop believing they are
# frame-aligned exports of one recording. Two renders of the same meeting end
# within a frame or two of each other; what actually varies is container and
# codec bookkeeping -- a trailing audio packet, a final partial GOP, a muxer
# that rounds the last frame's duration. That is tens of milliseconds, so a
# whole second is roughly thirty frames of headroom above the noise, while
# still being far below any real difference: a genuinely different recording,
# or an export that starts at a different moment, is out by seconds or minutes.
#
# It is a heuristic and it is deliberately only a WARN, because being wrong in
# the strict direction would block a perfectly good render over an encoder
# quirk. Being wrong in the lenient direction produces a clip whose face and
# whose slides are from different moments -- which is why the warning is worth
# emitting loudly even though it cannot be certain.
SOURCE_DURATION_TOLERANCE = 1.0

_MAX_DIFF_HUNKS = 6
_MAX_QUOTED_TOKENS = 8
_MAX_VERBATIM_CHARS = 1200


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"


# Rules `--fix` can actually repair, so the CLI can offer it. `time.hallucinated`
# is deliberately absent: snapping an invented timestamp to the nearest word
# would produce a well-formed clip of some other part of the talk, which is a
# worse outcome than the error. A fabricated time needs a human.
#
# `gate.opens_on_filler` is absent for the same reason. Dropping the leading
# word would leave the clip opening mid-sentence, and choosing a different span
# is an editorial decision about what the clip is -- not something a snapping
# routine should make on its own.
FIXABLE_RULES = frozenset(
    {
        "time.off_boundary",
        "time.mid_word",
        "gate.start_on_sentence_start",
        "gate.end_on_sentence_end",
    }
)


@dataclass(frozen=True)
class Finding:
    """One thing wrong with the edit list.

    `clip_id` is None for document-level findings. `message` is written to be
    read by whoever has to fix the file, so it names the values involved rather
    than describing the rule.
    """

    rule_id: str
    clip_id: str | None
    severity: Severity
    message: str
    fixable: bool = False

    def __str__(self) -> str:
        where = self.clip_id or "-"
        return f"{self.severity.value.upper():5} {self.rule_id:<32} [{where}] {self.message}"


@dataclass
class LintReport:
    findings: list[Finding] = field(default_factory=list)
    # Populated by `lint_job(..., fix=True)`: what --fix actually moved.
    fixed: list[str] = field(default_factory=list)

    def add(self, rule_id: str, clip_id: str | None, severity: Severity, message: str) -> None:
        self.findings.append(
            Finding(rule_id, clip_id, severity, message, fixable=rule_id in FIXABLE_RULES)
        )

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARN]

    @property
    def ok(self) -> bool:
        """True when nothing blocks the render. Warnings do not block."""
        return not self.errors

    def rule_ids(self) -> list[str]:
        return [f.rule_id for f in self.findings]

    def for_rule(self, rule_id: str) -> list[Finding]:
        return [f for f in self.findings if f.rule_id == rule_id]

    def format(self) -> str:
        if not self.findings:
            return "lint: clean"
        lines = [str(f) for f in self.findings]
        lines.append(f"lint: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


# Every rule this module can emit, with what it means. Kept as data so `ms lint
# --rules` can print it and so the test suite can assert that no rule is
# emitted without being documented here.
RULES: dict[str, str] = {
    # 1. no hallucinated time
    "time.no_words": "words.json contains no words, so no timestamp can be verified",
    "time.hallucinated": "a clip edge sits near no word edge in words.json",
    "time.mid_word": "a clip edge falls inside a spoken word and would clip it",
    "time.off_boundary": "a clip edge misses the nearest word edge but is close enough to fix",
    "text.empty_span": "the clip's time range contains no words at all",
    "text.mismatch": "source_text does not match the words.json span it claims",
    # 3. gates
    "gate.duration_short": "clip is shorter than gates.duration.min",
    "gate.duration_long": "clip is longer than gates.duration.max",
    "gate.internal_silence": "a silence span inside the clip exceeds gates.max_internal_silence",
    "gate.min_separation": "two clip midpoints are closer than gates.min_separation",
    "gate.max_clips": "the edit list has more clips than gates.max_clips",
    "gate.start_on_sentence_start": "clip does not begin on a sentence start",
    "gate.end_on_sentence_end": "clip does not end on a sentence end",
    "gate.opens_on_filler": "clip opens on a filler or connective word",
    "ref.transition_on_first_span": "the first span names a transition, which cannot apply",
    "gate.no_sentence_flags": "words.json has no sentence flags, so those gates were skipped",
    "gate.silence_unavailable": "no silence.json supplied, so max_internal_silence was skipped",
    # 4. rubric
    "rubric.missing_score": "a rubric criterion is missing from why.scores",
    "rubric.unknown_score": "why.scores contains a key that is not in the rubric",
    "rubric.weighted_score_mismatch": "why.weighted_score disagrees with the recomputed value",
    "rubric.veto_failed": "a scoring veto was not met",
    "rubric.below_min_score": "weighted score is below scoring.min_weighted_score",
    "rubric.theme_overrepresented": "more clips share a why.theme than diversity.max_per_theme",
    "rubric.criteria_changed": "criteria.yaml has changed since this edit list was written",
    # 5. visual dependency
    "visual.no_slide_region": "visual_dependency is true but no layout shows a slide region",
    # 6. referential integrity
    "ref.unknown_region": "a layout references a region id that does not exist",
    "ref.unknown_speaker": "speaker names a region id that does not exist",
    "ref.bad_clip_id": "clip id does not match the required pattern",
    "ref.duplicate_clip_id": "two clips share an id",
    "ref.span_start_not_zero": "the first layout span is not at 0.0",
    "ref.spans_unordered": "layout spans are not strictly ordered by `at`",
    "ref.span_out_of_bounds": "a layout span starts outside the clip",
    "ref.overlapping_clips": "two clips overlap in the source",
    "ref.out_of_source": "clip range falls outside the primary source's duration",
    "ref.unknown_source": "a region names a source that is not declared in `sources`",
    # 7. multi-source coherence
    "source.clip_out_of_range": "a source the clip puts on screen is too short to cover it",
    "source.duration_mismatch": "two sources differ in duration enough to doubt they are aligned",
    "source.missing_file": "a declared source file is not on disk",
    # loading
    "schema.invalid": "clips.json does not match the schema",
}


# --------------------------------------------------------------------------
# source_text normalization
# --------------------------------------------------------------------------
#
# The judgment call of the module. What follows is tolerated, and why:
#
#   case          -- Whisper capitalises sentence starts on acoustic guesswork;
#                    case carries no timing information.
#   punctuation   -- likewise a guess. Holding a human to the model's comma
#                    placement would produce failures nobody can act on.
#   whitespace    -- line breaks in a hand-edited JSON string are formatting.
#   apostrophes   -- removed rather than spaced, so "don't" and "dont" agree
#                    instead of becoming "don t".
#   unicode form  -- NFKC, so curly quotes and ligatures do not create phantom
#                    differences.
#
# What is deliberately NOT tolerated:
#
#   filler words  -- "um", "uh", "you know". Dropping them is the signature of
#                    a quote reconstructed from memory rather than copied from
#                    words.json, which is precisely what this check exists to
#                    detect. Tidying the transcript is a real editorial wish,
#                    but it must not be paid for with the only defence against
#                    invented quotes.
#   word order,
#   substitutions,
#   omissions     -- any of these mean the clip does not say what the file
#                    claims it says.
#   numerals      -- "18" and "eighteen" are different words on screen and in
#                    the captions rendered from words.json.

_APOSTROPHES = "'’ʼ´`"
_APOSTROPHE_RE = re.compile(f"[{re.escape(_APOSTROPHES)}]")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Fold away everything a transcript may legitimately disagree about."""
    s = unicodedata.normalize("NFKC", s)
    s = _APOSTROPHE_RE.sub("", s)
    s = _NON_WORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip().casefold()


def normalize_tokens(s: str) -> list[str]:
    n = normalize_text(s)
    return n.split() if n else []


def _quote(tokens: Sequence[str]) -> str:
    if len(tokens) > _MAX_QUOTED_TOKENS:
        shown = " ".join(tokens[:_MAX_QUOTED_TOKENS])
        return f'"{shown} …" ({len(tokens)} words)'
    return '"' + " ".join(tokens) + '"'


def text_diff(expected: Sequence[str], actual: Sequence[str]) -> list[str]:
    """A word-level diff a human can act on.

    "expected" is words.json — the ground truth — and "actual" is what
    clips.json claims. Reported per differing run rather than per word, because
    a dropped clause should read as one problem, not as nine.
    """
    lines: list[str] = []
    matcher = difflib.SequenceMatcher(a=list(expected), b=list(actual), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if len(lines) >= _MAX_DIFF_HUNKS:
            lines.append("    … further differences suppressed")
            break
        if tag == "replace":
            lines.append(
                f"    at word {i1 + 1}: words.json has {_quote(expected[i1:i2])} "
                f"but clips.json has {_quote(actual[j1:j2])}"
            )
        elif tag == "delete":
            lines.append(
                f"    at word {i1 + 1}: missing from clips.json: {_quote(expected[i1:i2])}"
            )
        elif tag == "insert":
            lines.append(
                f"    at word {i1 + 1}: not in words.json: {_quote(actual[j1:j2])}"
            )
    return lines


def _truncate(s: str, limit: int = _MAX_VERBATIM_CHARS) -> str:
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# The linter
# --------------------------------------------------------------------------


def lint_doc(
    doc: ClipsDoc,
    *,
    words: WordsDoc,
    criteria: Criteria,
    silence: SilenceDoc | None = None,
    check_files: bool = False,
) -> LintReport:
    """Check an edit list against the artifacts that can prove it wrong.

    `check_files` decides whether the declared source paths are looked for on
    disk. It is off by default because a `ClipsDoc` is very often held without
    any filesystem to resolve against -- built in a test, generated in memory,
    or read from somewhere other than the directory its paths are relative to.
    Warning "file missing" in those cases would be a lie told confidently. The
    entry points that do know they are looking at a real job pass it through.
    """
    report = LintReport()
    gates = criteria.gates

    _check_document(doc, words, criteria, silence, report)
    _check_sources(doc, report, check_files=check_files)
    if words.words:
        for clip in doc.clips:
            _check_time_and_text(clip, words, gates, report)
        for clip in doc.clips:
            _check_gates(clip, words, silence, gates, report)
        _check_separation(doc, gates, report)
    for clip in doc.clips:
        _check_rubric(clip, criteria, report)
    _check_diversity(doc, criteria, report)
    for clip in doc.clips:
        _check_visual(doc, clip, gates, report)
    for clip in doc.clips:
        _check_referential(doc, clip, report)
    for clip in doc.clips:
        _check_clip_sources(doc, clip, report)
    _check_clip_relations(doc, report)
    return report


def _check_document(
    doc: ClipsDoc,
    words: WordsDoc,
    criteria: Criteria,
    silence: SilenceDoc | None,
    report: LintReport,
) -> None:
    if not words.words:
        report.add(
            "time.no_words",
            None,
            Severity.ERROR,
            "words.json contains no words — every timestamp in this edit list is unverifiable",
        )
    elif not snap.has_sentence_flags(words):
        report.add(
            "gate.no_sentence_flags",
            None,
            Severity.WARN,
            "words.json carries no sentence_start/sentence_end flags; the "
            "start_on_sentence_start and end_on_sentence_end gates were skipped",
        )
    if silence is None:
        report.add(
            "gate.silence_unavailable",
            None,
            Severity.WARN,
            "no silence.json supplied; gates.max_internal_silence was not checked",
        )
    if len(doc.clips) > criteria.gates.max_clips:
        report.add(
            "gate.max_clips",
            None,
            Severity.ERROR,
            f"{len(doc.clips)} clips, but gates.max_clips is {criteria.gates.max_clips}",
        )
    if criteria.sha256 and doc.criteria_ref.sha256 != criteria.sha256:
        report.add(
            "rubric.criteria_changed",
            None,
            Severity.WARN,
            f"criteria.yaml has changed since this edit list was written "
            f"(recorded {doc.criteria_ref.version!r}/{doc.criteria_ref.sha256[:12]}…, "
            f"current {criteria.version!r}/{criteria.sha256[:12]}…). The scores below were "
            "given against the old rubric.",
        )


# -- 1. no hallucinated time ------------------------------------------------


def _check_time_and_text(
    clip: Clip,
    words: WordsDoc,
    gates: Gates,
    report: LintReport,
) -> None:
    if gates.snap_to_word_boundaries:
        tol = gates.boundary_tolerance
        _check_edge(clip, clip.start, "start", words, tol, gates.pad_in, report)
        _check_edge(clip, clip.end, "end", words, tol, gates.pad_out, report)

    span = snap.words_in_span(words, clip.start, clip.end)
    if not span:
        report.add(
            "text.empty_span",
            clip.id,
            Severity.ERROR,
            f"no words in words.json fall between {clip.start:.2f} and {clip.end:.2f} — "
            "this range does not correspond to anything that was said",
        )
        return

    expected_raw = snap.span_text(span)
    expected = normalize_tokens(expected_raw)
    actual = normalize_tokens(clip.source_text)
    if expected == actual:
        return

    lines = [
        f"source_text does not match words.json over [{clip.start:.2f}, {clip.end:.2f}] "
        f"({len(span)} words). Comparison ignores case, punctuation and whitespace only.",
        "  differences (word-level):",
        *text_diff(expected, actual),
        "  words.json says, verbatim:",
        f"    {_truncate(expected_raw)}",
    ]
    report.add("text.mismatch", clip.id, Severity.ERROR, "\n".join(lines))


def _check_edge(
    clip: Clip,
    t: float,
    side: snap.Side,
    words: WordsDoc,
    tolerance: float,
    pad: float,
    report: LintReport,
) -> None:
    check = snap.check_boundary(words, t, side, tolerance=tolerance, pad=pad)
    if check.ok:
        return
    nearest = check.nearest
    if nearest is None:  # pragma: no cover - words.words is non-empty here
        return
    distance = check.distance or 0.0
    edge = "starts" if side == "start" else "ends"

    # Landing inside a real word proves the number came from the transcript,
    # so this is a bad cut rather than a fabrication — but it is still an
    # error, because it clips a syllable.
    if check.inside_word is not None:
        w = check.inside_word
        report.add(
            "time.mid_word",
            clip.id,
            Severity.ERROR,
            f"{side} {t:.3f} falls inside the word {w.text.strip()!r} "
            f"({w.start:.3f}–{w.end:.3f}); the clip {edge} mid-word. "
            f"Nearest legal edge is {nearest.time:.3f}. Run `ms lint --fix`.",
        )
        return

    if distance > HALLUCINATION_WINDOW:
        report.add(
            "time.hallucinated",
            clip.id,
            Severity.ERROR,
            f"{side} {t:.3f} is {distance:.3f}s from the nearest word {side} in words.json "
            f"({nearest.time:.3f}, {nearest.text!r}) — beyond the {HALLUCINATION_WINDOW}s "
            "window in which a timestamp can be called a rounding error. This value was not "
            "read from words.json.",
        )
        return

    report.add(
        "time.off_boundary",
        clip.id,
        Severity.WARN,
        f"{side} {t:.3f} misses the nearest word {side} ({nearest.time:.3f}, {nearest.text!r}) "
        f"by {distance:.3f}s, over the {tolerance}s tolerance. Run `ms lint --fix`.",
    )


# -- 3. gates ---------------------------------------------------------------


def _check_gates(
    clip: Clip,
    words: WordsDoc,
    silence: SilenceDoc | None,
    gates: Gates,
    report: LintReport,
) -> None:
    dur = clip.duration
    if dur < gates.duration.min - snap.EPS:
        report.add(
            "gate.duration_short",
            clip.id,
            Severity.ERROR,
            f"{dur:.2f}s is under gates.duration.min of {gates.duration.min}s",
        )
    if dur > gates.duration.max + snap.EPS:
        report.add(
            "gate.duration_long",
            clip.id,
            Severity.ERROR,
            f"{dur:.2f}s is over gates.duration.max of {gates.duration.max}s",
        )

    if silence is not None:
        _check_internal_silence(clip, silence, gates, report)

    if snap.has_sentence_flags(words):
        if gates.start_on_sentence_start:
            _check_sentence_edge(clip, words, gates, "start", report)
        if gates.end_on_sentence_end:
            _check_sentence_edge(clip, words, gates, "end", report)

    _check_opening_filler(clip, gates, report)
    _check_first_span_transition(clip, report)


def _check_first_span_transition(clip: Clip, report: LintReport) -> None:
    """A transition entering the first span has nothing to come from.

    A warning rather than an error: the render is still correct (the engine has
    nothing to blend and hard-cuts), but the edit list says something it cannot
    mean, and a reader would reasonably expect a fade-in. Edge fades in
    render.yaml are what actually produce that.
    """
    spans = clip.spans
    if spans and getattr(spans[0], "transition", None):
        report.add(
            "ref.transition_on_first_span",
            clip.id,
            Severity.WARN,
            f"the first layout span names transition "
            f"{spans[0].transition!r}, but there is nothing before it to "
            "transition from — it will hard-cut. For a fade from black, set "
            "`fade.in` in render.yaml, which applies to every clip.",
        )


_WORD_RE = re.compile(r"[A-Za-z']+")


def _check_opening_filler(clip: Clip, gates: Gates, report: LintReport) -> None:
    """Reject a clip whose first spoken word is filler.

    Read off `source_text` rather than words.json on purpose: source_text is
    already verified against the transcript by `text.mismatch`, so it is the
    same words, and this stays correct for a clip whose start was padded into
    the silence before the first word.
    """
    fillers = {f.lower() for f in gates.forbid_opening_fillers}
    if not fillers:
        return
    found = _WORD_RE.findall(clip.source_text)
    if not found:
        return
    first = found[0].lower()
    if first not in fillers:
        return

    opening = " ".join(found[:6])
    severity = Severity.ERROR if gates.opening_filler_is_error else Severity.WARN
    report.add(
        "gate.opens_on_filler",
        clip.id,
        severity,
        f"opens on the filler word {found[0]!r}: {opening!r}...  A cold open that "
        f"begins on a connective spends the hook window saying nothing. Pick a "
        f"span that starts on the point itself, or drop {found[0]!r} from "
        f"gates.forbid_opening_fillers if it reads fine here.",
    )


def _check_internal_silence(
    clip: Clip, silence: SilenceDoc, gates: Gates, report: LintReport
) -> None:
    """Intersect the clip range with every silence span.

    Clamping to the clip is what makes this correct at the edges: a long pause
    that merely *touches* the start of a clip contributes only the fraction of
    itself that is actually inside, which is at most the pad. Dead air that
    sits wholly inside the clip is what the gate is about.
    """
    for span in silence.spans:
        lo = max(span.start, clip.start)
        hi = min(span.end, clip.end)
        inside = hi - lo
        if inside > gates.max_internal_silence + snap.EPS:
            report.add(
                "gate.internal_silence",
                clip.id,
                Severity.ERROR,
                f"{inside:.2f}s of dead air at {lo:.2f}–{hi:.2f} (silence span "
                f"{span.start:.2f}–{span.end:.2f}) exceeds gates.max_internal_silence "
                f"of {gates.max_internal_silence}s",
            )


def _check_sentence_edge(
    clip: Clip, words: WordsDoc, gates: Gates, side: snap.Side, report: LintReport
) -> None:
    t = clip.start if side == "start" else clip.end
    pad = gates.pad_in if side == "start" else gates.pad_out
    check = snap.check_boundary(words, t, side, tolerance=gates.boundary_tolerance, pad=pad)
    if not check.ok or check.nearest is None:
        # Already reported as a time.* finding; a sentence verdict on a
        # timestamp we do not believe would only add noise.
        return
    # Anchor on the word actually *inside* the clip, not on whatever edge is
    # numerically closest. With padding applied, the closest edge to a clip end
    # is often the NEXT word's onset -- pad_out legitimately runs into the gap
    # before it -- and asking whether that word is a sentence end is asking
    # about a word the clip never contains. The last word to finish before the
    # cut is the one whose sentence flag matters.
    # Containment, not proximity. A word that merely touches the cut is not in
    # the clip: `They` starting at exactly the end timestamp is the next
    # sentence, however close its own end happens to be.
    slack = gates.boundary_tolerance + pad
    eps = 1e-6
    if side == "start":
        # The first word that is still being spoken after the cut.
        candidates = [x for x in words.words if x.end > t + eps]
        w = min(candidates, key=lambda x: x.start, default=check.nearest.word)
        if w.start > t + slack:
            w = check.nearest.word
    else:
        # The last word that had already begun before the cut.
        candidates = [x for x in words.words if x.start < t - eps]
        w = max(candidates, key=lambda x: x.end, default=check.nearest.word)
        if w.end < t - slack:
            w = check.nearest.word

    flagged = w.sentence_start if side == "start" else w.sentence_end
    if flagged:
        return
    kind: snap.BoundaryKind = "sentence_start" if side == "start" else "sentence_end"
    nearest_sentence = snap.nearest_boundary(words, t, kind, prefer="earlier")
    where = (
        f" Nearest sentence {side} is {nearest_sentence.time:.3f} ({nearest_sentence.text!r})."
        if nearest_sentence
        else ""
    )
    verb = "opens" if side == "start" else "closes"
    message = (
        f"clip {verb} mid-sentence: {side} {t:.3f} lands on the word {w.text.strip()!r}, "
        f"which is not a sentence {side}.{where}"
    )
    if side == "start":
        report.add("gate.start_on_sentence_start", clip.id, Severity.ERROR, message)
    else:
        report.add("gate.end_on_sentence_end", clip.id, Severity.ERROR, message)


def _check_separation(doc: ClipsDoc, gates: Gates, report: LintReport) -> None:
    mids = [(c, (c.start + c.end) / 2) for c in doc.clips]
    for i, (a, ma) in enumerate(mids):
        for b, mb in mids[i + 1 :]:
            gap = abs(mb - ma)
            if gap < gates.min_separation - snap.EPS:
                report.add(
                    "gate.min_separation",
                    b.id,
                    Severity.ERROR,
                    f"midpoint {mb:.2f} is only {gap:.2f}s from clip {a.id!r} (midpoint "
                    f"{ma:.2f}); gates.min_separation is {gates.min_separation}s",
                )


# -- 4. rubric --------------------------------------------------------------


def _check_rubric(clip: Clip, criteria: Criteria, report: LintReport) -> None:
    scores = {k: v.score for k, v in clip.why.scores.items()}
    missing = [cid for cid in criteria.ids if cid not in scores]
    for cid in missing:
        report.add(
            "rubric.missing_score",
            clip.id,
            Severity.ERROR,
            f"why.scores has no entry for rubric criterion {cid!r} — every criterion in "
            "criteria.yaml must be scored with evidence",
        )
    unknown = [k for k in scores if k not in set(criteria.ids)]
    for k in sorted(unknown):
        report.add(
            "rubric.unknown_score",
            clip.id,
            Severity.WARN,
            f"why.scores contains {k!r}, which is not a criterion in criteria.yaml; "
            "it is ignored by the weighted score",
        )

    for cid, actual, required in criteria.failed_vetoes(scores):
        report.add(
            "rubric.veto_failed",
            clip.id,
            Severity.ERROR,
            f"veto on {cid!r}: scored {actual}, minimum is {required}. A veto failure kills "
            "the clip regardless of the weighted mean.",
        )

    if missing:
        # Cannot recompute without a complete set; the missing findings above
        # are the actionable ones.
        return

    computed = criteria.weighted_score(scores)
    stated = clip.why.weighted_score
    if abs(computed - stated) > SCORE_TOLERANCE:
        report.add(
            "rubric.weighted_score_mismatch",
            clip.id,
            Severity.ERROR,
            f"why.weighted_score says {stated:.4g} but the scores as written work out to "
            f"{computed:.4f} (weighted mean, weights from criteria.yaml)",
        )
    if computed < criteria.scoring.min_weighted_score - SCORE_TOLERANCE:
        report.add(
            "rubric.below_min_score",
            clip.id,
            Severity.ERROR,
            f"weighted score {computed:.2f} is below scoring.min_weighted_score of "
            f"{criteria.scoring.min_weighted_score}",
        )


def _check_diversity(doc: ClipsDoc, criteria: Criteria, report: LintReport) -> None:
    cap = criteria.diversity.max_per_theme
    seen: dict[str, list[str]] = {}
    for clip in doc.clips:
        theme = clip.why.theme
        seen.setdefault(theme, []).append(clip.id)
        if len(seen[theme]) > cap:
            report.add(
                "rubric.theme_overrepresented",
                clip.id,
                Severity.ERROR,
                f"theme {theme!r} already has {cap} clip(s) ({', '.join(seen[theme][:cap])}); "
                f"diversity.max_per_theme is {cap}",
            )


# -- 5. visual dependency ---------------------------------------------------


def _check_visual(doc: ClipsDoc, clip: Clip, gates: Gates, report: LintReport) -> None:
    if not (clip.visual_dependency and gates.require_slide_region_when_visually_dependent):
        return
    # resolved_regions() spans every source, so a slide living in a different
    # file than the face satisfies this rule exactly as one in the same file
    # does. That is the point of a multi-source job: the sharp slides and the
    # usable face are rarely in the same export.
    regions = doc.resolved_regions()
    referenced = {rid for span in clip.spans for rid in layout_region_ids(span)}
    if any(regions[r].kind == "slide" for r in referenced if r in regions):
        return
    multi = len(doc.resolved_sources()) > 1
    available = [
        f"{r} (in source {regions[r].source!r})" if multi else r
        for r in sorted(regions)
        if regions[r].kind == "slide"
    ]
    where = "sources.*.regions" if multi else "source.regions"
    hint = (
        f" Slide regions available: {', '.join(available)}."
        if available
        else f" No region of kind 'slide' is declared in {where} at all."
    )
    report.add(
        "visual.no_slide_region",
        clip.id,
        Severity.ERROR,
        f"visual_dependency is true, but the layout only shows "
        f"{', '.join(sorted(referenced)) or 'nothing'} — none of which is a slide region."
        + hint,
    )


# -- 6. referential integrity -----------------------------------------------


def _known_regions(doc: ClipsDoc) -> str:
    """The region ids a layout may name, for the "here is what exists" half of
    an unknown-reference message.

    With one source this is a flat list, which is all it ever needed to be.
    With several, a flat list is actively misleading -- `slides` and `cam_a`
    reading as peers hides the fact that they are measured against different
    files -- so the ids are grouped under the source that owns them.
    """
    regions = doc.resolved_regions()
    srcs = doc.resolved_sources()
    if len(srcs) == 1:
        return ", ".join(sorted(regions))
    primary = doc.primary_source_name
    owner = {rid: (r.source or primary) for rid, r in regions.items()}
    parts = [
        f"{name}: {', '.join(sorted(r for r in regions if owner[r] == name))}"
        for name in srcs
        if any(owner[r] == name for r in regions)
    ]
    # Regions whose source does not exist are reported by ref.unknown_source,
    # but they must still appear here or this list silently loses ids.
    orphans = sorted(r for r in regions if owner[r] not in srcs)
    if orphans:
        parts.append(f"(undeclared source): {', '.join(orphans)}")
    return "; ".join(parts)


def _check_referential(doc: ClipsDoc, clip: Clip, report: LintReport) -> None:
    regions = doc.resolved_regions()
    known = _known_regions(doc)
    declared_in = "source.regions" if len(doc.resolved_sources()) == 1 else "sources.*.regions"

    if not re.match(CLIP_ID_PATTERN, clip.id):
        report.add(
            "ref.bad_clip_id",
            clip.id,
            Severity.ERROR,
            f"clip id {clip.id!r} does not match {CLIP_ID_PATTERN} — the id drives the "
            "output filename, so it must be sortable and filesystem-safe",
        )

    unknown: list[str] = []
    for span in clip.spans:
        for rid in layout_region_ids(span):
            if rid not in regions and rid not in unknown:
                unknown.append(rid)
    for rid in unknown:
        report.add(
            "ref.unknown_region",
            clip.id,
            Severity.ERROR,
            f"layout references region {rid!r}, which is not in {declared_in}. "
            f"Known regions: {known}",
        )
    if clip.speaker is not None and clip.speaker not in regions:
        report.add(
            "ref.unknown_speaker",
            clip.id,
            Severity.ERROR,
            f"speaker {clip.speaker!r} is not a region in {declared_in}. "
            f"Known regions: {known}",
        )

    ats = [s.at for s in clip.spans]
    if ats:
        if abs(ats[0]) > snap.EPS:
            report.add(
                "ref.span_start_not_zero",
                clip.id,
                Severity.ERROR,
                f"the first layout span is at {ats[0]}, but a clip must have a layout from "
                "its first frame",
            )
        if ats != sorted(ats) or len(set(ats)) != len(ats):
            report.add(
                "ref.spans_unordered",
                clip.id,
                Severity.ERROR,
                f"layout spans must be strictly ordered by `at`; got {ats}",
            )
        for a in ats:
            if a < -snap.EPS or a >= clip.duration - snap.EPS:
                report.add(
                    "ref.span_out_of_bounds",
                    clip.id,
                    Severity.ERROR,
                    f"layout span at {a} lies outside the clip, which is "
                    f"{clip.duration:.2f}s long",
                )

    # The primary source is the job's clock: every timestamp in the edit list
    # is an instant in it, whether or not this particular clip puts it on
    # screen. Companion sources are checked separately, by
    # source.clip_out_of_range, against the clips that actually use them.
    primary = doc.resolved_sources()[doc.primary_source_name]
    if clip.start < -snap.EPS or clip.end > primary.duration + snap.EPS:
        which = (
            f"the source, which is {primary.duration:.2f}s long"
            if len(doc.resolved_sources()) == 1
            else f"the primary source {doc.primary_source_name!r}, which is "
            f"{primary.duration:.2f}s long"
        )
        report.add(
            "ref.out_of_source",
            clip.id,
            Severity.ERROR,
            f"clip range {clip.start:.2f}–{clip.end:.2f} falls outside {which}",
        )


def _check_clip_relations(doc: ClipsDoc, report: LintReport) -> None:
    seen: dict[str, int] = {}
    for clip in doc.clips:
        seen[clip.id] = seen.get(clip.id, 0) + 1
        if seen[clip.id] == 2:
            report.add(
                "ref.duplicate_clip_id",
                clip.id,
                Severity.ERROR,
                f"clip id {clip.id!r} is used more than once; ids drive output filenames "
                "and must be unique",
            )

    ordered = sorted(doc.clips, key=lambda c: c.start)
    for a, b in zip(ordered, ordered[1:]):
        if b.start < a.end - snap.EPS:
            report.add(
                "ref.overlapping_clips",
                b.id,
                Severity.ERROR,
                f"overlaps clip {a.id!r} ({a.start:.2f}–{a.end:.2f}) by "
                f"{a.end - b.start:.2f}s",
            )


# -- 7. multi-source coherence ----------------------------------------------
#
# A multi-source job rests on one assumption that nothing else in the pipeline
# can verify: that the files are frame-aligned views of a single recording, so
# that a timestamp means the same instant in all of them. Everywhere else in
# this module, a shared clock is simply true. Here it is a claim, and these
# rules are what turn it into a checked one.


def _check_sources(doc: ClipsDoc, report: LintReport, *, check_files: bool) -> None:
    srcs = doc.resolved_sources()

    for rid, region in sorted(doc.resolved_regions().items()):
        if region.source is not None and region.source not in srcs:
            report.add(
                "ref.unknown_source",
                None,
                Severity.ERROR,
                f"region {rid!r} is measured against source {region.source!r}, which is not "
                f"declared. Known sources: {', '.join(srcs)}",
            )

    if len(srcs) > 1:
        primary_name = doc.primary_source_name
        primary = srcs[primary_name]
        for name, spec in srcs.items():
            if name == primary_name:
                continue
            delta = abs(spec.duration - primary.duration)
            if delta > SOURCE_DURATION_TOLERANCE:
                report.add(
                    "source.duration_mismatch",
                    None,
                    Severity.WARN,
                    f"source {name!r} is {spec.duration:.2f}s but {primary_name!r} is "
                    f"{primary.duration:.2f}s — a difference of {delta:.2f}s, over the "
                    f"{SOURCE_DURATION_TOLERANCE}s these files are expected to agree within. "
                    "Sources are composited on the assumption that a timestamp means the "
                    "same instant in each of them; if these are not frame-aligned exports "
                    "of one recording, every clip that mixes them will show a face and a "
                    "slide from different moments, and nothing later in the pipeline will "
                    "notice. Check that both files came from the same session and neither "
                    "has been trimmed.",
                )

    if check_files:
        for name, spec in srcs.items():
            if not Path(spec.path).exists():
                report.add(
                    "source.missing_file",
                    None,
                    Severity.WARN,
                    f"source {name!r} points at {spec.path!r}, which is not on disk. The "
                    "render will fail when it reaches this file; everything else in this "
                    "report was still checked.",
                )


def _sources_used_by(doc: ClipsDoc, clip: Clip) -> dict[str, list[str]]:
    """Source name -> the region ids that put that source on screen.

    Layout regions only. `speaker` is editorial metadata about who is talking,
    not an instruction to show anything, so it does not make a file part of
    this clip's render.
    """
    regions = doc.resolved_regions()
    used: dict[str, list[str]] = {}
    for span in clip.spans:
        for rid in layout_region_ids(span):
            region = regions.get(rid)
            if region is None:
                continue  # already reported as ref.unknown_region
            name = region.source or doc.primary_source_name
            ids = used.setdefault(name, [])
            if rid not in ids:
                ids.append(rid)
    return used


def _check_clip_sources(doc: ClipsDoc, clip: Clip, report: LintReport) -> None:
    """Every companion file the clip draws on must actually reach that far.

    This is the failure the multi-source feature invites and the one that must
    never reach the renderer: a clip cut at 3400s of a long recording, composed
    against a companion export that stops at 1200s. The edit list looks
    entirely reasonable, the primary source covers the range, and the render
    dies -- or worse, silently produces a frozen or black region -- well after
    the point where anyone is still watching.
    """
    srcs = doc.resolved_sources()
    primary_name = doc.primary_source_name
    for name, region_ids in _sources_used_by(doc, clip).items():
        if name == primary_name:
            continue  # covered by ref.out_of_source, which owns the job's clock
        spec = srcs.get(name)
        if spec is None:
            continue  # already reported as ref.unknown_source
        if clip.end > spec.duration + snap.EPS:
            short = clip.end - spec.duration
            report.add(
                "source.clip_out_of_range",
                clip.id,
                Severity.ERROR,
                f"clip runs to {clip.end:.2f}s, but source {name!r} ({spec.path}) is only "
                f"{spec.duration:.2f}s long — {short:.2f}s short of the end of this clip. "
                f"The layout shows {', '.join(sorted(region_ids))} from that file, so there "
                "is nothing there to render.",
            )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _load(path: str | Path, model: type) -> Any:
    return model.model_validate(json.loads(Path(path).read_text()))


def lint_paths(
    clips_path: str | Path,
    words_path: str | Path,
    *,
    silence_path: str | Path | None = None,
    criteria_path: str | Path | None = None,
    criteria: Criteria | None = None,
    check_files: bool = False,
) -> LintReport:
    """Lint a job's files. A clips.json that will not even parse is itself a
    finding (`schema.invalid`) rather than an exception, so the CLI has one
    uniform way to report failure."""
    if criteria is None:
        criteria = (
            load_criteria(criteria_path) if criteria_path is not None else load_criteria()
        )
    words: WordsDoc = _load(words_path, WordsDoc)
    silence: SilenceDoc | None = (
        _load(silence_path, SilenceDoc) if silence_path is not None else None
    )
    return _lint_clips_file(
        clips_path, words=words, criteria=criteria, silence=silence, check_files=check_files
    )


def _lint_clips_file(
    clips_path: str | Path,
    *,
    words: WordsDoc,
    criteria: Criteria,
    silence: SilenceDoc | None,
    check_files: bool = False,
) -> LintReport:
    try:
        doc: ClipsDoc = _load(clips_path, ClipsDoc)
    except ValidationError as exc:
        report = LintReport()
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            report.add(
                "schema.invalid",
                None,
                Severity.ERROR,
                f"{loc or '<root>'}: {err['msg']}",
            )
        return report
    except json.JSONDecodeError as exc:
        report = LintReport()
        report.add("schema.invalid", None, Severity.ERROR, f"not valid JSON: {exc}")
        return report
    return lint_doc(
        doc, words=words, criteria=criteria, silence=silence, check_files=check_files
    )


def lint_job(
    job: Any, criteria: Criteria, *, fix: bool = False, check_files: bool = False
) -> LintReport:
    """Lint one job directory. The entry point `ms lint` calls.

    `job` is duck-typed on purpose — it needs `clips_json`, `words_json`,
    `silence_json` and `load_words()` / `load_silence()`, and nothing in the
    editorial layer should have to import the job model to be tested.

    With `fix=True` the mechanical repairs are applied and written back
    *before* the checks run, so what is reported is the state of the file the
    user is left holding.
    """
    words: WordsDoc = job.load_words()
    silence: SilenceDoc | None = (
        job.load_silence() if Path(job.silence_json).exists() else None
    )

    fixed: list[str] = []
    clips_path = Path(job.clips_json)
    if fix and clips_path.exists():
        result = fix_text(clips_path.read_text(), words, criteria)
        if result.changed:
            clips_path.write_text(result.text)
            fixed = [str(c) for c in result.changes]

    report = _lint_clips_file(
        clips_path, words=words, criteria=criteria, silence=silence, check_files=check_files
    )
    report.fixed = fixed
    return report


# --------------------------------------------------------------------------
# 2. --fix
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixChange:
    clip_id: str
    field: snap.Side
    old: float
    new: float

    @property
    def delta(self) -> float:
        return self.new - self.old

    def __str__(self) -> str:
        return (
            f"{self.clip_id}: {self.field} {self.old:.3f} → {self.new:.3f} "
            f"({self.delta:+.3f}s)"
        )


@dataclass
class FixResult:
    text: str
    changes: list[FixChange] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _layout_ats(clip: dict[str, Any]) -> list[float]:
    """The `at` of every layout span, read straight out of the raw JSON."""
    layout = clip.get("layout")
    spans = layout if isinstance(layout, list) else [layout]
    return [
        float(s["at"])
        for s in spans
        if isinstance(s, dict) and isinstance(s.get("at"), (int, float))
    ]


def _detect_indent(text: str) -> str | int:
    """Match the file's own indentation. The user hand-edits this file; coming
    back to find it silently reflowed is its own small betrayal."""
    m = re.search(r"\n([ \t]+)\S", text)
    if not m:
        return 2
    lead = m.group(1)
    return "\t" if lead.startswith("\t") else len(lead)


def fix_text(
    text: str,
    words: WordsDoc,
    criteria: Criteria,
) -> FixResult:
    """Snap every clip's start/end to real boundaries and apply the pads.

    Only `start` and `end` are touched. `source_text` is deliberately left
    alone: rewriting it from words.json would make every file self-consistent,
    including the ones where the model invented the quote, and would erase the
    single most valuable finding this linter produces. If snapping moves an
    edge far enough to change the words, the next lint run says so and prints
    the verbatim replacement text to paste in.

    Key order is preserved (JSON objects round-trip as ordered dicts) and the
    file's own indentation is reused.
    """
    data: dict[str, Any] = json.loads(text)
    gates = criteria.gates
    # Snapping is clamped to the job's clock, which is the single `source` or,
    # for a multi-source job, the first entry of `sources` -- the same source
    # ClipsDoc.primary_source_name resolves to. Read from the raw JSON, because
    # --fix runs on files that may not load as a ClipsDoc at all.
    source_duration = None
    src = data.get("source")
    if src is None:
        sources = data.get("sources")
        if isinstance(sources, dict) and sources:
            src = next(iter(sources.values()))
    if isinstance(src, dict) and isinstance(src.get("duration"), (int, float)):
        source_duration = float(src["duration"])

    changes: list[FixChange] = []
    skipped: list[str] = []
    clips = data.get("clips")
    if not isinstance(clips, list):
        return FixResult(text=text)

    for i, clip in enumerate(clips):
        cid = clip.get("id", f"<clip {i}>") if isinstance(clip, dict) else f"<clip {i}>"
        if not isinstance(clip, dict) or not isinstance(clip.get("start"), (int, float)) or (
            not isinstance(clip.get("end"), (int, float))
        ):
            skipped.append(f"{cid}: start and end must be numbers before they can be snapped")
            continue
        old_start = float(clip["start"])
        old_end = float(clip["end"])
        result = snap.snap_clip(
            words,
            old_start,
            old_end,
            to_sentence_start=gates.start_on_sentence_start,
            to_sentence_end=gates.end_on_sentence_end,
            pad_in=gates.pad_in,
            pad_out=gates.pad_out,
            source_duration=source_duration,
        )
        # Snapping to a sentence boundary can shorten a clip by seconds, which
        # could strand a mid-clip layout change past the new end -- and a file
        # that no longer loads is a worse outcome than the misplaced edge this
        # was meant to repair. A repair tool must not break the thing it is
        # repairing.
        ats = _layout_ats(clip)
        if ats and max(ats) >= result.end - result.start - snap.EPS:
            skipped.append(
                f"{cid}: snapping to {result.start:.3f}–{result.end:.3f} would leave the "
                f"layout span at {max(ats)}s outside the clip; move the layout first"
            )
            continue

        if abs(result.start - old_start) > snap.EPS:
            clip["start"] = result.start
            changes.append(FixChange(str(cid), "start", old_start, result.start))
        if abs(result.end - old_end) > snap.EPS:
            clip["end"] = result.end
            changes.append(FixChange(str(cid), "end", old_end, result.end))

    out = json.dumps(data, indent=_detect_indent(text), ensure_ascii=False)
    if text.endswith("\n"):
        out += "\n"
    return FixResult(text=out, changes=changes, skipped=skipped)


def fix_file(
    clips_path: str | Path,
    words_path: str | Path,
    *,
    criteria_path: str | Path | None = None,
    criteria: Criteria | None = None,
    write: bool = True,
) -> FixResult:
    """`ms lint --fix`. Writes back in place unless `write=False`."""
    if criteria is None:
        criteria = (
            load_criteria(criteria_path) if criteria_path is not None else load_criteria()
        )
    words: WordsDoc = _load(words_path, WordsDoc)
    path = Path(clips_path)
    result = fix_text(path.read_text(), words, criteria)
    if write and result.changed:
        path.write_text(result.text)
    return result
