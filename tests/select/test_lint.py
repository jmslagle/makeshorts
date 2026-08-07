"""Fixture-driven tests for `ms lint` — the gate.

The shape of this suite: one valid baseline (`fixtures/clips.valid.json`,
generated together with the words.json and silence.json it must agree with),
then one deliberately-broken variant per failure mode. The headline failures —
an invented timestamp and an invented quote — are committed as whole files,
because those are what a human will actually stare at. The rest are single-key
mutations of the baseline, which keeps each test's breakage visible in the test
itself rather than buried in a near-identical 200-line JSON file.

Every test asserts the specific `rule_id`. "Something failed" is not a useful
guarantee from a linter whose whole job is telling you *what* failed.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from makeshorts.artifacts import SilenceDoc, WordsDoc
from makeshorts.select import snap
from makeshorts.select.criteria import Criteria, load_criteria
from makeshorts.select.lint import (
    FIXABLE_RULES,
    RULES,
    Severity,
    fix_file,
    fix_text,
    lint_doc,
    lint_job,
    lint_paths,
    normalize_text,
    normalize_tokens,
)
from makeshorts.select.schema import ClipsDoc, FocusLayout

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parents[2]

WORDS = WordsDoc.model_validate(json.loads((FIXTURES / "words.json").read_text()))
SILENCE = SilenceDoc.model_validate(json.loads((FIXTURES / "silence.json").read_text()))
CRITERIA = load_criteria(FIXTURES / "criteria.yaml")
BASELINE = json.loads((FIXTURES / "clips.valid.json").read_text())

PAD_IN, PAD_OUT = CRITERIA.gates.pad_in, CRITERIA.gates.pad_out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def lint(data: dict, *, silence: SilenceDoc | None = SILENCE, criteria: Criteria = CRITERIA):
    return lint_doc(
        ClipsDoc.model_validate(data), words=WORDS, criteria=criteria, silence=silence
    )


def broken(mutate, **kwargs):
    """Apply one mutation to a copy of the baseline and lint the result."""
    data = copy.deepcopy(BASELINE)
    mutate(data)
    return lint(data, **kwargs)


def ids(report) -> set[str]:
    return set(report.rule_ids())


def error_ids(report) -> set[str]:
    return {f.rule_id for f in report.errors}


def word_indices(clip: dict) -> list[int]:
    """Indices into words.json of the words a clip contains."""
    return [
        i
        for i, x in enumerate(WORDS.words)
        if clip["start"] <= (x.start + x.end) / 2 <= clip["end"]
    ]


def sentence_ends(clip: dict) -> list[int]:
    """Indices of the words that close a sentence inside a clip."""
    return [i for i in word_indices(clip) if WORDS.words[i].sentence_end]


def retime(clip: dict, first: int, last: int) -> None:
    """Move a clip onto a different span of words, keeping source_text honest.

    Retiming without rewriting the quote would trip text.mismatch in every
    test that only means to break one other thing.
    """
    ws = WORDS.words
    start, end, _, _ = snap.pad_range(
        WORDS, ws[first].start, ws[last].end, pad_in=PAD_IN, pad_out=PAD_OUT
    )
    clip["start"] = round(start, 3)
    clip["end"] = round(end, 3)
    clip["source_text"] = snap.span_text(ws[first : last + 1])


def with_gates(**kw) -> Criteria:
    gates = type(CRITERIA.gates).model_validate({**CRITERIA.gates.model_dump(), **kw})
    return CRITERIA.model_copy(update={"gates": gates})


def set_score(clip: dict, criterion: str, value: int) -> None:
    """Change a score and keep the weighted mean consistent, so a test about
    vetoes is not also a test about arithmetic."""
    clip["why"]["scores"][criterion]["score"] = value
    scores = {k: v["score"] for k, v in clip["why"]["scores"].items()}
    clip["why"]["weighted_score"] = round(CRITERIA.weighted_score(scores), 2)


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


def test_baseline_is_clean() -> None:
    report = lint(BASELINE)
    assert report.findings == [], report.format()
    assert report.ok


def test_baseline_through_the_file_entry_point_is_clean() -> None:
    report = lint_paths(
        FIXTURES / "clips.valid.json",
        FIXTURES / "words.json",
        silence_path=FIXTURES / "silence.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert report.ok and not report.findings, report.format()


def test_warnings_do_not_block_the_render() -> None:
    report = lint_paths(
        FIXTURES / "clips.unsnapped.json",
        FIXTURES / "words.json",
        silence_path=FIXTURES / "silence.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert ids(report) == {"time.off_boundary"}
    assert report.warnings and not report.errors
    assert report.ok


# --------------------------------------------------------------------------
# 1. no hallucinated time
# --------------------------------------------------------------------------


def test_hallucinated_timestamp_is_an_error() -> None:
    report = lint_paths(
        FIXTURES / "clips.hallucinated-time.json",
        FIXTURES / "words.json",
        silence_path=FIXTURES / "silence.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert "time.hallucinated" in error_ids(report)
    assert not report.ok
    msg = report.for_rule("time.hallucinated")[0].message
    assert "words.json" in msg and "nearest word start" in msg


def test_a_cut_inside_a_word_is_an_error_not_a_warning() -> None:
    """Landing mid-word proves the number came from the transcript, so it is
    not a fabrication — but it still clips a syllable."""

    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        first = WORDS.words[word_indices(clip)[0]]
        clip["start"] = round((first.start + first.end) / 2, 3)

    report = broken(mutate)
    assert error_ids(report) == {"time.mid_word"}
    assert "falls inside the word" in report.for_rule("time.mid_word")[0].message


def test_an_edge_off_its_boundary_but_close_is_only_a_warning() -> None:
    def mutate(data: dict) -> None:
        data["clips"][0]["start"] -= 0.18

    report = broken(mutate)
    assert ids(report) == {"time.off_boundary"}
    assert report.ok


def test_the_padded_window_is_not_reported_as_off_boundary() -> None:
    """A start sits pad_in before its word by construction; the tolerance has
    to allow for that or every freshly fixed file fails."""
    clip = BASELINE["clips"][0]
    first = WORDS.words[word_indices(clip)[0]]
    assert clip["start"] == pytest.approx(first.start - PAD_IN)
    assert lint(BASELINE).findings == []


def test_source_text_mismatch_is_an_error_with_a_readable_diff() -> None:
    report = lint_paths(
        FIXTURES / "clips.text-mismatch.json",
        FIXTURES / "words.json",
        silence_path=FIXTURES / "silence.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert error_ids(report) == {"text.mismatch"}
    msg = report.for_rule("text.mismatch")[0].message
    # The three planted differences, each named with its position.
    assert 'missing from clips.json: "and almost nobody checks it"' in msg
    assert 'words.json has "four" but clips.json has "five"' in msg
    assert 'words.json has "companies" but clips.json has "startups"' in msg
    # And the verbatim replacement text, so the fix is a copy-paste.
    assert "Eighteen months. That is the number that kills companies," in msg


def test_a_range_containing_no_words_is_an_error() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        clip["start"], clip["end"] = 109.9, 113.4  # the dead air before the block

    report = broken(mutate)
    assert "text.empty_span" in error_ids(report)


TWO_SENTENCES = (
    "Eighteen months. That is the number that kills companies, "
    "and almost nobody checks it."
)


@pytest.mark.parametrize(
    "variant",
    [
        TWO_SENTENCES,
        # case
        "eighteen MONTHS. that is the number that kills companies, "
        "and almost nobody checks it.",
        # punctuation, including a missing comma and a changed terminator
        "Eighteen months — that is the number that kills companies "
        "and almost nobody checks it!",
        # whitespace, as a hand-wrapped JSON string
        "Eighteen  months.\n   That is the number that kills companies,\n"
        "   and almost\tnobody checks it.",
        # curly quotes and an ellipsis
        "“Eighteen months.” That is the number that kills companies, "
        "and almost nobody checks it…",
    ],
)
def test_normalization_tolerates_case_punctuation_and_whitespace(variant: str) -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        idx = word_indices(clip)
        retime(clip, idx[0], sentence_ends(clip)[1])  # the first two sentences
        clip["source_text"] = variant

    report = broken(mutate, criteria=with_gates(duration={"min": 1, "target": 45, "max": 60}))
    assert "text.mismatch" not in ids(report), report.format()


def test_normalization_does_not_tolerate_dropped_filler_words() -> None:
    """Tidying 'um' out of a quote is a real editorial wish, but tolerating it
    would blunt the only check that catches a quote written from memory."""
    assert normalize_tokens("So, um, yeah — and then?") == ["so", "um", "yeah", "and", "then"]
    assert normalize_text("don't") == "dont"
    assert normalize_text("Well-known  facts.") == "well known facts"
    assert normalize_text("“Curly” quotes") == "curly quotes"


def test_a_dropped_word_is_a_mismatch() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        clip["source_text"] = clip["source_text"].replace("almost nobody", "nobody")

    report = broken(mutate)
    assert error_ids(report) == {"text.mismatch"}


# --------------------------------------------------------------------------
# 2. --fix
# --------------------------------------------------------------------------


def test_fix_snaps_an_unsnapped_file_back_onto_its_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "clips.json"
    path.write_text((FIXTURES / "clips.unsnapped.json").read_text())
    result = fix_file(path, FIXTURES / "words.json", criteria=CRITERIA)

    assert [(c.clip_id, c.field) for c in result.changes] == [
        ("01-cac-payback-math", "start"),
        ("01-cac-payback-math", "end"),
    ]
    assert json.loads(path.read_text()) == BASELINE
    assert lint(json.loads(path.read_text())).findings == []


def test_fix_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "clips.json"
    path.write_text((FIXTURES / "clips.unsnapped.json").read_text())
    fix_file(path, FIXTURES / "words.json", criteria=CRITERIA)
    after_first = path.read_text()
    second = fix_file(path, FIXTURES / "words.json", criteria=CRITERIA)
    assert second.changes == []
    assert not second.changed
    assert path.read_text() == after_first


def test_fix_of_a_clean_file_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "clips.json"
    original = (FIXTURES / "clips.valid.json").read_text()
    path.write_text(original)
    result = fix_file(path, FIXTURES / "words.json", criteria=CRITERIA)
    assert not result.changed
    assert path.read_text() == original


def test_fix_preserves_key_order_and_indentation() -> None:
    text = (FIXTURES / "clips.unsnapped.json").read_text()
    result = fix_text(text, WORDS, CRITERIA)
    before = json.loads(text)
    after = json.loads(result.text)
    assert list(after) == list(before)
    assert list(after["clips"][0]) == list(before["clips"][0])
    assert result.text.startswith('{\n  "schema_version"')
    assert result.text.endswith("\n")


def test_fix_does_not_rewrite_source_text() -> None:
    """Rewriting the quote from words.json would make every file
    self-consistent, including the ones where the model made the quote up."""
    text = (FIXTURES / "clips.text-mismatch.json").read_text()
    result = fix_text(text, WORDS, CRITERIA)
    assert json.loads(result.text)["clips"][0]["source_text"] == (
        json.loads(text)["clips"][0]["source_text"]
    )


def test_fix_snaps_to_sentence_boundaries_when_the_gate_asks_for_it() -> None:
    """A start dropped in the middle of a sentence is pulled back to the
    sentence's first word, not merely to the nearest word."""
    data = copy.deepcopy(BASELINE)
    clip = data["clips"][0]
    idx = word_indices(clip)
    opener = WORDS.words[idx[2]]  # the second sentence's first word
    assert opener.sentence_start and not WORDS.words[idx[4]].sentence_start
    clip["start"] = WORDS.words[idx[4]].start  # a word start, but mid-sentence
    result = fix_text(json.dumps(data, indent=2) + "\n", WORDS, CRITERIA)
    fixed = json.loads(result.text)["clips"][0]["start"]
    assert fixed == round(opener.start - PAD_IN, 3)


def test_fix_skips_clips_it_cannot_understand() -> None:
    data = copy.deepcopy(BASELINE)
    data["clips"][1]["start"] = "somewhere around here"
    result = fix_text(json.dumps(data, indent=2), WORDS, CRITERIA)
    assert len(result.skipped) == 1
    assert result.skipped[0].startswith("02-first-sales-hire:")
    assert not [c for c in result.changes if c.clip_id == "02-first-sales-hire"]


def test_fix_refuses_to_strand_a_layout_span_outside_the_clip() -> None:
    """Snapping to a sentence boundary can shorten a clip past a mid-clip
    layout change. Writing that back would produce a clips.json that no longer
    loads — worse than the edge it was fixing."""
    data = copy.deepcopy(BASELINE)
    clip = data["clips"][1]
    ends = sentence_ends(clip)
    clip["end"] = round(WORDS.words[ends[0]].end + PAD_OUT, 3)  # first sentence only, ~5s
    result = fix_text(json.dumps(data, indent=2), WORDS, CRITERIA)
    assert result.changes == []
    assert "layout span at 8.0s outside the clip" in result.skipped[0]


# --------------------------------------------------------------------------
# 3. gates
# --------------------------------------------------------------------------


def test_duration_under_the_minimum() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        idx = word_indices(clip)
        retime(clip, idx[0], sentence_ends(clip)[1])  # first two sentences, about 8s

    report = broken(mutate)
    assert error_ids(report) == {"gate.duration_short"}
    assert "gates.duration.min" in report.for_rule("gate.duration_short")[0].message


def test_duration_over_the_maximum() -> None:
    report = lint(BASELINE, criteria=with_gates(duration={"min": 20, "target": 25, "max": 30}))
    long_clips = {f.clip_id for f in report.for_rule("gate.duration_long")}
    assert long_clips == {"01-cac-payback-math", "02-first-sales-hire"}  # 34.8s and 30.6s
    assert error_ids(report) == {"gate.duration_long"}


def test_excessive_internal_silence() -> None:
    silence = SilenceDoc.model_validate(
        json.loads((FIXTURES / "silence.excessive.json").read_text())
    )
    report = lint(BASELINE, silence=silence)
    assert error_ids(report) == {"gate.internal_silence"}
    finding = report.for_rule("gate.internal_silence")[0]
    assert finding.clip_id == "01-cac-payback-math"
    assert "2.50s of dead air" in finding.message


def test_silence_that_only_touches_a_clip_edge_is_not_internal() -> None:
    """The four-second gap before clip 01 overlaps it by the pad and no more."""
    assert "gate.internal_silence" not in ids(lint(BASELINE))


def test_clips_too_close_together() -> None:
    report = lint(BASELINE, criteria=with_gates(min_separation=120))
    assert error_ids(report) == {"gate.min_separation"}
    assert len(report.for_rule("gate.min_separation")) == 3  # every pair


def test_too_many_clips() -> None:
    report = lint(BASELINE, criteria=with_gates(max_clips=2))
    assert error_ids(report) == {"gate.max_clips"}
    finding = report.for_rule("gate.max_clips")[0]
    assert finding.clip_id is None  # a property of the edit list, not of a clip


def test_clip_must_start_on_a_sentence_start() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        idx = word_indices(clip)
        retime(clip, idx[1], idx[-1])  # start on the second word of a sentence

    report = broken(mutate)
    assert error_ids(report) == {"gate.start_on_sentence_start"}
    assert "mid-sentence" in report.for_rule("gate.start_on_sentence_start")[0].message


def test_clip_must_end_on_a_sentence_end() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        idx = word_indices(clip)
        retime(clip, idx[0], idx[-2])

    report = broken(mutate)
    assert error_ids(report) == {"gate.end_on_sentence_end"}


def test_sentence_gates_are_skipped_when_the_transcript_has_no_flags() -> None:
    """No segmentation means we cannot tell; failing every clip would be
    confidently wrong."""
    flat = WordsDoc(
        model=WORDS.model,
        language=WORDS.language,
        words=[w.model_copy(update={"sentence_start": False, "sentence_end": False})
               for w in WORDS.words],
    )
    report = lint_doc(
        ClipsDoc.model_validate(BASELINE), words=flat, criteria=CRITERIA, silence=SILENCE
    )
    assert ids(report) == {"gate.no_sentence_flags"}
    assert report.ok


def test_missing_silence_json_is_a_warning() -> None:
    report = lint(BASELINE, silence=None)
    assert ids(report) == {"gate.silence_unavailable"}
    assert report.ok


def test_an_empty_transcript_makes_every_timestamp_unverifiable() -> None:
    empty = WordsDoc(model="x", language="en", words=[])
    report = lint_doc(
        ClipsDoc.model_validate(BASELINE), words=empty, criteria=CRITERIA, silence=SILENCE
    )
    assert "time.no_words" in error_ids(report)


# --------------------------------------------------------------------------
# 4. rubric
# --------------------------------------------------------------------------


def test_missing_rubric_criterion() -> None:
    def mutate(data: dict) -> None:
        del data["clips"][0]["why"]["scores"]["payoff"]

    report = broken(mutate)
    assert error_ids(report) == {"rubric.missing_score"}
    assert "'payoff'" in report.for_rule("rubric.missing_score")[0].message


def test_a_score_that_is_not_in_the_rubric_is_only_a_warning() -> None:
    def mutate(data: dict) -> None:
        data["clips"][0]["why"]["scores"]["vibes"] = {"score": 5, "evidence": "trust me"}

    report = broken(mutate)
    assert ids(report) == {"rubric.unknown_score"}
    assert report.ok


def test_weighted_score_is_recomputed_not_trusted() -> None:
    def mutate(data: dict) -> None:
        data["clips"][2]["why"]["weighted_score"] = 4.9  # actually 4.0

    report = broken(mutate)
    assert error_ids(report) == {"rubric.weighted_score_mismatch"}
    assert "4.0000" in report.for_rule("rubric.weighted_score_mismatch")[0].message


def test_weighted_score_within_tolerance_passes() -> None:
    def mutate(data: dict) -> None:
        data["clips"][2]["why"]["weighted_score"] = 4.009

    assert broken(mutate).findings == []


def test_veto_violation() -> None:
    def mutate(data: dict) -> None:
        set_score(data["clips"][0], "hook_strength", 3)  # veto requires 4

    report = broken(mutate)
    assert error_ids(report) == {"rubric.veto_failed"}
    assert "scored 3, minimum is 4" in report.for_rule("rubric.veto_failed")[0].message


def test_below_the_minimum_weighted_score() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][0]
        for cid in ("single_idea", "payoff", "specificity", "quotability", "housekeeping"):
            clip["why"]["scores"][cid]["score"] = 1
        set_score(clip, "hook_strength", 4)  # vetoes still satisfied
        set_score(clip, "standalone", 4)

    report = broken(mutate)
    assert error_ids(report) == {"rubric.below_min_score"}


def test_too_many_clips_on_one_theme() -> None:
    def mutate(data: dict) -> None:
        data["clips"][1]["why"]["theme"] = "unit-economics"

    report = broken(mutate)
    assert error_ids(report) == {"rubric.theme_overrepresented"}
    finding = report.for_rule("rubric.theme_overrepresented")[0]
    assert finding.clip_id == "03-churn-is-pricing"  # the third one, not the first two


def test_a_changed_rubric_warns_rather_than_failing() -> None:
    """The edit list is not wrong — it was scored against a rubric that has
    since moved. That is a fact the reviewer needs, not a blocked render."""
    report = lint(BASELINE, criteria=load_criteria(REPO / "config" / "criteria.yaml"))
    assert "rubric.criteria_changed" in {f.rule_id for f in report.warnings}
    assert "rubric.criteria_changed" not in error_ids(report)


# --------------------------------------------------------------------------
# 5. visual dependency
# --------------------------------------------------------------------------


def test_visual_dependency_without_a_slide_region() -> None:
    def mutate(data: dict) -> None:
        clip = data["clips"][1]
        clip["layout"] = [{"at": 0.0, "mode": "focus", "region": "cam_a", "fit": "cover"}]

    report = broken(mutate)
    assert error_ids(report) == {"visual.no_slide_region"}
    assert "Slide regions available: slides" in (
        report.for_rule("visual.no_slide_region")[0].message
    )


def test_visual_dependency_satisfied_by_any_span() -> None:
    """The slide only has to be on screen at some point in the clip."""
    assert "visual.no_slide_region" not in ids(lint(BASELINE))


def test_no_slide_region_declared_at_all_is_reported_differently() -> None:
    def mutate(data: dict) -> None:
        data["source"]["regions"] = [data["source"]["regions"][0]]
        data["clips"][1]["layout"] = [
            {"at": 0.0, "mode": "focus", "region": "cam_a", "fit": "cover"}
        ]
        data["clips"][2]["layout"] = [
            {"at": 0.0, "mode": "focus", "region": "frame", "fit": "contain_blur"}
        ]

    report = broken(mutate)
    assert error_ids(report) == {"visual.no_slide_region"}
    assert "No region of kind 'slide' is declared" in (
        report.for_rule("visual.no_slide_region")[0].message
    )


# --------------------------------------------------------------------------
# 6. referential integrity
# --------------------------------------------------------------------------


def test_unknown_region_in_a_layout() -> None:
    def mutate(data: dict) -> None:
        data["clips"][0]["layout"] = {"mode": "focus", "region": "cam_b", "fit": "cover"}

    report = broken(mutate)
    assert error_ids(report) == {"ref.unknown_region"}
    assert "Known regions: cam_a, frame, slides" in (
        report.for_rule("ref.unknown_region")[0].message
    )


def test_the_implicit_frame_region_is_always_valid() -> None:
    """`frame` is never listed in source.regions, and must still resolve."""
    assert "frame" in json.dumps(BASELINE["clips"][2]["layout"])
    assert "ref.unknown_region" not in ids(lint(BASELINE))


def test_unknown_speaker_region() -> None:
    def mutate(data: dict) -> None:
        data["clips"][0]["speaker"] = "cam_z"

    assert error_ids(broken(mutate)) == {"ref.unknown_speaker"}


def test_overlapping_clips() -> None:
    def mutate(data: dict) -> None:
        idx = word_indices(data["clips"][0])
        retime(data["clips"][1], idx[20], idx[-1])  # land clip 02 inside clip 01

    report = broken(mutate)
    assert "ref.overlapping_clips" in error_ids(report)
    assert "overlaps clip '01-cac-payback-math'" in (
        report.for_rule("ref.overlapping_clips")[0].message
    )


def test_clip_outside_the_source_duration() -> None:
    def mutate(data: dict) -> None:
        data["source"]["duration"] = 200.0

    report = broken(mutate)
    assert error_ids(report) == {"ref.out_of_source"}
    assert {f.clip_id for f in report.for_rule("ref.out_of_source")} == {
        "02-first-sales-hire",
        "03-churn-is-pricing",
    }


# The next few break rules the schema also enforces, so they cannot be
# expressed as a loadable file. `model_copy` bypasses validation, which is how
# a doc built in code (rather than parsed) could reach the linter -- and the
# linter is the gate, so it must not depend on having been handed a
# schema-validated object.


def base_doc() -> ClipsDoc:
    return ClipsDoc.model_validate(BASELINE)


def test_bad_clip_id() -> None:
    doc = base_doc()
    doc = doc.model_copy(
        update={"clips": [doc.clips[0].model_copy(update={"id": "Clip One"})]}
    )
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.bad_clip_id" in error_ids(report)


def test_duplicate_clip_ids() -> None:
    doc = base_doc()
    doc = doc.model_copy(update={"clips": [doc.clips[0], doc.clips[0]]})
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.duplicate_clip_id" in error_ids(report)


def test_layout_span_outside_the_clip() -> None:
    doc = base_doc()
    clip = doc.clips[2].model_copy(
        update={
            "layout": [
                FocusLayout(at=0.0, region="cam_a"),
                FocusLayout(at=900.0, region="slides"),
            ]
        }
    )
    doc = doc.model_copy(update={"clips": [clip]})
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.span_out_of_bounds" in error_ids(report)


def test_layout_spans_out_of_order() -> None:
    doc = base_doc()
    clip = doc.clips[2].model_copy(
        update={
            "layout": [
                FocusLayout(at=0.0, region="cam_a"),
                FocusLayout(at=20.0, region="slides"),
                FocusLayout(at=10.0, region="cam_a"),
            ]
        }
    )
    doc = doc.model_copy(update={"clips": [clip]})
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.spans_unordered" in error_ids(report)


def test_first_layout_span_must_be_at_zero() -> None:
    doc = base_doc()
    clip = doc.clips[2].model_copy(update={"layout": [FocusLayout(at=5.0, region="cam_a")]})
    doc = doc.model_copy(update={"clips": [clip]})
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.span_start_not_zero" in error_ids(report)


# --------------------------------------------------------------------------
# 7. multi-source coherence
# --------------------------------------------------------------------------
#
# The motivating shape is a Zoom export: several frame-aligned renders of one
# meeting, where the sharp slides and the only usable face are in different
# files. Mixing them is only safe because they share a clock, so these tests
# are all about that assumption being checked rather than trusted.


def two_source(data: dict | None = None, *, screen_duration: float = 298.92) -> dict:
    """The baseline edit list re-expressed as two frame-aligned files.

    Same clips, same timestamps, same layouts — only the face and the slides
    now live in different exports.
    """
    data = copy.deepcopy(BASELINE if data is None else data)
    src = data.pop("source")
    cam, slides = src["regions"]
    assert cam["id"] == "cam_a" and slides["id"] == "slides"
    data["sources"] = {
        "main": {**src, "regions": [cam]},
        "screen": {
            "path": "jobs/acme-q3-webinar/screen.mp4",
            "duration": screen_duration,
            "resolution": [2378, 1410],
            "regions": [slides],
        },
    }
    return data


def test_the_same_edit_list_split_across_two_sources_is_clean() -> None:
    """Nothing about splitting the regions across frame-aligned files makes an
    otherwise valid edit list invalid."""
    report = lint(two_source())
    assert report.findings == [], report.format()


def test_a_clip_running_past_a_companion_source_is_an_error() -> None:
    """The failure this feature invites: a clip the primary source covers
    comfortably, composed against a companion export that stops earlier."""
    report = lint(two_source(screen_duration=200.0))
    assert error_ids(report) == {"source.clip_out_of_range"}
    assert not report.ok
    # Clip 01 shows only the face, so the short file never comes into it.
    assert {f.clip_id for f in report.for_rule("source.clip_out_of_range")} == {
        "02-first-sales-hire",
        "03-churn-is-pricing",
    }
    msg = report.for_rule("source.clip_out_of_range")[0].message
    assert "source 'screen'" in msg and "200.00s long" in msg
    assert "4.59s short" in msg  # clip 02 runs to 204.59
    assert "slides" in msg


def test_a_clip_inside_every_source_it_uses_is_not_reported() -> None:
    """256.98s is the last frame any clip needs; a 260s companion covers it."""
    report = lint(two_source(screen_duration=260.0))
    assert "source.clip_out_of_range" not in ids(report)


def test_mismatched_source_durations_warn_without_blocking() -> None:
    """A heuristic must not stop a render — but a silent misalignment produces
    a clip whose face and slides are from different moments, which is very hard
    to spot afterwards."""
    report = lint(two_source(screen_duration=260.0))
    assert ids(report) == {"source.duration_mismatch"}
    assert report.ok
    finding = report.warnings[0]
    assert finding.severity is Severity.WARN
    assert finding.clip_id is None  # a property of the job, not of a clip
    assert "38.92s" in finding.message and "same instant" in finding.message


@pytest.mark.parametrize(
    ("screen_duration", "warns"),
    [
        (298.92, False),  # identical
        (298.42, False),  # half a second: container and codec bookkeeping
        (297.95, False),  # just inside the tolerance
        (297.42, True),  # a second and a half: not the same recording
        (120.0, True),
    ],
)
def test_the_duration_tolerance_allows_for_encoder_bookkeeping(
    screen_duration: float, warns: bool
) -> None:
    report = lint(two_source(screen_duration=screen_duration))
    assert ("source.duration_mismatch" in ids(report)) is warns


def test_a_region_naming_a_source_that_is_not_declared() -> None:
    data = two_source()
    data["sources"]["screen"]["regions"][0]["source"] = "camera_two"
    report = lint(data)
    assert error_ids(report) == {"ref.unknown_source"}
    finding = report.for_rule("ref.unknown_source")[0]
    assert finding.clip_id is None
    assert "region 'slides'" in finding.message
    assert "'camera_two'" in finding.message
    assert "Known sources: main, screen" in finding.message


def test_the_primary_source_still_bounds_every_clip_in_a_multi_source_job() -> None:
    """`ref.out_of_source` owns the job's clock and keeps working when the
    single `source` key is not there to read."""
    data = two_source()
    data["sources"]["main"]["duration"] = 200.0
    report = lint(data)
    assert error_ids(report) == {"ref.out_of_source"}
    assert {f.clip_id for f in report.for_rule("ref.out_of_source")} == {
        "02-first-sales-hire",
        "03-churn-is-pricing",
    }
    assert "primary source 'main'" in report.for_rule("ref.out_of_source")[0].message


def test_visual_dependency_is_satisfied_by_a_slide_in_another_source() -> None:
    """The whole point of a multi-source job: the sharp slides and the usable
    face are rarely in the same export."""
    doc = ClipsDoc.model_validate(two_source())
    assert doc.source_of("slides").path.endswith("screen.mp4")
    assert doc.source_of("cam_a").path.endswith("source.mp4")
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert report.findings == [], report.format()


def test_the_slide_region_hint_names_the_source_when_there_are_several() -> None:
    data = two_source()
    data["clips"][1]["layout"] = [{"at": 0.0, "mode": "focus", "region": "cam_a"}]
    report = lint(data)
    assert error_ids(report) == {"visual.no_slide_region"}
    assert "slides (in source 'screen')" in (
        report.for_rule("visual.no_slide_region")[0].message
    )


def test_known_regions_are_grouped_by_source() -> None:
    """A flat list would read as though these regions were peers, hiding the
    fact that they are measured against different files."""
    data = two_source()
    data["clips"][0]["layout"] = {"mode": "focus", "region": "cam_b", "fit": "cover"}
    report = lint(data)
    assert error_ids(report) == {"ref.unknown_region"}
    assert (
        "Known regions: main: cam_a, frame, main_frame; screen: screen_frame, slides"
        in report.for_rule("ref.unknown_region")[0].message
    )


def test_a_missing_source_file_is_a_warning(tmp_path: Path) -> None:
    present = tmp_path / "screen.mp4"
    present.write_bytes(b"")
    data = two_source()
    data["sources"]["main"]["path"] = str(tmp_path / "gone.mp4")
    data["sources"]["screen"]["path"] = str(present)
    report = lint_doc(
        ClipsDoc.model_validate(data),
        words=WORDS,
        criteria=CRITERIA,
        silence=SILENCE,
        check_files=True,
    )
    assert ids(report) == {"source.missing_file"}
    assert report.ok  # the render will fail anyway; saying so early is a kindness
    assert "source 'main'" in report.for_rule("source.missing_file")[0].message


def test_source_files_are_not_looked_for_unless_asked(tmp_path: Path) -> None:
    """A ClipsDoc is very often held with no filesystem to resolve against, and
    a confident "file missing" would be a lie in every one of those cases."""
    data = two_source()
    data["sources"]["main"]["path"] = str(tmp_path / "gone.mp4")
    assert "source.missing_file" not in ids(lint(data))


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_a_file_that_does_not_match_the_schema_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    data = copy.deepcopy(BASELINE)
    data["clips"][0]["id"] = "1-not-two-digits"
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(data))
    report = lint_paths(
        path,
        FIXTURES / "words.json",
        silence_path=FIXTURES / "silence.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert error_ids(report) == {"schema.invalid"}
    assert "clips.0.id" in report.findings[0].message


def test_unparseable_json_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "clips.json"
    path.write_text("{ not json at all")
    report = lint_paths(
        path,
        FIXTURES / "words.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert error_ids(report) == {"schema.invalid"}


# --------------------------------------------------------------------------
# the job entry point `ms lint` calls
# --------------------------------------------------------------------------


class FakeJob:
    """The whole of what lint_job needs from a job. Duck-typed on purpose:
    the editorial layer should not have to import the job model."""

    def __init__(self, root: Path, *, clips: str, silence: bool = True) -> None:
        self.root = root
        self.clips_json = root / "clips.json"
        self.words_json = root / "words.json"
        self.silence_json = root / "silence.json"
        self.clips_json.write_text((FIXTURES / clips).read_text())
        self.words_json.write_text((FIXTURES / "words.json").read_text())
        if silence:
            self.silence_json.write_text((FIXTURES / "silence.json").read_text())

    def load_words(self) -> WordsDoc:
        return WordsDoc.model_validate(json.loads(self.words_json.read_text()))

    def load_silence(self) -> SilenceDoc:
        return SilenceDoc.model_validate(json.loads(self.silence_json.read_text()))


def test_lint_job_on_a_clean_job(tmp_path: Path) -> None:
    report = lint_job(FakeJob(tmp_path, clips="clips.valid.json"), CRITERIA)
    assert report.ok and not report.findings and report.fixed == []


def test_lint_job_without_silence_json_still_runs(tmp_path: Path) -> None:
    report = lint_job(FakeJob(tmp_path, clips="clips.valid.json", silence=False), CRITERIA)
    assert ids(report) == {"gate.silence_unavailable"}
    assert report.ok


def test_lint_job_with_fix_repairs_then_reports_the_repaired_file(tmp_path: Path) -> None:
    job = FakeJob(tmp_path, clips="clips.unsnapped.json")
    report = lint_job(job, CRITERIA, fix=True)
    assert report.findings == [], report.format()
    assert report.fixed == [
        "01-cac-payback-math: start 113.410 → 113.590 (+0.180s)",
        "01-cac-payback-math: end 148.590 → 148.390 (-0.200s)",
    ]
    assert json.loads(job.clips_json.read_text()) == BASELINE


def test_lint_job_can_be_asked_to_look_for_the_source_file(tmp_path: Path) -> None:
    """The fixture's source path is not a real file, which is exactly the
    situation `check_files` exists to report — and exactly why it is off by
    default for a doc that may have been built anywhere."""
    job = FakeJob(tmp_path, clips="clips.valid.json")
    assert lint_job(job, CRITERIA).findings == []
    report = lint_job(job, CRITERIA, check_files=True)
    assert ids(report) == {"source.missing_file"}
    assert report.ok


def test_lint_job_without_fix_leaves_the_file_alone(tmp_path: Path) -> None:
    job = FakeJob(tmp_path, clips="clips.unsnapped.json")
    before = job.clips_json.read_text()
    report = lint_job(job, CRITERIA, fix=False)
    assert job.clips_json.read_text() == before
    assert ids(report) == {"time.off_boundary"}


def test_findings_say_whether_fix_can_repair_them() -> None:
    """The CLI offers `--fix` on the strength of this flag, so it must not
    promise a repair for an invented timestamp — snapping one produces a
    well-formed clip of some other part of the talk."""
    report = lint_paths(
        FIXTURES / "clips.unsnapped.json",
        FIXTURES / "words.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert all(f.fixable for f in report.for_rule("time.off_boundary"))

    report = lint_paths(
        FIXTURES / "clips.hallucinated-time.json",
        FIXTURES / "words.json",
        criteria_path=FIXTURES / "criteria.yaml",
    )
    assert not any(f.fixable for f in report.for_rule("time.hallucinated"))
    assert set(FIXABLE_RULES) <= set(RULES)


# --------------------------------------------------------------------------
# the report itself
# --------------------------------------------------------------------------


def test_every_rule_the_linter_can_emit_is_documented() -> None:
    """RULES is what `ms lint` prints when asked what it checks. A rule that
    fires but is not listed is a rule nobody can look up."""
    source = (REPO / "makeshorts" / "select" / "lint.py").read_text()
    emitted = set(re.findall(r'report\.add\(\s*"([^"]+)"', source))
    assert emitted - set(RULES) == set()
    assert set(RULES) - emitted == set()


def test_findings_are_structured_and_printable() -> None:
    def mutate(data: dict) -> None:
        data["clips"][0]["speaker"] = "nobody"

    finding = broken(mutate).findings[0]
    assert finding.rule_id == "ref.unknown_speaker"
    assert finding.clip_id == "01-cac-payback-math"
    assert finding.severity is Severity.ERROR
    assert "ref.unknown_speaker" in str(finding)
    assert "[01-cac-payback-math]" in str(finding)


def test_report_format_summarises() -> None:
    assert lint(BASELINE).format() == "lint: clean"

    def mutate(data: dict) -> None:
        data["clips"][0]["speaker"] = "nobody"

    text = broken(mutate).format()
    assert "1 error(s), 0 warning(s)" in text


def test_transition_on_the_first_span_warns() -> None:
    """It cannot apply -- there is nothing before it -- but it must not block."""
    doc = base_doc()
    clip = doc.clips[0]
    spans = [s.model_copy(deep=True) for s in clip.spans]
    spans[0] = spans[0].model_copy(update={"transition": "dissolve"})
    doc = doc.model_copy(
        update={"clips": [clip.model_copy(update={"layout": spans})] + list(doc.clips[1:])}
    )
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.transition_on_first_span" in report.rule_ids()
    assert report.ok, "a meaningless transition must not block the render"


def test_a_transition_on_a_later_span_is_fine() -> None:
    doc = base_doc()
    multi = next((c for c in doc.clips if len(c.spans) > 1), None)
    if multi is None:
        pytest.skip("fixture has no multi-span clip")
    spans = [s.model_copy(deep=True) for s in multi.spans]
    spans[1] = spans[1].model_copy(update={"transition": "dissolve"})
    doc = doc.model_copy(
        update={
            "clips": [
                c.model_copy(update={"layout": spans}) if c.id == multi.id else c
                for c in doc.clips
            ]
        }
    )
    report = lint_doc(doc, words=WORDS, criteria=CRITERIA, silence=SILENCE)
    assert "ref.transition_on_first_span" not in report.rule_ids()
