"""PROMPT.md generation.

The tests that matter here are not "does it produce text". They are:

1. **No rubric prose is hardcoded.** `test_rubric_text_is_not_hardcoded` edits a
   criterion in a temp YAML, regenerates, and asserts the new wording appears
   while the old wording is gone. `test_no_rubric_prose_appears_in_the_source`
   attacks the same requirement from the other side by grepping prompt.py for
   any of the real rubric's own sentences. Between them, a criterion that got
   copied into a Python string cannot survive.

2. **The example is a valid clips.json.** A prompt containing an invalid
   example teaches the reader the wrong shape, and it would do so
   convincingly. So the fenced JSON block is parsed and validated against the
   real `ClipsDoc`, and checked against the rubric's own thresholds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from makeshorts.artifacts import IMPLICIT_FRAME_REGION_ID, RegionsDoc
from makeshorts.select.criteria import load_criteria
from makeshorts.select.prompt import (
    INLINE_TRANSCRIPT_MAX_CHARS,
    build_prompt,
    example_clips_doc,
    write_prompt,
)
from makeshorts.select.schema import Clip, ClipsDoc

REPO = Path(__file__).resolve().parents[2]
CRITERIA_YAML = REPO / "config" / "criteria.yaml"
FIXTURES = Path(__file__).resolve().parent / "prompt_fixtures"

TRANSCRIPT = "\n".join(
    f"[{m:02d}:{s:02d}.00] Sentence number {m * 60 + s} of the talk."
    for m in range(3)
    for s in range(0, 60, 15)
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def criteria():
    return load_criteria(CRITERIA_YAML)


@pytest.fixture
def regions() -> RegionsDoc:
    return RegionsDoc.model_validate(json.loads((FIXTURES / "regions.json").read_text()))


@pytest.fixture
def prompt(criteria, regions) -> str:
    return build_prompt(
        job_slug="acme-q3-webinar",
        criteria=criteria,
        transcript=TRANSCRIPT,
        regions=regions,
        source_path="jobs/acme-q3-webinar/source.mp4",
        duration=3612.44,
        resolution=(1920, 1080),
        criteria_file="config/criteria.yaml",
    )


def _example_json(text: str) -> dict:
    """The one fenced ```json block in the prompt, parsed."""
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    assert len(blocks) == 1, f"expected exactly one JSON example, found {len(blocks)}"
    return json.loads(blocks[0])


# --------------------------------------------------------------------------
# 1. The rubric is data, not a prompt string
# --------------------------------------------------------------------------


def test_rubric_text_is_not_hardcoded(tmp_path: Path, regions: RegionsDoc) -> None:
    """The load-bearing test of this module.

    Rewrite a criterion's question, its anchors, its weight, the score
    thresholds, and its id in a temp criteria.yaml. Every one of those changes
    must show up in the regenerated prompt, and the originals must be gone. If
    any rubric text is a literal in prompt.py, this fails.
    """
    raw = yaml.safe_load(CRITERIA_YAML.read_text())

    raw["rubric"][0]["id"] = "plectrum_energy"
    raw["rubric"][0]["weight"] = 9
    raw["rubric"][0]["question"] = (
        "Does the speaker reach for a plectrum within the opening bars?"
    )
    raw["rubric"][0]["anchors"] = {
        1: "No plectrum anywhere; the clip opens on tuning noise.",
        3: "A plectrum appears, but only once the second chorus has started.",
        5: "The plectrum is the first thing in frame and it is already moving.",
    }
    raw["scoring"]["vetoes"] = {"plectrum_energy": 5}
    raw["scoring"]["min_weighted_score"] = 4.62
    raw["gates"]["max_clips"] = 6
    raw["gates"]["hook_window"] = 7.5
    raw["gates"]["max_internal_silence"] = 0.9

    path = tmp_path / "criteria.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))

    text = build_prompt(
        job_slug="tuned",
        criteria=load_criteria(path),
        transcript=TRANSCRIPT,
        regions=regions,
        duration=3612.44,
        criteria_file=str(path),
    )

    # The new rubric arrived, verbatim.
    assert "Does the speaker reach for a plectrum within the opening bars?" in text
    assert "No plectrum anywhere; the clip opens on tuning noise." in text
    assert "The plectrum is the first thing in frame and it is already moving." in text
    assert "plectrum_energy" in text

    # The new numbers arrived too -- weights, thresholds and gates are data.
    assert "weight 9" in text
    assert "4.62" in text
    assert "max 6" in text
    assert "7.5" in text
    assert "0.9" in text

    # And the original rubric is nowhere in the output.
    assert "makes a scrolling stranger stop" not in text
    assert "The first sentence IS the hook" not in text
    assert "Starts mid-thought or in filler" not in text
    assert "hook_strength" not in text
    assert "3.8" not in text


def test_no_rubric_prose_appears_in_the_source(criteria) -> None:
    """The same requirement, checked against the file rather than the output.

    Every criterion's own words must be absent from prompt.py. A distinctive
    slice of each question and each anchor is enough to catch a copy-paste.
    """
    source = (REPO / "makeshorts" / "select" / "prompt.py").read_text()
    offenders = []
    for c in criteria.rubric:
        for kind, prose in [("question", c.question), *(("anchor %d" % k, v) for k, v in c.anchors.items())]:
            slice_ = " ".join(prose.split())[:40]
            if slice_ and slice_ in source:
                offenders.append(f"{c.id} {kind}: {slice_!r}")
    assert not offenders, (
        "rubric prose is hardcoded in prompt.py -- it belongs in criteria.yaml:\n  "
        + "\n  ".join(offenders)
    )


def test_every_rubric_id_question_and_anchor_is_rendered(prompt: str, criteria) -> None:
    for c in criteria.rubric:
        assert c.id in prompt, f"criterion id {c.id} missing from PROMPT.md"
        assert " ".join(c.question.split()) in prompt, f"{c.id} question missing"
        for score, anchor in c.anchors.items():
            assert " ".join(anchor.split()) in prompt, f"{c.id} anchor {score} missing"


def test_every_weight_and_the_weight_total_are_rendered(prompt: str, criteria) -> None:
    total = sum(c.weight for c in criteria.rubric)
    assert f"Weights total {total:g}" in prompt
    for c in criteria.rubric:
        assert f"weight {c.weight:g} of {total:g}" in prompt


def test_scoring_thresholds_and_vetoes_are_rendered(prompt: str, criteria) -> None:
    assert str(criteria.scoring.min_weighted_score) in prompt
    assert criteria.scoring.method.replace("_", " ") in prompt
    for cid, required in criteria.scoring.vetoes.items():
        assert re.search(rf"`{cid}` must be at least \*\*{required}\*\*", prompt), (
            f"veto {cid} >= {required} not stated as a threshold"
        )
    assert str(criteria.diversity.max_per_theme) in prompt


# --------------------------------------------------------------------------
# 2. Mechanical gates
# --------------------------------------------------------------------------


def test_every_mechanical_gate_is_rendered(prompt: str, criteria) -> None:
    g = criteria.gates
    assert f"{g.duration.min:g}–{g.duration.max:g} s" in prompt
    assert f"aim for {g.duration.target:g} s" in prompt
    assert f"max {g.max_internal_silence:g} s" in prompt
    assert f"min {g.min_separation:g} s" in prompt
    assert f"max {g.max_clips}" in prompt
    assert f"first {g.hook_window:g} s" in prompt
    assert f"within {g.boundary_tolerance:g} s" in prompt
    assert "sentence" in prompt.lower()
    assert "ms lint" in prompt


def test_gates_are_framed_as_a_gate_not_a_suggestion(prompt: str) -> None:
    section = prompt.split("## 3. Mechanical gates")[1].split("## 4.")[0]
    assert "rejects the file" in section
    assert "ms lint" in section


# --------------------------------------------------------------------------
# 3. Regions and layout
# --------------------------------------------------------------------------


def test_regions_are_listed_with_id_kind_rect_and_confidence(
    prompt: str, regions: RegionsDoc
) -> None:
    for r in regions.regions:
        assert f"`{r.id}`" in prompt
        assert r.kind in prompt
        if r.confidence is not None:
            assert f"{r.confidence:.2f}" in prompt
    # rects, normalized
    assert "[0, 0, 0.5, 1]" in prompt
    assert "[0.5, 0, 0.5, 0.5]" in prompt


def test_frame_region_is_always_documented(prompt: str) -> None:
    assert IMPLICIT_FRAME_REGION_ID in prompt
    assert "[0, 0, 1, 1]" in prompt


def test_regions_are_described_as_a_correctable_proposal(prompt: str) -> None:
    section = prompt.split("## 4. The source frame")[1].split("## 5.")[0]
    assert "proposes" in section
    assert "correct it" in section


def test_all_three_layout_modes_are_explained(prompt: str) -> None:
    section = prompt.split("### The three layout modes")[1].split("## 5.")[0]
    for mode in ["focus", "hero_inset", "stack"]:
        assert f"`{mode}`" in section
    assert "contain_blur" in section
    assert "cover" in section
    # `at` is the field most easily misread as absolute source time.
    assert "relative to the start of the clip" in section


def test_zero_detected_regions_is_handled(criteria) -> None:
    text = build_prompt(
        job_slug="bare",
        criteria=criteria,
        transcript=TRANSCRIPT,
        regions=RegionsDoc(grid="8x6", frames_sampled=120, regions=[]),
        duration=1800.0,
    )
    assert "no distinct regions" in text
    assert IMPLICIT_FRAME_REGION_ID in text
    # The example must still be a legal edit list against zero regions.
    doc = ClipsDoc.model_validate(_example_json(text))
    assert doc.source.regions == []
    assert doc.clips[0].visual_dependency is False
    for span in doc.clips[0].spans:
        assert IMPLICIT_FRAME_REGION_ID in doc.resolved_regions()


def test_regions_omitted_entirely_behaves_like_zero_regions(criteria) -> None:
    text = build_prompt(job_slug="bare", criteria=criteria, transcript=TRANSCRIPT)
    assert "no distinct regions" in text
    ClipsDoc.model_validate(_example_json(text))


# --------------------------------------------------------------------------
# 4. visual_dependency
# --------------------------------------------------------------------------


def test_visual_dependency_rule_is_stated_and_tied_to_lint(prompt: str) -> None:
    section = prompt.split("## 5. `visual_dependency`")[1].split("## 6.")[0]
    assert "visual_dependency: true" in section
    assert "slide" in section
    assert "lint" in section


# --------------------------------------------------------------------------
# 5. The example edit list
# --------------------------------------------------------------------------


def test_example_json_parses_and_validates_against_clipsdoc(prompt: str) -> None:
    doc = ClipsDoc.model_validate(_example_json(prompt))
    assert len(doc.clips) == 1


def test_example_covers_every_field_of_the_schema(prompt: str) -> None:
    """A documented shape that has quietly lost a field is worse than none."""
    data = _example_json(prompt)
    assert set(ClipsDoc.model_fields) <= set(data)
    assert set(Clip.model_fields) <= set(data["clips"][0])


def test_example_scores_every_rubric_criterion(prompt: str, criteria) -> None:
    scores = _example_json(prompt)["clips"][0]["why"]["scores"]
    assert set(scores) == set(criteria.ids)


def test_example_clip_would_survive_its_own_rubric(prompt: str, criteria) -> None:
    """The example must not teach a clip that `ms lint` would throw out."""
    clip = ClipsDoc.model_validate(_example_json(prompt)).clips[0]
    scores = {k: v.score for k, v in clip.why.scores.items()}

    assert criteria.failed_vetoes(scores) == []
    recomputed = criteria.weighted_score(scores)
    assert recomputed >= criteria.scoring.min_weighted_score
    assert abs(clip.why.weighted_score - recomputed) <= 0.01
    assert criteria.gates.duration.min <= clip.duration <= criteria.gates.duration.max


def test_example_uses_a_slide_region_when_it_claims_visual_dependency(
    prompt: str,
) -> None:
    from makeshorts.select.schema import layout_region_ids

    doc = ClipsDoc.model_validate(_example_json(prompt))
    clip = doc.clips[0]
    resolved = doc.resolved_regions()
    used = {rid for span in clip.spans for rid in layout_region_ids(span)}

    assert used <= set(resolved), "example layout references an unknown region"
    if clip.visual_dependency:
        assert any(resolved[rid].kind == "slide" for rid in used)


def test_example_drops_detection_only_region_fields(prompt: str) -> None:
    """regions.json carries confidence and motion; a confirmed edit list should
    not pretend those were editorial decisions."""
    for region in _example_json(prompt)["source"]["regions"]:
        assert "confidence" not in region
        assert "motion" not in region
        assert "change_points" not in region


def test_example_doc_construction_fails_loudly_if_the_schema_grows(criteria) -> None:
    """`example_clips_doc` builds real models, so a new required field on Clip
    breaks here rather than producing a plausible, invalid example."""
    doc = example_clips_doc(
        job_slug="j",
        criteria=criteria,
        regions=[],
        source_path="jobs/j/source.mp4",
        duration=None,
        resolution=(1920, 1080),
        criteria_file="config/criteria.yaml",
    )
    assert isinstance(doc, ClipsDoc)


# --------------------------------------------------------------------------
# 6. Anti-hallucination
# --------------------------------------------------------------------------


def test_anti_hallucination_instruction_is_present_and_specific(prompt: str) -> None:
    section = prompt.split("## 7. Do not invent")[1].split("## 8.")[0]
    assert "words.json" in section
    assert "verbatim" in section
    assert "source_text" in section
    assert "Never compute one." in section


# --------------------------------------------------------------------------
# 7. The transcript
# --------------------------------------------------------------------------


def test_small_transcript_is_inlined(prompt: str) -> None:
    assert TRANSCRIPT in prompt
    assert "Inlined below in full" in prompt


def test_large_transcript_becomes_a_pointer_with_a_format_sample(criteria) -> None:
    big = "\n".join(f"[00:{i:02d}:00.00] Line {i}." for i in range(20_000))
    assert len(big) > INLINE_TRANSCRIPT_MAX_CHARS

    text = build_prompt(
        job_slug="long-one",
        criteria=criteria,
        transcript=big,
        duration=20_000.0,
        transcript_filename="transcript.txt",
    )
    assert big not in text
    assert "Read `transcript.txt`" in text
    assert "[00:00:00.00] Line 0." in text  # the format sample survives
    assert len(text) < INLINE_TRANSCRIPT_MAX_CHARS


def test_empty_transcript_says_to_run_prepare(criteria) -> None:
    text = build_prompt(job_slug="empty", criteria=criteria, transcript="")
    assert "ms prepare" in text


# --------------------------------------------------------------------------
# 8. Writing into a job
# --------------------------------------------------------------------------


def test_write_prompt_writes_into_the_job_dir(tmp_path: Path, regions) -> None:
    from makeshorts.jobs import Job

    job = Job.create("acme", jobs_root=tmp_path)
    job.media_json.write_text(
        json.dumps(
            {
                "path": "jobs/acme/source.mp4",
                "duration": 3612.44,
                "resolution": [1920, 1080],
                "fps": 30.0,
                "has_audio": True,
                "audio_channels": 2,
                "sha256": "0" * 64,
            }
        )
    )
    job.regions_json.write_text(regions.model_dump_json())
    job.transcript_txt.write_text(TRANSCRIPT)

    path = write_prompt(job, criteria_path=CRITERIA_YAML)

    assert path == job.prompt_md
    text = path.read_text()
    assert text.startswith("# Editorial brief — `acme`")
    assert "cam_a" in text
    ClipsDoc.model_validate(_example_json(text))


def test_write_prompt_tolerates_missing_regions_json(tmp_path: Path) -> None:
    from makeshorts.jobs import Job

    job = Job.create("bare", jobs_root=tmp_path)
    job.media_json.write_text(
        json.dumps(
            {
                "path": "jobs/bare/source.mp4",
                "duration": 900.0,
                "resolution": [1280, 720],
                "fps": 25.0,
                "has_audio": True,
                "audio_channels": 1,
                "sha256": "1" * 64,
            }
        )
    )
    job.transcript_txt.write_text(TRANSCRIPT)

    text = write_prompt(job, criteria_path=CRITERIA_YAML).read_text()
    assert "no distinct regions" in text
    assert "1280×720" in text
