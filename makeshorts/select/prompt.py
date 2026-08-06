"""PROMPT.md — the editorial brief, generated from the rubric.

This module renders `config/criteria.yaml` into the document a human (or Claude
Code) reads before writing `clips.json`. It is the entire interface between the
rubric and the editorial act.

**The hard rule of this file: not one word of rubric prose may live here.**
Every criterion question, every anchor, every weight, and every threshold is
read from `criteria.py` and formatted. The point of the design is that the
definition of "compelling" is explicit and tunable in one YAML file rather than
buried in a prompt string -- if you edit a criterion and PROMPT.md does not
change, the seam has broken. `tests/select/test_prompt.py` proves this by
regenerating against a modified rubric and asserting the old text is gone.

Structural scaffolding ("Output valid JSON matching this shape") is a literal
here, and that is fine. It is not the rubric; it is the paper the rubric is
printed on.

Two further things are derived rather than written:

* the documented field list comes from `schema.py` via a real `ClipsDoc` that
  is constructed, validated, and dumped -- so a prompt containing an invalid
  example is impossible, and a new required field cannot be silently omitted;
* the 1-5 score range comes from the constraints on `CriterionScore.score`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from makeshorts.artifacts import IMPLICIT_FRAME_REGION_ID, Region, RegionsDoc
from makeshorts.select.criteria import (
    DEFAULT_CRITERIA_PATH,
    Criteria,
    Criterion,
    load_criteria,
)
from makeshorts.select.schema import (
    Audio,
    Captions,
    Clip,
    ClipsDoc,
    CriteriaRef,
    CriterionScore,
    FocusLayout,
    HeroInsetLayout,
    OutputSpec,
    RejectedAlternative,
    SourceSpec,
    Why,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from makeshorts.jobs import Job

__all__ = [
    "INLINE_TRANSCRIPT_MAX_CHARS",
    "build_prompt",
    "example_clips_doc",
    "write_prompt",
]


# A transcript under this many characters is pasted into PROMPT.md whole;
# anything larger becomes a pointer to transcript.txt plus a format sample.
# ~120k characters is roughly 30k tokens -- an hour of talk fits comfortably
# alongside the rubric, while a three-hour recording would crowd out the
# instructions it is meant to be read against.
INLINE_TRANSCRIPT_MAX_CHARS = 120_000

# How many lines of a large transcript to show as a format sample.
_SAMPLE_LINES = 12

# Fields on a detected Region that are detection telemetry, not part of the
# confirmed edit list. Shown in the regions table, omitted from the example.
_DETECTION_ONLY_REGION_FIELDS = {"motion", "change_points", "confidence"}

# Used for the example clip's timing only when the source duration is unknown.
_FALLBACK_EXAMPLE_START = 1284.10


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _num(value: float) -> str:
    """`3.0` -> `3`, `1.2` -> `1.2`. Thresholds read better without noise."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _flow(text: str) -> str:
    """Collapse a YAML folded scalar into one paragraph."""
    return " ".join(text.split())


def _cell(text: str) -> str:
    """Make a value safe to drop into a markdown table cell."""
    return _flow(text).replace("|", r"\|")


def _rect(rect: tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{v:.3g}" for v in rect) + "]"


def _score_range() -> tuple[int, int]:
    """The scoring scale, read off the schema rather than asserted here."""
    lo, hi = 1, 5
    for meta in CriterionScore.model_fields["score"].metadata:
        if getattr(meta, "ge", None) is not None:
            lo = meta.ge
        if getattr(meta, "le", None) is not None:
            hi = meta.le
    return int(lo), int(hi)


# --------------------------------------------------------------------------
# The example edit list
# --------------------------------------------------------------------------


def _example_scores(criteria: Criteria) -> dict[str, int]:
    """Scores for the example clip that actually clear the rubric's own bars.

    Derived, never written down: heavier criteria get top marks, vetoes get at
    least what they demand, and the whole thing is nudged up until it clears
    `min_weighted_score`. An example that would be rejected by `ms lint`
    teaches the wrong lesson.
    """
    lo, hi = _score_range()
    heaviest = max(c.weight for c in criteria.rubric)
    scores: dict[str, int] = {}
    for c in criteria.rubric:
        base = hi if c.weight >= heaviest else hi - 1
        required = criteria.scoring.vetoes.get(c.id, lo)
        scores[c.id] = max(min(base, hi), min(required, hi))

    # Raise the cheapest scores first until the weighted mean clears the bar.
    order = sorted(criteria.rubric, key=lambda c: c.weight)
    for c in order:
        if criteria.weighted_score(scores) >= criteria.scoring.min_weighted_score:
            break
        scores[c.id] = hi
    return scores


def _example_evidence(criterion: Criterion, score: int) -> str:
    """Evidence text for the example.

    Deliberately a placeholder rather than invented prose: `evidence` is
    supposed to be a quote from *this* transcript, and a plausible-looking
    fabricated one in the example is an invitation to fabricate.
    """
    return (
        f"<the words in this clip that make it a {score} for {criterion.id} — "
        f"quote them>"
    )


def _example_layout_and_dependency(
    regions: list[Region],
    *,
    clip_duration: float,
) -> tuple[list[FocusLayout | HeroInsetLayout], bool, str | None]:
    """Pick an example layout that is legal for the regions this job has.

    With a speaker and a slide region the example shows the interesting case: a
    mid-clip layout change and `visual_dependency: true` backed by a slide
    region. With nothing detected it degenerates to the whole frame, which is
    exactly the fallback the region model is designed to make unremarkable.
    """
    speaker = next((r for r in regions if r.kind == "speaker"), None)
    slide = next((r for r in regions if r.kind == "slide"), None)
    fallback = next((r for r in regions if r.kind == "unknown"), None)
    hero_at = round(clip_duration * 0.26, 2)

    if speaker and slide:
        return (
            [
                FocusLayout(at=0.0, region=speaker.id, fit="cover"),
                HeroInsetLayout(
                    at=hero_at,
                    hero=slide.id,
                    inset=speaker.id,
                    inset_corner="bottom_right",
                    inset_scale=0.28,
                ),
            ],
            True,
            speaker.id,
        )

    only = speaker or fallback or (regions[0] if regions else None)
    region_id = only.id if only else IMPLICIT_FRAME_REGION_ID
    fit = "cover" if only else "contain_blur"
    return ([FocusLayout(at=0.0, region=region_id, fit=fit)], False, only.id if only else None)


def example_clips_doc(
    *,
    job_slug: str,
    criteria: Criteria,
    regions: list[Region],
    source_path: str,
    duration: float | None,
    resolution: tuple[int, int],
    criteria_file: str,
) -> ClipsDoc:
    """A complete, valid `clips.json` with one clip in it.

    Built out of the real pydantic models so the example in PROMPT.md cannot
    drift from the schema: a new required field breaks this function loudly
    instead of producing a prompt that documents a file shape which no longer
    validates.
    """
    target = criteria.gates.duration.target

    if duration and duration > target:
        start = round(min(duration * 0.35, duration - target), 2)
    elif duration:
        start = 0.0
    else:
        start = _FALLBACK_EXAMPLE_START
    end = round(start + target, 2)
    source_duration = duration if duration else round(end + 1800.0, 2)

    scores = _example_scores(criteria)
    layout, visual_dependency, speaker = _example_layout_and_dependency(
        regions, clip_duration=target
    )

    clip = Clip(
        id="01-cac-payback-math",
        title="The CAC payback number nobody checks",
        start=start,
        end=end,
        source_text=(
            "<the transcript text of this exact span, copied verbatim — "
            "not paraphrased, not tidied up>"
        ),
        why=Why(
            one_line="<one sentence: why this clip and not the ten seconds either side of it>",
            theme="unit-economics",
            scores={
                cid: CriterionScore(
                    score=score,
                    evidence=_example_evidence(criteria.by_id(cid), score),
                )
                for cid, score in scores.items()
            },
            weighted_score=round(criteria.weighted_score(scores), 2),
            rejected_alternatives=[
                RejectedAlternative(
                    start=round(max(0.0, start - 94.1), 2),
                    end=round(max(0.0, start - 44.1), 2),
                    reason="<why the nearby candidate lost, in rubric terms>",
                )
            ],
        ),
        visual_dependency=visual_dependency,
        speaker=speaker,
        layout=layout if len(layout) > 1 else layout[0],
        captions=Captions(),
        audio=Audio(),
    )

    return ClipsDoc(
        job=job_slug,
        source=SourceSpec(
            path=source_path,
            duration=source_duration,
            resolution=resolution,
            regions=[
                Region(id=r.id, kind=r.kind, rect=r.rect, label=r.label) for r in regions
            ],
        ),
        output=OutputSpec(),
        criteria_ref=CriteriaRef(
            file=criteria_file,
            version=criteria.version,
            sha256=criteria.sha256,
        ),
        clips=[clip],
    )


def _example_json(doc: ClipsDoc) -> str:
    """Serialise the example, and refuse to emit one that has lost a field.

    Nulls are kept rather than dropped: §6 promises a complete field
    inventory, and `"speaker": null` documents an optional field where an
    absent key would just look like an oversight.
    """
    data = doc.model_dump(
        mode="json",
        exclude={"source": {"regions": {"__all__": _DETECTION_ONLY_REGION_FIELDS}}},
    )

    missing_doc = set(ClipsDoc.model_fields) - set(data)
    missing_clip = set(Clip.model_fields) - set(data["clips"][0])
    if missing_doc or missing_clip:
        raise RuntimeError(
            "the PROMPT.md example no longer covers every field of the schema — "
            f"missing document fields {sorted(missing_doc)}, clip fields "
            f"{sorted(missing_clip)}. Update example_clips_doc() in "
            "makeshorts/select/prompt.py."
        )
    return _compact_number_arrays(json.dumps(data, indent=2))


# A JSON array of nothing but numbers, however `indent` chose to break it up.
_NUMBER_ARRAY = re.compile(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*)\s*\]")


def _compact_number_arrays(text: str) -> str:
    """Put `[0.0, 0.0, 0.5, 1.0]` back on one line.

    `json.dumps(indent=2)` explodes every array, which turns a rect into four
    lines and makes the example harder to read than the thing it documents.
    Rects and resolutions are the only arrays of pure numbers in clips.json.
    """
    return _NUMBER_ARRAY.sub(
        lambda m: "[" + ", ".join(part.strip() for part in m.group(1).split(",")) + "]",
        text,
    )


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def _section_rubric(criteria: Criteria, criteria_file: str) -> list[str]:
    lo, hi = _score_range()
    total = sum(c.weight for c in criteria.rubric)
    out = [
        "## 1. The rubric",
        "",
        f"Score every clip you propose against **all {len(criteria.rubric)} criteria**, "
        f"{lo}-{hi}, and cite the evidence for each score. These questions are not "
        f"advice — `ms lint` requires one entry per id below in every clip's "
        "`why.scores`, and rejects the file if one is missing.",
        "",
        f"Source: `{criteria_file}` version `{criteria.version}` "
        f"(sha256 `{criteria.sha256[:12]}…`). Weights total {_num(total)}.",
        "",
    ]
    for c in criteria.rubric:
        share = 100.0 * c.weight / total if total else 0.0
        out += [
            f"### `{c.id}` — weight {_num(c.weight)} of {_num(total)} ({share:.0f}%)",
            "",
            _flow(c.question),
            "",
        ]
        if c.anchors:
            out.append("What the scores mean:")
            out.append("")
            for score in sorted(c.anchors):
                out.append(f"- **{score}** — {_flow(c.anchors[score])}")
            out.append("")
    return out


def _section_scoring(criteria: Criteria) -> list[str]:
    lo, hi = _score_range()
    total = sum(c.weight for c in criteria.rubric)
    terms = " + ".join(f"{_num(c.weight)}×{c.id}" for c in criteria.rubric)
    method = criteria.scoring.method.replace("_", " ")

    out = [
        "## 2. How the scores decide",
        "",
        f"The method is a **{method}**. `ms lint` recomputes it from `why.scores` "
        "and compares it against the `weighted_score` you write, so the arithmetic "
        "has to be honest:",
        "",
        "```",
        f"weighted_score = ({terms}) / {_num(total)}",
        "```",
        "",
        "A clip ships only if it clears **both** of these:",
        "",
        f"- **Weighted mean at least {_num(criteria.scoring.min_weighted_score)}.** "
        f"Below that, the clip is not proposed at all.",
    ]
    if criteria.scoring.vetoes:
        out.append(
            "- **Every veto met.** A veto is a single-criterion floor: failing one "
            "kills the clip no matter how well it scores elsewhere."
        )
        out.append("")
        for cid, required in criteria.scoring.vetoes.items():
            out.append(
                f"  - `{cid}` must be at least **{required}** "
                f"(of {hi}) — a {required - 1} here is fatal."
            )
    out += [
        "",
        f"Also: **at most {criteria.diversity.max_per_theme} clips may share a "
        "`why.theme`.** Label the theme honestly; lint enforces the cap and will "
        "reject the whole file rather than pick which one to drop.",
        "",
        "Record the clips you considered and rejected in `why.rejected_alternatives` "
        "with the criterion that killed them. That record is most of what makes this "
        "file worth reviewing.",
        "",
    ]
    return out


def _section_gates(criteria: Criteria) -> list[str]:
    g = criteria.gates
    out = [
        "## 3. Mechanical gates — you will be checked against these",
        "",
        "These are not editorial judgements and there is nothing to weigh. "
        "`ms lint` checks each one against `words.json` and `silence.json` and "
        "**rejects the file** on any violation.",
        "",
        "| Gate | Value | What it means for you |",
        "| --- | --- | --- |",
        f"| Duration | {_num(g.duration.min)}–{_num(g.duration.max)} s "
        f"(aim for {_num(g.duration.target)} s) | `end - start` outside this range "
        "is rejected. |",
    ]
    if g.snap_to_word_boundaries:
        out.append(
            f"| Word boundaries | within {_num(g.boundary_tolerance)} s | Every "
            "`start` and `end` must land on a real word edge in `words.json`. |"
        )
    if g.start_on_sentence_start:
        out.append(
            "| Start of sentence | required | The first word of the clip must begin "
            "a sentence. Opening mid-sentence sounds like a mistake. |"
        )
    if g.end_on_sentence_end:
        out.append(
            "| End of sentence | required | The last word must end one. Cutting "
            "mid-sentence sounds like a truncation. |"
        )
    out += [
        f"| Internal silence | max {_num(g.max_internal_silence)} s | No dead-air "
        "span longer than this anywhere inside the clip. |",
        f"| Separation | min {_num(g.min_separation)} s | Between the midpoints of "
        "any two clips. Spread the set across the talk. |",
        f"| Clip count | max {g.max_clips} | Propose fewer if fewer earn it. A weak "
        f"{g.max_clips}th clip costs you more than the empty slot. |",
        f"| Hook window | first {_num(g.hook_window)} s | The span the hook criterion "
        "is judged on. |",
    ]
    if g.require_slide_region_when_visually_dependent:
        out.append(
            "| Visual dependency | enforced | `visual_dependency: true` requires a "
            "slide region in the layout — see §5. |"
        )
    out += [
        "",
        f"Clips must not overlap, and clip ids must be unique and of the form "
        f"`01-kebab-case-slug` — the number orders them, the slug names them, and "
        "together they become the output filename.",
        "",
        f"You do not need to add handles: `ms lint --fix` applies "
        f"{_num(g.pad_in)} s before and {_num(g.pad_out)} s after, snapping to "
        "boundaries. Give it the true speech boundaries.",
        "",
    ]
    return out


def _section_regions(regions: list[Region], resolution: tuple[int, int]) -> list[str]:
    # A region carries the name of the file it was measured against as soon as
    # a job has more than one. Several frame-aligned exports of one meeting is
    # the case that produces it -- the sharp slides and the usable face are in
    # different files -- and it changes what the editorial step must write, so
    # it is stated here rather than left to be inferred from the ids.
    source_names = list(dict.fromkeys(r.source for r in regions if r.source))
    multi = len(source_names) > 1

    which = "The primary source is" if multi else "The source is"
    out = [
        "## 4. The source frame: regions and layout",
        "",
        f"{which} {resolution[0]}×{resolution[1]} and the output is vertical, "
        "so something has to be chosen out of the frame. Regions are how you say "
        "what.",
        "",
        "A region is a named rectangle, normalized `[x, y, w, h]` in 0..1 from the "
        "top-left. Region detection **proposes** the list below by looking at which "
        "parts of the frame move; it is a starting point, not a decision. If a "
        "`rect` is wrong or a `kind` is mislabelled, correct it — confidence "
        "figures are there to tell you which ones to distrust.",
        "",
    ]

    if multi:
        listed = ", ".join(f"`{n}`" for n in source_names)
        out += [
            f"**This job has {len(source_names)} sources: {listed}.** They are "
            "frame-aligned views of one recording — the same meeting exported "
            "several times over, sharing a clock and an audio track, so a timestamp "
            "means the same instant in every one of them. Typically one file holds "
            "the usable face and another holds the sharp slides.",
            "",
            "Two things follow. First, the edit list declares them as a `sources` "
            "map — `\"sources\": {\"" + source_names[0] + '": {"path": …, "duration": '
            "…, \"resolution\": …, \"regions\": […]}, …}` — in place of the single "
            "`source` object the example in §6 shows; the first entry is primary. "
            "Each region lives under the source it was measured against, and every "
            "region id below is already namespaced with that source's name. "
            "Second, **a layout may mix sources**: "
            '`{"mode": "hero_inset", "hero": "<a slide region of one source>", '
            '"inset": "<a camera region of another>"}` is exactly what having '
            "several views is for, and is the layout most of these clips want.",
            "",
        ]

    if regions:
        header = "| id | kind | label | rect `[x, y, w, h]` | confidence |"
        divider = "| --- | --- | --- | --- | --- |"
        if multi:
            header = "| id | source | kind | label | rect `[x, y, w, h]` | confidence |"
            divider = "| --- | --- | --- | --- | --- | --- |"
        out += [header, divider]
        for r in regions:
            conf = "—" if r.confidence is None else f"{r.confidence:.2f}"
            label = _cell(r.label) if r.label else "—"
            source = f"`{r.source}` | " if multi else ""
            out.append(f"| `{r.id}` | {source}{r.kind} | {label} | `{_rect(r.rect)}` | {conf} |")
        frame_source = f"`{source_names[0]}` | " if multi else ""
        out.append(
            f"| `{IMPLICIT_FRAME_REGION_ID}` | {frame_source}— | the whole source frame | "
            "`[0, 0, 1, 1]` | implicit, always present |"
        )
        if multi:
            for name in source_names:
                out.append(
                    f"| `{name}_frame` | `{name}` | — | the whole frame of source "
                    f"`{name}` | `[0, 0, 1, 1]` | implicit, always present |"
                )
        out.append("")
    else:
        out += [
            "**Detection found no distinct regions in this source.** That is a "
            "normal outcome for a single full-frame camera or a full-screen "
            "screenshare — it is not an error and it does not block you. You have "
            f"`{IMPLICIT_FRAME_REGION_ID}`, and you may still declare regions by "
            "hand in `source.regions` if you can see from the video that the frame "
            "has distinct areas worth naming.",
            "",
        ]

    out += [
        f"`{IMPLICIT_FRAME_REGION_ID}` always exists as a region id, whether or not "
        "anything was detected, and always means the whole source frame "
        "`[0, 0, 1, 1]`. It needs no entry in the declared regions. "
        f"`{{\"mode\": \"focus\", \"region\": \"{IMPLICIT_FRAME_REGION_ID}\", "
        '"fit": "contain_blur"}` is the honest fallback whenever no crop is better '
        "than a guessed one.",
        "",
    ]
    if multi:
        out += [
            f"With several sources, `{IMPLICIT_FRAME_REGION_ID}` means the whole frame "
            f"of the primary source (`{source_names[0]}`), and `<name>_frame` means the "
            "whole frame of the source you name — those are implicit too, and are "
            "often all you need: a view that is nothing but the shared screen wants "
            "`slides_frame`, not a detected rectangle inside it.",
            "",
        ]
    out += [
        "### The three layout modes",
        "",
        "Each is a claim about what the viewer needs to be looking at, not a "
        "picture-in-picture preference:",
        "",
        "- **`focus`** — one region fills the vertical frame. Use it when one thing "
        "carries the moment. `fit: \"cover\"` crops to fill; "
        "`fit: \"contain_blur\"` fits the whole region in and softens the "
        "remainder, which is what you want for a wide region you must not crop "
        "(a full slide, a two-shot).",
        "- **`hero_inset`** — one region fills the frame, a second sits small in a "
        "corner over it. Use it when the words and the visual are both load-bearing: "
        "a slide as `hero` with the speaker as `inset` keeps the face present "
        "without letting it compete.",
        "- **`stack`** — two or more regions tiled vertically, each fit to its band. "
        "Use it for genuine dialogue, where losing either face loses the exchange. "
        "It costs the most vertical space per region, so it should earn it.",
        "",
        "`layout` may be a single object, or an ordered list of timed spans. In list "
        "form each span carries `at`, **seconds relative to the start of the clip** "
        "(not the source), the first span must be at `0.0`, and spans must be in "
        "increasing order. Change layout when what matters on screen changes — when "
        "the speaker turns to a slide, not on a timer.",
        "",
    ]
    return out


def _section_visual_dependency(criteria: Criteria) -> list[str]:
    out = [
        "## 5. `visual_dependency` — say when the words need the picture",
        "",
        "Set `visual_dependency: true` when the clip's words lean on what is on "
        'screen: "as you can see here", "this second column", "the shape of this '
        'curve", a number read off a chart rather than said aloud.',
        "",
        "This is an editorial claim about meaning, and it is the one such claim the "
        "pipeline can check.",
        "",
    ]
    if criteria.gates.require_slide_region_when_visually_dependent:
        out += [
            "**If you set it true, the clip's layout must include a region of "
            "`kind: \"slide\"` — `ms lint` rejects the clip otherwise.** The two "
            "ways to satisfy it are to show the slide (`hero_inset` with the slide "
            "as hero, or `focus` on it for the span that needs it), or to conclude "
            "the clip does not really need it and set the flag false.",
            "",
            "What you must not do is set it false to get past the check on a clip "
            "that is incomprehensible without the screen. The honest fix is to "
            "score it accordingly or choose a different span.",
            "",
        ]
        # Name the criterion only if it is actually in the rubric. Referring to
        # it by a hardcoded id would go stale the moment someone renames or
        # removes it -- the same failure this module exists to avoid.
        if "standalone" in criteria.ids:
            out[-1] = (
                "Such a clip has a `standalone` problem; setting the flag false "
                "hides it rather than fixing it."
            )
            out.append("")
    return out


def _section_shape(doc: ClipsDoc, criteria: Criteria) -> list[str]:
    lo, hi = _score_range()
    ids = ", ".join(f"`{i}`" for i in criteria.ids)
    return [
        "## 6. The file you write",
        "",
        "Write `clips.json` in this job directory. Output valid JSON matching this "
        "shape — no comments, no trailing commas, no extra keys (unknown keys are "
        "rejected, not ignored).",
        "",
        "Field notes, because a few are easy to get subtly wrong:",
        "",
        "- `start` / `end` — **absolute seconds in the source**, not relative to "
        "anything, and not `mm:ss`.",
        "- `source_text` — the transcript of that exact span, copied verbatim.",
        f"- `why.scores` — exactly one entry per rubric id: {ids}. Each is an "
        f"integer {lo}-{hi} plus `evidence`.",
        "- `why.weighted_score` — computed with the formula in §2, to two decimals.",
        "- `layout[].at` — seconds **relative to the clip start**.",
        "- `speaker` — which region is talking, or `null`.",
        "- `captions.style` and `captions.position` are *names*. Resolution is the "
        "renderer's business; nothing about how a clip is encoded belongs in this "
        "file.",
        "- `criteria_ref` — copy it exactly as shown; it records which rubric "
        "produced this edit list.",
        "",
        "The values below are illustrative and every one of them is meant to be "
        "replaced — but the shape, the key names, and the score keys are exact:",
        "",
        "```json",
        _example_json(doc),
        "```",
        "",
    ]


def _section_no_hallucination(criteria: Criteria) -> list[str]:
    g = criteria.gates
    return [
        "## 7. Do not invent a timestamp or a quote",
        "",
        "**This is the failure mode the whole pipeline exists to catch.** A clip "
        "with a plausible, confident, invented `start` renders 45 seconds of the "
        "wrong thing, and nothing downstream notices. So it is checked "
        "mechanically, not trusted:",
        "",
        f"- Every `start` and `end` is matched against a real word edge in "
        f"`words.json`, within {_num(g.boundary_tolerance)} s. A timestamp you "
        "estimated, interpolated, or rounded to something tidy will not match.",
        "- `source_text` is compared against the words `words.json` holds for that "
        "span, verbatim after normalization. Paraphrasing fails. Tidying up "
        "grammar fails. Merging two sentences that were not adjacent fails.",
        "",
        "So, concretely:",
        "",
        "1. **Copy timestamps from the transcript. Never compute one.** If you want "
        "a clip to begin at a particular word, use that word's timestamp as written.",
        "2. **Copy `source_text` from the transcript by selection**, from the first "
        "word to the last. Do not retype it from memory.",
        "3. **Every quote in an `evidence` field must appear in the clip's own span.** "
        "Evidence quoting something outside the clip is evidence for a different clip.",
        "4. If you cannot find a span that genuinely clears the bar, **propose fewer "
        f"clips**. {g.max_clips} is a ceiling, not a target, and an empty slot costs "
        "nothing. Inventing a clip to fill one costs everything.",
        "",
        "A file that fails lint is not a disaster — it is the system working, and "
        "the message will name the clip and the rule. Fix it and re-run.",
        "",
    ]


def _section_transcript(
    transcript: str,
    *,
    transcript_filename: str,
    inline_max_chars: int,
) -> list[str]:
    text = transcript.rstrip("\n")
    lines = text.splitlines()
    out = [
        "## 8. The transcript",
        "",
        f"Timestamped, produced mechanically from the audio. `words.json` in the "
        "same directory holds the same words with per-word timings and is the "
        "authority every timestamp is checked against; this is the readable view "
        "of it.",
        "",
    ]

    if not text:
        out += [
            f"**`{transcript_filename}` is empty or missing.** Run `ms prepare` "
            "before writing an edit list — there is nothing to select from until "
            "the transcript exists.",
            "",
        ]
        return out

    if len(text) <= inline_max_chars:
        out += [
            f"Inlined below in full ({len(lines):,} lines, {len(text):,} characters) "
            "so it can be read against the rubric without leaving this file.",
            "",
            "```",
            text,
            "```",
            "",
        ]
        return out

    sample = "\n".join(lines[:_SAMPLE_LINES])
    out += [
        f"**Read `{transcript_filename}` in this directory.** It is "
        f"{len(text):,} characters over {len(lines):,} lines — too large to inline "
        "here without burying the rubric it is supposed to be read against. Open it "
        "directly; read all of it before choosing, because the best moments in a "
        "long talk are rarely in the first third.",
        "",
        f"The first {min(_SAMPLE_LINES, len(lines))} lines, so you know the format "
        "and the timestamps you will be copying:",
        "",
        "```",
        sample,
        "```",
        "",
    ]
    return out


def _section_checklist(criteria: Criteria) -> list[str]:
    g = criteria.gates
    return [
        "## Before you save",
        "",
        "- [ ] Every `start`/`end` was copied from the transcript, not estimated.",
        "- [ ] Every `source_text` was copied verbatim from the span it claims.",
        f"- [ ] Every clip is {_num(g.duration.min)}–{_num(g.duration.max)} s, starts "
        "at a sentence start and ends at a sentence end.",
        f"- [ ] Every clip has all {len(criteria.rubric)} scores with real evidence, "
        "and `weighted_score` is the actual arithmetic.",
        f"- [ ] Every clip clears {_num(criteria.scoring.min_weighted_score)} and "
        "every veto.",
        f"- [ ] No more than {criteria.diversity.max_per_theme} clips share a theme; "
        f"midpoints are at least {_num(g.min_separation)} s apart.",
        "- [ ] Every region id used in a layout exists; "
        "`visual_dependency` clips show a slide.",
        "- [ ] You wrote down what you rejected and why.",
        "",
        "Then run `ms lint <slug>`. Nothing renders until it passes.",
        "",
    ]


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def build_prompt(
    *,
    job_slug: str,
    criteria: Criteria,
    transcript: str,
    regions: RegionsDoc | list[Region] | None = None,
    source_path: str = "",
    duration: float | None = None,
    resolution: tuple[int, int] = (1920, 1080),
    criteria_file: str = str(DEFAULT_CRITERIA_PATH),
    transcript_filename: str = "transcript.txt",
    inline_transcript_max_chars: int = INLINE_TRANSCRIPT_MAX_CHARS,
) -> str:
    """Render PROMPT.md.

    Every rubric word in the result comes from `criteria`. Everything else is
    scaffolding: what the deliverable is, how the file is shaped, and what will
    be checked.
    """
    if isinstance(regions, RegionsDoc):
        region_list = list(regions.regions)
    else:
        region_list = list(regions or [])

    source = source_path or f"jobs/{job_slug}/source.mp4"
    doc = example_clips_doc(
        job_slug=job_slug,
        criteria=criteria,
        regions=region_list,
        source_path=source,
        duration=duration,
        resolution=resolution,
        criteria_file=criteria_file,
    )

    lines: list[str] = [
        f"# Editorial brief — `{job_slug}`",
        "",
        "You are choosing which moments of a long recording deserve to be short "
        "vertical clips. Everything mechanical has already been done: the audio is "
        "transcribed with per-word timings, dead air is measured, and the frame is "
        "divided into regions. What is left is the judgement, which is the only "
        "part of this that is not deterministic.",
        "",
        "**Your deliverable is one file: `clips.json` in this job directory.** It is "
        "not a message back to anyone — it is an artifact a person will open, read, "
        "disagree with, edit by hand, and re-render from without re-running any of "
        "this. Write it for that reader. The `why` block on each clip is not "
        "paperwork; it is the argument they will be arguing with, and a score "
        "without evidence is an assertion rather than a case.",
        "",
        "Read the whole of this file before you start. The rubric is the standard, "
        "the gates are non-negotiable, and §7 is the part that gets people.",
        "",
        "---",
        "",
    ]

    lines += _section_rubric(criteria, criteria_file)
    lines += ["---", ""]
    lines += _section_scoring(criteria)
    lines += ["---", ""]
    lines += _section_gates(criteria)
    lines += ["---", ""]
    lines += _section_regions(region_list, resolution)
    lines += ["---", ""]
    lines += _section_visual_dependency(criteria)
    lines += ["---", ""]
    lines += _section_shape(doc, criteria)
    lines += ["---", ""]
    lines += _section_no_hallucination(criteria)
    lines += ["---", ""]
    lines += _section_transcript(
        transcript,
        transcript_filename=transcript_filename,
        inline_max_chars=inline_transcript_max_chars,
    )
    lines += ["---", ""]
    lines += _section_checklist(criteria)

    return "\n".join(lines).rstrip("\n") + "\n"


def write_prompt(
    job: Job,
    *,
    criteria: Criteria | None = None,
    criteria_path: str | Path = DEFAULT_CRITERIA_PATH,
    inline_transcript_max_chars: int = INLINE_TRANSCRIPT_MAX_CHARS,
) -> Path:
    """Generate PROMPT.md for a prepared job and write it into the job dir.

    Missing `regions.json` is tolerated -- a source with no detected regions is
    a normal outcome, and the brief says so and falls back to the whole frame.
    """
    from makeshorts.jobs import StageNotRun

    crit = criteria if criteria is not None else load_criteria(criteria_path)

    try:
        regions_doc: RegionsDoc | None = job.load_regions()
    except StageNotRun:
        regions_doc = None

    media = job.load_media()
    transcript = job.transcript_txt.read_text() if job.transcript_txt.exists() else ""

    text = build_prompt(
        job_slug=job.slug,
        criteria=crit,
        transcript=transcript,
        regions=regions_doc,
        source_path=str(media.path),
        duration=media.duration,
        resolution=media.resolution,
        criteria_file=str(criteria_path),
        transcript_filename=job.transcript_txt.name,
        inline_transcript_max_chars=inline_transcript_max_chars,
    )
    job.root.mkdir(parents=True, exist_ok=True)
    job.prompt_md.write_text(text)
    return job.prompt_md
