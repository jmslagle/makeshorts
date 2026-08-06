"""Several frame-aligned views of one recording, prepared as one job.

The case is a Zoom cloud recording: the same meeting exported as an
active-speaker camera, a shared-screen render and a gallery view, all the same
length, all sharing one audio track. What has to be true of that job is the
subject of this file -- every view probed and region-detected, exactly one of
them transcribed, and one regions.json in which it is unambiguous which
rectangle belongs to which file.

Region detection is stubbed here for the same reason transcription is stubbed
in `test_run.py`: whether the tile heuristic finds a slide in `smptebars` is
`test_regions.py`'s question, and asserting on real detections would make these
tests fail for reasons that have nothing to do with the wiring under test.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from types import ModuleType

import pytest

from makeshorts.artifacts import Region, RegionsDoc
from makeshorts.jobs import Job
from makeshorts.prepare import run as run_mod

from .conftest import evenly_timed, requires_ffmpeg

pytestmark = requires_ffmpeg

TRANSCRIPT = "Eighteen months is the number. Nobody ever checks it."

REGION_ID = re.compile(r"^[a-z][a-z0-9_]*$")

CRITERIA_YAML = pathlib.Path(__file__).resolve().parents[2] / "config" / "criteria.yaml"


@pytest.fixture
def views(tmp_path, av_file, slides_view_file) -> dict[str, object]:
    """A job with two named sources ingested, camera first."""
    job = Job.create("two-views", tmp_path / "jobs")
    cam = job.source_for(av_file, "cam")
    cam.write_bytes(av_file.read_bytes())
    slides = job.source_for(slides_view_file, "slides")
    slides.write_bytes(slides_view_file.read_bytes())
    return {"job": job, "sources": {"cam": cam, "slides": slides}}


@pytest.fixture
def stub_transcribe(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return evenly_timed(TRANSCRIPT)

    monkeypatch.setattr(run_mod.transcribe_mod, "transcribe", fake)
    return calls


@pytest.fixture
def stub_regions(monkeypatch) -> list:
    """A detector that proposes one region per source, with colliding ids.

    Colliding on purpose: `cam_a` is exactly what the real detector names the
    first camera it finds in *any* file, so two views of one meeting produce
    two different rectangles under one id unless prepare does something about
    it.
    """
    seen: list = []
    module = ModuleType("makeshorts.prepare.regions")

    def detect_regions(source, **kwargs):
        seen.append(source)
        return RegionsDoc(
            grid="4x4",
            frames_sampled=12,
            regions=[
                Region(id="cam_a", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0), confidence=0.5),
                Region(id="slides", kind="slide", rect=(0.5, 0.0, 0.5, 1.0)),
            ],
        )

    module.detect_regions = detect_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)
    return seen


# --------------------------------------------------------------------------
# regions.json
# --------------------------------------------------------------------------


def test_one_regions_json_holds_every_source(views, stub_transcribe, stub_regions) -> None:
    job = views["job"]
    result = run_mod.prepare(job, sources=views["sources"])

    assert job.regions_json in result.written
    assert len(stub_regions) == 2, "region detection did not run once per source"

    doc = job.load_regions()
    assert [r.source for r in doc.regions] == ["cam", "cam", "slides", "slides"]
    assert doc.frames_sampled == 24


def test_region_ids_are_unique_across_sources_and_still_legal(
    views, stub_transcribe, stub_regions
) -> None:
    """Two views of one meeting both contain a `cam_a`. They are different
    rectangles of different files, and a layout has to be able to say which."""
    run_mod.prepare(views["job"], sources=views["sources"])
    ids = [r.id for r in views["job"].load_regions().regions]

    assert ids == ["cam__cam_a", "cam__slides", "slides__cam_a", "slides__slides"]
    assert len(set(ids)) == len(ids)
    for rid in ids:
        assert REGION_ID.match(rid), f"{rid} is not a legal region id"


def test_the_source_a_region_belongs_to_survives_the_round_trip(
    views, stub_transcribe, stub_regions
) -> None:
    run_mod.prepare(views["job"], sources=views["sources"])
    raw = json.loads(views["job"].regions_json.read_text())
    by_id = {r["id"]: r for r in raw["regions"]}

    assert by_id["cam__cam_a"]["source"] == "cam"
    assert by_id["slides__slides"]["source"] == "slides"
    # Detection telemetry is carried through the merge, not flattened away.
    assert by_id["cam__cam_a"]["confidence"] == 0.5
    assert by_id["cam__cam_a"]["rect"] == [0.0, 0.0, 0.5, 1.0]


def test_one_source_failing_detection_does_not_lose_the_others(
    views, stub_transcribe, monkeypatch
) -> None:
    module = ModuleType("makeshorts.prepare.regions")

    def detect_regions(source, **kwargs):
        if "slides" in str(source):
            raise RuntimeError("no stable tiles found")
        return RegionsDoc(
            grid="4x4",
            frames_sampled=9,
            regions=[Region(id="cam_a", kind="speaker", rect=(0.0, 0.0, 1.0, 1.0))],
        )

    module.detect_regions = detect_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)

    job = views["job"]
    result = run_mod.prepare(job, sources=views["sources"])

    assert [r.id for r in job.load_regions().regions] == ["cam__cam_a"]
    assert any("no stable tiles found" in w and "[slides]" in w for w in result.warnings)


def test_a_source_with_no_video_is_probed_but_not_region_detected(
    tmp_path, av_file, audio_only_file, stub_transcribe, stub_regions
) -> None:
    """The audio-only export that arrives beside the video views is a source
    like any other -- it just has no frame to divide up."""
    job = Job.create("with-audio-export", tmp_path / "jobs")
    cam = job.source_for(av_file, "cam")
    cam.write_bytes(av_file.read_bytes())
    audio = job.source_for(audio_only_file, "audio")
    audio.write_bytes(audio_only_file.read_bytes())

    result = run_mod.prepare(job, sources={"cam": cam, "audio": audio}, audio_from="audio")

    assert set(job.load_media_sources()) == {"cam", "audio"}
    assert len(stub_regions) == 1
    assert {r.source for r in job.load_regions().regions} == {"cam"}
    assert any("no video stream" in w and "[audio]" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Transcribing once
# --------------------------------------------------------------------------


def test_transcription_runs_once_no_matter_how_many_sources(
    views, stub_transcribe, stub_regions
) -> None:
    """The views share an audio track. Transcribing each would cost minutes per
    file to produce the same words.json."""
    result = run_mod.prepare(views["job"], sources=views["sources"])

    assert len(stub_transcribe) == 1
    assert stub_transcribe[0]["path"] == views["sources"]["cam"]
    assert result.audio_source == "cam"


def test_audio_from_chooses_which_source_is_transcribed(
    views, stub_transcribe, stub_regions
) -> None:
    result = run_mod.prepare(views["job"], sources=views["sources"], audio_from="slides")

    assert len(stub_transcribe) == 1
    assert stub_transcribe[0]["path"] == views["sources"]["slides"]
    assert result.audio_source == "slides"


def test_which_source_the_transcript_came_from_is_said_out_loud(
    views, stub_transcribe, stub_regions
) -> None:
    lines: list[str] = []
    result = run_mod.prepare(
        views["job"], sources=views["sources"], audio_from="slides", log=lines.append
    )

    assert any("audio from source 'slides'" in line for line in lines)
    assert "transcript from source 'slides'" in run_mod.summarize(result)


def test_media_json_records_which_source_the_audio_came_from(
    views, stub_transcribe, stub_regions
) -> None:
    """Which file produced these timestamps is not a question the filesystem
    can answer later, so it is written down."""
    job = views["job"]
    run_mod.prepare(job, sources=views["sources"], audio_from="slides")

    assert json.loads(job.media_json.read_text())["audio_from"] == "slides"
    assert job.audio_source_name == "slides"


def test_an_unknown_audio_from_is_rejected(views) -> None:
    with pytest.raises(ValueError, match="not one of this job's sources"):
        run_mod.prepare(views["job"], sources=views["sources"], audio_from="gallery")


# --------------------------------------------------------------------------
# media.json
# --------------------------------------------------------------------------


def test_media_json_is_keyed_by_source_name(views, stub_transcribe, stub_regions) -> None:
    job = views["job"]
    run_mod.prepare(job, sources=views["sources"])

    raw = json.loads(job.media_json.read_text())
    assert raw["primary"] == "cam"
    assert list(raw["sources"]) == ["cam", "slides"]
    assert raw["sources"]["cam"]["resolution"] == [320, 240]
    assert raw["sources"]["slides"]["resolution"] == [480, 270]
    # The two views have different hashes; nothing is being written twice.
    assert raw["sources"]["cam"]["sha256"] != raw["sources"]["slides"]["sha256"]


def test_load_media_still_returns_one_document_and_it_is_the_primary(
    views, stub_transcribe, stub_regions
) -> None:
    """Everything downstream that only ever wanted one file keeps working."""
    job = views["job"]
    run_mod.prepare(job, sources=views["sources"])

    assert job.load_media().resolution == (320, 240)
    assert list(job.load_media_sources()) == ["cam", "slides"]


def test_a_source_added_later_is_probed_without_re_running_anything_else(
    views, tmp_path, slides_view_file, stub_transcribe, stub_regions
) -> None:
    job = views["job"]
    run_mod.prepare(job, sources={"cam": views["sources"]["cam"]})
    assert len(stub_transcribe) == 1

    result = run_mod.prepare(job, sources=views["sources"])
    assert "transcribe" in result.skipped
    assert len(stub_transcribe) == 1, "adding a source re-transcribed the audio"
    assert list(job.load_media_sources()) == ["cam", "slides"]


def test_a_source_added_after_regions_json_says_so(
    views, stub_transcribe, stub_regions
) -> None:
    """A cached regions.json is not wrong about the new source, it is silent
    about it -- and nothing downstream can tell that apart from "this view has
    no regions"."""
    job = views["job"]
    run_mod.prepare(job, sources={"cam": views["sources"]["cam"]})

    result = run_mod.prepare(job, sources=views["sources"])
    assert any("regions.json proposes nothing for slides" in w for w in result.warnings)


def test_a_cached_words_json_keeps_saying_which_source_it_came_from(
    views, stub_transcribe, stub_regions
) -> None:
    """media.json's `audio_from` is provenance. Re-running with a different
    --audio-from must not relabel a transcript nobody re-made."""
    job = views["job"]
    run_mod.prepare(job, sources=views["sources"], audio_from="cam")

    result = run_mod.prepare(job, sources=views["sources"], audio_from="slides")
    assert len(stub_transcribe) == 1
    assert job.audio_source_name == "cam"
    assert any("was transcribed from source 'cam'" in w for w in result.warnings)

    forced = run_mod.prepare(job, sources=views["sources"], audio_from="slides", force=True)
    assert len(stub_transcribe) == 2
    assert job.audio_source_name == "slides"
    assert forced.audio_source == "slides"


# --------------------------------------------------------------------------
# The job directory
# --------------------------------------------------------------------------


def test_named_sources_land_beside_each_other_in_the_job_dir(views) -> None:
    job = views["job"]
    assert (job.root / "source-cam.mp4").exists()
    assert (job.root / "source-slides.mp4").exists()
    assert not (job.root / "source.mp4").exists()
    assert job.has_named_sources


def test_the_primary_source_is_the_one_prepare_was_given_first(
    views, stub_transcribe, stub_regions
) -> None:
    """Alphabetically `cam` already sorts first, so order it the other way to
    prove the answer comes from media.json rather than from the directory."""
    job = views["job"]
    reversed_order = {"slides": views["sources"]["slides"], "cam": views["sources"]["cam"]}
    run_mod.prepare(job, sources=reversed_order)

    assert list(job.sources) == ["slides", "cam"]
    assert job.source == job.root / "source-slides.mp4"
    assert job.load_media().resolution == (480, 270)


def test_a_source_name_that_could_not_be_a_region_prefix_is_rejected(views) -> None:
    with pytest.raises(ValueError, match="not usable"):
        run_mod.prepare(views["job"], sources={"Cam 1": views["sources"]["cam"]})


def test_source_and_sources_together_is_rejected(views) -> None:
    with pytest.raises(ValueError, match="not both"):
        run_mod.prepare(
            views["job"],
            source=views["sources"]["cam"],
            sources=views["sources"],
        )


def test_a_missing_source_file_names_which_one(views, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no source media.*'gone'"):
        run_mod.prepare(
            views["job"],
            sources={"cam": views["sources"]["cam"], "gone": tmp_path / "nope.mp4"},
        )


# --------------------------------------------------------------------------
# The single-source form is untouched
# --------------------------------------------------------------------------


def test_a_single_source_job_writes_exactly_what_it_always_did(
    tmp_path, av_file, stub_transcribe, stub_regions
) -> None:
    """The common case must not pay for the multi-source one: no name in the
    filename, no wrapper in media.json, no prefix on a region id."""
    job = Job.create("one-file", tmp_path / "jobs")
    job.source_for(av_file).write_bytes(av_file.read_bytes())

    run_mod.prepare(job)

    assert (job.root / "source.mp4").exists()
    media = json.loads(job.media_json.read_text())
    assert set(media) == {"path", "duration", "resolution", "fps", "has_audio",
                          "audio_channels", "sha256"}

    regions = json.loads(job.regions_json.read_text())
    assert [r["id"] for r in regions["regions"]] == ["cam_a", "slides"]
    assert all("source" not in r for r in regions["regions"])
    assert regions["grid"] == "4x4"


def test_the_brief_says_which_source_each_region_came_from(
    views, stub_transcribe, stub_regions
) -> None:
    """A merged regions.json is only useful if the editorial step can tell the
    views apart, so PROMPT.md names the source of every region and says that a
    layout may mix them -- which is the whole reason to ingest several files.
    """
    from makeshorts.select.criteria import load_criteria
    from makeshorts.select.prompt import build_prompt

    job = views["job"]
    run_mod.prepare(job, sources=views["sources"])

    text = build_prompt(
        job_slug=job.slug,
        criteria=load_criteria(CRITERIA_YAML),
        transcript=TRANSCRIPT,
        regions=job.load_regions(),
        resolution=(320, 240),
    )
    section = text.split("## 4.")[1].split("\n## ")[0]

    assert "| id | source | kind |" in section
    assert "| `cam__cam_a` | `cam` | speaker |" in section
    assert "| `slides__slides` | `slides` | slide |" in section
    assert "a layout may mix sources" in section.lower()
    # The implicit per-source frames are the layout most of these jobs want.
    assert "`slides_frame`" in section


def test_a_single_source_brief_has_no_source_column(
    tmp_path, av_file, stub_transcribe, stub_regions
) -> None:
    from makeshorts.select.criteria import load_criteria
    from makeshorts.select.prompt import build_prompt

    job = Job.create("one-file-brief", tmp_path / "jobs")
    job.source_for(av_file).write_bytes(av_file.read_bytes())
    run_mod.prepare(job)

    text = build_prompt(
        job_slug=job.slug,
        criteria=load_criteria(CRITERIA_YAML),
        transcript=TRANSCRIPT,
        regions=job.load_regions(),
        resolution=(320, 240),
    )
    section = text.split("## 4.")[1].split("\n## ")[0]

    assert "| id | kind | label |" in section
    # No source column, no per-source frame ids, nothing about mixing sources:
    # a one-file job is not made to read about a distinction it does not have.
    assert "| id | source |" not in section
    assert "_frame`" not in section
    assert "mix sources" not in section


def test_a_single_named_source_gets_a_name_but_no_region_prefix(
    tmp_path, av_file, stub_transcribe, stub_regions
) -> None:
    """Nothing to collide with, so nothing to disambiguate."""
    job = Job.create("one-named", tmp_path / "jobs")
    cam = job.source_for(av_file, "cam")
    cam.write_bytes(av_file.read_bytes())

    run_mod.prepare(job, sources={"cam": cam})

    assert [r.id for r in job.load_regions().regions] == ["cam_a", "slides"]
    assert {r.source for r in job.load_regions().regions} == {"cam"}
    assert list(job.load_media_sources()) == ["cam"]
