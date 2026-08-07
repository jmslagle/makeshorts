"""Job directory layout and output naming.

The filename `out/<slug>--<clip.id>.mp4` is the one piece of this system that
leaves the repository: rendered clips get dragged into an upload queue, and the
name has to still say which webinar and which moment it came from. Everything
here pins that down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeshorts.jobs import (
    PRIMARY_SOURCE_NAME,
    ArtifactInvalid,
    Job,
    JobError,
    JobNotFound,
    StageNotRun,
    iter_jobs,
)


def test_job_at_touches_nothing(tmp_path: Path) -> None:
    job = Job.at("acme-q3", tmp_path)
    assert job.root == tmp_path / "acme-q3"
    assert not job.root.exists()


def test_open_missing_job_names_the_ones_that_exist(tmp_path: Path) -> None:
    (tmp_path / "acme-q3").mkdir()
    with pytest.raises(JobNotFound) as exc:
        Job.open("typo", tmp_path)
    assert "acme-q3" in str(exc.value)


def test_create_refuses_to_clobber_and_says_how_to_proceed(tmp_path: Path) -> None:
    Job.create("acme-q3", tmp_path)
    with pytest.raises(JobError) as exc:
        Job.create("acme-q3", tmp_path)
    assert "--force" in str(exc.value)

    # exist_ok is what --force turns on.
    assert Job.create("acme-q3", tmp_path, exist_ok=True).slug == "acme-q3"


def test_create_makes_the_directories_a_render_will_need(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    assert job.out_dir.is_dir()
    assert job.cache_dir.is_dir()


def test_artifact_paths_are_flat_and_predictable(tmp_path: Path) -> None:
    job = Job.at("acme-q3", tmp_path)
    root = tmp_path / "acme-q3"
    assert job.media_json == root / "media.json"
    assert job.words_json == root / "words.json"
    assert job.silence_json == root / "silence.json"
    assert job.regions_json == root / "regions.json"
    assert job.transcript_txt == root / "transcript.txt"
    assert job.raw_srt == root / "raw.srt"
    assert job.prompt_md == root / "PROMPT.md"
    assert job.clips_json == root / "clips.json"
    assert job.config_dir == root / "config"


def test_output_path_repeats_the_slug(tmp_path: Path) -> None:
    job = Job.at("acme-q3-webinar", tmp_path)
    out = job.output_path("01-cac-payback-math")
    assert out.name == "acme-q3-webinar--01-cac-payback-math.mp4"
    assert out.parent == job.out_dir


def test_receipt_sits_beside_the_clip_with_the_same_stem(tmp_path: Path) -> None:
    job = Job.at("acme-q3", tmp_path)
    clip = job.output_path("03-pricing-floor")
    receipt = job.receipt_path("03-pricing-floor")
    assert receipt.parent == clip.parent
    assert receipt.stem == clip.stem
    assert receipt.suffix == ".json"


def test_source_follows_whatever_was_ingested(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    # Nothing ingested yet: the .mp4 name is the placeholder.
    assert job.source.name == "source.mp4"

    (job.root / "source.mkv").write_bytes(b"x")
    assert job.source.name == "source.mkv"


def test_source_for_keeps_the_container_extension(tmp_path: Path) -> None:
    job = Job.at("acme-q3", tmp_path)
    assert job.source_for("/downloads/Recording.MOV").name == "source.mov"


# --------------------------------------------------------------------------
# Several sources
#
# One recording exported as several frame-aligned views is one job. What the
# directory has to keep straight is which file is which and which one is
# primary -- `frame`, and any region that names no source, mean that one.
# --------------------------------------------------------------------------


def test_a_single_source_job_reports_one_source_under_the_primary_name(
    tmp_path: Path,
) -> None:
    job = Job.create("acme-q3", tmp_path)
    (job.root / "source.mp4").write_bytes(b"x")

    assert job.sources == {PRIMARY_SOURCE_NAME: job.root / "source.mp4"}
    assert not job.has_named_sources


def test_the_primary_source_name_agrees_with_the_edit_list_schema() -> None:
    """jobs.py repeats the constant rather than importing select/, so the two
    have to be checked against each other somewhere."""
    from makeshorts.select.schema import PRIMARY_SOURCE_NAME as SCHEMA_NAME

    assert PRIMARY_SOURCE_NAME == SCHEMA_NAME


def test_named_sources_are_found_and_source_falls_back_to_the_primary(
    tmp_path: Path,
) -> None:
    job = Job.create("acme-q3", tmp_path)
    (job.root / "source-cam.mp4").write_bytes(b"x")
    (job.root / "source-slides.mp4").write_bytes(b"x")

    assert job.has_named_sources
    assert set(job.sources) == {"cam", "slides"}
    # No media.json yet, so disk order decides and `source` is the first of it.
    assert job.source == job.root / "source-cam.mp4"


def test_media_json_decides_which_named_source_is_primary(tmp_path: Path) -> None:
    """Alphabetical order is arbitrary; which view is primary is a decision,
    and it was made when the job was prepared."""
    job = Job.create("acme-q3", tmp_path)
    (job.root / "source-cam.mp4").write_bytes(b"x")
    (job.root / "source-slides.mp4").write_bytes(b"x")
    job.media_json.write_text(
        json.dumps({"primary": "slides", "sources": {"slides": {}, "cam": {}}})
    )

    assert list(job.sources) == ["slides", "cam"]
    assert job.source == job.root / "source-slides.mp4"


def test_a_malformed_media_json_cannot_make_a_source_path_unreadable(
    tmp_path: Path,
) -> None:
    """These are path properties. They feed error messages, including the one
    about media.json being broken."""
    job = Job.create("acme-q3", tmp_path)
    (job.root / "source-cam.mp4").write_bytes(b"x")
    (job.root / "source-slides.mp4").write_bytes(b"x")
    job.media_json.write_text("{not json")

    assert set(job.sources) == {"cam", "slides"}
    assert job.source.exists()


def test_source_for_names_a_file_after_its_source(tmp_path: Path) -> None:
    job = Job.at("acme-q3", tmp_path)
    assert job.source_for("/downloads/Recording.MOV", "cam").name == "source-cam.mov"
    with pytest.raises(JobError, match="not usable"):
        job.source_for("/downloads/Recording.mp4", "Cam 1")


def test_load_media_reads_both_shapes_of_media_json(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    one = {
        "path": "jobs/acme-q3/source.mp4",
        "duration": 60.0,
        "resolution": [1920, 1080],
        "fps": 30.0,
        "has_audio": True,
        "audio_channels": 1,
        "sha256": "a" * 64,
    }
    job.media_json.write_text(json.dumps(one))
    assert job.load_media().resolution == (1920, 1080)
    assert list(job.load_media_sources()) == [PRIMARY_SOURCE_NAME]

    other = dict(one, resolution=[2378, 1410], path="jobs/acme-q3/source-slides.mp4")
    job.media_json.write_text(
        json.dumps({"primary": "cam", "audio_from": "cam",
                    "sources": {"slides": other, "cam": one}})
    )
    assert list(job.load_media_sources()) == ["cam", "slides"]
    assert job.load_media().resolution == (1920, 1080)
    assert job.audio_source_name == "cam"


def test_an_invalid_entry_in_a_keyed_media_json_names_the_source(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    job.media_json.write_text(json.dumps({"primary": "cam", "sources": {"cam": {"path": 1}}}))

    with pytest.raises(ArtifactInvalid, match="source 'cam'"):
        job.load_media()


def test_stage_advances_as_artifacts_appear(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    assert job.stage == "empty"

    (job.root / "source.mp4").write_bytes(b"x")
    assert job.stage == "source"

    job.media_json.write_text("{}")
    job.words_json.write_text("{}")
    assert job.stage == "prepared"

    job.prompt_md.write_text("#")
    assert job.stage == "prompt"

    job.clips_json.write_text("{}")
    assert job.stage == "clips"

    job.output_path("01-a").write_bytes(b"x")
    assert job.stage == "rendered"


def test_a_gap_does_not_hide_later_progress(tmp_path: Path) -> None:
    """A missing PROMPT.md must not mask a clips.json that exists.

    Stages are skippable by design: PROMPT.md is a convenience for the
    editorial step rather than a prerequisite, an edit list can be written by
    hand, and `--skip transcribe` deliberately leaves no words.json. Reporting
    the furthest *contiguous* milestone therefore understates a job badly --
    it called a job with sixteen rendered clips `empty` and told the user to
    run `ms prepare` over the top of it. The furthest milestone actually
    reached is both accurate and useful.
    """
    job = Job.create("acme-q3", tmp_path)
    (job.root / "source.mp4").write_bytes(b"x")
    job.media_json.write_text("{}")
    job.words_json.write_text("{}")
    job.clips_json.write_text("{}")
    assert job.stage == "clips"


def test_stage_is_empty_only_when_nothing_exists(tmp_path: Path) -> None:
    job = Job.create("acme-q4", tmp_path)
    assert job.stage == "empty"


def test_a_rendered_job_reports_rendered_even_without_a_source_copy(tmp_path: Path) -> None:
    """The case that exposed this: an edit list assembled by hand, rendered,
    with no source file ever copied into the job."""
    job = Job.create("acme-q5", tmp_path)
    job.clips_json.write_text("{}")
    job.ensure_dirs()
    (job.out_dir / "acme-q5--01-x.mp4").write_bytes(b"x")
    assert job.stage == "rendered"


def test_rendered_clip_ids_strips_the_slug_prefix(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    job.output_path("02-b").write_bytes(b"x")
    job.output_path("01-a").write_bytes(b"x")
    # A stray file from another job must not be counted.
    (job.out_dir / "other--09-z.mp4").write_bytes(b"x")

    assert job.rendered_clip_ids() == ["01-a", "02-b"]


def test_loading_an_artifact_before_its_stage_ran_names_the_remedy(tmp_path: Path) -> None:
    job = Job.create("acme-q3", tmp_path)
    with pytest.raises(StageNotRun) as exc:
        job.load_words()
    assert exc.value.remedy == "ms prepare acme-q3"
    assert "ms prepare acme-q3" in str(exc.value)


def test_iter_jobs_is_sorted_and_ignores_dotfiles(tmp_path: Path) -> None:
    for name in ("beta", "alpha", ".hidden"):
        (tmp_path / name).mkdir()
    (tmp_path / "loose-file.mp4").write_bytes(b"x")

    assert [j.slug for j in iter_jobs(tmp_path)] == ["alpha", "beta"]


def test_iter_jobs_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert list(iter_jobs(tmp_path / "nope")) == []
