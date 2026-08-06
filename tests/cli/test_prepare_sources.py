"""`ms prepare` with several named inputs.

The mechanical stage itself is stubbed throughout: what these tests are about
is the surface -- how a multi-source job is asked for, what lands in the job
directory, and whether a mistyped `--source` produces a sentence a person can
act on. `tests/prepare/test_multi_source.py` covers what prepare then does
with those files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from makeshorts import cli as cli_mod
from makeshorts.jobs import Job

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _text(result) -> str:
    out = result.stdout or ""
    try:
        err = result.stderr or ""
    except ValueError:  # stderr not captured separately by this click version
        err = ""
    return _ANSI.sub("", out + err)


@pytest.fixture
def recordings(tmp_path: Path) -> dict[str, Path]:
    """Two stand-in files. Nothing here decodes them."""
    videos = tmp_path / "videos"
    videos.mkdir()
    files = {}
    for name, size in (("cam", 2048), ("slides", 4096)):
        path = videos / f"GMT20260728-165920_Recording_{name}.mp4"
        path.write_bytes(b"\0" * size)
        files[name] = path
    return files


@pytest.fixture
def stub_prepare(monkeypatch):
    """Record what the CLI asked the mechanical stage to do."""
    import makeshorts.prepare.run as run_mod

    calls: list[dict] = []

    def fake(job, **kwargs):
        calls.append({"job": job, **kwargs})
        return run_mod.PrepareResult(job=job)

    monkeypatch.setattr(run_mod, "prepare", fake)
    return calls


def _invoke(runner, jobs_dir: Path, config_dir: Path, *args: str):
    return runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_dir), "--config-dir", str(config_dir), "prepare", *args],
    )


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_named_sources_are_ingested_side_by_side(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--slug", "webinar",
        "--source", f"cam={recordings['cam']}",
        "--source", f"slides={recordings['slides']}",
    )
    assert result.exit_code == 0, _text(result)

    job = Job.at("webinar", jobs_dir)
    assert (job.root / "source-cam.mp4").exists()
    assert (job.root / "source-slides.mp4").exists()
    assert not (job.root / "source.mp4").exists()

    kwargs = stub_prepare[0]
    assert list(kwargs["sources"]) == ["cam", "slides"]
    assert kwargs["sources"]["cam"] == job.root / "source-cam.mp4"
    # The files inside the job are what gets prepared, never the originals.
    assert all(p.parent == job.root for p in kwargs["sources"].values())


def test_the_first_source_given_is_the_primary_one(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    """Order is meaningful: the primary source is what `frame` refers to."""
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--slug", "webinar",
        "--source", f"slides={recordings['slides']}",
        "--source", f"cam={recordings['cam']}",
    )
    assert result.exit_code == 0, _text(result)
    assert list(stub_prepare[0]["sources"]) == ["slides", "cam"]


def test_audio_from_is_passed_through_and_reported(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--slug", "webinar",
        "--source", f"cam={recordings['cam']}",
        "--source", f"slides={recordings['slides']}",
        "--audio-from", "slides",
    )
    assert result.exit_code == 0, _text(result)
    assert stub_prepare[0]["audio_from"] == "slides"
    assert "from source 'slides'" in _text(result)


def test_without_audio_from_the_primary_source_is_named_as_the_one_transcribed(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--slug", "webinar",
        "--source", f"cam={recordings['cam']}",
        "--source", f"slides={recordings['slides']}",
    )
    assert result.exit_code == 0, _text(result)
    assert "audio_from" not in stub_prepare[0]
    assert "from source 'cam'" in _text(result)


def test_the_slug_is_derived_from_the_first_source_when_none_is_given(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--source", f"cam={recordings['cam']}",
        "--source", f"slides={recordings['slides']}",
    )
    assert result.exit_code == 0, _text(result)
    assert stub_prepare[0]["job"].slug == "gmt20260728-165920-recording-cam"


# --------------------------------------------------------------------------
# The single-file form, unchanged
# --------------------------------------------------------------------------


def test_a_single_positional_input_still_lands_as_source_ext(
    runner, jobs_dir: Path, config_dir: Path, recordings, stub_prepare
) -> None:
    result = _invoke(runner, jobs_dir, config_dir, "--slug", "one", str(recordings["cam"]))
    assert result.exit_code == 0, _text(result)

    job = Job.at("one", jobs_dir)
    assert (job.root / "source.mp4").exists()
    kwargs = stub_prepare[0]
    assert kwargs["source"] == job.root / "source.mp4"
    assert "sources" not in kwargs
    assert "audio_from" not in kwargs


# --------------------------------------------------------------------------
# Mistakes
# --------------------------------------------------------------------------


def test_a_source_without_an_equals_sign_says_what_the_shape_is(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(runner, jobs_dir, config_dir, "--source", str(recordings["cam"]))
    assert result.exit_code == 1
    text = _text(result)
    assert "is not NAME=PATH" in text
    assert "--source cam=" in text
    assert "Traceback" not in text


@pytest.mark.parametrize("name", ["Cam", "cam 1", "1cam", "cam-a", ""])
def test_a_name_that_could_not_be_a_region_prefix_is_rejected(
    runner, jobs_dir: Path, config_dir: Path, recordings, name: str
) -> None:
    result = _invoke(runner, jobs_dir, config_dir, "--source", f"{name}={recordings['cam']}")
    assert result.exit_code == 1
    assert "Traceback" not in _text(result)


def test_the_rejection_of_a_bad_name_explains_what_names_are_for(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(runner, jobs_dir, config_dir, "--source", f"Cam={recordings['cam']}")
    text = _text(result)
    assert "not usable" in text
    assert "region-id prefix" in text


def test_a_source_pointing_at_nothing_fails_before_any_job_is_created(
    runner, jobs_dir: Path, config_dir: Path
) -> None:
    result = _invoke(runner, jobs_dir, config_dir, "--source", "cam=/nope/missing.mp4")
    assert result.exit_code == 1
    assert "no such file" in _text(result)
    assert list(jobs_dir.iterdir()) == []


def test_the_same_name_twice_is_rejected(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--source", f"cam={recordings['cam']}",
        "--source", f"cam={recordings['slides']}",
    )
    assert result.exit_code == 1
    assert "given twice" in _text(result)


def test_an_input_and_a_source_together_is_rejected(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        str(recordings["cam"]),
        "--source", f"slides={recordings['slides']}",
    )
    assert result.exit_code == 1
    text = _text(result)
    assert "not both" in text
    assert "--source cam=" in text


def test_neither_an_input_nor_a_source_says_what_to_pass(
    runner, jobs_dir: Path, config_dir: Path
) -> None:
    result = _invoke(runner, jobs_dir, config_dir)
    assert result.exit_code == 1
    assert "nothing to ingest" in _text(result)


def test_audio_from_naming_an_unknown_source_lists_the_real_ones(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(
        runner,
        jobs_dir,
        config_dir,
        "--source", f"cam={recordings['cam']}",
        "--source", f"slides={recordings['slides']}",
        "--audio-from", "gallery",
    )
    assert result.exit_code == 1
    text = _text(result)
    assert "'gallery' is not one of the sources" in text
    assert "cam, slides" in text


def test_audio_from_without_any_named_source_explains_why_it_exists(
    runner, jobs_dir: Path, config_dir: Path, recordings
) -> None:
    result = _invoke(
        runner, jobs_dir, config_dir, str(recordings["cam"]), "--audio-from", "cam"
    )
    assert result.exit_code == 1
    assert "--audio-from names one of the --source inputs" in _text(result)
