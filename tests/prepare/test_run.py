"""Orchestration.

The transcription stage is stubbed throughout: whether faster-whisper works is
not what these tests are about, and downloading a model to assert that a file
gets written would be absurd. What is under test is the wiring -- stage order,
caching, skipping, and graceful degradation when regions.py is absent.
"""

from __future__ import annotations

import json
import sys

import pytest

from makeshorts.artifacts import MediaDoc, SilenceDoc, WordsDoc
from makeshorts.jobs import Job
from makeshorts.prepare import run as run_mod

from .conftest import evenly_timed, requires_ffmpeg

pytestmark = requires_ffmpeg

TRANSCRIPT = "Eighteen months is the number. Nobody ever checks it."


@pytest.fixture
def job(tmp_path, av_file) -> Job:
    """A job directory with the synthesised source already ingested."""
    j = Job.create("fixture", tmp_path / "jobs")
    j.source_for(av_file).write_bytes(av_file.read_bytes())
    return j


@pytest.fixture
def stub_transcribe(monkeypatch) -> list[dict]:
    """Replace the model call, recording the arguments it was given."""
    calls: list[dict] = []

    def fake(path, **kwargs):
        calls.append({"path": path, **kwargs})
        return evenly_timed(TRANSCRIPT)

    monkeypatch.setattr(run_mod.transcribe_mod, "transcribe", fake)
    return calls


@pytest.fixture
def no_regions_module(monkeypatch):
    """Region detection is written by other code and may not exist yet."""
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", None)


def test_prepare_writes_every_artifact(job, stub_transcribe, no_regions_module) -> None:
    result = run_mod.prepare(job)

    assert job.media_json.exists()
    assert job.silence_json.exists()
    assert job.words_json.exists()
    assert job.transcript_txt.exists()
    assert job.raw_srt.exists()
    assert set(result.written) >= {
        job.media_json,
        job.silence_json,
        job.words_json,
        job.transcript_txt,
        job.raw_srt,
    }


def test_written_json_validates_against_the_contract(job, stub_transcribe, no_regions_module) -> None:
    run_mod.prepare(job)
    assert isinstance(job.load_media(), MediaDoc)
    assert isinstance(job.load_words(), WordsDoc)
    assert isinstance(job.load_silence(), SilenceDoc)


def test_json_is_pretty_printed(job, stub_transcribe, no_regions_module) -> None:
    """These files exist to be read and diffed by hand."""
    run_mod.prepare(job)
    raw = job.media_json.read_text()
    assert raw.endswith("\n")
    assert raw.count("\n") > 3
    assert '  "duration"' in raw
    json.loads(raw)


def test_the_media_duration_is_passed_to_silence_detection(job, stub_transcribe, no_regions_module) -> None:
    """This is what closes an unterminated trailing silence."""
    run_mod.prepare(job)
    media = job.load_media()
    silence = job.load_silence()
    for span in silence.spans:
        assert span.end <= media.duration + 0.01


def test_a_second_run_reuses_existing_artifacts(job, stub_transcribe, no_regions_module) -> None:
    run_mod.prepare(job)
    assert len(stub_transcribe) == 1

    second = run_mod.prepare(job)
    assert len(stub_transcribe) == 1, "transcription re-ran despite words.json existing"
    assert "transcribe" in second.skipped
    assert "probe" in second.skipped
    # Text outputs are pure functions of words.json, so they are always rewritten.
    assert job.transcript_txt in second.written


def test_force_re_runs_every_stage(job, stub_transcribe, no_regions_module) -> None:
    run_mod.prepare(job)
    run_mod.prepare(job, force=True)
    assert len(stub_transcribe) == 2


def test_deleting_one_artifact_re_runs_only_that_stage(job, stub_transcribe, no_regions_module) -> None:
    """The artifacts are the state -- that is the whole point of the job dir."""
    run_mod.prepare(job)
    job.words_json.unlink()
    result = run_mod.prepare(job)
    assert len(stub_transcribe) == 2
    assert "probe" in result.skipped
    assert "silence" in result.skipped


def test_skipping_transcription_still_produces_media_and_silence(job, no_regions_module) -> None:
    result = run_mod.prepare(job, skip=["transcribe"])
    assert job.media_json.exists()
    assert job.silence_json.exists()
    assert not job.words_json.exists()
    assert not job.raw_srt.exists()
    assert any("no words.json" in w for w in result.warnings)


def test_an_unknown_skip_name_is_rejected(job) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        run_mod.prepare(job, skip=["transcirbe"])


def test_transcription_options_reach_the_model(job, stub_transcribe, no_regions_module) -> None:
    run_mod.prepare(job, model="medium", compute_type="float16", device="cuda", language="de")
    call = stub_transcribe[0]
    assert call["model"] == "medium"
    assert call["compute_type"] == "float16"
    assert call["device"] == "cuda"
    assert call["language"] == "de"


def test_a_missing_regions_module_is_a_warning_not_a_failure(job, stub_transcribe, no_regions_module) -> None:
    """`frame` is always a valid layout target, so a job without regions.json is
    degraded, not broken."""
    result = run_mod.prepare(job)
    assert not job.regions_json.exists()
    assert any("region detection failed or is unavailable" in w for w in result.warnings)


def test_a_crashing_region_detector_does_not_lose_the_transcript(job, stub_transcribe, monkeypatch) -> None:
    """Region detection is the only optional stage. A heuristic tripping over
    an odd frame must not cost us words.json."""
    from types import ModuleType

    module = ModuleType("makeshorts.prepare.regions")

    def detect_regions(source, **kwargs):
        raise RuntimeError("no stable tiles found")

    module.detect_regions = detect_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)

    result = run_mod.prepare(job)
    assert not job.regions_json.exists()
    assert job.words_json.exists()
    assert job.transcript_txt.exists()
    assert any("no stable tiles found" in w for w in result.warnings)


def test_regions_are_written_when_detection_is_available(job, stub_transcribe, monkeypatch) -> None:
    from types import ModuleType

    from makeshorts.artifacts import Region, RegionsDoc

    module = ModuleType("makeshorts.prepare.regions")
    seen: list[dict] = []

    def detect_regions(source):
        seen.append({"source": source})
        return RegionsDoc(
            grid="4x4",
            frames_sampled=12,
            regions=[Region(id="cam_a", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0))],
        )

    module.detect_regions = detect_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)

    result = run_mod.prepare(job)
    assert job.regions_json in result.written
    assert job.load_regions().regions[0].id == "cam_a"
    # A detector that declares only `source` is called with only `source`: the
    # signature belongs to that module, not to this one.
    assert set(seen[0]) == {"source"}


def test_a_regions_detector_taking_kwargs_is_offered_every_media_fact(job, stub_transcribe, monkeypatch) -> None:
    from types import ModuleType

    from makeshorts.artifacts import RegionsDoc

    module = ModuleType("makeshorts.prepare.regions")
    seen: list[dict] = []

    def detect_regions(source, **kwargs):
        seen.append(kwargs)
        return RegionsDoc(grid="2x2", frames_sampled=1, regions=[])

    module.detect_regions = detect_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)

    run_mod.prepare(job)
    assert set(seen[0]) == {"duration", "resolution", "fps"}


def test_regions_detector_receives_media_facts_when_it_asks_for_them(job, stub_transcribe, monkeypatch) -> None:
    from types import ModuleType

    from makeshorts.artifacts import RegionsDoc

    module = ModuleType("makeshorts.prepare.regions")
    seen: list[dict] = []

    def propose_regions(source, *, duration, fps):
        seen.append({"duration": duration, "fps": fps})
        return RegionsDoc(grid="3x3", frames_sampled=5, regions=[])

    module.propose_regions = propose_regions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "makeshorts.prepare.regions", module)

    run_mod.prepare(job)
    assert seen[0]["duration"] == pytest.approx(6.0, abs=0.2)
    assert seen[0]["fps"] == pytest.approx(30.0)


def test_a_source_with_no_audio_produces_empty_silence_and_no_transcript(
    tmp_path, video_only_file, no_regions_module
) -> None:
    j = Job.create("mute", tmp_path / "jobs")
    j.source_for(video_only_file).write_bytes(video_only_file.read_bytes())

    result = run_mod.prepare(j)
    assert j.load_media().has_audio is False
    # Not one long silent span: that would fail every clip's max_internal_silence.
    assert j.load_silence().spans == []
    assert not j.words_json.exists()
    assert any("no audio stream" in w for w in result.warnings)


def test_a_job_with_no_source_fails_clearly(tmp_path) -> None:
    j = Job.create("empty", tmp_path / "jobs")
    with pytest.raises(FileNotFoundError, match="no source media"):
        run_mod.prepare(j)


def test_summarize_mentions_what_was_written(job, stub_transcribe, no_regions_module) -> None:
    text = run_mod.summarize(run_mod.prepare(job))
    assert "prepared fixture" in text
    assert "words.json" in text
