"""The ffmpeg engine, against real encodes.

These tests actually run ffmpeg. Fixtures are synthesized with `-f lavfi` --
nothing binary is committed -- and kept to a few seconds so the suite stays
usable.

Two things are worth knowing before editing this file:

- Every ffmpeg invocation is wrapped in a hard timeout (`_guard_ffmpeg`, which
  is autouse). The failure mode this module exists to prevent -- an unbounded
  `color` canvas feeding `overlay` -- does not raise, it *hangs*, so a plain
  assertion would never fire and the suite would simply stop.
- The multi-source cases lean on the crop geometry itself as the correctness
  check: `main` is 1280x720 and `slides` is 800x600, so wiring a placement to
  the wrong input asks ffmpeg to crop 720 rows out of a 600-row frame and the
  encode fails outright.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from makeshorts.artifacts import Region
from makeshorts.render import ffmpeg_engine as FE
from makeshorts.render.ffmpeg_engine import EncodeSettings, FFmpegEngine
from makeshorts.select.schema import (
    Clip,
    ClipsDoc,
    CriteriaRef,
    CriterionScore,
    FocusLayout,
    HeroInsetLayout,
    SourceSpec,
    Why,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed"
)

# Deliberately different shapes: a 16:9 camera render and a 4:3 screen render,
# as a real multi-view export would be.
MAIN_SIZE = (1280, 720)
SLIDES_SIZE = (800, 600)
FIXTURE_SECONDS = 5.0
OUTPUT_SIZE = (1080, 1920)

# Long enough that a slow machine encoding two 1080x1920 seconds never trips
# it, short enough that a hang is caught inside a coffee break.
FFMPEG_TIMEOUT = 90.0


# --------------------------------------------------------------------------
# Hang guard
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _guard_ffmpeg(monkeypatch):
    """Give every ffmpeg call a deadline and record what was run.

    An unbounded filter source produces an encode that never terminates rather
    than one that fails, so the timeout *is* the assertion.
    """
    calls: list[list[str]] = []
    real_run = subprocess.run

    def guarded(args: list[str]) -> str:
        calls.append(list(args))
        try:
            proc = real_run(args, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(
                f"ffmpeg did not terminate within {FFMPEG_TIMEOUT}s -- almost "
                f"certainly an unbounded filter source:\n{' '.join(args)}"
            ) from exc
        if proc.returncode != 0:
            raise FE.FFmpegError(args, proc.stderr)
        return proc.stderr

    monkeypatch.setattr(FE, "_run", guarded)
    return calls


# --------------------------------------------------------------------------
# Fixtures on disk
# --------------------------------------------------------------------------


def _synth(path: Path, video: str, size: tuple[int, int], audio: bool) -> Path:
    w, h = size
    args = [FFMPEG, "-hide_banner", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"{video}=size={w}x{h}:rate=30"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    args += ["-t", f"{FIXTURE_SECONDS}",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    args.append(str(path))
    subprocess.run(args, capture_output=True, text=True, timeout=120, check=True)
    return path


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    """Two frame-aligned renders of the same imaginary meeting.

    `main` carries the audio, as the primary source always does.
    """
    d = tmp_path_factory.mktemp("media")
    return {
        "main": _synth(d / "main.mp4", "testsrc2", MAIN_SIZE, audio=True),
        "slides": _synth(d / "slides.mp4", "smptebars", SLIDES_SIZE, audio=False),
    }


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def _why() -> Why:
    return Why(
        one_line="fixture",
        theme="test",
        scores={"hook_strength": CriterionScore(score=5, evidence="n/a")},
        weighted_score=5.0,
    )


def _clip(layout, start: float = 1.0, end: float = 3.0) -> Clip:
    return Clip(
        id="01-fixture",
        title="Fixture",
        start=start,
        end=end,
        source_text="fixture",
        why=_why(),
        layout=layout,
        captions={"enabled": False},
    )


CRITERIA = CriteriaRef(file="config/criteria.yaml", version="test", sha256="0" * 64)


def _multi_doc(media: dict[str, Path], clip: Clip) -> ClipsDoc:
    """Two named sources. `main` declares a camera region on its left half;
    `slides` relies on the implicit `slides_frame`."""
    return ClipsDoc(
        job="fixture",
        sources={
            "main": SourceSpec(
                path=str(media["main"]),
                duration=FIXTURE_SECONDS,
                resolution=MAIN_SIZE,
                regions=[Region(id="cam", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0))],
            ),
            "slides": SourceSpec(
                path=str(media["slides"]),
                duration=FIXTURE_SECONDS,
                resolution=SLIDES_SIZE,
            ),
        },
        criteria_ref=CRITERIA,
        clips=[clip],
    )


def _single_doc(media: dict[str, Path], clip: Clip) -> ClipsDoc:
    """The pre-existing shape: one `source`, no names anywhere."""
    return ClipsDoc(
        job="fixture",
        source=SourceSpec(
            path=str(media["main"]),
            duration=FIXTURE_SECONDS,
            resolution=MAIN_SIZE,
            regions=[Region(id="cam", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0))],
        ),
        criteria_ref=CRITERIA,
        clips=[clip],
    )


@pytest.fixture
def engine() -> FFmpegEngine:
    # ultrafast/crf 30 is about the encode finishing, not about looking good.
    return FFmpegEngine(EncodeSettings(preset="ultrafast", crf=30))


# --------------------------------------------------------------------------
# ffprobe helpers
# --------------------------------------------------------------------------


def probe(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "width": video["width"],
        "height": video["height"],
        "duration": float(data["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def span_commands(receipt) -> list[str]:
    """The per-span encodes, picked out of the receipt by their canvas."""
    return [c for c in receipt.commands if "color=c=black" in c]


def input_count(command: str) -> int:
    return command.split().count("-i")


def inputs_and_graph(command: str) -> tuple[list[str], str]:
    """The files a span command opens, in order, and its filtergraph.

    Everything else in the command is the temp workdir, which differs run to
    run and says nothing about how the graph was built.
    """
    tokens = command.split()
    files = [tokens[i + 1] for i, t in enumerate(tokens) if t == "-i"]
    graph = tokens[tokens.index("-filter_complex") + 1]
    return files, graph


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_hero_inset_composites_two_sources(engine, media, tmp_path):
    """Hero from the 800x600 screen render, inset from the 1280x720 camera."""
    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam", inset_scale=0.3))
    doc = _multi_doc(media, clip)
    out = tmp_path / "clip.mp4"

    receipt = engine.render(doc, clip, media["main"], out)

    info = probe(out)
    assert (info["width"], info["height"]) == OUTPUT_SIZE
    assert info["duration"] == pytest.approx(clip.duration, abs=0.15)
    assert info["has_audio"]
    assert receipt.duration == pytest.approx(clip.duration, abs=0.15)


def test_two_source_span_opens_exactly_two_inputs(engine, media, tmp_path):
    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam"))
    doc = _multi_doc(media, clip)

    receipt = engine.render(doc, clip, media["main"], tmp_path / "clip.mp4")

    spans = span_commands(receipt)
    assert len(spans) == 1
    assert input_count(spans[0]) == 2
    # One split per input, sized to that input's own consumers.
    assert "[0:v]split=" in spans[0] and "[1:v]split=" in spans[0]
    assert "[2:v]" not in spans[0]
    # Both files are seeked to the same absolute instant: they share a clock.
    assert spans[0].count(f"-ss {clip.start:.3f}") == 2


def test_focus_on_one_source_opens_one_input(engine, media, tmp_path):
    """A span that reads a single file must still produce a single-input
    command, even in a job that declares several sources."""
    clip = _clip(FocusLayout(region="slides_frame"))
    doc = _multi_doc(media, clip)

    receipt = engine.render(doc, clip, media["main"], tmp_path / "clip.mp4")

    spans = span_commands(receipt)
    assert len(spans) == 1
    assert input_count(spans[0]) == 1
    assert str(media["slides"]) in spans[0]
    assert str(media["main"]) not in spans[0].split("-filter_complex")[0]
    assert probe(tmp_path / "clip.mp4")["width"] == OUTPUT_SIZE[0]


def test_single_source_document_unchanged(engine, media, tmp_path):
    """The pre-existing single-`source` shape renders exactly as before, from
    the path the caller passed."""
    clip = _clip(FocusLayout(region="cam"))
    doc = _single_doc(media, clip)
    out = tmp_path / "clip.mp4"

    receipt = engine.render(doc, clip, media["main"], out)

    spans = span_commands(receipt)
    assert len(spans) == 1
    assert input_count(spans[0]) == 1
    info = probe(out)
    assert (info["width"], info["height"]) == OUTPUT_SIZE
    assert info["has_audio"]


def test_contain_blur_reads_one_input_twice(engine, media, tmp_path):
    """A contain_blur backdrop is a second placement on the SAME file: one
    input, split two ways."""
    clip = _clip(FocusLayout(region="slides_frame", fit="contain_blur"))
    doc = _multi_doc(media, clip)

    receipt = engine.render(doc, clip, media["main"], tmp_path / "clip.mp4")

    span = span_commands(receipt)[0]
    assert input_count(span) == 1
    assert "[0:v]split=2" in span
    assert "boxblur" in span


def test_multi_span_clip_concatenates(engine, media, tmp_path):
    """Layout change mid-clip: one span per layout, concatenated, then one
    continuous audio stream over the whole thing."""
    clip = _clip(
        [
            FocusLayout(at=0.0, region="cam"),
            HeroInsetLayout(at=1.0, hero="slides_frame", inset="cam"),
        ],
        start=1.0,
        end=3.0,
    )
    doc = _multi_doc(media, clip)
    out = tmp_path / "clip.mp4"

    receipt = engine.render(doc, clip, media["main"], out)

    spans = span_commands(receipt)
    assert len(spans) == 2
    assert input_count(spans[0]) == 1  # focus on `main` alone
    assert input_count(spans[1]) == 2  # hero_inset spans both files
    assert any("-f concat" in c for c in receipt.commands)

    info = probe(out)
    assert (info["width"], info["height"]) == OUTPUT_SIZE
    assert info["duration"] == pytest.approx(clip.duration, abs=0.2)
    assert info["has_audio"]


def test_input_order_is_deterministic(engine, media, tmp_path):
    """The receipt records the command; identical inputs must produce an
    identical command, or a re-render cannot be checked against it."""
    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam"))
    doc = _multi_doc(media, clip)

    first = span_commands(engine.render(doc, clip, media["main"], tmp_path / "a.mp4"))
    second = span_commands(engine.render(doc, clip, media["main"], tmp_path / "b.mp4"))

    assert [inputs_and_graph(c) for c in first] == [inputs_and_graph(c) for c in second]
    # Inputs follow first use among the back-to-front placements, so the hero's
    # file is input 0.
    files, _ = inputs_and_graph(first[0])
    assert files == [str(media["slides"]), str(media["main"])]


def test_render_terminates(engine, media, tmp_path):
    """Guard against the infinite-canvas class of bug.

    `_guard_ffmpeg` kills any single call that overruns; this pins the whole
    render to a wall-clock budget so a graph that stalls fails loudly instead
    of stopping the suite.
    """
    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam"))
    doc = _multi_doc(media, clip)

    started = time.monotonic()
    engine.render(doc, clip, media["main"], tmp_path / "clip.mp4")
    elapsed = time.monotonic() - started

    assert elapsed < FFMPEG_TIMEOUT, f"render took {elapsed:.1f}s"


def test_relative_source_paths_resolve(engine, media, tmp_path):
    """`SourceSpec.path` as written by hand: relative to the job directory."""
    job = tmp_path / "jobs" / "fixture"
    job.mkdir(parents=True)
    for name, src in media.items():
        shutil.copy(src, job / f"{name}.mp4")

    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam"))
    doc = _multi_doc(media, clip).model_copy(deep=True)
    doc.sources["main"].path = "jobs/fixture/main.mp4"
    doc.sources["slides"].path = "jobs/fixture/slides.mp4"

    paths = engine._resolve_source_paths(doc, job / "main.mp4")

    assert paths["main"] == (job / "main.mp4").resolve()
    assert paths["slides"] == (job / "slides.mp4").resolve()


def test_missing_source_names_itself(engine, media, tmp_path):
    clip = _clip(HeroInsetLayout(hero="slides_frame", inset="cam"))
    doc = _multi_doc(media, clip).model_copy(deep=True)
    doc.sources["slides"].path = "nowhere/absent.mp4"

    with pytest.raises(FileNotFoundError) as excinfo:
        engine.render(doc, clip, media["main"], tmp_path / "clip.mp4")

    assert "slides" in str(excinfo.value)
    assert "absent.mp4" in str(excinfo.value)
