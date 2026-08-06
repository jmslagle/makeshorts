"""Region detection against synthesized video whose layout we chose.

Every fixture is built here with `ffmpeg -f lavfi`, so the ground truth is not
an opinion: we know the speaker is the left half because we put it there, and
we know the deck changed at 7s and 14s because we concatenated it that way.
Nothing binary is committed.

Two moving sources are used deliberately. `testsrc2` animates continuously,
which is what a camera does. `smptehdbars` is pixel-identical frame to frame
and is swapped for a transformed copy of itself at known instants, which is
what a slide deck does. That contrast is the entire signal the detector reads.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from makeshorts.artifacts import RegionsDoc
from makeshorts.prepare.regions import (
    RegionParams,
    _activity_split,
    _change_points,
    _grow_into_frozen,
    _snap,
    analyze,
    detect_content_box,
    detect_regions,
    explain,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FPS = 25
# One tile of the default 8x8 grid. Rects are quantized to it, so this is the
# natural unit for "close enough".
TILE = 1.0 / 8
RECT_TOL = TILE
# Uniform sampling of a 21s clip lands ~11 frames a second; a change point
# cannot be off by more than an interval or two.
TIME_TOL = 0.3


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _encode(path, inputs: list[str], filter_complex: str, duration: float, gop: int = 0) -> str:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for spec in inputs:
        cmd += ["-f", "lavfi", "-i", spec]
    cmd += ["-filter_complex", filter_complex, "-map", "[out]"]
    if gop:
        cmd += ["-g", str(gop)]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-t", str(duration), str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


# Every deck below is three static bar patterns concatenated: static within a
# segment and completely different across one, so it changes exactly twice, at
# known instants. `negate` and `hflip` keep the spatial detail (a flat colour
# would read as an empty tile) while making each segment a different image.


@pytest.fixture(scope="session")
def two_up(tmp_path_factory):
    """testsrc2 | stepping deck, split exactly down the middle.

    speaker = [0, 0, .5, 1]; slide = [.5, 0, .5, 1]; changes at 7s and 14s.
    """
    path = tmp_path_factory.mktemp("regions") / "two_up.mp4"
    inputs = [f"testsrc2=s=640x720:r={FPS}:d=21"] + [f"smptehdbars=s=640x720:r={FPS}:d=7"] * 3
    fc = "[2:v]negate[b];[3:v]hflip[c];[1:v][b][c]concat=n=3:v=1:a=0[deck];[0:v][deck]hstack=inputs=2[out]"
    return _encode(path, inputs, fc, 21)


@pytest.fixture(scope="session")
def two_up_dense_keyframes(tmp_path_factory):
    """The same layout with a keyframe every 13 frames.

    13 and testsrc2's 25-frame animation period are coprime, so keyframes fall
    at every phase of the animation. That matters: a keyframe interval that
    divides the source's period samples the same instant of the loop every
    time and makes a moving picture look frozen.
    """
    path = tmp_path_factory.mktemp("regions") / "two_up_g13.mp4"
    inputs = [f"testsrc2=s=640x720:r={FPS}:d=21"] + [f"smptehdbars=s=640x720:r={FPS}:d=7"] * 3
    fc = "[2:v]negate[b];[3:v]hflip[c];[1:v][b][c]concat=n=3:v=1:a=0[deck];[0:v][deck]hstack=inputs=2[out]"
    return _encode(path, inputs, fc, 21, gop=13)


@pytest.fixture(scope="session")
def single_speaker(tmp_path_factory):
    """One moving source filling the frame. There is nothing to decompose."""
    path = tmp_path_factory.mktemp("regions") / "single.mp4"
    return _encode(path, [f"testsrc2=s=1280x720:r={FPS}:d=12"], "[0:v]copy[out]", 12)


@pytest.fixture(scope="session")
def slide_dominant(tmp_path_factory):
    """Stepping deck full frame with a small camera inset bottom-right.

    speaker = [.75, .75, .25, .25]; slide = the whole frame; changes at 4s, 8s.
    """
    path = tmp_path_factory.mktemp("regions") / "slide_dominant.mp4"
    inputs = [f"smptehdbars=s=1280x720:r={FPS}:d=4"] * 3 + [f"testsrc2=s=320x180:r={FPS}:d=12"]
    fc = (
        "[1:v]negate[b];[2:v]hflip[c];[0:v][b][c]concat=n=3:v=1:a=0[deck];"
        "[deck][3:v]overlay=x=960:y=540[out]"
    )
    return _encode(path, inputs, fc, 12)


@pytest.fixture(scope="session")
def pillarboxed(tmp_path_factory):
    """The two-up layout as 4:3 content pillarboxed into a 16:9 frame.

    Content occupies x in [.125, .875]; speaker = [.125, 0, .375, 1].
    """
    path = tmp_path_factory.mktemp("regions") / "pillar.mp4"
    inputs = [f"testsrc2=s=480x720:r={FPS}:d=21"] + [f"smptehdbars=s=480x720:r={FPS}:d=7"] * 3
    fc = (
        "[2:v]negate[b];[3:v]hflip[c];[1:v][b][c]concat=n=3:v=1:a=0[deck];"
        "[0:v][deck]hstack=inputs=2,pad=1280:720:160:0:black[out]"
    )
    return _encode(path, inputs, fc, 21)


@pytest.fixture(scope="session")
def audio_only(tmp_path_factory):
    path = tmp_path_factory.mktemp("regions") / "audio.m4a"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "5", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def only(regions, kind):
    matching = [r for r in regions if r.kind == kind]
    assert len(matching) == 1, f"expected one {kind}, got {[(r.id, r.kind, r.rect) for r in regions]}"
    return matching[0]


def assert_rect(actual, expected, tol=RECT_TOL, what=""):
    assert all(abs(a - e) <= tol for a, e in zip(actual, expected)), (
        f"{what} rect {tuple(round(v, 3) for v in actual)} not within {tol} of {expected}"
    )


def assert_times(actual, expected, tol=TIME_TOL, what=""):
    assert len(actual) == len(expected), f"{what} expected {len(expected)} change points, got {actual}"
    for a, e in zip(actual, expected):
        assert abs(a - e) <= tol, f"{what} change point {a} not within {tol} of {e}"


# --------------------------------------------------------------------------
# The three layouts
# --------------------------------------------------------------------------


def test_two_up_splits_speaker_from_deck(two_up):
    doc = detect_regions(two_up)

    assert doc.grid == "8x8"
    assert doc.frames_sampled > 100
    assert len(doc.regions) == 2, [(r.id, r.kind, r.rect) for r in doc.regions]

    speaker = only(doc.regions, "speaker")
    slide = only(doc.regions, "slide")
    assert_rect(speaker.rect, (0.0, 0.0, 0.5, 1.0), what="speaker")
    assert_rect(slide.rect, (0.5, 0.0, 0.5, 1.0), what="slide")

    # The boundary is the load-bearing number: it must land on the seam we
    # created, not merely somewhere in the middle.
    assert abs(speaker.rect[0] + speaker.rect[2] - 0.5) <= 0.02
    assert abs(slide.rect[0] - 0.5) <= 0.02

    assert_times(slide.change_points, [7.0, 14.0], what="deck")
    assert speaker.change_points == []
    assert speaker.motion > slide.motion * 5


def test_single_speaker_is_not_split(single_speaker):
    doc = detect_regions(single_speaker)

    assert len(doc.regions) <= 1, (
        f"a single full-frame source was decomposed into {[(r.id, r.rect) for r in doc.regions]}"
    )
    if doc.regions:
        region = doc.regions[0]
        assert region.kind == "speaker"
        assert_rect(region.rect, (0.0, 0.0, 1.0, 1.0), what="single speaker")
        # Specifically, not half the frame.
        assert region.rect[2] > 0.8 and region.rect[3] > 0.8


def test_slide_dominant_finds_the_inset(slide_dominant):
    doc = detect_regions(slide_dominant)

    speaker = only(doc.regions, "speaker")
    slide = only(doc.regions, "slide")

    assert_rect(speaker.rect, (0.75, 0.75, 0.25, 0.25), what="inset speaker")
    # The deck really does span the whole frame; the camera sits on top of it.
    assert slide.rect[2] > 0.8 and slide.rect[3] > 0.8
    assert speaker.rect[2] * speaker.rect[3] < 0.15

    assert_times(slide.change_points, [4.0, 8.0], what="deck")


def test_pillarboxed_source_is_measured_on_its_content(pillarboxed):
    """Bars must be stripped before gridding, and rects reported against the
    full source frame — not against the content box."""
    analysis = analyze(pillarboxed)
    x0, _, x1, _ = analysis.crop
    assert x0 > 0 and x1 < analysis.grid.frame_w, f"pillarbox not detected: {analysis.crop}"

    speaker = only(analysis.regions, "speaker")
    slide = only(analysis.regions, "slide")
    assert_rect(speaker.rect, (0.125, 0.0, 0.375, 1.0), what="speaker")
    assert_rect(slide.rect, (0.5, 0.0, 0.375, 1.0), what="slide")
    # Nothing may be proposed inside the black bars.
    assert speaker.rect[0] >= 0.1
    assert slide.rect[0] + slide.rect[2] <= 0.9


# --------------------------------------------------------------------------
# Sampling strategies and degenerate input
# --------------------------------------------------------------------------


def test_keyframe_sampling_reaches_the_same_answer(two_up_dense_keyframes):
    """Long sources are sampled by decoding keyframes only. The layout must
    come out the same, and confidence should drop with the frame count."""
    params = RegionParams(keyframe_min_duration=1.0)
    analysis = analyze(two_up_dense_keyframes, params=params)

    assert analysis.sample.method == "keyframes"
    assert 24 <= analysis.sample.count < 100

    speaker = only(analysis.regions, "speaker")
    slide = only(analysis.regions, "slide")
    assert_rect(speaker.rect, (0.0, 0.0, 0.5, 1.0), what="speaker")
    assert_rect(slide.rect, (0.5, 0.0, 0.5, 1.0), what="slide")
    # Sampling every ~0.5s, a change point can be half an interval out.
    assert_times(slide.change_points, [7.0, 14.0], tol=0.6, what="deck")

    dense = only(detect_regions(two_up_dense_keyframes).regions, "speaker")
    assert speaker.confidence < dense.confidence


def test_audio_only_source_yields_no_regions(audio_only):
    doc = detect_regions(audio_only)
    assert doc.frames_sampled == 0
    assert doc.regions == []


def test_output_validates_and_ids_are_well_formed(two_up, slide_dominant):
    for path in (two_up, slide_dominant):
        doc = detect_regions(path)
        round_tripped = RegionsDoc.model_validate(json.loads(doc.model_dump_json()))
        assert round_tripped == doc
        assert len({r.id for r in doc.regions}) == len(doc.regions)
        for region in doc.regions:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", region.id), region.id
            assert 0.0 < region.confidence < 1.0
            x, y, w, h = region.rect
            assert 0.0 <= x and 0.0 <= y and w > 0 and h > 0
            assert x + w <= 1.0001 and y + h <= 1.0001


def test_explain_names_every_region(two_up):
    analysis = analyze(two_up)
    text = explain(analysis)
    for region in analysis.regions:
        assert region.id in text
    assert "activity gap" in text


# --------------------------------------------------------------------------
# The pieces, without ffmpeg
# --------------------------------------------------------------------------


def test_activity_split_finds_a_bimodal_gap():
    values = np.array([0.008] * 32 + [0.3] * 32)
    threshold, separation = _activity_split(values, RegionParams())
    assert separation > 10
    assert 0.008 < threshold < 0.3


def test_activity_split_refuses_a_unimodal_distribution():
    """A single camera filling the frame. Inventing a boundary here would be
    the worst thing this module could do, so the gap must read as absent."""
    values = np.linspace(0.10, 0.29, 64)
    _, separation = _activity_split(values, RegionParams())
    assert separation < RegionParams().separation_min


def test_content_box_strips_bars_but_not_dark_content():
    frame = np.zeros((100, 200), dtype=np.uint8)
    frame[10:90, 20:180] = 200
    frames = np.stack([frame, frame])
    assert detect_content_box(frames, RegionParams()) == (20, 10, 180, 90)

    # A dark-but-real image is not a bar: one bright pixel per row is enough.
    dark = np.full((100, 200), 5, dtype=np.uint8)
    dark[:, 0] = 200
    dark[:, -1] = 200
    assert detect_content_box(np.stack([dark, dark]), RegionParams()) == (0, 0, 200, 100)


def test_content_box_refuses_an_implausibly_large_crop():
    """Most of the frame being dark means a dark video, not a huge matte."""
    frame = np.zeros((100, 200), dtype=np.uint8)
    frame[45:55, 95:105] = 255
    assert detect_content_box(np.stack([frame, frame]), RegionParams()) == (0, 0, 200, 100)


def test_snapping_recovers_thirds_but_leaves_odd_edges_alone():
    params = RegionParams()
    tol = params.snap_tol_tiles / 8
    assert _snap(0.375, tol, params) == (pytest.approx(1 / 3), True)
    assert _snap(0.5, tol, params) == (0.5, True)
    assert _snap(0.19, tol, params) == (0.19, False)


def test_frozen_tiles_never_bridge_two_regions():
    """A still gutter between a camera and a deck must not glue them together.

    This is the failure the two-phase fill exists to prevent: growing frozen
    tiles by union-find merged every unrelated motionless area in the frame
    into one meaningless rectangle spanning the whole picture.
    """

    class _Grid:
        rows, cols = 1, 5

    labels = np.array([["moving", "moving", "frozen", "static", "static"]], dtype=object)
    components = [[(0, 0), (0, 1)], [(0, 3), (0, 4)]]
    _grow_into_frozen(_Grid(), labels, components)

    assert len(components) == 2
    assert sum(len(c) for c in components) == 5
    assert (0, 2) in components[0] or (0, 2) in components[1]


def test_change_points_collapse_a_run_into_one_event():
    """A crossfade spans several sampled intervals but is one change."""

    class _Grid:
        delta = np.zeros((10, 1, 1), dtype=np.float32)
        times = np.arange(11, dtype=np.float64)

    _Grid.delta[4:7, 0, 0] = 0.5
    points = _change_points(_Grid(), [(0, 0)], RegionParams())
    assert points == [pytest.approx(5.5)]


def test_change_points_ignore_a_continuously_moving_region():
    class _Grid:
        rng = np.random.default_rng(0)
        delta = rng.uniform(0.05, 0.15, size=(200, 1, 1)).astype(np.float32)
        times = np.arange(201, dtype=np.float64)

    assert _change_points(_Grid(), [(0, 0)], RegionParams()) == []
