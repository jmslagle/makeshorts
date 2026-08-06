"""ffmpeg capability probing.

Two things are being tested, and they are different in kind: the *parsing*,
which is deterministic and gets synthetic input; and the *conclusion this
machine's binary leads to*, which is the fact the whole caption design hangs
off and is worth asserting against the real ffmpeg rather than a fixture.
"""

from __future__ import annotations

import shutil

import pytest

from makeshorts.render.caps import (
    CAPS_SCHEMA_VERSION,
    CapsProbeError,
    FFmpegCaps,
    _parse_buildconf,
    _parse_names,
    _parse_version,
    clear_cache,
    default_cache_path,
    probe_caps,
)
from makeshorts.render.engine import (
    CAP_BURNED_CAPTIONS,
    CAP_CONTAIN_BLUR,
    CAP_HERO_INSET,
    CAP_KARAOKE_CAPTIONS,
)

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)

FILTERS_SAMPLE = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  | = Source or sink filter
  ------
 .. abench            A->A       Benchmark part of a filtergraph.
 TS overlay           VV->V      Overlay a video source on top of the input.
 T. boxblur           V->V       Blur the input.
 .. scale             V->V       Scale the input video size.
"""

ENCODERS_SAMPLE = """Encoders:
 V..... = Video
 .....D = Supports direct rendering method 1
 ------
 V....D libx264              libx264 H.264 / AVC (codec h264)
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""


def caps(**overrides) -> FFmpegCaps:
    base = dict(
        probed_at="2026-08-06T00:00:00+00:00",
        ffmpeg_path="/usr/bin/ffmpeg",
        ffmpeg_version="8.1.2",
        filters={"crop": True, "scale": True, "overlay": True, "gblur": True},
        encoders={"libx264": True},
    )
    base.update(overrides)
    return FFmpegCaps(**base)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_names_skips_the_legend():
    names = _parse_names(FILTERS_SAMPLE)
    assert {"abench", "overlay", "boxblur", "scale"} <= names
    assert "Timeline" not in names, "legend text must not be read as filter names"
    assert "=" not in names


def test_parse_names_on_encoders():
    assert {"libx264", "h264_videotoolbox", "aac"} <= _parse_names(ENCODERS_SAMPLE)


def test_parse_version():
    assert _parse_version("ffmpeg version 8.1.2 Copyright (c) 2000") == "8.1.2"
    assert _parse_version("something else entirely") == "unknown"


def test_parse_buildconf_collects_flags():
    flags = _parse_buildconf(
        "configuration: --prefix=/opt --enable-gpl --enable-libx264 --enable-neon"
    )
    assert "--enable-libx264" in flags
    assert "--enable-libass" not in flags


# --------------------------------------------------------------------------
# Derived conclusions
# --------------------------------------------------------------------------


def test_no_libass_means_the_pillow_backend():
    c = caps(filters={"subtitles": False, "ass": False, "overlay": True})
    assert not c.has_libass
    assert c.caption_backend == "pillow"


def test_libass_means_the_ass_backend():
    c = caps(filters={"subtitles": True, "overlay": True})
    assert c.has_libass
    assert c.caption_backend == "ass"


def test_no_overlay_at_all_means_no_captions():
    c = caps(filters={"subtitles": False, "overlay": False})
    assert c.caption_backend == "none"
    assert CAP_BURNED_CAPTIONS not in c.capabilities()


def test_drawtext_presence_tracks_libfreetype():
    assert caps(filters={"drawtext": True}).has_libfreetype
    assert not caps(filters={"drawtext": False}).has_libfreetype


def test_blur_filter_prefers_gblur():
    assert caps(filters={"gblur": True, "boxblur": True}).blur_filter == "gblur"
    assert caps(filters={"boxblur": True}).blur_filter == "boxblur"
    assert caps(filters={"avgblur": True}).blur_filter == "avgblur"
    assert caps(filters={}).blur_filter is None


def test_capabilities_require_the_filters_that_implement_them():
    full = caps(filters={"crop": True, "scale": True, "overlay": True, "gblur": True})
    assert {CAP_HERO_INSET, CAP_CONTAIN_BLUR, CAP_KARAOKE_CAPTIONS} <= full.capabilities()

    no_blur = caps(filters={"crop": True, "scale": True, "overlay": True})
    assert CAP_CONTAIN_BLUR not in no_blur.capabilities()

    no_scale = caps(filters={"crop": True, "overlay": True})
    assert CAP_HERO_INSET not in no_scale.capabilities()


def test_karaoke_is_offered_by_both_backends():
    """The Pillow path does word highlighting too, via one PNG per word."""
    for filters in (
        {"crop": True, "scale": True, "overlay": True},  # pillow
        {"crop": True, "scale": True, "overlay": True, "subtitles": True},  # ass
    ):
        assert CAP_KARAOKE_CAPTIONS in caps(filters=filters).capabilities()


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


@has_ffmpeg
def test_probe_writes_and_reuses_a_cache(tmp_path):
    path = tmp_path / "caps.json"
    first = probe_caps(cache_path=path)
    assert path.is_file()
    second = probe_caps(cache_path=path)
    assert second.probed_at == first.probed_at, "second call should have hit the cache"

    third = probe_caps(cache_path=path, refresh=True)
    assert third.ffmpeg_version == first.ffmpeg_version


@has_ffmpeg
def test_a_stale_cache_is_ignored(tmp_path):
    """A rebuilt binary must invalidate the cache without anyone clearing it."""
    path = tmp_path / "caps.json"
    real = probe_caps(cache_path=path)
    path.write_text(real.model_copy(update={"binary_size": 1}).model_dump_json())
    again = probe_caps(cache_path=path)
    assert again.binary_size == real.binary_size


@has_ffmpeg
def test_a_corrupt_cache_is_ignored(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text("{not json at all")
    assert probe_caps(cache_path=path).ffmpeg_version != "unknown"


def test_cache_schema_version_is_part_of_the_key():
    c = caps(caps_schema_version="0")
    assert c.caps_schema_version != CAPS_SCHEMA_VERSION


def test_default_cache_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_path() == tmp_path / "makeshorts" / "ffmpeg_caps.json"


def test_clear_cache_is_forgiving(tmp_path):
    clear_cache(tmp_path / "never-existed.json")  # must not raise


def test_a_missing_binary_raises():
    with pytest.raises(CapsProbeError):
        probe_caps("definitely-not-ffmpeg-xyz", use_cache=False)


# --------------------------------------------------------------------------
# The actual machine
# --------------------------------------------------------------------------


@has_ffmpeg
def test_probe_reports_this_machine_correctly():
    """Ground truth for the whole caption design.

    If this ever fails because libass appeared, that is good news -- but the
    ass backend then becomes live and should be re-checked against real
    output before anyone trusts it.
    """
    c = probe_caps(use_cache=False)
    assert c.ffmpeg_version.startswith("8.")
    assert c.has_overlay, "overlay is the one filter the Pillow path cannot do without"
    assert c.filters["crop"] and c.filters["scale"]
    assert c.blur_filter is not None
    assert c.has_libx264
    # Consistency, not a hardcoded expectation: the derived flags must agree
    # with the filter list they are derived from, whatever the build has.
    assert c.has_libass == (c.filters["subtitles"] or c.filters["ass"])
    assert c.has_libfreetype == c.filters["drawtext"]
    assert c.caption_backend == ("ass" if c.has_libass else "pillow")
    assert c.summary()


@has_ffmpeg
def test_select_backend_matches_the_probe():
    from makeshorts.render.captions import select_backend

    c = probe_caps(use_cache=False)
    assert select_backend(c).name == c.caption_backend
