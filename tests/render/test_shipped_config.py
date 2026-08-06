"""The caption styles that actually ship, and the ASS document ffmpeg sees.

`config/styles.yaml` is owned by another part of the project, so this is the
seam between the model here and the file there. A field renamed on either side
should fail here rather than at render time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from makeshorts.render.captions.ass_backend import build_ass
from makeshorts.render.captions.cues import CueOptions, build_cues
from makeshorts.render.captions.pillow_backend import PillowCaptionBackend
from makeshorts.render.captions.style import (
    DEFAULT_STYLE_NAME,
    default_style_name,
    load_styles,
    resolve_font_path,
)

from tests.render.conftest import evenly_spaced

REPO = Path(__file__).resolve().parents[2]
STYLES_YAML = REPO / "config" / "styles.yaml"
OUT = (1080, 1920)

needs_config = pytest.mark.skipif(
    not STYLES_YAML.is_file(), reason="config/styles.yaml has not been written yet"
)


@needs_config
def test_the_shipped_styles_load():
    styles = load_styles(STYLES_YAML)
    assert DEFAULT_STYLE_NAME in styles, "clips.json defaults to pill-karaoke"
    for name, style in styles.items():
        assert style.name == name
        assert resolve_font_path(style.font_file).is_file()


@needs_config
def test_the_shipped_default_names_a_real_style():
    assert default_style_name(STYLES_YAML) in load_styles(STYLES_YAML)


@needs_config
def test_every_shipped_style_renders(tmp_path):
    """Cheap end-to-end over the real config: cues in, PNGs out."""
    doc = evenly_spaced(
        "Eighteen months is the number that kills companies quietly.", per_word=0.4
    )
    for name, style in load_styles(STYLES_YAML).items():
        cues = build_cues(
            doc,
            0.0,
            20.0,
            CueOptions(
                max_chars_per_line=style.max_chars_per_line, max_lines=style.max_lines
            ),
        )
        assets = PillowCaptionBackend().render(cues, style, OUT, tmp_path / name)
        assert assets.overlays, f"{name} produced nothing"
        for o in assets.overlays:
            assert o.x >= 0 and o.x + o.width <= OUT[0], f"{name} overflows horizontally"
            assert o.y >= 0 and o.y + o.height <= OUT[1], f"{name} overflows vertically"


# --------------------------------------------------------------------------
# ASS validation
#
# This ffmpeg cannot *render* ASS -- that is libass's job -- but libavformat
# still demuxes and decodes it, so the file can be checked against a real
# parser rather than only against these tests' idea of the format.
# --------------------------------------------------------------------------

has_ffprobe = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe is not installed"
)


@has_ffprobe
def test_ffmpeg_parses_the_generated_ass(tmp_path):
    from makeshorts.render.captions.style import DEFAULT_STYLES

    doc = evenly_spaced(
        "Eighteen months is the number that kills companies quietly. "
        "Nobody checks it and that is the problem.",
        per_word=0.4,
    )
    cues = build_cues(doc, 0.0, 30.0, CueOptions())
    path = tmp_path / "cues.ass"
    path.write_text(build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT), encoding="utf-8")

    proc = subprocess.run(
        [
            "ffprobe", "-hide_banner", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_packets",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"ffprobe rejected the file:\n{proc.stderr}"
    assert not proc.stderr.strip(), f"ffprobe warned about the file:\n{proc.stderr}"

    data = json.loads(proc.stdout)
    assert data["streams"][0]["codec_name"] == "ass"
    # One decoded packet per Dialogue line: the events really did parse.
    assert len(data["packets"]) == len(cues)


@has_ffprobe
def test_ffmpeg_reads_back_the_cue_timings(tmp_path):
    from makeshorts.render.captions.style import DEFAULT_STYLES

    doc = evenly_spaced("one two three four five six seven eight", per_word=0.5)
    cues = build_cues(doc, 0.0, 10.0, CueOptions())
    path = tmp_path / "cues.ass"
    path.write_text(build_ass(cues, DEFAULT_STYLES["pill-karaoke"], OUT), encoding="utf-8")

    proc = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error", "-print_format", "json",
         "-show_packets", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    packets = json.loads(proc.stdout)["packets"]
    for cue, pkt in zip(cues, packets, strict=True):
        assert float(pkt["pts_time"]) == pytest.approx(cue.start, abs=0.01)
        assert float(pkt["duration_time"]) == pytest.approx(cue.duration, abs=0.02)
