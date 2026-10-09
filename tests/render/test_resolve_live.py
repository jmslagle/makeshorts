"""The Resolve engine against a running DaVinci Resolve Studio.

Opt-in, because it drives a real application on the desktop:

    MS_RESOLVE_LIVE=1 uv run pytest tests/render/test_resolve_live.py

It builds in a project named `makeshorts-test` (left behind, empty of
timelines) and puts back whichever project was open. What it guards is the
one thing the pure tests cannot: that Resolve still means by Pan, Tilt, Zoom
and Crop what `transform_for` assumes. Those units are measured, not
documented, so a Resolve update could move them; if it does, this is where a
box stops landing where layout put it.

The fixture is synthesized with `-f lavfi`: a grey 1920x1080 frame with a red
square in its top-left corner, which the clip shows as an inset over the
whole frame letterboxed. A transform that is off by any real amount puts grey
where red should be.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from makeshorts.artifacts import Region
from makeshorts.render import layout as L
from makeshorts.render.resolve_engine import ResolveEngine, ResolveSettings
from makeshorts.select.schema import (
    Clip,
    ClipsDoc,
    CriteriaRef,
    CriterionScore,
    HeroInsetLayout,
    SourceSpec,
    Why,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
OUT = (1080, 1920)


def _reachable() -> bool:
    if os.environ.get("MS_RESOLVE_LIVE") != "1" or not (FFMPEG and FFPROBE):
        return False
    return not ResolveEngine().version().startswith("unavailable")


pytestmark = pytest.mark.skipif(
    not _reachable(), reason="set MS_RESOLVE_LIVE=1 with Resolve Studio running"
)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("resolve") / "src.mp4"
    subprocess.run([
        FFMPEG, "-hide_banner", "-nostdin", "-y",
        "-f", "lavfi", "-i",
        "color=c=gray:s=1920x1080:r=30:d=4,drawbox=x=0:y=0:w=200:h=200:c=red:t=fill",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True, capture_output=True, timeout=120)
    return path


def _doc(source: Path) -> tuple[ClipsDoc, Clip]:
    clip = Clip(
        id="01-live", title="Live", start=1.0, end=3.0, source_text="live",
        why=Why(one_line="x", theme="t", weighted_score=5.0,
                scores={"hook_strength": CriterionScore(score=5, evidence="n/a")}),
        layout=[HeroInsetLayout(hero="frame", hero_fit="contain_blur", inset="red",
                                inset_corner="bottom_right", inset_scale=0.3)],
        captions={"enabled": False},
    )
    doc = ClipsDoc(
        job="live", criteria_ref=CriteriaRef(file="c", version="t", sha256="0" * 64),
        clips=[clip],
        source=SourceSpec(path=str(source), duration=4.0, resolution=(1920, 1080),
                          regions=[Region(id="red", kind="unknown",
                                          rect=(0.0, 0.0, 200 / 1920, 200 / 1080))]),
    )
    return doc, clip


def test_inset_lands_where_layout_put_it(source, tmp_path):
    doc, clip = _doc(source)
    out = tmp_path / "out" / "live--01-live.mp4"
    engine = ResolveEngine(ResolveSettings(project="makeshorts-test",
                                           keep_timelines=False, restore_project=True))

    receipt = engine.render(doc, clip, source, out)

    probe = json.loads(subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == OUT
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    assert float(probe["format"]["duration"]) == pytest.approx(clip.duration, abs=0.1)
    assert receipt.duration == pytest.approx(clip.duration, abs=1e-6)

    frame = tmp_path / "frame.png"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "1.0", "-i", str(out),
                    "-frames:v", "1", str(frame)], check=True)
    img = Image.open(frame).convert("RGB")

    [span] = L.plan_clip(clip, doc.resolved_regions(), doc, OUT)
    inset = span.plan.placements[-1].dest_rect

    def red(x, y):
        r, g, b = img.getpixel((x, y))
        return r > 150 and g < 90 and b < 90

    m = 4  # stay clear of the encoder's chroma bleed at the edges
    for x, y in ((inset.x + m, inset.y + m), (inset.right - m, inset.bottom - m),
                 (inset.x + inset.w // 2, inset.y + inset.h // 2)):
        assert red(x, y), f"expected the inset at ({x}, {y})"
    for x, y in ((inset.x - m, inset.y + inset.h // 2), (inset.x + inset.w // 2, inset.y - m)):
        assert not red(x, y), f"inset spills past its rect at ({x}, {y})"
