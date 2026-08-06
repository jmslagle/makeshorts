"""Enforce the hard requirement: the render engine is swappable.

Two independent checks, because the requirement can be broken two ways.

1. *Layering* -- `select/` and `prepare/` must not import or mention the
   renderer. A leak here means the edit list has quietly grown a dependency on
   ffmpeg even if clips.json still looks clean.

2. *Vocabulary in the edit list* -- a fully-populated clips.json must contain no
   codec, filter, pixel geometry, or file path belonging to a specific engine.
   This is the check that actually protects the artifact.

These are cheap to run and catch the leak on the day it happens rather than the
day someone tries to write a second engine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Vocabulary that belongs to a rendering backend and must never appear upstream
# of it. Deliberately includes the *shapes* of ffmpeg filter syntax, not just
# the word "ffmpeg" -- `crop=w:h:x:y` is the kind of thing that leaks first.
ENGINE_VOCABULARY = [
    r"\bffmpeg\b",
    r"\bffprobe\b",
    r"\blibx264\b",
    r"\blibx265\b",
    r"\bvideotoolbox\b",
    r"\blibass\b",
    r"\bcrf\b",
    r"-vf\b",
    r"-filter_complex\b",
    r"\bcrop=",
    r"\bscale=",
    r"\boverlay=",
    r"\bvstack\b",
    r"\bhstack\b",
    r"\bboxblur\b",
    r"\bdrawtext\b",
    r"\bsilencedetect\b",
    r"\byuv420p\b",
]

# prepare/ legitimately shells out to ffmpeg/ffprobe -- it is a mechanical
# ingest stage, not part of the edit list. What matters is that `select/` (the
# editorial contract) stays clean, and that no module upstream of the engine
# emits filter syntax.
LAYERING_EXEMPT = {"prepare"}


def _python_files(package: str) -> list[Path]:
    return sorted((REPO / "makeshorts" / package).rglob("*.py"))


def _strip_comments_and_docstrings(src: str) -> str:
    """Crude but adequate: we care about executable references, and a docstring
    explaining *why* ffmpeg is absent should not fail the test."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


@pytest.mark.parametrize("package", ["select"])
def test_editorial_layer_has_no_engine_vocabulary(package: str) -> None:
    """`select/` decides WHAT to cut. It must not know HOW it gets rendered."""
    offenders: list[str] = []
    for path in _python_files(package):
        body = _strip_comments_and_docstrings(path.read_text())
        for pattern in ENGINE_VOCABULARY:
            for m in re.finditer(pattern, body):
                line = body[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO)}:{line}: {m.group(0)!r}")
    assert not offenders, (
        "engine vocabulary leaked into the editorial layer:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("package", ["select", "prepare"])
def test_no_upstream_package_imports_the_render_engine(package: str) -> None:
    """Import-level layering. `render/` may import `select/` (it consumes the
    edit list); the reverse would make the edit list engine-dependent."""
    offenders = [
        f"{p.relative_to(REPO)}"
        for p in _python_files(package)
        if re.search(r"^\s*(from|import)\s+makeshorts\.render", p.read_text(), re.M)
    ]
    assert not offenders, (
        f"{package}/ must not import makeshorts.render: {offenders}"
    )


def test_clips_json_contains_no_engine_concepts() -> None:
    """The artifact itself. Walk every key and string value of a fully-populated
    edit list and assert nothing engine-specific appears.

    Style and layout names are semantic and expected; a font path, codec name,
    or filter string is not.
    """
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(
        {
            "job": "agnostic-check",
            "source": {
                "path": "jobs/agnostic-check/source.mp4",
                "duration": 3600.0,
                "resolution": [1920, 1080],
                "regions": [
                    {"id": "cam_a", "kind": "speaker", "rect": [0.0, 0.0, 0.5, 1.0]},
                    {"id": "slides", "kind": "slide", "rect": [0.5, 0.0, 0.5, 1.0]},
                ],
            },
            "criteria_ref": {
                "file": "config/criteria.yaml",
                "version": "2026-08-06",
                "sha256": "0" * 64,
            },
            "clips": [
                {
                    "id": "01-example",
                    "title": "Example",
                    "start": 100.0,
                    "end": 145.0,
                    "source_text": "text",
                    "visual_dependency": True,
                    "speaker": "cam_a",
                    "why": {
                        "one_line": "why",
                        "theme": "t",
                        "scores": {"hook_strength": {"score": 5, "evidence": "e"}},
                        "weighted_score": 5.0,
                    },
                    "layout": [
                        {"at": 0.0, "mode": "focus", "region": "cam_a", "fit": "cover"},
                        {
                            "at": 12.0,
                            "mode": "hero_inset",
                            "hero": "slides",
                            "inset": "cam_a",
                            "inset_corner": "bottom_right",
                        },
                        {"at": 30.0, "mode": "stack", "regions": ["cam_a", "slides"]},
                    ],
                }
            ],
        }
    )

    blob = json.dumps(doc.model_dump(mode="json")).lower()
    for pattern in ENGINE_VOCABULARY:
        assert not re.search(pattern, blob), (
            f"engine concept {pattern!r} appeared in clips.json -- the edit list "
            "must stay engine-agnostic"
        )

    # Positive assertions: the semantic vocabulary that SHOULD survive.
    for expected in ["focus", "hero_inset", "stack", "cam_a", "slides", "cover"]:
        assert expected in blob, f"expected semantic term {expected!r} missing"


def test_frame_region_is_always_resolvable() -> None:
    """The degenerate whole-frame layout must work without any detected region,
    since that is what makes a separate `full` layout mode unnecessary."""
    from makeshorts.artifacts import IMPLICIT_FRAME_REGION_ID
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(
        {
            "job": "bare",
            "source": {"path": "s.mp4", "duration": 60.0, "resolution": [1920, 1080]},
            "criteria_ref": {"file": "c.yaml", "version": "1", "sha256": "0" * 64},
            "clips": [],
        }
    )
    regions = doc.resolved_regions()
    assert IMPLICIT_FRAME_REGION_ID in regions
    assert regions[IMPLICIT_FRAME_REGION_ID].rect == (0.0, 0.0, 1.0, 1.0)
