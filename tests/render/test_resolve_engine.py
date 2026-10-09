"""The Resolve engine's decisions, without Resolve.

Everything the engine decides -- transforms, frame windows, tracks, fades,
caption states -- is made by pure functions before the driver runs, so it is
tested here on any machine. The driver itself is exercised only by
`test_resolve_live.py`, which needs a running Resolve Studio.

The transform tests do not check `transform_for` against itself. They push its
output back through the mapping *measured* in Resolve 21.1 (see the
resolve_engine docstring) and assert the source rect lands on the dest rect.
If Resolve ever changes its units, the live test is what will notice; these
pin down the algebra.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from makeshorts.artifacts import Region
from makeshorts.config import OutroConfig, RenderConfig, TransitionConfig
from makeshorts.render import layout as L
from makeshorts.render import resolve_driver as RD
from makeshorts.render import resolve_engine as RE
from makeshorts.render.captions.base import CaptionOverlay
from makeshorts.render.engine import CAP_CONTAIN_BLUR, CAP_FOCUS, available_engines
from makeshorts.render.resolve_engine import (
    ResolveEngine,
    ResolveError,
    ResolveSettings,
    build_plan,
    caption_states,
    fade_alphas,
    transform_for,
)
from makeshorts.select.schema import (
    Clip,
    ClipsDoc,
    CriteriaRef,
    CriterionScore,
    FocusLayout,
    HeroInsetLayout,
    SourceSpec,
    StackLayout,
    Why,
)

OUT = (1080, 1920)
FPS = 30
CRITERIA = CriteriaRef(file="config/criteria.yaml", version="test", sha256="0" * 64)


# --------------------------------------------------------------------------
# The mapping Resolve was measured to apply (Scaling = Crop)
# --------------------------------------------------------------------------


def resolve_maps(props, src, tl, sx, sy):
    """Where Resolve draws source pixel (sx, sy), per the measured model."""
    sw, sh = src
    tw, th = tl
    x = tw / 2 + props["ZoomX"] * (sx - sw / 2) + props["Pan"] * sw / tw
    y = th / 2 + props["ZoomY"] * (sy - sh / 2) - props["Tilt"] * sh / th
    return x, y


def resolve_crop_window(props, src, tl):
    """The source rectangle Resolve leaves visible, in source pixels."""
    sw, sh = src
    k = max(sw / tl[0], sh / tl[1])
    return (props["CropLeft"] * k, props["CropTop"] * k,
            sw - props["CropRight"] * k, sh - props["CropBottom"] * k)


def assert_lands(placement, src, tl=OUT):
    props = transform_for(placement, src, tl)
    s, d = placement.source_rect, placement.dest_rect
    for corner_s, corner_d in (((s.x, s.y), (d.x, d.y)),
                               ((s.right, s.bottom), (d.right, d.bottom))):
        x, y = resolve_maps(props, src, tl, *corner_s)
        assert x == pytest.approx(corner_d[0], abs=1e-6)
        assert y == pytest.approx(corner_d[1], abs=1e-6)
    assert resolve_crop_window(props, src, tl) == pytest.approx(
        (s.x, s.y, s.right, s.bottom), abs=1e-6)
    assert props["Scaling"] == RD.SCALE_CROP


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def _why() -> Why:
    return Why(one_line="fixture", theme="test",
               scores={"hook_strength": CriterionScore(score=5, evidence="n/a")},
               weighted_score=5.0)


def _clip(layout, start=10.0, end=14.0, **kw) -> Clip:
    kw.setdefault("captions", {"enabled": False})
    return Clip(id="01-fixture", title="Fixture", start=start, end=end,
                source_text="fixture", why=_why(), layout=layout, **kw)


def _doc(clip: Clip, tmp_path: Path, *, two_sources=False) -> ClipsDoc:
    main = tmp_path / "main.mp4"
    main.write_bytes(b"")
    regions = [Region(id="cam", kind="speaker", rect=(0.0, 0.0, 0.5, 1.0)),
               Region(id="slides", kind="slide", rect=(0.5, 0.0, 0.5, 1.0))]
    if not two_sources:
        return ClipsDoc(job="fixture", criteria_ref=CRITERIA, clips=[clip],
                        source=SourceSpec(path=str(main), duration=60.0,
                                          resolution=(1920, 1080), regions=regions))
    screen = tmp_path / "screen.mp4"
    screen.write_bytes(b"")
    return ClipsDoc(job="fixture", criteria_ref=CRITERIA, clips=[clip], sources={
        "main": SourceSpec(path=str(main), duration=60.0, resolution=(1280, 720),
                           regions=[regions[0]]),
        "screen": SourceSpec(path=str(screen), duration=60.0, resolution=(800, 600)),
    })


def _plan(doc, clip, tmp_path, **kw):
    spans = L.plan_clip(clip, doc.resolved_regions(), doc, OUT)
    paths = {name: Path(spec.path) for name, spec in doc.resolved_sources().items()}
    out = tmp_path / "out" / "fixture--01-fixture.mp4"
    return build_plan(doc, clip, spans, paths, out, tmp_path / "work",
                      settings=kw.pop("settings", ResolveSettings()), job=doc.job, **kw)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src", [(1920, 1080), (1280, 720), (1080, 1920), (800, 600), (800, 800)])
@pytest.mark.parametrize("layout", [
    FocusLayout(region="cam_a", fit="cover"),
    FocusLayout(region="frame", fit="contain_blur"),
    HeroInsetLayout(hero="slides", inset="cam_a", inset_corner="top_left", inset_scale=0.3),
    StackLayout(regions=["slides", "cam_a"]),
])
def test_every_placement_lands_where_layout_put_it(src, layout, two_up):
    plan = L.compute_placements(layout, two_up, src, OUT)
    for p in plan.placements:
        assert_lands(p, src)


@pytest.mark.parametrize("tl", [(1080, 1920), (1920, 1080), (1200, 1200)])
def test_transform_holds_for_any_timeline_shape(tl, two_up):
    plan = L.compute_placements(HeroInsetLayout(hero="slides", inset="cam_a"), two_up,
                                (1920, 1080), tl)
    for p in plan.placements:
        assert_lands(p, (1920, 1080), tl)


def test_the_measured_fixture_reproduces():
    """The case the model was confirmed with on screen: the green box at
    (960, 270, 480, 540) in a 1920x1080 source drawn at (100, 200, 720, 810)
    of a vertical timeline rendered exactly there."""
    p = L.Placement(region_id="x", source="main",
                    source_rect=L.PixelRect(x=960, y=270, w=480, h=540),
                    dest_rect=L.PixelRect(x=100, y=200, w=720, h=810))
    props = transform_for(p, (1920, 1080), OUT)
    assert props["Pan"] == pytest.approx(-247.5)
    assert props["Tilt"] == pytest.approx(631.111111)
    assert props["CropLeft"] == pytest.approx(540.0)
    assert props["CropTop"] == pytest.approx(151.875)


def test_backdrop_blur_is_scaled_into_source_pixels(two_up):
    plan = L.compute_placements(FocusLayout(region="frame", fit="contain_blur"),
                                two_up, (1920, 1080), OUT,
                                L.LayoutOptions(blur_radius_pct=0.05))
    backdrop, front = plan.placements
    zoom = backdrop.dest_rect.w / backdrop.source_rect.w
    assert RE._blur_size(backdrop) == pytest.approx(backdrop.blur_radius_px / zoom, abs=1e-3)
    assert RE._blur_size(front) == 0.0


# --------------------------------------------------------------------------
# Plan: time and tracks
# --------------------------------------------------------------------------


def test_single_span_is_one_item_one_audio(tmp_path):
    clip = _clip([FocusLayout(region="cam")])
    doc = _doc(clip, tmp_path)
    plan, layers = _plan(doc, clip, tmp_path)

    assert plan["timeline"] == {"name": "fixture--01-fixture", "width": 1080,
                                "height": 1920, "fps": 30, "video_tracks": 1}
    [item] = plan["video"]
    assert (item["track"], item["record"], item["frames"]) == (1, 0, 120)
    assert item["anchor"] == clip.start
    assert plan["audio"]["media"] == "main"
    assert plan["audio"]["frames"] == 120
    assert plan["audio"]["normalize"] == {"mode": "ITU-R BS.1770-4", "target": -14.0}
    assert Path(plan["media"]["main"]["path"]).is_absolute()
    assert plan["render"]["name"] == "fixture--01-fixture"
    assert plan["render"]["ext"] == "mp4"
    assert not layers.captions and not layers.head and not layers.tail
    json.dumps(plan)  # the driver reads it as JSON


def test_normalize_off_is_respected(tmp_path):
    clip = _clip([FocusLayout(region="cam")], audio={"normalize": False})
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path)
    assert plan["audio"]["normalize"] is None


def test_backdrop_sits_below_its_foreground(tmp_path):
    clip = _clip([FocusLayout(region="frame", fit="contain_blur")])
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path)
    back, front = plan["video"]
    assert back["track"] < front["track"]
    assert back["blur"] > 0 and "blur" not in front


def test_spans_tile_with_no_gap_or_overlap_on_shared_tracks(tmp_path):
    clip = _clip([FocusLayout(at=0.0, region="cam"),
                  HeroInsetLayout(at=1.27, hero="slides", inset="cam"),
                  FocusLayout(at=2.9, region="slides")], start=10.013, end=14.0)
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path)
    total = round(clip.duration * FPS)

    bottom = [v for v in plan["video"] if v["track"] == 1]
    assert [v["record"] for v in bottom] == [0, 38, 87]
    for a, b in zip(bottom, bottom[1:]):
        assert a["record"] + a["frames"] == b["record"]
    assert bottom[-1]["record"] + bottom[-1]["frames"] == total
    # No transitions: tracks are reused, not stacked per span.
    assert plan["timeline"]["video_tracks"] == 2
    assert all("fade_out" not in v for v in plan["video"])


def test_a_transition_extends_and_fades_the_outgoing_span_above(tmp_path):
    clip = _clip([FocusLayout(at=0.0, region="cam"),
                  HeroInsetLayout(at=2.0, hero="slides", inset="cam", transition="dissolve")])
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path,
                    transitions=[None, TransitionConfig(type="fade", duration=0.4)])
    out_span = [v for v in plan["video"] if v["record"] == 0]
    in_span = [v for v in plan["video"] if v["record"] == 60]

    assert [v["frames"] for v in out_span] == [72]  # 60 + 12 frames of overlap
    assert [v.get("fade_out") for v in out_span] == [12]
    assert all("fade_out" not in v for v in in_span)
    # Outgoing above incoming, or the dissolve would happen out of sight.
    assert min(v["track"] for v in out_span) > max(v["track"] for v in in_span)
    assert plan["timeline"]["video_tracks"] == 3


def test_a_transition_cannot_outlast_the_span_it_enters(tmp_path):
    clip = _clip([FocusLayout(at=0.0, region="cam"),
                  FocusLayout(at=3.9, region="slides", transition="soft")])
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path,
                    transitions=[None, TransitionConfig(type="fade", duration=2.0)])
    first = next(v for v in plan["video"] if v["record"] == 0)
    assert first["fade_out"] == 2  # the 3-frame span keeps one frame of its own


def test_two_sources_each_cropped_in_their_own_pixels(tmp_path):
    clip = _clip([HeroInsetLayout(hero="screen_frame", inset="cam")])
    doc = _doc(clip, tmp_path, two_sources=True)
    plan, _ = _plan(doc, clip, tmp_path)

    assert set(plan["media"]) == {"main", "screen"}
    spans = L.plan_clip(clip, doc.resolved_regions(), doc, OUT)
    for item, p in zip(plan["video"], spans[0].plan.placements):
        assert item["media"] == p.source
        size = spans[0].plan.size_of(p)
        assert item["props"] == transform_for(p, size, OUT)
    # Audio only ever from the primary source.
    assert plan["audio"]["media"] == "main"


# --------------------------------------------------------------------------
# Plan: fades, captions, outro
# --------------------------------------------------------------------------


def test_fade_curve_matches_ffmpegs():
    head, tail = fade_alphas(100, 4, 5)
    assert head == [1.0, 0.75, 0.5, 0.25]
    assert tail == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_edge_fades_are_one_layer_above_the_video(tmp_path):
    clip = _clip([FocusLayout(region="frame", fit="contain_blur")])
    plan, layers = _plan(_doc(clip, tmp_path), clip, tmp_path, fade_in=0.15, fade_out=0.35)
    fades = [v for v in plan["video"] if v["media"] == "@fades"]
    head, tail = fades

    assert head["record"] == 0 and head["frames"] == 4 and head["src_in"] == 0
    assert tail["record"] + tail["frames"] == 120 and tail["src_in"] == head["frames"]
    assert {head["track"], tail["track"]} == {3}
    assert plan["media"]["@fades"]["end"] == head["frames"] + tail["frames"] - 1
    assert len(layers.head) == head["frames"] and len(layers.tail) == tail["frames"]


def _overlay(png, start, end, x=0, y=0):
    return CaptionOverlay(png_path=str(png), x=x, y=y, width=10, height=10,
                          start=start, end=end)


def test_caption_states_are_half_open_and_ordered(tmp_path):
    a = _overlay("a.png", 0.0, 4.0)       # branding, the whole clip
    b = _overlay("b.png", 0.0, 0.1)
    c = _overlay("c.png", 0.1, 0.2)
    states = caption_states([a, b, c], 7, 30)
    assert states == [(0, 1), (0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0,)]


def test_captions_ride_on_top_of_everything(tmp_path):
    clip = _clip([FocusLayout(region="cam")], captions={"enabled": True})
    ov = [_overlay("a.png", 0.5, 1.0)]
    plan, layers = _plan(_doc(clip, tmp_path), clip, tmp_path, fade_in=0.1, overlays=ov)
    caps = next(v for v in plan["video"] if v["media"] == "@captions")
    assert caps["track"] == max(v["track"] for v in plan["video"])
    assert caps["frames"] == 120 and len(layers.captions) == 120
    assert plan["timeline"]["video_tracks"] == caps["track"]


def test_disabled_captions_draw_nothing(tmp_path):
    clip = _clip([FocusLayout(region="cam")], captions={"enabled": False})
    plan, layers = _plan(_doc(clip, tmp_path), clip, tmp_path,
                         overlays=[_overlay("a.png", 0.0, 1.0)])
    assert "@captions" not in plan["media"] and not layers.captions


def test_outro_follows_the_clip(tmp_path):
    clip = _clip([FocusLayout(region="cam")])
    bumper = tmp_path / "outro.mp4"
    bumper.write_bytes(b"")
    plan, _ = _plan(_doc(clip, tmp_path), clip, tmp_path,
                    outro=OutroConfig(file=str(bumper), audio="mute", max_duration=2.0))
    assert plan["outro"] == {"media": "@outro", "record": 120, "max_seconds": 2.0,
                             "audio": False}
    assert plan["media"]["@outro"]["path"] == str(bumper.resolve())


# --------------------------------------------------------------------------
# Image layers on disk
# --------------------------------------------------------------------------


def test_layers_are_linked_sequences_of_distinct_images(tmp_path):
    png = tmp_path / "cue.png"
    Image.new("RGBA", (40, 20), (255, 255, 0, 255)).save(png)
    layers = RE._Layers(captions=[(), (0,), (0,), ()], head=[1.0, 0.5], tail=[0.0, 0.5])
    work = tmp_path / "work"
    RE._write_layers(layers, [_overlay(png, 0, 1, x=100, y=200)], (320, 480), work)

    caps = sorted((work / "captions").iterdir())
    assert [p.name for p in caps] == [RE.FRAME_PATTERN % k for k in range(4)]
    assert len({os.stat(p).st_ino for p in caps}) == 2  # blank + one caption state
    lit = Image.open(caps[1])
    assert lit.size == (320, 480)
    assert lit.getpixel((110, 205)) == (255, 255, 0, 255)
    assert lit.getpixel((10, 10))[3] == 0

    fades = sorted((work / "fades").iterdir())
    assert [Image.open(p).getpixel((0, 0))[3] for p in fades] == [255, 128, 0, 128]


def test_rerendering_replaces_the_sequence(tmp_path):
    work = tmp_path / "work"
    RE._write_layers(RE._Layers(head=[1.0, 0.5, 0.25]), [], (8, 8), work)
    RE._write_layers(RE._Layers(head=[1.0]), [], (8, 8), work)
    assert len(list((work / "fades").iterdir())) == 1


# --------------------------------------------------------------------------
# Driver arithmetic and the subprocess seam
# --------------------------------------------------------------------------


class _FakeMediaItem:
    def __init__(self, fps):
        self.fps = fps

    def GetClipProperty(self, key):  # noqa: N802 — Resolve's spelling
        return {"FPS": self.fps}[key]


def test_driver_windows_tile_at_matching_frame_rates():
    media = _FakeMediaItem(30.0)
    a = RD._source_window({"anchor": 10.013, "record": 0, "frames": 38}, media, 30.0)
    b = RD._source_window({"anchor": 10.013, "record": 38, "frames": 49}, media, 30.0)
    assert a == (300, 338) and b == (338, 387)


def test_driver_windows_retime_other_frame_rates():
    media = _FakeMediaItem(25.0)
    assert RD._source_window({"anchor": 4.0, "record": 30, "frames": 30}, media, 30.0) == (125, 150)
    assert RD._source_window({"src_in": 7, "frames": 3, "record": 0}, media, 30.0) == (7, 10)


def _fake_run(stdout, stderr=""):
    def run(args, **kw):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=stderr)
    return run


def test_driver_errors_surface_as_resolve_errors(monkeypatch):
    monkeypatch.setattr(RE.subprocess, "run",
                        _fake_run(json.dumps({"ok": False, "error": "Resolve is closed"})))
    with pytest.raises(ResolveError, match="Resolve is closed"):
        ResolveEngine()._drive(["version"], timeout=5)


def test_a_driver_that_dies_says_how(monkeypatch):
    monkeypatch.setattr(RE.subprocess, "run", _fake_run("", stderr="Segmentation fault"))
    with pytest.raises(ResolveError, match="Segmentation fault"):
        ResolveEngine()._drive(["version"], timeout=5)


def test_version_degrades_rather_than_raising(monkeypatch):
    monkeypatch.setattr(RE.subprocess, "run",
                        _fake_run(json.dumps({"ok": False, "error": "not running"})))
    assert ResolveEngine().version().startswith("unavailable")


def test_render_writes_the_plan_and_reports_the_driver(monkeypatch, tmp_path):
    clip = _clip([FocusLayout(region="cam")])
    doc = _doc(clip, tmp_path)
    out = tmp_path / "out" / "fixture--01-fixture.mp4"
    seen = {}

    def drive(self, args, timeout):
        seen["plan"] = json.loads(Path(args[1]).read_text())
        out.write_bytes(b"mp4")
        return {"ok": True, "version": "DaVinci Resolve Studio 21.1", "project": "makeshorts",
                "timeline": out.stem, "items": 1, "audio": True, "frames": 120,
                "output": str(out)}

    monkeypatch.setattr(ResolveEngine, "_drive", drive)
    receipt = ResolveEngine().render(doc, clip, Path(doc.source.path), out)

    assert seen["plan"]["timeline"]["name"] == out.stem
    assert (out.parent / ".resolve" / out.stem / "plan.json").is_file()
    assert receipt.engine == "resolve"
    assert receipt.engine_version == "DaVinci Resolve Studio 21.1"
    assert receipt.duration == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Registration and config
# --------------------------------------------------------------------------


def test_registered_under_its_name():
    assert "resolve" in available_engines()


def test_no_capabilities_without_resolve_installed(monkeypatch):
    monkeypatch.setattr(RE, "script_api_dir", lambda: None)
    assert ResolveEngine().capabilities() == set()


def test_full_capabilities_with_resolve_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(RE, "script_api_dir", lambda: tmp_path)
    caps = ResolveEngine().capabilities()
    assert {CAP_FOCUS, CAP_CONTAIN_BLUR} <= caps


def test_configure_reads_render_yaml():
    cfg = RenderConfig.model_validate({
        "resolve": {"project": "shorts", "codec": "H265", "quality": 0,
                    "keep_timelines": False},
        "audio": {"normalize": {"integrated": -16.0}, "sample_rate": 44100},
    })
    engine = ResolveEngine()
    engine.configure(cfg)
    s = engine.settings
    assert (s.project, s.codec, s.quality, s.keep_timelines) == ("shorts", "H265", 0, False)
    assert (s.normalize_target, s.sample_rate) == (-16.0, 44100)
