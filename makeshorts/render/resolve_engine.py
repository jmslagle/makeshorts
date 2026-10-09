"""The DaVinci Resolve render engine.

Builds each clip as a real Resolve timeline -- one track per layer, transforms
on the Inspector's own controls -- renders it, and leaves the timeline in a
dedicated project so a clip can be opened and finished by hand. That last part
is the point of having this engine at all: ffmpeg's output is a file, Resolve's
is a file *and* an edit you can keep working on.

## Shape

    resolve_engine (this module)   pure: edit list + layout plan -> JSON plan
    resolve_driver                 thin: JSON plan -> Resolve API calls

The driver runs in a child process because Resolve's scripting module crashes
during interpreter teardown once it has been used; see its docstring. That
split also puts every decision on this side, where it is tested without
Resolve installed.

## Geometry

`layout.plan_clip()` already decided which source pixels go where. What is
left is expressing a `Placement` in Resolve's Transform and Cropping controls,
whose units are not pixels and are documented nowhere. Measured against
Resolve Studio 21.1 by rendering synthetic sources of known geometry, with the
item's Scaling set to Crop (source pixels 1:1, centred):

    timeline_x = TW/2 + ZoomX * (sx - SW/2) + Pan  * SW/TW
    timeline_y = TH/2 + ZoomY * (sy - SH/2) - Tilt * SH/TH

    one Crop unit = max(SW/TW, SH/TH) source pixels, masking only

(SW, SH: the source file; TW, TH: the timeline.) Pan and Tilt are scaled by
the source-to-timeline ratio on their *own* axis; crop by the larger of the
two, i.e. in pixels of the source as scaled-to-fit. Crop does not move the
image. `transform_for` is the inverse of the above, and is tested against it.

## Layers

    tracks 1..n   video, one track per placement, back to front
    next          edge fades: black frames with falling/rising alpha
    top           captions and branding, one full-frame image per frame

Captions are an image *sequence*, not one still per cue: Resolve ignores
`endFrame` for a still and lays it down at the project's default still
duration. A sequence honours it, and each output frame is a hard link to the
image for the caption state on screen at that instant, so a 45-second clip is
~1350 links to perhaps 150 distinct images.

Crossfades between layout spans are opacity fades on the outgoing span, which
sits above the incoming one -- so with transitions, each span gets its own
block of tracks, earliest on top. Every named transition renders as a
dissolve; xfade's wipes and slides have no counterpart that works across
several tracks. On a span with a backdrop and a foreground, both layers fade,
so the backdrop shows faintly through the foreground mid-dissolve.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import layout as L
from .engine import (
    CAP_BURNED_CAPTIONS,
    CAP_CONTAIN_BLUR,
    CAP_FOCUS,
    CAP_HERO_INSET,
    CAP_KARAOKE_CAPTIONS,
    CAP_MULTI_SPAN,
    CAP_STACK,
    Receipt,
    register,
)
from .resolve_driver import SCALE_CROP, script_api_dir
from .sources import resolve_source_paths
from ..select.schema import Clip, ClipsDoc

__all__ = [
    "ResolveEngine",
    "ResolveError",
    "ResolveSettings",
    "build_plan",
    "caption_states",
    "fade_alphas",
    "transform_for",
]

# Image-sequence frame names. Five digits covers 55 minutes at 30fps.
FRAME_PATTERN = "f_%05d.png"


class ResolveError(RuntimeError):
    """Resolve could not build or render a clip. Carries the driver's reason."""


@dataclass(frozen=True)
class ResolveSettings:
    """Sourced from config/render.yaml's `resolve:` and `audio:` blocks."""

    project: str = "makeshorts"
    keep_timelines: bool = True
    restore_project: bool = False
    format: str = "mp4"
    codec: str = "H264"
    quality: str | int = 12000
    normalize_mode: str = "ITU-R BS.1770-4"
    normalize_target: float = -14.0
    audio_codec: str = "aac"
    sample_rate: int = 48000
    timeout: float = 1800.0
    # Unsharp amount for the upscaled picture (0 = off). Shared knob with the
    # ffmpeg engine; applied here as a Fusion sharpen on each non-backdrop item.
    sharpen: float = 0.0


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def transform_for(
    placement: L.Placement,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> dict[str, Any]:
    """A placement as Resolve Transform + Cropping properties.

    The inverse of the measured mapping in the module docstring: zoom so the
    source rect is the dest rect's size, pan/tilt so their centres coincide,
    crop to exactly the source rect. Zoom is per axis because layout's even
    rounding leaves the two aspects a fraction of a pixel apart, and ganging
    them would spread that error to an edge.
    """
    sw, sh = source_size
    tw, th = output_size
    s, d = placement.source_rect, placement.dest_rect
    zx, zy = d.w / s.w, d.h / s.h
    pan = (d.x + d.w / 2 - tw / 2 - zx * (s.x + s.w / 2 - sw / 2)) * tw / sw
    tilt = -(d.y + d.h / 2 - th / 2 - zy * (s.y + s.h / 2 - sh / 2)) * th / sh
    unit = max(sw / tw, sh / th)
    return {
        "Scaling": SCALE_CROP,
        "ZoomGang": False,
        "ZoomX": zx,
        "ZoomY": zy,
        "Pan": pan,
        "Tilt": tilt,
        "CropLeft": s.x / unit,
        "CropRight": (sw - s.right) / unit,
        "CropTop": s.y / unit,
        "CropBottom": (sh - s.bottom) / unit,
    }


def _blur_size(placement: L.Placement) -> float:
    """Fusion Blur size for a backdrop, in source pixels.

    Fusion blurs before the item is scaled, so the output-pixel radius layout
    suggests is divided by the zoom. Gaussian rather than ffmpeg's box blur;
    the two read alike at these strengths.
    """
    if not placement.blur_backdrop or placement.blur_radius_px <= 0:
        return 0.0
    zoom = placement.dest_rect.w / placement.source_rect.w
    return round(placement.blur_radius_px / zoom, 3)


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def _frames(seconds: float, fps: int) -> int:
    return int(round(seconds * fps))


def caption_states(overlays: Sequence[Any], n_frames: int, fps: int) -> list[tuple[int, ...]]:
    """For each output frame, the overlays on screen, back to front.

    Frame k shows the instant k/fps, matching how the ffmpeg engine gates an
    overlay on `between(t, start, end)` -- except half-open, so that two cues
    sharing a boundary never both claim the frame on it.
    """
    spans = [(float(_ov(o, "start")), float(_ov(o, "end"))) for o in overlays]
    states = []
    for k in range(n_frames):
        t = k / fps
        states.append(tuple(i for i, (s, e) in enumerate(spans) if s <= t < e))
    return states


def fade_alphas(n_frames: int, fade_in: int, fade_out: int) -> tuple[list[float], list[float]]:
    """Opacity of the black layer over the head and tail fade windows.

    Same discrete curve as ffmpeg's `fade`, so the two engines agree frame for
    frame: a fade-in's frame k has black at 1 - k/N, starting fully black; a
    fade-out's frame k has black at k/N, so its last frame keeps 1/N of the
    picture rather than reaching black.
    """
    fade_in = min(fade_in, n_frames)
    fade_out = min(fade_out, n_frames)
    head = [1.0 - k / fade_in for k in range(fade_in)]
    tail = [k / fade_out for k in range(fade_out)]
    return head, tail


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass
class _Layers:
    """Image sequences to write before the plan can run."""

    captions: list[tuple[int, ...]] = field(default_factory=list)
    head: list[float] = field(default_factory=list)
    tail: list[float] = field(default_factory=list)


def build_plan(
    doc: ClipsDoc,
    clip: Clip,
    spans: Sequence[L.SpanPlan],
    paths: Mapping[str, Path],
    out: Path,
    work: Path,
    *,
    settings: ResolveSettings,
    transitions: Sequence[Any | None] = (),
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    overlays: Sequence[Any] = (),
    outro: Any | None = None,
    job: str = "makeshorts",
) -> tuple[dict[str, Any], _Layers]:
    """Everything the driver needs, decided here.

    Returns the JSON-ready plan and the image layers it refers to, which the
    caller writes to `work` before running the driver. Nothing in this
    function touches the filesystem or Resolve.

    `transitions[i]` is the transition entering span i, or None for a cut --
    the same convention as the ffmpeg engine.
    """
    fps = int(doc.output.fps)
    out_w, out_h = doc.output.width, doc.output.height
    total = _frames(clip.duration, fps)
    trans = list(transitions) + [None] * (len(spans) - len(transitions))

    bounds = [(_frames(sp.start, fps), _frames(sp.end, fps)) for sp in spans]
    bounds[-1] = (bounds[-1][0], total)
    # A transition entering span i is paid for by span i-1, which runs that
    # many frames longer and fades out over them. Clamped so a dissolve can
    # never outlast the span it dissolves into.
    overlap = [0] * len(spans)
    for i in range(1, len(spans)):
        if trans[i] is not None:
            overlap[i] = max(0, min(_frames(trans[i].duration, fps),
                                    bounds[i][1] - bounds[i][0] - 1))

    # Without transitions nothing overlaps, so every span can reuse tracks
    # 1..n and the timeline reads like one a person would build. With them,
    # an outgoing span must sit above the incoming one: blocks, earliest on top.
    layers = [len(sp.plan.placements) for sp in spans]
    if any(overlap):
        base = [1 + sum(layers[i + 1:]) for i in range(len(spans))]
        n_video = sum(layers)
    else:
        base = [1] * len(spans)
        n_video = max(layers)

    video: list[dict[str, Any]] = []
    for i, sp in enumerate(spans):
        start, end = bounds[i]
        extra = overlap[i + 1] if i + 1 < len(spans) else 0
        for k, p in enumerate(sp.plan.placements):
            item: dict[str, Any] = {
                "media": p.source,
                "track": base[i] + k,
                "record": start,
                "frames": end - start + extra,
                "anchor": clip.start,
                "props": transform_for(p, sp.plan.size_of(p), (out_w, out_h)),
                "role": p.role,
                "region": p.region_id,
            }
            blur = _blur_size(p)
            if blur:
                item["blur"] = blur
            # Sharpen the real picture, not the blurred backdrop behind it.
            if settings.sharpen > 0 and p.role != "backdrop":
                item["sharpen"] = settings.sharpen
            if extra:
                item["fade_out"] = extra
            video.append(item)

    media: dict[str, Any] = {
        name: {"path": str(Path(paths[name]).resolve())}
        for name in _unique(p["media"] for p in video)
    }

    layers_out = _Layers()
    next_track = n_video + 1

    head, tail = fade_alphas(total, _frames(fade_in, fps), _frames(fade_out, fps))
    if head or tail:
        layers_out.head, layers_out.tail = head, tail
        n = len(head) + len(tail)
        media["@fades"] = {"sequence": str((work / "fades").resolve() / FRAME_PATTERN),
                          "start": 0, "end": n - 1}
        if head:
            video.append({"media": "@fades", "track": next_track, "record": 0,
                          "frames": len(head), "src_in": 0, "props": {"Scaling": SCALE_CROP}})
        if tail:
            video.append({"media": "@fades", "track": next_track, "record": total - len(tail),
                          "frames": len(tail), "src_in": len(head),
                          "props": {"Scaling": SCALE_CROP}})
        next_track += 1

    if overlays and clip.captions.enabled:
        states = caption_states(overlays, total, fps)
        if any(states):
            layers_out.captions = states
            media["@captions"] = {"sequence": str((work / "captions").resolve() / FRAME_PATTERN),
                                 "start": 0, "end": total - 1}
            video.append({"media": "@captions", "track": next_track, "record": 0,
                          "frames": total, "src_in": 0, "props": {"Scaling": SCALE_CROP}})
            next_track += 1

    primary = doc.primary_source_name
    media.setdefault(primary, {"path": str(Path(paths[primary]).resolve())})
    audio = {
        "media": primary, "track": 1, "record": 0, "frames": total, "anchor": clip.start,
        "normalize": ({"mode": settings.normalize_mode, "target": settings.normalize_target}
                      if clip.audio.normalize else None),
    }

    outro_spec = None
    if outro is not None:
        media["@outro"] = {"path": str(Path(outro.file).resolve())}
        outro_spec = {"media": "@outro", "record": total, "max_seconds": outro.max_duration,
                      "audio": outro.audio != "mute"}

    plan = {
        "project": settings.project,
        "bin": job,
        "keep_timeline": settings.keep_timelines,
        "restore_project": settings.restore_project,
        "timeout": settings.timeout,
        "timeline": {"name": out.stem, "width": out_w, "height": out_h, "fps": fps,
                     "video_tracks": next_track - 1},
        "media": media,
        "video": video,
        "audio": audio,
        "outro": outro_spec,
        "render": {
            "dir": str(out.parent.resolve()), "name": out.stem,
            "ext": out.suffix.lstrip(".") or settings.format,
            "format": settings.format, "codec": settings.codec, "quality": settings.quality,
            "audio_codec": settings.audio_codec, "sample_rate": settings.sample_rate,
            "width": out_w, "height": out_h, "fps": fps,
        },
    }
    return plan, layers_out


# --------------------------------------------------------------------------
# Writing the image layers
# --------------------------------------------------------------------------


def _write_sequence(directory: Path, frames: Sequence[Path]) -> None:
    """Lay out `frames` as f_00000.png, f_00001.png ... by hard link.

    Links rather than copies: most frames repeat the one before. A fresh
    directory every time, because Resolve keys its cache on the path.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for k, src in enumerate(frames):
        dst = directory / (FRAME_PATTERN % k)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def _write_layers(layers: _Layers, overlays: Sequence[Any], size: tuple[int, int],
                  work: Path) -> None:
    from PIL import Image  # noqa: PLC0415 — only needed once a clip is actually rendered

    images = work / "images"
    images.mkdir(parents=True, exist_ok=True)

    if layers.head or layers.tail:
        frames = []
        for alpha in [*layers.head, *layers.tail]:
            a = int(round(alpha * 255))
            path = images / f"black_{a:03d}.png"
            if not path.exists():
                Image.new("RGBA", size, (0, 0, 0, a)).save(path, compress_level=1)
            frames.append(path)
        _write_sequence(work / "fades", frames)

    if layers.captions:
        # Rewritten every render: a caption PNG keeps its path when its style
        # changes, so a cache keyed on paths would serve the old look.
        rendered: dict[tuple[int, ...], Path] = {}
        sources = {i: Image.open(_ov(o, "png_path", "path", "png")).convert("RGBA")
                   for i, o in enumerate(overlays)
                   if any(i in s for s in layers.captions)}
        frames = []
        for state in layers.captions:
            if state not in rendered:
                canvas = Image.new("RGBA", size, (0, 0, 0, 0))
                for i in state:
                    o = overlays[i]
                    canvas.alpha_composite(sources[i], (int(_ov(o, "x", "dest_x")),
                                                        int(_ov(o, "y", "dest_y"))))
                key = hashlib.sha1(repr(state).encode()).hexdigest()[:16]
                path = images / f"cap_{key}.png"
                canvas.save(path, compress_level=1)
                rendered[state] = path
            frames.append(rendered[state])
        _write_sequence(work / "captions", frames)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class ResolveEngine:
    name = "resolve"

    def __init__(self, settings: ResolveSettings | None = None,
                 layout_options: L.LayoutOptions | None = None,
                 python: str | None = None) -> None:
        self.settings = settings or ResolveSettings()
        self.layout_options = layout_options
        # Same presentation policy the CLI hands every engine; empty means hard
        # cuts, no edge fades and no bumper.
        self.transitions: dict[str, Any] = {}
        self.fade_in: float = 0.0
        self.fade_out: float = 0.0
        self.outro: dict[str, Any] = {}
        self.default_outro: str | None = None
        self.python = python or sys.executable
        self._version: str | None = None

    def configure(self, render_cfg: Any) -> None:
        """Take render.yaml. `resolve:` is this engine's own block; loudness
        target and audio format are shared with ffmpeg."""
        r, audio = render_cfg.resolve, render_cfg.audio
        self.settings = replace(
            self.settings,
            project=r.project, keep_timelines=r.keep_timelines,
            restore_project=r.restore_project, format=r.format, codec=r.codec,
            quality=r.quality, normalize_mode=r.normalize_mode, timeout=r.timeout,
            normalize_target=audio.normalize.integrated,
            audio_codec=audio.codec, sample_rate=audio.sample_rate,
            sharpen=render_cfg.sharpen,
        )

    def summary(self) -> str:
        s = self.settings
        return f"{s.format}/{s.codec} quality {s.quality} · project {s.project!r}"

    # -- capabilities -----------------------------------------------------

    def capabilities(self) -> set[str]:
        """Everything, if Resolve's scripting module is installed.

        Deliberately does not contact the running app: `ms caps` lists every
        engine, and must not stall on one that is closed. Whether Resolve is
        actually running is found out when a render starts, and said plainly.
        """
        if script_api_dir() is None:
            return set()
        return {CAP_FOCUS, CAP_HERO_INSET, CAP_STACK, CAP_CONTAIN_BLUR, CAP_MULTI_SPAN,
                CAP_BURNED_CAPTIONS, CAP_KARAOKE_CAPTIONS}

    def version(self) -> str:
        if self._version is None:
            try:
                self._version = self._drive(["version"], timeout=30)["version"]
            except ResolveError as exc:
                return f"unavailable ({exc})"
        return self._version

    # -- driver -----------------------------------------------------------

    def _drive(self, args: list[str], timeout: float) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [self.python, "-m", "makeshorts.render.resolve_driver", *args],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ResolveError(f"Resolve did not answer within {timeout:.0f}s") from None
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        try:
            result = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
            raise ResolveError(f"the Resolve driver failed without a result:\n{tail}") from None
        if not result.get("ok"):
            raise ResolveError(result.get("error", "unknown error"))
        return result

    # -- presentation policy ------------------------------------------------

    def _transition_for(self, span: Any) -> Any:
        """The transition entering this span, or None. Unknown names are a
        hard cut; the CLI warns about them."""
        name = getattr(span, "transition", None)
        if not name or name == "none":
            return None
        return self.transitions.get(name)

    def _outro_for(self, clip: Clip) -> Any:
        name = getattr(clip, "outro", None) or self.default_outro
        if not name or name == "none":
            return None
        preset = self.outro.get(name)
        if preset is None or not preset.file:
            return None
        return preset if Path(preset.file).exists() else None

    # -- entry point ------------------------------------------------------

    def render(self, doc: ClipsDoc, clip: Clip, source: Path, out: Path,
               caption_overlays: list | None = None) -> Receipt:
        out.parent.mkdir(parents=True, exist_ok=True)
        paths = resolve_source_paths(doc, Path(source))
        size = (doc.output.width, doc.output.height)
        spans = L.plan_clip(clip, doc.resolved_regions(), doc, size, self.layout_options)
        overlays = list(caption_overlays or [])

        # Kept after the render, not temporary: the timeline left in Resolve
        # points at these frames, and would go offline without them.
        work = out.parent / ".resolve" / out.stem
        work.mkdir(parents=True, exist_ok=True)

        plan, layers = build_plan(
            doc, clip, spans, paths, out, work,
            settings=self.settings,
            transitions=[self._transition_for(s) for s in clip.spans],
            fade_in=self.fade_in, fade_out=self.fade_out,
            overlays=overlays, outro=self._outro_for(clip),
            job=doc.job,
        )
        _write_layers(layers, overlays, size, work)
        plan_path = work / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

        result = self._drive(["build", str(plan_path)], timeout=self.settings.timeout + 120)
        self._version = result.get("version", self._version)
        written = Path(result["output"])
        if written.resolve() != out.resolve():
            shutil.move(str(written), out)

        return Receipt(
            clip_id=clip.id,
            output=str(out),
            engine=self.name,
            engine_version=self._version or "unknown",
            source_sha256="",
            clip_sha256=_clip_hash(clip),
            duration=result["frames"] / doc.output.fps,
            commands=[
                f"resolve project={result['project']!r} timeline={result['timeline']!r} "
                f"items={result['items']} audio={result['audio']}",
                f"plan {plan_path}",
            ],
        )


def _unique(names: Any) -> list[str]:
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _ov(obj: Any, *names: str) -> Any:
    """Read the first present attribute or key of a caption overlay."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    raise AttributeError(f"caption overlay missing any of {names}: {obj!r}")


def _clip_hash(clip: Clip) -> str:
    payload = json.dumps(clip.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


register(ResolveEngine())
