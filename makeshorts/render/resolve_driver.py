"""Executes a timeline plan inside a running DaVinci Resolve.

Run as a child process, never imported by `ms` itself:

    python -m makeshorts.render.resolve_driver version
    python -m makeshorts.render.resolve_driver build plan.json

and prints exactly one JSON object on its last line of stdout.

Why a separate process: Resolve's `fusionscript` module segfaults during
interpreter teardown once it has been used. Loaded into `ms render`, every
batch would end in a crash however well it went. Here the process leaves via
`os._exit`, which skips the teardown, and the parent reads the result rather
than the exit status.

Why so dumb: every decision -- which frames, which track, which transform --
was made by `resolve_engine.build_plan`, which is pure and tested without
Resolve. This file only translates a plan into API calls, so the part that
cannot be tested in CI is as small as it can be made.

Standard library only. It must import cleanly with nothing but Resolve's own
scripting module available.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["script_api_dir", "main"]

# TimelineItem "Scaling" values -- resolve.SCALE_* in the scripting docs.
SCALE_CROP = 1  # source pixels 1:1, centred: the frame every transform assumes
SCALE_FIT = 2

_DEFAULT_API = {
    "darwin": "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
    "win32": os.path.expandvars(
        r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
    ),
    "linux": "/opt/resolve/Developer/Scripting",
}


class DriverError(RuntimeError):
    pass


def script_api_dir() -> Path | None:
    """Where Resolve's scripting module lives, or None if it is not installed.

    `RESOLVE_SCRIPT_API` wins, as Resolve's own README describes.
    """
    env = os.environ.get("RESOLVE_SCRIPT_API")
    candidates = [env] if env else []
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    if platform in _DEFAULT_API:
        candidates.append(_DEFAULT_API[platform])
    for c in candidates:
        if c and (Path(c) / "Modules" / "DaVinciResolveScript.py").is_file():
            return Path(c)
    return None


def _connect() -> Any:
    api = script_api_dir()
    if api is None:
        raise DriverError(
            "DaVinci Resolve's scripting module was not found. Install Resolve "
            "Studio, or set RESOLVE_SCRIPT_API to its Developer/Scripting directory."
        )
    sys.path.append(str(api / "Modules"))
    import DaVinciResolveScript as dvr  # noqa: PLC0415 — only exists inside Resolve's install

    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise DriverError(
            "could not reach DaVinci Resolve. Start it, and check Preferences > "
            "System > General > External scripting using: Local."
        )
    return resolve


# --------------------------------------------------------------------------
# Project, bin, media
# --------------------------------------------------------------------------


def _open_project(resolve: Any, name: str) -> tuple[Any, Any, str | None]:
    pm = resolve.GetProjectManager()
    current = pm.GetCurrentProject()
    previous = current.GetName() if current else None
    if previous == name:
        return pm, current, previous
    # Switching projects in the UI asks whether to save. There is no one to
    # ask here, and discarding someone's edits is the worse of the two.
    if current is not None:
        pm.SaveProject()
    project = pm.LoadProject(name) or pm.CreateProject(name)
    if project is None:
        raise DriverError(f"could not open or create Resolve project {name!r}")
    return pm, project, previous


def _bin(media_pool: Any, name: str) -> Any:
    root = media_pool.GetRootFolder()
    for folder in root.GetSubFolderList() or []:
        if folder.GetName() == name:
            return folder
    folder = media_pool.AddSubFolder(root, name)
    if folder is None:
        raise DriverError(f"could not create media pool bin {name!r}")
    return folder


def _import_media(media_pool: Any, folder: Any, media: dict[str, Any],
                  fps: float) -> dict[str, Any]:
    """Name -> MediaPoolItem.

    Sources are reused across clips when the bin already holds the same file:
    re-importing a two-hour recording for every clip would fill the bin with
    duplicates. Image sequences are always replaced, because a re-render
    rewrites their frames and Resolve would otherwise keep showing the old
    ones from cache.
    """
    media_pool.SetCurrentFolder(folder)
    existing = {}
    for clip in folder.GetClipList() or []:
        existing.setdefault(clip.GetClipProperty("File Path"), clip)

    items: dict[str, Any] = {}
    for name, spec in media.items():
        if "sequence" in spec:
            seq_dir = str(Path(spec["sequence"]).parent)
            stale = [c for path, c in existing.items() if path and path.startswith(seq_dir)]
            if stale:
                media_pool.DeleteClips(stale)
            got = media_pool.ImportMedia([{
                "FilePath": spec["sequence"],
                "StartIndex": spec["start"],
                "EndIndex": spec["end"],
            }])
            if not got:
                raise DriverError(f"Resolve could not import image sequence {spec['sequence']}")
            item = got[0]
            # A sequence takes the project's frame rate, which need not be
            # this clip's; its frames are one-per-output-frame by construction.
            if float(item.GetClipProperty("FPS") or 0) != fps:
                item.SetClipProperty("FPS", _fps_str(fps))
        else:
            path = spec["path"]
            item = existing.get(path)
            if item is None:
                got = media_pool.ImportMedia([path])
                if not got:
                    raise DriverError(f"Resolve could not import {path}")
                item = got[0]
        items[name] = item
    return items


def _fps_str(fps: float) -> str:
    return str(int(fps)) if float(fps).is_integer() else str(fps)


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


def _delete_timeline(project: Any, media_pool: Any, name: str) -> None:
    for i in range(project.GetTimelineCount(), 0, -1):
        tl = project.GetTimelineByIndex(i)
        if tl is not None and tl.GetName() == name:
            media_pool.DeleteTimelines([tl])


def _new_timeline(project: Any, media_pool: Any, spec: dict[str, Any]) -> Any:
    _delete_timeline(project, media_pool, spec["name"])
    tl = media_pool.CreateEmptyTimeline(spec["name"])
    if tl is None:
        raise DriverError(f"could not create timeline {spec['name']!r}")
    project.SetCurrentTimeline(tl)
    settings = {
        "useCustomSettings": "1",
        "timelineResolutionWidth": str(spec["width"]),
        "timelineResolutionHeight": str(spec["height"]),
        "timelineFrameRate": _fps_str(spec["fps"]),
    }
    for key, value in settings.items():
        if not tl.SetSetting(key, value):
            raise DriverError(f"timeline refused {key}={value}")
    while tl.GetTrackCount("video") < spec["video_tracks"]:
        if not tl.AddTrack("video"):
            raise DriverError("could not add a video track")
    return tl


def _source_window(item: dict[str, Any], media_item: Any, tl_fps: float) -> tuple[int, int]:
    """[in, out) in the media's own frames.

    Every layer of a clip is anchored to the same absolute instant and offset
    by its record position, so layers of one span cut on the same source frame
    and consecutive spans tile with no gap or repeat. When the source runs at
    the timeline's rate this is exact; when it does not, Resolve retimes and
    the rounding is per frame.
    """
    if "src_in" in item:
        return item["src_in"], item["src_in"] + item["frames"]
    src_fps = float(media_item.GetClipProperty("FPS") or tl_fps)
    base = round(item["anchor"] * src_fps)
    ratio = src_fps / tl_fps
    return (base + round(item["record"] * ratio),
            base + round((item["record"] + item["frames"]) * ratio))


def _append(media_pool: Any, tl: Any, media_item: Any, item: dict[str, Any],
            media_type: int | None, tl_fps: float) -> Any | None:
    src_in, src_out = _source_window(item, media_item, tl_fps)
    info = {
        "mediaPoolItem": media_item,
        "startFrame": src_in,
        "endFrame": src_out,  # exclusive
        "trackIndex": item["track"],
        "recordFrame": tl.GetStartFrame() + item["record"],
    }
    if media_type is not None:
        info["mediaType"] = media_type
    got = media_pool.AppendToTimeline([info])
    return got[0] if got else None


def _add_blur(timeline_item: Any, size: float) -> None:
    """Gaussian blur via a Fusion comp on the item.

    Do NOT wrap this in `comp.Lock()`/`Unlock()`. Fusion's usual batching
    idiom silently leaves the comp out of the render: the graph reads back
    correctly wired and the output is unblurred.
    """
    comp = timeline_item.AddFusionComp()
    if comp is None:
        raise DriverError("could not add a Fusion comp for the backdrop blur")
    media_in = comp.FindTool("MediaIn1")
    media_out = comp.FindTool("MediaOut1")
    blur = comp.AddTool("Blur", -32768, -32768)
    if not (media_in and media_out and blur):
        raise DriverError("Fusion comp is missing MediaIn1/MediaOut1")
    blur.ConnectInput("Input", media_in)
    blur.SetInput("XBlurSize", float(size))
    media_out.ConnectInput("Input", blur)


def _add_sharpen(timeline_item: Any, amount: float) -> None:
    """Unsharp mask via a Fusion comp on the item, to counter upscaling softness.

    Same no-Lock() rule as `_add_blur`. Uses the UnsharpMask tool, whose `Gain`
    is the strength knob; the makeshorts `sharpen` amount maps straight onto it.
    """
    comp = timeline_item.AddFusionComp()
    if comp is None:
        raise DriverError("could not add a Fusion comp for sharpening")
    media_in = comp.FindTool("MediaIn1")
    media_out = comp.FindTool("MediaOut1")
    sharp = comp.AddTool("UnsharpMask", -32768, -32768)
    if not (media_in and media_out and sharp):
        raise DriverError("Fusion comp is missing MediaIn1/MediaOut1 or the sharpen tool")
    sharp.ConnectInput("Input", media_in)
    sharp.SetInput("Gain", float(amount))
    media_out.ConnectInput("Input", sharp)


def _place_video(media_pool: Any, tl: Any, media: dict[str, Any],
                 items: list[dict[str, Any]], fps: float) -> int:
    for spec in items:
        placed = _append(media_pool, tl, media[spec["media"]], spec, 1, fps)
        if placed is None:
            raise DriverError(
                f"could not place {spec['media']!r} on V{spec['track']} at frame {spec['record']}"
            )
        if spec.get("props") and not placed.SetProperty(spec["props"]):
            raise DriverError(f"Resolve rejected transform {spec['props']}")
        if spec.get("blur"):
            _add_blur(placed, spec["blur"])
        if spec.get("sharpen"):
            _add_sharpen(placed, spec["sharpen"])
        if spec.get("fade_in") or spec.get("fade_out"):
            placed.SetFades({"FadeIn": spec.get("fade_in", 0), "FadeOut": spec.get("fade_out", 0)})
    return len(items)


def _place_audio(media_pool: Any, tl: Any, media: dict[str, Any],
                 spec: dict[str, Any] | None, fps: float) -> bool:
    """One continuous audio item for the whole clip, from one file."""
    if spec is None:
        return False
    placed = _append(media_pool, tl, media[spec["media"]], spec, 2, fps)
    if placed is None:
        # A source with no audio stream is legitimate; keep the video.
        return False
    norm = spec.get("normalize")
    if norm:
        tl.NormalizeAudioLevel([placed], {
            "normalizationMode": norm["mode"],
            "targetLoudness": float(norm["target"]),
        })
    return True


def _place_outro(media_pool: Any, tl: Any, media: dict[str, Any],
                 spec: dict[str, Any] | None, fps: float) -> None:
    if spec is None:
        return
    item = media[spec["media"]]
    src_fps = float(item.GetClipProperty("FPS") or fps)
    total = int(item.GetClipProperty("Frames") or 0)
    if spec.get("max_seconds"):
        total = min(total, round(spec["max_seconds"] * src_fps)) if total else round(
            spec["max_seconds"] * src_fps)
    if total <= 0:
        return
    window = {"src_in": 0, "frames": total, "track": 1, "record": spec["record"]}
    video = _append(media_pool, tl, item, window, 1, fps)
    if video is None:
        raise DriverError("could not place the outro")
    # Fit, not crop: a bumper is designed whole and must not lose its edges.
    video.SetProperty({"Scaling": SCALE_FIT})
    if spec.get("audio"):
        _append(media_pool, tl, item, window, 2, fps)


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def _render(project: Any, spec: dict[str, Any], timeout: float) -> Path:
    if not project.SetCurrentRenderFormatAndCodec(spec["format"], spec["codec"]):
        codecs = project.GetRenderCodecs(spec["format"]) or {}
        raise DriverError(
            f"Resolve has no {spec['format']!r}/{spec['codec']!r} render option; "
            f"codecs for that format: {sorted(codecs.values())}"
        )
    project.SetCurrentRenderMode(1)  # single clip

    target = Path(spec["dir"]) / f"{spec['name']}.{spec['ext']}"
    if target.exists():
        # Otherwise Resolve may write `name_1.mp4` beside it and leave the old
        # file where `ms render` expects the new one.
        target.unlink()

    settings = {
        "SelectAllFrames": True,
        "TargetDir": spec["dir"],
        "CustomName": spec["name"],
        "ExportVideo": True,
        "ExportAudio": True,
        "FormatWidth": spec["width"],
        "FormatHeight": spec["height"],
        "FrameRate": float(spec["fps"]),
        "VideoQuality": spec["quality"],
        "AudioCodec": spec["audio_codec"],
        "AudioSampleRate": spec["sample_rate"],
    }
    if not project.SetRenderSettings(settings):
        raise DriverError(f"Resolve rejected render settings {settings}")

    job = project.AddRenderJob()
    if not job:
        raise DriverError("Resolve would not queue a render job")
    try:
        if not project.StartRendering([job], False):
            raise DriverError("Resolve would not start rendering")
        deadline = time.monotonic() + timeout
        while project.IsRenderingInProgress():
            if time.monotonic() > deadline:
                project.StopRendering()
                raise DriverError(f"render did not finish within {timeout:.0f}s")
            time.sleep(0.25)
        status = project.GetRenderJobStatus(job) or {}
    finally:
        # Only our job, never the user's queue.
        project.DeleteRenderJob(job)

    if status.get("JobStatus") != "Complete":
        raise DriverError(
            f"render {status.get('JobStatus', 'unknown')}: {status.get('Error', 'no detail')}"
        )
    if not target.exists():
        raise DriverError(f"Resolve reported success but {target} was not written")
    return target


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _version() -> dict[str, Any]:
    resolve = _connect()
    return {"ok": True, "version": f"{resolve.GetProductName()} {resolve.GetVersionString()}"}


def _build(plan: dict[str, Any]) -> dict[str, Any]:
    resolve = _connect()
    version = f"{resolve.GetProductName()} {resolve.GetVersionString()}"
    pm, project, previous = _open_project(resolve, plan["project"])
    try:
        media_pool = project.GetMediaPool()
        folder = _bin(media_pool, plan["bin"])
        fps = float(plan["timeline"]["fps"])
        media = _import_media(media_pool, folder, plan["media"], fps)

        tl = _new_timeline(project, media_pool, plan["timeline"])
        placed = _place_video(media_pool, tl, media, plan["video"], fps)
        has_audio = _place_audio(media_pool, tl, media, plan.get("audio"), fps)
        _place_outro(media_pool, tl, media, plan.get("outro"), fps)
        frames = tl.GetEndFrame() - tl.GetStartFrame()

        output = _render(project, plan["render"], float(plan.get("timeout", 1800)))
        if not plan.get("keep_timeline", True):
            media_pool.DeleteTimelines([tl])
        pm.SaveProject()
    finally:
        if plan.get("restore_project") and previous and previous != plan["project"]:
            pm.SaveProject()
            pm.LoadProject(previous)

    return {
        "ok": True,
        "version": version,
        "project": plan["project"],
        "timeline": plan["timeline"]["name"],
        "items": placed,
        "audio": has_audio,
        "frames": frames,
        "output": str(output),
    }


def main(argv: list[str]) -> dict[str, Any]:
    if not argv:
        raise DriverError("usage: resolve_driver (version | build PLAN.json)")
    if argv[0] == "version":
        return _version()
    if argv[0] == "build" and len(argv) == 2:
        return _build(json.loads(Path(argv[1]).read_text()))
    raise DriverError(f"unknown command {argv!r}")


if __name__ == "__main__":
    try:
        result = main(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001 — everything goes back to the parent as JSON
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result), flush=True)
    sys.stderr.flush()
    # Skip interpreter teardown: fusionscript crashes in it.
    os._exit(0)
