"""The ffmpeg render engine — the only place in this project that knows what
ffmpeg is.

Everything upstream speaks in semantic layouts and normalized rects.
`layout.plan_clip()` turns those into pixel `Placement`s. This module is the
last mile: placements become filtergraphs, and nothing else needs to care.

## Pipeline

A clip may change layout partway through, so it is built in stages rather than
as one heroic filtergraph:

    1. one video-only segment per layout span
    2. concat the segments            -> continuous silent video
    3. burn captions in ONE pass over the concatenated video
    4. mux against ONE continuous audio stream for the whole clip

Steps 3 and 4 are deliberately not per-span. Per-span audio would put a seam at
every layout change, and per-span captions would cut a caption cue in half when
a layout change lands mid-sentence. Doing both once over the whole clip makes
those failure modes structurally impossible rather than merely unlikely.

## Several input files

A job may have more than one source file -- a Zoom export gives frame-aligned
renders of the same meeting, with the sharp slides in one file and the only
usable face in another, and a single clip may composite both. Those files share
a clock: an absolute timestamp means the same instant in every one of them, so
every input gets the *same* `-ss`/`-t` seek and no re-timing is needed
anywhere.

Two rules keep that from spreading:

- a span opens exactly the inputs its own placements read, so a single-source
  clip still produces a single-input command;
- audio always comes from one file, the primary source, in one pass over the
  whole clip. Per-source audio would reintroduce the seam that step 4 exists to
  avoid.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from makeshorts.render import layout as L
from makeshorts.render.caps import probe_caps
from makeshorts.render.engine import (
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
from makeshorts.select.schema import Clip, ClipsDoc

# Beyond this many PNG overlays in one filtergraph, ffmpeg's graph gets
# unwieldy and command lines get long, so caption burning is split into
# successive passes. Karaoke produces one PNG per highlighted word, so a
# 45-second clip routinely exceeds this.
MAX_OVERLAYS_PER_PASS = 40


class FFmpegError(RuntimeError):
    def __init__(self, args: list[str], stderr: str) -> None:
        self.args_run = args
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(f"ffmpeg failed: {' '.join(args[:6])} ...\n{tail}")


@dataclass(frozen=True)
class EncodeSettings:
    """Everything ffmpeg-shaped. Sourced from config/render.yaml, never from
    the edit list."""

    video_codec: str = "libx264"
    crf: int = 19
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "160k"
    loudnorm_i: float = -14.0  # LUFS; the usual target for social platforms
    fps: int = 30

    @property
    def video_args(self) -> list[str]:
        if self.video_codec == "h264_videotoolbox":
            # videotoolbox ignores crf and wants a bitrate instead
            return ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
        return [
            "-c:v", self.video_codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
        ]


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(args, proc.stderr)
    return proc.stderr


def _esc(path: Path) -> str:
    """Escape a path for use inside a filtergraph argument."""
    return str(path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


class FFmpegEngine:
    name = "ffmpeg"

    def __init__(self, settings: EncodeSettings | None = None, ffmpeg: str | None = None) -> None:
        self.settings = settings or EncodeSettings()
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = shutil.which("ffprobe") or "ffprobe"

    # -- capabilities -----------------------------------------------------

    def capabilities(self) -> set[str]:
        """What this binary can actually do.

        Caption capability is probed rather than assumed: a build without
        libass/libfreetype has no `subtitles` or `drawtext` filter, and the
        Pillow backend covers for it by compositing PNGs with `overlay`, which
        every build has. Layout capabilities need only crop/scale/overlay.
        """
        caps = {CAP_FOCUS, CAP_HERO_INSET, CAP_STACK, CAP_CONTAIN_BLUR, CAP_MULTI_SPAN}
        probed = probe_caps()
        backend = getattr(probed, "caption_backend", None)
        if backend in ("pillow", "ass"):
            caps |= {CAP_BURNED_CAPTIONS, CAP_KARAOKE_CAPTIONS}
        return caps

    def version(self) -> str:
        out = subprocess.run([self.ffmpeg, "-version"], capture_output=True, text=True).stdout
        return out.splitlines()[0] if out else "unknown"

    # -- filtergraph construction ----------------------------------------

    def _span_filtergraph(
        self,
        plan: L.LayoutPlan,
        duration: float,
        names: list[str],
        input_of: Mapping[str, int],
    ) -> str:
        """One layout span -> a filtergraph string.

        Placements arrive already ordered back-to-front, so this is a straight
        fold: start from a black canvas and overlay each layer in turn. All the
        geometry decisions were made in layout.py; nothing here re-derives
        them.

        `names[i]` is the source placement `i` reads from and `input_of` maps
        that name to an ffmpeg input index -- so a placement's crop is applied
        to the file it was measured against, not to whichever file happened to
        be listed first.

        `duration` must be passed explicitly and bound onto the `color` source:
        a filter-generated source is infinite, and `-t` before `-i` bounds only
        the file input, so an unbounded canvas plus `overlay=shortest=0` yields
        an encode that never terminates.
        """
        w, h = plan.output_width, plan.output_height
        parts = [
            f"color=c=black:s={w}x{h}:r={self.settings.fps}:d={duration:.3f}[canvas]"
        ]

        # One split per input, sized to how many placements read that input:
        # ffmpeg requires an explicit split to consume a stream more than once,
        # and the count is per stream rather than per span.
        by_input: dict[int, list[int]] = {}
        for i, name in enumerate(names):
            by_input.setdefault(input_of[name], []).append(i)
        for idx in sorted(by_input):
            users = by_input[idx]
            outs = "".join(f"[src{i}]" for i in users)
            parts.append(f"[{idx}:v]split={len(users)}{outs}")

        current = "canvas"
        for i, p in enumerate(plan.placements):
            s, d = p.source_rect, p.dest_rect
            chain = f"[src{i}]crop={s.w}:{s.h}:{s.x}:{s.y},scale={d.w}:{d.h}"
            if p.blur_backdrop and p.blur_radius_px > 0:
                # boxblur's radius must stay under half the smaller dimension
                r = max(1, min(p.blur_radius_px, min(d.w, d.h) // 2 - 1))
                chain += f",boxblur={r}:1"
            chain += f"[l{i}]"
            parts.append(chain)

            nxt = f"c{i}"
            parts.append(f"[{current}][l{i}]overlay=x={d.x}:y={d.y}:shortest=0[{nxt}]")
            current = nxt

        parts.append(f"[{current}]format={self.settings.pix_fmt}[vout]")
        return ";".join(parts)

    def _render_span(self, sources: Mapping[str, Path],
                     abs_start: float, duration: float,
                     plan: L.LayoutPlan, out: Path) -> list[str]:
        """Encode one layout span, opening one input per source it reads.

        `plan.source_names` is layout's own back-to-front first-use order, so
        identical plans produce byte-identical commands -- which the receipt
        records and a re-render is expected to reproduce. A span whose
        placements all read one file gets one `-i`, exactly as before this
        module learned about several.
        """
        names = [p.source for p in plan.placements]
        order = list(plan.source_names)
        missing = [n for n in order if n not in sources]
        if missing:
            raise FileNotFoundError(
                f"layout reads source(s) {missing} but only {sorted(sources)} "
                f"were resolved from the edit list"
            )
        input_of = {name: i for i, name in enumerate(order)}
        graph = self._span_filtergraph(plan, duration, names, input_of)

        args = [self.ffmpeg, "-hide_banner", "-nostdin", "-y"]
        for name in order:
            # Every input is seeked to the same ABSOLUTE time: the files are
            # frame-aligned renders of one recording and share a clock.
            # -ss before -i seeks fast; -accurate_seek keeps it frame-exact,
            # which matters because clip boundaries were snapped to words.
            args += [
                "-accurate_seek", "-ss", f"{abs_start:.3f}",
                "-t", f"{duration:.3f}",
                "-i", str(sources[name]),
            ]
        args += [
            "-filter_complex", graph,
            "-map", "[vout]", "-an",
            *self.settings.video_args,
            "-r", str(self.settings.fps),
            # Belt and braces: bound the OUTPUT too. The canvas is already
            # bounded above, but a stray unbounded filter source must never be
            # able to produce an encode that runs forever.
            "-t", f"{duration:.3f}",
            str(out),
        ]
        _run(args)
        return args

    def _concat(self, segments: list[Path], out: Path, workdir: Path) -> list[str]:
        """Stream-copy concat. Safe because every segment was encoded by
        _render_span with identical settings."""
        listing = workdir / "segments.txt"
        listing.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
        args = [
            self.ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(out),
        ]
        _run(args)
        return args

    def _burn_captions(self, video: Path, overlays: list, out: Path,
                       workdir: Path) -> list[str]:
        """Composite caption PNGs over the whole clip.

        `overlays` is the CaptionBackend contract: objects carrying
        `png_path`, `x`, `y`, `start`, `end` (seconds relative to clip start).
        Timeline-gated `overlay` is used because this ffmpeg has no `subtitles`
        or `drawtext` filter -- see caps.py.
        """
        if not overlays:
            shutil.copy(video, out)
            return []

        cmds: list[str] = []
        src = video
        chunks = [
            overlays[i : i + MAX_OVERLAYS_PER_PASS]
            for i in range(0, len(overlays), MAX_OVERLAYS_PER_PASS)
        ]
        for pass_i, chunk in enumerate(chunks):
            dst = out if pass_i == len(chunks) - 1 else workdir / f"cap{pass_i}.mp4"
            args = [self.ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(src)]
            for o in chunk:
                args += ["-i", str(_png_path(o))]

            parts, current = [], "0:v"
            for i, o in enumerate(chunk, start=1):
                x, y = int(_attr(o, "x", "dest_x")), int(_attr(o, "y", "dest_y"))
                s, e = float(_attr(o, "start", "start_time")), float(_attr(o, "end", "end_time"))
                nxt = f"o{i}"
                parts.append(
                    f"[{current}][{i}:v]overlay=x={x}:y={y}:"
                    f"enable='between(t,{s:.3f},{e:.3f})'[{nxt}]"
                )
                current = nxt
            parts.append(f"[{current}]format={self.settings.pix_fmt}[vout]")

            args += [
                "-filter_complex", ";".join(parts),
                "-map", "[vout]", "-an",
                *self.settings.video_args, "-r", str(self.settings.fps),
                str(dst),
            ]
            _run(args)
            cmds.append(" ".join(args))
            src = dst
        return cmds

    def _mux_audio(self, video: Path, source: Path, clip: Clip, out: Path,
                   normalize: bool) -> list[str]:
        """Attach one continuous audio stream for the whole clip.

        Extracted in a single pass over [clip.start, clip.end] so layout
        changes leave no audible seam.
        """
        af = f"loudnorm=I={self.settings.loudnorm_i}:TP=-1.5:LRA=11" if normalize else "anull"
        args = [
            self.ffmpeg, "-hide_banner", "-nostdin", "-y",
            "-i", str(video),
            "-accurate_seek", "-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}",
            "-i", str(source),
            "-filter_complex", f"[1:a]{af},asetpts=PTS-STARTPTS[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", self.settings.audio_codec, "-b:a", self.settings.audio_bitrate,
            "-shortest", str(out),
        ]
        try:
            _run(args)
        except FFmpegError:
            # A source with no audio stream is legitimate; keep the video.
            shutil.copy(video, out)
            return []
        return args

    # -- inputs ------------------------------------------------------------

    def _resolve_source_paths(self, doc: ClipsDoc, source: Path) -> dict[str, Path]:
        """Source name -> a file that exists on disk.

        One source is the overwhelmingly common case and is answered without
        touching the filesystem: the caller's path wins outright, which keeps
        every existing single-source caller working unchanged.

        With several, `SourceSpec.path` is what we have, and it is written the
        way a human writes it in `clips.json` -- usually relative to the repo
        root (`jobs/<slug>/screen.mp4`), sometimes just a filename sitting next
        to the primary source. So each is tried against the primary source's
        directory, then each directory above it, then the cwd.
        """
        specs = doc.resolved_sources()
        primary = doc.primary_source_name
        if len(specs) == 1:
            return {primary: source}

        roots = _search_roots(source)
        paths: dict[str, Path] = {}
        missing: list[str] = []
        for name, spec in specs.items():
            found = _first_existing(Path(spec.path), roots)
            if found is None and name == primary and source.exists():
                # The edit list may name the primary file something the job
                # directory does not; the caller handed us the real one.
                found = source
            if found is None:
                missing.append(f"{name!r} -> {spec.path!r}")
            else:
                paths[name] = found.resolve()
        if missing:
            where = "\n  ".join(str(r) for r in roots)
            raise FileNotFoundError(
                "cannot find source file(s) named by the edit list: "
                + ", ".join(missing)
                + f"\nlooked (relative paths only) in:\n  {where}"
            )
        return paths

    # -- entry point -------------------------------------------------------

    def render(self, doc: ClipsDoc, clip: Clip, source: Path, out: Path,
               caption_overlays: list | None = None) -> Receipt:
        """Render one clip.

        `source` stays in the signature: for a single-source job it *is* the
        answer, and it is the file `ms render` actually ingested, which need not
        be spelled the way the edit list spells it. When the edit list names
        several sources the real paths come from `doc.resolved_sources()`
        instead.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        regions = doc.resolved_regions()
        primary = doc.primary_source_name
        paths = self._resolve_source_paths(doc, Path(source))
        out_size = (doc.output.width, doc.output.height)

        # layout reads each source's resolution off the document, so every
        # region is measured against the file it actually lives in.
        spans = L.plan_clip(clip, regions, doc, out_size)
        commands: list[str] = []

        with TemporaryDirectory(prefix=f"ms-{clip.id}-") as td:
            work = Path(td)
            segments = []
            for i, sp in enumerate(spans):
                seg = work / f"span{i:02d}.mp4"
                args = self._render_span(
                    paths, clip.start + sp.start, sp.end - sp.start, sp.plan, seg
                )
                commands.append(" ".join(args))
                segments.append(seg)

            silent = work / "video.mp4"
            if len(segments) == 1:
                shutil.move(str(segments[0]), silent)
            else:
                commands.append(" ".join(self._concat(segments, silent, work)))

            captioned = work / "captioned.mp4"
            if clip.captions.enabled and caption_overlays:
                commands += self._burn_captions(silent, caption_overlays, captioned, work)
            else:
                shutil.copy(silent, captioned)

            # ONE audio stream for the whole clip, from ONE file. Even when the
            # video was composited from several renders of the meeting, only
            # the primary source's audio is used.
            args = self._mux_audio(
                captioned, paths[primary], clip, out, clip.audio.normalize
            )
            if args:
                commands.append(" ".join(args))

        return Receipt(
            clip_id=clip.id,
            output=str(out),
            engine=self.name,
            engine_version=self.version(),
            source_sha256="",
            clip_sha256=_clip_hash(clip),
            duration=_probe_duration(self.ffprobe, out),
            commands=commands,
        )


def _unique(names: Iterable[str]) -> list[str]:
    """Deduplicate, preserving first-use order."""
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _search_roots(source: Path) -> list[Path]:
    """Directories a relative source path is tried against, nearest first."""
    base = source.parent if source.parent != Path("") else Path.cwd()
    roots = [base, *base.parents, Path.cwd()]
    return [Path(p) for p in _unique(str(p) for p in roots)]


def _first_existing(path: Path, roots: list[Path]) -> Path | None:
    if path.is_absolute():
        return path if path.exists() else None
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return None


def _attr(obj, *names):
    """Read the first present attribute or key. Keeps the engine tolerant of
    the caption backend's exact field naming."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    raise AttributeError(f"caption overlay missing any of {names}: {obj!r}")


def _png_path(o):
    return _attr(o, "png_path", "path", "png")


def _clip_hash(clip: Clip) -> str:
    import hashlib

    payload = json.dumps(clip.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _probe_duration(ffprobe: str, path: Path) -> float:
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


register(FFmpegEngine())
