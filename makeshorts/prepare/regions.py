"""Propose named regions of the source frame from per-tile temporal statistics.

Webinar recordings are composites. Two speakers side by side, one speaker
filling the frame, a speaker plus a shared deck, a gallery grid — the geometry
is different every time and nothing in the container records it. This module
recovers it mechanically, and only ever *proposes*: `regions.json` is read by
the editorial step, which confirms or corrects it into `clips.json`.

The signal
----------
A camera feed changes a little in almost every frame — a person is never
perfectly still, and even a still person sits in front of a noisy sensor. A
slide or screenshare is byte-identical from frame to frame and then changes
completely, once, when the deck advances.

So the discriminator is not *how much* a tile changes but *how often*: the
fraction of sampled intervals in which a tile changed at all. We call that the
tile's **activity**. Cameras land near 0.2-0.9, slides near 1/n. Two orders of
magnitude apart, and — importantly — the threshold between them is not a
constant we have to guess. It is read off the gap in the distribution of tile
activities for this particular video, so a noisy camera and a pristine
screengrab separate just as well as a clean camera and a dithered one.

Everything downstream is bookkeeping: connect similar neighbouring tiles,
bound them with a rectangle, name it, and say how much we believe it.

What this module deliberately does not do
-----------------------------------------
Invent structure. If the tile activities do not separate, every tile gets the
same label, one region comes out covering the whole frame, and the caller falls
back to the implicit `frame` region. Zero or one region is a correct answer.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from makeshorts.artifacts import Region, RegionKind, RegionsDoc, Rect

__all__ = [
    "RegionParams",
    "RegionDetectionError",
    "TileGrid",
    "FrameSample",
    "RegionAnalysis",
    "detect_regions",
    "analyze",
    "explain",
]


class RegionDetectionError(RuntimeError):
    """ffmpeg could not be run, or produced nothing usable."""


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionParams:
    """Every tunable in one place, so a bad proposal can be re-run by hand with
    one value changed and the difference attributed."""

    # -- sampling ----------------------------------------------------------
    grid_x: int = 8
    grid_y: int = 8
    #: Frames to aim for on the uniform-sampling path.
    target_frames: int = 240
    #: Hard cap on the keyframe path, which has no rate control of its own.
    max_frames: int = 900
    #: Below this many usable frames the statistics are not worth trusting.
    min_frames: int = 24
    #: Width the source is downscaled to before any statistics. Height follows
    #: the source aspect. Small enough to be free, large enough that an 8x8 tile
    #: still covers hundreds of pixels.
    sample_width: int = 256
    #: Videos at least this long are sampled by decoding keyframes only. It is
    #: dramatically faster on a 60-minute recording and — because keyframes are
    #: ~2s apart while uniform sampling of a 60-minute file is ~15s apart —
    #: gives *denser* coverage, not sparser.
    keyframe_min_duration: float = 120.0

    # -- letterbox ---------------------------------------------------------
    #: A pixel never brighter than this (0..1) over the whole video is a bar.
    bar_level: float = 0.098  # 25/255, matching cropdetect's default limit
    #: Refuse to call more than this fraction of an axis "bar". Beyond it we are
    #: almost certainly eating a dark slide deck rather than a mask.
    max_crop_frac: float = 0.35

    # -- tile classification ----------------------------------------------
    #: A tile counts as having changed in an interval when its mean absolute
    #: luma delta exceeds this fraction of its own 99.5th-percentile delta.
    #: Relative to the tile's own peak, so the absolute noise floor of the
    #: source does not matter.
    change_rel: float = 0.15
    #: ...but never below this, so a frozen tile's rounding noise is not motion.
    change_abs: float = 0.0008
    #: The activity gap must be at least this wide (multiplicatively) before we
    #: believe the frame decomposes into moving and static parts.
    separation_min: float = 3.0
    #: With no gap to split on, these decide what the whole frame is.
    unimodal_moving_activity: float = 0.05
    unimodal_static_activity: float = 0.02
    #: A tile with less spatial detail than this and no changes at all carries
    #: no signal — a black bar, a flat backdrop. Excluded rather than labelled.
    content_min: float = 0.02
    #: Largest delta a tile must reach to be considered ever to have changed.
    peak_min: float = 0.01

    # -- clustering --------------------------------------------------------
    #: Neighbouring camera tiles join when their activities are within this
    #: factor. Cameras are not internally uniform (a face moves more than the
    #: wall behind it), so this is deliberately loose.
    speaker_activity_ratio: float = 4.0
    #: Neighbouring slide tiles join when their change *schedules* correlate.
    #: Two screenshares that advance at different times therefore separate.
    slide_corr_min: float = 0.5
    #: Smallest acceptable region, as a fraction of the frame and in tiles.
    min_area_frac: float = 0.02
    min_tiles: int = 2

    # -- change points -----------------------------------------------------
    #: An interval is a step when it dwarfs the region's own baseline...
    step_over_baseline: float = 8.0
    #: ...and is comparable to the largest change the region ever showed. A
    #: slide advance repaints most of the region; a mouse cursor crossing it
    #: does not, and this is what keeps the cursor out of `change_points`.
    step_rel_peak: float = 0.25
    step_min_delta: float = 0.008
    max_change_points: int = 200

    # -- snapping ----------------------------------------------------------
    #: Edges land on tile boundaries, so 1/4, 1/2 and 3/4 are already exact on
    #: an 8-column grid and snapping to them is a no-op; they are listed so
    #: that landing on one still counts as a clean boundary. 1/3 and 2/3 are
    #: the ones actually worth recovering.
    snap_to: tuple[float, ...] = (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0)
    #: As a fraction of one tile. Under half a tile, so snapping can only ever
    #: move an edge within the cell it was already known to lie in.
    snap_tol_tiles: float = 0.4

    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


# --------------------------------------------------------------------------
# Intermediate results (kept whole so `explain` can answer "why this rect?")
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameSample:
    """Greyscale frames plus the source timestamp of each one."""

    frames: np.ndarray  # (n, h, w) uint8
    times: np.ndarray  # (n,) float seconds
    method: str  # "keyframes" | "uniform"

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True)
class TileGrid:
    """Per-tile temporal signatures over the letterbox-stripped content area.

    All rectangles here are in *sample* pixel coordinates of the full frame,
    which normalize directly to source-frame coordinates.
    """

    cols: int
    rows: int
    #: Boundary indices into the full sample frame, length cols+1 / rows+1.
    xs: np.ndarray
    ys: np.ndarray
    frame_w: int
    frame_h: int
    #: (m, rows, cols) mean absolute luma delta per interval.
    delta: np.ndarray
    #: (m+1,) timestamps; interval i spans times[i]..times[i+1].
    times: np.ndarray
    activity: np.ndarray  # (rows, cols) fraction of intervals with a change
    energy: np.ndarray  # (rows, cols) mean absolute delta
    peak: np.ndarray  # (rows, cols) 99.5th-percentile delta
    content: np.ndarray  # (rows, cols) spatial std of the time-mean frame
    corr_h: np.ndarray  # (rows, cols-1) correlation with right neighbour
    corr_v: np.ndarray  # (rows-1, cols) correlation with lower neighbour


@dataclass(frozen=True)
class RegionAnalysis:
    """Everything `detect_regions` decided, and what it decided it from."""

    grid: TileGrid
    sample: FrameSample
    #: Per tile: "moving", "static", "unknown" or "dead".
    tile_class: np.ndarray
    #: The activity value the moving/static split was made at, and how clean it
    #: was. `separation` of 1.0 means no split was made.
    split_at: float
    separation: float
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in sample pixels
    components: list[Component] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)


@dataclass
class Component:
    """One connected run of similar tiles, before it becomes a Region."""

    tiles: list[tuple[int, int]]
    kind: RegionKind
    bbox: tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive
    fill: float
    activity: float
    rect: Rect
    snapped: bool
    change_points: list[float]
    confidence: float


# --------------------------------------------------------------------------
# ffmpeg / ffprobe
# --------------------------------------------------------------------------

_PTS_TIME = re.compile(rb"pts_time:(-?[0-9.]+)")
_SIZE = re.compile(rb" s:(\d+)x(\d+)")


def _run(cmd: list[str], *, want_stdout: bool) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:  # ffmpeg not installed
        raise RegionDetectionError(f"{cmd[0]} not found on PATH") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise RegionDetectionError(f"{cmd[0]} failed:\n" + "\n".join(tail))
    if want_stdout and not proc.stdout:
        raise RegionDetectionError(f"{cmd[0]} produced no frames")
    return proc


def probe_video(path: str | Path, params: RegionParams = RegionParams()) -> tuple[float, float] | None:
    """`(duration, fps)` of the first video stream, or None if there isn't one.

    Deliberately local rather than reusing `prepare/probe.py`: region detection
    needs exactly two numbers and should stay runnable on a bare file path.
    """
    cmd = [
        params.ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = _run(cmd, want_stdout=True)
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        return None
    stream = streams[0]

    duration = 0.0
    for candidate in (stream.get("duration"), (data.get("format") or {}).get("duration")):
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break

    fps = 0.0
    rate = stream.get("avg_frame_rate") or "0/0"
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    return duration, fps


def _decode(cmd: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Run one ffmpeg rawvideo pipe and return `(frames, times)`.

    `showinfo` is in the filter chain, so the timestamps are the ones ffmpeg
    actually emitted rather than ones we assumed. `-fps_mode passthrough` keeps
    the muxer from padding the output back up to a constant frame rate, which
    would otherwise hand us hundreds of duplicate frames with zero deltas —
    i.e. a video that looks entirely static.
    """
    proc = _run(cmd, want_stdout=True)
    size = _SIZE.search(proc.stderr)
    if size is None:
        raise RegionDetectionError("ffmpeg showinfo reported no frame size")
    w, h = int(size.group(1)), int(size.group(2))
    stride = w * h
    n = len(proc.stdout) // stride
    if n == 0:
        raise RegionDetectionError("ffmpeg returned a partial frame")
    frames = np.frombuffer(proc.stdout[: n * stride], dtype=np.uint8).reshape(n, h, w)
    times = np.array([float(m.group(1)) for m in _PTS_TIME.finditer(proc.stderr)], dtype=np.float64)
    if times.size < n:
        # showinfo lost lines to ffmpeg's log rate limiter. Extrapolate at the
        # spacing we did observe; only `change_points` care, and they will be
        # approximately right rather than absent.
        spacing = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        times = np.arange(n, dtype=np.float64) * max(spacing, 1e-6)
    return frames, times[:n]


def sample_frames(
    path: str | Path,
    *,
    duration: float,
    fps: float,
    params: RegionParams = RegionParams(),
) -> FrameSample:
    """Downscaled greyscale frames spread across the video, in one decode.

    Two strategies, both single-pass — 240 seeks would cost more than the decode
    they were meant to avoid:

    - **keyframes**: `-skip_frame nokey` skips the inter-frame reconstruction
      entirely. Used on anything long, where it is both faster and better
      sampled than a uniform rate.
    - **uniform**: an `fps` filter. Used on short input, where there may be
      only a handful of keyframes.
    """
    vf_tail = f"scale={params.sample_width}:-2:flags=area,format=gray,showinfo"
    base = [params.ffmpeg, "-nostdin", "-loglevel", "info"]
    tail = ["-fps_mode", "passthrough", "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "gray", "-"]

    if duration >= params.keyframe_min_duration:
        cmd = [*base, "-skip_frame", "nokey", "-i", str(path), "-vf", vf_tail, *tail]
        try:
            frames, times = _decode(cmd)
        except RegionDetectionError:
            frames = np.empty((0, 0, 0), dtype=np.uint8)
            times = np.empty(0)
        if frames.shape[0] >= params.min_frames:
            if frames.shape[0] > params.max_frames:
                keep = np.linspace(0, frames.shape[0] - 1, params.max_frames).round().astype(int)
                frames, times = frames[keep], times[keep]
            return FrameSample(frames=frames, times=times, method="keyframes")

    # Never ask for frames faster than the source has them: the fps filter
    # duplicates to fill, and duplicates read as a perfectly static video.
    rate = params.target_frames / duration if duration > 0 else 1.0
    if fps > 0:
        rate = min(rate, fps)
    rate = max(rate, 1e-3)
    cmd = [*base, "-i", str(path), "-vf", f"fps={rate:.6f},{vf_tail}", *tail]
    frames, times = _decode(cmd)
    return FrameSample(frames=frames, times=times, method="uniform")


# --------------------------------------------------------------------------
# Letterbox
# --------------------------------------------------------------------------


def detect_content_box(frames: np.ndarray, params: RegionParams) -> tuple[int, int, int, int]:
    """`(x0, y0, x1, y1)` of the real content, stripping letterbox/pillarbox.

    This is what `cropdetect` computes, done on the frames we already have
    rather than in a second full decode of the file. A row is a bar only if no
    pixel in it is ever brighter than `bar_level` in *any* sampled frame, which
    is a stricter test than cropdetect's per-frame one and will not eat a dark
    slide that happens to open on black.
    """
    h, w = frames.shape[1:]
    brightest = frames.max(axis=0).astype(np.float32) / 255.0
    row_bar = brightest.max(axis=1) <= params.bar_level
    col_bar = brightest.max(axis=0) <= params.bar_level

    def _edges(bar: np.ndarray, size: int) -> tuple[int, int]:
        lo = 0
        while lo < size and bar[lo]:
            lo += 1
        hi = size
        while hi > lo and bar[hi - 1]:
            hi -= 1
        if lo >= hi:  # the entire axis is dark; not a crop, just a dark video
            return 0, size
        if (size - (hi - lo)) / size > params.max_crop_frac:
            return 0, size
        return lo, hi

    y0, y1 = _edges(row_bar, h)
    x0, x1 = _edges(col_bar, w)
    if x1 - x0 < params.grid_x or y1 - y0 < params.grid_y:
        return 0, 0, w, h
    return x0, y0, x1, y1


# --------------------------------------------------------------------------
# Tile statistics
# --------------------------------------------------------------------------


def _boundaries(lo: int, hi: int, n: int) -> np.ndarray:
    """`n+1` strictly increasing indices splitting `[lo, hi)` into n blocks."""
    edges = np.linspace(lo, hi, n + 1).round().astype(int)
    for i in range(1, len(edges)):  # linspace can round two edges together
        edges[i] = max(edges[i], edges[i - 1] + 1)
    edges[-1] = hi
    return edges


def _tile_mean(block: np.ndarray, ys: np.ndarray, xs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Block-average `(..., H, W)` down to `(..., rows, cols)`."""
    summed = np.add.reduceat(block, ys[:-1], axis=-2)
    summed = np.add.reduceat(summed, xs[:-1], axis=-1)
    return summed / counts


def tile_grid(sample: FrameSample, params: RegionParams = RegionParams()) -> TileGrid:
    """Per-tile temporal signatures. This is the whole measurement."""
    frames = sample.frames
    n, h, w = frames.shape
    x0, y0, x1, y1 = detect_content_box(frames, params)

    cols = min(params.grid_x, x1 - x0)
    rows = min(params.grid_y, y1 - y0)
    xs = _boundaries(0, x1 - x0, cols)
    ys = _boundaries(0, y1 - y0, rows)
    counts = (np.diff(ys)[:, None] * np.diff(xs)[None, :]).astype(np.float32)

    content_frames = frames[:, y0:y1, x0:x1]

    # Deltas in time-chunks: the float32 expansion of a 900-frame sample is the
    # only thing here big enough to be worth caring about.
    delta = np.empty((n - 1, rows, cols), dtype=np.float32)
    mean_sum = np.zeros((rows, cols), dtype=np.float64)
    sq_sum = np.zeros((rows, cols), dtype=np.float64)
    mean_img = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
    step = 128
    for start in range(0, n - 1, step):
        stop = min(start + step, n - 1)
        chunk = content_frames[start : stop + 1].astype(np.float32) / 255.0
        delta[start:stop] = _tile_mean(np.abs(np.diff(chunk, axis=0)), ys, xs, counts)
    for start in range(0, n, step):
        stop = min(start + step, n)
        mean_img += (content_frames[start:stop].astype(np.float64) / 255.0).sum(axis=0)
    mean_img /= n
    mean_sum = _tile_mean(mean_img, ys, xs, counts)
    sq_sum = _tile_mean(mean_img**2, ys, xs, counts)
    content = np.sqrt(np.maximum(sq_sum - mean_sum**2, 0.0)).astype(np.float32)

    peak = np.percentile(delta, 99.5, axis=0).astype(np.float32)
    threshold = np.maximum(peak * params.change_rel, params.change_abs)
    activity = (delta > threshold).mean(axis=0).astype(np.float32)
    energy = delta.mean(axis=0).astype(np.float32)

    # Correlation of the delta *schedules* of neighbouring tiles. Slides that
    # belong to the same surface step at the same instants; independent panels
    # do not.
    flat = delta.reshape(n - 1, -1).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(flat, axis=0)
    unit = (flat / np.where(norm > 1e-9, norm, 1.0)).reshape(n - 1, rows, cols)
    corr_h = (unit[:, :, :-1] * unit[:, :, 1:]).sum(axis=0).astype(np.float32)
    corr_v = (unit[:, :-1, :] * unit[:, 1:, :]).sum(axis=0).astype(np.float32)

    return TileGrid(
        cols=cols,
        rows=rows,
        xs=xs + x0,
        ys=ys + y0,
        frame_w=w,
        frame_h=h,
        delta=delta,
        times=sample.times,
        activity=activity,
        energy=energy,
        peak=peak,
        content=content,
        corr_h=corr_h,
        corr_v=corr_v,
    )


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def _activity_split(values: np.ndarray, params: RegionParams) -> tuple[float, float]:
    """Split the tile activities at their widest multiplicative gap.

    Returning `(threshold, separation)`. `separation` is the size of that gap;
    when it is small the distribution is unimodal and there is nothing to split
    — one camera filling the frame looks exactly like that, and inventing a
    boundary in it would be the single worst thing this module could do.
    """
    if values.size < 4:
        return 0.0, 1.0
    floor = max(1.0 / max(values.size, 1), 1e-4)
    v = np.sort(np.maximum(values.ravel(), floor))
    ratios = v[1:] / v[:-1]
    # Both sides must be big enough to be a region at all.
    margin = max(params.min_tiles, 1)
    if ratios.size <= 2 * margin:
        return 0.0, 1.0
    inner = ratios[margin - 1 : ratios.size - margin + 1]
    k = int(np.argmax(inner)) + margin - 1
    return float(math.sqrt(v[k] * v[k + 1])), float(ratios[k])


def classify_tiles(grid: TileGrid, params: RegionParams) -> tuple[np.ndarray, float, float]:
    """Label every tile `moving` / `static` / `unknown` / `frozen` / `dead`.

    `frozen` means the tile never changed at all, so it has no activity to
    compare and no schedule to correlate — a slide's template border, the
    matte around a camera feed. Those are held back from the split and from
    clustering, and grown into whichever region ends up surrounding them.
    Letting them cluster on their own merges every unrelated still area in the
    frame into one meaningless rectangle.
    """
    frozen = grid.peak < params.peak_min
    blank = grid.content < params.content_min
    live = ~frozen

    labels = np.full((grid.rows, grid.cols), "dead", dtype=object)
    labels[frozen & ~blank] = "frozen"
    if not live.any():
        return labels, 0.0, 1.0

    threshold, separation = _activity_split(grid.activity[live], params)
    if separation >= params.separation_min:
        labels[live & (grid.activity > threshold)] = "moving"
        labels[live & (grid.activity <= threshold)] = "static"
        return labels, threshold, separation

    # No gap: the frame is one thing. Say which, or say we do not know.
    median = float(np.median(grid.activity[live]))
    if median >= params.unimodal_moving_activity:
        whole = "moving"
    elif median <= params.unimodal_static_activity:
        whole = "static"
    else:
        whole = "unknown"
    labels[live] = whole
    return labels, threshold, separation


_KIND_OF: dict[str, RegionKind] = {"moving": "speaker", "static": "slide", "unknown": "unknown"}


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _similar(grid: TileGrid, labels: np.ndarray, a: tuple[int, int], b: tuple[int, int], horizontal: bool, params: RegionParams) -> bool:
    """Should these two adjacent tiles be part of the same region?

    Flood fill over this predicate rather than a clustering library, because
    the answer has to be explainable one pair at a time: "these joined because
    both are moving and their activities are within 4x" is a sentence a person
    can check against the video.
    """
    la, lb = labels[a], labels[b]
    if la != lb or la in ("dead", "frozen"):
        return False
    if la == "static":
        corr = grid.corr_h[a[0], min(a[1], b[1])] if horizontal else grid.corr_v[min(a[0], b[0]), a[1]]
        return bool(corr >= params.slide_corr_min)
    hi = max(grid.activity[a], grid.activity[b])
    lo = min(grid.activity[a], grid.activity[b])
    return bool(hi <= lo * params.speaker_activity_ratio)


def connected_components(grid: TileGrid, labels: np.ndarray, params: RegionParams) -> list[list[tuple[int, int]]]:
    ds = _DisjointSet(grid.rows * grid.cols)
    idx = lambda r, c: r * grid.cols + c  # noqa: E731
    for r in range(grid.rows):
        for c in range(grid.cols):
            if c + 1 < grid.cols and _similar(grid, labels, (r, c), (r, c + 1), True, params):
                ds.union(idx(r, c), idx(r, c + 1))
            if r + 1 < grid.rows and _similar(grid, labels, (r, c), (r + 1, c), False, params):
                ds.union(idx(r, c), idx(r + 1, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            if labels[r, c] in ("dead", "frozen"):
                continue
            groups.setdefault(ds.find(idx(r, c)), []).append((r, c))
    components = list(groups.values())
    _grow_into_frozen(grid, labels, components)
    return components


def _grow_into_frozen(grid: TileGrid, labels: np.ndarray, components: list[list[tuple[int, int]]]) -> None:
    """Hand every never-changing tile to the region that surrounds it.

    Region growing rather than union: a frozen tile joins a component but can
    never merge two, so a still gutter between a camera and a deck cannot glue
    them into one rectangle. Repeated until nothing more attaches, so a frozen
    interior spreads inward from its edges.
    """
    owner: dict[tuple[int, int], int] = {}
    for i, tiles in enumerate(components):
        for tile in tiles:
            owner[tile] = i
    pending = {(r, c) for r in range(grid.rows) for c in range(grid.cols) if labels[r, c] == "frozen"}

    while pending:
        assigned: dict[tuple[int, int], int] = {}
        for r, c in sorted(pending):
            votes: dict[int, int] = {}
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols and (nr, nc) in owner:
                    i = owner[nr, nc]
                    votes[i] = votes.get(i, 0) + 1
            if votes:
                assigned[r, c] = max(votes, key=lambda i: (votes[i], len(components[i]), -i))
        if not assigned:
            return
        for tile, i in assigned.items():
            components[i].append(tile)
            owner[tile] = i
            pending.discard(tile)


# --------------------------------------------------------------------------
# Rectangles
# --------------------------------------------------------------------------


def _snap(value: float, tol: float, params: RegionParams) -> tuple[float, bool]:
    best = min(params.snap_to, key=lambda t: abs(t - value))
    if abs(best - value) <= tol:
        return best, True
    return value, False


def _rect_for(grid: TileGrid, bbox: tuple[int, int, int, int], params: RegionParams) -> tuple[Rect, bool]:
    r0, c0, r1, c1 = bbox
    x0 = grid.xs[c0] / grid.frame_w
    x1 = grid.xs[c1 + 1] / grid.frame_w
    y0 = grid.ys[r0] / grid.frame_h
    y1 = grid.ys[r1 + 1] / grid.frame_h
    # Tolerance is a fraction of one tile *as it actually is* — on a
    # pillarboxed source the tiles cover only the content box, so they are
    # narrower than 1/cols of the frame and the tolerance must shrink with them.
    tile_w = (grid.xs[-1] - grid.xs[0]) / (grid.frame_w * grid.cols)
    tile_h = (grid.ys[-1] - grid.ys[0]) / (grid.frame_h * grid.rows)
    tol_x = params.snap_tol_tiles * tile_w
    tol_y = params.snap_tol_tiles * tile_h
    x0, s0 = _snap(x0, tol_x, params)
    x1, s1 = _snap(x1, tol_x, params)
    y0, s2 = _snap(y0, tol_y, params)
    y1, s3 = _snap(y1, tol_y, params)
    rect: Rect = (
        round(max(0.0, x0), 4),
        round(max(0.0, y0), 4),
        round(min(1.0, x1) - max(0.0, x0), 4),
        round(min(1.0, y1) - max(0.0, y0), 4),
    )
    return rect, all((s0, s1, s2, s3))


def _change_points(grid: TileGrid, tiles: list[tuple[int, int]], params: RegionParams) -> list[float]:
    """Timestamps at which this region's content stepped.

    A slide advance is a single interval in which nearly every pixel of the
    region changes at once, against a baseline of nothing happening. Runs of
    consecutive flagged intervals collapse to one point so a crossfade is one
    change, not eight.
    """
    rr = np.array([t[0] for t in tiles])
    cc = np.array([t[1] for t in tiles])
    series = grid.delta[:, rr, cc].mean(axis=1)
    if series.size == 0:
        return []
    baseline = float(np.median(series))
    peak = float(np.percentile(series, 99.5))
    threshold = max(
        baseline * params.step_over_baseline,
        peak * params.step_rel_peak,
        params.step_min_delta,
    )
    hits = np.flatnonzero(series > threshold)
    if hits.size == 0:
        return []
    points: list[float] = []
    run_start = hits[0]
    prev = hits[0]
    for i in hits[1:]:
        if i != prev + 1:
            points.append(_midpoint(grid.times, run_start, prev))
            run_start = i
        prev = i
    points.append(_midpoint(grid.times, run_start, prev))
    return [round(p, 3) for p in points[: params.max_change_points]]


def _midpoint(times: np.ndarray, first: int, last: int) -> float:
    end = min(last + 1, times.size - 1)
    return float((times[first] + times[end]) / 2.0)


def _label_for(rect: Rect) -> str:
    x, y, w, h = rect
    if w >= 0.9 and h >= 0.9:
        return "full frame"
    cx, cy = x + w / 2, y + h / 2
    parts = []
    if h < 0.9:
        parts.append("top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle")
    if w < 0.9:
        parts.append("left" if cx < 0.4 else "right" if cx > 0.6 else "centre")
    return "-".join(parts) or "centre"


def _confidence(
    component: Component,
    *,
    separation: float,
    frames: int,
    n_regions: int,
    params: RegionParams,
) -> float:
    """How much of this proposal we actually stand behind.

    Deliberately capped below 1.0. Nothing here has looked at a face or read a
    slide; it has looked at how often rectangles of pixels change.
    """
    score = 0.30
    if n_regions > 1:
        score += 0.25 * min(1.0, math.log(max(separation, 1.0)) / math.log(20.0))
    else:
        score += 0.10  # nothing to separate, but nothing was invented either
    score += 0.20 * component.fill**2
    score += 0.10 * min(1.0, frames / 120.0)
    if component.snapped:
        score += 0.10
    if component.kind == "unknown":
        score -= 0.20
    return round(min(0.95, max(0.05, score)), 2)


def _name_regions(components: list[Component]) -> tuple[list[Region], list[Component]]:
    """Ids and labels, in a stable order: cameras left-to-right, then slides.

    Also returns the components in that same order, so an analysis can be
    read alongside its own output.
    """
    order = {"speaker": 0, "slide": 1, "unknown": 2}
    ordered = sorted(components, key=lambda comp: (order[comp.kind], comp.rect[0], comp.rect[1]))
    counters: dict[RegionKind, int] = {}
    by_kind: dict[RegionKind, int] = {}
    for comp in ordered:
        by_kind[comp.kind] = by_kind.get(comp.kind, 0) + 1

    regions: list[Region] = []
    for comp in ordered:
        n = counters.get(comp.kind, 0)
        counters[comp.kind] = n + 1
        suffix = chr(ord("a") + n) if n < 26 else str(n + 1)
        if comp.kind == "speaker":
            rid = f"cam_{suffix}"
        elif comp.kind == "slide":
            rid = "slides" if by_kind[comp.kind] == 1 else f"slides_{suffix}"
        else:
            rid = f"region_{suffix}"
        regions.append(
            Region(
                id=rid,
                kind=comp.kind,
                rect=comp.rect,
                label=_label_for(comp.rect),
                motion=round(float(comp.activity), 4),
                change_points=comp.change_points if comp.kind != "speaker" else [],
                confidence=comp.confidence,
            )
        )
    return regions, ordered


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def analyze(
    source: str | Path,
    *,
    duration: float | None = None,
    fps: float | None = None,
    params: RegionParams = RegionParams(),
) -> RegionAnalysis | None:
    """Full detection with every intermediate kept. None if there is no video."""
    if duration is None or fps is None:
        probed = probe_video(source, params)
        if probed is None:
            return None
        duration = duration if duration is not None else probed[0]
        fps = fps if fps is not None else probed[1]
    if duration <= 0:
        return None

    sample = sample_frames(source, duration=duration, fps=fps, params=params)
    if sample.count < 2:
        raise RegionDetectionError(f"only {sample.count} frame(s) sampled from {source}")

    grid = tile_grid(sample, params)
    labels, split_at, separation = classify_tiles(grid, params)
    crop = (int(grid.xs[0]), int(grid.ys[0]), int(grid.xs[-1]), int(grid.ys[-1]))

    total_tiles = grid.rows * grid.cols
    min_tiles = max(params.min_tiles, math.ceil(params.min_area_frac * total_tiles))

    components: list[Component] = []
    for tiles in connected_components(grid, labels, params):
        if len(tiles) < min_tiles:
            continue
        rs = [t[0] for t in tiles]
        cs = [t[1] for t in tiles]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
        # Tiles with no signal at all inside the bbox are not evidence against
        # the rectangle: a black gutter inside a camera feed is still the feed.
        dead_inside = sum(
            1
            for r in range(bbox[0], bbox[2] + 1)
            for c in range(bbox[1], bbox[3] + 1)
            if labels[r, c] == "dead"
        )
        fill = len(tiles) / max(area - dead_inside, 1)
        rect, snapped = _rect_for(grid, bbox, params)
        # Tiles that never changed were grown into this component; they say
        # nothing about how much it moves or when it stepped, so the region's
        # own statistics come from the tiles that carried the signal.
        signal = [t for t in tiles if labels[t] != "frozen"] or tiles
        kind = _KIND_OF[str(labels[signal[0]])]
        components.append(
            Component(
                tiles=tiles,
                kind=kind,
                bbox=bbox,
                fill=min(1.0, fill),
                activity=float(np.mean([grid.activity[t] for t in signal])),
                rect=rect,
                snapped=snapped,
                change_points=_change_points(grid, signal, params) if kind != "speaker" else [],
                confidence=0.0,
            )
        )

    components = _drop_duplicates(components)
    for comp in components:
        comp.confidence = _confidence(
            comp,
            separation=separation,
            frames=sample.count,
            n_regions=len(components),
            params=params,
        )

    regions, ordered = _name_regions(components)
    return RegionAnalysis(
        grid=grid,
        sample=sample,
        tile_class=labels,
        split_at=split_at,
        separation=separation,
        crop=crop,
        components=ordered,
        regions=regions,
    )


def _drop_duplicates(components: list[Component]) -> list[Component]:
    """Two rectangles for the same area is noise, not two regions.

    Nesting is left alone: a slide filling the frame with a speaker inset in
    the corner is a real and common layout, and both rectangles are true.
    """
    keep: list[Component] = []
    for comp in sorted(components, key=lambda c: -len(c.tiles)):
        if any(_iou(comp.rect, other.rect) > 0.8 for other in keep):
            continue
        keep.append(comp)
    return keep


def _iou(a: Rect, b: Rect) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_regions(
    source: str | Path,
    *,
    duration: float | None = None,
    fps: float | None = None,
    params: RegionParams = RegionParams(),
) -> RegionsDoc:
    """Propose the named regions of `source`.

    Returns a `RegionsDoc` whose `regions` may legitimately be empty: an
    audio-only file, a video whose frame does not decompose, or one too short
    to measure all produce zero regions, and every caller already has the
    implicit whole-frame region to fall back on.
    """
    analysis = analyze(source, duration=duration, fps=fps, params=params)
    if analysis is None:
        return RegionsDoc(grid=f"{params.grid_x}x{params.grid_y}", frames_sampled=0, regions=[])
    return RegionsDoc(
        grid=f"{analysis.grid.cols}x{analysis.grid.rows}",
        frames_sampled=analysis.sample.count,
        regions=analysis.regions,
    )


def explain(analysis: RegionAnalysis) -> str:
    """A human-readable account of why the proposal came out as it did.

    Not written to the job directory; this is for the person asking "why is
    there no slide region?" at a REPL or from a `--explain` flag.
    """
    grid = analysis.grid
    lines = [
        f"sampled {analysis.sample.count} frames by {analysis.sample.method}",
        f"content box {analysis.crop} of {grid.frame_w}x{grid.frame_h}",
        f"grid {grid.cols}x{grid.rows}",
    ]
    lines.append(f"widest activity gap {analysis.separation:.1f}x at {analysis.split_at:.4f}")
    lines.append("tile classes (activity):")
    symbol = {"moving": "M", "static": "S", "unknown": "?", "frozen": "-", "dead": "."}
    for r in range(grid.rows):
        row = " ".join(
            f"{symbol[str(analysis.tile_class[r, c])]}{grid.activity[r, c]:.2f}" for c in range(grid.cols)
        )
        lines.append("  " + row)
    for comp, region in zip(analysis.components, analysis.regions, strict=False):
        lines.append(
            f"{region.id}: {region.kind} rect={region.rect} tiles={len(comp.tiles)} "
            f"fill={comp.fill:.2f} snapped={comp.snapped} conf={region.confidence} "
            f"changes={len(region.change_points)}"
        )
    if not analysis.regions:
        lines.append("no regions proposed — the frame does not decompose")
    return "\n".join(lines)
