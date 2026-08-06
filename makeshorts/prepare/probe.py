"""ffprobe -> media.json.

Facts about the source file and nothing else. Every downstream stage trusts
`duration` and `resolution` from here rather than re-probing, so this runs once
and is cached in the job directory.

The sha256 is what makes a render receipt meaningful: it says *which* bytes
produced a clip, so a re-render against a re-exported source fails loudly
instead of silently producing different output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from makeshorts.artifacts import MediaDoc

# Read in chunks: webinar sources are multi-gigabyte and there is no reason to
# hold one in memory to hash it.
HASH_CHUNK_BYTES = 1 << 20


class ProbeError(RuntimeError):
    """ffprobe failed, is missing, or returned something unusable."""


def find_tool(name: str) -> str:
    """Locate an ffmpeg-family binary.

    Homebrew's path is the fallback because a GUI-launched process does not
    always inherit a shell PATH that includes /opt/homebrew/bin.
    """
    override = os.environ.get(f"MAKESHORTS_{name.upper()}")
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    homebrew = Path("/opt/homebrew/bin") / name
    if homebrew.exists():
        return str(homebrew)
    raise ProbeError(
        f"{name} not found on PATH. Install it (`brew install ffmpeg`) or set "
        f"MAKESHORTS_{name.upper()} to its path."
    )


def ffprobe_path() -> str:
    return find_tool("ffprobe")


def sha256_file(path: str | Path, *, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_json(path: str | Path) -> dict:
    """Raw `-show_format -show_streams` output, parsed."""
    src = Path(path)
    if not src.exists():
        raise ProbeError(f"no such file: {src}")
    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(src),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed on {src}:\n{proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - ffprobe would have to lie
        raise ProbeError(f"ffprobe returned malformed JSON for {src}: {exc}") from exc


def parse_fps(value: str | None) -> float:
    """`"30000/1001"` -> 29.97. Returns 0.0 for ffprobe's `"0/0"` placeholder."""
    if not value:
        return 0.0
    try:
        frac = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(frac)


def _first_stream(streams: list[dict], codec_type: str) -> dict | None:
    for stream in streams:
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _duration(info: dict, video: dict | None, audio: dict | None) -> float:
    """Container duration, falling back to a stream's.

    Some MKV and fragmented-MP4 exports carry no duration in the container
    header; the stream usually still has one.
    """
    for candidate in (
        info.get("format", {}).get("duration"),
        (video or {}).get("duration"),
        (audio or {}).get("duration"),
    ):
        try:
            value = float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    raise ProbeError(
        "could not determine duration -- the file may be truncated or still being written"
    )


def media_doc_from_probe(
    info: dict,
    *,
    path: str | Path,
    sha256: str,
) -> MediaDoc:
    """Pure: ffprobe output -> MediaDoc. Split out so it is testable without
    running ffprobe."""
    streams = info.get("streams") or []
    video = _first_stream(streams, "video")
    audio = _first_stream(streams, "audio")

    if video is not None:
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        # avg_frame_rate is the honest number for VFR sources; r_frame_rate is
        # the container's claimed base rate and is often a useless multiple.
        fps = parse_fps(video.get("avg_frame_rate")) or parse_fps(video.get("r_frame_rate"))
    else:
        # Audio-only input. Job.source accepts .mp3/.m4a/.wav, so this is a real
        # case, not a defensive branch.
        width = height = 0
        fps = 0.0

    return MediaDoc(
        path=str(path),
        duration=round(_duration(info, video, audio), 3),
        resolution=(width, height),
        fps=round(fps, 6),
        has_audio=audio is not None,
        audio_channels=int(audio.get("channels") or 0) if audio else 0,
        sha256=sha256,
    )


def probe(path: str | Path, *, sha256: str | None = None) -> MediaDoc:
    """Probe `path` and hash it.

    `sha256` may be passed in when the caller has already hashed the file (the
    ingest step does), to avoid a second full read.
    """
    src = Path(path)
    info = ffprobe_json(src)
    return media_doc_from_probe(
        info,
        path=str(src),
        sha256=sha256 if sha256 is not None else sha256_file(src),
    )


__all__ = [
    "HASH_CHUNK_BYTES",
    "ProbeError",
    "ffprobe_path",
    "find_tool",
    "ffprobe_json",
    "media_doc_from_probe",
    "parse_fps",
    "probe",
    "sha256_file",
]
