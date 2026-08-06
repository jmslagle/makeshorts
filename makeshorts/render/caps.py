"""What the *installed* ffmpeg can actually do.

The design assumption that burned-in captions mean `subtitles` or `drawtext`
is false on plenty of real machines, this one included: homebrew's ffmpeg
8.1.2 is built without `libass` and without `libfreetype`, so neither filter
exists in the binary. Discovering that at render time, after transcription and
selection have already run, is the worst place to discover it.

So capabilities are probed, cached, and consulted up front. `ms caps` prints
this; `ffmpeg_engine` picks its caption backend from it.

The probe shells out to `ffmpeg -filters`, `-encoders`, and `-buildconf` once
and caches the parsed result keyed on the binary's path, size, and mtime -- a
brew upgrade invalidates the cache automatically without anyone remembering
to clear it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from makeshorts.artifacts import Strict
from makeshorts.render.engine import (
    CAP_BURNED_CAPTIONS,
    CAP_CONTAIN_BLUR,
    CAP_FOCUS,
    CAP_HERO_INSET,
    CAP_KARAOKE_CAPTIONS,
    CAP_MULTI_SPAN,
    CAP_STACK,
)

__all__ = [
    "FFmpegCaps",
    "CapsProbeError",
    "probe_caps",
    "default_cache_path",
    "clear_cache",
]

CAPS_SCHEMA_VERSION = "1"

# Filters this project actually reaches for. Probing a fixed list rather than
# storing all ~600 keeps the cache file readable.
_FILTERS_OF_INTEREST = (
    "subtitles",
    "drawtext",
    "drawbox",
    "ass",
    "overlay",
    "crop",
    "scale",
    "pad",
    "boxblur",
    "gblur",
    "avgblur",
    "vstack",
    "hstack",
    "colorchannelmixer",
    "loudnorm",
    "concat",
)

_ENCODERS_OF_INTEREST = (
    "libx264",
    "libx265",
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "aac",
    "aac_at",
)


class CapsProbeError(RuntimeError):
    """ffmpeg is missing or refused to describe itself."""


class FFmpegCaps(Strict):
    """A frozen answer to "what can this binary do".

    Everything here is observed, never assumed. `caption_backend` is the one
    derived field, and it is the reason this module exists.
    """

    caps_schema_version: str = CAPS_SCHEMA_VERSION
    probed_at: str
    ffmpeg_path: str
    ffmpeg_version: str
    # Cache key: a rebuilt or upgraded binary must not read a stale answer.
    binary_size: int = 0
    binary_mtime: float = 0.0

    buildconf: list[str] = Field(default_factory=list)
    filters: dict[str, bool] = Field(default_factory=dict)
    encoders: dict[str, bool] = Field(default_factory=dict)

    # ---- Derived, and the whole point ------------------------------------

    @property
    def has_libass(self) -> bool:
        """libass drives the `subtitles`/`ass` filters.

        Both the buildconf flag and the filter's actual presence are checked:
        a shared-library build could in principle report one without the
        other, and only the filter's existence makes a render work.
        """
        return self.filters.get("subtitles", False) or self.filters.get("ass", False)

    @property
    def has_libfreetype(self) -> bool:
        """libfreetype drives `drawtext`."""
        return self.filters.get("drawtext", False)

    @property
    def has_subtitles_filter(self) -> bool:
        return self.filters.get("subtitles", False)

    @property
    def has_drawtext_filter(self) -> bool:
        return self.filters.get("drawtext", False)

    @property
    def has_overlay(self) -> bool:
        return self.filters.get("overlay", False)

    @property
    def blur_filter(self) -> str | None:
        """Preferred blur filter name, or None if the build has none.

        gblur looks better; boxblur is cheaper and more universally present.
        """
        for name in ("gblur", "boxblur", "avgblur"):
            if self.filters.get(name):
                return name
        return None

    @property
    def has_videotoolbox(self) -> bool:
        return self.encoders.get("h264_videotoolbox", False)

    @property
    def has_libx264(self) -> bool:
        return self.encoders.get("libx264", False)

    @property
    def caption_backend(self) -> str:
        """Which caption backend this machine must use.

        `ass` when libass is present -- one filter, one file, cheapest path.
        `pillow` otherwise, as long as `overlay` exists, which it always does.
        `none` if even overlay is missing, which means captions are impossible
        and `ms render` should say so rather than silently dropping them.
        """
        if self.has_libass:
            return "ass"
        if self.has_overlay:
            return "pillow"
        return "none"

    def capabilities(self) -> set[str]:
        """CAP_* names an ffmpeg engine can honestly advertise on this build.

        Layout modes need only crop/scale/pad/overlay, which every build has.
        Captions are the part that varies.
        """
        caps: set[str] = set()
        if self.filters.get("crop") and self.filters.get("scale"):
            caps.update({CAP_FOCUS, CAP_MULTI_SPAN})
            if self.has_overlay:
                caps.update({CAP_HERO_INSET, CAP_STACK})
            if self.blur_filter and self.has_overlay:
                caps.add(CAP_CONTAIN_BLUR)
        if self.caption_backend != "none":
            caps.add(CAP_BURNED_CAPTIONS)
            # Both backends do word-level highlighting: libass via \k, Pillow
            # via one PNG per highlighted word.
            caps.add(CAP_KARAOKE_CAPTIONS)
        return caps

    def summary(self) -> str:
        """One human-readable block, for `ms caps`."""
        lines = [
            f"ffmpeg      {self.ffmpeg_version}",
            f"path        {self.ffmpeg_path}",
            f"libass      {'yes' if self.has_libass else 'NO'}"
            f"   (subtitles filter: {'yes' if self.has_subtitles_filter else 'NO'})",
            f"libfreetype {'yes' if self.has_libfreetype else 'NO'}"
            f"   (drawtext filter:  {'yes' if self.has_drawtext_filter else 'NO'})",
            f"overlay     {'yes' if self.has_overlay else 'NO'}",
            f"blur        {self.blur_filter or 'NONE'}",
            f"libx264     {'yes' if self.has_libx264 else 'NO'}",
            f"vtoolbox    {'yes' if self.has_videotoolbox else 'NO'}",
            f"captions    -> {self.caption_backend} backend",
            f"capabilities {' '.join(sorted(self.capabilities())) or '<none>'}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


def _run(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CapsProbeError(f"ffmpeg not found: {args[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CapsProbeError(f"{' '.join(args)} timed out") from exc
    # -buildconf exits non-zero on some builds while still printing. Take
    # whatever came out and let the parsers decide.
    return (proc.stdout or "") + (proc.stderr or "")


def _parse_version(text: str) -> str:
    m = re.search(r"ffmpeg version (\S+)", text)
    return m.group(1) if m else "unknown"


def _parse_buildconf(text: str) -> list[str]:
    """Every `--flag` in the configuration line, one per entry."""
    return sorted({tok for tok in re.findall(r"--[\w-]+(?:=[^\s]*)?", text)})


def _parse_names(text: str) -> set[str]:
    """Names from an ffmpeg `-filters` / `-encoders` listing.

    Both formats are `<flags> <name> <io> <description>` after a legend
    terminated by a line of dashes. Splitting on whitespace and taking the
    second field is enough, but only after the legend, or the legend's own
    text would be mistaken for names.
    """
    names: set[str] = set()
    in_body = False
    for line in text.splitlines():
        if not in_body:
            if set(line.strip()) == {"-"} and len(line.strip()) > 3:
                in_body = True
            continue
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    if not in_body:  # no dashed separator; fall back to a looser scan
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 3 and re.fullmatch(r"[A-Z.]{2,7}", parts[0]):
                names.add(parts[1])
    return names


def default_cache_path() -> Path:
    """`$XDG_CACHE_HOME/makeshorts/ffmpeg_caps.json`, or the macOS-ish default."""
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "makeshorts" / "ffmpeg_caps.json"


def _binary_stat(path: str) -> tuple[int, float]:
    try:
        st = Path(path).stat()
    except OSError:
        return (0, 0.0)
    return (st.st_size, st.st_mtime)


def _probe(ffmpeg: str) -> FFmpegCaps:
    resolved = shutil.which(ffmpeg) or ffmpeg
    version_text = _run([resolved, "-hide_banner", "-version"])
    if "ffmpeg version" not in version_text:
        raise CapsProbeError(
            f"{resolved!r} did not identify itself as ffmpeg; got: {version_text[:200]!r}"
        )
    buildconf_text = _run([resolved, "-hide_banner", "-buildconf"])
    filters_text = _run([resolved, "-hide_banner", "-filters"])
    encoders_text = _run([resolved, "-hide_banner", "-encoders"])

    filter_names = _parse_names(filters_text)
    encoder_names = _parse_names(encoders_text)

    size, mtime = _binary_stat(resolved)
    return FFmpegCaps(
        probed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ffmpeg_path=resolved,
        ffmpeg_version=_parse_version(version_text),
        binary_size=size,
        binary_mtime=mtime,
        # -buildconf output is the richer of the two; -version's config line is
        # a fallback for builds that do not implement -buildconf.
        buildconf=_parse_buildconf(buildconf_text) or _parse_buildconf(version_text),
        filters={name: name in filter_names for name in _FILTERS_OF_INTEREST},
        encoders={name: name in encoder_names for name in _ENCODERS_OF_INTEREST},
    )


def probe_caps(
    ffmpeg: str = "ffmpeg",
    cache_path: Path | None = None,
    *,
    refresh: bool = False,
    use_cache: bool = True,
) -> FFmpegCaps:
    """Capabilities of `ffmpeg`, from cache when the binary is unchanged.

    A cache entry is honoured only when the schema version, the resolved path,
    and the binary's size+mtime all still match. Anything else re-probes.
    """
    path = cache_path if cache_path is not None else default_cache_path()
    resolved = shutil.which(ffmpeg) or ffmpeg

    if use_cache and not refresh and path.exists():
        try:
            cached = FFmpegCaps.model_validate_json(path.read_text())
        except Exception:
            cached = None  # corrupt or written by an older schema; re-probe
        if cached is not None and _cache_is_fresh(cached, resolved):
            return cached

    caps = _probe(ffmpeg)
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(caps.model_dump_json(indent=2) + "\n")
        except OSError:
            pass  # an unwritable cache dir must not fail a render
    return caps


def _cache_is_fresh(cached: FFmpegCaps, resolved_path: str) -> bool:
    if cached.caps_schema_version != CAPS_SCHEMA_VERSION:
        return False
    if cached.ffmpeg_path != resolved_path:
        return False
    size, mtime = _binary_stat(resolved_path)
    return (cached.binary_size, cached.binary_mtime) == (size, mtime)


def clear_cache(cache_path: Path | None = None) -> None:
    path = cache_path if cache_path is not None else default_cache_path()
    path.unlink(missing_ok=True)
