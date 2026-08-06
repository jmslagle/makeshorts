"""Caption rendering: cues, styles, and two interchangeable backends.

The backend is chosen by probing the toolchain, never by configuration --
see `select_backend`. On a machine whose ffmpeg lacks libass (which is the
machine this was built on) that resolves to Pillow, and nothing above this
package needs to know.
"""

from __future__ import annotations

from makeshorts.render.captions.base import (
    CaptionAssets,
    CaptionBackend,
    CaptionOverlay,
    CaptionPosition,
    CaptionRenderError,
    OverlayAssets,
    SubtitleAssets,
)
from makeshorts.render.captions.cues import Cue, CueOptions, CueWord, build_cues, cues_for_clip
from makeshorts.render.captions.style import (
    DEFAULT_STYLE_NAME,
    DEFAULT_STYLES,
    CaptionStyle,
    SafeArea,
    StyleError,
    get_style,
    load_styles,
)

__all__ = [
    "CaptionAssets",
    "CaptionBackend",
    "CaptionOverlay",
    "CaptionPosition",
    "CaptionRenderError",
    "OverlayAssets",
    "SubtitleAssets",
    "Cue",
    "CueOptions",
    "CueWord",
    "build_cues",
    "cues_for_clip",
    "CaptionStyle",
    "SafeArea",
    "StyleError",
    "DEFAULT_STYLES",
    "DEFAULT_STYLE_NAME",
    "get_style",
    "load_styles",
    "select_backend",
]


def select_backend(caps: object | None = None) -> CaptionBackend:
    """The caption backend this machine can actually run.

    `caps` is an `FFmpegCaps`; omitted, it is probed (and cached). Imports are
    deferred so that importing this package does not shell out to ffmpeg --
    `cues.py` and `style.py` are useful in contexts with no video toolchain at
    all, and unit tests should not pay for a subprocess.
    """
    from makeshorts.render.caps import probe_caps

    probed = caps if caps is not None else probe_caps()
    if getattr(probed, "caption_backend", "pillow") == "ass":
        from makeshorts.render.captions.ass_backend import AssCaptionBackend

        return AssCaptionBackend(caps=probed)

    from makeshorts.render.captions.pillow_backend import PillowCaptionBackend

    return PillowCaptionBackend()
