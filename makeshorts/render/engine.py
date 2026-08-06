"""The swappability contract.

ffmpeg is today's renderer. It is not the only one this design admits, and the
edit list must never learn its name. A second engine (Remotion, MoviePy,
Resolve, a cloud service) implements `RenderEngine`, registers a name, and
`ms render --engine X` selects it. Nothing in `select/` or `prepare/` may
import anything below this module.

`layout.py` is deliberately NOT part of any engine: it is pure geometry, shared
by all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from makeshorts.select.schema import Clip, ClipsDoc


class Receipt(BaseModel):
    """What was rendered, by what, from what.

    Written next to each output as `<name>.json` so a clip on disk can always
    be traced back to the edit list and toolchain that produced it.
    """

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    output: str
    engine: str
    engine_version: str
    source_sha256: str
    clip_sha256: str  # hash of this clip's entry in clips.json
    duration: float
    commands: list[str] = []


class Capability(str):
    """Capability names an engine may advertise."""


# Layout modes and caption features an engine may or may not support. `ms
# render` checks the edit list against the selected engine's capabilities and
# fails before encoding rather than halfway through a batch.
CAP_FOCUS = "focus"
CAP_HERO_INSET = "hero_inset"
CAP_STACK = "stack"
CAP_CONTAIN_BLUR = "contain_blur"
CAP_MULTI_SPAN = "multi_span"
CAP_BURNED_CAPTIONS = "burned_captions"
CAP_KARAOKE_CAPTIONS = "karaoke_captions"


@runtime_checkable
class RenderEngine(Protocol):
    name: str

    def capabilities(self) -> set[str]:
        """Which of the CAP_* features this engine can actually deliver on this
        machine. May be probed at runtime -- the ffmpeg engine's caption
        capabilities depend on how the installed binary was built."""
        ...

    def version(self) -> str:
        ...

    def render(self, doc: ClipsDoc, clip: Clip, source: Path, out: Path) -> Receipt:
        """Render one clip to `out`. Must be idempotent: same inputs, same
        bytes-equivalent output, no reliance on prior runs."""
        ...


class UnsupportedByEngine(RuntimeError):
    """Raised when an edit list asks for something the selected engine cannot
    do. Carries the missing capabilities so the CLI can explain itself."""

    def __init__(self, engine: str, missing: set[str]) -> None:
        self.engine = engine
        self.missing = missing
        super().__init__(
            f"engine {engine!r} cannot satisfy this edit list; missing: {sorted(missing)}"
        )


_REGISTRY: dict[str, RenderEngine] = {}


def register(engine: RenderEngine) -> RenderEngine:
    _REGISTRY[engine.name] = engine
    return engine


def get_engine(name: str) -> RenderEngine:
    if name not in _REGISTRY:
        raise KeyError(f"unknown render engine {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def required_capabilities(doc: ClipsDoc) -> set[str]:
    """What this edit list demands of an engine. Pure inspection of the
    semantic layout -- no engine involved."""
    needed: set[str] = set()
    for clip in doc.clips:
        spans = clip.spans
        if len(spans) > 1:
            needed.add(CAP_MULTI_SPAN)
        for span in spans:
            needed.add(span.mode)
            if getattr(span, "fit", None) == "contain_blur":
                needed.add(CAP_CONTAIN_BLUR)
            if getattr(span, "hero_fit", None) == "contain_blur":
                needed.add(CAP_CONTAIN_BLUR)
        if clip.captions.enabled:
            needed.add(CAP_BURNED_CAPTIONS)
    return needed
