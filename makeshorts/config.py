"""Typed loaders for `config/render.yaml` and `config/styles.yaml`.

Two files, two jobs:

    render.yaml   how clips are encoded -- codecs, CRF, output frame, ffmpeg
                  binary. Everything engine-shaped lives here so that none of
                  it has to live in clips.json.
    styles.yaml   how captions look -- font, size, colours, safe area. Named
                  styles, because clips.json refers to a style by *name* and
                  must never carry a font path.

Neither file is consulted by `select/` or `prepare/`. Both are resolved through
this module and nowhere else, so there is exactly one place that knows what a
default is.

**Per-job overrides.** A job may carry a partial copy of either file at
`jobs/<slug>/config/`. It is merged key-by-key (recursively) over the
repo-level file, so a job that only wants a faster encoder writes four lines:

    video:
      codec: h264_videotoolbox

Lists replace wholesale rather than concatenating -- appending to `extra_args`
by accident is a worse failure than restating it.

Every load failure raises `ConfigError` with the file path and, for a schema
violation, pydantic's field-by-field report. A typo in a config file should
name itself; it should never surface as a `KeyError` three modules later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_DIR = Path("config")

RENDER_FILENAME = "render.yaml"
STYLES_FILENAME = "styles.yaml"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ConfigError(RuntimeError):
    """Base for everything that can go wrong loading a config file."""


class ConfigNotFound(ConfigError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"config file not found: {path}")


class ConfigInvalid(ConfigError):
    """The file exists but is not usable.

    `detail` is the parser's or pydantic's own report, indented under a first
    line that names the file -- so the message reads top-down from "which file"
    to "which field".
    """

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        indented = "\n".join(f"  {line}" for line in detail.splitlines())
        super().__init__(f"{path} is not valid:\n{indented}")


class UnknownStyle(ConfigError):
    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        super().__init__(f"unknown caption style {name!r}; defined styles: {', '.join(known)}")


# --------------------------------------------------------------------------
# YAML plumbing
# --------------------------------------------------------------------------


class Strict(BaseModel):
    """Unknown keys are an error.

    A silently-ignored `codec:` under the wrong parent is the exact bug this
    project cannot afford: the render would succeed and be wrong.
    """

    model_config = ConfigDict(extra="forbid")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Parse one YAML file into a mapping, or raise `ConfigError`.

    An empty file is an empty mapping -- a per-job override file with
    everything commented out is legal and means "no overrides".
    """
    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError as exc:
        raise ConfigNotFound(p) from exc
    except OSError as exc:
        raise ConfigInvalid(p, str(exc)) from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigInvalid(p, _yaml_error_detail(exc)) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigInvalid(
            p, f"expected a mapping at the top level, found {type(data).__name__}"
        )
    return data


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    """PyYAML's own message, which already carries line/column and a caret.

    Kept rather than reformatted: `mark` gives the exact offending line, which
    is the only thing anyone wants when a config file will not parse.
    """
    problem = getattr(exc, "problem", None)
    mark = getattr(exc, "problem_mark", None)
    if problem and mark is not None:
        return f"malformed YAML: {problem} (line {mark.line + 1}, column {mark.column + 1})"
    return f"malformed YAML: {exc}"


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """`overlay` over `base`, recursing into nested mappings.

    Lists and scalars are replaced, not merged. A job overriding
    `video.extra_args` means exactly what it says.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _validate(model: type[BaseModel], data: Mapping[str, Any], sources: list[Path]) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigInvalid(sources[-1], _validation_detail(exc, sources)) from exc


def _validation_detail(exc: ValidationError, sources: list[Path]) -> str:
    """One line per bad field: `video.crf: input should be a valid integer`.

    Pydantic's default rendering repeats the model name and the input value on
    separate lines, which buries the field path. The field path is the whole
    answer here, so it goes first.
    """
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"{loc}: {err['msg']}")
    if len(sources) > 1:
        merged_from = " + ".join(str(s) for s in sources)
        lines.append(f"(after merging {merged_from})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# render.yaml
# --------------------------------------------------------------------------

# Enumerated rather than free strings: a typo'd codec should fail while reading
# the config, not 40 minutes into a batch when ffmpeg finally rejects it.
Codec = Literal["libx264", "libx265", "h264_videotoolbox", "hevc_videotoolbox"]

Preset = Literal[
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]

LogLevel = Literal["quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug"]

CaptionBackend = Literal["auto", "ass", "pillow"]


class VideoConfig(Strict):
    codec: Codec = "libx264"
    crf: int = Field(default=20, ge=0, le=51)
    preset: Preset = "medium"
    pix_fmt: str = "yuv420p"
    # Hardware encoders have no CRF mode; this is what they use instead.
    bitrate: str = "8M"
    profile: str = "high"
    level: str = "4.1"
    faststart: bool = True
    # Appended verbatim to the output side of the command. The escape hatch for
    # flags this schema does not model.
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("level", "bitrate", "profile", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        """`level: 4.1` unquoted is a float in YAML and a string everywhere
        else. Accept both rather than making the user learn that."""
        return str(v) if isinstance(v, (int, float)) else v

    @property
    def is_hardware(self) -> bool:
        """Hardware encoders ignore `crf`/`preset` and are not bit-deterministic."""
        return self.codec.endswith("_videotoolbox")


class LoudnormConfig(Strict):
    """EBU R128 targets, passed to ffmpeg's `loudnorm`.

    -14 LUFS integrated is what the major platforms normalize to; delivering at
    their target is what stops them pulling the clip down.
    """

    integrated: float = Field(default=-14.0, le=0.0)
    true_peak: float = Field(default=-1.5, le=0.0)
    range: float = Field(default=11.0, gt=0.0)


class AudioConfig(Strict):
    codec: str = "aac"
    bitrate: str = "192k"
    sample_rate: int = 48000
    channels: int = Field(default=2, ge=1, le=8)
    normalize: LoudnormConfig = Field(default_factory=LoudnormConfig)

    @field_validator("bitrate", mode="before")
    @classmethod
    def _stringify(cls, v: Any) -> Any:
        return str(v) if isinstance(v, (int, float)) else v


class OutputConfig(Strict):
    # Relative to the job directory.
    dir: str = "out"
    container: str = "mp4"
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    fps: int = Field(default=30, gt=0)
    overwrite: bool = False


class FfmpegConfig(Strict):
    binary: str = "ffmpeg"
    probe_binary: str = "ffprobe"
    # 0 = one thread per core, ffmpeg's own default.
    threads: int = Field(default=0, ge=0)
    loglevel: LogLevel = "warning"


class CaptionsConfig(Strict):
    # `auto` probes the installed ffmpeg and falls back to the Pillow backend
    # when it was built without libass.
    backend: CaptionBackend = "auto"
    # Resolved against styles.yaml. A clip naming its own style wins over this.
    default_style: str = "pill-karaoke"
    cache: bool = True
    # Cues per filtergraph stage, so the graph stays parseable on a long clip.
    overlay_chunk: int = Field(default=40, ge=1)


class BrandingConfig(Strict):
    """A watermark composited over every clip.

    Lives here rather than in clips.json for the same reason caption styling
    does: which image, how big, which corner is a property of the channel, not
    an editorial judgment about a particular clip. An edit list names a preset
    or says "none"; it never carries a file path.
    """

    enabled: bool = False
    image: str | None = None
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_right"
    # Fraction of output width the logo should span.
    scale: float = Field(default=0.16, gt=0.01, le=0.5)
    opacity: float = Field(default=0.85, gt=0.0, le=1.0)
    # Fraction of output width kept clear at the edges.
    margin: float = Field(default=0.04, ge=0.0, le=0.2)


# xfade's own names. Enumerated so a typo fails while reading config rather
# than 40 minutes into a batch.
TransitionType = Literal[
    "fade", "fadeblack", "fadewhite", "dissolve",
    "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "circleopen", "circleclose", "smoothleft", "smoothright",
    "smoothup", "smoothdown", "pixelize", "radial", "hblur",
]


class TransitionConfig(Strict):
    """What a named transition looks like.

    `clips.json` says *which cuts* get one -- an editorial call about this
    clip. This says what one is, which is presentation policy shared by every
    engine.
    """

    type: TransitionType = "fade"
    # Seconds of overlap. Kept short by default: on a 15-60s clip a long
    # dissolve spends real runtime and reads as sluggish.
    duration: float = Field(default=0.4, gt=0.0, le=3.0)


class FadeConfig(Strict):
    """Fade from and to black at the clip's own edges.

    Uniform across a set rather than per clip, so it lives here with no
    clips.json counterpart. `in_` is deliberately small: a fade-in eats the
    hook window, which is the three seconds `hook_strength` is scored on.
    """

    in_: float = Field(default=0.0, ge=0.0, le=2.0, alias="in")
    out: float = Field(default=0.0, ge=0.0, le=2.0)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class OutroConfig(Strict):
    """A bumper concatenated after the clip.

    A supplied file rather than a generated card: the design belongs in a
    design tool, and normalising an existing video is less code than
    reimplementing motion graphics badly.
    """

    file: str | None = None
    # Bumpers often carry their own music. "mute" silences it, which is the
    # safer default when the clip's own audio has been loudness-normalised and
    # the bumper has not.
    audio: Literal["keep", "mute"] = "keep"
    # Trim a long bumper down; None uses the whole file.
    max_duration: float | None = Field(default=None, gt=0.0)


class RenderConfig(Strict):
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    ffmpeg: FfmpegConfig = Field(default_factory=FfmpegConfig)
    captions: CaptionsConfig = Field(default_factory=CaptionsConfig)
    # Preset name -> settings. `clips.json` may name one, or "none".
    branding: dict[str, BrandingConfig] = Field(default_factory=dict)
    default_branding: str | None = None
    # Geometry policy handed to layout.py. Validated against the real
    # LayoutOptions model in Config.layout_options() rather than mirrored
    # here, so the two cannot drift apart.
    layout: dict[str, Any] = Field(default_factory=dict)
    # Preset name -> what that transition looks like. clips.json names one.
    transitions: dict[str, TransitionConfig] = Field(default_factory=dict)
    fade: FadeConfig = Field(default_factory=FadeConfig)
    outro: dict[str, OutroConfig] = Field(default_factory=dict)
    default_outro: str | None = None


# --------------------------------------------------------------------------
# styles.yaml
#
# The CaptionStyle model lives in `makeshorts/render/captions/style.py` and is
# owned by the render side, which is the only code that has an opinion about
# what a stroke width means. Defining a second model here would guarantee the
# two drift, so this module does the one thing the render side does not: it
# layers the repo-level file under a per-job override, and turns every failure
# into the same ConfigError the rest of this module raises.
#
# The import is deliberately deferred into `load_styles()`. `cli.py` imports
# this module at startup and must keep working against a tree where the render
# package is absent or broken.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StylesDoc:
    """Named caption styles, plus which one to use when nobody says.

    `styles` values are `render.captions.style.CaptionStyle` instances. They
    are not annotated as such because that would drag the render package into
    this module's import graph.
    """

    styles: dict[str, Any]
    default: str = "pill-karaoke"

    def get(self, name: str | None = None) -> Any:
        key = name or self.default
        try:
            return self.styles[key]
        except KeyError as exc:
            raise UnknownStyle(key, sorted(self.styles)) from exc

    @property
    def names(self) -> list[str]:
        return sorted(self.styles)


def _split_styles(data: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], str | None]:
    """`{default: ..., styles: {...}}`, or a bare mapping of name -> style.

    Both shapes are accepted because `style.py` accepts both, and a config file
    that loads in one place and not the other would be the worst outcome here.
    """
    if "styles" in data:
        raw = data["styles"]
        if not isinstance(raw, Mapping):
            raise ConfigInvalid(path, "`styles` must be a mapping of name -> style")
        return dict(raw), data.get("default")
    reserved = {"default", "version"}
    return {k: v for k, v in data.items() if k not in reserved}, data.get("default")


def load_styles(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    overlay_dir: str | Path | None = None,
) -> StylesDoc:
    """`config/styles.yaml`, with `<overlay_dir>/styles.yaml` merged over it.

    A missing styles.yaml is not an error: `style.py` ships a complete set of
    built-in styles, and a checkout with no config file should still render.
    Each YAML entry overrides the built-in of the same name field by field, so
    a style may name only what it changes.
    """
    try:
        from makeshorts.render.captions import style as style_mod
    except ImportError as exc:
        raise ConfigError(
            f"cannot load caption styles: {exc}. The style model lives in "
            f"makeshorts/render/captions/style.py."
        ) from exc

    paths = [p for p in _layer_paths(STYLES_FILENAME, config_dir, overlay_dir, required=False)]
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(path))

    last = paths[-1] if paths else Path(config_dir) / STYLES_FILENAME
    entries, default = _split_styles(merged, last)

    styles = {name: s.model_copy(deep=True) for name, s in style_mod.DEFAULT_STYLES.items()}
    for name, fields in entries.items():
        fields = fields or {}
        if not isinstance(fields, Mapping):
            raise ConfigInvalid(last, f"style {name!r} must be a mapping")
        base = styles.get(name)
        data = (base.model_dump() if base is not None else {}) | dict(fields)
        # The style knows its own name; the YAML key is the single source of it.
        data["name"] = name
        # Nested, so a style can nudge one inset without restating the rest.
        if base is not None and isinstance(fields.get("safe_area"), Mapping):
            data["safe_area"] = base.safe_area.model_dump() | dict(fields["safe_area"])
        try:
            styles[name] = style_mod.CaptionStyle.model_validate(data)
        except ValidationError as exc:
            raise ConfigInvalid(last, f"style {name!r}:\n{_validation_detail(exc, paths)}") from exc
        except Exception as exc:  # StyleError from an unparseable colour
            raise ConfigInvalid(last, f"style {name!r}: {exc}") from exc

    chosen = default or style_mod.DEFAULT_STYLE_NAME
    if chosen not in styles:
        raise ConfigInvalid(
            last,
            f"default style {chosen!r} is not defined; defined styles: "
            f"{', '.join(sorted(styles))}",
        )
    return StylesDoc(styles=styles, default=chosen)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Everything the render side reads from disk, loaded once.

    `sources` records which files actually contributed, in merge order, so an
    unexpected setting can be traced back to the file that set it.
    """

    render: RenderConfig
    styles: StylesDoc
    sources: tuple[Path, ...]

    def style_for(self, name: str | None = None) -> Any:
        """The style a clip asked for, falling back to render.yaml's default.

        Returns a `render.captions.style.CaptionStyle`.
        """
        return self.styles.get(name or self.render.captions.default_style)

    def layout_options(self) -> Any:
        """`render.yaml`'s `layout:` block as a `render.layout.LayoutOptions`.

        Deferred import for the same reason as styles: the model belongs to the
        render side, which is the only code with an opinion about what a blur
        radius means, and mirroring it here would guarantee the two drift.

        These are presentation policy, not editorial decisions -- how strong
        the backdrop blur is, how large an inset sits, how much room captions
        are left. They are shared by any engine, which is why they live beside
        codec settings rather than in `clips.json`.
        """
        from makeshorts.render.layout import LayoutOptions  # noqa: PLC0415

        try:
            return LayoutOptions.model_validate(self.render.layout or {})
        except ValidationError as exc:
            raise ConfigInvalid(
                Path(self.sources[0]) if self.sources else Path(RENDER_FILENAME),
                f"invalid `layout:` block: {exc}",
            ) from exc


def _layer_paths(
    filename: str,
    config_dir: str | Path,
    overlay_dir: str | Path | None,
    *,
    required: bool = True,
) -> list[Path]:
    """Repo-level file first, per-job override second.

    A missing override is always fine — that is the normal case. A missing base
    file is fatal for render.yaml, which has no meaningful defaults, and fine
    for styles.yaml, which does.
    """
    layers = []
    base = Path(config_dir) / filename
    if base.exists():
        layers.append(base)
    elif required:
        raise ConfigNotFound(base)
    if overlay_dir is not None:
        override = Path(overlay_dir) / filename
        if override.exists():
            layers.append(override)
    return layers


def _load_layered(
    filename: str,
    model: type[BaseModel],
    config_dir: str | Path,
    overlay_dir: str | Path | None,
) -> tuple[Any, list[Path]]:
    paths = _layer_paths(filename, config_dir, overlay_dir)
    data: dict[str, Any] = {}
    for path in paths:
        data = deep_merge(data, read_yaml(path))
    return _validate(model, data, paths), paths


def load_render_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    overlay_dir: str | Path | None = None,
) -> RenderConfig:
    """`config/render.yaml`, with `<overlay_dir>/render.yaml` merged over it."""
    cfg, _ = _load_layered(RENDER_FILENAME, RenderConfig, config_dir, overlay_dir)
    return cfg


def load_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    *,
    overlay_dir: str | Path | None = None,
) -> Config:
    """Both files at once.

    `overlay_dir` is a job's own `config/` directory. Pass `job.config_dir`;
    it need not exist.
    """
    render, render_paths = _load_layered(RENDER_FILENAME, RenderConfig, config_dir, overlay_dir)
    styles = load_styles(config_dir, overlay_dir=overlay_dir)
    style_paths = _layer_paths(STYLES_FILENAME, config_dir, overlay_dir, required=False)
    return Config(render=render, styles=styles, sources=tuple(render_paths + style_paths))


__all__ = [
    "DEFAULT_CONFIG_DIR",
    "RENDER_FILENAME",
    "STYLES_FILENAME",
    "ConfigError",
    "ConfigNotFound",
    "ConfigInvalid",
    "UnknownStyle",
    "VideoConfig",
    "LoudnormConfig",
    "AudioConfig",
    "OutputConfig",
    "FfmpegConfig",
    "CaptionsConfig",
    "RenderConfig",
    "StylesDoc",
    "Config",
    "deep_merge",
    "read_yaml",
    "load_render_config",
    "load_styles",
    "load_config",
]
