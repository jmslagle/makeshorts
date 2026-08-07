"""`ms` — the command line. This is the interface a person actually lives in.

Six commands, one per stage of the pipeline, in the order you run them:

    ms prepare <input>     mechanical ingest; writes the artifacts and PROMPT.md
    ms plan <slug>         regenerate PROMPT.md from criteria.yaml
    ms lint <slug>         the gate — nothing renders until this passes
    ms render <slug>       encode the clips
    ms caps                what the installed ffmpeg can actually do
    ms jobs                what state every job is in

Two rules shape this module.

**Nothing heavy is imported at module scope.** `prepare/`, `select/lint`,
`select/prompt` and the render engines are imported *inside* the command that
needs them. That keeps `ms --help` instant, keeps a broken optional dependency
from breaking every command, and means this file imports cleanly against a
half-built tree. A missing module produces a sentence, not a traceback.

**Findings are the product.** `ms lint` output is the thing this tool is for:
it is what tells you an edit list is wrong and why. It is grouped by clip,
column-aligned, and uses colour for exactly one thing — severity. Everything
else is plain text, because a terminal full of colour communicates nothing.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Iterable, NoReturn, Sequence

import typer

from makeshorts import config as config_mod
from makeshorts import jobs as jobs_mod

app = typer.Typer(
    name="ms",
    help="Turn long-form webinar recordings into short vertical clips.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# --------------------------------------------------------------------------
# Terminal output
#
# Colour is reserved for severity. Structure is carried by indentation and
# alignment, which survive being piped to a file; colour does not.
# --------------------------------------------------------------------------

ERROR = "error"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}
_SEVERITY_LABEL = {ERROR: "ERROR", WARNING: "WARN", INFO: "INFO"}
_SEVERITY_COLOR = {ERROR: typer.colors.RED, WARNING: typer.colors.YELLOW, INFO: None}

# Anything wider than this wraps. Narrow terminals get narrow output; very wide
# ones do not get 200-character lines, which are unreadable.
_MAX_WIDTH = 100


def _term_width() -> int:
    return min(shutil.get_terminal_size((88, 24)).columns, _MAX_WIDTH)


def _out(line: str = "") -> None:
    typer.echo(line)


def _err(line: str = "") -> None:
    typer.echo(line, err=True)


def _bold(text: str) -> str:
    return typer.style(text, bold=True)


def _dim(text: str) -> str:
    return typer.style(text, dim=True)


def _severity(sev: str, width: int = 5) -> str:
    label = _SEVERITY_LABEL.get(sev, sev.upper())
    color = _SEVERITY_COLOR.get(sev)
    padded = label.ljust(width)
    if color is None:
        return _dim(padded)
    return typer.style(padded, fg=color, bold=(sev == ERROR))


def _wrap(text: str, indent: int) -> str:
    """Wrap to the terminal, hanging-indented so continuation lines line up
    under the first one instead of resetting to the margin.

    `indent` is part of the width, not additional to it -- a line is `indent`
    spaces plus text, totalling at most the terminal width.
    """
    pad = " " * indent
    width = max(_term_width(), indent + 30)
    return "\n".join(
        textwrap.fill(para, width=width, initial_indent=pad, subsequent_indent=pad)
        for para in text.splitlines() or [""]
    )


def _die(message: str, *, hint: str | None = None, code: int = 1) -> NoReturn:
    """Print a failure and stop.

    Every failure the user can act on gets a hint naming the command that fixes
    it. `StageNotRun` from jobs.py already carries one; this is the same idea
    applied to everything else.
    """
    _err(typer.style("error: ", fg=typer.colors.RED, bold=True) + message)
    if hint:
        _err(_dim(_wrap(hint, 2)))
    raise typer.Exit(code)


# --------------------------------------------------------------------------
# Shared options
# --------------------------------------------------------------------------


@dataclass
class Settings:
    """Where things live. Set once by the app callback, read by every command.

    Both are overridable so the CLI can be pointed at a scratch tree — which is
    what the tests do, and what you want when trying a different rubric without
    disturbing a real job.
    """

    jobs_dir: Path = Path(jobs_mod.JOBS_DIR_NAME)
    config_dir: Path = config_mod.DEFAULT_CONFIG_DIR


def _settings(ctx: typer.Context) -> Settings:
    if not isinstance(ctx.obj, Settings):
        ctx.obj = Settings()
    return ctx.obj


@app.callback()
def _main(
    ctx: typer.Context,
    jobs_dir: Annotated[
        Path,
        typer.Option(
            "--jobs-dir",
            envvar="MS_JOBS_DIR",
            metavar="DIR",
            help="Directory holding job folders.",
        ),
    ] = Path(jobs_mod.JOBS_DIR_NAME),
    config_dir: Annotated[
        Path,
        typer.Option(
            "--config-dir",
            envvar="MS_CONFIG_DIR",
            metavar="DIR",
            help="Directory holding render.yaml, criteria.yaml and styles.yaml.",
        ),
    ] = config_mod.DEFAULT_CONFIG_DIR,
) -> None:
    ctx.obj = Settings(jobs_dir=jobs_dir, config_dir=config_dir)


# --------------------------------------------------------------------------
# Lazy imports of the other build tracks
#
# These modules are written by other tracks and may legitimately be absent or
# half-finished. Importing them at module scope would make `ms --help` fail on
# an unrelated bug, so every one of them is resolved here, at call time, with a
# message that says which piece is missing rather than dumping a stack.
# --------------------------------------------------------------------------


def _need_module(dotted: str, *, command: str, provides: str) -> ModuleType:
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as exc:
        missing = exc.name or dotted
        if missing == dotted or dotted.startswith(f"{missing}."):
            _die(
                f"`ms {command}` needs `{dotted}`, which is not present in this checkout.",
                hint=f"That module provides {provides}. Every other command still works.",
            )
        _die(
            f"`ms {command}` could not import `{dotted}`: no module named {missing!r}.",
            hint=f"That looks like a missing dependency rather than missing code. "
            f"Try `uv sync`, or install {missing!r}.",
        )
    except ImportError as exc:
        _die(f"`ms {command}` could not import `{dotted}`: {exc}")


def _need_attr(dotted: str, *names: str, command: str, provides: str) -> Any:
    """The first of `names` the module actually defines.

    Several names are accepted because these entry points are being written in
    parallel with this file; when the tracks converge, the surviving name is
    the one to keep. If none matches, the error lists what the module does
    export, which is the fastest way to find the right name.
    """
    module = _need_module(dotted, command=command, provides=provides)
    for name in names:
        found = getattr(module, name, None)
        if callable(found):
            return found
    public = sorted(n for n in vars(module) if not n.startswith("_"))
    _die(
        f"`{dotted}` defines none of {', '.join(repr(n) for n in names)}.",
        hint=f"It exports: {', '.join(public) or '(nothing)'}. "
        f"`ms {command}` calls one of those names to get {provides}.",
    )


# --------------------------------------------------------------------------
# Job and config resolution
# --------------------------------------------------------------------------


def _open_job(ctx: typer.Context, slug: str) -> jobs_mod.Job:
    try:
        return jobs_mod.Job.open(slug, _settings(ctx).jobs_dir)
    except jobs_mod.JobNotFound as exc:
        _die(str(exc), hint="`ms jobs` lists every job and the stage it has reached.")


def _load_config(ctx: typer.Context, job: jobs_mod.Job | None = None) -> config_mod.Config:
    """render.yaml + styles.yaml, with the job's own `config/` merged over."""
    settings = _settings(ctx)
    overlay = job.config_dir if job is not None else None
    try:
        return config_mod.load_config(settings.config_dir, overlay_dir=overlay)
    except config_mod.ConfigNotFound as exc:
        _die(str(exc), hint="Point at a different tree with `--config-dir`.")
    except config_mod.ConfigError as exc:
        _die(str(exc))


def _criteria_path(ctx: typer.Context, job: jobs_mod.Job | None = None) -> Path:
    """The rubric in force: the job's own copy if it has one, else the repo's.

    A job that pins its rubric keeps producing the same judgments after the
    repo-level file is tuned, which is what makes an old clips.json still
    readable against the criteria that produced it.
    """
    if job is not None:
        override = job.config_dir / "criteria.yaml"
        if override.exists():
            return override
    return Path(_settings(ctx).config_dir) / "criteria.yaml"


def _load_criteria(ctx: typer.Context, job: jobs_mod.Job | None = None):
    """criteria.yaml — the rubric. Read through `select.criteria` so the CLI,
    the prompt and the linter can never disagree about what it says."""
    from makeshorts.select.criteria import load_criteria

    path = _criteria_path(ctx, job)
    try:
        return load_criteria(path)
    except FileNotFoundError as exc:
        _die(str(exc), hint="Point at a different tree with `--config-dir`.")
    except Exception as exc:  # pydantic ValidationError, yaml errors
        _die(f"{path} is not a valid rubric:\n{textwrap.indent(str(exc), '  ')}")


def _load_clips(job: jobs_mod.Job):
    try:
        return job.load_clips()
    except jobs_mod.StageNotRun as exc:
        _die(
            f"{exc.artifact} not found at {exc.path}.",
            hint=f"Run `{exc.remedy}`, then read {job.prompt_md} and write the edit list.",
        )
    except jobs_mod.ArtifactInvalid as exc:
        _die(str(exc), hint="clips.json is hand-editable; the field path above says where to look.")


# --------------------------------------------------------------------------
# Findings — the normalized shape `ms lint` renders
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    message: str
    clip_id: str | None = None
    fixable: bool = False


def _norm_severity(value: Any) -> str:
    raw = str(getattr(value, "value", value) or INFO).strip().lower()
    if raw.startswith("err") or raw in {"fatal", "critical"}:
        return ERROR
    if raw.startswith("warn"):
        return WARNING
    return INFO


def _first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        if value is not None:
            return value
    return default


def _normalize(raw: Any) -> Finding:
    """Accept a finding in whatever shape the linter emits.

    The linter is owned by another module; this insulates the display from its
    exact attribute names so a rename there cannot turn a lint failure into a
    crash — which would be the single worst failure mode for a gate.
    """
    return Finding(
        severity=_norm_severity(_first_attr(raw, "severity", "level", default=ERROR)),
        rule=str(_first_attr(raw, "rule", "rule_id", "id", "code", default="lint")),
        message=str(_first_attr(raw, "message", "msg", "detail", default=str(raw))),
        clip_id=_first_attr(raw, "clip_id", "clip", "target"),
        fixable=bool(_first_attr(raw, "fixable", "auto_fixable", default=False)),
    )


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors


def _collect(raw_report: Any) -> Report:
    findings = _first_attr(raw_report, "findings", "issues", "problems", default=None)
    if findings is None:
        findings = raw_report if isinstance(raw_report, Iterable) else []
    fixed = _first_attr(raw_report, "fixed", "fixes", "changes", default=[]) or []
    return Report(
        findings=[_normalize(f) for f in findings],
        fixed=[str(x) for x in fixed],
    )


# --------------------------------------------------------------------------
# Rendering a report
# --------------------------------------------------------------------------

_DOCUMENT_GROUP = "(whole edit list)"


def _print_report(report: Report, *, job: jobs_mod.Job, clip_ids: Sequence[str]) -> None:
    """Findings grouped by clip, in edit-list order, aligned into columns.

    Clip order follows clips.json rather than severity, so the output reads
    like the file it is criticising — you can go down the list with the edit
    list open beside it. Within a clip, errors come before warnings.
    """
    _out(f"{_bold(str(job.clips_json))}  {_dim(f'{len(clip_ids)} clips')}")

    if not report.findings:
        _out()
        _out("  " + typer.style("clean", fg=typer.colors.GREEN) + " — no findings.")
        return

    by_clip: dict[str, list[Finding]] = {}
    for finding in report.findings:
        by_clip.setdefault(finding.clip_id or _DOCUMENT_GROUP, []).append(finding)

    # Document-level findings first — they are about the file as a whole
    # (duplicate ids, overlapping clips, a stale criteria_ref) and reading them
    # after per-clip detail gets the story backwards.
    order = [_DOCUMENT_GROUP] + list(clip_ids)
    groups = [g for g in order if g in by_clip]
    groups += [g for g in by_clip if g not in order]

    # One column width for the whole report so rules line up across clips.
    rule_width = min(max(len(f.rule) for f in report.findings), 28)

    for group in groups:
        items = sorted(by_clip[group], key=lambda f: (_SEVERITY_ORDER[f.severity], f.rule))
        _out()
        _out("  " + _bold(group))
        for finding in items:
            head = f"    {_severity(finding.severity)}  {finding.rule.ljust(rule_width)}  "
            # Indent is computed from the visible text, not the styled string,
            # so ANSI escapes do not throw the alignment off.
            visible_indent = 4 + 5 + 2 + rule_width + 2
            body = _wrap(finding.message, visible_indent).lstrip()
            _out(head + body)

    _out()
    affected = len([g for g in by_clip if g != _DOCUMENT_GROUP])
    doc_level = len(by_clip.get(_DOCUMENT_GROUP, []))
    _out("  " + _summary_line(report, affected, len(clip_ids), doc_level))


def _summary_line(report: Report, affected: int, total_clips: int, doc_level: int) -> str:
    """`6 errors · 2 warnings — 4 of 8 clips, plus 2 on the edit list itself`.

    The count exists so the size of the problem is visible without counting
    lines, and so one run can be compared against the last at a glance.
    """
    parts = []
    n_err, n_warn = len(report.errors), len(report.warnings)
    n_info = len(report.findings) - n_err - n_warn
    if n_err:
        parts.append(_plural(n_err, "error"))
    if n_warn:
        parts.append(_plural(n_warn, "warning"))
    if n_info:
        parts.append(_plural(n_info, "note"))

    where = f"{affected} of {_plural(total_clips, 'clip')}" if total_clips else "the edit list"
    if doc_level:
        where += f", plus {doc_level} on the edit list itself"
    return _dim(f"{' · '.join(parts)} — {where}")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


@app.command()
def prepare(
    ctx: typer.Context,
    input: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            metavar="INPUT",
            help="The recording to ingest. Copied into the job as source.<ext>. "
            "Omit it and pass --source when one recording exists as several files.",
        ),
    ] = None,
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            metavar="NAME=PATH",
            help="A named input; repeatable. Use this when one recording was "
            "exported as several frame-aligned views (Zoom's active-speaker, "
            "shared-screen and gallery renders of the same meeting). The first "
            "one given is primary. NAME becomes the region-id prefix and the key "
            "in clips.json `sources`, so it must be lowercase letters, digits and "
            "underscores, starting with a letter.",
        ),
    ] = None,
    audio_from: Annotated[
        str | None,
        typer.Option(
            "--audio-from",
            metavar="NAME",
            help="Which --source to transcribe. Frame-aligned views share one "
            "audio track, so exactly one is transcribed; defaults to the first.",
        ),
    ] = None,
    slug: Annotated[
        str | None,
        typer.Option(
            "--slug",
            "-s",
            help="Name the job. Defaults to a slug derived from the filename, "
            "suffixed if that name is taken.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            metavar="NAME",
            help="Whisper model for transcription. Larger is slower and more "
            "accurate; word-level timestamps are required either way.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Reuse an existing job directory and redo the mechanical stage, "
            "overwriting its artifacts. clips.json is not touched.",
        ),
    ] = False,
    skip: Annotated[
        list[str] | None,
        typer.Option(
            "--skip",
            metavar="STAGE",
            help="Skip a mechanical stage; repeatable. One of: "
            "probe, silence, transcribe, regions, text. Useful to re-run "
            "region detection without paying for transcription again.",
        ),
    ] = None,
) -> None:
    """Mechanical ingest: probe, transcribe, detect silence and regions, write PROMPT.md.

    Deterministic and re-runnable. No AI, no judgment, nothing here depends on
    the rubric. Everything it writes can be deleted and regenerated.

    One recording, one file:

        ms prepare webinar.mp4

    One recording exported as several frame-aligned views — the case Zoom
    cloud recordings produce, where the usable face and the sharp slides are
    in different files:

        ms prepare --source cam=GMT..._avo.mp4 --source slides=GMT..._as.mp4

    Every source is probed and region-detected; the audio is transcribed once,
    from the first source or from `--audio-from`.
    """
    settings = _settings(ctx)
    jobs_root = settings.jobs_dir

    named_sources = _parse_sources(source)
    if input is not None and named_sources:
        _die(
            "give either a positional INPUT or --source NAME=PATH, not both.",
            hint="A single file is `ms prepare recording.mp4`. Several views of one "
            "recording are `ms prepare --source cam=a.mp4 --source slides=b.mp4`; "
            "the first --source is the primary one.",
        )
    if input is None and not named_sources:
        _die(
            "nothing to ingest.",
            hint="Pass a recording (`ms prepare recording.mp4`), or name several "
            "frame-aligned views of one recording with repeated "
            "`--source NAME=PATH`.",
        )
    if audio_from and not named_sources:
        _die(
            "--audio-from names one of the --source inputs, and this run has none.",
            hint="It exists because frame-aligned views of one meeting share an audio "
            "track, so only one of them is transcribed. With a single input there "
            "is nothing to choose.",
        )
    if audio_from and audio_from not in named_sources:
        _die(
            f"--audio-from {audio_from!r} is not one of the sources given.",
            hint=f"Sources: {', '.join(named_sources)}.",
        )

    slug_from = input if input is not None else next(iter(named_sources.values()))
    if slug:
        chosen = slug
    elif force:
        # --force means "redo this job", so it must land on the job the input
        # already maps to. Uniquifying first would append a -2 suffix and
        # create a second job directory, which is the opposite of reuse.
        chosen = jobs_mod.slug_for_input(slug_from)
    else:
        chosen = jobs_mod.unique_slug(jobs_mod.slug_for_input(slug_from), jobs_root)
    if not jobs_mod.slugify(chosen):
        _die(f"--slug {chosen!r} contains no usable characters.")

    try:
        job = jobs_mod.Job.create(chosen, jobs_root, exist_ok=force)
    except jobs_mod.JobError as exc:
        _die(str(exc))

    run_prepare = _need_attr(
        "makeshorts.prepare.run",
        "prepare",
        command="prepare",
        provides="the mechanical ingest stage",
    )

    def run_stages() -> tuple[str, ...]:
        import importlib

        return tuple(importlib.import_module("makeshorts.prepare.run").STAGES)
    _out(f"{_bold(job.slug)}  {_dim(str(job.root))}")

    kwargs: dict[str, Any] = {"force": force, "log": _log_stage}
    if named_sources:
        ingested = {name: _ingest(path, job, name=name) for name, path in named_sources.items()}
        kwargs["sources"] = ingested
        if audio_from:
            kwargs["audio_from"] = audio_from
        # Named rather than left implicit: this is the one stage that does not
        # run per source, and a source named `audio` must not be confused with
        # the fact that the audio came from it.
        chosen = audio_from or next(iter(ingested))
        _out(f"  transcript  {_dim(f'from source {chosen!r}')}")
    else:
        assert input is not None  # guarded above
        kwargs["source"] = _ingest(input, job)
    if model:
        kwargs["model"] = model
        _out(f"  model    {model}")
    _out()

    if skip:
        # Validate here rather than letting a typo silently run the full stage:
        # "--skip transcibe" costing a 40-minute transcription is exactly the
        # mistake this flag exists to avoid.
        unknown = [s for s in skip if s not in run_stages()]
        if unknown:
            _die(
                f"unknown stage(s) to skip: {', '.join(sorted(unknown))}. "
                f"Valid stages: {', '.join(run_stages())}."
            )
        kwargs["skip"] = skip
    try:
        result = run_prepare(job, **kwargs)
    except Exception as exc:  # noqa: BLE001 — the stage owns its own messages
        _die(f"prepare failed: {exc}")

    # The stage log has already said what ran. What is left is what did *not*
    # go to plan -- a skipped stage or a degraded result is exactly the thing a
    # scrolling wall of "wrote ..." lines would bury.
    for stage in getattr(result, "skipped", ()):
        _out(f"  {_severity(INFO)}  skipped {stage}")
    for warning in getattr(result, "warnings", ()):
        _out(f"  {_severity(WARNING)}  {warning}")

    generated = _write_prompt(ctx, job, quiet=True)

    _out()
    _print_artifacts(job)
    elapsed = getattr(result, "elapsed", 0.0)
    if elapsed:
        _out(_dim(f"  {elapsed:.1f}s"))

    _out()
    if generated:
        _out(f"Next: read {_bold(str(job.prompt_md))}, then write {job.clips_json}.")
    else:
        _out(f"Next: `ms plan {job.slug}` to generate the editorial brief.")


def _log_stage(message: str) -> None:
    _out(_dim(f"  {message}"))


def _parse_sources(specs: list[str] | None) -> dict[str, Path]:
    """`--source NAME=PATH` into an ordered name -> path mapping.

    Order is preserved because it is meaningful: the first source is the
    primary one, which is what `frame` and any region that names no source
    refer to.
    """
    parsed: dict[str, Path] = {}
    for spec in specs or []:
        name, sep, raw = spec.partition("=")
        name = name.strip()
        raw = raw.strip()
        if not sep or not name or not raw:
            _die(
                f"--source {spec!r} is not NAME=PATH.",
                hint="For example: --source cam=videos/meeting_avo.mp4 "
                "--source slides=videos/meeting_as.mp4",
            )
        if not jobs_mod.SOURCE_NAME_RE.match(name):
            _die(
                f"--source name {name!r} is not usable.",
                hint=f"Names must match {jobs_mod.SOURCE_NAME_RE.pattern} — lowercase, "
                "starting with a letter. The name becomes a region-id prefix "
                "(`cam` gives `cam__cam_a`) and a key in clips.json `sources`, both "
                "of which are held to that shape.",
            )
        if name in parsed:
            _die(
                f"--source {name!r} was given twice.",
                hint=f"Already pointing at {parsed[name]}. Each name identifies one file.",
            )
        path = Path(raw)
        if not path.is_file():
            _die(
                f"--source {name}={raw}: no such file.",
                hint="Every source must exist before ingest; they are hardlinked "
                "into the job directory, not fetched.",
            )
        parsed[name] = path
    return parsed


def _ingest(input_path: Path, job: jobs_mod.Job, *, name: str | None = None) -> Path:
    """Put the recording inside the job as `source.<ext>`, or `source-<name>.<ext>`.

    Hardlinked when the filesystem allows it, because a webinar recording is
    several gigabytes and copying one to start a job is a bad first impression.
    A hardlink is safe here: nothing in this pipeline ever writes to the source.
    """
    dest = job.source_for(input_path, name)
    label = "source  " if name is None else f"{name}".ljust(8)
    src = input_path.resolve()
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        _out(f"  {label} {dest} {_dim('(already ingested)')}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    how = "linked"
    try:
        dest.hardlink_to(src)
    except OSError:
        how = "copied"
        try:
            shutil.copy2(src, dest)
        except OSError as exc:
            _die(f"could not ingest {input_path}: {exc}")
    _out(f"  {label} {dest} {_dim(f'({how} from {input_path})')}")
    return dest


def _print_artifacts(job: jobs_mod.Job) -> None:
    """What landed on disk. The point of the checkpoint is that you inspect
    these before anything editorial happens, so the paths go on screen."""
    artifacts = [
        ("media.json", job.media_json),
        ("words.json", job.words_json),
        ("silence.json", job.silence_json),
        ("regions.json", job.regions_json),
        ("transcript.txt", job.transcript_txt),
        ("raw.srt", job.raw_srt),
        ("PROMPT.md", job.prompt_md),
    ]
    sources = job.sources
    if len(sources) > 1:
        # Which file is which matters more here than anywhere else: every
        # region id in regions.json is namespaced by these names.
        artifacts = [(f"source:{name}", path) for name, path in sources.items()] + artifacts
    width = max(len(name) for name, _ in artifacts)
    for name, path in artifacts:
        if path.exists():
            size = _human_size(path.stat().st_size)
            _out(f"  {name.ljust(width)}  {_dim(size.rjust(8))}  {path}")
        else:
            _out(f"  {name.ljust(width)}  {_dim('  missing')}")


def _human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}G"


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@app.command()
def plan(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Job slug. `ms jobs` lists them.")],
) -> None:
    """Regenerate PROMPT.md from criteria.yaml.

    The rubric is data: editing config/criteria.yaml and re-running this
    changes the brief the model is given, the score keys `ms lint` demands, and
    the thresholds it enforces — all three together, because they are all read
    from the same file.
    """
    job = _open_job(ctx, slug)
    if not _write_prompt(ctx, job):
        raise typer.Exit(1)


def _write_prompt(ctx: typer.Context, job: jobs_mod.Job, *, quiet: bool = False) -> bool:
    """Generate PROMPT.md and report where it landed.

    `quiet` is for the tail of `ms prepare`, which has already said plenty; a
    prepared job with no brief is a dead end, so it is generated there too
    rather than making `ms plan` a step you can forget.
    """
    criteria = _load_criteria(ctx, job)
    path_of_criteria = _criteria_path(ctx, job)

    write_prompt = _need_attr(
        "makeshorts.select.prompt",
        "write_prompt",
        "generate",
        command="plan",
        provides="PROMPT.md generation from the rubric",
    )

    try:
        result = write_prompt(job, criteria=criteria, criteria_path=path_of_criteria)
    except jobs_mod.StageNotRun as exc:
        if quiet:
            _err(_dim(f"  PROMPT.md not written: {exc}"))
            return False
        _die(str(exc), hint=f"Run `{exc.remedy}` first.")
    except Exception as exc:  # noqa: BLE001
        if quiet:
            _err(_dim(f"  PROMPT.md not written: {exc}"))
            return False
        _die(f"could not generate PROMPT.md: {exc}")

    if quiet:
        # `ms prepare` lists PROMPT.md in its artifact table; saying it twice
        # in one screen of output is noise.
        return True

    path = Path(result) if isinstance(result, (str, Path)) else job.prompt_md
    n = len(criteria.rubric)
    _out(f"{_bold(str(path))}")
    _out(
        _dim(
            f"  rubric {criteria.version} · "
            f"{n} criteri{'on' if n == 1 else 'a'} · "
            f"max {criteria.gates.max_clips} clips · "
            f"{criteria.gates.duration.min:g}-{criteria.gates.duration.max:g}s each"
        )
    )
    _out()
    _out("Read this file, then write clips.json.")
    _out(_dim(f"When it is written: ms lint {job.slug}"))
    return True


# --------------------------------------------------------------------------
# lint
# --------------------------------------------------------------------------


@app.command()
def lint(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Job slug.")],
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Snap start/end to word and sentence boundaries and apply "
            "pad_in/pad_out, rewriting clips.json in place. Only mechanical "
            "fixes; nothing editorial is touched.",
        ),
    ] = False,
) -> None:
    """Check clips.json against words.json, silence.json and the rubric.

    This is the gate. It exits nonzero on any ERROR, and `ms render` refuses to
    run a job that fails it — which is what makes a bad edit list fail loudly
    instead of quietly rendering something wrong.
    """
    job = _open_job(ctx, slug)
    criteria = _load_criteria(ctx, job)
    doc = _load_clips(job)

    run_lint = _need_attr(
        "makeshorts.select.lint",
        "lint_job",
        "lint",
        "run",
        command="lint",
        provides="the mechanical checks against words.json and the rubric",
    )

    try:
        raw = run_lint(job, criteria, fix=fix)
    except jobs_mod.StageNotRun as exc:
        _die(str(exc), hint=f"Run `{exc.remedy}` first.")
    except TypeError as exc:
        _die(
            f"the linter rejected the arguments the CLI passed: {exc}",
            hint="`ms lint` calls lint_job(job, criteria, fix=...). "
            "If the linter's signature changed, cli.py needs updating to match.",
        )
    except Exception as exc:  # noqa: BLE001
        _die(f"lint failed to run: {exc}")

    report = _collect(raw)
    _print_report(report, job=job, clip_ids=[c.id for c in doc.clips])

    if fix and report.fixed:
        _out()
        _out("  " + _bold("fixed"))
        for line in report.fixed:
            _out(f"    {line}")

    _out()
    if report.ok:
        _out(typer.style("lint passed.", fg=typer.colors.GREEN) + f" `ms render {job.slug}`")
        return

    fixable = [f for f in report.errors if f.fixable]
    _err(typer.style("lint failed.", fg=typer.colors.RED, bold=True) + " Nothing will render.")
    if fixable and not fix:
        _err(_dim(f"  {_plural(len(fixable), 'error')} can be fixed by `ms lint {job.slug} --fix`."))
    raise typer.Exit(1)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


@app.command()
def render(
    ctx: typer.Context,
    slug: Annotated[str, typer.Argument(help="Job slug.")],
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            metavar="IDS",
            help="Comma-separated clip ids or their numeric prefixes, e.g. "
            "`--only 01,03`. Default: every clip in the edit list.",
        ),
    ] = None,
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            metavar="NAME",
            help="Render backend. ffmpeg is the only one today; the edit list "
            "is engine-agnostic, so a second one changes nothing upstream.",
        ),
    ] = "ffmpeg",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite clips that already exist in out/."),
    ] = False,
) -> None:
    """Encode each clip to out/<slug>--<clip.id>.mp4, with a receipt beside it.

    Refuses to start if `ms lint` reports an ERROR. Also refuses before
    encoding anything if the edit list asks for a layout or caption feature the
    selected engine cannot deliver — failing after twenty minutes of the batch
    is worse than failing immediately.
    """
    job = _open_job(ctx, slug)
    cfg = _load_config(ctx, job)
    criteria = _load_criteria(ctx, job)
    doc = _load_clips(job)

    _gate_on_lint(job, criteria)

    clips = _select_clips(doc, only)
    selected_engine = _resolve_engine(engine, doc)
    _apply_render_config(selected_engine, cfg, doc)

    job.ensure_dirs()
    overwrite = force or cfg.render.output.overwrite

    _out(f"{_bold(job.slug)}  {_dim(f'{len(clips)} of {len(doc.clips)} clips')}")
    _out(
        _dim(
            f"  engine {selected_engine.name} {_engine_version(selected_engine)} · "
            f"{doc.output.width}x{doc.output.height}@{doc.output.fps} · "
            f"{cfg.render.video.codec} crf {cfg.render.video.crf}"
        )
    )
    _out()

    failures = 0
    for index, clip in enumerate(clips, start=1):
        out_path = job.output_path(clip.id)
        prefix = f"  [{index}/{len(clips)}] {clip.id}"
        if out_path.exists() and not overwrite:
            _out(f"{prefix}  {_dim('skipped — already rendered; --force to redo')}")
            continue
        try:
            # Branding first so captions composite over it if they ever meet.
            overlays = _branding_overlays(job, clip, doc, cfg) + _caption_overlays(
                job, clip, doc, cfg
            )
            receipt = selected_engine.render(
                doc, clip, job.source, out_path, caption_overlays=overlays
            )
        except Exception as exc:  # noqa: BLE001 — one bad clip must not kill the batch
            failures += 1
            _out(f"{prefix}  {_severity(ERROR, 0)}")
            _out(_wrap(str(exc), 6))
            continue
        _write_receipt(job, clip, receipt)
        caption_note = f" · {len(overlays)} caption states" if overlays else ""
        _out(f"{prefix}  {_dim(f'{clip.duration:.1f}s{caption_note}')}  {out_path}")

    _out()
    if failures:
        _err(
            typer.style(f"{_plural(failures, 'clip')} failed.", fg=typer.colors.RED, bold=True)
            + f" {len(clips) - failures} written to {job.out_dir}/"
        )
        raise typer.Exit(1)
    _out(
        typer.style("done.", fg=typer.colors.GREEN)
        + f" {_plural(len(clips), 'clip')} in {job.out_dir}/"
    )


def _caption_overlays(
    job: jobs_mod.Job, clip: Any, doc: Any, cfg: Any
) -> list[Any]:
    """Rasterize this clip's captions, or return [] if it wants none.

    Cue building and rasterization live in `render/captions/`; this only picks
    the style the edit list named and hands over the words. A clip with
    captions disabled, or a job with no words.json, renders silently without
    them rather than failing -- captions are a presentation layer, and losing
    them should never cost you the encode.
    """
    if not getattr(clip.captions, "enabled", True):
        return []
    try:
        words = job.load_words()
    except Exception:  # noqa: BLE001 — no transcript is a legitimate state
        return []

    cues_mod = importlib.import_module("makeshorts.render.captions.cues")
    pillow_mod = importlib.import_module("makeshorts.render.captions.pillow_backend")

    # clips.json names a style; config/styles.yaml defines what it looks like.
    # That indirection is what keeps typography out of the edit list.
    style = cfg.style_for(clip.captions.style)
    cues = cues_mod.cues_for_clip(words, clip.start, clip.end, style)
    if not cues:
        return []

    backend = pillow_mod.PillowCaptionBackend()
    assets = backend.render(
        cues,
        style,
        (doc.output.width, doc.output.height),
        job.caption_cache_dir(clip.id),
        position=clip.captions.position,
    )
    return list(assets.overlays)


def _branding_overlays(job: jobs_mod.Job, clip: Any, doc: Any, cfg: Any) -> list[Any]:
    """The watermark, as a single overlay spanning the whole clip.

    Reuses the caption overlay contract rather than adding a second
    compositing path in the engine: a watermark is just an image at a fixed
    spot for a fixed window, which is exactly what a caption cue already is.

    Resolution order is clip -> config default, with the literal name "none"
    suppressing it for one clip.
    """
    name = getattr(clip, "branding", None) or getattr(cfg.render, "default_branding", None)
    if not name or name == "none":
        return []
    preset = getattr(cfg.render, "branding", {}).get(name)
    if preset is None or not preset.enabled or not preset.image:
        return []

    src = Path(preset.image)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.exists():
        _out(f"  {_severity(WARNING)}  branding image not found: {src} — rendering without it")
        return []

    from PIL import Image  # noqa: PLC0415 — optional at import time

    from makeshorts.render.captions.base import CaptionOverlay  # noqa: PLC0415

    out_w, out_h = doc.output.width, doc.output.height
    logo = Image.open(src).convert("RGBA")
    logo = logo.crop(logo.getbbox() or (0, 0, logo.width, logo.height))
    target_w = max(2, int(out_w * preset.scale))
    logo = logo.resize((target_w, max(2, round(logo.height * target_w / logo.width))), Image.LANCZOS)
    if preset.opacity < 1.0:
        alpha = logo.getchannel("A").point(lambda v: int(v * preset.opacity))
        logo.putalpha(alpha)

    margin = int(out_w * preset.margin)
    x = margin if preset.corner.endswith("left") else out_w - logo.width - margin
    y = margin if preset.corner.startswith("top") else out_h - logo.height - margin

    cache = job.cache_dir / "branding"
    cache.mkdir(parents=True, exist_ok=True)
    png = cache / f"{name}-{target_w}-{int(preset.opacity * 100)}.png"
    if not png.exists():
        logo.save(png, "PNG")

    return [
        CaptionOverlay(
            png_path=str(png), x=x, y=y,
            width=logo.width, height=logo.height,
            start=0.0, end=float(clip.duration),
        )
    ]


def _write_receipt(job: jobs_mod.Job, clip: Any, receipt: Any) -> None:
    """Persist what produced this file, beside it.

    A clip on disk should always be traceable back to the edit list and the
    toolchain that made it -- otherwise "why does this one look different"
    has no answer six weeks later.
    """
    if receipt is None:
        return
    path = job.receipt_path(clip.id)
    try:
        payload = (
            receipt.model_dump(mode="json")
            if hasattr(receipt, "model_dump")
            else dataclasses.asdict(receipt)
        )
        # The engine is handed a path, not the job, so it cannot know the
        # source hash. Fill it in here: without it the receipt cannot prove
        # which recording the clip actually came from.
        if not payload.get("source_sha256"):
            try:
                payload["source_sha256"] = job.load_media().sha256
            except Exception:  # noqa: BLE001 — media.json is not required to render
                pass
        path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:  # noqa: BLE001 — a missing receipt must not fail the render
        pass


def _gate_on_lint(job: jobs_mod.Job, criteria: Any) -> None:
    """`ms render` will not run an edit list that fails the gate.

    Deliberately re-runs the linter rather than trusting a previous `ms lint`:
    clips.json is hand-editable, so a pass recorded five minutes ago says
    nothing about the file on disk now.
    """
    run_lint = _need_attr(
        "makeshorts.select.lint",
        "lint_job",
        "lint",
        "run",
        command="render",
        provides="the gate that render refuses to bypass",
    )
    try:
        report = _collect(run_lint(job, criteria, fix=False))
    except Exception as exc:  # noqa: BLE001
        _die(f"could not verify the edit list before rendering: {exc}")

    if report.ok:
        return
    _err(
        typer.style("refusing to render: ", fg=typer.colors.RED, bold=True)
        + f"{_plural(len(report.errors), 'lint error')} in {job.clips_json}."
    )
    for finding in report.errors[:5]:
        where = f"{finding.clip_id}: " if finding.clip_id else ""
        _err(f"  {_severity(ERROR)}  {where}{finding.message}")
    if len(report.errors) > 5:
        _err(_dim(f"  … {len(report.errors) - 5} more"))
    _err(_dim(f"  Full report: ms lint {job.slug}"))
    raise typer.Exit(1)


def _select_clips(doc: Any, only: str | None) -> list[Any]:
    """Resolve `--only 01,03` against the edit list.

    A token matches a clip id exactly, or its numeric prefix — `01` finds
    `01-cac-payback-math`, because nobody wants to type the whole slug.
    """
    if not only:
        if not doc.clips:
            _die("the edit list contains no clips.")
        return list(doc.clips)

    by_id = {clip.id: clip for clip in doc.clips}
    picked: list[Any] = []
    unknown: list[str] = []
    for token in (t.strip() for t in only.split(",")):
        if not token:
            continue
        if token in by_id:
            match = [by_id[token]]
        else:
            match = [c for c in doc.clips if c.id.split("-", 1)[0] == token]
        if not match:
            unknown.append(token)
            continue
        for clip in match:
            if clip not in picked:
                picked.append(clip)

    if unknown:
        _die(
            f"--only matched nothing for: {', '.join(unknown)}",
            hint="Clips in this edit list: " + ", ".join(by_id) or "(none)",
        )
    if not picked:
        _die("--only selected no clips.")
    # Keep edit-list order regardless of the order they were named.
    return [c for c in doc.clips if c in picked]


def _resolve_engine(name: str, doc: Any) -> Any:
    """Look up the engine and check it against what the edit list demands."""
    engine_mod = _need_module(
        "makeshorts.render.engine", command="render", provides="the render engine registry"
    )
    # Engines register themselves on import; the registry is empty until at
    # least one backend module has been loaded.
    _load_engine_backends()

    try:
        engine = engine_mod.get_engine(name)
    except KeyError:
        available = engine_mod.available_engines()
        _die(
            f"unknown render engine {name!r}.",
            hint=f"Available: {', '.join(available) or '(none registered)'}.",
        )

    needed = set(engine_mod.required_capabilities(doc))
    missing = needed - set(engine.capabilities())
    if missing:
        _die(
            f"engine {name!r} cannot render this edit list.",
            hint=f"Missing: {', '.join(sorted(missing))}. "
            f"Either change the layouts in clips.json or use another engine.",
        )
    return engine


def _apply_render_config(engine: Any, cfg: config_mod.Config, doc: Any) -> None:
    """Hand render.yaml to the engine.

    INTEGRATION SEAM. `RenderEngine` in engine.py has no `configure` method, so
    engines come out of the registry holding their own defaults and render.yaml
    would otherwise be read and ignored — the worst kind of bug, because the
    output looks fine and is not what the config asked for.

    Until the protocol grows a hook, this adapts generically: a `configure`
    method is used if one exists, otherwise a dataclass of settings on the
    instance is rebuilt with whichever of these names it declares. Nothing here
    is ffmpeg-specific and no backend module is imported.
    """
    # Geometry policy, if the engine accepts any. Set before the `configure`
    # branch so it applies to both kinds of engine — an engine that grows a
    # configure() hook should not silently lose its layout settings.
    if hasattr(engine, "layout_options"):
        try:
            engine.layout_options = cfg.layout_options()
        except config_mod.ConfigError as exc:
            _die(str(exc))

    configure = getattr(engine, "configure", None)
    if callable(configure):
        configure(cfg.render)
        return

    settings = getattr(engine, "settings", None)
    if settings is None or not dataclasses.is_dataclass(settings):
        return

    video, audio = cfg.render.video, cfg.render.audio
    # fps comes from the edit list, which render.yaml's own comment says wins.
    wanted = {
        "video_codec": video.codec,
        "codec": video.codec,
        "crf": video.crf,
        "preset": video.preset,
        "pix_fmt": video.pix_fmt,
        "bitrate": video.bitrate,
        "audio_codec": audio.codec,
        "audio_bitrate": audio.bitrate,
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "loudnorm_i": audio.normalize.integrated,
        "loudnorm_tp": audio.normalize.true_peak,
        "loudnorm_lra": audio.normalize.range,
        "fps": doc.output.fps,
        "threads": cfg.render.ffmpeg.threads,
    }
    declared = {f.name for f in dataclasses.fields(settings)}
    applied = {k: v for k, v in wanted.items() if k in declared}
    if applied:
        engine.settings = dataclasses.replace(settings, **applied)


def _load_engine_backends() -> None:
    """Import backend modules for their registration side effect.

    Absence is not an error here — `ms caps` and `ms render` report an empty
    registry themselves, which is a clearer message than an import traceback.
    """
    for dotted in ("makeshorts.render.ffmpeg_engine",):
        try:
            importlib.import_module(dotted)
        except ImportError:
            continue


def _engine_version(engine: Any) -> str:
    try:
        return str(engine.version())
    except Exception:  # noqa: BLE001
        return "?"


# --------------------------------------------------------------------------
# caps
# --------------------------------------------------------------------------


@app.command()
def caps(ctx: typer.Context) -> None:
    """What the installed ffmpeg can actually do.

    Worth running first on a new machine. The homebrew ffmpeg is commonly built
    without libass and libfreetype, which removes the `subtitles` and
    `drawtext` filters entirely — burned-in captions then have to come from the
    Pillow backend, and this is where you find that out.
    """
    cfg = _load_config(ctx)

    probe_caps = _need_attr(
        "makeshorts.render.caps",
        "probe_caps",
        "probe",
        command="caps",
        provides="probing the installed ffmpeg",
    )
    try:
        probed = probe_caps(cfg.render.ffmpeg.binary)
    except Exception as exc:  # noqa: BLE001
        _die(f"could not probe ffmpeg: {exc}", hint=f"Configured binary: {cfg.render.ffmpeg.binary}")

    _out(_bold("ffmpeg"))
    for label, value in _caps_rows(probed):
        _out(f"  {label.ljust(22)}  {_render_value(value)}")

    _out()
    _out(_bold("engines"))
    _load_engine_backends()
    engine_mod = _need_module(
        "makeshorts.render.engine", command="caps", provides="the render engine registry"
    )
    names = engine_mod.available_engines()
    if not names:
        _out(_dim("  none registered"))
    for name in names:
        engine = engine_mod.get_engine(name)
        _out(f"  {name.ljust(22)}  {_engine_version(engine)}")
        _out(_dim(f"    {', '.join(sorted(engine.capabilities())) or 'no capabilities'}"))

    _out()
    _out(_bold("configured"))
    _out(f"  {'captions.backend'.ljust(22)}  {cfg.render.captions.backend}")
    _out(f"  {'captions.style'.ljust(22)}  {cfg.render.captions.default_style}")
    _out(f"  {'video.codec'.ljust(22)}  {cfg.render.video.codec}")
    _out(f"  {'styles defined'.ljust(22)}  {', '.join(cfg.styles.names)}")


# The observed facts worth showing, then the conclusions drawn from them. The
# conclusions are what anyone is actually here for -- `caption_backend` in
# particular, because a build without libass silently changes how captions get
# made and this is where you find out.
_CAPS_FIELDS = ("ffmpeg_path", "ffmpeg_version")
_CAPS_DERIVED = (
    "has_libass",
    "has_libfreetype",
    "has_overlay",
    "has_videotoolbox",
    "has_libx264",
    "blur_filter",
    "caption_backend",
)


def _caps_rows(probed: Any) -> list[tuple[str, Any]]:
    """Label/value pairs from the probe result.

    Named fields first so the important ones are not buried, then anything the
    probe grew that this list has not caught up with -- the probe's shape
    belongs to the render track, and the CLI should not hide a new field just
    because it predates this list.
    """
    rows: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for name in _CAPS_FIELDS + _CAPS_DERIVED:
        if hasattr(probed, name):
            rows.append((name, getattr(probed, name)))
            seen.add(name)

    dumped = probed.model_dump() if hasattr(probed, "model_dump") else {}
    for key, value in dumped.items():
        if key in seen or str(key).startswith("_"):
            continue
        # Present the filter/encoder probe results as "what is available",
        # since the absent ones are the default assumption anyway.
        if isinstance(value, dict) and value and all(isinstance(v, bool) for v in value.values()):
            rows.append((str(key), sorted(k for k, v in value.items() if v)))
        elif key in ("buildconf", "binary_size", "binary_mtime", "caps_schema_version"):
            continue
        else:
            rows.append((str(key), value))
    return rows


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return typer.style("yes", fg=typer.colors.GREEN) if value else _dim("no")
    if isinstance(value, (list, tuple, set)):
        items = sorted(str(v) for v in value)
        shown = ", ".join(items[:12])
        return shown + (_dim(f" … +{len(items) - 12}") if len(items) > 12 else "")
    return str(value)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

# Pipeline order. `Job.stage` reports the furthest one reached.
_STAGE_NEXT = {
    "empty": "ms prepare <input> --slug {slug}",
    "source": "ms prepare <input> --slug {slug} --force",
    "prepared": "ms plan {slug}",
    "prompt": "read PROMPT.md, write clips.json",
    "clips": "ms lint {slug}",
    "rendered": "",
}


@app.command()
def jobs(ctx: typer.Context) -> None:
    """List every job and the stage it has reached.

    A job is a directory and its state is whichever artifacts exist, so this is
    `ls` with the pipeline order applied — there is no database to disagree
    with the filesystem.
    """
    settings = _settings(ctx)
    all_jobs = list(jobs_mod.iter_jobs(settings.jobs_dir))
    if not all_jobs:
        _out(_dim(f"no jobs under {settings.jobs_dir}/"))
        _out("Start one with `ms prepare <recording>`.")
        return

    rows = []
    for job in all_jobs:
        stage = job.stage
        rendered = job.rendered_clip_ids()
        rows.append(
            (
                job.slug,
                stage,
                _clip_count(job),
                str(len(rendered)) if rendered else _dim("–"),
                _STAGE_NEXT.get(stage, "").format(slug=job.slug),
            )
        )

    headers = ("job", "stage", "clips", "rendered", "next")
    widths = [
        max(len(headers[i]), max(_visible_len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    _out(_dim("  ".join(h.ljust(w) for h, w in zip(headers, widths))))
    for row in rows:
        cells = [_ljust_visible(row[i], widths[i]) for i in range(len(headers))]
        _out("  ".join(cells).rstrip())


def _clip_count(job: jobs_mod.Job) -> str:
    """How many clips the edit list declares, without validating it.

    A clips.json that does not parse is a real state a job can be in, and
    `ms jobs` should say so rather than refusing to list anything.
    """
    if not job.clips_json.exists():
        return _dim("–")
    try:
        import json

        data = json.loads(job.clips_json.read_text())
        return str(len(data.get("clips", [])))
    except Exception:  # noqa: BLE001
        return typer.style("bad", fg=typer.colors.RED)


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escapes, so styled cells still align."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\x1b":
            j = text.find("m", i)
            i = len(text) if j == -1 else j + 1
            continue
        out += 1
        i += 1
    return out


def _ljust_visible(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_len(text))


if __name__ == "__main__":  # pragma: no cover
    app()
