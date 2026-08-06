"""The mechanical stage, end to end.

`ms prepare` is this function. It fills a job directory with everything the
editorial step needs and nothing it has to think about. Every stage is
skippable and every stage is idempotent: the artifacts are the state, so
deleting `words.json` and re-running re-transcribes and nothing else.

Stage order is a dependency order, not a preference. `media.json` first because
its duration closes an unterminated trailing silence and its `has_audio` decides
whether the audio stages run at all; `transcript.txt` and `raw.srt` last because
they are views of `words.json`.

A job may have more than one source. Zoom exports one meeting as several
frame-aligned views -- active-speaker camera, shared screen, gallery -- and the
usable face and the sharp slides are in different files. Those views share a
clock and share an audio track, which decides how this module treats them:
every source is probed and every source is region-detected, but the audio
stages run exactly once, against one named source. Transcribing four views of
the same meeting would cost four times as long to produce the same words.json.
"""

from __future__ import annotations

import importlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from makeshorts.artifacts import MediaDoc, Region, RegionsDoc, SilenceDoc, WordsDoc
from makeshorts.jobs import PRIMARY_SOURCE_NAME, SOURCE_NAME_RE, Job
from makeshorts.prepare import silence as silence_mod
from makeshorts.prepare import text_outputs, transcribe as transcribe_mod
from makeshorts.prepare.probe import probe

# Named so `--skip regions` reads the way a user would say it.
STAGES = ("probe", "silence", "transcribe", "regions", "text")

Logger = Callable[[str], None]


def _noop(message: str) -> None:  # pragma: no cover - trivial
    pass


@dataclass
class PrepareResult:
    """What a run produced, for the CLI to report and for tests to assert on."""

    job: Job
    written: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    media: MediaDoc | None = None
    words: WordsDoc | None = None
    silence: SilenceDoc | None = None
    elapsed: float = 0.0
    #: Every input this run worked over, by name, primary first. One entry for
    #: a single-source job.
    sources: dict[str, Path] = field(default_factory=dict)
    #: Per-source ffprobe facts. `media` is the primary source's entry.
    media_by_source: dict[str, MediaDoc] = field(default_factory=dict)
    #: The source words.json was transcribed from. The others were never
    #: decoded for audio at all.
    audio_source: str = PRIMARY_SOURCE_NAME

    def wrote(self, path: Path) -> Path:
        self.written.append(path)
        return path

    @property
    def multi_source(self) -> bool:
        return len(self.sources) > 1


def write_json(path: Path, doc: object) -> Path:
    """Pretty-printed, newline-terminated, `None` fields omitted.

    Pretty because these files exist to be read and diffed by hand -- a
    one-line words.json would defeat the point of the whole mechanical stage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = doc.model_dump(mode="json", exclude_none=True)  # type: ignore[attr-defined]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _resolve_sources(
    job: Job,
    source: str | Path | None,
    sources: Mapping[str, str | Path] | None,
) -> tuple[dict[str, Path], bool]:
    """`(name -> path, named)` for this run.

    `named` is the shape decision, not a count: it says whether the caller
    chose names for these files. It stays False for the single positional
    input, which is what keeps a one-file job's media.json byte-for-byte the
    document it has always been.
    """
    if source is not None and sources is not None:
        raise ValueError("pass `source` (one file) or `sources` (named files), not both")

    if sources is not None:
        if not sources:
            raise ValueError("`sources` is empty -- give at least one NAME=PATH")
        resolved: dict[str, Path] = {}
        for name, path in sources.items():
            if not SOURCE_NAME_RE.match(name):
                raise ValueError(
                    f"source name {name!r} is not usable: names must match "
                    f"{SOURCE_NAME_RE.pattern}"
                )
            resolved[name] = Path(path)
        return resolved, True

    if source is not None:
        return {PRIMARY_SOURCE_NAME: Path(source)}, False

    # Nothing passed: read what the job directory already holds.
    return dict(job.sources), job.has_named_sources


def _write_media_json(
    path: Path,
    docs: dict[str, MediaDoc],
    *,
    audio_from: str,
    named: bool,
) -> Path:
    """media.json, in whichever of its two shapes this job needs.

    One unnamed source keeps the bare `MediaDoc` every existing job has. Named
    sources get a mapping, because `MediaDoc` describes one file and the facts
    that matter here -- which file is primary, and which one the transcript was
    taken from -- are about the set.
    """
    if not named:
        return write_json(path, docs[next(iter(docs))])
    payload = {
        "primary": next(iter(docs)),
        "audio_from": audio_from,
        "sources": {name: doc.model_dump(mode="json") for name, doc in docs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _merge_regions(docs: dict[str, object], *, prefix: bool) -> RegionsDoc:
    """One regions.json out of a per-source detection each.

    Ids are namespaced with the source name, because `cam_a` from the
    active-speaker view and `cam_a` from the gallery view are different
    rectangles of different files and a layout has to be able to say which. The
    prefix is skipped when the job has a single source: there is nothing to
    collide with, and every id in every existing regions.json would otherwise
    change name for no reason.
    """
    regions: list[Region] = []
    grids: list[str] = []
    frames = 0
    for name, doc in docs.items():
        grids.append(f"{name}={getattr(doc, 'grid', '?')}")
        frames += int(getattr(doc, "frames_sampled", 0) or 0)
        for raw in getattr(doc, "regions", None) or []:
            region = raw if isinstance(raw, Region) else Region.model_validate(raw)
            rid = f"{name}__{region.id}" if prefix else region.id
            regions.append(region.model_copy(update={"id": rid, "source": name}))
    return RegionsDoc(grid=" ".join(grids), frames_sampled=frames, regions=regions)


def _detect_regions(source: Path, media: MediaDoc):
    """Call `prepare.regions` if it is present.

    Region detection is optional by design: `frame` is always a valid layout
    target, so a job with no `regions.json` is degraded, not broken. The module
    is imported here rather than at the top so this file does not fail to
    import while that module is still being written, and the entry point is
    looked up by name because it is owned by different code.
    """
    # importlib rather than `from makeshorts.prepare import regions`: a
    # from-import reads the attribute already bound on the parent package once
    # the submodule has been imported anywhere in the process, which silently
    # ignores sys.modules. For an optional dependency that callers need to be
    # able to substitute or remove, resolving through sys.modules is the
    # behaviour we actually want.
    regions_mod = importlib.import_module("makeshorts.prepare.regions")  # noqa: PLC0415

    for name in ("detect_regions", "propose_regions", "analyze_regions", "detect"):
        fn = getattr(regions_mod, name, None)
        if callable(fn):
            break
    else:
        raise AttributeError("makeshorts.prepare.regions exposes no detection entry point")

    # Pass only what it actually accepts -- the signature is not ours.
    available = {
        "duration": media.duration,
        "resolution": media.resolution,
        "fps": media.fps,
    }
    params = inspect.signature(fn).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in available.items() if accepts_any or k in params}
    return fn(source, **kwargs)


def prepare(
    job: Job,
    *,
    source: str | Path | None = None,
    sources: Mapping[str, str | Path] | None = None,
    audio_from: str | None = None,
    skip: Iterable[str] = (),
    force: bool = False,
    threshold_db: float = silence_mod.DEFAULT_THRESHOLD_DB,
    min_silence: float = silence_mod.DEFAULT_MIN_DURATION,
    model: str = transcribe_mod.DEFAULT_MODEL,
    device: str = transcribe_mod.DEFAULT_DEVICE,
    compute_type: str = transcribe_mod.DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    log: Logger = _noop,
) -> PrepareResult:
    """Run the mechanical stage over `job`.

    `force` re-runs stages whose artifacts already exist; without it, an
    existing artifact is loaded and reused, which is what makes re-running
    after a crash cheap. Stages in `skip` are never run.

    `source` is one file. `sources` is a mapping of name to file for a
    recording exported as several frame-aligned views; they are mutually
    exclusive, and passing neither reads whatever the job directory holds.
    `audio_from` names the source to transcribe, defaulting to the first --
    the views share an audio track, so exactly one of them is decoded for it.
    """
    started = time.monotonic()
    skip_set = {s.strip().lower() for s in skip}
    unknown = skip_set - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage(s) to skip: {sorted(unknown)}; known: {list(STAGES)}")

    job.ensure_dirs()
    src_map, named = _resolve_sources(job, source, sources)
    for name, path in src_map.items():
        if not path.exists():
            where = f"{path}" if not named else f"{path} (source {name!r})"
            raise FileNotFoundError(
                f"job {job.slug!r} has no source media at {where} -- ingest one first"
            )

    primary = next(iter(src_map))
    audio_name = audio_from or primary
    if audio_name not in src_map:
        raise ValueError(
            f"--audio-from {audio_name!r} is not one of this job's sources: "
            f"{', '.join(src_map)}"
        )
    audio_path = src_map[audio_name]

    result = PrepareResult(job=job, sources=dict(src_map), audio_source=audio_name)

    def should_run(stage: str, artifact: Path) -> bool:
        if stage in skip_set:
            result.skipped.append(stage)
            log(f"skip   {stage}")
            return False
        if artifact.exists() and not force:
            result.skipped.append(stage)
            log(f"cached {stage} ({artifact.name})")
            return False
        return True

    # media.json records which source words.json was transcribed from, so it
    # has to record the source that actually produced the file on disk. A
    # cached words.json was made from whatever it was made from, and saying
    # otherwise because this run passed a different --audio-from would be a
    # lie in an artifact whose whole job is provenance.
    will_transcribe = "transcribe" not in skip_set and (force or not job.words_json.exists())
    audio_record = audio_name
    if not will_transcribe and job.words_json.exists() and job.media_json.exists():
        audio_record = job.audio_source_name
        if audio_record != audio_name:
            result.warnings.append(
                f"words.json already exists and was transcribed from source "
                f"{audio_record!r}, not {audio_name!r}; delete it or pass --force to "
                f"re-transcribe from {audio_name!r}"
            )

    def of_source(text: str, name: str) -> str:
        """Tag a message with the source it is about, but only when there is
        more than one and the tag therefore carries information."""
        return text if len(src_map) == 1 else f"{text} [{name}]"

    # -- media.json --------------------------------------------------------
    media_by_source: dict[str, MediaDoc] = {}
    if should_run("probe", job.media_json):
        log("probe  ffprobe + sha256")
        for name, path in src_map.items():
            if named:
                log(f"       {name.ljust(10)} {path.name}")
            media_by_source[name] = probe(path)
        result.wrote(
            _write_media_json(job.media_json, media_by_source, audio_from=audio_record, named=named)
        )
    elif job.media_json.exists():
        media_by_source = job.load_media_sources()
        # A source added to an existing job has no entry yet. Probing just that
        # one is the whole point of the artifacts being the state.
        fresh = {name: path for name, path in src_map.items() if name not in media_by_source}
        if fresh:
            for name, path in fresh.items():
                log(f"probe  {name} (new source)")
                media_by_source[name] = probe(path)
            media_by_source = {n: media_by_source[n] for n in src_map if n in media_by_source}
            result.wrote(
                _write_media_json(
                    job.media_json, media_by_source, audio_from=audio_record, named=named
                )
            )

    result.media_by_source = media_by_source
    result.media = media_by_source.get(primary)

    if result.media is None:
        # Everything below needs a duration. Without media.json there is
        # nothing honest to do.
        raise RuntimeError("media.json is required by the later stages but was skipped")

    media = result.media
    # Every audio decision is about the file the transcript comes from, which
    # on a multi-source job need not be the primary one.
    audio_media = media_by_source.get(audio_name, media)
    if not audio_media.has_audio:
        result.warnings.append(
            of_source("source has no audio stream -- no transcript will be produced", audio_name)
        )

    # -- silence.json ------------------------------------------------------
    if should_run("silence", job.silence_json):
        if audio_media.has_audio:
            log(f"silence  {threshold_db}dB / {min_silence}s")
            result.silence = silence_mod.detect_silence(
                audio_path,
                threshold_db=threshold_db,
                min_duration=min_silence,
                duration=audio_media.duration,
            )
        else:
            result.silence = silence_mod.silence_doc_for_no_audio(
                threshold_db=threshold_db, min_duration=min_silence
            )
        result.wrote(write_json(job.silence_json, result.silence))
    elif job.silence_json.exists():
        result.silence = job.load_silence()

    # -- words.json --------------------------------------------------------
    if should_run("transcribe", job.words_json):
        if not audio_media.has_audio:
            result.warnings.append("transcribe skipped: no audio stream")
        else:
            log(f"transcribe  {model} ({compute_type} on {device})")
            if len(src_map) > 1:
                # Said out loud because it is the one stage that does *not*
                # run per source, and a reader of words.json is entitled to
                # know which file's audio produced it.
                log(f"           audio from source {audio_name!r} ({audio_path.name})")
            result.words = transcribe_mod.transcribe(
                audio_path,
                model=model,
                device=device,
                compute_type=compute_type,
                language=language,
            )
            result.wrote(write_json(job.words_json, result.words))
            log(f"           {len(result.words.words)} words")
    elif job.words_json.exists():
        result.words = job.load_words()

    # -- regions.json ------------------------------------------------------
    if should_run("regions", job.regions_json):
        detected: dict[str, object] = {}
        for name, path in src_map.items():
            source_media = media_by_source[name]
            if source_media.resolution[0] <= 0:
                result.warnings.append(
                    of_source("regions skipped: source has no video stream", name)
                )
                continue
            try:
                log(of_source("regions  tile variance", name))
                detected[name] = _detect_regions(path, source_media)
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad. Region detection is a *proposal* and the
                # only optional stage: `frame` is always a valid layout target,
                # so a job without regions.json is degraded, not broken. Losing
                # a transcript because a heuristic tripped over an odd frame
                # would be a much worse trade.
                result.warnings.append(
                    of_source(
                        f"region detection failed or is unavailable "
                        f"({type(exc).__name__}: {exc}); `frame` remains usable as a layout "
                        f"target and regions can be declared by hand in clips.json",
                        name,
                    )
                )
        if detected and not named:
            # One unnamed source: write exactly what the detector returned.
            result.wrote(write_json(job.regions_json, detected[primary]))
        elif detected:
            merged = _merge_regions(detected, prefix=len(src_map) > 1)
            result.wrote(write_json(job.regions_json, merged))
    elif len(src_map) > 1 and job.regions_json.exists():
        # A cached regions.json predates any source added since. It is not
        # wrong, it is incomplete, and nothing downstream can tell the
        # difference between "this view has no regions" and "this view was
        # never looked at".
        try:
            covered = {r.source for r in job.load_regions().regions if r.source}
        except Exception:  # noqa: BLE001 - a bad regions.json is its own problem
            covered = set()
        uncovered = [name for name in src_map if name not in covered]
        if covered and uncovered:
            result.warnings.append(
                f"regions.json proposes nothing for {', '.join(uncovered)}; if "
                f"{'those sources were' if len(uncovered) > 1 else 'that source was'} "
                "added after it was written, delete it or pass --force to detect "
                "regions for them"
            )

    # -- transcript.txt + raw.srt ------------------------------------------
    if "text" in skip_set:
        result.skipped.append("text")
    elif result.words is None:
        result.warnings.append("transcript.txt and raw.srt skipped: no words.json")
    else:
        # Always regenerated when words are present: they are pure functions of
        # words.json and cost milliseconds, so caching them buys nothing and
        # risks a transcript that disagrees with the timestamps beside it.
        result.wrote(text_outputs.write_transcript(result.words, job.transcript_txt))
        result.wrote(text_outputs.write_srt(result.words, job.raw_srt))
        log("text   transcript.txt + raw.srt")

    result.elapsed = time.monotonic() - started
    for warning in result.warnings:
        log(f"warn   {warning}")
    return result


def summarize(result: PrepareResult) -> str:
    """One block of text for the CLI to print."""
    lines = [f"prepared {result.job.slug} in {result.elapsed:.1f}s"]
    if result.multi_source:
        for name, path in result.sources.items():
            lines.append(f"  source  {name} = {path}")
        lines.append(f"  audio   transcript from source {result.audio_source!r}")
    for path in result.written:
        lines.append(f"  wrote   {path}")
    for stage in result.skipped:
        lines.append(f"  skipped {stage}")
    for warning in result.warnings:
        lines.append(f"  warn    {warning}")
    return "\n".join(lines)


__all__ = ["STAGES", "PrepareResult", "prepare", "summarize", "write_json"]
