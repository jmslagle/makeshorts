"""Job directory layout, slug derivation, and artifact loading.

A job is a directory. Everything a run produces lives inside it, named
predictably, so the state of the pipeline is readable with `ls` and re-runnable
by deleting a file. There is no database and no hidden state.

    jobs/<slug>/
      source.mp4       the input, copied or linked in
      source-cam.mp4   …or several named inputs, when one recording was
      source-slides.mp4  exported as multiple frame-aligned views
      media.json       ffprobe facts
      words.json       word-level transcript -- the source of truth for time
      silence.json     dead-air spans
      regions.json     proposed frame regions
      transcript.txt   human-readable, timestamped
      raw.srt          subtitle export of words.json
      PROMPT.md        generated from criteria.yaml; the editorial brief
      clips.json       the edit list -- AI-authored, human-editable
      cache/           caption PNGs and other regenerable intermediates
      out/<slug>--<clip.id>.mp4    rendered clip
      out/<slug>--<clip.id>.json   render receipt
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from makeshorts.artifacts import MediaDoc, RegionsDoc, SilenceDoc, WordsDoc
    from makeshorts.select.schema import ClipsDoc

JOBS_DIR_NAME = "jobs"

# Long enough to stay recognisable, short enough that
# `<slug>--<clip.id>.mp4` survives every filesystem we care about.
MAX_SLUG_LEN = 60

_NON_SLUG = re.compile(r"[^a-z0-9]+")
_SOURCE_STEM = "source"

# Video containers we will accept as a job's source, in preference order when
# more than one somehow exists.
SOURCE_SUFFIXES = (".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi", ".mp3", ".m4a", ".wav")

# The name a single-source job's one file goes by. Must agree with
# `select.schema.PRIMARY_SOURCE_NAME`; repeated rather than imported so that
# this module stays free of any dependency on the edit-list schema.
# `tests/cli/test_job_paths.py` asserts the two agree.
PRIMARY_SOURCE_NAME = "main"

# A source name is used as a region-id prefix and as a key in clips.json's
# `sources`, so it obeys the same rule as a region id.
SOURCE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# `source-cam.mp4` — a named input of a multi-source job.
_NAMED_SOURCE_RE = re.compile(rf"^{_SOURCE_STEM}-([a-z][a-z0-9_]*)$")


class JobError(RuntimeError):
    """Anything wrong with a job directory."""


class JobNotFound(JobError):
    pass


class StageNotRun(JobError):
    """An artifact was requested before the stage that writes it has run.

    Carries a remedy in the message -- the whole point is that the user is told
    which command to run next rather than being handed a FileNotFoundError.
    """

    def __init__(self, artifact: str, path: Path, remedy: str) -> None:
        self.artifact = artifact
        self.path = path
        self.remedy = remedy
        super().__init__(f"{artifact} not found at {path} — run `{remedy}` first")


class ArtifactInvalid(JobError):
    """An artifact exists but does not parse or does not validate."""

    def __init__(self, artifact: str, path: Path, detail: str) -> None:
        self.artifact = artifact
        self.path = path
        self.detail = detail
        super().__init__(f"{artifact} at {path} is not valid:\n{detail}")


# --------------------------------------------------------------------------
# Slugs
# --------------------------------------------------------------------------


def slugify(text: str, *, max_len: int = MAX_SLUG_LEN) -> str:
    """Lowercase kebab-case, ASCII only.

    Accents are folded rather than dropped so `Café Q3` becomes `cafe-q3` and
    not `caf-q3`.
    """
    folded = unicodedata.normalize("NFKD", text)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    kebab = _NON_SLUG.sub("-", ascii_only.lower()).strip("-")
    if len(kebab) > max_len:
        kebab = kebab[:max_len].rstrip("-")
    return kebab


def slug_for_input(input_path: str | Path) -> str:
    """Slug derived from an input filename, ignoring its extension."""
    stem = Path(input_path).stem
    slug = slugify(stem)
    return slug or "job"


def unique_slug(base: str, jobs_root: str | Path) -> str:
    """`base`, or `base-2`, `base-3`… — the first that is not taken.

    Collisions are common in practice: two exports of the same webinar land as
    `webinar.mp4` and `webinar (1).mp4`, which slugify identically.
    """
    root = Path(jobs_root)
    if not (root / base).exists():
        return base
    n = 2
    while (root / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One job directory. Pure paths plus validating loaders; creates nothing
    until you ask it to."""

    slug: str
    root: Path

    # -- construction ------------------------------------------------------

    @classmethod
    def at(cls, slug: str, jobs_root: str | Path = JOBS_DIR_NAME) -> Job:
        """A Job handle. Does not touch the filesystem."""
        return cls(slug=slug, root=Path(jobs_root) / slug)

    @classmethod
    def open(cls, slug: str, jobs_root: str | Path = JOBS_DIR_NAME) -> Job:
        """An existing job. Raises `JobNotFound` if the directory is absent."""
        job = cls.at(slug, jobs_root)
        if not job.root.is_dir():
            known = [j.slug for j in iter_jobs(jobs_root)]
            hint = f" Known jobs: {', '.join(known)}" if known else ""
            raise JobNotFound(f"no job {slug!r} under {Path(jobs_root)}/.{hint}")
        return job

    @classmethod
    def create(
        cls,
        slug: str,
        jobs_root: str | Path = JOBS_DIR_NAME,
        *,
        exist_ok: bool = False,
    ) -> Job:
        job = cls.at(slug, jobs_root)
        if job.root.exists() and not exist_ok:
            raise JobError(
                f"job {slug!r} already exists at {job.root} — pass --force to reuse it, "
                f"or --slug to name a new one"
            )
        job.ensure_dirs()
        return job

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

    # -- paths -------------------------------------------------------------

    @property
    def source(self) -> Path:
        """The job's primary source media.

        The extension follows whatever was ingested, so this looks for
        `source.*` and falls back to `source.mp4` for the not-yet-ingested case.
        A multi-source job has no `source.*`; there the primary named file is
        returned, so every existing caller keeps meaning "the file this job is
        mostly about".
        """
        for suffix in SOURCE_SUFFIXES:
            candidate = self.root / f"{_SOURCE_STEM}{suffix}"
            if candidate.exists():
                return candidate
        found = sorted(self.root.glob(f"{_SOURCE_STEM}.*"))
        if found:
            return found[0]
        named = self._named_sources()
        if named:
            return next(iter(named.values()))
        return self.root / f"{_SOURCE_STEM}.mp4"

    @property
    def sources(self) -> dict[str, Path]:
        """Every input file in this job, by name, primary first.

        A single-source job has exactly one entry under `PRIMARY_SOURCE_NAME`,
        which is what makes the multi-source case additive: callers that only
        ever want one file keep using `.source`, and callers that can handle
        several read this and never branch on how many there are.
        """
        named = self._named_sources()
        if named:
            return named
        return {PRIMARY_SOURCE_NAME: self.source}

    @property
    def has_named_sources(self) -> bool:
        """True when this job was ingested as several named files."""
        return bool(self._named_sources())

    def _named_sources(self) -> dict[str, Path]:
        """`source-<name>.<ext>` files, ordered by media.json where it says.

        Filesystem order is alphabetical and therefore arbitrary, but which
        source is *primary* is a real decision — it is the one `frame` and an
        unqualified region refer to. media.json records the order the sources
        were ingested in; disk order is only the fallback.
        """
        if not self.root.is_dir():
            return {}
        found: dict[str, Path] = {}
        for path in sorted(self.root.glob(f"{_SOURCE_STEM}-*.*")):
            match = _NAMED_SOURCE_RE.match(path.stem)
            if match and path.suffix.lower() in SOURCE_SUFFIXES:
                found.setdefault(match.group(1), path)
        if len(found) < 2:
            return found
        ordered = {name: found[name] for name in self._recorded_source_order() if name in found}
        ordered.update({name: path for name, path in found.items() if name not in ordered})
        return ordered

    def _recorded_source_order(self) -> list[str]:
        """Source names in the order media.json lists them. Never raises: this
        feeds a path property, and a malformed artifact must not make a path
        unreadable."""
        try:
            data = json.loads(self.media_json.read_text())
        except Exception:  # noqa: BLE001 - absent, unreadable or malformed
            return []
        if not isinstance(data, dict):
            return []
        sources = data.get("sources")
        names = list(sources) if isinstance(sources, dict) else []
        primary = data.get("primary")
        if isinstance(primary, str) and primary in names:
            names.remove(primary)
            names.insert(0, primary)
        return names

    def source_for(self, input_path: str | Path, name: str | None = None) -> Path:
        """Where an input file should land inside this job, keeping its
        container extension.

        Unnamed for a single-source job (`source.mp4`), named for a
        multi-source one (`source-cam.mp4`).
        """
        suffix = Path(input_path).suffix.lower()
        if name is None:
            return self.root / f"{_SOURCE_STEM}{suffix}"
        if not SOURCE_NAME_RE.match(name):
            raise JobError(
                f"source name {name!r} is not usable: names must match "
                f"{SOURCE_NAME_RE.pattern} — they become region-id prefixes and "
                f"keys in clips.json `sources`."
            )
        return self.root / f"{_SOURCE_STEM}-{name}{suffix}"

    @property
    def media_json(self) -> Path:
        return self.root / "media.json"

    @property
    def words_json(self) -> Path:
        return self.root / "words.json"

    @property
    def silence_json(self) -> Path:
        return self.root / "silence.json"

    @property
    def regions_json(self) -> Path:
        return self.root / "regions.json"

    @property
    def transcript_txt(self) -> Path:
        return self.root / "transcript.txt"

    @property
    def raw_srt(self) -> Path:
        return self.root / "raw.srt"

    @property
    def prompt_md(self) -> Path:
        return self.root / "PROMPT.md"

    @property
    def clips_json(self) -> Path:
        return self.root / "clips.json"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def cache_dir(self) -> Path:
        """Regenerable intermediates -- caption PNGs, segment files. Safe to
        delete at any time."""
        return self.root / "cache"

    @property
    def config_dir(self) -> Path:
        """Per-job config overrides. Optional; see `makeshorts.config`."""
        return self.root / "config"

    # -- output naming -----------------------------------------------------

    def output_path(self, clip_id: str, *, suffix: str = ".mp4") -> Path:
        """`out/<slug>--<clip.id>.mp4`.

        The slug is repeated in the filename deliberately: rendered clips get
        dragged out of the job dir into an upload queue, and the filename has to
        survive that trip and still say which webinar it came from.
        """
        return self.out_dir / f"{self.slug}--{clip_id}{suffix}"

    def receipt_path(self, clip_id: str) -> Path:
        """The render receipt next to the clip: same name, `.json`."""
        return self.output_path(clip_id, suffix=".json")

    def caption_cache_dir(self, clip_id: str) -> Path:
        return self.cache_dir / "captions" / clip_id

    # -- loaders -----------------------------------------------------------

    def _read_json(self, path: Path, artifact: str, remedy: str) -> object:
        if not path.exists():
            raise StageNotRun(artifact, path, remedy)
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactInvalid(artifact, path, f"malformed JSON: {exc}") from exc

    def _load(self, path: Path, artifact: str, remedy: str, model: type) -> object:
        from pydantic import ValidationError

        data = self._read_json(path, artifact, remedy)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ArtifactInvalid(artifact, path, str(exc)) from exc

    def load_media(self) -> MediaDoc:
        """The primary source's ffprobe facts.

        Single-source jobs -- which is nearly all of them -- have exactly this
        in media.json and nothing else.
        """
        docs = self.load_media_sources()
        return docs[next(iter(docs))]

    def load_media_sources(self) -> dict[str, MediaDoc]:
        """Every source's ffprobe facts, by name, primary first.

        media.json has two shapes and this is the one place that knows it: a
        bare `MediaDoc` for a job with one file, or `{"primary", "audio_from",
        "sources": {name: MediaDoc}}` for a job with several. The second exists
        because `MediaDoc` describes one file and a job may now have four; the
        first is kept verbatim so no existing job directory has to be migrated.
        """
        remedy = f"ms prepare {self.slug}"
        data = self._read_json(self.media_json, "media.json", remedy)
        if not isinstance(data, dict) or "sources" not in data:
            return {PRIMARY_SOURCE_NAME: self._load_media_doc(data, PRIMARY_SOURCE_NAME)}

        raw = data.get("sources")
        if not isinstance(raw, dict) or not raw:
            raise ArtifactInvalid(
                "media.json", self.media_json, "`sources` must be a non-empty object"
            )
        names = list(raw)
        primary = data.get("primary")
        if isinstance(primary, str) and primary in names:
            names.remove(primary)
            names.insert(0, primary)
        return {name: self._load_media_doc(raw[name], name) for name in names}

    def _load_media_doc(self, data: object, name: str) -> MediaDoc:
        from pydantic import ValidationError

        from makeshorts.artifacts import MediaDoc

        try:
            return MediaDoc.model_validate(data)
        except ValidationError as exc:
            where = "" if name == PRIMARY_SOURCE_NAME else f"source {name!r}: "
            raise ArtifactInvalid("media.json", self.media_json, f"{where}{exc}") from exc

    @property
    def audio_source_name(self) -> str:
        """Which source the transcript was taken from, per media.json.

        Frame-aligned exports of one meeting share an audio track, so exactly
        one of them is transcribed and the rest are silent as far as
        `words.json` is concerned. Recorded rather than inferred, because
        "which file did these timestamps come from" is not a question the
        filesystem can answer later.
        """
        try:
            data = json.loads(self.media_json.read_text())
        except Exception:  # noqa: BLE001
            return next(iter(self.sources))
        if isinstance(data, dict):
            audio_from = data.get("audio_from")
            if isinstance(audio_from, str) and audio_from:
                return audio_from
        return next(iter(self.sources))

    def load_words(self) -> WordsDoc:
        from makeshorts.artifacts import WordsDoc

        return self._load(self.words_json, "words.json", f"ms prepare {self.slug}", WordsDoc)

    def load_silence(self) -> SilenceDoc:
        from makeshorts.artifacts import SilenceDoc

        return self._load(self.silence_json, "silence.json", f"ms prepare {self.slug}", SilenceDoc)

    def load_regions(self) -> RegionsDoc:
        from makeshorts.artifacts import RegionsDoc

        return self._load(self.regions_json, "regions.json", f"ms prepare {self.slug}", RegionsDoc)

    def load_clips(self) -> ClipsDoc:
        from makeshorts.select.schema import ClipsDoc

        return self._load(
            self.clips_json,
            "clips.json",
            f"ms plan {self.slug}",
            ClipsDoc,
        )

    def write_clips(self, doc: ClipsDoc) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.clips_json.write_text(doc.model_dump_json(indent=2, exclude_none=True) + "\n")
        return self.clips_json

    # -- state -------------------------------------------------------------

    @property
    def stages(self) -> dict[str, bool]:
        """Which milestones this job has reached, in pipeline order."""
        return {
            "source": self.source.exists(),
            "prepared": self.words_json.exists() and self.media_json.exists(),
            "prompt": self.prompt_md.exists(),
            "clips": self.clips_json.exists(),
            "rendered": any(self.out_dir.glob("*.mp4")) if self.out_dir.is_dir() else False,
        }

    @property
    def stage(self) -> str:
        """The furthest stage reached, as a single word for `ms jobs`.

        The *furthest*, not the last before a gap. Stages are legitimately
        skippable -- `--skip transcribe` leaves no words.json, a hand-assembled
        edit list may have no copied source -- and stopping at the first gap
        reported a job with sixteen rendered clips as `empty`, then advised
        re-running `ms prepare` over it.
        """
        reached = "empty"
        for name, done in self.stages.items():
            if done:
                reached = name
        return reached

    def rendered_clip_ids(self) -> list[str]:
        if not self.out_dir.is_dir():
            return []
        prefix = f"{self.slug}--"
        return sorted(
            p.stem[len(prefix) :] for p in self.out_dir.glob("*.mp4") if p.stem.startswith(prefix)
        )


def iter_jobs(jobs_root: str | Path = JOBS_DIR_NAME) -> Iterator[Job]:
    """Every job directory under `jobs_root`, slug-sorted."""
    root = Path(jobs_root)
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            yield Job(slug=child.name, root=child)


__all__ = [
    "JOBS_DIR_NAME",
    "MAX_SLUG_LEN",
    "PRIMARY_SOURCE_NAME",
    "SOURCE_NAME_RE",
    "SOURCE_SUFFIXES",
    "JobError",
    "JobNotFound",
    "StageNotRun",
    "ArtifactInvalid",
    "slugify",
    "slug_for_input",
    "unique_slug",
    "Job",
    "iter_jobs",
]
