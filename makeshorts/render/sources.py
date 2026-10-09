"""Which file on disk each source named by the edit list actually is.

Shared by every engine. It is not geometry, so it does not belong in
`layout.py`, and it is not engine-shaped, so it does not belong in any one
engine: ffmpeg opens the files as `-i` inputs, Resolve imports them into a
media pool, and both have to agree on what `clips.json` meant by a name.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..select.schema import ClipsDoc

__all__ = ["resolve_source_paths", "search_roots", "first_existing"]


def resolve_source_paths(doc: ClipsDoc, source: Path) -> dict[str, Path]:
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

    roots = search_roots(source)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name, spec in specs.items():
        found = first_existing(Path(spec.path), roots)
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


def _unique(names: Iterable[str]) -> list[str]:
    """Deduplicate, preserving first-use order."""
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def search_roots(source: Path) -> list[Path]:
    """Directories a relative source path is tried against, nearest first."""
    base = source.parent if source.parent != Path("") else Path.cwd()
    roots = [base, *base.parents, Path.cwd()]
    return [Path(p) for p in _unique(str(p) for p in roots)]


def first_existing(path: Path, roots: list[Path]) -> Path | None:
    if path.is_absolute():
        return path if path.exists() else None
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return None
