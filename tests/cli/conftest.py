"""Fixtures for the CLI and config tests.

The interesting one is `blocked_tracks`. This package is built by several
agents at once, and `cli.py` is required to import and render its help against
a tree where `prepare/`, `select/lint`, `select/prompt` and `render/` do not
exist at all. Rather than deleting files, those imports are blocked at the
`sys.meta_path` level, which reproduces the same `ModuleNotFoundError` without
touching the working copy.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]

# Everything owned by another build track. `select.schema`, `select.criteria`,
# `jobs`, `artifacts` and `config` are contracts and stay importable.
OTHER_TRACKS = (
    "makeshorts.prepare",
    "makeshorts.select.lint",
    "makeshorts.select.prompt",
    "makeshorts.select.snap",
    "makeshorts.render",
)


class _Blocker:
    """A meta-path finder that makes named packages look uninstalled."""

    def __init__(self, prefixes: tuple[str, ...]) -> None:
        self.prefixes = prefixes

    def _blocked(self, name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in self.prefixes)

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001, D102
        if self._blocked(fullname):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def blocked_tracks():
    """Run the body with the other tracks' modules unimportable.

    `makeshorts.cli` and `makeshorts.config` are evicted and re-imported so the
    test exercises a cold import against the reduced tree, which is the case
    that actually matters -- a top-level import of another track would only
    show up on first import.
    """
    blocker = _Blocker(OTHER_TRACKS)
    reset = ("makeshorts.cli", "makeshorts.config")

    # Only the modules this fixture evicts are put back. Restoring a whole
    # snapshot of sys.modules would also evict anything imported *during* the
    # test -- typer pulls in more of rich lazily while rendering help -- and
    # the next import would then produce a second copy of those classes, whose
    # instances the already-bound references reject.
    removed = {}
    for name in list(sys.modules):
        if blocker._blocked(name) or name in reset:
            removed[name] = sys.modules.pop(name)

    package = sys.modules.get("makeshorts")
    saved_attrs = {
        attr: getattr(package, attr)
        for attr in ("cli", "config", "prepare", "render")
        if package is not None and hasattr(package, attr)
    }

    sys.meta_path.insert(0, blocker)
    try:
        yield importlib.import_module("makeshorts.cli")
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if blocker._blocked(name) or name in reset:
                del sys.modules[name]
        sys.modules.update(removed)
        # `from makeshorts import cli` reads the package attribute, not
        # sys.modules, so the re-imported module has to be unbound there too.
        for attr, value in saved_attrs.items():
            setattr(package, attr, value)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A copy of the repo's real config/, so tests exercise the shipped files
    rather than a fixture that can drift away from them."""
    dest = tmp_path / "config"
    dest.mkdir()
    for name in ("render.yaml", "styles.yaml", "criteria.yaml"):
        (dest / name).write_text((REPO / "config" / name).read_text())
    return dest


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "jobs"
    d.mkdir()
    return d
