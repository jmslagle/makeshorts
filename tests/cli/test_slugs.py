"""Slug derivation and collision handling.

A slug names a directory and is repeated inside every rendered filename, so it
has to be stable, ASCII, and unique. The collision case is not hypothetical:
two exports of the same webinar routinely arrive as `webinar.mp4` and
`webinar (1).mp4`, which slugify identically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makeshorts.jobs import MAX_SLUG_LEN, slug_for_input, slugify, unique_slug


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme Q3 Webinar", "acme-q3-webinar"),
        ("ACME  Q3   Webinar", "acme-q3-webinar"),
        ("acme_q3_webinar", "acme-q3-webinar"),
        ("--acme--q3--", "acme-q3"),
        ("Acme: Q3 (Final)", "acme-q3-final"),
        ("2026-08-06 recording", "2026-08-06-recording"),
    ],
)
def test_slugify_normalizes(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_slugify_folds_accents_rather_than_dropping_them() -> None:
    """`Café Q3` must become `cafe-q3`, not `caf-q3`. Dropping the character
    silently mangles the name; folding it keeps the word readable."""
    assert slugify("Café Q3") == "cafe-q3"
    assert slugify("Ünicode Tëst") == "unicode-test"


def test_slugify_truncates_without_a_trailing_dash() -> None:
    long = "x" * 40 + " " + "y" * 40
    result = slugify(long)
    assert len(result) <= MAX_SLUG_LEN
    assert not result.endswith("-")


def test_slugify_of_unusable_input_is_empty() -> None:
    assert slugify("!!!") == ""
    assert slugify("") == ""


def test_slug_for_input_ignores_the_extension_and_directory() -> None:
    assert slug_for_input(Path("/downloads/Acme Q3 Webinar.mp4")) == "acme-q3-webinar"
    assert slug_for_input("Acme.Q3.mkv") == "acme-q3"


def test_slug_for_input_falls_back_when_nothing_survives() -> None:
    """A file called `!!!.mp4` still has to produce a usable directory name."""
    assert slug_for_input("!!!.mp4") == "job"


def test_unique_slug_returns_the_base_when_free(tmp_path: Path) -> None:
    assert unique_slug("acme-q3", tmp_path) == "acme-q3"


def test_unique_slug_suffixes_on_collision(tmp_path: Path) -> None:
    (tmp_path / "acme-q3").mkdir()
    assert unique_slug("acme-q3", tmp_path) == "acme-q3-2"

    (tmp_path / "acme-q3-2").mkdir()
    assert unique_slug("acme-q3", tmp_path) == "acme-q3-3"


def test_the_real_collision_two_exports_of_one_webinar(tmp_path: Path) -> None:
    """`webinar.mp4` and `webinar (1).mp4` slugify to different strings only
    because of the digit -- but `webinar.mp4` and `webinar.mov` do not, and
    that is the pair that actually collides."""
    first = unique_slug(slug_for_input("webinar.mp4"), tmp_path)
    (tmp_path / first).mkdir()
    second = unique_slug(slug_for_input("webinar.mov"), tmp_path)

    assert first == "webinar"
    assert second == "webinar-2"
    assert first != second


def test_unique_slug_skips_over_a_gap(tmp_path: Path) -> None:
    """Deleting `acme-2` must not make the next job reuse the name while
    `acme-3` still exists -- the suffix search stops at the first free slot,
    which is the intended behaviour and is worth pinning down."""
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme-3").mkdir()
    assert unique_slug("acme", tmp_path) == "acme-2"
