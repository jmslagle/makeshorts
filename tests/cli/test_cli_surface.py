"""The CLI surface itself.

Two obligations, and the first is unusual enough to state plainly: **importing
`makeshorts.cli` and rendering every help screen must work with none of the
other build tracks present.** `prepare/`, `select/lint`, `select/prompt` and
`render/` are imported inside the command functions for exactly this reason. A
top-level import of any of them would make `ms --help` fail because of a bug in
a module the user was not invoking, and would make this package impossible to
build in parallel.

The second is that `ms lint` exits nonzero on an ERROR. `ms render` refuses to
run on a job that fails the gate, so the exit code is load-bearing and not a
cosmetic detail.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import typer

from makeshorts import cli as cli_mod
from makeshorts.jobs import Job

COMMANDS = ("prepare", "plan", "lint", "render", "caps", "jobs")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Help screens are rendered with per-character styling, so the literal
    string `--slug` does not appear in the raw output. Strip the escapes."""
    return _ANSI.sub("", text)


def _text(result) -> str:
    """Everything the command printed, wherever it went, without styling.

    Errors go to stderr and successes to stdout, which is correct behaviour and
    an annoyance to assert on; these tests care about the content.
    """
    out = result.stdout or ""
    try:
        err = result.stderr or ""
    except ValueError:  # stderr not captured separately by this click version
        err = ""
    return _plain(out + err)


# --------------------------------------------------------------------------
# Importable and helpful with the other tracks absent
# --------------------------------------------------------------------------


def test_cli_imports_in_a_fresh_interpreter_with_the_other_tracks_blocked() -> None:
    """A subprocess, so nothing already in this process's sys.modules can hide
    a top-level import of another track."""
    script = """
import sys

BLOCKED = (
    "makeshorts.prepare",
    "makeshorts.select.lint",
    "makeshorts.select.prompt",
    "makeshorts.select.snap",
    "makeshorts.render",
)


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == b or fullname.startswith(b + ".") for b in BLOCKED):
            raise ModuleNotFoundError("blocked: " + fullname, name=fullname)
        return None


sys.meta_path.insert(0, Blocker())
import makeshorts.cli  # noqa: F401

for name in BLOCKED:
    assert name not in sys.modules, name + " was imported at module scope"
print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_root_help_renders_with_the_other_tracks_blocked(blocked_tracks, runner) -> None:
    result = runner.invoke(blocked_tracks.app, ["--help"])
    assert result.exit_code == 0, _text(result)
    for command in COMMANDS:
        assert command in _text(result)


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_help_renders_with_the_other_tracks_blocked(
    blocked_tracks, runner, command: str
) -> None:
    result = runner.invoke(blocked_tracks.app, [command, "--help"])
    assert result.exit_code == 0, _text(result)
    assert command in _text(result)


def test_no_args_shows_help_rather_than_failing_silently(runner) -> None:
    result = runner.invoke(cli_mod.app, [])
    assert "Usage" in _text(result)


def test_documented_flags_are_actually_declared(runner) -> None:
    """The flags the design promises, checked against the parser rather than
    against the docs."""
    expected = {
        "prepare": ["--slug", "--model", "--force", "--source", "--audio-from"],
        "plan": [],
        "lint": ["--fix"],
        "render": ["--only", "--engine"],
        "caps": [],
        "jobs": [],
    }
    for command, flags in expected.items():
        out = _text(runner.invoke(cli_mod.app, [command, "--help"]))
        for flag in flags:
            assert flag in out, f"ms {command} is missing {flag}"


def test_a_missing_track_produces_a_sentence_not_a_traceback(
    blocked_tracks, runner, tmp_path: Path, config_dir: Path
) -> None:
    Job.create("acme-q3", tmp_path / "jobs")
    result = runner.invoke(
        blocked_tracks.app,
        ["--jobs-dir", str(tmp_path / "jobs"), "--config-dir", str(config_dir), "plan", "acme-q3"],
    )
    assert result.exit_code == 1
    text = _text(result)
    assert "makeshorts.select.prompt" in text
    assert "Traceback" not in text
    assert result.exception is None or isinstance(result.exception, SystemExit)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


def test_jobs_on_an_empty_tree_says_how_to_start_one(runner, tmp_path: Path) -> None:
    result = runner.invoke(cli_mod.app, ["--jobs-dir", str(tmp_path / "jobs"), "jobs"])
    assert result.exit_code == 0
    assert "ms prepare" in _text(result)


def test_jobs_lists_each_job_with_the_stage_it_reached(runner, tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    bare = Job.create("bare-job", jobs_root)
    prepared = Job.create("prepared-job", jobs_root)
    (prepared.root / "source.mp4").write_bytes(b"x")
    prepared.media_json.write_text("{}")
    prepared.words_json.write_text("{}")

    result = runner.invoke(cli_mod.app, ["--jobs-dir", str(jobs_root), "jobs"])
    assert result.exit_code == 0
    assert "bare-job" in _text(result)
    assert "empty" in _text(result)
    assert "prepared-job" in _text(result)
    assert "prepared" in _text(result)
    # The next step is named, because a list of states you cannot act on is
    # less useful than a list you can.
    assert f"ms plan {prepared.slug}" in _text(result)
    assert bare.slug in _text(result)


def test_jobs_survives_a_clips_json_that_does_not_parse(runner, tmp_path: Path) -> None:
    """A half-written clips.json is a real state a job can be in. Refusing to
    list anything because one job is broken would be the wrong trade."""
    jobs_root = tmp_path / "jobs"
    job = Job.create("broken", jobs_root)
    job.clips_json.write_text("{not json")

    result = runner.invoke(cli_mod.app, ["--jobs-dir", str(jobs_root), "jobs"])
    assert result.exit_code == 0
    assert "broken" in _text(result)
    assert "bad" in _text(result)


# --------------------------------------------------------------------------
# lint — the gate
# --------------------------------------------------------------------------


def _clips_doc(job_slug: str, clip_ids: tuple[str, ...] = ("01-first", "02-second")) -> dict:
    """A minimal edit list that validates against select/schema.py."""
    return {
        "job": job_slug,
        "source": {
            "path": f"jobs/{job_slug}/source.mp4",
            "duration": 3600.0,
            "resolution": [1920, 1080],
            "regions": [{"id": "cam_a", "kind": "speaker", "rect": [0.0, 0.0, 0.5, 1.0]}],
        },
        "criteria_ref": {"file": "config/criteria.yaml", "version": "2026-08-06", "sha256": "0" * 64},
        "clips": [
            {
                "id": clip_id,
                "title": f"Clip {clip_id}",
                "start": 100.0 + 100 * index,
                "end": 140.0 + 100 * index,
                "source_text": "some words",
                "why": {
                    "one_line": "because",
                    "theme": "pricing",
                    "scores": {"hook_strength": {"score": 5, "evidence": "e"}},
                    "weighted_score": 5.0,
                },
                "layout": {"mode": "focus", "region": "cam_a"},
            }
            for index, clip_id in enumerate(clip_ids)
        ],
    }


@pytest.fixture
def linted_job(tmp_path: Path, config_dir: Path):
    """A job with a valid clips.json, ready for a stubbed linter."""
    jobs_root = tmp_path / "jobs"
    job = Job.create("acme-q3", jobs_root)
    job.clips_json.write_text(json.dumps(_clips_doc(job.slug)))
    return job, jobs_root, config_dir


@pytest.fixture
def fake_lint(monkeypatch):
    """Install a stand-in `makeshorts.select.lint`.

    The real linter belongs to another track. What is under test here is the
    CLI's contract with it: findings get grouped and rendered, and an ERROR
    sets the exit code.
    """

    def install(findings, fixed=()):
        module = types.ModuleType("makeshorts.select.lint")

        class Finding:
            def __init__(self, severity, rule, message, clip_id=None, fixable=False):
                self.severity = severity
                self.rule = rule
                self.message = message
                self.clip_id = clip_id
                self.fixable = fixable

        class Report:
            def __init__(self):
                self.findings = [Finding(*f) for f in findings]
                self.fixed = list(fixed)

        def lint_job(job, criteria, fix=False):
            return Report()

        module.lint_job = lint_job
        module.Finding = Finding
        monkeypatch.setitem(sys.modules, "makeshorts.select.lint", module)
        return module

    return install


def test_lint_passes_cleanly_and_points_at_render(runner, linted_job, fake_lint) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint([])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    assert result.exit_code == 0, _text(result)
    assert "lint passed" in _text(result)
    assert f"ms render {job.slug}" in _text(result)


def test_lint_exits_nonzero_on_an_error(runner, linted_job, fake_lint) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint([("error", "time.word_boundary", "start 100.0 is not on a word edge", "01-first")])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    assert result.exit_code == 1, _text(result)


def test_lint_exits_zero_when_only_warnings(runner, linted_job, fake_lint) -> None:
    """A warning is advice. Failing the build on advice would train people to
    ignore the exit code, which is the one thing it cannot survive."""
    job, jobs_root, config_dir = linted_job
    fake_lint([("warning", "gate.duration", "43.8s is under the 45s target", "01-first")])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    assert result.exit_code == 0, _text(result)


def test_lint_groups_findings_by_clip_and_shows_rule_and_message(
    runner, linted_job, fake_lint
) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint(
        [
            ("error", "time.word_boundary", "start is not on a word edge", "01-first"),
            ("warning", "gate.duration", "under the target duration", "01-first"),
            ("error", "score.missing", "why.scores is missing `payoff`", "02-second"),
            ("error", "doc.overlap", "clips 01 and 02 overlap", None),
        ]
    )

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    text = _text(result)

    # Every clip with a finding gets a heading, and it appears once.
    assert text.count("01-first") == 1
    assert text.count("02-second") == 1

    # Rule id and message both survive.
    for token in ("time.word_boundary", "score.missing", "why.scores is missing"):
        assert token in text

    # Severity labels are present and distinguishable.
    assert "ERROR" in text
    assert "WARN" in text

    # Document-level findings lead, before any per-clip detail.
    assert text.index("clips 01 and 02 overlap") < text.index("01-first")

    # And a count, so the size of the problem is visible without counting lines.
    # Document-level findings are counted apart from the clips, so "2 of 2
    # clips" stays true rather than becoming "3 of 2".
    assert "3 errors" in text and "1 warning" in text
    assert "2 of 2 clips" in text
    assert "1 on the edit list itself" in text


def test_long_findings_wrap_into_the_message_column(runner, linted_job, fake_lint) -> None:
    """A finding is only useful if it can say something specific, which means
    messages get long. They must wrap under the message column rather than
    running off the terminal or resetting to the left margin."""
    long_message = (
        "source_text does not match words.json over 100.00-145.00; first divergence at "
        "word 12: wrote 'eighteen months', transcript has 'eighty months'"
    )
    job, jobs_root, config_dir = linted_job
    fake_lint([("error", "text.source_mismatch", long_message, "01-first")])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    lines = [ln for ln in _text(result).splitlines() if "months" in ln or "divergence" in ln]
    assert len(lines) > 1, "a long message should have wrapped"

    # Every wrapped line starts at the same column as the first line's message.
    starts = {len(ln) - len(ln.lstrip()) for ln in lines[1:]}
    assert len(starts) == 1, f"continuation lines are ragged: {starts}"
    assert all(len(ln) <= 100 for ln in lines), "wrapped past the terminal width"


def test_lint_mentions_fix_only_when_something_is_fixable(runner, linted_job, fake_lint) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint([("error", "time.word_boundary", "off by 0.4s", "01-first", True)])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug],
    )
    assert f"ms lint {job.slug} --fix" in _text(result)


def test_lint_reports_what_fix_changed(runner, linted_job, fake_lint) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint([], fixed=["01-first: start 1284.10 -> 1283.68"])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", job.slug, "--fix"],
    )
    assert result.exit_code == 0, _text(result)
    assert "1284.10 -> 1283.68" in _text(result)


def test_lint_without_a_clips_json_says_which_command_writes_it(
    runner, tmp_path: Path, config_dir: Path, fake_lint
) -> None:
    fake_lint([])
    jobs_root = tmp_path / "jobs"
    Job.create("acme-q3", jobs_root)

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", "acme-q3"],
    )
    assert result.exit_code == 1
    assert "clips.json" in _text(result)


def test_an_unknown_slug_lists_the_known_ones(runner, tmp_path: Path, config_dir: Path) -> None:
    jobs_root = tmp_path / "jobs"
    Job.create("acme-q3", jobs_root)

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "lint", "typo"],
    )
    assert result.exit_code == 1
    assert "acme-q3" in _text(result)


# --------------------------------------------------------------------------
# render — refuses to bypass the gate
# --------------------------------------------------------------------------


def test_render_refuses_when_lint_reports_an_error(runner, linted_job, fake_lint) -> None:
    job, jobs_root, config_dir = linted_job
    fake_lint([("error", "time.word_boundary", "start is not on a word edge", "01-first")])

    result = runner.invoke(
        cli_mod.app,
        ["--jobs-dir", str(jobs_root), "--config-dir", str(config_dir), "render", job.slug],
    )
    assert result.exit_code == 1
    text = _text(result)
    assert "refusing to render" in text
    assert f"ms lint {job.slug}" in text
    assert not list(job.out_dir.glob("*.mp4"))


def test_only_selects_by_numeric_prefix() -> None:
    """`--only 01,03` has to find `01-cac-payback-math` -- nobody types the
    whole slug."""
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(_clips_doc("acme", ("01-first", "02-second", "03-third")))
    picked = cli_mod._select_clips(doc, "01,03")
    assert [c.id for c in picked] == ["01-first", "03-third"]


def test_only_accepts_a_full_clip_id() -> None:
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(_clips_doc("acme"))
    assert [c.id for c in cli_mod._select_clips(doc, "02-second")] == ["02-second"]


def test_only_keeps_edit_list_order_regardless_of_how_it_was_named() -> None:
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(_clips_doc("acme"))
    assert [c.id for c in cli_mod._select_clips(doc, "02,01")] == ["01-first", "02-second"]


def test_only_with_no_match_lists_what_is_available(runner) -> None:
    from makeshorts.select.schema import ClipsDoc

    doc = ClipsDoc.model_validate(_clips_doc("acme"))
    with pytest.raises(typer.Exit):
        cli_mod._select_clips(doc, "99")


# --------------------------------------------------------------------------
# caps
# --------------------------------------------------------------------------


def test_caps_reports_the_caption_backend(runner, config_dir: Path) -> None:
    """The one thing `ms caps` exists to tell you: this machine's ffmpeg has no
    libass, so captions have to come from the Pillow backend."""
    pytest.importorskip("makeshorts.render.caps")
    result = runner.invoke(cli_mod.app, ["--config-dir", str(config_dir), "caps"])
    assert result.exit_code == 0, _text(result)
    assert "caption_backend" in _text(result)
    assert "has_libass" in _text(result)


# -- ms install -------------------------------------------------------------


def test_install_copies_the_skill(runner, tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert (dest / "makeshorts" / "SKILL.md").is_file()


def test_install_refuses_to_clobber_without_force(runner, tmp_path: Path) -> None:
    """Silently replacing an edited skill would lose work with no warning."""
    dest = tmp_path / "skills"
    runner.invoke(cli_mod.app, ["install", "--dest", str(dest)])
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(dest)])
    assert result.exit_code != 0
    assert "--force" in result.output


def test_install_force_replaces(runner, tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    runner.invoke(cli_mod.app, ["install", "--dest", str(dest)])
    (dest / "makeshorts" / "SKILL.md").write_text("edited")
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(dest), "--force"])
    assert result.exit_code == 0
    assert (dest / "makeshorts" / "SKILL.md").read_text() != "edited"


def test_install_link_tracks_the_checkout(runner, tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(dest), "--link"])
    assert result.exit_code == 0
    assert (dest / "makeshorts").is_symlink()


def test_uninstall_removes_it(runner, tmp_path: Path) -> None:
    dest = tmp_path / "skills"
    runner.invoke(cli_mod.app, ["install", "--dest", str(dest)])
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(dest), "--uninstall"])
    assert result.exit_code == 0
    assert not (dest / "makeshorts").exists()


def test_uninstall_on_nothing_is_not_an_error(runner, tmp_path: Path) -> None:
    result = runner.invoke(cli_mod.app, ["install", "--dest", str(tmp_path / "x"), "--uninstall"])
    assert result.exit_code == 0
