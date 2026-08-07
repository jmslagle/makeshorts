"""Loading and validating config/render.yaml and config/styles.yaml.

Two things matter beyond "it parses". First, the per-job override has to merge
key-by-key, because a job that wants a faster encoder should write four lines
and not a copy of the whole file. Second, a broken config must name the file
and the field -- a config error that surfaces as a KeyError three modules later
is the failure mode this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from makeshorts.config import (
    ConfigError,
    ConfigInvalid,
    ConfigNotFound,
    UnknownStyle,
    deep_merge,
    load_config,
    load_render_config,
    load_styles,
)


# --------------------------------------------------------------------------
# The shipped file
# --------------------------------------------------------------------------


def test_the_repos_own_render_yaml_loads(config_dir: Path) -> None:
    cfg = load_render_config(config_dir)
    assert cfg.video.codec == "libx264"
    assert cfg.video.crf == 20
    assert cfg.output.width == 1080
    assert cfg.output.height == 1920
    assert cfg.output.fps == 30
    assert cfg.audio.normalize.integrated == -14.0
    assert cfg.captions.default_style == "pill-karaoke"


def test_the_repos_own_styles_yaml_loads_and_defines_pill_karaoke(config_dir: Path) -> None:
    styles = load_styles(config_dir)
    assert styles.default == "pill-karaoke"
    assert "pill-karaoke" in styles.names
    style = styles.get()
    assert style.name == "pill-karaoke"
    assert Path(style.font_file).exists(), "the default style must name a font that is present"


def test_style_sizes_resolve_to_sane_pixels(config_dir: Path) -> None:
    """The whole point of fractional sizing is that a style survives a change
    of output resolution. Check it lands somewhere readable at 1080x1920."""
    style = load_styles(config_dir).get("pill-karaoke")
    assert 60 <= style.font_px(1920) <= 120
    assert style.font_px(1280) < style.font_px(1920)


def test_load_config_reports_which_files_contributed(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    names = [p.name for p in cfg.sources]
    assert "render.yaml" in names
    assert "styles.yaml" in names


def test_style_for_falls_back_to_the_render_config_default(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    assert cfg.style_for().name == cfg.render.captions.default_style
    assert cfg.style_for("impact-punch").name == "impact-punch"


def test_unknown_style_lists_the_ones_that_exist(config_dir: Path) -> None:
    cfg = load_config(config_dir)
    with pytest.raises(UnknownStyle) as exc:
        cfg.style_for("no-such-style")
    assert "pill-karaoke" in str(exc.value)


# --------------------------------------------------------------------------
# Per-job override
# --------------------------------------------------------------------------


def test_deep_merge_recurses_into_mappings() -> None:
    base = {"video": {"codec": "libx264", "crf": 20}, "audio": {"codec": "aac"}}
    over = {"video": {"crf": 18}}
    assert deep_merge(base, over) == {
        "video": {"codec": "libx264", "crf": 18},
        "audio": {"codec": "aac"},
    }


def test_deep_merge_replaces_lists_rather_than_appending() -> None:
    """Appending to `extra_args` by accident is worse than restating it: you
    would get both the old flags and the new ones, silently."""
    assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


def test_job_override_changes_only_what_it_names(config_dir: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "render.yaml").write_text("video:\n  codec: h264_videotoolbox\n")

    cfg = load_render_config(config_dir, overlay_dir=overlay)
    assert cfg.video.codec == "h264_videotoolbox"
    assert cfg.video.is_hardware
    # Untouched keys survive the merge.
    assert cfg.video.crf == 20
    assert cfg.video.pix_fmt == "yuv420p"
    assert cfg.audio.codec == "aac"


def test_a_missing_job_override_is_not_an_error(config_dir: Path, tmp_path: Path) -> None:
    cfg = load_render_config(config_dir, overlay_dir=tmp_path / "does-not-exist")
    assert cfg.video.codec == "libx264"


def test_an_empty_override_file_means_no_overrides(config_dir: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "render.yaml").write_text("# everything commented out\n")
    assert load_render_config(config_dir, overlay_dir=overlay).video.crf == 20


def test_job_override_can_restyle_one_field_of_one_style(
    config_dir: Path, tmp_path: Path
) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "styles.yaml").write_text("styles:\n  pill-karaoke:\n    max_lines: 3\n")

    styles = load_styles(config_dir, overlay_dir=overlay)
    style = styles.get("pill-karaoke")
    assert style.max_lines == 3
    # Everything else in that style is inherited, not reset to the model default.
    assert style.max_chars_per_line == 26
    assert style.pill_color == "#000000B3"


def test_job_override_can_nudge_one_safe_area_inset(config_dir: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "styles.yaml").write_text(
        "styles:\n  pill-karaoke:\n    safe_area:\n      bottom_pct: 0.2\n"
    )
    area = load_styles(config_dir, overlay_dir=overlay).get("pill-karaoke").safe_area
    assert area.bottom_pct == 0.2
    assert area.top_pct == 0.08


# --------------------------------------------------------------------------
# Bad input
# --------------------------------------------------------------------------


def test_malformed_yaml_names_the_file_and_the_line(tmp_path: Path) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "render.yaml").write_text(
        "video:\n  codec: libx264\n   crf: 20\n  preset: medium\n"  # bad indent on line 3
    )

    with pytest.raises(ConfigInvalid) as exc:
        load_render_config(bad)

    message = str(exc.value)
    assert str(bad / "render.yaml") in message
    assert "malformed YAML" in message
    assert "line 3" in message, f"the line number is the whole point: {message}"


def test_unknown_key_is_rejected_and_named(tmp_path: Path) -> None:
    """A silently-ignored key under the wrong parent would render successfully
    and be wrong, which is the one outcome this project cannot afford."""
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "render.yaml").write_text("video:\n  codek: libx264\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_render_config(bad)
    assert "video.codek" in str(exc.value)


def test_a_bad_enum_value_names_the_field(tmp_path: Path) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "render.yaml").write_text("video:\n  codec: mpeg1video\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_render_config(bad)
    assert "video.codec" in str(exc.value)


def test_out_of_range_value_names_the_field(tmp_path: Path) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "render.yaml").write_text("video:\n  crf: 99\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_render_config(bad)
    assert "video.crf" in str(exc.value)


def test_a_top_level_list_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "render.yaml").write_text("- video\n- audio\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_render_config(bad)
    assert "mapping" in str(exc.value)


def test_missing_render_yaml_names_the_path_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(ConfigNotFound) as exc:
        load_render_config(tmp_path / "nowhere")
    assert "render.yaml" in str(exc.value)


def test_missing_styles_yaml_falls_back_to_the_built_ins(tmp_path: Path) -> None:
    """styles.yaml is optional -- the style model ships a complete set, and a
    checkout with no config file should still be able to render."""
    empty = tmp_path / "config"
    empty.mkdir()
    styles = load_styles(empty)
    assert "pill-karaoke" in styles.names


def test_a_broken_style_names_the_style_and_the_field(config_dir: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "styles.yaml").write_text("styles:\n  pill-karaoke:\n    font_size_pct: 9.0\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_styles(config_dir, overlay_dir=overlay)
    message = str(exc.value)
    assert "pill-karaoke" in message
    assert "font_size_pct" in message


def test_an_unparseable_colour_is_rejected(config_dir: Path, tmp_path: Path) -> None:
    overlay = tmp_path / "job-config"
    overlay.mkdir()
    (overlay / "styles.yaml").write_text("styles:\n  pill-karaoke:\n    fill: nearly-white\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_styles(config_dir, overlay_dir=overlay)
    assert "pill-karaoke" in str(exc.value)


def test_an_unknown_default_style_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "config"
    bad.mkdir()
    (bad / "styles.yaml").write_text("default: no-such-style\nstyles: {}\n")

    with pytest.raises(ConfigInvalid) as exc:
        load_styles(bad)
    assert "no-such-style" in str(exc.value)


def test_every_config_failure_is_a_config_error(tmp_path: Path) -> None:
    """Callers catch one exception type. `cli.py` relies on this to turn any
    config problem into a sentence instead of a traceback."""
    assert issubclass(ConfigInvalid, ConfigError)
    assert issubclass(ConfigNotFound, ConfigError)
    assert issubclass(UnknownStyle, ConfigError)


# --------------------------------------------------------------------------
# Coercions worth having
# --------------------------------------------------------------------------


def test_unquoted_numeric_level_is_accepted(tmp_path: Path) -> None:
    """`level: 4.1` unquoted is a float in YAML. Nobody should have to learn
    that, so it is coerced rather than rejected."""
    d = tmp_path / "config"
    d.mkdir()
    (d / "render.yaml").write_text("video:\n  level: 4.1\n")
    assert load_render_config(d).video.level == "4.1"


# -- layout geometry --------------------------------------------------------


def test_layout_options_come_from_render_yaml(tmp_path: Path) -> None:
    """The `layout:` block reaches layout.py's own model.

    Validated against the real LayoutOptions rather than a mirror in config.py,
    so a field added there is usable from YAML immediately and the two cannot
    drift.
    """
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "render.yaml").write_text(
        "video:\n  codec: libx264\nlayout:\n  blur_radius_pct: 0.09\n  stack_gap_px: 12\n"
    )
    opts = load_config(cfg_dir).layout_options()
    assert opts.blur_radius_pct == 0.09
    assert opts.stack_gap_px == 12
    # Unspecified keys keep layout.py's defaults rather than becoming zero.
    assert opts.inset_max_height_pct == 0.45


def test_layout_block_is_optional(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "render.yaml").write_text("video:\n  codec: libx264\n")
    opts = load_config(cfg_dir).layout_options()
    assert opts.blur_radius_pct == 0.02  # layout.py's default


def test_a_bad_layout_value_names_the_field(tmp_path: Path) -> None:
    """Out of range should fail while reading config, not silently clamp."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "render.yaml").write_text(
        "video:\n  codec: libx264\nlayout:\n  blur_radius_pct: 5.0\n"
    )
    with pytest.raises(ConfigError) as exc:
        load_config(cfg_dir).layout_options()
    assert "blur_radius_pct" in str(exc.value)
