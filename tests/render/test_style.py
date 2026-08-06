"""Caption style model and loading."""

from __future__ import annotations

import pytest
import yaml

from makeshorts.render.captions.style import (
    DEFAULT_STYLE_NAME,
    DEFAULT_STYLES,
    CaptionStyle,
    SafeArea,
    StyleError,
    get_style,
    load_styles,
    parse_color,
    resolve_font_path,
    styles_yaml_template,
)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("#FFFFFF", (255, 255, 255, 255)),
        ("#000000B3", (0, 0, 0, 179)),
        ("#FFF", (255, 255, 255, 255)),
        ("#F008", (255, 0, 0, 136)),
        ("ffd400", (255, 212, 0, 255)),
        ("white", (255, 255, 255, 255)),
        ("transparent", (0, 0, 0, 0)),
    ],
)
def test_parse_color(value, expected):
    assert parse_color(value) == expected


@pytest.mark.parametrize("bad", ["", "#GGGGGG", "rgb(1,2,3)", "#12345"])
def test_bad_colours_are_rejected(bad):
    with pytest.raises(StyleError):
        parse_color(bad)


def test_a_bad_colour_fails_the_model_not_the_render():
    with pytest.raises(Exception):
        CaptionStyle(fill="nonsense")


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------


def test_sizes_scale_with_the_output_height():
    style = DEFAULT_STYLES[DEFAULT_STYLE_NAME]
    assert style.font_px(1920) == round(style.font_size_pct * 1920)
    # Halving the output halves the type, so a style is resolution-independent.
    assert style.font_px(960) == pytest.approx(style.font_px(1920) / 2, abs=1)


def test_safe_area_insets_are_css_ordered():
    area = SafeArea(top_pct=0.1, right_pct=0.2, bottom_pct=0.3, left_pct=0.4)
    assert area.insets(1000, 2000) == (200, 200, 600, 400)


def test_default_bottom_safe_area_clears_platform_ui():
    assert DEFAULT_STYLES[DEFAULT_STYLE_NAME].safe_area.bottom_pct >= 0.10


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------


def test_pill_karaoke_exists_and_is_karaoke():
    style = DEFAULT_STYLES["pill-karaoke"]
    assert style.karaoke
    assert style.pill_rgba[3] > 0, "the pill must actually be visible"
    assert style.highlight_rgba != style.fill_rgba


def test_every_default_style_is_self_consistent():
    for name, style in DEFAULT_STYLES.items():
        assert style.name == name
        assert resolve_font_path(style.font_file).is_file()
        assert style.font_px(1920) >= 12
        assert style.max_chars_per_line * style.max_lines >= 12


def test_static_block_is_not_karaoke():
    assert not DEFAULT_STYLES["static-block"].karaoke


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------


def test_font_falls_back_rather_than_failing():
    assert resolve_font_path("/nope/does-not-exist.ttf").is_file()
    assert resolve_font_path(None).is_file()


# --------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------


def test_missing_yaml_falls_back_to_the_defaults(tmp_path):
    styles = load_styles(tmp_path / "absent.yaml")
    assert set(styles) == set(DEFAULT_STYLES)
    assert styles["pill-karaoke"].fill == DEFAULT_STYLES["pill-karaoke"].fill


def test_yaml_overrides_field_by_field(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text(yaml.safe_dump({"styles": {"pill-karaoke": {"highlight_fill": "#FF00FF"}}}))
    style = load_styles(p)["pill-karaoke"]
    assert style.highlight_fill == "#FF00FF"
    # Untouched fields keep their built-in values.
    assert style.font_file == DEFAULT_STYLES["pill-karaoke"].font_file
    assert style.max_lines == DEFAULT_STYLES["pill-karaoke"].max_lines


def test_a_bare_top_level_mapping_also_works(tmp_path):
    """styles.yaml is owned elsewhere; both plausible shapes are accepted."""
    p = tmp_path / "styles.yaml"
    p.write_text(yaml.safe_dump({"pill-karaoke": {"max_lines": 3}}))
    assert load_styles(p)["pill-karaoke"].max_lines == 3


def test_yaml_can_add_a_new_style(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text(yaml.safe_dump({"styles": {"house": {"fill": "#00FF00"}}}))
    styles = load_styles(p)
    assert styles["house"].name == "house"
    assert styles["house"].fill == "#00FF00"


def test_nested_safe_area_merges(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text(
        yaml.safe_dump({"styles": {"pill-karaoke": {"safe_area": {"bottom_pct": 0.2}}}})
    )
    area = load_styles(p)["pill-karaoke"].safe_area
    assert area.bottom_pct == 0.2
    assert area.top_pct == DEFAULT_STYLES["pill-karaoke"].safe_area.top_pct


def test_a_typo_in_yaml_fails_loudly(tmp_path):
    """Silently ignoring `fil: red` would render the wrong thing forever."""
    p = tmp_path / "styles.yaml"
    p.write_text(yaml.safe_dump({"styles": {"pill-karaoke": {"fil": "#FF0000"}}}))
    with pytest.raises(StyleError, match="pill-karaoke"):
        load_styles(p)


def test_malformed_yaml_fails_loudly(tmp_path):
    p = tmp_path / "styles.yaml"
    p.write_text("styles: [this is: not: a mapping")
    with pytest.raises(StyleError):
        load_styles(p)


def test_get_style_names_what_exists(tmp_path):
    with pytest.raises(StyleError, match="pill-karaoke"):
        get_style("no-such-style", tmp_path / "absent.yaml")


def test_the_yaml_template_round_trips():
    """Whoever writes config/styles.yaml can start from this."""
    text = styles_yaml_template()
    parsed = yaml.safe_load(text)
    assert set(parsed["styles"]) == set(DEFAULT_STYLES)
    for name, fields in parsed["styles"].items():
        rebuilt = CaptionStyle.model_validate(fields | {"name": name})
        assert rebuilt == DEFAULT_STYLES[name]
