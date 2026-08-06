from __future__ import annotations

import hashlib

import pytest

from makeshorts.artifacts import MediaDoc
from makeshorts.prepare.probe import (
    ProbeError,
    media_doc_from_probe,
    parse_fps,
    probe,
    sha256_file,
)

from .conftest import requires_ffmpeg


@pytest.mark.parametrize(
    "value,expected",
    [
        ("30/1", 30.0),
        ("30000/1001", pytest.approx(29.97, abs=0.01)),
        ("25", 25.0),
        ("0/0", 0.0),
        ("", 0.0),
        (None, 0.0),
        ("garbage", 0.0),
    ],
)
def test_parse_fps(value, expected) -> None:
    assert parse_fps(value) == expected


def test_media_doc_from_probe_reads_the_video_stream() -> None:
    info = {
        "format": {"duration": "3612.44"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "60/1",
            },
            {"codec_type": "audio", "channels": 2},
        ],
    }
    doc = media_doc_from_probe(info, path="jobs/x/source.mp4", sha256="a" * 64)
    assert doc.resolution == (1920, 1080)
    # avg_frame_rate wins: r_frame_rate is often a meaningless multiple.
    assert doc.fps == pytest.approx(29.97, abs=0.01)
    assert doc.duration == 3612.44
    assert doc.has_audio is True
    assert doc.audio_channels == 2


def test_media_doc_falls_back_to_stream_duration() -> None:
    """Some MKV and fragmented-MP4 exports carry no container duration."""
    info = {
        "format": {},
        "streams": [
            {"codec_type": "video", "width": 640, "height": 480, "avg_frame_rate": "25/1",
             "duration": "12.5"},
        ],
    }
    assert media_doc_from_probe(info, path="s.mkv", sha256="b" * 64).duration == 12.5


def test_media_doc_without_a_duration_anywhere_raises() -> None:
    info = {"format": {}, "streams": [{"codec_type": "video", "width": 1, "height": 1}]}
    with pytest.raises(ProbeError, match="duration"):
        media_doc_from_probe(info, path="s.mp4", sha256="c" * 64)


def test_media_doc_for_a_file_with_no_audio_stream() -> None:
    info = {
        "format": {"duration": "10.0"},
        "streams": [
            {"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "25/1"}
        ],
    }
    doc = media_doc_from_probe(info, path="s.mp4", sha256="d" * 64)
    assert doc.has_audio is False
    assert doc.audio_channels == 0


def test_media_doc_for_an_audio_only_file() -> None:
    """Job.source accepts .mp3/.m4a/.wav, so a missing video stream is real."""
    info = {
        "format": {"duration": "90.0"},
        "streams": [{"codec_type": "audio", "channels": 1}],
    }
    doc = media_doc_from_probe(info, path="s.m4a", sha256="e" * 64)
    assert doc.resolution == (0, 0)
    assert doc.fps == 0.0
    assert doc.has_audio is True


def test_sha256_matches_hashlib(tmp_path) -> None:
    payload = b"x" * (3 * (1 << 20) + 17)  # spans several read chunks
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()
    # Chunk size must not change the answer.
    assert sha256_file(target, chunk_bytes=7) == hashlib.sha256(payload).hexdigest()


def test_probe_on_a_missing_file_raises() -> None:
    with pytest.raises(ProbeError, match="no such file"):
        probe("/nonexistent/nope.mp4")


@requires_ffmpeg
def test_probe_a_synthesised_av_file(av_file) -> None:
    doc = probe(av_file)
    assert isinstance(doc, MediaDoc)
    assert doc.resolution == (320, 240)
    assert doc.fps == pytest.approx(30.0)
    assert doc.duration == pytest.approx(6.0, abs=0.15)
    assert doc.has_audio is True
    assert doc.audio_channels == 1
    assert len(doc.sha256) == 64
    assert doc.sha256 == sha256_file(av_file)


@requires_ffmpeg
def test_probe_a_file_with_no_audio_stream(video_only_file) -> None:
    doc = probe(video_only_file)
    assert doc.has_audio is False
    assert doc.audio_channels == 0
    assert doc.resolution == (640, 360)
    assert doc.fps == pytest.approx(25.0)


@requires_ffmpeg
def test_supplied_hash_is_used_verbatim(av_file) -> None:
    doc = probe(av_file, sha256="f" * 64)
    assert doc.sha256 == "f" * 64


@requires_ffmpeg
def test_media_doc_round_trips_through_json(av_file) -> None:
    doc = probe(av_file)
    assert MediaDoc.model_validate(doc.model_dump(mode="json")) == doc
