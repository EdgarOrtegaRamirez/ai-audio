"""Tests for speech-to-text module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_audio.stt import (
    TranscribeConfig,
    TranscribeResult,
    _format_srt_time,
    _format_vtt_time,
    _merge_transcription_results,
    _needs_chunking,
)


class TestFormatTime:
    """Test time formatting for subtitles."""

    def test_srt_zero(self):
        assert _format_srt_time(0) == "00:00:00,000"

    def test_srt_seconds(self):
        assert _format_srt_time(5.5) == "00:00:05,500"

    def test_srt_minutes(self):
        assert _format_srt_time(65.123) == "00:01:05,123"

    def test_srt_hours(self):
        assert _format_srt_time(3661.5) == "01:01:01,500"

    def test_vtt_zero(self):
        assert _format_vtt_time(0) == "00:00:00.000"

    def test_vtt_seconds(self):
        assert _format_vtt_time(5.5) == "00:00:05.500"

    def test_vtt_minutes(self):
        assert _format_vtt_time(65.123) == "00:01:05.123"

    def test_vtt_hours(self):
        assert _format_vtt_time(3661.5) == "01:01:01.500"


class TestTranscribeResult:
    """Test TranscribeResult methods."""

    def test_segments_text(self):
        result = TranscribeResult(
            text="Hello world",
            language="en",
            duration=10.0,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "Hello"},
                {"start": 5.0, "end": 10.0, "text": "world"},
            ],
        )
        output = result.segments_text
        assert "[0.0s - 5.0s] Hello" in output
        assert "[5.0s - 10.0s] world" in output

    def test_to_srt(self):
        result = TranscribeResult(
            text="Hello world",
            language="en",
            duration=10.0,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "Hello"},
                {"start": 5.0, "end": 10.0, "text": "world"},
            ],
        )
        srt = result.to_srt()
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:05,000" in srt
        assert "Hello" in srt
        assert "2\n" in srt

    def test_to_vtt(self):
        result = TranscribeResult(
            text="Hello world",
            language="en",
            duration=10.0,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "Hello"},
            ],
        )
        vtt = result.to_vtt()
        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:05.000" in vtt
        assert "Hello" in vtt

    def test_empty_segments(self):
        result = TranscribeResult(
            text="", language="en", duration=0, segments=[]
        )
        assert result.segments_text == ""
        assert result.to_srt() == ""
        assert "WEBVTT" in result.to_vtt()


class TestTranscribeConfig:
    """Test TranscribeConfig defaults."""

    def test_defaults(self):
        config = TranscribeConfig()
        assert config.model == "whisper-1"
        assert config.language is None
        assert config.temperature == 0.0

    def test_custom(self):
        config = TranscribeConfig(model="whisper-1", language="es", temperature=0.5)
        assert config.model == "whisper-1"
        assert config.language == "es"
        assert config.temperature == 0.5


class TestNeedsChunking:
    """Test file size check for chunking."""

    def test_small_file(self, tmp_path):
        small_file = tmp_path / "small.mp3"
        small_file.write_bytes(b"x" * 1000)
        assert not _needs_chunking(small_file, 25 * 1024 * 1024)

    def test_large_file(self, tmp_path):
        large_file = tmp_path / "large.mp3"
        large_file.write_bytes(b"x" * (25 * 1024 * 1024 + 1))
        assert _needs_chunking(large_file, 25 * 1024 * 1024)

    def test_exact_limit(self, tmp_path):
        exact_file = tmp_path / "exact.mp3"
        exact_file.write_bytes(b"x" * (25 * 1024 * 1024))
        assert not _needs_chunking(exact_file, 25 * 1024 * 1024)


class TestMergeResults:
    """Test transcription result merging."""

    def test_single_result(self):
        results = [{
            "text": "Hello",
            "language": "en",
            "duration": 5.0,
            "segments": [{"start": 0.0, "end": 5.0, "text": "Hello"}],
        }]
        merged = _merge_transcription_results(results, 5.0)
        assert merged.text == "Hello"
        assert merged.language == "en"
        assert len(merged.segments) == 1

    def test_multiple_results_offset(self):
        results = [
            {
                "text": "Part one",
                "language": "en",
                "duration": 5.0,
                "segments": [{"start": 0.0, "end": 5.0, "text": "Part one"}],
            },
            {
                "text": "Part two",
                "language": "en",
                "duration": 5.0,
                "segments": [{"start": 0.0, "end": 5.0, "text": "Part two"}],
            },
        ]
        merged = _merge_transcription_results(results, 10.0)
        assert "Part one" in merged.text
        assert "Part two" in merged.text
        assert len(merged.segments) == 2
        # Second segment should have time offset
        assert merged.segments[1]["start"] == 5.0
        assert merged.segments[1]["end"] == 10.0

    def test_words_merge(self):
        results = [
            {
                "text": "Hello",
                "language": "en",
                "duration": 2.0,
                "segments": [],
                "words": [{"word": "Hello", "start": 0.0, "end": 1.0}],
            },
            {
                "text": "world",
                "language": "en",
                "duration": 2.0,
                "segments": [],
                "words": [{"word": "world", "start": 0.0, "end": 1.0}],
            },
        ]
        merged = _merge_transcription_results(results, 4.0)
        assert merged.words is not None
        assert len(merged.words) == 2
        assert merged.words[1]["start"] == 2.0  # Offset applied
