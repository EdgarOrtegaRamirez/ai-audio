"""Tests for audio utility functions."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_audio.audio_utils import (
    AudioInfo,
    detect_audio_format,
    split_audio_sync,
)


class TestAudioInfo:
    """Test AudioInfo dataclass."""

    def test_duration_human_seconds(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=45.0,
                         sample_rate=44100, channels=2, bitrate=128000,
                         codec="mp3", file_size_bytes=1000)
        assert info.duration_human == "45s"

    def test_duration_human_minutes(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=125.0,
                         sample_rate=44100, channels=2, bitrate=128000,
                         codec="mp3", file_size_bytes=1000)
        assert info.duration_human == "2m 5s"

    def test_duration_human_hours(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=3661.0,
                         sample_rate=44100, channels=2, bitrate=128000,
                         codec="mp3", file_size_bytes=1000)
        assert info.duration_human == "1h 1m 1s"

    def test_file_size_human_bytes(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=0,
                         sample_rate=0, channels=0, bitrate=0,
                         codec="mp3", file_size_bytes=500)
        assert info.file_size_human == "500.0B"

    def test_file_size_human_kb(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=0,
                         sample_rate=0, channels=0, bitrate=0,
                         codec="mp3", file_size_bytes=1500)
        assert "KB" in info.file_size_human

    def test_file_size_human_mb(self):
        info = AudioInfo(path="test.mp3", format="mp3", duration_seconds=0,
                         sample_rate=0, channels=0, bitrate=0,
                         codec="mp3", file_size_bytes=5 * 1024 * 1024)
        assert "MB" in info.file_size_human


class TestDetectAudioFormat:
    """Test format detection."""

    def test_mp3(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"\xff\xfb\x90\x00")  # MP3 header bytes
        assert detect_audio_format(f) == "mp3"

    def test_wav(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF")
        assert detect_audio_format(f) == "wav"

    def test_ogg(self, tmp_path):
        f = tmp_path / "test.ogg"
        f.write_bytes(b"OggS")
        assert detect_audio_format(f) == "ogg"

    def test_flac(self, tmp_path):
        f = tmp_path / "test.flac"
        f.write_bytes(b"fLaC")
        assert detect_audio_format(f) == "flac"

    def test_unknown_ext(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_bytes(b"")
        # Falls back to unknown
        result = detect_audio_format(f)
        assert result == "unknown"


class TestSplitAudioSync:
    """Test audio splitting."""

    @patch("ai_audio.audio_utils.get_audio_info")
    def test_short_file_no_split(self, mock_info, tmp_path):
        """Short file should not be split."""
        mock_info.return_value = AudioInfo(
            path="short.mp3", format="mp3", duration_seconds=10.0,
            sample_rate=44100, channels=2, bitrate=128000,
            codec="mp3", file_size_bytes=1000,
        )
        audio = tmp_path / "short.mp3"
        audio.write_bytes(b"fake audio data")
        result = split_audio_sync(audio, chunk_duration=300)
        assert len(result) == 1
        assert result[0] == audio

    @patch("ai_audio.audio_utils.get_audio_info")
    @patch("ai_audio.audio_utils.subprocess.run")
    def test_long_file_split(self, mock_run, mock_info, tmp_path):
        """Long file should be split into chunks."""
        mock_info.return_value = AudioInfo(
            path="test.mp3", format="mp3", duration_seconds=700.0,
            sample_rate=44100, channels=2, bitrate=128000,
            codec="mp3", file_size_bytes=10000000,
        )
        mock_run.return_value = MagicMock(returncode=0)

        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"fake audio data")
        result = split_audio_sync(audio, chunk_duration=300)

        assert len(result) == 3  # 700s / 300s = 3 chunks
        assert all(str(p).endswith(".mp3") for p in result)
        # Verify ffmpeg was called for each chunk
        assert mock_run.call_count == 3
