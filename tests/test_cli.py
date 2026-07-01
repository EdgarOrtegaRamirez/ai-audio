"""Tests for the CLI interface."""

import pytest
from click.testing import CliRunner

from ai_audio.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    """Test CLI commands."""

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "ai-audio" in result.output

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Text-to-speech" in result.output

    def test_speak_help(self, runner):
        result = runner.invoke(main, ["speak", "--help"])
        assert result.exit_code == 0
        assert "voice" in result.output.lower()

    def test_transcribe_help(self, runner):
        result = runner.invoke(main, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "language" in result.output.lower()

    def test_voices_help(self, runner):
        result = runner.invoke(main, ["voices", "--help"])
        assert result.exit_code == 0

    def test_info_nonexistent(self, runner):
        result = runner.invoke(main, ["info", "nonexistent.mp3"])
        assert result.exit_code == 1

    def test_transcribe_nonexistent(self, runner):
        result = runner.invoke(main, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code == 1

    def test_speak_empty_stdin(self, runner):
        """Test speak with empty stdin."""
        result = runner.invoke(main, ["speak"], input="")
        # Should fail with no text
        assert result.exit_code == 1
