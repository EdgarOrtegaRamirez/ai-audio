"""Speech-to-text engine using OpenAI Whisper API with intelligent chunking."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path

from .audio_utils import convert_audio, get_audio_info, split_audio_sync

# Whisper API limit: 25MB per request
WHISPER_MAX_FILE_SIZE_MB = 24
WHISPER_MAX_FILE_SIZE = WHISPER_MAX_FILE_SIZE_MB * 1024 * 1024
WHISPER_CHUNK_DURATION = 600.0  # 10 minutes per chunk


@dataclass
class TranscribeConfig:
    """Configuration for speech-to-text transcription."""

    model: str = "whisper-1"
    language: str | None = None  # Auto-detect if None
    response_format: str = "verbose_json"  # json, verbose_json, text, srt, vtt
    temperature: float = 0.0
    prompt: str | None = None  # Context prompt for better accuracy
    timestamp_granularities: list[str] = field(default_factory=lambda: ["segment"])

    # Chunking
    chunk_duration: float = WHISPER_CHUNK_DURATION
    max_file_size: int = WHISPER_MAX_FILE_SIZE

    # API
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class TranscribeResult:
    """Result of a transcription."""

    text: str
    language: str
    duration: float
    segments: list[dict]
    words: list[dict] | None = None

    @property
    def segments_text(self) -> str:
        lines = []
        for seg in self.segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """Convert segments to SRT subtitle format."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lines.append(f"{i}")
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self) -> str:
        """Convert segments to WebVTT format."""
        lines = ["WEBVTT", ""]
        for seg in self.segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lines.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _get_openai_client(config: TranscribeConfig):
    """Create OpenAI client with proper API key and base URL."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required for transcription. Install with: pip install 'ai-audio[transcribe]'"
        ) from e

    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass --api-key")

    kwargs = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return OpenAI(**kwargs)


def _needs_chunking(filepath: Path, max_size: int) -> bool:
    """Check if file needs to be chunked for the API."""
    return filepath.stat().st_size > max_size


def _transcribe_chunk_sync(
    client,
    chunk_path: Path,
    config: TranscribeConfig,
) -> dict:
    """Transcribe a single audio chunk."""
    with open(chunk_path, "rb") as f:
        kwargs = {
            "model": config.model,
            "file": f,
            "response_format": config.response_format,
            "temperature": config.temperature,
        }
        if config.language:
            kwargs["language"] = config.language
        if config.prompt:
            kwargs["prompt"] = config.prompt
        if "timestamp_granularities" in config.response_format:
            kwargs["timestamp_granularities"] = config.timestamp_granularities

        result = client.audio.transcriptions.create(**kwargs)

    # Convert to dict
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    # Fallback for string response
    return {"text": str(result), "language": config.language or "en", "duration": 0, "segments": []}


def _merge_transcription_results(results: list[dict], total_duration: float) -> TranscribeResult:
    """Merge multiple chunk transcription results into one."""
    all_text = []
    all_segments = []
    all_words = []
    time_offset = 0.0
    detected_language = results[0].get("language", "en") if results else "en"

    for result in results:
        text = result.get("text", "")
        all_text.append(text)

        chunk_duration = result.get("duration", 0)

        for seg in result.get("segments", []):
            adjusted_seg = {
                **seg,
                "start": seg.get("start", 0) + time_offset,
                "end": seg.get("end", 0) + time_offset,
            }
            all_segments.append(adjusted_seg)

        for word in result.get("words", []):
            adjusted_word = {
                **word,
                "start": word.get("start", 0) + time_offset,
                "end": word.get("end", 0) + time_offset,
            }
            all_words.append(adjusted_word)

        time_offset += chunk_duration

    return TranscribeResult(
        text=" ".join(all_text).strip(),
        language=detected_language,
        duration=total_duration,
        segments=all_segments,
        words=all_words if all_words else None,
    )


def transcribe_sync(
    filepath: str | Path,
    config: TranscribeConfig | None = None,
) -> TranscribeResult:
    """Synchronous transcription entry point.

    Handles chunking for large files, API calls, and result merging.
    """
    if config is None:
        config = TranscribeConfig()

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    info = get_audio_info(filepath)
    client = _get_openai_client(config)

    # Check if chunking is needed
    file_size = filepath.stat().st_size
    needs_chunk = file_size > config.max_file_size or info.duration_seconds > config.chunk_duration

    if not needs_chunk:
        # Direct transcription
        # Convert to mp3 if needed (Whisper works best with mp3/wav)
        work_path = filepath
        if filepath.suffix.lower() not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"):
            work_path = filepath.with_suffix(".mp3")
            import asyncio

            asyncio.run(convert_audio(filepath, work_path, output_format="mp3"))

        result = _transcribe_chunk_sync(client, work_path, config)
        return TranscribeResult(
            text=result.get("text", ""),
            language=result.get("language", config.language or "en"),
            duration=result.get("duration", info.duration_seconds),
            segments=result.get("segments", []),
            words=result.get("words"),
        )

    # Chunk and transcribe
    chunks = split_audio_sync(filepath, config.chunk_duration)
    chunk_results = []

    for _i, chunk_path in enumerate(chunks):
        # Ensure chunk is in a compatible format
        work_chunk = chunk_path
        if chunk_path.suffix.lower() not in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"):
            work_chunk = chunk_path.with_suffix(".mp3")
            import asyncio

            asyncio.run(convert_audio(chunk_path, work_chunk, output_format="mp3"))

        result = _transcribe_chunk_sync(client, work_chunk, config)
        chunk_results.append(result)

    return _merge_transcription_results(chunk_results, info.duration_seconds)


async def transcribe(
    filepath: str | Path,
    config: TranscribeConfig | None = None,
) -> TranscribeResult:
    """Async transcription entry point."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_sync, filepath, config)
