"""Audio utility functions for format detection, normalization, and processing."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioInfo:
    """Metadata about an audio file."""

    path: str
    format: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bitrate: int
    codec: str
    file_size_bytes: int

    @property
    def duration_human(self) -> str:
        mins, secs = divmod(int(self.duration_seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h {mins}m {secs}s"
        if mins:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    @property
    def file_size_human(self) -> str:
        size = self.file_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


def get_audio_info(filepath: str | Path) -> AudioInfo:
    """Get audio file metadata using ffprobe."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(filepath),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), streams[0] if streams else {})

    return AudioInfo(
        path=str(filepath),
        format=fmt.get("format_name", filepath.suffix.lstrip(".")),
        duration_seconds=float(fmt.get("duration", 0)),
        sample_rate=int(audio_stream.get("sample_rate", 0)),
        channels=int(audio_stream.get("channels", 0)),
        bitrate=int(fmt.get("bit_rate", 0)),
        codec=audio_stream.get("codec_name", "unknown"),
        file_size_bytes=filepath.stat().st_size,
    )


async def convert_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_format: str = "mp3",
    sample_rate: int | None = None,
    channels: int | None = None,
    bitrate: str | None = None,
    normalize: bool = False,
) -> Path:
    """Convert audio between formats using ffmpeg."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    if channels:
        cmd.extend(["-ac", str(channels)])
    if bitrate:
        cmd.extend(["-b:a", bitrate])
    if normalize:
        cmd.extend(["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"])

    cmd.append(str(output_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    return output_path


async def normalize_audio(input_path: str | Path, output_path: str | Path) -> Path:
    """Normalize audio volume using EBU R128 loudness normalization."""
    return await convert_audio(input_path, output_path, normalize=True)


async def extract_segment(
    input_path: str | Path,
    output_path: str | Path,
    start: float,
    end: float,
) -> Path:
    """Extract a time segment from an audio file."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-c",
        "copy",
        str(output_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg segment extraction failed")
    return output_path


def split_audio_sync(input_path: str | Path, chunk_duration: float = 300.0) -> list[Path]:
    """Split audio into chunks of chunk_duration seconds. Synchronous version."""
    input_path = Path(input_path)
    info = get_audio_info(input_path)

    if info.duration_seconds <= chunk_duration:
        return [input_path]

    chunks = []
    start = 0.0
    idx = 0
    while start < info.duration_seconds:
        end = min(start + chunk_duration, info.duration_seconds)
        chunk_path = input_path.parent / f"{input_path.stem}_chunk_{idx:03d}{input_path.suffix}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            str(start),
            "-t",
            str(end - start),
            "-c",
            "copy",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append(chunk_path)
        start = end
        idx += 1

    return chunks


def detect_audio_format(filepath: str | Path) -> str:
    """Detect audio format from file extension or content."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower().lstrip(".")
    format_map = {
        "mp3": "mp3",
        "wav": "wav",
        "ogg": "ogg",
        "flac": "flac",
        "m4a": "m4a",
        "aac": "aac",
        "wma": "wma",
        "opus": "opus",
        "webm": "webm",
        "mp4": "mp4",
    }
    if ext in format_map:
        return format_map[ext]

    # Fallback to ffprobe
    try:
        info = get_audio_info(filepath)
        return info.format.split(",")[0]
    except Exception:
        return "unknown"
