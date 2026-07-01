"""Text-to-speech engine using Microsoft Edge TTS."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

import edge_tts

from .audio_utils import convert_audio


@dataclass
class TTSConfig:
    """Configuration for text-to-speech synthesis."""
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    output_format: str = "mp3"
    output_dir: str = "."

    # Text chunking
    max_chunk_chars: int = 4000  # Edge TTS limit per request
    sentence_pause_ms: int = 200
    paragraph_pause_ms: int = 500


# Popular voices by language
VOICE_PRESETS: dict[str, list[str]] = {
    "en": [
        "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
        "en-US-AndrewNeural", "en-US-BrianNeural", "en-US-EmmaNeural",
        "en-GB-SoniaNeural", "en-GB-RyanNeural",
        "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    ],
    "es": [
        "es-ES-ElviraNeural", "es-ES-AlvaroNeural",
        "es-MX-DaliaNeural", "es-MX-JorgeNeural",
        "es-CO-SalomeNeural", "es-CO-GonzaloNeural",
    ],
    "fr": [
        "fr-FR-DeniseNeural", "fr-FR-HenriNeural",
        "fr-CA-SylvieNeural", "fr-CA-JeanNeural",
    ],
    "de": [
        "de-DE-KatjaNeural", "de-DE-ConradNeural",
    ],
    "pt": [
        "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural",
        "pt-PT-RaquelNeural", "pt-PT-DuarteNeural",
    ],
    "ja": [
        "ja-JP-NanamiNeural", "ja-JP-KeitaNeural",
    ],
    "ko": [
        "ko-KR-SunHiNeural", "ko-KR-InJoonNeural",
    ],
    "zh": [
        "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
        "zh-CN-YunjianNeural",
    ],
    "hi": [
        "hi-IN-SwaraNeural", "hi-IN-MadhurNeural",
    ],
    "ar": [
        "ar-SA-ZariyahNeural", "ar-SA-HamedNeural",
    ],
    "it": [
        "it-IT-ElsaNeural", "it-IT-DiegoNeural",
    ],
    "ru": [
        "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural",
    ],
    "nl": [
        "nl-NL-ColetteNeural", "nl-NL-MaartenNeural",
    ],
    "pl": [
        "pl-PL-AgnieszkaNeural", "pl-PL-MarekNeural",
    ],
}


async def list_voices(language: str | None = None) -> list[dict]:
    """List available Edge TTS voices, optionally filtered by language."""
    voices = await edge_tts.list_voices()
    if language:
        lang = language.lower()
        voices = [v for v in voices if v.get("Locale", "").startswith(lang)]
    return voices


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks for TTS processing, respecting sentence boundaries.

    Uses a greedy algorithm that:
    1. Splits on paragraph boundaries first
    2. Then on sentence boundaries
    3. Then on clause boundaries (commas, semicolons)
    4. Finally on word boundaries as a last resort
    """
    if not text.strip():
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Find the best split point
        segment = remaining[:max_chars]

        # Priority 1: paragraph boundary
        split_pos = segment.rfind("\n\n")
        if split_pos > max_chars * 0.3:
            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()
            continue

        # Priority 2: sentence boundary (. ! ?)
        split_pos = max(
            segment.rfind(". "),
            segment.rfind("! "),
            segment.rfind("? "),
            segment.rfind(".\n"),
        )
        if split_pos > max_chars * 0.3:
            chunks.append(remaining[: split_pos + 1].strip())
            remaining = remaining[split_pos + 1 :].strip()
            continue

        # Priority 3: clause boundary (; , —)
        split_pos = max(
            segment.rfind("; "),
            segment.rfind(", "),
            segment.rfind(" — "),
            segment.rfind(" - "),
        )
        if split_pos > max_chars * 0.3:
            chunks.append(remaining[:split_pos + 1].strip())
            remaining = remaining[split_pos + 1 :].strip()
            continue

        # Priority 4: word boundary
        split_pos = segment.rfind(" ")
        if split_pos > max_chars * 0.3:
            chunks.append(remaining[:split_pos].strip())
            remaining = remaining[split_pos:].strip()
            continue

        # Last resort: hard split
        chunks.append(remaining[:max_chars].strip())
        remaining = remaining[max_chars:].strip()

    return [c for c in chunks if c.strip()]


async def synthesize_chunk(text: str, config: TTSConfig, output_path: Path) -> Path:
    """Synthesize a single text chunk to audio."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=config.voice,
        rate=config.rate,
        pitch=config.pitch,
        volume=config.volume,
    )

    tmp_path = output_path.with_suffix(".tmp.mp3")
    await communicate.save(str(tmp_path))

    # Convert if needed
    if config.output_format != "mp3":
        final_path = output_path.with_suffix(f".{config.output_format}")
        await convert_audio(tmp_path, final_path, output_format=config.output_format)
        tmp_path.unlink()
        return final_path

    tmp_path.rename(output_path)
    return output_path


async def synthesize(
    text: str,
    config: TTSConfig | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Synthesize text to speech.

    Handles text chunking, multi-part synthesis, and audio concatenation.
    """
    if config is None:
        config = TTSConfig()

    if not text.strip():
        raise ValueError("Text cannot be empty")

    # Determine output path
    if output_path is None:
        output_path = Path(config.output_dir) / f"speech.{config.output_format}"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunk text
    chunks = chunk_text(text, config.max_chunk_chars)

    if len(chunks) == 1:
        return await synthesize_chunk(chunks[0], config, output_path)

    # Multi-chunk: synthesize each, then concatenate
    import tempfile

    chunk_paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, chunk in enumerate(chunks):
            chunk_path = Path(tmp_dir) / f"chunk_{i:03d}.mp3"
            await synthesize_chunk(chunk, config, chunk_path)
            chunk_paths.append(chunk_path)

        # Concatenate with ffmpeg
        concat_file = Path(tmp_dir) / "concat.txt"
        with open(concat_file, "w") as f:
            for cp in chunk_paths:
                f.write(f"file '{cp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("Failed to concatenate audio chunks")

    return output_path


async def speak(
    text: str,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    output: str | None = None,
    output_format: str = "mp3",
) -> Path:
    """High-level speak function."""
    config = TTSConfig(
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
        output_format=output_format,
    )
    return await synthesize(text, config, output)
