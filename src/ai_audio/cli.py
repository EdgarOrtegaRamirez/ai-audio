"""CLI interface for ai-audio: text-to-speech and speech-to-text."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from . import __version__
from .tts import speak, list_voices, TTSConfig, VOICE_PRESETS
from .stt import transcribe, TranscribeConfig, TranscribeResult
from .audio_utils import get_audio_info, detect_audio_format

console = Console()


def _run_async(coro):
    """Run an async function from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _read_stdin_or_text(text: str | None, prompt: str = "Enter text: ") -> str:
    """Read text from argument, stdin, or interactive prompt."""
    if text:
        return text
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return click.prompt(prompt, type=str)


@click.group()
@click.version_option(__version__, prog_name="ai-audio")
def main():
    """Text-to-speech and speech-to-text from your terminal."""
    pass


# ─── SPEAK ────────────────────────────────────────────────────────────
@main.command()
@click.argument("text", required=False, default=None)
@click.option("-v", "--voice", default="en-US-AriaNeural", help="Voice name (e.g. en-US-GuyNeural)")
@click.option("-r", "--rate", default="+0%", help="Speech rate (e.g. +20%%, -10%%)")
@click.option("-p", "--pitch", default="+0Hz", help="Pitch adjustment (e.g. +5Hz)")
@click.option("-l", "--volume", default="+0%", help="Volume adjustment (e.g. +10%%)")
@click.option("-o", "--output", default=None, help="Output file path (default: speech.mp3)")
@click.option("-f", "--format", "output_format", default="mp3", help="Output format (mp3, wav, ogg, flac)")
@click.option("--list-voices", "list_voices_flag", is_flag=True, help="List available voices")
@click.option("--lang", default=None, help="Filter voices by language (en, es, fr, de, pt, ja, ko, zh, hi)")
def speak_cmd(text, voice, rate, pitch, volume, output, output_format, list_voices_flag, lang):
    """Convert text to speech.

    Examples:
        ai audio speak "Hello world"
        ai audio speak "Hola mundo" --voice es-CO-SalomeNeural
        ai audio speak -o output.wav -f wav "Some text"
        ai audio speak --list-voices --lang es
        echo "Hello" | ai audio speak
        ai audio speak  # interactive prompt
    """
    if list_voices_flag:
        _show_voices(lang)
        return

    # Show voice presets
    if voice == "list":
        _show_voices(lang)
        return

    text_content = _read_stdin_or_text(text)
    if not text_content:
        console.print("[red]No text provided. Use 'ai audio speak --list-voices' to see available voices.[/red]")
        raise SystemExit(1)

    # Determine output path
    if output is None:
        output = f"speech.{output_format}"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Synthesizing with voice [cyan]{voice}[/cyan]...[/dim]")
    console.print(f"[dim]Text ({len(text_content)} chars): {text_content[:80]}{'...' if len(text_content) > 80 else ''}[/dim]")

    try:
        result = _run_async(speak(
            text=text_content,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
            output=str(output_path),
            output_format=output_format,
        ))
        info = get_audio_info(result)
        console.print(f"\n[green]✓ Saved to {result}[/green]")
        console.print(f"[dim]  Format: {info.format} | Duration: {info.duration_human} | Size: {info.file_size_human}[/dim]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)


def _show_voices(lang: str | None = None):
    """Display available voices in a table."""
    voices = _run_async(list_voices(lang))

    table = Table(title="Available Voices", box=box.ROUNDED)
    table.add_column("Voice", style="cyan")
    table.add_column("Language", style="green")
    table.add_column("Gender")
    table.add_column("Locale")

    for v in voices:
        name = v.get("ShortName", "unknown")
        locale = v.get("Locale", "unknown")
        gender = v.get("Gender", "unknown")
        lang_name = v.get("LocaleName", locale)
        table.add_row(name, lang_name, gender, locale)

    console.print(table)
    console.print(f"\n[dim]Total: {len(voices)} voices. Use --lang to filter (e.g. --lang es)[/dim]")


# ─── TRANSCRIBE ──────────────────────────────────────────────────────
@main.command()
@click.argument("audio_file", required=False, default=None)
@click.option("-l", "--language", default=None, help="Language code (e.g. en, es, fr) — auto-detect if omitted")
@click.option("-m", "--model", default="whisper-1", help="Whisper model to use")
@click.option("-o", "--output", default=None, help="Output file (default: stdout)")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json", "srt", "vtt", "verbose"]), help="Output format")
@click.option("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
@click.option("--base-url", default=None, help="Custom API base URL")
@click.option("--prompt", default=None, help="Context prompt for better accuracy")
@click.option("--temperature", default=0.0, type=float, help="Temperature (0-1)")
def transcribe_cmd(audio_file, language, model, output, output_format, api_key, base_url, prompt, temperature):
    """Transcribe speech to text.

    Examples:
        ai audio transcribe recording.mp3
        ai audio transcribe meeting.wav --language es
        ai audio transcribe podcast.m4a --format srt -o subtitles.srt
        cat audio.ogg | ai audio transcribe -
        ai audio transcribe --format json recording.mp3
    """
    # Read from stdin or file
    if audio_file == "-" or (audio_file is None and not sys.stdin.isatty()):
        # Save stdin to temp file
        import tempfile
        stdin_data = sys.stdin.buffer.read()
        if not stdin_data:
            console.print("[red]No audio data on stdin[/red]")
            raise SystemExit(1)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(stdin_data)
            audio_file = tmp.name
        console.print(f"[dim]Read {len(stdin_data)} bytes from stdin[/dim]")

    if audio_file is None:
        console.print("[red]No audio file specified. Provide a file path or pipe audio to stdin.[/red]")
        raise SystemExit(1)

    audio_path = Path(audio_file)
    if not audio_path.exists():
        console.print(f"[red]File not found: {audio_path}[/red]")
        raise SystemExit(1)

    # Show file info
    try:
        info = get_audio_info(audio_path)
        console.print(f"[dim]Transcribing: {info.path} | Format: {info.format} | Duration: {info.duration_human} | Size: {info.file_size_human}[/dim]")
    except Exception:
        console.print(f"[dim]Transcribing: {audio_path}[/dim]")

    config = TranscribeConfig(
        model=model,
        language=language,
        response_format="verbose_json",
        temperature=temperature,
        prompt=prompt,
        api_key=api_key,
        base_url=base_url,
    )

    try:
        result = _run_async(transcribe(str(audio_path), config))
        _output_result(result, output_format, output)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Transcription error: {e}[/red]")
        raise SystemExit(1)


def _output_result(result: TranscribeResult, fmt: str, output_file: str | None):
    """Output transcription result in the specified format."""
    import json

    if fmt == "text":
        content = result.text
    elif fmt == "json":
        data = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
        }
        if result.words:
            data["words"] = result.words
        content = json.dumps(data, indent=2, ensure_ascii=False)
    elif fmt == "verbose":
        content = result.segments_text
    elif fmt == "srt":
        content = result.to_srt()
    elif fmt == "vtt":
        content = result.to_vtt()
    else:
        content = result.text

    if output_file:
        Path(output_file).write_text(content, encoding="utf-8")
        console.print(f"[green]✓ Saved to {output_file}[/green]")
    else:
        console.print(content)

    # Stats
    console.print(f"\n[dim]Language: {result.language} | Duration: {result.duration:.1f}s | Segments: {len(result.segments)}[/dim]")


# ─── VOICES ──────────────────────────────────────────────────────────
@main.command("voices")
@click.option("-l", "--lang", default=None, help="Filter by language code")
def voices_cmd(lang):
    """List available TTS voices."""
    _show_voices(lang)


# ─── INFO ─────────────────────────────────────────────────────────────
@main.command()
@click.argument("audio_file")
def info_cmd(audio_file):
    """Show audio file metadata."""
    try:
        info = get_audio_info(audio_file)
        table = Table(title=f"Audio Info: {info.path}", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Format", info.format)
        table.add_row("Codec", info.codec)
        table.add_row("Duration", info.duration_human)
        table.add_row("Sample Rate", f"{info.sample_rate} Hz")
        table.add_row("Channels", str(info.channels))
        table.add_row("Bitrate", f"{info.bitrate // 1000} kbps")
        table.add_row("File Size", info.file_size_human)
        console.print(table)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


# ─── FORMAT ───────────────────────────────────────────────────────────
@main.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("-f", "--format", "output_format", default="mp3", help="Target format")
@click.option("--normalize", is_flag=True, help="Normalize audio volume")
@click.option("--sample-rate", default=None, type=int, help="Target sample rate")
@click.option("--bitrate", default=None, help="Target bitrate (e.g. 192k)")
def convert_cmd(input_file, output_file, output_format, normalize, sample_rate, bitrate):
    """Convert audio between formats."""
    from .audio_utils import convert_audio

    async def _convert():
        return await convert_audio(
            input_file, output_file,
            output_format=output_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            normalize=normalize,
        )

    try:
        result = _run_async(_convert())
        info = get_audio_info(result)
        console.print(f"[green]✓ Converted to {result}[/green]")
        console.print(f"[dim]  Format: {info.format} | Duration: {info.duration_human} | Size: {info.file_size_human}[/dim]")
    except Exception as e:
        console.print(f"[red]Conversion error: {e}[/red]")
        raise SystemExit(1)


# ─── ALIASES ──────────────────────────────────────────────────────────
@main.command("speak-alias", hidden=True)
def speak_alias():
    """Hidden alias for backwards compatibility."""
    pass


if __name__ == "__main__":
    main()
