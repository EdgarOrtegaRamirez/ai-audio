# ai-audio

Run `ai audio speak <prompt>` and `ai audio transcribe <file>` from your terminal.

## Commands

- `ai-audio speak "text"` — Text-to-speech (Edge TTS, free)
- `ai-audio transcribe file.mp3` — Speech-to-text (Whisper API)
- `ai-audio voices` — List available voices
- `ai-audio info file.mp3` — Show audio metadata
- `ai-audio convert in.wav out.mp3` — Format conversion

## Setup

```bash
pip install ai-audio
pip install 'ai-audio[transcribe]'  # for STT
```

## Testing

```bash
python -m pytest tests/ -v
```
