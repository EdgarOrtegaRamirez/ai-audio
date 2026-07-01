# ai-audio

**Text-to-speech and speech-to-text from your terminal.**

```bash
ai audio speak "Hello world"
ai audio transcribe recording.mp3
```

## Features

- **Text-to-Speech (TTS)** via Microsoft Edge TTS — free, high-quality, 100+ voices
- **Speech-to-Text (STT)** via OpenAI Whisper API — automatic language detection
- Smart text chunking for long documents
- Large file chunking for Whisper's 25MB limit
- Multiple output formats: MP3, WAV, OGG, FLAC
- Subtitle output: SRT, VTT, JSON
- stdin pipe support
- Audio format conversion and normalization

## Install

```bash
pip install ai-audio
```

| Install | Command | `speak` (TTS) | `transcribe` (STT) | API Key Needed |
|---------|---------|:---:|:---:|:---:|
| **Basic** | `pip install ai-audio` | ✅ | ❌ | No |
| **Transcribe** | `pip install 'ai-audio[transcribe]'` | ✅ | ✅ | Yes (OpenAI) |

- **Basic** — `speak` uses Edge TTS (free, no API key, 100+ voices)
- **Transcribe** — adds `transcribe` via OpenAI Whisper API (paid, needs `OPENAI_API_KEY`)

## Usage

### Text-to-Speech

```bash
# Basic speech
ai audio speak "Hello world"

# With specific voice
ai audio speak "Hola mundo" --voice es-CO-SalomeNeural

# Custom rate and pitch
ai audio speak "Fast speech" --rate +50%
ai audio speak "Deep voice" --pitch -5Hz

# Output to file
ai audio speak "Save me" -o output.wav -f wav

# From stdin
echo "Hello from pipe" | ai audio speak
cat story.txt | ai audio speak -o story.mp3

# List voices
ai audio voices
ai audio voices --lang es
ai audio speak --list-voices --lang fr
```

### Speech-to-Text

```bash
# Basic transcription
ai audio transcribe recording.mp3

# With language hint
ai audio transcribe meeting.wav --language es

# Output as SRT subtitles
ai audio transcribe podcast.m4a --format srt -o subtitles.srt

# JSON output
ai audio transcribe interview.mp3 --format json

# From stdin
cat audio.ogg | ai audio transcribe -

# Save to file
ai audio transcribe lecture.wav -o transcript.txt
```

### Audio Info & Conversion

```bash
# Show audio metadata
ai audio info recording.mp3

# Convert formats
ai audio convert input.wav output.mp3
ai audio convert input.mp3 output.ogg --normalize
```

## Available Voices

| Language | Top Voices |
|----------|-----------|
| English  | en-US-AriaNeural, en-US-GuyNeural, en-US-JennyNeural |
| Spanish  | es-ES-ElviraNeural, es-MX-DaliaNeural, es-CO-SalomeNeural |
| French   | fr-FR-DeniseNeural, fr-FR-HenriNeural |
| German   | de-DE-KatjaNeural, de-DE-ConradNeural |
| Portuguese | pt-BR-FranciscaNeural, pt-BR-AntonioNeural |
| Japanese | ja-JP-NanamiNeural, ja-JP-KeitaNeural |
| Korean   | ko-KR-SunHiNeural, ko-KR-InJoonNeural |
| Chinese  | zh-CN-XiaoxiaoNeural, zh-CN-YunxiNeural |

Run `ai audio voices --lang <code>` for full voice lists.

## Configuration

### Environment Variables

```bash
# For transcription
export OPENAI_API_KEY="sk-..."

# Optional: custom API endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Rate/Pitch/Volume Formats

- Rate: `+0%` (normal), `+20%` (faster), `-10%` (slower)
- Pitch: `+0Hz` (normal), `+5Hz` (higher), `-5Hz` (lower)
- Volume: `+0%` (normal), `+10%` (louder), `-10%` (quieter)

## How It Works

### TTS Pipeline
1. Text is chunked at sentence/paragraph boundaries (4000 char limit per chunk)
2. Each chunk is synthesized via Edge TTS (WebSocket API)
3. Chunks are concatenated via ffmpeg
4. Output in requested format

### STT Pipeline
1. Audio file is analyzed (size, duration, format)
2. If >25MB or >10min, split into chunks using ffmpeg
3. Each chunk is sent to OpenAI Whisper API
4. Results are merged with time offset correction
5. Output in requested format (text, JSON, SRT, VTT)

## License

MIT
