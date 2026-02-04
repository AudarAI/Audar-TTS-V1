<h1 align="center">Audar-TTS</h1>

<p align="center">
  <strong>Lightning-Fast Zero-Shot Voice Cloning</strong><br/>
  Clone any voice with 3 seconds of audio · 2x faster than real-time · Bilingual EN/AR
</p>

<p align="center">
  <a href="https://huggingface.co/audarai/audar_tts_flash_v1_gguf">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/RTF-0.46x-brightgreen.svg" alt="RTF"/>
  <img src="https://img.shields.io/badge/latency-500ms-orange.svg" alt="Latency"/>
</p>

---

## Architecture

<p align="center">
  <img src="assets/audar_tts_diagrams.png?v=2" alt="Audar-TTS Architecture" width="100%"/>
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| **Zero-Shot Cloning** | Clone any voice without fine-tuning |
| **Real-Time Speed** | 0.46x RTF (2.2x faster than playback) |
| **Low Latency** | ~500ms to first audio chunk |
| **Streaming** | Generate audio progressively |
| **Bilingual** | English + Arabic with code-switching |
| **Speaker Caching** | Instant reuse of encoded voices |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AudarAI/Audar-TTS-V1.git
cd Audar-TTS-V1

# Install dependencies
pip install -r requirements.txt

# System dependency (for phonemization)
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt-get install espeak-ng
```

---

## Quick Start

### Python API

```python
from audar_tts import AudarTTS

# Initialize (auto-downloads model from HuggingFace)
tts = AudarTTS()

# Basic synthesis
audio, metrics = tts.speak(
    "Hello! Welcome to Audar TTS.",
    speaker="Eve",
    output="hello.wav"
)

# Streaming (low latency)
for chunk, info in tts.stream("Long text here...", speaker="Eve"):
    play_audio(chunk)  # Play each ~1s chunk

# Clone a new voice
tts.clone_voice(
    name="my_voice",
    audio_path="reference.wav",
    transcript="What I said in the recording"
)
tts.speak("Now in cloned voice!", speaker="my_voice")
```

### Command Line

```bash
# Basic synthesis
python audar_tts.py "Hello world!" -s Eve -o output.wav

# Streaming mode
python audar_tts.py "Long text here" -s Eve --stream

# Arabic text
python audar_tts.py "مرحبا بالعالم" -s Eve --lang ar

# List voices
python audar_tts.py --list-voices
```

---

## API Reference

### AudarTTS Class

```python
AudarTTS(
    model_path=None,      # Path to GGUF (auto-downloads if None)
    voices_dir=None,      # Voice profiles directory
    n_ctx=8192,           # Context window
    verbose=False
)
```

### Methods

| Method | Description |
|--------|-------------|
| `speak(text, speaker, output, lang)` | Synthesize speech → `(audio, metrics)` |
| `stream(text, speaker, lang)` | Stream synthesis → Generator |
| `clone_voice(name, audio_path, transcript)` | Clone voice → `Speaker` |
| `voices` | List available voices |

---

## Performance

| Metric | Value |
|--------|-------|
| Real-Time Factor | 0.46x |
| First Chunk Latency | ~500ms |
| Model Size | ~800MB |
| Sample Rate | 24kHz |

---

## TTS Comparison

Comparison of Audar TTS Flash V1 against ElevenLabs Flash across different scenarios.

| Scenario | Speaker | Audar TTS Flash V1 | ElevenLabs Flash |
|----------|---------|:------------------:|:----------------:|
| English | Eve | :white_check_mark: | :white_check_mark: |
| Arabic | Salama | :white_check_mark: | :white_check_mark: |
| Mixed EN/AR | Amal | :white_check_mark: | :white_check_mark: |
| `[laughs]` | Eve | :white_check_mark: | :x: |
| `[whispers]` | Eve | :white_check_mark: | :x: |
| `[excited]` | Eve | :white_check_mark: | :x: |
| `[sarcastic]` | Eve | :white_check_mark: | :x: |

### Key Differentiators

| Feature | Audar TTS Flash V1 | ElevenLabs Flash |
|---------|:------------------:|:----------------:|
| Expressive Tags | :white_check_mark: | :x: |
| Code-Switching (EN/AR) | :white_check_mark: | :white_check_mark: |
| Zero-Shot Cloning | :white_check_mark: | :white_check_mark: |
| On-Premise Deployment | :white_check_mark: | :x: |
| Open Source | :white_check_mark: | :x: |

### Available Speakers

| Name | Language | Style |
|------|----------|-------|
| Eve | EN/AR | Clear, expressive |
| Salama | EN/AR | Calm, peaceful |
| Amal | EN/AR | Warm, hopeful |
| Hanaa | EN/AR | Bright, happy |
| Salem | EN/AR | Professional |
| Amin | EN/AR | Trustworthy |
| Wadee | EN/AR | Gentle, soft |

---

## Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** — Detailed architecture and implementation
- **[docs/streaming.md](docs/streaming.md)** — Streaming synthesis guide
- **[docs/voice-cloning.md](docs/voice-cloning.md)** — Voice cloning deep dive

---

## License

Apache 2.0 — See [LICENSE](LICENSE)

---

<p align="center">
  Built with ❤️ by <a href="https://audar.ai">Audar AI</a>
</p>
