# Zero-Shot Voice Cloning

Clone any voice with just 3-10 seconds of reference audio.

## Overview

Audar-TTS uses in-context learning to clone voices without fine-tuning. The model learns voice characteristics from reference audio codes in the prompt.

<p align="center">
  <img src="../assets/voice_cloning_overview.png" alt="Voice Cloning" width="100%"/>
</p>

## How It Works

1. **Reference Audio Encoding** — Convert 3-10s audio to discrete codes
2. **Speaker Profile Caching** — Store codes with MD5-based invalidation
3. **In-Context Learning** — LLM sees ref_codes, generates target in same voice
4. **Audio Synthesis** — Decode generated codes to waveform

## Quick Start

```python
from audar_tts import AudarTTS

tts = AudarTTS()

# Clone a voice
speaker = tts.clone_voice(
    name="my_voice",
    audio_path="reference.wav",      # 3-10 seconds
    transcript="Text spoken in audio" # Optional, improves quality
)

# Use the cloned voice
audio, metrics = tts.speak(
    "Hello in my cloned voice!",
    speaker="my_voice"
)
```

## Requirements

| Requirement | Value |
|-------------|-------|
| Audio Duration | 3-10 seconds |
| Audio Quality | Clean, single speaker |
| Sample Rate | 16kHz+ |
| Format | WAV, MP3, FLAC |

## Voice Profile Structure

```python
Speaker(
    name="my_voice",
    audio_path="/path/to/reference.wav",
    transcript="What was said",
    phonemes="/fəˈnɛtɪk/",
    codes=[127, 4521, 892, ...],  # Discrete audio codes
    lang="en",
    duration=5.2
)
```

## Caching

Speaker profiles are automatically cached:

```
.cache/speakers/
├── my_voice_a1b2c3d4.json    # MD5 hash of audio file
└── ...
```

Cache invalidates when audio file changes (MD5 mismatch).

## Best Practices

### Reference Audio Quality

- **Clean recording** — Minimal background noise
- **Single speaker** — No overlapping voices
- **Natural speech** — Not reading, conversational preferred
- **Consistent volume** — No sudden loud/quiet sections

### Duration Guidelines

| Duration | Quality |
|----------|---------|
| < 3s | Poor — insufficient voice data |
| 3-5s | Good — recommended minimum |
| 5-10s | Best — optimal for most voices |
| > 10s | Diminishing returns |

### Transcript Impact

Providing a transcript improves quality by:
- Enabling better phoneme alignment
- Improving reference code segmentation
- Reducing hallucination

## Bilingual Cloning

Clone voices for both English and Arabic:

```python
# Clone with English reference
tts.clone_voice("speaker_en", "english_ref.wav", "Hello there")

# Clone with Arabic reference  
tts.clone_voice("speaker_ar", "arabic_ref.wav", "مرحبا بكم")

# Use either for any language (code-switching)
tts.speak("Welcome مرحبا to Audar", speaker="speaker_en")
```

## Technical Details

### Voice Encoding

```
Reference Audio (3-10s WAV)
    ↓
DistillNeuCodec Encoder (50Hz)
    ↓
Discrete Codes [0-8191]
    ↓
Speaker Profile Cache
```

### In-Context Cloning

The LLM learns voice characteristics via prompt conditioning:

```
[REF_SPEECH] speaker_codes  → Voice pattern
[TARGET_TEXT] phonemes      → What to say
[TARGET_CODES] ???          → Generate in same voice
```

The model sees reference codes first, then generates target codes that match the voice style.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Voice sounds different | Use longer reference (5-10s) |
| Pronunciation errors | Provide accurate transcript |
| Robotic output | Check audio quality, reduce noise |
| Cache not updating | Delete `.cache/speakers/` directory |
