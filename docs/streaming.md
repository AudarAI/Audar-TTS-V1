# Streaming Synthesis

Real-time audio generation with low latency.

## Overview

Audar-TTS generates audio in ~1 second chunks, enabling playback to begin before full synthesis completes.

<p align="center">
  <img src="../assets/streaming_pipeline.png" alt="Streaming Pipeline" width="100%"/>
</p>

## Key Metrics

| Metric | Value |
|--------|-------|
| First Chunk Latency | ~500ms |
| Chunk Duration | ~1 second (50 codes) |
| Real-Time Factor | 0.46x |
| Codes per Second | 50 |

## Usage

```python
from audar_tts import AudarTTS

tts = AudarTTS()

# Stream synthesis
for audio_chunk, metrics in tts.stream("Your text here", speaker="Eve"):
    # Each chunk is ~1 second of audio
    # First chunk arrives in ~500ms
    play_audio(audio_chunk)
    
    print(f"Chunk {metrics['chunk']}: {metrics['duration']:.1f}s")
```

## How It Works

1. **Token Generation** — LLM generates speech codes autoregressively
2. **Chunk Buffer** — Accumulates 50 codes (~1 second of audio)
3. **Codec Decode** — Converts codes to waveform immediately
4. **Yield** — Returns audio chunk for playback

## Parameters

```python
tts.stream(
    text,                    # Text to synthesize
    speaker="Eve",           # Voice to use
    lang="en",               # Language (en/ar)
    chunk_size=50            # Codes per chunk (default: 50)
)
```

## Benefits

- **Low Latency** — Playback starts in ~500ms
- **Memory Efficient** — No full-sequence buffering
- **Constant Memory** — Works for any length text
- **Parallel Processing** — Generate while playing

## Implementation Details

The streaming architecture uses a generator pattern:

```python
def stream(self, text, speaker, lang="en", chunk_size=50):
    # Encode speaker and phonemize text
    prompt = self._build_prompt(speaker, phonemes)
    
    codes_buffer = []
    for token in self._llm.generate(prompt):
        if is_speech_token(token):
            codes_buffer.append(extract_code(token))
            
            if len(codes_buffer) >= chunk_size:
                audio = self._codec.decode(codes_buffer)
                yield audio, metrics
                codes_buffer = []
    
    # Flush remaining codes
    if codes_buffer:
        audio = self._codec.decode(codes_buffer)
        yield audio, metrics
```

## Performance Tuning

| Setting | Effect |
|---------|--------|
| `chunk_size=50` | Balance of latency vs. audio quality |
| `chunk_size=25` | Lower latency, more frequent yields |
| `chunk_size=100` | Higher quality, less frequent yields |
