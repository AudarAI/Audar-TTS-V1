---
license: apache-2.0
language:
  - en
  - ar
tags:
  - tts
  - text-to-speech
  - voice-cloning
  - gguf
  - zero-shot
pipeline_tag: text-to-speech
library_name: llama-cpp-python
---

# Audar-TTS Flash v1 (GGUF)

Zero-shot voice cloning TTS. GGUF format for CPU/GPU inference via `llama-cpp-python`.

## Specs

| Param | Value |
|-------|-------|
| Base | Qwen2.5-0.5B |
| Quant | Q8_0 |
| Size | 766 MB |
| Context | 8192 |
| Codebook | 65536 |
| Sample Rate | 24 kHz |
| Codes/sec | 50 |

## Files

```
audar-tts-flash-v1-q8_0.gguf   # Model
tokenizer.json                  # Tokenizer
tokenizer_config.json
special_tokens_map.json
```

## Quick Start

```python
from audar_tts import AudarTTS

tts = AudarTTS()

# Synthesis
audio, metrics = tts.speak("Hello world", speaker="Eve", output="out.wav")

# Streaming
for chunk, info in tts.stream("Long text here", speaker="Eve"):
    play(chunk)  # ~1s chunks

# Voice cloning
tts.clone_voice("my_voice", "ref.wav", "transcript text")
tts.speak("In cloned voice", speaker="my_voice")
```

## Low-Level API

```python
from llama_cpp import Llama
from transformers import AutoTokenizer
from neucodec import DistillNeuCodec
import torch, soundfile as sf

# Load
llm = Llama(model_path="audar-tts-flash-v1-q8_0.gguf", n_ctx=8192)
tokenizer = AutoTokenizer.from_pretrained("audarai/audar_tts_flash_v1_gguf")
codec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")

# Prompt template
prompt = """user: Convert the text to speech:<|REF_TEXT_START|>{ref_ph}<|REF_TEXT_END|><|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|><|TARGET_TEXT_START|>{tgt_ph}<|TARGET_TEXT_END|>
assistant:<|TARGET_CODES_START|>"""

# Generate
output = llm(prompt, max_tokens=500, stop=["<|TARGET_CODES_END|>"])
tokens = tokenizer.encode(output["choices"][0]["text"])

# Decode speech codes (filter speech_* tokens: 151671-217206)
codes = [t - 151671 for t in tokens if 151671 <= t <= 217206]
wav = codec.decode_code(torch.tensor(codes).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
sf.write("output.wav", wav, 24000)
```

## Delimiters

| Token | Purpose |
|-------|---------|
| `<\|REF_TEXT_START\|>` | Reference phonemes start |
| `<\|REF_SPEECH_START\|>` | Reference codes start (`<\|speech_0\|>`..`<\|speech_65535\|>`) |
| `<\|TARGET_TEXT_START\|>` | Target phonemes start |
| `<\|TARGET_CODES_START\|>` | Generated codes start |

## Links

- [Full Implementation](https://github.com/AudarAI/Audar-TTS-V1)
- [Codec](https://huggingface.co/neuphonic/distill-neucodec)

## License

Apache 2.0
