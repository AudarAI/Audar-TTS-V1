# Audar TTS Model Family

Audar TTS is available in three tiers designed to meet different needs - from open-source research and experimentation to production-scale commercial deployments.

---

## Model Overview

| Model | Parameters | License | Best For |
|-------|------------|---------|----------|
| **Flash** | 0.5B | Apache 2.0 | Research, prototyping, edge deployment |
| **Turbo** | 1.5B | Commercial | Production workloads, enhanced quality |
| **Pro** | 4B | Commercial | Studio-grade output, maximum quality |

---

## Audar TTS Flash (Open Source)

**Flash** is our fully open-source model, designed to democratize access to state-of-the-art multilingual TTS.

### Specifications

| Specification | Value |
|---------------|-------|
| Parameters | 0.5B |
| Architecture | Transformer-based with neural codec |
| Training Data | ~200K hours (open-source + proprietary mix) |
| Languages | English, Arabic |
| Sample Rate | 24kHz |
| First Chunk Latency | ~500ms |
| Real-Time Factor | 0.46x |

### Capabilities

- **Zero-Shot Voice Cloning**: Clone any voice from 5 seconds of reference audio
- **Expressive Tags**: Full support for `[laughs]`, `[whispers]`, `[sighs]`, and more
- **Code-Switching**: Seamless transitions between English and Arabic mid-sentence
- **Streaming**: Real-time audio generation with chunk-by-chunk output
- **GGUF Format**: Optimized for efficient CPU inference

### Architecture

Flash is built on a transformer backbone with our proprietary neural codec for high-fidelity audio synthesis. The architecture uses:

- **Qwen 0.5B** as the language model backbone
- **Distill-NeuCodec** for speech encoding (50Hz, single codebook)
- **Separator token architecture** for clean reference/target separation
- **8 explicit delimiters** for structured generation

For full technical details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

### Installation

```bash
git clone https://github.com/AudarAI/Audar-TTS-V1.git
cd Audar-TTS-V1
pip install -r requirements.txt
```

### Quick Example

```python
from audar_tts import AudarTTS

tts = AudarTTS()
audio, metrics = tts.speak(
    "Hello! [laughs] This is Audar TTS Flash.",
    speaker="Eve",
    output="output.wav"
)
```

---

## Audar TTS Turbo (Commercial)

**Turbo** is our mid-tier commercial model, offering enhanced quality and performance for production workloads.

### Specifications

| Specification | Value |
|---------------|-------|
| Parameters | 1.5B |
| Architecture | Proprietary |
| Training Data | Massive proprietary dataset |
| Languages | English, Arabic (+ more coming) |
| First Chunk Latency | ~400ms |
| Voice Quality | Superior |

### Enhanced Capabilities

- **Enhanced Code-Switching**: Improved fluency in multilingual transitions
- **Better Voice Consistency**: More stable voice characteristics across long generations
- **Lower Latency**: Optimized inference pipeline
- **Priority Support**: Direct access to Audar AI engineering team

### Access

Turbo is available through our commercial API.

<p align="center">
  <a href="https://dev.audarai.com">
    <strong>Try Turbo</strong>
  </a>
</p>

---

## Audar TTS Pro (Commercial)

**Pro** is our flagship model, delivering studio-grade voice synthesis for the most demanding applications.

### Specifications

| Specification | Value |
|---------------|-------|
| Parameters | 4B |
| Architecture | Proprietary |
| Training Data | Massive proprietary dataset |
| Languages | English, Arabic (+ more coming) |
| First Chunk Latency | ~300ms |
| Voice Quality | Studio-grade |

### Premium Capabilities

- **Studio-Grade Output**: Broadcast-quality voice synthesis
- **Enhanced Code-Switching**: State-of-the-art multilingual performance
- **Maximum Voice Consistency**: Professional-level stability
- **Dedicated Support**: Custom integration assistance

### Access

Pro is available through our commercial API.

<p align="center">
  <a href="https://dev.audarai.com">
    <strong>Try Pro</strong>
  </a>
</p>

---

## Model Comparison

| Feature | Flash | Turbo | Pro |
|---------|:-----:|:-----:|:---:|
| **Open Source** | Yes | No | No |
| **Parameters** | 0.5B | 1.5B | 4B |
| **Zero-Shot Cloning** | 5s | 5s | 5s |
| **Expressive Tags** | Full | Full | Full |
| **Code-Switching** | Native | Enhanced | Enhanced |
| **First Chunk Latency** | ~500ms | ~400ms | ~300ms |
| **Voice Quality** | Excellent | Superior | Studio-grade |
| **Languages** | EN, AR | EN, AR + more | EN, AR + more |
| **Support** | Community | Priority | Dedicated |

---

## Which Model Should I Use?

### Choose **Flash** if:
- You're doing research or experimentation
- You need a fully open-source solution
- You're deploying to edge devices with limited resources
- You want to fine-tune or modify the model
- You're building a proof-of-concept

### Choose **Turbo** if:
- You're building a production application
- You need enhanced voice quality and consistency
- Lower latency is important for your use case
- You want priority support for integration help
- You're scaling beyond prototype stage

### Choose **Pro** if:
- You need studio-grade voice quality
- You're building for broadcast or professional media
- Maximum quality is more important than cost
- You want dedicated support and custom integration
- You're an enterprise with demanding requirements

---

## Commercial Licensing

**Flash** is released under Apache 2.0 and is free for both commercial and non-commercial use.

**Turbo** and **Pro** are available under commercial licenses. Contact us for pricing and terms.

<p align="center">
  <a href="https://www.audarai.com">Website</a> ·
  <a href="https://dev.audarai.com">Try Demo</a>
</p>

---

## Technical Details

Architecture details for **Turbo** and **Pro** are proprietary. These models are trained on massive proprietary datasets and incorporate architectural innovations not present in the open-source Flash model.

For Flash architecture details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
