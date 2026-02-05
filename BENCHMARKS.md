# Audar TTS Benchmarks

Comprehensive performance evaluation of Audar TTS models against open-source and commercial alternatives.

> **Last Updated**: February 2025  
> **Evaluation Protocol**: Standardized TTS Benchmark Suite v2.1

---

## Table of Contents

1. [Methodology](#methodology)
2. [Audar TTS Flash (0.5B) Benchmarks](#audar-tts-flash-05b-benchmarks)
3. [Audar TTS Turbo (1.5B) Benchmarks](#audar-tts-turbo-15b-benchmarks)
4. [Audar TTS Pro (4B) Benchmarks](#audar-tts-pro-4b-benchmarks)
5. [Cross-Model Comparison](#cross-model-comparison)
6. [Detailed Metrics Definitions](#detailed-metrics-definitions)

---

## Methodology

### Evaluation Framework

All benchmarks follow standardized TTS evaluation protocols with the following specifications:

| Parameter | Value |
|-----------|-------|
| **Test Set Size** | 500 utterances per language |
| **Evaluators (MOS)** | 50 native speakers per language |
| **Hardware** | NVIDIA H100 80GB (commercial), Apple M2 Max (edge) |
| **Audio Quality** | 24kHz, 16-bit PCM |
| **Statistical Significance** | p < 0.05 (two-tailed t-test) |

### Metrics Overview

| Metric | Range | Optimal | Description |
|--------|-------|---------|-------------|
| **MOS** | 1.0 - 5.0 | Higher | Mean Opinion Score for naturalness |
| **SMOS** | 1.0 - 5.0 | Higher | Speaker similarity for voice cloning |
| **WER** | 0% - 100% | Lower | Word Error Rate (intelligibility) |
| **CER** | 0% - 100% | Lower | Character Error Rate (Arabic) |
| **RTF** | 0.0 - ∞ | Lower (<1) | Real-Time Factor |
| **TTFC** | ms | Lower | Time To First Chunk (latency) |
| **PESQ** | 1.0 - 4.5 | Higher | Perceptual speech quality |
| **STOI** | 0.0 - 1.0 | Higher | Short-Time Objective Intelligibility |

### Test Datasets

| Dataset | Language | Utterances | Domain |
|---------|----------|------------|--------|
| **LibriTTS-test** | English | 250 | Audiobooks |
| **VCTK-test** | English | 250 | Multi-speaker |
| **MGB-3** | Arabic | 250 | Broadcast news |
| **QASR** | Arabic | 250 | Conversational |
| **Custom-CS** | EN/AR Mixed | 100 | Code-switching |

---

## Audar TTS Flash (0.5B) Benchmarks

### Executive Summary

**Audar TTS Flash** is the **best-in-class 0.5B parameter TTS model** optimized for edge devices, delivering state-of-the-art quality in the lightweight model category while supporting features typically reserved for larger models.

### English Performance (0.5B Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | WER ↓ | RTF ↓ | TTFC (ms) ↓ |
|-------|------------|-------|--------|-------|-------|-------------|
| **Audar Flash** | **0.5B** | **4.21** | **4.08** | **3.2%** | **0.46** | **487** |
| SpeechT5 | 0.3B | 3.67 | 3.21 | 5.8% | 0.52 | 612 |
| VITS | 0.4B | 3.82 | 3.45 | 4.9% | 0.61 | 534 |
| Piper TTS | 0.2B | 3.41 | 2.89 | 7.2% | 0.38 | 423 |
| Edge TTS (Azure) | API | 3.95 | N/A | 4.1% | API | 245 |

> **Key Finding**: Audar Flash achieves **4.21 MOS** - the highest among all sub-1B parameter models, with voice cloning quality (SMOS 4.08) previously only achievable by models 3x larger.

### Arabic Performance (0.5B Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | CER ↓ | RTF ↓ | TTFC (ms) ↓ |
|-------|------------|-------|--------|-------|-------|-------------|
| **Audar Flash** | **0.5B** | **4.18** | **4.02** | **2.8%** | **0.48** | **502** |
| Coqui XTTS-v2 | 1.6B | 3.89 | 3.67 | 4.2% | 0.87 | 856 |
| Azure Arabic | API | 3.92 | N/A | 3.9% | API | 267 |
| Google TTS | API | 3.78 | N/A | 4.5% | API | 312 |
| MMS-TTS | 0.3B | 3.12 | 2.45 | 8.7% | 0.42 | 398 |

> **Key Finding**: Audar Flash is the **only sub-1B model** achieving native Arabic quality comparable to commercial APIs, with **2.8% CER** - 34% lower than the next best open-source alternative.

### Code-Switching Performance (EN↔AR)

| Model | Fluency MOS ↑ | Transition Quality ↑ | Accent Consistency ↑ |
|-------|---------------|---------------------|---------------------|
| **Audar Flash** | **4.15** | **4.22** | **4.31** |
| Coqui XTTS-v2 | 3.56 | 3.12 | 3.34 |
| Azure Neural | 3.78 | 3.45 | 3.67 |
| Google WaveNet | 3.62 | 3.23 | 3.45 |

> **Key Finding**: Audar Flash demonstrates **superior code-switching** with natural mid-sentence language transitions - a capability typically absent in models under 1B parameters.

### Expressive Speech (Tag Support)

| Model | Tag Support | [laughs] MOS | [whispers] MOS | [sighs] MOS | Overall Expressiveness |
|-------|-------------|--------------|----------------|-------------|----------------------|
| **Audar Flash** | **Full** | **4.12** | **4.08** | **4.05** | **4.08** |
| Coqui XTTS-v2 | Partial | 3.23 | 3.12 | 2.98 | 3.11 |
| Bark | Full | 3.78 | 3.65 | 3.71 | 3.71 |
| VALL-E X | Limited | 3.45 | 3.21 | 3.12 | 3.26 |

> **Key Finding**: Audar Flash offers **full expressive tag support** in a 0.5B model - a feature typically requiring 2B+ parameters. Competitors either lack this feature or deliver significantly lower quality.

### Edge Device Performance (CPU)

Benchmarked on Apple M2, Raspberry Pi 5, and Snapdragon 8 Gen 3:

| Device | Audar Flash RTF | Audar Flash TTFC | SpeechT5 RTF | VITS RTF |
|--------|-----------------|------------------|--------------|----------|
| **Apple M2** | **0.46** | **487ms** | 0.52 | 0.61 |
| **Snapdragon 8 Gen 3** | **0.68** | **623ms** | 0.89 | 1.12 |
| **Raspberry Pi 5** | **0.92** | **945ms** | 1.34 | 1.67 |

> **Key Finding**: Audar Flash maintains **real-time performance (RTF < 1.0)** across all tested edge devices, making it the most deployment-friendly model in its class.

### Flash Statistical Significance

All Audar Flash results show statistical significance (p < 0.05) compared to:
- ✅ All open-source models in the 0.5B category
- ✅ Most commercial edge TTS solutions
- ⚠️ Premium commercial APIs (comparable quality, lower latency)

---

## Audar TTS Turbo (1.5B) Benchmarks

### Executive Summary

**Audar TTS Turbo** delivers **production-grade quality** that matches or exceeds 3B parameter models while maintaining efficient inference. It represents the optimal balance between quality and computational cost.

### English Performance (1.5B vs 3B Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | WER ↓ | PESQ ↑ | RTF ↓ |
|-------|------------|-------|--------|-------|--------|-------|
| **Audar Turbo** | **1.5B** | **4.52** | **4.41** | **2.1%** | **4.12** | **0.31** |
| Coqui XTTS-v2 | 1.6B | 4.12 | 3.89 | 3.4% | 3.78 | 0.87 |
| Tortoise TTS | 1.0B | 4.23 | 4.02 | 2.9% | 3.92 | 4.20 |
| YourTTS | 1.2B | 3.98 | 3.76 | 3.8% | 3.65 | 0.92 |
| VALL-E X | 3.0B | 4.45 | 4.28 | 2.4% | 4.05 | 1.23 |
| Bark | 1.0B | 4.01 | 3.54 | 4.2% | 3.71 | 2.10 |

> **Key Finding**: Audar Turbo achieves **4.52 MOS** - matching or exceeding 3B models like VALL-E X while using **50% fewer parameters** and **4x faster inference**.

### Arabic Performance (1.5B vs 3B Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | CER ↓ | STOI ↑ | RTF ↓ |
|-------|------------|-------|--------|-------|--------|-------|
| **Audar Turbo** | **1.5B** | **4.48** | **4.35** | **1.9%** | **0.94** | **0.33** |
| Coqui XTTS-v2 | 1.6B | 3.89 | 3.67 | 4.2% | 0.87 | 0.87 |
| Azure Neural | API | 4.21 | N/A | 2.8% | 0.91 | API |
| Google WaveNet | API | 4.15 | N/A | 3.1% | 0.89 | API |
| Amazon Polly | API | 3.92 | N/A | 3.8% | 0.86 | API |

> **Key Finding**: Audar Turbo is the **best Arabic TTS in the 1.5B-3B category**, outperforming commercial APIs with **1.9% CER** and maintaining excellent STOI (0.94).

### Commercial API Comparison

| Model | Provider | MOS (EN) | MOS (AR) | SMOS | Latency | Offline |
|-------|----------|----------|----------|------|---------|---------|
| **Audar Turbo** | **Audar AI** | **4.52** | **4.48** | **4.41** | **398ms** | **Yes** |
| Eleven Multilingual v2 | ElevenLabs | 4.56 | 4.12 | 4.45 | 523ms | No |
| Azure Neural | Microsoft | 4.38 | 4.21 | N/A | 267ms | No |
| WaveNet | Google | 4.31 | 4.15 | N/A | 312ms | No |
| Neural | Amazon | 4.12 | 3.92 | N/A | 345ms | No |

> **Key Finding**: Audar Turbo matches **ElevenLabs quality in English** and **significantly exceeds it in Arabic** (+0.36 MOS), while offering on-premise deployment capability.

### Voice Cloning Quality (Zero-Shot)

5-second reference audio evaluation:

| Model | SMOS ↑ | Speaker Verification (EER) ↓ | Consistency Score ↑ |
|-------|--------|------------------------------|---------------------|
| **Audar Turbo** | **4.41** | **4.2%** | **0.91** |
| ElevenLabs | 4.45 | 3.9% | 0.92 |
| Coqui XTTS-v2 | 3.89 | 6.8% | 0.84 |
| YourTTS | 3.76 | 8.2% | 0.79 |
| VALL-E X | 4.28 | 5.1% | 0.88 |

> **Key Finding**: Audar Turbo achieves **near-ElevenLabs voice cloning quality** (SMOS 4.41 vs 4.45) with the advantage of **on-premise deployment**.

### Latency Analysis

| Model | TTFC (ms) ↓ | P95 Latency ↓ | Throughput (chars/s) ↑ |
|-------|-------------|---------------|------------------------|
| **Audar Turbo** | **398** | **456** | **892** |
| ElevenLabs | 523 | 612 | 756 |
| Coqui XTTS-v2 | 856 | 1023 | 423 |
| Tortoise | 3200 | 4100 | 112 |

> **Key Finding**: Audar Turbo offers **24% lower latency** than ElevenLabs with **18% higher throughput**.

### Turbo Statistical Significance

All Audar Turbo results show statistical significance (p < 0.05) compared to:
- ✅ All open-source models up to 3B parameters
- ✅ Commercial APIs for Arabic performance
- ⚠️ ElevenLabs English (comparable, within margin of error)

---

## Audar TTS Pro (4B) Benchmarks

### Executive Summary

**Audar TTS Pro** is the **market-leading Arabic + English TTS**, achieving the highest combined bilingual performance of any system evaluated. It sets new standards for studio-grade multilingual voice synthesis.

### English Performance (Premium Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | WER ↓ | PESQ ↑ | Studio Grade |
|-------|------------|-------|--------|-------|--------|--------------|
| **Audar Pro** | **4B** | **4.68** | **4.62** | **1.4%** | **4.28** | **Yes** |
| ElevenLabs v2 | Unknown | 4.56 | 4.45 | 1.8% | 4.15 | Yes |
| Azure Neural HD | API | 4.45 | N/A | 2.1% | 4.08 | Yes |
| OpenAI TTS HD | Unknown | 4.52 | 4.21 | 1.9% | 4.12 | Yes |
| Google Studio | API | 4.42 | N/A | 2.3% | 4.02 | Yes |

> **Key Finding**: Audar Pro achieves **4.68 MOS** in English - the highest score among all evaluated systems, representing **+2.6%** improvement over ElevenLabs.

### Arabic Performance (Premium Category)

| Model | Parameters | MOS ↑ | SMOS ↑ | CER ↓ | STOI ↑ | Dialect Support |
|-------|------------|-------|--------|-------|--------|-----------------|
| **Audar Pro** | **4B** | **4.71** | **4.58** | **1.2%** | **0.97** | **MSA + Gulf + Levantine** |
| ElevenLabs v2 | Unknown | 4.12 | 3.89 | 3.4% | 0.88 | MSA only |
| Azure Neural | API | 4.21 | N/A | 2.8% | 0.91 | MSA + Gulf |
| Google WaveNet | API | 4.15 | N/A | 3.1% | 0.89 | MSA only |
| Amazon Polly | API | 3.92 | N/A | 3.8% | 0.86 | MSA only |

> **Key Finding**: Audar Pro delivers **4.71 MOS in Arabic** - a **14% improvement over ElevenLabs** and the highest Arabic TTS score ever recorded in our evaluation suite.

### Combined Bilingual Score

Weighted average across English and Arabic test sets:

| Model | Combined MOS ↑ | Combined SMOS ↑ | Bilingual Consistency ↑ |
|-------|----------------|-----------------|------------------------|
| **Audar Pro** | **4.70** | **4.60** | **0.96** |
| ElevenLabs v2 | 4.34 | 4.17 | 0.82 |
| Azure Neural | 4.33 | N/A | 0.89 |
| Google WaveNet | 4.28 | N/A | 0.87 |

> **Key Finding**: Audar Pro achieves **8.3% higher combined bilingual MOS** than the next best competitor (ElevenLabs), with significantly better voice consistency across languages.

### Code-Switching Excellence

| Model | CS Fluency ↑ | Transition MOS ↑ | Accent Preservation ↑ | Natural Flow ↑ |
|-------|--------------|------------------|----------------------|----------------|
| **Audar Pro** | **4.72** | **4.68** | **4.75** | **4.71** |
| ElevenLabs v2 | 3.89 | 3.67 | 3.78 | 3.72 |
| Azure Neural | 3.95 | 3.78 | 3.82 | 3.81 |
| Google WaveNet | 3.82 | 3.56 | 3.71 | 3.65 |

> **Key Finding**: Audar Pro demonstrates **21% improvement in code-switching fluency** over competitors - the largest performance gap in any category.

### Expressive Speech (Advanced Tags)

| Model | Tag Vocabulary | Emotion Range | [laughs] | [whispers] | [sighs] | Custom Tags |
|-------|---------------|---------------|----------|------------|---------|-------------|
| **Audar Pro** | **Extended** | **12 emotions** | **4.65** | **4.61** | **4.58** | **Yes** |
| ElevenLabs v2 | Standard | 8 emotions | 4.45 | 4.38 | 4.32 | No |
| Bark | Extended | 10 emotions | 3.78 | 3.65 | 3.71 | Partial |
| OpenAI TTS | Limited | 4 emotions | 3.92 | 3.78 | 3.65 | No |

> **Key Finding**: Audar Pro supports **12 emotion categories** with custom tag capability, delivering **+4.5% expressiveness** over ElevenLabs.

### Studio Production Metrics

| Model | Dynamic Range | S/N Ratio | Frequency Response | Artifact Score ↓ |
|-------|--------------|-----------|-------------------|------------------|
| **Audar Pro** | **68 dB** | **92 dB** | **20Hz-20kHz ±1dB** | **0.8%** |
| ElevenLabs v2 | 65 dB | 89 dB | 50Hz-18kHz ±2dB | 1.2% |
| Azure Neural HD | 62 dB | 87 dB | 80Hz-16kHz ±3dB | 1.8% |

> **Key Finding**: Audar Pro meets **broadcast production standards** with the lowest artifact score (0.8%) among all evaluated systems.

### Pro Latency Performance

| Model | TTFC (ms) ↓ | P50 Latency | P99 Latency | Streaming Support |
|-------|-------------|-------------|-------------|-------------------|
| **Audar Pro** | **312** | **298** | **412** | **Yes** |
| ElevenLabs v2 | 523 | 489 | 678 | Yes |
| Azure Neural HD | 412 | 378 | 534 | Limited |
| OpenAI TTS HD | 567 | 523 | 712 | No |

> **Key Finding**: Audar Pro achieves **40% lower latency** than ElevenLabs while delivering higher quality output.

### Pro Statistical Significance

All Audar Pro results show statistical significance (p < 0.01) compared to:
- ✅ All open-source models
- ✅ All commercial APIs for Arabic performance
- ✅ ElevenLabs for combined bilingual performance
- ✅ All competitors for code-switching quality

---

## Cross-Model Comparison

### Audar Model Family Overview

| Metric | Flash (0.5B) | Turbo (1.5B) | Pro (4B) |
|--------|--------------|--------------|----------|
| **MOS (English)** | 4.21 | 4.52 | 4.68 |
| **MOS (Arabic)** | 4.18 | 4.48 | 4.71 |
| **SMOS** | 4.08 | 4.41 | 4.62 |
| **WER (English)** | 3.2% | 2.1% | 1.4% |
| **CER (Arabic)** | 2.8% | 1.9% | 1.2% |
| **RTF** | 0.46 | 0.31 | 0.22 |
| **TTFC** | 487ms | 398ms | 312ms |
| **Expressive Tags** | Full | Full | Extended |
| **Code-Switching** | Native | Enhanced | Superior |

### Market Position Matrix

```
Quality (MOS)
    ↑
4.7 │                              ★ Audar Pro
    │                    ★ Audar Turbo
4.5 │              ○ ElevenLabs
    │         ○ OpenAI
4.3 │    ★ Audar Flash
    │    ○ Azure
4.1 │ ○ Google
    │ ○ Coqui XTTS
3.9 │
    │ ○ Bark
3.7 │        ○ Tortoise
    │
    └────────────────────────────────────→
         0.5B    1.5B    3B    4B+    API
                   Model Size / Type
    
★ = Audar TTS    ○ = Competitor
```

### Use Case Recommendations

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Mobile Apps** | Flash | Best quality under 1GB, real-time on mobile CPUs |
| **Edge Devices** | Flash | Optimized GGUF, runs on Raspberry Pi |
| **Web Applications** | Turbo | Optimal quality/latency balance |
| **Production APIs** | Turbo | Enterprise-ready, scalable |
| **Broadcast/Media** | Pro | Studio-grade quality |
| **Arabic-First Products** | Pro | Unmatched Arabic performance |
| **Bilingual Content** | Pro | Best code-switching, consistency |

---

## Detailed Metrics Definitions

### Quality Metrics

| Metric | Full Name | Description | Evaluation Method |
|--------|-----------|-------------|-------------------|
| **MOS** | Mean Opinion Score | Overall naturalness rating | 50 human evaluators, 1-5 scale |
| **SMOS** | Speaker Mean Opinion Score | Voice similarity to reference | A/B comparison with reference |
| **PESQ** | Perceptual Evaluation of Speech Quality | ITU-T P.862 standard | Automated, reference-based |
| **STOI** | Short-Time Objective Intelligibility | Speech clarity metric | Automated, 0-1 scale |

### Accuracy Metrics

| Metric | Full Name | Description | Evaluation Method |
|--------|-----------|-------------|-------------------|
| **WER** | Word Error Rate | Transcription accuracy (EN) | ASR + manual verification |
| **CER** | Character Error Rate | Transcription accuracy (AR) | ASR + manual verification |
| **EER** | Equal Error Rate | Speaker verification accuracy | Automated speaker ID |

### Performance Metrics

| Metric | Full Name | Description | Evaluation Method |
|--------|-----------|-------------|-------------------|
| **RTF** | Real-Time Factor | Processing time / audio length | Automated benchmark |
| **TTFC** | Time To First Chunk | Latency to first audio | Streaming benchmark |
| **P95/P99** | Percentile Latency | Tail latency distribution | Statistical analysis |

---

## Reproducibility

### Hardware Specifications

| Component | Cloud Benchmark | Edge Benchmark |
|-----------|-----------------|----------------|
| **GPU** | NVIDIA H100 80GB | N/A |
| **CPU** | AMD EPYC 7763 | Apple M2 Max / Snapdragon 8 Gen 3 |
| **RAM** | 512GB | 16-32GB |
| **Storage** | NVMe SSD | NVMe SSD |

### Software Environment

```
Python: 3.11
PyTorch: 2.2.0
CUDA: 12.1
llama-cpp-python: 0.2.56
```

### Evaluation Scripts

Benchmarking code and datasets available upon request for academic reproducibility.

---

## Citation

If you use these benchmarks in research, please cite:

```bibtex
@misc{audar2025benchmarks,
  title={Audar TTS Benchmark Report},
  author={Audar AI Research},
  year={2025},
  howpublished={\url{https://github.com/AudarAI/Audar-TTS-V1}},
}
```

---

<p align="center">
  <a href="https://www.audarai.com">Website</a> ·
  <a href="https://dev.audarai.com">Try Demo</a> ·
  <a href="MODELS.md">Model Details</a>
</p>
