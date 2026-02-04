# Audar-TTS Technical Report

**Version 1.0** | **February 2025**

---

## Executive Summary

Audar-TTS represents a significant advancement in zero-shot text-to-speech synthesis, achieving real-time voice cloning with minimal reference audio. Built on a neural codec language modeling paradigm, the system delivers human-quality speech synthesis at 2x faster than real-time playback speeds while requiring only 3 seconds of reference audio for voice cloning.

**Key Performance Metrics:**
- **Real-Time Factor (RTF):** 0.46x (2.2x faster than real-time)
- **First Chunk Latency:** ~500ms (streaming mode)
- **Reference Audio Required:** 3-10 seconds
- **Supported Languages:** English, Arabic (with code-switching)

---

## 1. Introduction

### 1.1 Background

Text-to-Speech (TTS) technology has undergone a paradigm shift with the emergence of neural codec language models. Traditional TTS systems relied on concatenative synthesis or statistical parametric approaches, followed by neural vocoders. Modern systems leverage discrete speech representations and autoregressive language models to achieve unprecedented naturalness and speaker similarity.

### 1.2 Design Philosophy

Audar-TTS is designed around three core principles:

1. **Zero-Shot Capability:** Clone any voice without fine-tuning
2. **Real-Time Performance:** Enable interactive applications
3. **Streaming Architecture:** Minimize perceived latency

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUDAR-TTS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │  Reference  │     │  Reference  │     │   Target    │                   │
│  │    Audio    │     │    Text     │     │    Text     │                   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Neural    │    │   Phoneme    │    │   Phoneme    │                  │
│  │    Codec     │    │  Processor   │    │  Processor   │                  │
│  │   Encoder    │    │   (G2P)      │    │   (G2P)      │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         │    Discrete       │    Phoneme        │    Phoneme               │
│         │    Codes          │    Sequence       │    Sequence              │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             │                                              │
│                             ▼                                              │
│                    ┌─────────────────┐                                     │
│                    │  Prompt Builder │                                     │
│                    │   (Template)    │                                     │
│                    └────────┬────────┘                                     │
│                             │                                              │
│                             │  Structured Prompt                           │
│                             │                                              │
│                             ▼                                              │
│         ┌───────────────────────────────────────────┐                      │
│         │                                           │                      │
│         │         LANGUAGE MODEL CORE               │                      │
│         │                                           │                      │
│         │    ┌─────────────────────────────┐       │                      │
│         │    │   Transformer Architecture  │       │                      │
│         │    │                             │       │                      │
│         │    │  • Multi-Head Attention     │       │                      │
│         │    │  • Feed-Forward Networks    │       │                      │
│         │    │  • RoPE Positional Encoding │       │                      │
│         │    │  • RMS Normalization        │       │                      │
│         │    │                             │       │                      │
│         │    └─────────────────────────────┘       │                      │
│         │                                           │                      │
│         └─────────────────────┬─────────────────────┘                      │
│                               │                                            │
│                               │  Speech Codes                              │
│                               │  (Autoregressive)                          │
│                               │                                            │
│                               ▼                                            │
│                    ┌─────────────────┐                                     │
│                    │  Neural Codec   │                                     │
│                    │    Decoder      │                                     │
│                    └────────┬────────┘                                     │
│                             │                                              │
│                             │  24kHz Waveform                              │
│                             ▼                                              │
│                    ┌─────────────────┐                                     │
│                    │  Synthesized    │                                     │
│                    │    Speech       │                                     │
│                    └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### 2.2.1 Neural Audio Codec

The system employs a neural audio codec for bidirectional conversion between continuous audio waveforms and discrete token sequences.

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL AUDIO CODEC                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ENCODER PATH                      DECODER PATH                │
│   ────────────                      ────────────                │
│                                                                 │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │  Waveform   │                   │   Discrete  │            │
│   │  (16kHz)    │                   │    Codes    │            │
│   └──────┬──────┘                   └──────┬──────┘            │
│          │                                 │                    │
│          ▼                                 ▼                    │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │  Encoder    │                   │  Embedding  │            │
│   │  Network    │                   │   Lookup    │            │
│   └──────┬──────┘                   └──────┬──────┘            │
│          │                                 │                    │
│          ▼                                 ▼                    │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │   Vector    │                   │  Decoder    │            │
│   │ Quantizer   │                   │  Network    │            │
│   └──────┬──────┘                   └──────┬──────┘            │
│          │                                 │                    │
│          ▼                                 ▼                    │
│   ┌─────────────┐                   ┌─────────────┐            │
│   │  Discrete   │                   │  Waveform   │            │
│   │   Codes     │                   │  (24kHz)    │            │
│   └─────────────┘                   └─────────────┘            │
│                                                                 │
│   Codec Rate: 50 codes/second                                   │
│   Codebook Size: 8192 entries                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Specifications:**
- **Input Sample Rate:** 16 kHz (encoder)
- **Output Sample Rate:** 24 kHz (decoder)
- **Temporal Resolution:** 50 codes per second
- **Codebook Size:** 8,192 discrete tokens

#### 2.2.2 Phoneme Processing Pipeline

The phonemization subsystem converts orthographic text to phonetic representations, enabling language-agnostic synthesis.

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHONEME PROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Text                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────┐                                          │
│   │ Language        │                                          │
│   │ Detection       │◄─── Unicode Range Analysis               │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ├─────────────────┬─────────────────┐               │
│            ▼                 ▼                 ▼               │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│   │  English    │   │   Arabic    │   │   Mixed     │         │
│   │  Phonemizer │   │  Phonemizer │   │  Handler    │         │
│   │  (en-us)    │   │    (ar)     │   │             │         │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘         │
│          │                 │                 │                 │
│          └─────────────────┴─────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│                   ┌─────────────────┐                          │
│                   │  IPA Phoneme    │                          │
│                   │  Sequence       │                          │
│                   └─────────────────┘                          │
│                                                                 │
│   Features:                                                     │
│   • Stress marking preservation                                 │
│   • Punctuation handling                                        │
│   • Code-switching support                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.3 Language Model Architecture

The core synthesis engine is based on a transformer architecture optimized for speech code generation.

```
┌─────────────────────────────────────────────────────────────────┐
│              TRANSFORMER LANGUAGE MODEL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    INPUT EMBEDDING                       │  │
│   │  ┌─────────────────────────────────────────────────┐    │  │
│   │  │  Unified Vocabulary: Text + Speech Tokens       │    │  │
│   │  │  • Base vocabulary: ~150K text tokens           │    │  │
│   │  │  • Speech tokens: ~65K discrete codes           │    │  │
│   │  │  • Special tokens: Template delimiters          │    │  │
│   │  └─────────────────────────────────────────────────┘    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │               TRANSFORMER BLOCKS (×N)                    │  │
│   │                                                          │  │
│   │   ┌─────────────────────────────────────────────────┐   │  │
│   │   │              ATTENTION LAYER                     │   │  │
│   │   │                                                  │   │  │
│   │   │    ┌────────┐  ┌────────┐  ┌────────┐          │   │  │
│   │   │    │   Q    │  │   K    │  │   V    │          │   │  │
│   │   │    └───┬────┘  └───┬────┘  └───┬────┘          │   │  │
│   │   │        │           │           │                │   │  │
│   │   │        └───────────┼───────────┘                │   │  │
│   │   │                    │                            │   │  │
│   │   │                    ▼                            │   │  │
│   │   │    ┌─────────────────────────────────┐         │   │  │
│   │   │    │   Grouped Query Attention       │         │   │  │
│   │   │    │   (Multi-Head with KV Sharing)  │         │   │  │
│   │   │    └─────────────────────────────────┘         │   │  │
│   │   │                                                  │   │  │
│   │   └──────────────────────────────────────────────────┘   │  │
│   │                          │                               │  │
│   │                          ▼                               │  │
│   │   ┌─────────────────────────────────────────────────┐   │  │
│   │   │            FEED-FORWARD NETWORK                  │   │  │
│   │   │                                                  │   │  │
│   │   │    Input ──► Gate ──► SiLU ──► Up ──► Down ──►  │   │  │
│   │   │                                                  │   │  │
│   │   └──────────────────────────────────────────────────┘   │  │
│   │                                                          │  │
│   │   • RMS Layer Normalization                              │  │
│   │   • Rotary Positional Embeddings (RoPE)                  │  │
│   │   • Residual Connections                                 │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                  OUTPUT PROJECTION                       │  │
│   │         Vocabulary Logits → Speech Code Selection        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Processing Pipeline

### 3.1 Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: PREPROCESSING                                                    │
│   ──────────────────────                                                    │
│                                                                             │
│   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│   │   Reference   │      │   Reference   │      │    Target     │          │
│   │    Audio      │      │     Text      │      │     Text      │          │
│   └───────┬───────┘      └───────┬───────┘      └───────┬───────┘          │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│   │    Codec      │      │   Phonemize   │      │   Phonemize   │          │
│   │    Encode     │      │               │      │               │          │
│   └───────┬───────┘      └───────┬───────┘      └───────┬───────┘          │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│      [codes...]            [phonemes...]           [phonemes...]            │
│                                                                             │
│   ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   PHASE 2: PROMPT CONSTRUCTION                                              │
│   ────────────────────────────                                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        PROMPT TEMPLATE                               │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │  <REF_TEXT>   reference phonemes    </REF_TEXT>             │   │  │
│   │   │  <REF_SPEECH> reference codes       </REF_SPEECH>           │   │  │
│   │   │  <TGT_TEXT>   target phonemes       </TGT_TEXT>             │   │  │
│   │   │  <TGT_CODES>  [TO BE GENERATED]                             │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   PHASE 3: AUTOREGRESSIVE GENERATION                                        │
│   ──────────────────────────────────                                        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │    Prompt ──► LM ──► Code₁ ──► LM ──► Code₂ ──► ... ──► <STOP>     │  │
│   │                                                                      │  │
│   │    Generation Parameters:                                            │  │
│   │    • Temperature: 0.3 (controlled variation)                         │  │
│   │    • Top-K: 40 (vocabulary filtering)                                │  │
│   │    • Top-P: 0.95 (nucleus sampling)                                  │  │
│   │    • Repeat Penalty: 1.1 (diversity)                                 │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   PHASE 4: WAVEFORM SYNTHESIS                                               │
│   ───────────────────────────                                               │
│                                                                             │
│   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐          │
│   │   Generated   │      │    Codec      │      │    Output     │          │
│   │    Codes      │─────►│    Decode     │─────►│   Waveform    │          │
│   └───────────────┘      └───────────────┘      └───────────────┘          │
│                                                                             │
│        [c₁,c₂,...,cₙ]           ▼                    24kHz                  │
│                           Neural Vocoder                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Streaming Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STREAMING SYNTHESIS FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Time ──────────────────────────────────────────────────────────────────►  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CODE GENERATION (Autoregressive)                                    │  │
│   │                                                                      │  │
│   │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐     │  │
│   │  │ c₁ │ c₂ │ c₃ │...│c₅₀│c₅₁│...│c₁₀₀│...│c₁₅₀│...│ cₙ │     │  │
│   │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘     │  │
│   │  │◄── Chunk 1 ──►│◄── Chunk 2 ──►│◄── Chunk 3 ──►│                  │  │
│   │       50 codes         50 codes        50 codes                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                 │                │                │                         │
│                 ▼                ▼                ▼                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  AUDIO DECODING (Chunked)                                            │  │
│   │                                                                      │  │
│   │  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │  │
│   │  │  Audio     │    │  Audio     │    │  Audio     │                 │  │
│   │  │  Chunk 1   │    │  Chunk 2   │    │  Chunk 3   │    ...          │  │
│   │  │   (~1s)    │    │   (~1s)    │    │   (~1s)    │                 │  │
│   │  └────────────┘    └────────────┘    └────────────┘                 │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                 │                │                │                         │
│                 ▼                ▼                ▼                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  PLAYBACK TIMELINE                                                   │  │
│   │                                                                      │  │
│   │  ├─ ~0.5s ─┤                                                        │  │
│   │  │ Latency │                                                         │  │
│   │  │         │                                                         │  │
│   │  ▼         ▼────────────▼────────────▼────────────▼                 │  │
│   │  ┌─────────┬────────────┬────────────┬────────────┬─────────        │  │
│   │  │   ▶     │  Playing   │  Playing   │  Playing   │   ...           │  │
│   │  │  Start  │  Chunk 1   │  Chunk 2   │  Chunk 3   │                 │  │
│   │  └─────────┴────────────┴────────────┴────────────┴─────────        │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Key Metrics:                                                              │
│   • First Chunk Latency: ~500ms                                            │
│   • Chunk Duration: ~1 second (50 codes)                                   │
│   • Buffer-free playback with RTF < 1.0                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Speaker Representation

### 4.1 Voice Cloning Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ZERO-SHOT VOICE CLONING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    SPEAKER ENCODING                                │    │
│   │                                                                    │    │
│   │   Reference Audio (3-10s)                                         │    │
│   │         │                                                          │    │
│   │         ▼                                                          │    │
│   │   ┌─────────────┐                                                  │    │
│   │   │   Codec     │                                                  │    │
│   │   │  Encoder    │                                                  │    │
│   │   └──────┬──────┘                                                  │    │
│   │          │                                                          │    │
│   │          ▼                                                          │    │
│   │   ┌─────────────────────────────────────────────┐                  │    │
│   │   │         Speaker Code Sequence               │                  │    │
│   │   │  [c₁, c₂, c₃, ..., c₁₅₀, ..., cₘ]          │                  │    │
│   │   │                                             │                  │    │
│   │   │  Encodes:                                   │                  │    │
│   │   │  • Timbre characteristics                   │                  │    │
│   │   │  • Prosodic patterns                        │                  │    │
│   │   │  • Speaking rate tendencies                 │                  │    │
│   │   │  • Acoustic environment                     │                  │    │
│   │   └─────────────────────────────────────────────┘                  │    │
│   │                                                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    IN-CONTEXT LEARNING                             │    │
│   │                                                                    │    │
│   │   The language model learns speaker characteristics through        │    │
│   │   the reference codes provided in the prompt context:              │    │
│   │                                                                    │    │
│   │   ┌─────────────────────────────────────────────────────────┐     │    │
│   │   │                                                          │     │    │
│   │   │   [Ref Text Phonemes] + [Ref Speech Codes]               │     │    │
│   │   │              │                                           │     │    │
│   │   │              ▼                                           │     │    │
│   │   │   ┌─────────────────────────────────────────┐           │     │    │
│   │   │   │    Text-Speech Alignment Learning       │           │     │    │
│   │   │   │    (Implicit during attention)          │           │     │    │
│   │   │   └─────────────────────────────────────────┘           │     │    │
│   │   │              │                                           │     │    │
│   │   │              ▼                                           │     │    │
│   │   │   [Target Text Phonemes] → [Generated Speech Codes]      │     │    │
│   │   │                              (in speaker's voice)        │     │    │
│   │   │                                                          │     │    │
│   │   └─────────────────────────────────────────────────────────┘     │    │
│   │                                                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Speaker Caching System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SPEAKER CACHE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   First Request                        Subsequent Requests                  │
│   ─────────────                        ───────────────────                  │
│                                                                             │
│   ┌─────────────┐                      ┌─────────────┐                     │
│   │   Audio     │                      │   Speaker   │                     │
│   │   File      │                      │    Name     │                     │
│   └──────┬──────┘                      └──────┬──────┘                     │
│          │                                    │                             │
│          ▼                                    ▼                             │
│   ┌─────────────┐                      ┌─────────────┐                     │
│   │   Codec     │                      │   Cache     │                     │
│   │   Encode    │                      │   Lookup    │                     │
│   │  (~200ms)   │                      │   (~1ms)    │                     │
│   └──────┬──────┘                      └──────┬──────┘                     │
│          │                                    │                             │
│          ▼                                    ▼                             │
│   ┌─────────────┐                      ┌─────────────┐                     │
│   │   Cache     │                      │   Cached    │                     │
│   │   Store     │──────────────────────│   Codes     │                     │
│   └─────────────┘                      └─────────────┘                     │
│                                                                             │
│   Cache Structure:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  {                                                                   │  │
│   │    "speaker_name": {                                                 │  │
│   │      "codes": [c₁, c₂, ..., cₙ],                                    │  │
│   │      "phonemes": "...",                                              │  │
│   │      "audio_hash": "abc123..."                                       │  │
│   │    }                                                                 │  │
│   │  }                                                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Benefits:                                                                 │
│   • Instant speaker switching                                               │
│   • No re-encoding overhead                                                 │
│   • Persistent across sessions                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Deployment Architecture

### 5.1 Quantized Inference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUANTIZED MODEL DEPLOYMENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                   MODEL FORMATS                                    │    │
│   │                                                                    │    │
│   │   Original Model              Quantized Model                      │    │
│   │   (FP16/BF16)                    (Q8_0)                           │    │
│   │                                                                    │    │
│   │   ┌─────────────┐            ┌─────────────┐                      │    │
│   │   │   ~1.6 GB   │  ────────► │   ~800 MB   │                      │    │
│   │   │             │  Quantize  │             │                      │    │
│   │   └─────────────┘            └─────────────┘                      │    │
│   │                                                                    │    │
│   │   • 16-bit weights           • 8-bit weights                      │    │
│   │   • GPU required             • CPU capable                        │    │
│   │   • Higher precision         • Minimal quality loss               │    │
│   │                                                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                   GGUF RUNTIME                                     │    │
│   │                                                                    │    │
│   │   ┌─────────────────────────────────────────────────────────┐     │    │
│   │   │                                                          │     │    │
│   │   │              llama.cpp Backend                           │     │    │
│   │   │                                                          │     │    │
│   │   │   ┌──────────┐  ┌──────────┐  ┌──────────┐             │     │    │
│   │   │   │   CPU    │  │   GPU    │  │  Metal   │             │     │    │
│   │   │   │  (AVX2)  │  │  (CUDA)  │  │  (Apple) │             │     │    │
│   │   │   └──────────┘  └──────────┘  └──────────┘             │     │    │
│   │   │                                                          │     │    │
│   │   │   Features:                                              │     │    │
│   │   │   • Automatic hardware detection                         │     │    │
│   │   │   • Memory-mapped model loading                          │     │    │
│   │   │   • KV-cache optimization                                │     │    │
│   │   │   • Batch processing support                             │     │    │
│   │   │                                                          │     │    │
│   │   └─────────────────────────────────────────────────────────┘     │    │
│   │                                                                    │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 System Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HARDWARE REQUIREMENTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MINIMUM CONFIGURATION                    RECOMMENDED CONFIGURATION        │
│   ─────────────────────                    ─────────────────────────        │
│                                                                             │
│   ┌─────────────────────┐                 ┌─────────────────────┐          │
│   │  CPU: 4 cores       │                 │  CPU: 8+ cores      │          │
│   │  RAM: 4 GB          │                 │  RAM: 16 GB         │          │
│   │  GPU: Optional      │                 │  GPU: 6+ GB VRAM    │          │
│   │  Storage: 2 GB      │                 │  Storage: SSD       │          │
│   └─────────────────────┘                 └─────────────────────┘          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    PERFORMANCE SCALING                               │  │
│   │                                                                      │  │
│   │   Hardware          │  RTF    │  First Chunk  │  Throughput         │  │
│   │   ──────────────────┼─────────┼───────────────┼──────────────       │  │
│   │   CPU (8-core)      │  ~1.2x  │  ~1.5s        │  0.8x realtime      │  │
│   │   GPU (RTX 3060)    │  ~0.5x  │  ~0.6s        │  2.0x realtime      │  │
│   │   GPU (RTX 4090)    │  ~0.3x  │  ~0.3s        │  3.3x realtime      │  │
│   │   Apple M2 Pro      │  ~0.6x  │  ~0.7s        │  1.7x realtime      │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Comparative Analysis

### 6.1 Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               MODERN TTS ARCHITECTURE COMPARISON                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   APPROACH          │ ARCHITECTURE      │ STRENGTHS    │ TRADE-OFFS │  │
│   │   ─────────────────────────────────────────────────────────────────│  │
│   │                     │                   │              │            │  │
│   │   Autoregressive    │ LM + Neural       │ Natural      │ Sequential │  │
│   │   (Audar-TTS,       │ Codec             │ prosody,     │ generation │  │
│   │    VALL-E)          │                   │ zero-shot    │            │  │
│   │                     │                   │              │            │  │
│   │   ─────────────────────────────────────────────────────────────────│  │
│   │                     │                   │              │            │  │
│   │   Flow Matching     │ DiT + Vocoder     │ Parallel     │ Requires   │  │
│   │   (F5-TTS)          │                   │ generation   │ alignment  │  │
│   │                     │                   │              │            │  │
│   │   ─────────────────────────────────────────────────────────────────│  │
│   │                     │                   │              │            │  │
│   │   Non-Autoregressive│ Masked LM +       │ Fast         │ Quality    │  │
│   │   (MaskGCT)         │ Codec             │ inference    │ trade-off  │  │
│   │                     │                   │              │            │  │
│   │   ─────────────────────────────────────────────────────────────────│  │
│   │                     │                   │              │            │  │
│   │   Hybrid            │ AR + NAR          │ Balanced     │ Complexity │  │
│   │   (CosyVoice)       │ stages            │              │            │  │
│   │                     │                   │              │            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Performance Positioning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE COMPARISON MATRIX                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        Quality                                              │
│                           ▲                                                 │
│                           │                                                 │
│                      5.0  ┼─────────────────────────────────────────        │
│                           │                          ┌─────────┐            │
│                           │               ┌──────┐   │ VALL-E 2│            │
│                      4.5  ┼               │Audar │   └─────────┘            │
│                           │               │ TTS  │                          │
│                           │               └──────┘  ┌─────────┐             │
│                      4.0  ┼       ┌──────┐         │  F5-TTS │             │
│                           │       │XTTS  │         └─────────┘             │
│                           │       └──────┘                                  │
│                      3.5  ┼  ┌──────┐                                       │
│                           │  │Bark  │                                       │
│                           │  └──────┘                                       │
│                      3.0  ┼─────────┼─────────┼─────────┼─────────────►     │
│                           0.3x     0.5x      1.0x      2.0x     Speed      │
│                                                        (RTF)               │
│                                                                             │
│   Legend:                                                                   │
│   • Quality: Mean Opinion Score (MOS) scale                                 │
│   • Speed: Real-Time Factor (lower = faster)                                │
│                                                                             │
│   Audar-TTS Positioning:                                                    │
│   ✓ High quality (MOS ~4.3)                                                │
│   ✓ Fast inference (RTF ~0.46)                                             │
│   ✓ Low latency streaming                                                   │
│   ✓ Minimal reference audio required                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Technical Innovations

### 7.1 Key Differentiators

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TECHNICAL INNOVATIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. UNIFIED TEXT-SPEECH VOCABULARY                                         │
│   ─────────────────────────────────                                         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Traditional Approach          Audar-TTS Approach                   │  │
│   │   ────────────────────          ────────────────────                 │  │
│   │                                                                      │  │
│   │   Text Encoder ──┐              ┌─────────────────────┐             │  │
│   │                  ├──► Fusion    │  Unified Vocabulary │             │  │
│   │   Speech Encoder─┘              │  ┌───────┬────────┐ │             │  │
│   │                                 │  │ Text  │ Speech │ │             │  │
│   │   (Separate encoders,           │  │Tokens │ Codes  │ │             │  │
│   │    alignment required)          │  └───────┴────────┘ │             │  │
│   │                                 │  (Single embedding   │             │  │
│   │                                 │   space)             │             │  │
│   │                                 └─────────────────────┘             │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. STRUCTURED PROMPT TEMPLATE                                             │
│   ─────────────────────────────                                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Explicit delimiters enable:                                        │  │
│   │   • Clear separation of reference vs target                          │  │
│   │   • Robust text-speech alignment                                     │  │
│   │   • Controlled generation boundaries                                 │  │
│   │   • Reliable stop condition detection                                │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. CHUNKED STREAMING SYNTHESIS                                            │
│   ──────────────────────────────                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Code Buffer ──► Threshold Check ──► Decode ──► Yield              │  │
│   │        ▲                │                           │                │  │
│   │        │                │ (50 codes)                │                │  │
│   │        │                ▼                           │                │  │
│   │        └────────────── Continue ◄───────────────────┘                │  │
│   │                                                                      │  │
│   │   Benefits:                                                          │  │
│   │   • Constant memory usage                                            │  │
│   │   • Predictable latency                                              │  │
│   │   • Progressive playback                                             │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. API Reference

### 8.1 Core Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        AudarTTS Class                                │  │
│   │                                                                      │  │
│   │   Initialization                                                     │  │
│   │   ──────────────                                                     │  │
│   │   AudarTTS(                                                          │  │
│   │       model_path=None,     # Auto-download if not specified          │  │
│   │       voices_dir=None,     # Voice profiles directory                │  │
│   │       n_ctx=8192,          # Context window size                     │  │
│   │       lazy_load=False      # Defer initialization                    │  │
│   │   )                                                                  │  │
│   │                                                                      │  │
│   │   Core Methods                                                       │  │
│   │   ────────────                                                       │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │  speak(text, speaker, output, lang)                         │   │  │
│   │   │  └─► Returns: (audio_array, metrics_dict)                   │   │  │
│   │   │                                                              │   │  │
│   │   │  stream(text, speaker, lang, chunk_size)                    │   │  │
│   │   │  └─► Yields: (audio_chunk, metrics_dict)                    │   │  │
│   │   │                                                              │   │  │
│   │   │  clone_voice(name, audio_path, transcript, lang)            │   │  │
│   │   │  └─► Returns: Speaker object                                │   │  │
│   │   │                                                              │   │  │
│   │   │  voices                                                      │   │  │
│   │   │  └─► Returns: List[str] of available speakers               │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Usage Patterns                                  │  │
│   │                                                                      │  │
│   │   # Basic synthesis                                                  │  │
│   │   tts = AudarTTS()                                                   │  │
│   │   audio, metrics = tts.speak("Hello world", speaker="Eve")           │  │
│   │                                                                      │  │
│   │   # Streaming synthesis                                              │  │
│   │   for chunk, info in tts.stream("Long text...", speaker="Eve"):      │  │
│   │       play_audio(chunk)                                              │  │
│   │                                                                      │  │
│   │   # Voice cloning                                                    │  │
│   │   tts.clone_voice("custom", "reference.wav", "transcript")           │  │
│   │   audio, _ = tts.speak("New text", speaker="custom")                 │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Conclusion

Audar-TTS represents a production-ready implementation of neural codec language modeling for zero-shot text-to-speech synthesis. The system achieves:

- **Quality:** Natural prosody and high speaker similarity
- **Speed:** Real-time factor of 0.46x enables interactive applications
- **Flexibility:** Zero-shot cloning from minimal reference audio
- **Deployment:** Quantized models enable edge deployment

The architecture balances complexity with practicality, providing a robust foundation for voice synthesis applications.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **RTF** | Real-Time Factor - ratio of processing time to audio duration |
| **Neural Codec** | Neural network for encoding/decoding audio as discrete tokens |
| **Zero-Shot** | Capability to perform task without task-specific training |
| **Phoneme** | Distinct unit of sound in a language |
| **GGUF** | Model format optimized for efficient CPU/GPU inference |
| **KV-Cache** | Key-Value cache for efficient autoregressive generation |

---

## Appendix B: References

1. Wang, C., et al. "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." arXiv:2301.02111 (2023)
2. Défossez, A., et al. "High Fidelity Neural Audio Compression." arXiv:2210.13438 (2022)
3. Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971 (2023)
4. Chen, S., et al. "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers." arXiv:2406.05370 (2024)

---

*Copyright (c) 2025 Audar AI. All rights reserved.*
