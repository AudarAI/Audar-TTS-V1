<h1 align="center">
  <br>
  Audar TTS
  <br>
</h1>

<h3 align="center">Expressive Multilingual Voice AI</h3>

<p align="center">
  <strong>State-of-the-art Arabic + English TTS with expressive tags, zero-shot voice cloning, and real-time streaming</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/audarai/audar_tts_flash_v1_gguf">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface" alt="HuggingFace"/>
  </a>
  <a href="https://dev.audarai.com">
    <img src="https://img.shields.io/badge/Try-Demo-blue?logo=googlechrome" alt="Demo"/>
  </a>
  <a href="https://www.audarai.com">
    <img src="https://img.shields.io/badge/Website-audarai.com-green" alt="Website"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"/>
  <img src="https://img.shields.io/github/stars/AudarAI/Audar-TTS-V1?style=social" alt="Stars"/>
</p>

---

<div align="center">
  <h2>Listen to the Difference</h2>
  <p><strong>Side-by-side comparison: Audar TTS vs ElevenLabs</strong></p>
  <p>
    <a href="https://audarai.github.io/Audar-TTS-V1/comparison.html">
      <img src="https://img.shields.io/badge/OPEN%20INTERACTIVE%20PLAYER-Listen%20Now-blue?style=for-the-badge&logo=headphones&logoColor=white" alt="Interactive Comparison Player">
    </a>
  </p>
</div>

---

## Why Audar TTS?

<table>
<tr>
<td width="50%">

### Expressive Tags in Open Source

Unlike competitors who gate expressive speech behind premium tiers, **Audar TTS Flash** supports expressive tags out of the box:

```
[laughs] That's hilarious!
[whispers] Can you keep a secret?
[sighs] It's been a long day...
```

</td>
<td width="50%">

### Native Arabic + English

True bilingual architecture - not translation-based. Seamless code-switching mid-sentence:

```
"Welcome to our event, مرحباً بكم في حدثنا,
where innovation meets tradition."
```

</td>
</tr>
<tr>
<td width="50%">

### Zero-Shot Voice Cloning

Clone any voice from just **5 seconds** of audio. No fine-tuning required.

```python
tts.clone_voice("my_voice", "sample.wav")
tts.speak("Hello!", speaker="my_voice")
```

</td>
<td width="50%">

### Production Ready

Real-time streaming with **<500ms** first-chunk latency. Optimized for deployment.

```python
for chunk in tts.stream("Long text..."):
    play_audio(chunk)
```

</td>
</tr>
</table>

---

## Model Family

Audar TTS is available in three tiers to match your needs:

| | **Flash** | **Turbo** | **Pro** |
|:--|:--:|:--:|:--:|
| **Parameters** | 0.5B | 1.5B | 4B |
| **Training Data** | ~200K hrs (Open + Proprietary) | Massive Proprietary | Massive Proprietary |
| **Languages** | English, Arabic | English, Arabic + more | English, Arabic + more |
| **Zero-Shot Cloning** | 5s audio | 5s audio | 5s audio |
| **Expressive Tags** | Full Support | Full Support | Full Support |
| **Code-Switching** | Native | Enhanced | Enhanced |
| **First Chunk Latency** | ~500ms | ~400ms | ~300ms |
| **Voice Quality** | Excellent | Superior | Studio-grade |
| **License** | Apache 2.0 | Commercial | Commercial |
| **Access** | [**GitHub**](#quick-start) | [**Try Demo**](https://dev.audarai.com) | [**Try Demo**](https://dev.audarai.com) |

> **Flash** is fully open-source. **Turbo** and **Pro** are available through our commercial API with enhanced capabilities and dedicated support.

---

## Architecture

<p align="center">
  <img src="assets/audar_tts_diagrams.png?v=2" alt="Audar TTS Architecture" width="100%"/>
</p>

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/AudarAI/Audar-TTS-V1.git
cd Audar-TTS-V1
pip install -r requirements.txt

# System dependency
brew install espeak-ng  # macOS
# sudo apt-get install espeak-ng  # Ubuntu
```

### 2. Synthesize Speech

```python
from audar_tts import AudarTTS

tts = AudarTTS()

# Basic synthesis
audio, metrics = tts.speak(
    "Hello! Welcome to Audar TTS.",
    speaker="Eve",
    output="hello.wav"
)

# With expressive tags
audio, metrics = tts.speak(
    "[laughs] That was amazing! [sighs] But now back to work.",
    speaker="Eve",
    output="expressive.wav"
)
```

### 3. Clone a Voice

```python
# Clone from 5 seconds of audio
tts.clone_voice(
    name="my_voice",
    audio_path="reference.wav",
    transcript="What I said in the recording"
)

# Use cloned voice
tts.speak("Now speaking in the cloned voice!", speaker="my_voice")
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Zero-Shot Cloning** | Clone any voice from 5 seconds of audio |
| **Expressive Tags** | `[laughs]`, `[whispers]`, `[sighs]` and more |
| **Code-Switching** | Seamless Arabic + English in same sentence |
| **Real-Time Streaming** | <500ms latency, chunk-by-chunk generation |
| **7 Built-in Voices** | Eve, Salem, Amal, Salama, Amin, Hanaa, Wadee |
| **GGUF Support** | Optimized for CPU inference |

---

## Audio Comparison

### English Samples

| Speaker | Text | Audar TTS Flash | ElevenLabs Flash |
|---------|------|:---------------:|:----------------:|
| **Eve** | *"The future of artificial intelligence lies not in replacing human creativity, but in amplifying it beyond imagination."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_en_1_eve.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_en_1_eve.mp3) |
| **Salem** | *"After years of research, scientists finally discovered that the key to longevity was surprisingly simple: genuine human connection."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_en_2_salem.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_en_2_salem.mp3) |
| **Amal** | *"In the quiet moments between chaos and calm, we often find the answers we've been searching for all along."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_en_3_amal.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_en_3_amal.mp3) |

### Arabic Samples

| Speaker | Text | Audar TTS Flash | ElevenLabs Flash |
|---------|------|:---------------:|:----------------:|
| **Salama** | *"في عالم يتسارع فيه كل شيء، تبقى الحكمة هي البوصلة التي ترشدنا نحو القرارات الصائبة."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_ar_1_salama.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_ar_1_salama.mp3) |
| **Amin** | *"لا تقاس قيمة الإنسان بما يملك، بل بما يقدم للآخرين من خير وعطاء دون انتظار مقابل."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_ar_2_amin.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_ar_2_amin.mp3) |
| **Hanaa** | *"كل رحلة ألف ميل تبدأ بخطوة واحدة، فلا تستهن بالبدايات الصغيرة التي تصنع المستقبل العظيم."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_ar_3_hanaa.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_ar_3_hanaa.mp3) |

### Code-Switching (EN/AR)

| Speaker | Text | Audar TTS Flash | ElevenLabs Flash |
|---------|------|:---------------:|:----------------:|
| **Eve** | *"Welcome to our innovation hub, مرحباً بكم في مركز الابتكار، where ideas transform into reality."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_mix_1_eve.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_mix_1_eve.mp3) |
| **Amal** | *"The team worked tirelessly, والنتيجة كانت مذهلة، delivering beyond all expectations."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_mix_2_amal.mp3) | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/elevenlabs_mix_2_amal.mp3) |

### Expressive Tags (Audar TTS Feature)

Expressive tags are supported in **Audar TTS Flash** - competitors typically reserve this for premium tiers.

| Speaker | Tag | Text | Audar TTS Flash |
|---------|-----|------|:---------------:|
| **Eve** | `[laughs]` | *"I can't believe you actually did that! [laughs] That's the funniest thing I've heard all week!"* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_expr_1_eve.mp3) |
| **Salama** | `[whispers]` | *"[whispers] Listen carefully, I'm only going to say this once. The secret ingredient is... love."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_expr_2_salama.mp3) |
| **Amal** | `[sighs]` | *"[sighs] After everything we've been through, I never thought we'd actually make it here."* | [Play](https://audarai.github.io/Audar-TTS-V1/samples/comparison/audar_expr_3_amal.mp3) |

---

## Ready for Production?

<div align="center">

**Audar TTS Turbo and Pro** deliver enhanced capabilities for production workloads:

| | Turbo | Pro |
|:--|:--:|:--:|
| Lower latency | ~400ms | ~300ms |
| Voice consistency | Superior | Studio-grade |
| Additional languages | Coming soon | Coming soon |
| Priority support | Included | Dedicated |

<br/>

<a href="https://dev.audarai.com">
  <img src="https://img.shields.io/badge/Try%20Commercial%20Models-dev.audarai.com-blue?style=for-the-badge" alt="Try Demo"/>
</a>

</div>

---

## Documentation

| Document | Description |
|----------|-------------|
| [**BENCHMARKS.md**](BENCHMARKS.md) | Comprehensive performance benchmarks vs competitors |
| [**MODELS.md**](MODELS.md) | Detailed model comparison and specifications |
| [**TECHNICAL_REPORT.md**](TECHNICAL_REPORT.md) | Architecture deep-dive and research |
| [**docs/streaming.md**](docs/streaming.md) | Streaming synthesis guide |
| [**docs/voice-cloning.md**](docs/voice-cloning.md) | Voice cloning tutorial |

---

## Available Voices

| Name | Style | Languages |
|------|-------|-----------|
| **Eve** | Clear, expressive | EN, AR |
| **Salem** | Professional | EN, AR |
| **Amal** | Warm, hopeful | EN, AR |
| **Salama** | Calm, peaceful | EN, AR |
| **Amin** | Trustworthy | EN, AR |
| **Hanaa** | Bright, happy | EN, AR |
| **Wadee** | Gentle, soft | EN, AR |

---

## About Audar AI

**Audar AI** is pioneering expressive multilingual voice synthesis. We're on a mission to make advanced speech AI accessible across languages and cultures.

Our open-source **Flash** model represents our commitment to advancing the field through transparency, while our commercial **Turbo** and **Pro** models deliver production-grade capabilities trained on massive proprietary datasets.

<p>
  <a href="https://www.audarai.com">Website</a> ·
  <a href="https://linkedin.com/company/audarai">LinkedIn</a> ·
  <a href="https://dev.audarai.com">Try Demo</a>
</p>

---

## License

- **Audar TTS Flash**: Apache 2.0 - Free for commercial and non-commercial use
- **Audar TTS Turbo/Pro**: Commercial license - [Contact us](https://www.audarai.com)

---

<p align="center">
  Built with passion by <a href="https://www.audarai.com">Audar AI</a><br/>
  <em>Making expressive voice AI accessible worldwide</em>
</p>
