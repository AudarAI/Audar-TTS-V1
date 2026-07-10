<div align="center">

# Audar-TTS-V1

### Arabic-first, expressive zero-shot text-to-speech — clone any voice, speak any dialect.

**From Arabic to the world.**

[![Benchmarks](https://img.shields.io/badge/Arabic%20TTS-top%20intelligibility%20tier%20%C2%B7%20two--judge%20eval-2ea44f)](benchmarks/README.md)
[![Models](https://img.shields.io/badge/🤗%20Models-Flash%20%2B%20Turbo-ffcc4d)](https://huggingface.co/audarai)
[![Paper](https://img.shields.io/badge/📄%20Technical%20Report-PDF-blue)](Audar-TTS-V1-Technical-Report.pdf)
[![Code License](https://img.shields.io/badge/code-Apache%202.0-6f42c1)](LICENSE)
[![Website](https://img.shields.io/badge/🌐%20audarai.com-informational)](https://www.audarai.com)

<p>
<a href="#-models"><b>Models</b></a> ·
<a href="#-quickstart"><b>Quickstart</b></a> ·
<a href="#-voices--expression-tags"><b>Voices & Tags</b></a> ·
<a href="#-benchmarks"><b>Benchmarks</b></a> ·
<a href="#-downloads"><b>Downloads</b></a> ·
<a href="#-licenses"><b>Licenses</b></a> ·
<a href="https://www.audarai.com"><b>Audar API</b></a>
</p>

</div>

---

**Audar-TTS-V1** is a family of Arabic-first, expressive zero-shot text-to-speech models from
[AudarAI](https://www.audarai.com). It recasts synthesis as next-token prediction over a joint
vocabulary of text, control tokens, and discrete speech codes — **no phonemizer and no
per-language G2P**, so dialect coverage comes from data rather than brittle pronunciation rules. Each model **clones any voice from a 5–15 second
reference clip** with no per-speaker fine-tuning, shapes delivery with **inline expression tags**
(`[laughs]`, `[whispers]`, `[excited]`, …), and speaks **MSA, dialectal Arabic (Gulf/Emirati,
Egyptian, Levantine, Maghrebi), code-switched Arabic–English, and English** at studio-clean
**24 kHz**. The family is adapted in-house on **200,000+ hours** of audio–text data, with 15,000
hours of dedicated zero-shot voice-cloning SFT and a curriculum ending in **KTO preference
alignment** over a five-axis rubric (naturalness, similarity, prosody, expression fidelity,
pronunciation).

On **public Arabic benchmarks scored by two ASR judges**, the three tiers sit in the top
intelligibility tier and are statistically inseparable from each other; under the dialect-aware
judge, the Pro tier posts **significantly lower MSA WER than GPT-4o-mini-TTS** (*p* = 0.008,
*n* = 2,902). This repository is the developer hub: model pointers, the voice-profile registry, the
expression-tag reference, benchmarks, and copy-paste inference.

## 🧩 Models

| | **Audar-TTS-V1-Flash** | **Audar-TTS-V1-Turbo** | **Audar-TTS-V1-Pro** |
|---|---|---|---|
| **Tier** | Real-time · edge | Production default | Studio / long-form |
| **Parameters** | 0.55 B | 1.64 B | 4 B |
| **Runtimes** | 🤗 Transformers · GGUF (llama.cpp) | GGUF (llama.cpp) | Enterprise — via AudarAI |
| **Expression tags** | 8 | 8 | 17 |
| **Best for** | On-device, high-throughput, interactive agents | Balanced quality & latency | Maximum expressiveness & fidelity |
| **License** | [AudarAI Open v1.0](https://www.audarai.com/license/audarai-open-license-v1.0/) | [AudarAI Community v1.0](https://www.audarai.com/license/audarai-community-license-v1.0/) | [AudarAI Enterprise](https://www.audarai.com/license/audarAI-enterprise-license-agreement-v1.0-template/) |
| **Download** | **[🤗 audarai/Audar-TTS-V1-Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash)** | **[🤗 audarai/Audar-TTS-V1-Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo)** | *Hugging Face page coming soon* — [contact us](mailto:contact@audarai.com) |

All tiers share **one prompt protocol and one voice-profile registry**, so an application can
move between them for the right quality/cost trade-off **without any code change**. On intelligibility the tiers are statistically indistinguishable
([benchmarks](benchmarks/README.md)) — pick by expressiveness and cost.

## ⚡ Quickstart

Weights download automatically from the Hugging Face repos above.

### Transformers — Flash (Python, GPU)

```bash
pip install -r examples/requirements.txt
python examples/synthesize.py "مرحبا! [whispers] أهلاً وسهلاً بك." \
    --ref reference.wav --ref-text "transcript of the reference clip"
```

```python
# Or call the reference helpers directly:
from audar_tts import load_model, load_codec, synthesize
import soundfile as sf

model, tok = load_model("audarai/Audar-TTS-V1-Flash")   # HF repo id or local path
codec = load_codec()
wav = synthesize(model, tok, codec,
                 "Oh, you have to hear this — [excited] we just closed the deal!",
                 ref_wav="reference.wav", ref_text="transcript of the reference clip")
sf.write("out.wav", wav, 24000)
```

### GGUF — Flash or Turbo (llama.cpp: CPU/GPU/edge)

```bash
pip install llama-cpp-python neucodec soundfile torch librosa huggingface_hub
python examples/synthesize_gguf.py "مرحبا! [whispers] أهلاً وسهلاً بك." \
    --ref reference.wav --ref-text "transcript of the reference clip" --tier turbo
```

Q4_K_M / Q5_K_M / Q8_0 quantizations are published for both tiers; Flash's Q4 runs in **under
0.5 GB**.

### The prompt protocol

One zero-shot, reference-conditioned format across all tiers — the reference text and speech codes
are delimited from the target text, and the model emits the target codes:

```text
user: Convert the text to speech:
<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>
<|REF_SPEECH_START|>{ref_codes}<|REF_SPEECH_END|>
<|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>
assistant: <|TARGET_CODES_START|>{generated_codes}<|TARGET_CODES_END|>
```

A 5–15 s reference clip at 16 kHz is sufficient to clone a voice — **consented voices only**.

> 🎛️ **Sampling:** the Voice Gallery demos use `temperature=1.0 · top_k=40 · top_p=0.9 ·
> repetition_penalty=1.1` — a low repetition penalty (≈1.1) is what lets `[laughs]` through.
> Lower `temperature` toward `0.6–0.7` for steadier, more neutral delivery.

## 🎭 Voices & expression tags

- **[Voice-profile registry](voices/)** — six ready-to-use synthetic voices (interpolated from
  multiple speakers; they resemble no real individual), shared by all tiers and free to use.
  Machine-readable: [`voices/registry.json`](voices/registry.json). Listen to them in the Voice
  Gallery on the [Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash) and
  [Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo) model cards.
- **[Expression-tag reference](docs/expression_tags.md)** — inline tags are first-class vocabulary
  entries, no SSML. Flash/Turbo ship an 8-tag set
  (`[laughs]` `[curious]` `[excited]` `[sighs]` `[exhales]` `[mischievously]` `[whispers]`
  `[sarcastic]`); Pro carries the full 17-tag vocabulary (11 acoustically-grounded active tags +
  6 prosody tags) and supports tag stacking. Tags work in both Arabic and English.

## 📊 Benchmarks

Two evaluations, one lesson: **judge choice decides Arabic TTS rankings**, so we score every clip
with two ASR judges (MSA-leaning Whisper large-v3 + a dialect-aware judge) and report both.

- **Public two-judge intelligibility** (Habibi MSA/Saudi/UAE + a 2,902-clip full-MSA tie-breaker):
  all three tiers sit in the **top intelligibility tier** and are **statistically inseparable from
  each other**; under the dialect-aware judge **Pro significantly beats GPT-4o-mini-TTS on MSA**
  (7.34 % vs 7.86 % WER, *p* = 0.008) — the same system that leads under the Whisper judge.
- **In-House Expressive Benchmark** (1,364 clips, 10 systems): Pro **beats ElevenLabs v3 on
  resynthesis WER (−6.0 % relative) and SQUIM MOS, ties it on expression fidelity**, and posts the
  **best Gulf-dialect WER** of the compared systems.

**Full tables, protocol, significance tests, and machine-readable CSVs →
[benchmarks/](benchmarks/README.md).**

## 📥 Downloads

All weights and GGUF variants live on Hugging Face:

- **Flash** (Transformers + GGUF): **https://huggingface.co/audarai/Audar-TTS-V1-Flash**
- **Turbo** (GGUF): **https://huggingface.co/audarai/Audar-TTS-V1-Turbo**
- **Pro**: Hugging Face page coming soon — enterprise access via [contact@audarai.com](mailto:contact@audarai.com)
- Full model collection: **https://huggingface.co/audarai**

Prefer a managed, production-hosted endpoint (`client.tts`, model ids `audar-tts-v1-flash`,
`audar-tts-v1-turbo`)? See the **[Audar API](https://www.audarai.com)**.

## 🌍 Languages & capabilities

- **Primary**: Arabic — MSA and dialectal (Gulf/Emirati, Egyptian, Levantine, Maghrebi), plus
  **code-switched Arabic–English**; raw text in, no diacritization step to fail on
  diacritic-free input.
- **Also**: English, fully bilingual.
- **Tasks**: zero-shot voice cloning (5–15 s reference), expressive synthesis via inline tags,
  long-form narration (chunk long inputs at sentence boundaries).
- **Limitations**: stacked/multi-word tags are more fragile on Flash/Turbo; rare names, numbers,
  and code-switch boundaries can be mispronounced, as with all neural TTS.

## 📜 Licenses

- **This repository** (example code, reference helpers, docs) — **Apache-2.0** ([LICENSE](LICENSE)).
- **Model weights** — released under AudarAI model licenses (not Apache-2.0):
  - Audar-TTS-V1-Flash → **[AudarAI Open License v1.0](https://www.audarai.com/license/audarai-open-license-v1.0/)** (commercial use, redistribution, and fine-tuning permitted).
  - Audar-TTS-V1-Turbo → **[AudarAI Community License v1.0](https://www.audarai.com/license/audarai-community-license-v1.0/)** (research and limited commercial use for qualifying Community Entities).
  - Audar-TTS-V1-Pro → **[AudarAI Enterprise License Agreement v1.0](https://www.audarai.com/license/audarAI-enterprise-license-agreement-v1.0-template/)**.
- **Enterprise / large-scale / model-as-a-service** use may require an **AudarAI Enterprise License** —
  contact **contact@audarai.com** or visit **[audarai.com](https://www.audarai.com)**.

## 🛡️ Responsible use

Zero-shot cloning is intended for **consented voices only**. Do not use Audar-TTS to deceive,
defraud, or impersonate real people or organizations, and comply with applicable law. The shipped
voice profiles are synthetic and resemble no real individual.

## 📄 Citation

Technical report — *Audar-TTS-V1: A Multilingual, Arabic-First Expressive Speech Synthesis
Foundation Model*, Audar AI Team, 2026 —
**[read the PDF](Audar-TTS-V1-Technical-Report.pdf)** (also attached to the
[latest release](https://github.com/AudarAI/Audar-TTS-V1/releases/latest)).

```bibtex
@techreport{audar-tts-2026,
  title       = {Audar-TTS-V1: A Multilingual, Arabic-First Expressive Speech Synthesis Foundation Model},
  author      = {Audar AI Team},
  institution = {AudarAI},
  year        = {2026},
  url         = {https://github.com/AudarAI/Audar-TTS-V1}
}
```

---

<div align="center">

### Leading Arabic-First Multilingual Audio Intelligence

*AudarAI starts with Arabic — and expands to the world.*

**Arabic-first. Multilingual by design. Human-centered at heart.**

**[🌐 www.audarai.com](https://www.audarai.com)** · [🤗 Hugging Face](https://huggingface.co/audarai) · [GitHub](https://github.com/AudarAI) · contact@audarai.com

© 2026 AUDARAI PTE. LTD.

</div>
