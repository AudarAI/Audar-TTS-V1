# Voice-profile registry

Six ready-to-use voices ship with Audar-TTS, **free to use** and shared by all tiers (one
prompt protocol — a profile that works on Flash works unchanged on Turbo and Pro). They are
**synthetic voices created by interpolating multiple speakers — they do not replicate or
resemble any real individual.**

Machine-readable registry: [`registry.json`](registry.json). Listen to every voice speaking
fresh Arabic and English lines in the Voice Gallery on the model cards:
[Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash) ·
[Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo).

| Voice | Gender | Style |
|---|---|---|
| `demo_male_1` | male | warm, confident |
| `demo_male_2` | male | soft, intimate |
| `demo_male_3` | male | bright, curious |
| `demo_female_1` | female | vibrant, joyful |
| `demo_female_2` | female | velvety, playful |
| `demo_female_3` | female | airy, dreamy |

## Using a profile

A profile is a zero-shot reference — a 5–15 s clip (16 kHz mono) plus its transcript —
passed as the `REF_TEXT` / `REF_SPEECH` conditioning of the prompt protocol:

```bash
python examples/synthesize.py "مرحبا! [whispers] أهلاً وسهلاً بك." \
    --ref demo_female_1_source.wav --ref-text "<transcript of the clip>"
```

## Custom voices

Any consented 5–15 s clip works the same way — zero-shot, no per-speaker fine-tuning.
**Clone voices only with the speaker's explicit consent**; see Responsible use in the
[README](../README.md#responsible-use).
