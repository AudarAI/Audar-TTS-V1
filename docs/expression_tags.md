# Expression-tag reference

Audar-TTS expression tags are **first-class vocabulary entries**, recognized inline in the
target text — no SSML, no separate control channel. A tag scopes the span it precedes and
composes with the reference voice: the voice profile fixes timbre and accent while the tag
modulates affect and prosody. Tags work in both Arabic and English text.

```text
Oh, you have to hear this — [excited] we just closed the biggest deal of the entire year!
لا يمكنني الانتظار لأخبرك — [excited] لقد أنجزنا المشروع أخيراً، [laughs] وصدّقني، إنه أجمل شعور!
```

## Tag availability by tier

Expression-tag coverage is a tier feature: **Flash** and **Turbo** ship a compact 8-tag set
focused on prosody and affect; **Pro** carries the full 17-tag vocabulary.

### Flash / Turbo — 8 tags

| Tag | Effect |
|---|---|
| `[laughs]` | audible laughter |
| `[curious]` | rising, inquisitive intonation |
| `[excited]` | energetic, fast, bright delivery |
| `[sighs]` | audible sigh |
| `[exhales]` | audible exhale |
| `[mischievously]` | playful, conspiratorial tone |
| `[whispers]` | whispered delivery |
| `[sarcastic]` | dry, ironic intonation |

### Pro — 17 tags

Eleven **active** tags are acoustically grounded paralinguistic events, each trained on
annotated examples across multiple speakers:

`[gasp]` · `[crying]` · `[giggles]` · `[shouting]` · `[trembling]` · `[cough]` · `[yawn]` ·
`[panicked]` · `[tired]` · `[very slow]` · `[very fast]`

Six further **prosody** tags shape delivery style:

`[laughs]` · `[whispers]` · `[sighs]` · `[excited]` · `[curious]` · `[sarcastic]`

On Pro, tags can also be **stacked** — e.g. `[whispers] [trembling]` for a frightened
whisper.

## Usage guidance

- **Place the tag immediately before the span** it should affect; its influence fades at the
  next tag or sentence boundary.
- **Use tags sparingly.** One or two per sentence sounds natural; dense tagging degrades
  prosody.
- **Sampling matters for expressiveness.** The Voice Gallery demos use
  `temperature=1.0 · top_k=40 · top_p=0.9 · repetition_penalty=1.1` — a low repetition
  penalty (≈1.1) is what lets laughter and other bursts through. For steadier, more neutral
  delivery, lower `temperature` toward `0.6–0.7`.
- **On Flash/Turbo, stacked or multi-word tags are more fragile** than single tags; prefer
  one tag per span on the smaller tiers.
- Unknown tags are read as literal text — stick to the tier's published set.
