# Benchmarks — Audar-TTS-V1

Two complementary evaluations from the Audar-TTS-V1 technical report:

1. **[Public two-judge intelligibility](#1-public-two-judge-intelligibility-canonical)** —
   the canonical eval, on public Arabic test sets with frozen manifests. Machine-readable copy:
   [`public_habibi_two_judge.csv`](public_habibi_two_judge.csv).
2. **[In-House Expressive Benchmark](#2-in-house-expressive-benchmark)** — 1,364 clips across
   10 systems for expression fidelity, similarity, and per-dialect breakdowns. Machine-readable
   copy: [`inhouse_expressive.csv`](inhouse_expressive.csv).

**Headlines**

- All three Audar-TTS tiers sit in the **top intelligibility tier** on public Arabic
  benchmarks, and the tiers are **statistically inseparable from each other** on every set —
  pick a tier by cost and expressiveness, not intelligibility.
- Under the **dialect-aware judge**, Audar-TTS-V1-Pro posts **significantly lower MSA WER than
  GPT-4o-mini-TTS** (7.34 % vs 7.86 %, *p* = 0.008, *n* = 2,902) — the strongest system under
  the Whisper judge.
- **Rankings are judge-dependent** — the two ASR judges reverse the top-tier ordering, so any
  single-judge Arabic TTS leaderboard is judge-confounded. We report both judges throughout.
- On the in-house benchmark, Pro **beats ElevenLabs v3 on resynthesis WER (−6.0 % relative)
  and SQUIM MOS, ties it on expression fidelity**, and posts the **best Gulf-dialect WER**
  (0.165) of the compared systems.

---

## 1. Public two-judge intelligibility (canonical)

### Protocol

- **Test sets:** the public **Habibi** MSA, Saudi, and Emirati (UAE) sets — 250 clips each per
  system, frozen manifests, the *same* texts synthesized by every system — plus a
  **2,902-clip full-MSA tie-breaker** for the top systems.
- **Two judges per clip.** Arabic ASR judges differ in dialect handling, so every clip is
  scored by **Whisper large-v3** (WER_W; MSA-leaning) *and* the dialect-aware
  **Omnilingual-ASR-LLM-7B** prompted with dialect codes (WER_O).
- **Ground-truth floor:** the real spontaneous dialectal recordings are scored under the same
  pipeline; model scores should be read relative to that floor.
- **Significance:** paired bootstrap and Wilcoxon tests on shared texts.
- ElevenLabs v3 is run with its fixed MSA voice on all subsets.

### Results — WER % (lower is better; 250 clips per cell)

**Bold** = best system per column.

| System | WER_W MSA | WER_W Saudi | WER_W UAE | WER_O MSA | WER_O Saudi | WER_O UAE | Macro avg (WER_O) |
|---|--:|--:|--:|--:|--:|--:|--:|
| *Ground-truth floor* | *13.1* | *45.2* | *4.2* | *11.5* | *28.1* | *11.8* | *17.1* |
| **Audar-TTS-V1-Flash** | 7.9 | 15.7 | 6.2 | 6.4 | 14.9 | 7.1 | **9.5** |
| **Audar-TTS-V1-Turbo** | 7.9 | 15.0 | 6.6 | 7.6 | 14.7 | 6.9 | 9.7 |
| **Audar-TTS-V1-Pro** | 7.3 | 15.8 | 6.6 | **5.9** | 15.0 | 7.4 | **9.4** |
| GPT-4o-mini-TTS | **6.2** | **13.7** | **5.1** | 7.2 | 16.0 | 9.0 | 10.7 |
| Gemini-2.5-Flash-TTS | 7.8 | 15.3 | 7.6 | 6.7 | **14.4** | 7.8 | 9.6 |
| ElevenLabs v3 | 10.2 | 16.2 | 7.6 | 7.4 | 15.4 | **6.2** | 9.7 |

### Rankings are judge-dependent

The two judges **reverse the ordering of the top tier**: under the MSA-leaning Whisper judge,
GPT-4o-mini-TTS leads every column, while under the dialect-aware judge the Audar-TTS tiers
and Gemini-2.5-Flash populate the leading group (ElevenLabs v3 takes the Emirati column) and
Pro posts the best MSA score. The full-MSA tie-breaker (*n* = 2,902) makes this precise:

| Judge | Outcome |
|---|---|
| Whisper large-v3 | GPT-4o-mini-TTS ahead of all three Audar tiers (6.5 % vs 7.6–7.8 %, *p* < 0.0001) |
| Dialect-aware (Omnilingual) | **Pro significantly ahead of GPT-4o-mini-TTS (7.34 % vs 7.86 %, *p* = 0.008)**; Flash/Turbo directionally better too (7.3 % / 7.4 %, statistical ties) |

An MSA-biased judge forgives MSA-fied pronunciation of dialectal text; a dialect-aware judge
rewards dialect authenticity. **Any single-judge leaderboard of Arabic TTS is therefore
judge-confounded**, and we report both.

### The tiers are interchangeable on intelligibility

Across MSA, Saudi, and Emirati sets, **all pairwise comparisons among Flash, Turbo, and Pro
are statistical ties under both judges** — even at *n* = 2,902. The advantage also holds under
*true zero-shot* conditioning (every system cloning the benchmark's own reference voices):
the Audar tiers retain the strongest MSA and Emirati intelligibility of the systems evaluated
(zero-shot MSA WER 8.3–9.7 % across the tiers, vs 11.5 % for the next-best cloner).

---

## 2. In-House Expressive Benchmark

**1,364 scored clips across 10 systems** (8 Arabic-supporting + 2 English-only reference
baselines), 4 objective metric families. This benchmark probes what the public sets cannot:
**expression-tag fidelity, speaker similarity, and per-dialect coverage** (incl. Levantine and
Maghrebi). Audar-TTS-V1-Pro is the benchmarked headline configuration; Turbo runs behind the
same interface.

### Metrics

| Metric | How | Direction |
|---|---|---|
| WER / CER | ASR-resynthesis drift via **ElevenLabs Scribe v2** (external ASR, not our own) | lower ↓ |
| UTMOS / SQUIM | Predicted MOS — **UTMOSv2** and **SQUIM_SUBJECTIVE**, kept separate because they disagree | higher ↑ |
| SIM | **WavLM-base-plus-sv** embedding cosine to the reference | higher ↑ |
| EXPR | Expression fidelity — openSMILE eGeMAPSv02 + wav2vec2-superb-er | higher ↑ |

### Results (WER/CER as fractions; **ours in bold**)

| System | WER ↓ | CER ↓ | UTMOS ↑ | SQUIM ↑ | SIM ↑ | EXPR ↑ | n |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Audar-TTS-V1-Pro** | 0.167 | 0.055 | 3.09 | 4.19 | 0.925 | 0.990 | 247 |
| **Audar-TTS-V1-Turbo** | 0.224 | 0.271 | 2.96 | — | 0.931 | 0.943 | 184 |
| ElevenLabs v3 | 0.177 | 0.045 | 3.22 | 4.16 | 0.940 | 0.990 | 300 |
| GPT-4o-mini-TTS | 0.199 | 0.096 | 3.39 | 4.37 | 0.957 | 0.991 | 103 |
| Gemini-2.5-Flash | 0.164 | 0.041 | 2.98 | 4.18 | 0.930 | 0.989 | 123 |
| ElevenLabs Flash | 0.208 | 0.111 | 2.88 | 4.26 | 0.951 | 0.988 | 180 |
| Orpheus Arabic | 0.266 | 0.109 | 3.01 | 4.38 | 0.980 | 0.952 | 69 |
| GPT-4o-audio-preview | 0.538 | 0.404 | 2.85 | 4.13 | 0.920 | 0.926 | 93 |
| Kokoro *(en-only)* | 0.076 | 0.094 | 3.73 | 4.24 | 0.989 | 0.990 | 34 |
| Orpheus 3B *(en-only)* | 0.073 | 0.081 | 3.31 | 4.32 | 0.988 | 0.974 | 31 |

Against ElevenLabs v3 specifically, Pro **wins on WER** (0.167 vs 0.177, −6.0 % relative) and
**SQUIM MOS** (4.19 vs 4.16), **ties on EXPR** (both 0.990), and trails on UTMOS, CER, and
speaker similarity (SIM 0.925 vs 0.940).

### Per-dialect WER (Pro vs the primary Arabic-supporting systems)

Gulf *n* = 80, MSA *n* = 74, Egyptian *n* = 60, Levantine *n* = 42, Maghrebi *n* = 24.
**Bold** = best per column.

| System | MSA | Gulf | Egyptian | Levantine | Maghrebi |
|---|--:|--:|--:|--:|--:|
| **Audar-TTS-V1-Pro** | **0.202** | **0.165** | 0.213 | 0.316 | **0.335** |
| ElevenLabs v3 | 0.240 | 0.221 | **0.211** | 0.308 | 0.367 |
| GPT-4o mini | 0.217 | 0.186 | 0.217 | **0.276** | 0.440 |
| Orpheus Arabic | 0.219 | 0.213 | 0.214 | 0.317 | 0.385 |

Pro posts the **best Gulf, MSA, and Maghrebi WER** among the compared systems. All systems
degrade on Levantine and Maghrebi, consistent with their lower representation in current
Arabic TTS training corpora.

### Caveats (read before quoting numbers)

- Predicted-MOS metrics are **not calibrated for Arabic** and cannot separate the top tier —
  in one case a system scored competitive UTMOSv2 while producing largely unintelligible
  Arabic. We anchor evaluation on intelligibility.
- Similarity/expression proxies are English-biased.
- A formal **human MOS/CMOS listening study is planned, not yet completed**.

Full protocol, significance tests, and per-tag expression-fidelity breakdowns are in the
Audar-TTS-V1 technical report (see the [README citation](../README.md#-citation)).
