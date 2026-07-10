"""
Audar-TTS-V1 — official reference inference (Transformers).

Audar-TTS-V1 treats speech synthesis as next-token prediction: a decoder-only LM backbone
emits discrete `<|speech_N|>` audar-codec tokens from a structured zero-shot prompt, and a
neural codec decodes them to a 24 kHz waveform. Reference conditioning is a 5-15 s clip
(16 kHz mono) plus its transcript — no per-speaker fine-tuning, no phonemizer, no G2P.

Public API
----------
    model, tok = load_model("audarai/Audar-TTS-V1-Flash")   # HF repo id or local path
    codec      = load_codec()                                # NeuCodec-compatible decoder
    wav = synthesize(model, tok, codec,
                     "مرحبا! [whispers] أهلاً وسهلاً بك.",
                     ref_wav="reference.wav",
                     ref_text="transcript of the reference clip")
    # wav: float32 numpy array @ 24 kHz

This is the exact code path used to produce the Voice Gallery demos on the Hugging Face
model cards, so those samples are reproducible with it.

Note: only `Audar-TTS-V1-Flash` ships full-precision Transformers weights (in the
`transformers/` subfolder of its HF repo). `Audar-TTS-V1-Turbo` is distributed as GGUF for
llama.cpp — see `examples/synthesize_gguf.py`.
"""
from __future__ import annotations

import re

import numpy as np
import torch

REF_SR = 16_000   # codec encoder input rate (reference clip)
OUT_SR = 24_000   # codec decoder output rate

# Convenience map from tier name -> Hugging Face repo id.
HF_REPOS = {
    "flash": "audarai/Audar-TTS-V1-Flash",
    "turbo": "audarai/Audar-TTS-V1-Turbo",  # GGUF only — use examples/synthesize_gguf.py
}

# Voice Gallery demo settings (tested): tuned for maximum expressiveness. A low
# repetition_penalty (~1.1) is what lets [laughs] and other bursts through; lower
# temperature toward 0.6-0.7 for steadier, more neutral delivery.
GEN_DEFAULTS = dict(
    do_sample=True,
    temperature=1.0,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    min_new_tokens=50,
)

_SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")


def load_model(model_dir: str = HF_REPOS["flash"], device: str = "cuda:0", dtype=torch.bfloat16):
    """Load an Audar-TTS backbone (Transformers). Flash's bf16 weights live in the
    `transformers/` subfolder of the HF repo; a local path is used as-is."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"subfolder": "transformers"} if not model_dir.startswith(("/", ".")) else {}
    tok = AutoTokenizer.from_pretrained(model_dir, **kwargs)
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **kwargs)
        .eval()
        .to(device)
    )
    return model, tok


def load_codec(repo: str = "neuphonic/neucodec", device: str = "cuda:0"):
    """Load the codec used to encode the reference clip and decode generated tokens.

    audar-codec is Audar's Arabic-tuned NeuCodec; the tokens remain NeuCodec-compatible,
    so the public `neuphonic/neucodec` decodes them directly.
    """
    from neucodec import NeuCodec

    return NeuCodec.from_pretrained(repo).eval().to(device)


def encode_reference(codec, ref_wav) -> list[int]:
    """Encode a 5-15 s reference clip (path or float32 array @ 16 kHz) to codec ids."""
    if not isinstance(ref_wav, np.ndarray):
        import librosa

        ref_wav, _ = librosa.load(ref_wav, sr=REF_SR, mono=True)
    wav = torch.from_numpy(ref_wav.astype(np.float32))[None, None, :]
    device = next(codec.parameters()).device
    return codec.encode_code(wav.to(device)).squeeze().tolist()


def build_prompt(target_text: str, ref_text: str, ref_codes: list[int]) -> str:
    """Zero-shot reference-conditioned prompt — identical across all tiers."""
    ref = "".join(f"<|speech_{c}|>" for c in ref_codes)
    return (
        "user: Convert the text to speech:"
        f"<|REF_TEXT_START|>{ref_text}<|REF_TEXT_END|>"
        f"<|REF_SPEECH_START|>{ref}<|REF_SPEECH_END|>"
        f"<|TARGET_TEXT_START|>{target_text}<|TARGET_TEXT_END|>"
        "\nassistant:<|TARGET_CODES_START|>"
    )


def synthesize(
    model,
    tok,
    codec,
    target_text: str,
    ref_wav,
    ref_text: str,
    max_new_tokens: int = 1500,
    **gen_overrides,
) -> np.ndarray:
    """Synthesize `target_text` in the reference clip's voice. Returns float32 @ 24 kHz.

    `target_text` may carry inline expression tags, e.g.
    "Oh, you have to hear this — [excited] we just closed the deal!"
    (see docs/expression_tags.md for the per-tier tag sets).
    """
    ref_codes = encode_reference(codec, ref_wav)
    prompt = build_prompt(target_text, ref_text, ref_codes)

    ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    tce = tok.convert_tokens_to_ids("<|TARGET_CODES_END|>")
    gen = {**GEN_DEFAULTS, **gen_overrides}
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tce,
            pad_token_id=151643,
            **gen,
        )

    text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=False)
    codes = [int(x) for x in _SPEECH_RE.findall(text)]
    if not codes:
        raise RuntimeError("Model emitted no speech tokens — check the reference clip and prompt.")
    wav = codec.decode_code(torch.tensor(codes)[None, None, :].to(next(codec.parameters()).device))
    return wav.cpu().numpy()[0, 0, :]
