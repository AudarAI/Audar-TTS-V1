"""Audar-TTS-V1 — official reference inference helpers.

    from audar_tts import load_model, load_codec, synthesize

`load_model` accepts a local path *or* a Hugging Face repo id, e.g.
`load_model("audarai/Audar-TTS-V1-Flash")`.
"""
from .audar_tts_infer import (
    GEN_DEFAULTS,
    HF_REPOS,
    OUT_SR,
    REF_SR,
    build_prompt,
    encode_reference,
    load_codec,
    load_model,
    synthesize,
)

__all__ = [
    "GEN_DEFAULTS",
    "HF_REPOS",
    "OUT_SR",
    "REF_SR",
    "build_prompt",
    "encode_reference",
    "load_codec",
    "load_model",
    "synthesize",
]
__version__ = "1.0.0"
