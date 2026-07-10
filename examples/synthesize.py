#!/usr/bin/env python3
"""Zero-shot speech synthesis with Audar-TTS-V1-Flash via 🤗 Transformers (bf16, GPU).

    pip install -r examples/requirements.txt
    python examples/synthesize.py "مرحبا! [whispers] أهلاً وسهلاً بك." \
        --ref reference.wav --ref-text "transcript of the reference clip"
    python examples/synthesize.py "Oh, [excited] we just closed the deal!" \
        --ref reference.wav --ref-text "..." --out deal.wav

The reference clip is any consented 5-15 s voice sample (16 kHz mono is ideal; other rates
are resampled). Weights download automatically from
https://huggingface.co/audarai/Audar-TTS-V1-Flash
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import the repo's audar_tts package
from audar_tts import GEN_DEFAULTS, HF_REPOS, OUT_SR, load_codec, load_model, synthesize


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize speech in a cloned voice with Audar-TTS-V1-Flash.")
    ap.add_argument("text", help="Target text (Arabic and/or English; inline [tags] allowed).")
    ap.add_argument("--ref", required=True, help="Reference voice clip, 5-15 s (consented voices only).")
    ap.add_argument("--ref-text", required=True, help="Transcript of the reference clip.")
    ap.add_argument("--out", default="out.wav", help="Output wav path (24 kHz).")
    ap.add_argument("--model", default=HF_REPOS["flash"], help="HF repo id or local path.")
    ap.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0, cpu).")
    ap.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"],
                    help="1.0 = expressive demo setting; 0.6-0.7 = steadier, more neutral.")
    args = ap.parse_args()

    model, tok = load_model(args.model, device=args.device)
    codec = load_codec(device=args.device)
    wav = synthesize(model, tok, codec, args.text, ref_wav=args.ref, ref_text=args.ref_text,
                     temperature=args.temperature)
    sf.write(args.out, wav, OUT_SR)
    print(f"wrote {args.out} ({len(wav) / OUT_SR:.1f}s @ {OUT_SR} Hz)")


if __name__ == "__main__":
    main()
