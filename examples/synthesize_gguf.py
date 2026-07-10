#!/usr/bin/env python3
"""Zero-shot speech synthesis with Audar-TTS-V1 GGUF via llama.cpp — CPU, GPU, or edge.

Works for both tiers (Flash and Turbo); quantized weights download automatically from
Hugging Face.

    pip install llama-cpp-python neucodec soundfile torch librosa huggingface_hub
    python examples/synthesize_gguf.py "مرحبا! [whispers] أهلاً وسهلاً بك." \
        --ref reference.wav --ref-text "transcript of the reference clip" --tier turbo

CPU by default; pass --gpu-layers -1 to offload the whole backbone to GPU.
"""
import argparse
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audar_tts import HF_REPOS, OUT_SR, build_prompt, encode_reference

GGUF_FILES = {
    "flash": "Audar-TTS-V1-Flash-Q4_K_M.gguf",
    "turbo": "Audar-TTS-V1-Turbo-Q4_K_M.gguf",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize speech with Audar-TTS-V1 GGUF (llama.cpp).")
    ap.add_argument("text", help="Target text (Arabic and/or English; inline [tags] allowed).")
    ap.add_argument("--ref", required=True, help="Reference voice clip, 5-15 s (consented voices only).")
    ap.add_argument("--ref-text", required=True, help="Transcript of the reference clip.")
    ap.add_argument("--tier", choices=["flash", "turbo"], default="flash")
    ap.add_argument("--gguf", default=None, help="Override the GGUF filename (e.g. a Q8_0 variant).")
    ap.add_argument("--out", default="out.wav", help="Output wav path (24 kHz).")
    ap.add_argument("--gpu-layers", type=int, default=0, help="-1 offloads all layers to GPU.")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from neucodec import NeuCodec

    # 1) Backbone (GGUF) + codec. audar-codec tokens are NeuCodec-compatible, so the
    #    public neuphonic/neucodec decodes them directly.
    gguf = hf_hub_download(HF_REPOS[args.tier], args.gguf or GGUF_FILES[args.tier])
    llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=args.gpu_layers, verbose=False)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval()

    # 2) Encode the zero-shot reference and build the shared prompt.
    prompt = build_prompt(args.text, args.ref_text, encode_reference(codec, args.ref))

    # 3) Generate speech tokens; stop at <|TARGET_CODES_END|>.
    tce = llm.tokenize(b"<|TARGET_CODES_END|>", add_bos=False, special=True)[0]
    toks = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    ids = []
    for tid in llm.generate(toks, temp=1.0, top_k=40, top_p=0.9, repeat_penalty=1.1):
        if tid == tce or len(ids) >= 2048:
            break
        ids.append(tid)
    text = "".join(llm.detokenize([t], special=True).decode("utf-8", "ignore") for t in ids)

    # 4) Decode to 24 kHz audio.
    codes = [int(x) for x in re.findall(r"<\|speech_(\d+)\|>", text)]
    wav = codec.decode_code(torch.tensor(codes)[None, None, :]).cpu().numpy()[0, 0, :]
    sf.write(args.out, wav, OUT_SR)
    print(f"wrote {args.out} ({len(wav) / OUT_SR:.1f}s @ {OUT_SR} Hz)")


if __name__ == "__main__":
    main()
