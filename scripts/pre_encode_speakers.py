#!/usr/bin/env python3
"""
Pre-encode speaker audio to cached .pt files for faster inference.

This script generates pre-computed speaker codes that can be loaded instantly
during inference, eliminating the codec encoding overhead at runtime.

Usage:
    python scripts/pre_encode_speakers.py --voices-dir voices/ --output-dir voices/cache/
    python scripts/pre_encode_speakers.py  # Uses default paths
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Speaker metadata - All speakers are multilingual (EN/AR) by default
SPEAKER_METADATA = {
    "Eve": {"gender": "female", "lang": "multilingual", "description": "Clear expressive voice"},
    "Salama": {"gender": "female", "lang": "multilingual", "description": "Salama - Peace (سلامة)"},
    "Amal": {"gender": "female", "lang": "multilingual", "description": "Amal - Hope (أمل)"},
    "Hanaa": {"gender": "female", "lang": "multilingual", "description": "Hanaa - Happiness (هناء)"},
    "Salem": {"gender": "male", "lang": "multilingual", "description": "Salem - Peaceful (سالم)"},
    "Amin": {"gender": "male", "lang": "multilingual", "description": "Amin - Trustworthy (أمين)"},
    "Wadee": {"gender": "male", "lang": "multilingual", "description": "Wadee - Gentle (وديع)"},
}


def encode_audio(audio_path: Path, codec) -> List[int]:
    """Encode audio file to discrete codes using DistillNeuCodec."""
    import librosa
    
    wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        codes = codec.encode_code(audio_or_path=wav_tensor)
        return codes.squeeze(0).squeeze(0).tolist()


def phonemize_text(text: str, lang: str = "ar") -> str:
    """Convert text to phonemes with bilingual support."""
    from phonemizer.backend import EspeakBackend
    
    phonemizer_en = EspeakBackend(
        language='en-us', preserve_punctuation=True,
        with_stress=True, words_mismatch="ignore"
    )
    phonemizer_ar = EspeakBackend(
        language='ar', preserve_punctuation=True,
        with_stress=True, words_mismatch="ignore"
    )
    
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+'
    segments = []
    last_end = 0
    
    for match in re.finditer(arabic_pattern, text):
        if match.start() > last_end:
            non_arabic = text[last_end:match.start()].strip()
            if non_arabic:
                segments.append(('en', non_arabic))
        segments.append(('ar', match.group()))
        last_end = match.end()
    
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            segments.append(('en', remaining))
    
    if not segments:
        segments = [(lang, text)]
    
    phonemes_parts = []
    for seg_lang, seg_text in segments:
        phonemizer = phonemizer_ar if seg_lang == 'ar' else phonemizer_en
        phones = phonemizer.phonemize([seg_text])
        if phones and phones[0]:
            phonemes_parts.append(' '.join(phones[0].split()))
    
    return ' '.join(phonemes_parts)


def pre_encode_speakers(
    voices_dir: Path,
    output_dir: Path,
    speakers: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Pre-encode all speakers in voices_dir to .pt cache files.
    
    Args:
        voices_dir: Directory containing speaker .wav and .txt files
        output_dir: Directory to save .pt cache files
        speakers: Optional list of specific speakers to encode
        
    Returns:
        Dictionary mapping speaker names to cache file paths
    """
    from neucodec import DistillNeuCodec
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading DistillNeuCodec...")
    codec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec")
    codec.eval()
    
    wav_files = list(voices_dir.glob("*.wav"))
    if speakers:
        wav_files = [f for f in wav_files if f.stem in speakers]
    
    if not wav_files:
        logger.warning(f"No speaker files found in {voices_dir}")
        return {}
    
    logger.info(f"Found {len(wav_files)} speakers to encode")
    
    cache_files = {}
    manifest = {"speakers": [], "version": "1.0", "codec": "distill-neucodec"}
    
    for wav_path in wav_files:
        name = wav_path.stem
        txt_path = voices_dir / f"{name}.txt"
        
        logger.info(f"Encoding: {name}")
        
        # Get metadata
        meta = SPEAKER_METADATA.get(name, {"gender": "unknown", "lang": "ar", "description": name})
        
        # Read transcript
        transcript = ""
        if txt_path.exists():
            transcript = txt_path.read_text(encoding='utf-8').strip()
            # Clean expression markers
            transcript = re.sub(r'\[.*?\]', '', transcript).strip()
        
        # Encode audio
        try:
            codes = encode_audio(wav_path, codec)
        except Exception as e:
            logger.error(f"Failed to encode {name}: {e}")
            continue
        
        # Phonemize transcript
        phonemes = ""
        if transcript:
            try:
                phonemes = phonemize_text(transcript, meta.get("lang", "ar"))
            except Exception as e:
                logger.warning(f"Failed to phonemize {name}: {e}")
        
        # Create cache data
        cache_data = {
            "name": name,
            "codes": codes,
            "phonemes": phonemes,
            "transcript": transcript,
            "metadata": meta,
            "num_codes": len(codes),
            "duration_sec": len(codes) / 50.0,  # 50 codes/sec
        }
        
        # Save as .pt (fastest loading)
        cache_path = output_dir / f"{name}.pt"
        torch.save(cache_data, cache_path)
        cache_files[name] = cache_path
        
        # Also save JSON for inspection
        json_path = output_dir / f"{name}.json"
        json_data = {k: v for k, v in cache_data.items() if k != "codes"}
        json_data["codes_preview"] = codes[:10] if codes else []
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
        
        manifest["speakers"].append({
            "name": name,
            "cache_file": f"{name}.pt",
            "duration_sec": cache_data["duration_sec"],
            "num_codes": len(codes),
            **meta
        })
        
        logger.info(f"  → {len(codes)} codes ({cache_data['duration_sec']:.1f}s)")
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info(f"Manifest saved to {manifest_path}")
    
    return cache_files


def main():
    parser = argparse.ArgumentParser(description="Pre-encode speaker audio to cache files")
    parser.add_argument(
        "--voices-dir", "-v",
        type=Path,
        default=Path(__file__).parent.parent / "voices",
        help="Directory containing speaker .wav files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for cache files (default: voices/cache/)"
    )
    parser.add_argument(
        "--speakers", "-s",
        nargs="+",
        help="Specific speakers to encode (default: all)"
    )
    args = parser.parse_args()
    
    voices_dir = args.voices_dir.resolve()
    output_dir = args.output_dir or (voices_dir / "cache")
    
    if not voices_dir.exists():
        logger.error(f"Voices directory not found: {voices_dir}")
        sys.exit(1)
    
    logger.info(f"Voices directory: {voices_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    cache_files = pre_encode_speakers(voices_dir, output_dir, args.speakers)
    
    logger.info(f"\nPre-encoding complete! {len(cache_files)} speakers cached.")
    logger.info("Use these cache files for instant speaker loading during inference.")


if __name__ == "__main__":
    main()
