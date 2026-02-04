#!/usr/bin/env python3
"""
AudarTTS - Lightning-Fast Zero-Shot Voice Cloning

Clone any voice with just 3 seconds of audio. Generate natural speech in real-time.

Features:
    - Zero-shot voice cloning (no training required)
    - Real-time streaming synthesis (0.5s first-chunk latency)
    - Speaker caching for instant subsequent generations
    - Smart text chunking for unlimited length inputs
    - Bilingual support (English + Arabic)
    - 0.46x RTF (2x faster than real-time playback)

Quick Start:
    from audar_tts import AudarTTS
    
    tts = AudarTTS()
    tts.speak("Hello world!", speaker="Eve", output="hello.wav")
    
    # Stream for low latency
    for audio_chunk in tts.stream("Long text here", speaker="Eve"):
        play(audio_chunk)
    
    # Clone any voice
    tts.clone_voice("my_voice", "recording.wav", "transcript of recording")
    tts.speak("Now speaking in your voice!", speaker="my_voice")

Copyright (c) 2025 Audar AI
License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "Audar AI"
__all__ = ["AudarTTS", "Speaker", "split_text_chunks"]

import os
import sys
import re
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Generator, Tuple, Union
from functools import lru_cache
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Configuration
SAMPLE_RATE = 24000
CODES_PER_SECOND = 50
MAX_CONTEXT = 8192
DEFAULT_CHUNK_SIZE = 50

# HuggingFace model identifiers
HF_MODEL_REPO = "audarai/audar_tts_flash_v1_gguf"
HF_TOKENIZER_REPO = "AudarAI/audar-tts-flash-v1"
HF_CODEC_REPO = "neuphonic/distill-neucodec"


@dataclass
class Speaker:
    """Cached speaker profile for zero-shot voice cloning."""
    name: str
    audio_path: str
    transcript: str = ""
    phonemes: str = ""
    codes: List[int] = field(default_factory=list)
    lang: str = "en"
    _cached: bool = False
    
    @property
    def is_cached(self) -> bool:
        return len(self.codes) > 0
    
    @property
    def duration(self) -> float:
        return len(self.codes) / CODES_PER_SECOND if self.codes else 0
    
    def __repr__(self):
        status = "cached" if self.is_cached else "not loaded"
        return f"Speaker('{self.name}', {self.duration:.1f}s, {status})"


class SpeakerManager:
    """Manages speaker profiles with automatic caching and pre-encoded support."""
    
    def __init__(self, voices_dir: Path, cache_dir: Path, use_preencoded: bool = True):
        self.voices_dir = Path(voices_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._speakers: Dict[str, Speaker] = {}
        self._codec = None
        self._phonemizer_en = None
        self._phonemizer_ar = None
        self._use_preencoded = use_preencoded
        
        # Auto-load pre-encoded speakers if available
        if use_preencoded:
            self._load_preencoded_speakers()
    
    def _load_preencoded_speakers(self):
        """Load pre-encoded speakers from .pt cache files for instant availability."""
        import torch
        
        preencoded_dir = self.voices_dir / "cache"
        if not preencoded_dir.exists():
            return
        
        for pt_file in preencoded_dir.glob("*.pt"):
            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                name = data.get('name', pt_file.stem)
                
                speaker = Speaker(
                    name=name,
                    audio_path=str(self.voices_dir / f"{name}.wav"),
                    transcript=data.get('transcript', ''),
                    phonemes=data.get('phonemes', ''),
                    codes=data.get('codes', []),
                    lang=data.get('metadata', {}).get('lang', 'ar'),
                    _cached=True
                )
                self._speakers[name] = speaker
            except Exception:
                continue  # Skip invalid cache files
    
    def _ensure_codec(self):
        if self._codec is None:
            import torch
            from neucodec import DistillNeuCodec
            self._codec = DistillNeuCodec.from_pretrained(HF_CODEC_REPO)
            self._codec.eval()
    
    def _ensure_phonemizer(self):
        if self._phonemizer_en is None:
            from phonemizer.backend import EspeakBackend
            self._phonemizer_en = EspeakBackend(
                language='en-us', preserve_punctuation=True,
                with_stress=True, words_mismatch="ignore"
            )
            self._phonemizer_ar = EspeakBackend(
                language='ar', preserve_punctuation=True,
                with_stress=True, words_mismatch="ignore"
            )
    
    def _get_cache_path(self, audio_path: str) -> Path:
        audio_hash = hashlib.md5(Path(audio_path).read_bytes()).hexdigest()[:12]
        return self.cache_dir / f"{Path(audio_path).stem}_{audio_hash}.json"
    
    def _encode_audio(self, audio_path: str) -> List[int]:
        import torch
        import librosa
        
        self._ensure_codec()
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            codes = self._codec.encode_code(audio_or_path=wav_tensor)
            return codes.squeeze(0).squeeze(0).tolist()
    
    def _phonemize(self, text: str, lang: str = "en") -> str:
        self._ensure_phonemizer()
        
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
            phonemizer = self._phonemizer_ar if seg_lang == 'ar' else self._phonemizer_en
            phones = phonemizer.phonemize([seg_text])
            if phones and phones[0]:
                phonemes_parts.append(' '.join(phones[0].split()))
        
        return ' '.join(phonemes_parts)
    
    def list_voices(self) -> List[str]:
        """List all available voices including pre-encoded ones."""
        voices = set()
        # From wav files
        for f in self.voices_dir.glob("*.wav"):
            voices.add(f.stem)
        # From pre-encoded cache
        for name in self._speakers.keys():
            voices.add(name)
        return sorted(voices)
    
    def get(self, name: str) -> Speaker:
        """Get speaker by name, using pre-encoded cache if available."""
        # Return cached speaker if already loaded
        if name in self._speakers and self._speakers[name].is_cached:
            return self._speakers[name]
        
        # Try loading from pre-encoded .pt file
        preencoded_path = self.voices_dir / "cache" / f"{name}.pt"
        if preencoded_path.exists():
            import torch
            try:
                data = torch.load(preencoded_path, map_location='cpu', weights_only=False)
                speaker = Speaker(
                    name=name,
                    audio_path=str(self.voices_dir / f"{name}.wav"),
                    transcript=data.get('transcript', ''),
                    phonemes=data.get('phonemes', ''),
                    codes=data.get('codes', []),
                    lang=data.get('metadata', {}).get('lang', 'ar'),
                    _cached=True
                )
                self._speakers[name] = speaker
                return speaker
            except Exception:
                pass  # Fall through to standard loading
        
        # Standard loading from audio file
        audio_path = self.voices_dir / f"{name}.wav"
        if not audio_path.exists():
            available = self.list_voices()[:10]
            raise ValueError(f"Speaker '{name}' not found. Available: {available}")
        
        phonemes_path = self.voices_dir / f"{name}_phonemized.txt"
        text_path = self.voices_dir / f"{name}.txt"
        
        phonemes = ""
        transcript = ""
        if phonemes_path.exists():
            phonemes = phonemes_path.read_text().strip()
        elif text_path.exists():
            transcript = text_path.read_text().strip()
            phonemes = self._phonemize(transcript)
        
        cache_path = self._get_cache_path(str(audio_path))
        if cache_path.exists():
            cache_data = json.loads(cache_path.read_text())
            codes = cache_data.get('codes', [])
        else:
            codes = self._encode_audio(str(audio_path))
            cache_data = {'name': name, 'codes': codes, 'phonemes': phonemes}
            cache_path.write_text(json.dumps(cache_data))
        
        speaker = Speaker(
            name=name,
            audio_path=str(audio_path),
            transcript=transcript,
            phonemes=phonemes,
            codes=codes,
        )
        self._speakers[name] = speaker
        return speaker
    
    def add(
        self,
        name: str,
        audio_path: str,
        transcript: str = "",
        lang: str = "en"
    ) -> Speaker:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        phonemes = self._phonemize(transcript, lang) if transcript else ""
        codes = self._encode_audio(audio_path)
        
        speaker = Speaker(
            name=name,
            audio_path=audio_path,
            transcript=transcript,
            phonemes=phonemes,
            codes=codes,
            lang=lang,
        )
        self._speakers[name] = speaker
        
        cache_path = self._get_cache_path(audio_path)
        cache_data = {'name': name, 'codes': codes, 'phonemes': phonemes}
        cache_path.write_text(json.dumps(cache_data))
        
        return speaker
    
    def clear_cache(self):
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self._speakers.clear()


def split_text_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Split long text into synthesizable chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(sentence) > max_chars:
            clause_split = re.compile(r'(?<=[,;:])\s+')
            clauses = clause_split.split(sentence)
            
            for clause in clauses:
                if len(current_chunk) + len(clause) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {clause}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(clause) > max_chars:
                        words = clause.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= max_chars:
                                current_chunk = f"{current_chunk} {word}".strip()
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = word
                    else:
                        current_chunk = clause
        else:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


class AudarTTS:
    """
    Lightning-fast zero-shot text-to-speech engine.
    
    Clone any voice instantly. Generate natural speech 2x faster than real-time.
    
    Args:
        model_path: Path to GGUF model file. If None, downloads from HuggingFace.
        voices_dir: Directory containing voice profiles (.wav files).
        cache_dir: Directory for caching encoded speaker profiles.
        n_ctx: Context length for the language model (default 8192).
        verbose: Enable verbose logging.
        lazy_load: If True, defer model loading until first synthesis.
        use_preencoded: If True, load pre-encoded speaker codes from .pt files (recommended).
    
    Example:
        tts = AudarTTS()
        
        # Simple synthesis
        tts.speak("Hello world!", speaker="Eve", output="hello.wav")
        
        # Streaming for low latency
        for chunk in tts.stream("Long text...", speaker="Eve"):
            play(chunk)
        
        # Clone custom voice
        tts.clone_voice("friend", "friend.wav", "Hello this is my voice")
        tts.speak("Now I sound like my friend!", speaker="friend")
    """
    
    def __init__(
        self,
        model_path: str = None,
        voices_dir: str = None,
        cache_dir: str = None,
        n_ctx: int = MAX_CONTEXT,
        verbose: bool = False,
        lazy_load: bool = False,
        use_preencoded: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.voices_dir = Path(voices_dir) if voices_dir else Path("voices")
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        self._llm = None
        self._tokenizer = None
        self._codec = None
        self._speakers = SpeakerManager(self.voices_dir, self.cache_dir, use_preencoded)
        self._stop_token_id = None
        self._initialized = False
        
        if not lazy_load:
            self._initialize()
    
    def _download_model(self) -> Path:
        """Download GGUF model from HuggingFace if needed."""
        from huggingface_hub import hf_hub_download
        
        model_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="audar-tts-flash-v1-q8_0.gguf",
            cache_dir=str(self.cache_dir / "models")
        )
        return Path(model_file)
    
    def _initialize(self):
        if self._initialized:
            return
        
        print("=" * 60)
        print("  AudarTTS - Lightning-Fast Voice Cloning")
        print("=" * 60)
        
        # Download or locate model
        if self.model_path is None or not self.model_path.exists():
            print(f"\n[1/3] Downloading model from HuggingFace...")
            self.model_path = self._download_model()
        
        # Load GGUF model
        print(f"[1/3] Loading model...")
        from llama_cpp import Llama
        
        t0 = time.time()
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            verbose=self.verbose,
        )
        print(f"      {self.model_path.name} ({time.time()-t0:.1f}s)")
        
        # Load HF tokenizer
        print(f"[2/3] Loading tokenizer...")
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_REPO)
        self._stop_token_id = self._tokenizer.convert_tokens_to_ids('<|TARGET_CODES_END|>')
        
        # Load codec
        print(f"[3/3] Loading codec...")
        import torch
        from neucodec import DistillNeuCodec
        self._codec = DistillNeuCodec.from_pretrained(HF_CODEC_REPO)
        self._codec.eval()
        
        print("\n" + "=" * 60)
        print(f"  Ready! {len(self._speakers.list_voices())} voices available")
        print("=" * 60 + "\n")
        
        self._initialized = True
    
    def _build_prompt(self, speaker: Speaker, target_phonemes: str) -> str:
        ref_codes_str = "".join([f"<|speech_{c}|>" for c in speaker.codes])
        
        return (
            f"user: Convert the text to speech:"
            f"<|REF_TEXT_START|>{speaker.phonemes}<|REF_TEXT_END|>"
            f"<|REF_SPEECH_START|>{ref_codes_str}<|REF_SPEECH_END|>"
            f"<|TARGET_TEXT_START|>{target_phonemes}<|TARGET_TEXT_END|>\n"
            f"assistant:<|TARGET_CODES_START|>"
        )
    
    def _decode_audio(self, codes: List[int]) -> np.ndarray:
        import torch
        
        if not codes:
            return np.array([])
        
        codes_tensor = torch.tensor(codes, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            wav = self._codec.decode_code(codes_tensor)
        return wav.squeeze().numpy()
    
    def _estimate_max_tokens(self, text: str) -> int:
        estimated = int(len(text) / 10 * 50) + 100
        return max(150, min(estimated, 1500))
    
    @property
    def voices(self) -> List[str]:
        """List available voice profiles."""
        return self._speakers.list_voices()
    
    def clone_voice(
        self,
        name: str,
        audio_path: str,
        transcript: str = "",
        lang: str = "en"
    ) -> Speaker:
        """
        Clone a voice from an audio sample.
        
        Args:
            name: Unique name for this voice.
            audio_path: Path to audio file (3-10 seconds recommended).
            transcript: What is said in the audio (improves quality).
            lang: Primary language ('en' or 'ar').
        
        Returns:
            Speaker object ready for synthesis.
        """
        self._initialize()
        return self._speakers.add(name, audio_path, transcript, lang)
    
    def speak(
        self,
        text: str,
        speaker: str = "Eve",
        output: str = None,
        lang: str = "en",
        max_tokens: int = None,
        auto_chunk: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize (any length).
            speaker: Voice to use (default "Eve").
            output: Output file path (optional).
            lang: Language for phonemization ('en' or 'ar').
            max_tokens: Max tokens (auto-calculated if not set).
            auto_chunk: Auto-split long text (default True).
        
        Returns:
            Tuple of (audio_array, metrics_dict).
        """
        self._initialize()
        
        spk = self._speakers.get(speaker)
        chunks = split_text_chunks(text) if auto_chunk else [text]
        
        total_start = time.time()
        all_audio = []
        all_codes = []
        
        print(f"\n{'='*60}")
        print(f"  Speaker: {speaker} | Chunks: {len(chunks)}")
        print(f"{'='*60}")
        
        for i, chunk_text in enumerate(chunks, 1):
            chunk_start = time.time()
            
            target_phonemes = self._speakers._phonemize(chunk_text, lang)
            prompt = self._build_prompt(spk, target_phonemes)
            input_ids = self._tokenizer.encode(prompt)
            
            tokens_max = max_tokens or self._estimate_max_tokens(chunk_text)
            codes = []
            
            for token in self._llm.generate(
                tokens=input_ids,
                top_k=40, top_p=0.95, temp=0.3, repeat_penalty=1.1,
            ):
                if token == self._stop_token_id:
                    break
                if len(codes) >= tokens_max:
                    break
                
                token_text = self._tokenizer.decode([token])
                if token_text.startswith('<|speech_') and token_text.endswith('|>'):
                    try:
                        code = int(token_text[9:-2])
                        codes.append(code)
                    except ValueError:
                        pass
            
            audio = self._decode_audio(codes)
            all_audio.append(audio)
            all_codes.extend(codes)
            
            chunk_time = time.time() - chunk_start
            chunk_dur = len(audio) / SAMPLE_RATE
            tok_s = len(codes) / chunk_time if chunk_time > 0 else 0
            
            print(f"  [{i}/{len(chunks)}] {len(codes):3d} codes | "
                  f"{chunk_dur:.1f}s audio | {tok_s:.0f} tok/s")
        
        full_audio = np.concatenate(all_audio) if all_audio else np.array([])
        
        total_time = time.time() - total_start
        duration = len(full_audio) / SAMPLE_RATE
        rtf = total_time / duration if duration > 0 else 0
        
        print(f"{'='*60}")
        print(f"  Duration: {duration:.2f}s | Time: {total_time:.2f}s | RTF: {rtf:.2f}x")
        print(f"{'='*60}\n")
        
        if output:
            import soundfile as sf
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, full_audio, SAMPLE_RATE)
            print(f"  Saved: {output}\n")
        
        return full_audio, {
            'text': text,
            'speaker': speaker,
            'duration': duration,
            'total_time': total_time,
            'rtf': rtf,
            'num_codes': len(all_codes),
            'num_chunks': len(chunks),
            'tok_per_sec': len(all_codes) / total_time if total_time > 0 else 0,
        }
    
    def stream(
        self,
        text: str,
        speaker: str = "Eve",
        lang: str = "en",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_tokens: int = None,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Stream audio generation for low-latency playback.
        
        Yields audio chunks as they're generated (~1s chunks by default).
        First chunk available in ~0.5s.
        
        Args:
            text: Text to synthesize.
            speaker: Voice to use.
            lang: Language ('en' or 'ar').
            chunk_size: Codes per chunk (50 = ~1s audio).
            max_tokens: Maximum tokens to generate.
        
        Yields:
            Tuple of (audio_chunk, metrics).
        """
        self._initialize()
        
        spk = self._speakers.get(speaker)
        target_phonemes = self._speakers._phonemize(text, lang)
        prompt = self._build_prompt(spk, target_phonemes)
        input_ids = self._tokenizer.encode(prompt)
        
        tokens_max = max_tokens or self._estimate_max_tokens(text)
        
        codes_buffer = []
        total_codes = 0
        chunk_num = 0
        start_time = time.time()
        first_chunk_time = None
        
        for token in self._llm.generate(
            tokens=input_ids,
            top_k=40, top_p=0.95, temp=0.3, repeat_penalty=1.1,
        ):
            if token == self._stop_token_id:
                break
            if total_codes >= tokens_max:
                break
            
            token_text = self._tokenizer.decode([token])
            
            if token_text.startswith('<|speech_') and token_text.endswith('|>'):
                try:
                    code = int(token_text[9:-2])
                    codes_buffer.append(code)
                    total_codes += 1
                    
                    if len(codes_buffer) >= chunk_size:
                        chunk_num += 1
                        audio = self._decode_audio(codes_buffer)
                        elapsed = time.time() - start_time
                        
                        if first_chunk_time is None:
                            first_chunk_time = elapsed
                        
                        yield audio, {
                            'chunk': chunk_num,
                            'codes': len(codes_buffer),
                            'total_codes': total_codes,
                            'duration': len(codes_buffer) / CODES_PER_SECOND,
                            'elapsed': elapsed,
                            'tok_per_sec': total_codes / elapsed,
                            'first_chunk_latency': first_chunk_time,
                        }
                        codes_buffer.clear()
                        
                except ValueError:
                    pass
        
        if codes_buffer:
            chunk_num += 1
            audio = self._decode_audio(codes_buffer)
            elapsed = time.time() - start_time
            
            yield audio, {
                'chunk': chunk_num,
                'codes': len(codes_buffer),
                'total_codes': total_codes,
                'duration': len(codes_buffer) / CODES_PER_SECOND,
                'elapsed': elapsed,
                'tok_per_sec': total_codes / elapsed if elapsed > 0 else 0,
                'first_chunk_latency': first_chunk_time or elapsed,
                'final': True,
            }
    
    def stream_to_file(
        self,
        text: str,
        output: str,
        speaker: str = "Eve",
        lang: str = "en",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> dict:
        """
        Stream synthesis to a file with progress display.
        
        Args:
            text: Text to synthesize.
            output: Output file path.
            speaker: Voice to use.
            lang: Language ('en' or 'ar').
            chunk_size: Codes per chunk.
        
        Returns:
            Final metrics dictionary.
        """
        import soundfile as sf
        
        print(f"\n{'='*60}")
        print(f"  Streaming: {text[:40]}{'...' if len(text) > 40 else ''}")
        print(f"  Speaker: {speaker}")
        print(f"{'='*60}")
        
        all_audio = []
        final_metrics = {}
        
        for audio, metrics in self.stream(text, speaker, lang, chunk_size):
            all_audio.append(audio)
            final_metrics = metrics
            
            bar_len = min(20, metrics['chunk'])
            bar = "#" * bar_len
            print(f"  [{metrics['chunk']:2d}] {bar:<20} {metrics['duration']:.1f}s | "
                  f"{metrics['tok_per_sec']:.0f} tok/s")
        
        full_audio = np.concatenate(all_audio)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, full_audio, SAMPLE_RATE)
        
        duration = len(full_audio) / SAMPLE_RATE
        rtf = final_metrics['elapsed'] / duration if duration > 0 else 0
        
        print(f"{'='*60}")
        print(f"  First chunk: {final_metrics.get('first_chunk_latency', 0):.2f}s")
        print(f"  Total: {duration:.2f}s audio in {final_metrics['elapsed']:.2f}s")
        print(f"  RTF: {rtf:.2f}x | Saved: {output}")
        print(f"{'='*60}\n")
        
        return {
            'duration': duration,
            'total_time': final_metrics['elapsed'],
            'rtf': rtf,
            'first_chunk_latency': final_metrics.get('first_chunk_latency', 0),
            'num_codes': final_metrics['total_codes'],
            'num_chunks': final_metrics['chunk'],
        }


def main():
    """CLI interface for AudarTTS."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AudarTTS - Lightning-Fast Voice Cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic synthesis
  python audar_tts.py "Hello world!" -s Eve -o hello.wav
  
  # Streaming mode (lower latency)
  python audar_tts.py "Long text here" -s Ivy --stream
  
  # List available voices
  python audar_tts.py --list-voices
  
  # Clone a custom voice
  python audar_tts.py "Hello!" -s my_voice --clone my_recording.wav --transcript "Original text"
  
  # Arabic text
  python audar_tts.py "text" -s Abdullah --lang ar
        """
    )
    
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("-s", "--speaker", default="Eve", help="Voice to use")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--lang", default="en", choices=["en", "ar"], help="Language")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--chunk-size", type=int, default=50, help="Streaming chunk size")
    parser.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    
    parser.add_argument("--clone", help="Path to audio file for voice cloning")
    parser.add_argument("--transcript", default="", help="Transcript of clone audio")
    
    parser.add_argument("--voices-dir", default="voice_profiles", help="Voice profiles directory")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.list_voices:
        tts = AudarTTS(voices_dir=args.voices_dir, lazy_load=True)
        voices = tts.voices
        print(f"\nAvailable voices ({len(voices)}):")
        print("-" * 40)
        for i, v in enumerate(voices, 1):
            print(f"  {i:3d}. {v}")
        print()
        return
    
    if not args.text:
        parser.print_help()
        sys.exit(1)
    
    tts = AudarTTS(voices_dir=args.voices_dir, verbose=args.verbose)
    
    if args.clone:
        print(f"\nCloning voice from: {args.clone}")
        tts.clone_voice(args.speaker, args.clone, args.transcript, args.lang)
    
    output = args.output
    if not output:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_text = re.sub(r'[^\w\s-]', '', args.text)[:30].replace(' ', '_')
        mode = "stream" if args.stream else "batch"
        output = str(output_dir / f"{args.speaker}_{mode}_{safe_text}.wav")
    
    if args.stream:
        tts.stream_to_file(
            args.text,
            output,
            speaker=args.speaker,
            lang=args.lang,
            chunk_size=args.chunk_size,
        )
    else:
        tts.speak(
            args.text,
            speaker=args.speaker,
            output=output,
            lang=args.lang,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    main()
