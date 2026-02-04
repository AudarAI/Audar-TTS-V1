#!/usr/bin/env python3
"""
Streaming Text-to-Speech Synthesis
===================================
Real-time streaming example with chunk-by-chunk audio generation.
Ideal for low-latency applications and live playback.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import soundfile as sf
from audar_tts import AudarTTS


def main():
    # Initialize the TTS engine
    tts = AudarTTS()
    
    # Clone or load a speaker
    speaker = tts.clone_voice(
        name="streaming_demo",
        audio_path="reference.wav",
        transcript="Reference transcript here."
    )
    
    # Streaming synthesis - yields audio chunks as they're generated
    text = """
    Streaming synthesis enables real-time applications where audio 
    playback can begin before the entire utterance is generated. 
    This dramatically reduces perceived latency for end users.
    """
    
    all_chunks = []
    chunk_count = 0
    
    print("Streaming synthesis started...")
    print("-" * 40)
    
    for audio_chunk in tts.stream(
        text=text,
        speaker=speaker,
        lang="en",
        chunk_size=50  # ~1 second per chunk
    ):
        chunk_count += 1
        chunk_duration = len(audio_chunk) / 24000  # 24kHz sample rate
        all_chunks.append(audio_chunk)
        
        print(f"Chunk {chunk_count}: {chunk_duration:.2f}s ({len(audio_chunk)} samples)")
        
        # In a real application, you would play this chunk immediately
        # Example: audio_player.queue(audio_chunk)
    
    print("-" * 40)
    print(f"Total chunks: {chunk_count}")
    
    # Concatenate all chunks and save
    full_audio = np.concatenate(all_chunks)
    sf.write("streaming_output.wav", full_audio, 24000)
    print(f"Saved complete audio: streaming_output.wav ({len(full_audio)/24000:.2f}s)")


if __name__ == "__main__":
    main()
