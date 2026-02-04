#!/usr/bin/env python3
"""
Basic Text-to-Speech Synthesis
==============================
Simple example demonstrating core TTS functionality with Audar-TTS.
"""

import sys
sys.path.insert(0, '..')

from audar_tts import AudarTTS


def main():
    # Initialize the TTS engine (auto-downloads model if needed)
    tts = AudarTTS()
    
    # Clone a voice from a reference audio file
    speaker = tts.clone_voice(
        name="my_speaker",
        audio_path="reference.wav",  # Your reference audio (3-10 seconds)
        transcript="The exact words spoken in the reference audio."
    )
    
    # Synthesize speech
    audio, metrics = tts.speak(
        text="Hello! This is a demonstration of Audar TTS zero-shot voice cloning.",
        speaker=speaker,
        output="output.wav",
        lang="en"
    )
    
    # Print performance metrics
    print(f"Generated {metrics['duration']:.2f}s of audio")
    print(f"Real-time factor: {metrics['rtf']:.2f}x")
    print(f"Latency: {metrics['latency_ms']:.0f}ms")
    print(f"Saved to: output.wav")


if __name__ == "__main__":
    main()
