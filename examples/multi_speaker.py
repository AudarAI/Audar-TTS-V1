#!/usr/bin/env python3
"""
Multi-Speaker Voice Cloning
============================
Demonstrates cloning and caching multiple speaker voices for
dialogue generation and multi-character applications.
"""

import sys
sys.path.insert(0, '..')

from audar_tts import AudarTTS


def main():
    # Initialize the TTS engine
    tts = AudarTTS()
    
    # Clone multiple speakers (cached automatically after first use)
    speakers = {}
    
    # Speaker 1: Professional narrator
    speakers["narrator"] = tts.clone_voice(
        name="narrator",
        audio_path="samples/narrator.wav",
        transcript="Welcome to this demonstration of advanced voice synthesis."
    )
    
    # Speaker 2: Character voice
    speakers["character"] = tts.clone_voice(
        name="character",
        audio_path="samples/character.wav",
        transcript="Hello there, nice to meet you!"
    )
    
    print("Speakers cloned and cached.")
    print(f"Narrator duration: {speakers['narrator'].duration:.2f}s")
    print(f"Character duration: {speakers['character'].duration:.2f}s")
    
    # Generate a dialogue
    dialogue = [
        ("narrator", "The story begins on a quiet morning."),
        ("character", "What a beautiful day it is today!"),
        ("narrator", "Said the character, looking out the window."),
        ("character", "I wonder what adventures await me."),
    ]
    
    print("\nGenerating dialogue...")
    print("-" * 50)
    
    for i, (speaker_name, line) in enumerate(dialogue):
        speaker = speakers[speaker_name]
        output_file = f"dialogue_{i+1}_{speaker_name}.wav"
        
        audio, metrics = tts.speak(
            text=line,
            speaker=speaker,
            output=output_file,
            lang="en"
        )
        
        print(f"[{speaker_name.upper()}] {line}")
        print(f"  -> {output_file} ({metrics['duration']:.2f}s, RTF: {metrics['rtf']:.2f})")
    
    print("-" * 50)
    print("Dialogue generation complete!")
    
    # Note: Speaker profiles are automatically cached to disk
    # Subsequent runs will load from cache instead of re-encoding


if __name__ == "__main__":
    main()
