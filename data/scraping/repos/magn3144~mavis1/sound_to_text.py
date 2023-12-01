import openai
import os


def convert_audio_to_text(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript.text