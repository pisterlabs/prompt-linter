#!/usr/bin/env python3
import sys
import openai
audio_file= open(sys.argv[1], "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)