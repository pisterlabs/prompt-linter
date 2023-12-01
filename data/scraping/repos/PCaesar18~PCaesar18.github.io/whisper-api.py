import os
import config
import openai
openai.api_key = config.api_key
audio_file = open("/Users/vercingetorix/Downloads/REDSIGN_VOICE.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript.text)