# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os 
openai.api_key = os.getenv("openaikey")
# audio_file= open("audio.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)

audio_file= open("hindi.mp3", "rb")
transcript = openai.Audio.translate("whisper-1", audio_file,language="bn")

print(transcript)