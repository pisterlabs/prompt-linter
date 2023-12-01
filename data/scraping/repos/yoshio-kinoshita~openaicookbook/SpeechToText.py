# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
audio_file= open("data/001-sibutomo.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(transcript["text"])
