import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
audio_file = open("audio.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript['text'])


#Zadanie: Nagraj plik autio.mp3 i przetestuj działanie tego skryptu