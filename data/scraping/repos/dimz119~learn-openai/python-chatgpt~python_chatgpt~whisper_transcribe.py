import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file= open("sample/whisper_korean.mp4", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

# decode Unicode escape sequences in a string by using the unicode_escape codec
decoded_text = transcript['text'].encode('utf-8', 'unicode_escape').decode('utf-8')
print(decoded_text)
