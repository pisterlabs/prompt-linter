import openai
import api
import os

openai.api_key = api.key

# Get the  path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "input.wav"
file_path = os.path.join(script_dir, file_name)

# Open the audio file and extract aduio
audio_file = open(file_path, "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
text = transcript.text
audio_file.close()

print(text)
