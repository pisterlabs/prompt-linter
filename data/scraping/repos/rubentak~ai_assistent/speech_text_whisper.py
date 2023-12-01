# import openai
import openai
import json
import pandas as pd
import credentials
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "audio_output.wav")

#%%
def get_transcript_whisper():
    openai.api_key = credentials.api_key
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    text = transcription["text"]
    return text

# Main code
output = get_transcript_whisper()
print(output)
print(type(output))

#%%
