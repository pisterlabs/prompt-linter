import os
import whisper 
from openai import OpenAI

# load model 
model = whisper.load_model("medium")

files = os.listdir("./sample_audio")
for f in files:
    result = model.transcribe(f"./sample_audio/{f}")
    print(result)
    with open(f"./sample_audio_transcripts/{f}-transcript.txt", "w") as file:
        file.write(result["text"])