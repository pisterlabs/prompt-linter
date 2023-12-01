# -*- coding: utf-8 -*-
"""
This script takes a local MP3 file as input and outputs a transcript with speaker identification.
"""

import sys
import subprocess
import datetime
import wave
import contextlib
import numpy as np
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
import whisper
from openai import OpenAI

client = OpenAI(api_key="enter your key here") 
   # Replace with your OpenAI API key

# Check if an MP3 file path is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py [path_to_mp3_file]")
    sys.exit(1)

path = sys.argv[1]

# Number of speakers and language setting
num_speakers = 2
language = 'English'
model_size = 'large'

# Adjust the model name based on language
model_name = model_size
if language == 'English' and model_size != 'large':
    model_name += '.en'


# Load the Whisper model
model = whisper.load_model(model_size)

# Convert MP3 to WAV if necessary
if path[-3:] != 'wav':
    subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
    path = 'audio.wav'

# Perform transcription
result = model.transcribe(path)
segments = result["segments"]

# Calculate duration of the audio
with contextlib.closing(wave.open(path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Load the audio
audio = Audio()

def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])  # Adjust for the last segment
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    
    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:  # Check if the audio is not mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Add a batch dimension
    waveform = waveform.unsqueeze(0)
    
    return embedding_model(waveform)


# Load the embedding model
embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device("cuda"))

# Compute embeddings for each segment
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

# Perform clustering for speaker identification
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_

for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

# Function to format time
def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Write transcript to a file
with open("transcript.txt", "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')

print("Transcript saved to transcript.txt")

def summarise(file_path):
    # Create an OpenAI client instance
    client = OpenAI()

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are tasked with the role of a 'Stand-Up Meeting Summarizer'. In this capacity, your primary function is to meticulously analyze and interpret the transcripts of stand-up meetings. After receiving a transcript from me, your objective will be to systematically extract and enumerate the key action items (follow-ups) that emerged during the meeting. Additionally, you are expected to identify and list any potential obstacles or issues (blockers) that were highlighted during the discussion. This process demands a keen attention to detail and an ability to discern the most critical elements from the conversation for effective project management and team coordination."},
        {"role": "user", "content": text}
    ]

    # Make the API call using the client
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Access the response
    reply = response.choices[0].message.content
    print(reply)

    # Save the reply to a new text file
    output_file_path = file_path.replace('.txt', '_summary.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(reply)

    return reply

# Call the function with your transcript file
summarise("transcript.txt")
