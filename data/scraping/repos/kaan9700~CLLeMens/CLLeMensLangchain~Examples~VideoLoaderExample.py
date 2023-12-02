import openai
from dotenv import load_dotenv
from moviepy.editor import *
import os

from pydub import AudioSegment

# Pfad zum Video
video_path = 'video.mp4'
# Split the filepath into path and extension
path, _ = os.path.splitext(video_path)
print(path)

# Load the environment variables from the .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Das Video laden
video = VideoFileClip(video_path)

# Audio extrahieren
audio = video.audio

# Pfad zum Speichern der Audiodatei erstellen
audio_path = video_path.split('.')[0] + '.mp3'

# Audiodatei speichern
audio.write_audiofile(audio_path)

# Ressourcen freigeben
audio.close()
video.close()

# Load audio file
audio = AudioSegment.from_mp3(audio_path)
print("AUDIO: ", audio)

# Convert chunk length to milliseconds (PyDub uses milliseconds)
chunk_length = 120 * 1000

# Calculate number of chunks needed
num_chunks = len(audio) // chunk_length + 1  # +1 to account for any remaining part

# Create cache directory if it doesn't exist
cache_dir = "cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Initialize an empty string to store the full transcription
full_transcription = ""
print("NUM CHUNKS: ", num_chunks)
# Loop over the audio file, chunk by chunk
for i in range(num_chunks):
    print("I: ", i)
    # Extract the chunk
    start_time = i * chunk_length
    end_time = (i + 1) * chunk_length
    chunk = audio[start_time:end_time]

    # Export the chunk to a temporary file
    temp_file_name = os.path.join(cache_dir, f"temp_chunk_{i}.mp3")
    print(temp_file_name)
    chunk.export(temp_file_name, format="mp3")

    # Transcribe the chunk using OpenAI Whisper
    with open(temp_file_name, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        full_transcription += transcript['text'] + " "

print("FULL TRANSCRIPTION: ", full_transcription)