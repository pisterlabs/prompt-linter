import nltk
from nltk.tokenize import word_tokenize
import os
from openai_transcriber import OpenAITranscriber
from openai_api import OpenaiApi
import canvas
import math
import logging
import pydub
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont



logging.basicConfig(level=logging.INFO)

# These are your repository paths. Update them according to your file structure.
AUDIO_FILES_PATH = "./temp/"
TRANSCRIPTS_FOLDER = "./transcripts/"
NOTES_FOLDER = "./notes/"

# Define the maximum tokens that can be passed to the GPT model
MAX_TOKENS = 4096

from pydub import AudioSegment

def convert_m4a_to_mp3(m4a_path, mp3_path):
    # Load m4a file
    audio = AudioSegment.from_file(m4a_path, format="m4a")

    # Export as mp3
    audio.export(mp3_path, format="mp3")

def convert_mp4_to_mp3(mp4_path, mp3_path):
    # Load mp4 file
    audio = AudioSegment.from_file(mp4_path, format="mp4")

    # Export as mp3
    audio.export(mp3_path, format="mp3")

def split_transcript(transcript):
    # Tokenize the transcript
    tokens = word_tokenize(transcript)
    
    # Split the tokens into chunks of at most 4096 tokens
    chunks = []
    current_chunk = []
    current_token_count = 0
    for token in tokens:
        if current_token_count + len(token) > 4096:
            # If adding the next token would exceed the limit, we start a new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [token]
            current_token_count = len(token)
        else:
            # Otherwise, we add the token to the current chunk
            current_chunk.append(token)
            current_token_count += len(token)
    
    # Don't forget to add the last chunk if it's non-empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


transcriber = OpenAITranscriber(AUDIO_FILES_PATH, TRANSCRIPTS_FOLDER)
api = OpenaiApi()

# Convert all m4a files to mp3
for filename in os.listdir(AUDIO_FILES_PATH):
    if filename.endswith(".m4a"):
        convert_m4a_to_mp3(f"{AUDIO_FILES_PATH}/{filename}", f"{AUDIO_FILES_PATH}/{filename[:-4]}.mp3")
        
# convert all mp4 files to mp3
for filename in os.listdir(AUDIO_FILES_PATH):
    if filename.endswith(".mp4"):
        convert_mp4_to_mp3(f"{AUDIO_FILES_PATH}/{filename}", f"{AUDIO_FILES_PATH}/{filename[:-4]}.mp3")

# Iterate over the files in the repository
for filename in os.listdir(AUDIO_FILES_PATH):
    if filename.endswith(".mp3"):  # Or whatever format your audio files are in
        # Transcribe the audio file
        transcript = transcriber.transcribe(filename)
        transcript_text = transcript
        
        # If transcript is too long, split it
        chunks = split_transcript(transcript_text)
        
        all_notes = ""
        # Pass each chunk to GPT and save the notes
        for i, chunk in enumerate(chunks):
            system_prompt = "You are the world's best lecture notes taker. You will be passed in a transcript from a part of a lecture and you have to take nicely formatted notes on it. Use lot's of emojis and beautiful formatting. Ensure every single line has at least one emoji. The transcript may have errors so use your best judgement."
            prompt = "Please take nicely formatted notes on the following lecture transcript:\n\n" + chunk
            notes = api.query(system_prompt, prompt, "gpt-3.5-turbo")  # Fill in user_prompt and model as needed
            
            logging.info(notes)
        
            # Concatenate the notes
            all_notes += notes + "\n"
            
        # Save all notes to a single .txt file
        with open(f"{NOTES_FOLDER}{filename[:-4]}.txt", 'w') as file:
            file.write(all_notes)

