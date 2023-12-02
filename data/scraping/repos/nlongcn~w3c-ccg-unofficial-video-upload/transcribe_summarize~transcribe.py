from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
import whisper
from shutil import move
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import warnings

### create folders if not exist
def create_folders_if_not_exist(transcripts_folder, summaries_folder):
    if not os.path.exists(transcripts_folder):
        os.makedirs(transcripts_folder)
    
    if not os.path.exists(summaries_folder):
        os.makedirs(summaries_folder)

### get video files
def get_video_files(src_directory):
    return [f for f in os.listdir(src_directory) if f.endswith(".mp4")]

def transcribe_videos(video_files, src_directory, model, transcripts_folder):
    for video_file in video_files:
        video_path = os.path.join(src_directory, video_file)
        print(f"Transcribing {video_file}...this can take up to 20 minutes, depending on the length of the video.")
        transcription = model.transcribe(video_path)['text']
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        
        with open(os.path.join(transcripts_folder, f'{video_name}_transcript.txt'), 'w') as f:
            f.write(transcription)

        # Move the video to the 'ccg_videos' folder
        destination_folder = os.path.join(src_directory, 'ccg_videos_transcribed')
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        destination_path = os.path.join(destination_folder, video_file)
        move(video_path, destination_path)
        print(f"Video {video_file} has been processed and moved to {destination_path}.")
    
    # Filter and remove transcripts with less than specified number of words
    # Filter and remove transcripts with less than specified number of words
    min_words = 100  # Adjust as needed
    for transcript_file in [f for f in os.listdir(transcripts_folder) if os.path.isfile(os.path.join(transcripts_folder, f))]: # Ensures only files are processed
        try:
            with open(os.path.join(transcripts_folder, transcript_file), 'r') as f:
                transcript = f.read()
                if len(transcript.split()) < min_words:
                    # Removing the transcript
                    os.remove(os.path.join(transcripts_folder, transcript_file))
                    print(f"Deleted {transcript_file} due to insufficient words.")

                    # Removing the related video
                    video_name = transcript_file.replace('_transcript.txt', '.mp4') # Assuming the naming convention is consistent
                    video_path = os.path.join(src_directory, 'ccg_videos_transcribed', video_name)
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"Deleted corresponding video {video_name}.")

        except PermissionError:
            print(f"Permission error encountered when trying to access {transcript_file}. Skipping.")

def main():
    # This will suppress all warnings
    warnings.filterwarnings("ignore")

    # Set directories (you'd need to define these)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = os.getcwd()  # Gets the current working directory
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    while os.path.basename(script_dir) != 'w3c-ccg-unofficial-video-upload' and script_dir != os.path.dirname(script_dir):
        script_dir = os.path.dirname(script_dir)

    # At this point, script_directory is either the 'w3c-ccg-unofficial-video-upload' directory or the root directory if not found
    if os.path.basename(script_dir) != 'w3c-ccg-unofficial-video-upload':
        print("'w3c-ccg-unofficial-video-upload' directory not found in the path hierarchy of the script.")
        exit(1)
    
    src_directory = os.path.join(script_dir, 'ccg_videos_new')
    print(f"Expected video directory: {src_directory}")
    transcripts_folder = os.path.join(script_dir, 'transcripts')
    print(f"Expected transcripts directory: {transcripts_folder}")
    summaries_folder = os.path.join(script_dir, 'summaries')
    print(f"Expected summaries directory: {summaries_folder}")

    # Ensure required folders exist
    create_folders_if_not_exist(transcripts_folder, summaries_folder)

    # Get video files
    video_files = get_video_files(src_directory)

    # Load or define your model here
    model = whisper.load_model('base.en') 

    # Transcribe all videos
    transcribe_videos(video_files, src_directory, model, transcripts_folder)

    print("All videos have been processed.")

if __name__ == '__main__':
    main()
