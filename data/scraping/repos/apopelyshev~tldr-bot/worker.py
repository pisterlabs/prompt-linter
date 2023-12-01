import os
import openai
import logging

from pytube.exceptions import VideoUnavailable
from urllib.parse import urlparse, parse_qs
from moviepy.editor import *
from pytube import YouTube

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging
logger.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

prompt_template = """First, try to assess if the following text has a very high possibility of being 
song lyrics (presence of verses / chorus / etc)
- if it does, write "tldr-abort" and stop responding
- if it does not, write concisely the main takeaway from the text, in a casual style 
(like if you were explaining it to a friend), in under {length} characters


{{text}}


YOUR ANSWER:""".format(
    length=os.environ["PROMPT_SUMMARY_LEN"]
)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])


def is_valid_youtube_url(url: str) -> bool:
    # Check if the URL is a valid YouTube video URL
    try:
        # Create a YouTube object
        yt = YouTube(url)

        # Check if the video is available
        if not yt.video_id:
            return False

    except (VideoUnavailable, Exception):
        return False

    # Return True if the video is available
    return yt.streams.filter(adaptive=True).first() is not None


# Calculate YouTube video duration
def get_video_duration(url: str) -> float:
    yt = YouTube(url)
    video_length = round(yt.length / 60, 2)

    return video_length


# Calculate API call cost
def calculate_api_cost(video_length: float) -> float:
    return round(video_length * 0.009, 2)


# Get Video Title
def video_title(url: str):
    return YouTube(url).title


# Mark as song to avoid making a call in the future
def mark_as_song(url: str):
    with open(".known_songs", "a") as songs:
        songs.write(f"{YouTube(url).video_id}#")


# Check if video marked as song already
def is_song(url: str):
    try:
        with open(".known_songs", "r") as songs:
            return YouTube(url).video_id in songs.read()
    except OSError as err:
        return False


# Download YouTube video as Audio
def download_audio(url: str):
    yt = YouTube(url)

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # Get the first available audio stream and download it
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path="tmp/")

    # Convert the downloaded audio file to mp3 format
    audio_path = os.path.join("tmp/", audio_stream.default_filename)
    audio_clip = AudioFileClip(audio_path)
    audio_clip.write_audiofile(os.path.join("tmp/", f"{video_id}.mp3"))

    # Delete the original audio stream
    os.remove(audio_path)


# Transcription
def transcribe_audio(file_path, video_id):
    # The path of the transcript
    transcript_filepath = f"tmp/{video_id}.txt"

    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)

    # Convert bytes to megabytes
    file_size_in_mb = file_size / (1024 * 1024)

    # Check if the file size is less than 25 MB
    if file_size_in_mb < 25:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # Writing the content of transcript into a txt file
            with open(transcript_filepath, "w") as transcript_file:
                transcript_file.write(transcript["text"])

        # Deleting the mp3 file
        os.remove(file_path)

    else:
        print("Please provide a smaller audio file (less than 25mb).")


# Generating Video Summary
def generate_summary(api_key: str, url: str) -> str:
    summary = ""
    openai.api_key = api_key

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    text_splitter = CharacterTextSplitter()

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # The path of the audio file
    audio_path = f"tmp/{video_id}.mp3"

    # The path of the transcript
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exist
    download_audio(url)
    # Transcribe the mp3 audio to text
    transcribe_audio(audio_path, video_id)

    # Generating summary of the text file
    with open(transcript_filepath) as f:
        transcript_file = f.read()
        logger.info(transcript_file[0:300])

        texts = text_splitter.split_text(transcript_file)
        docs = [Document(page_content=t) for t in texts[:3]]
        chain = load_summarize_chain(
            llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT
        )
        summary = chain.run(docs)

        # Delete the temporary transcript and audio file after the summary is generated
        os.remove(audio_path)
        os.remove(transcript_filepath)

    return summary.strip()
