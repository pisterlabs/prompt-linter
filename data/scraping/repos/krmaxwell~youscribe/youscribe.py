import os
import sys

from urllib.parse import urlparse
from pytube import YouTube
from openai import OpenAI, audio

openai_key = ""


def get_yt_code(url: str):
    if url == "":
        raise ValueError("URL cannot be blank")

    parsed_url = urlparse(url)

    if parsed_url.netloc == "www.youtube.com":
        video_id = parsed_url.query.split("=")[1]
    elif parsed_url.netloc == "youtu.be":
        video_id = parsed_url.path[1:]
    else:
        raise ValueError("URL must be a YouTube URL")
    return video_id


def save_url_to_file(url: str):
    video_id = get_yt_code(url)
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True)[0]
    audio_stream.download(filename=f"{video_id}.mp4")
    return f"{video_id}.mp4"


def get_openai_key(file_location: str = ""):
    openai_key = ""

    if file_location:
        with open(file_location, "r") as f:
            openai_key = f.read().strip()
    elif "OPENAI_KEY" in os.environ:
        openai_key = os.environ["OPENAI_KEY"]

    if openai_key == "":
        raise ValueError("OpenAI key cannot be blank")
    return openai_key


def get_transcript_for_audio_file(file_name: str):
    if file_name == "":
        raise ValueError("File name cannot be blank")

    if openai_key == "":
        raise ValueError("OpenAI key cannot be blank")

    client = OpenAI(api_key=openai_key)
    audio_file = open(file_name, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    audio_file.close()

    # get file name without extension
    plain_file_name = os.path.splitext(file_name)[0]
    with open(f"{plain_file_name}.txt", "w") as f:
        f.write(transcript.text)

    return transcript.text


if __name__ == "__main__":
    openai_key = get_openai_key()
    # get URL from command line parameter
    url = sys.argv[1]
    file_name = save_url_to_file(url)
    transcript = get_transcript_for_audio_file(file_name)
    print(transcript)
