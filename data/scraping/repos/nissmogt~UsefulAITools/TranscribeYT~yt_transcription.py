#
# This python function uses the transformers library to transcribe an audio file from YouTube to text.
#
# Author: @nissmogt
# Date: 2022-12-09
# Version: 1.0
# Uses python 3.8.10
#
# Usage: python yt_transcription.py <youtube_url>

import dependency_check
import os
import whisper
from yt_dlp import YoutubeDL
import subprocess
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Check if the audio file exists
def check_audio_file(audiofile):
    import contextlib
    import wave
    try:
        with contextlib.closing(wave.open(audiofile)) as f:
            return True
    except:
        return False


# Download the audio from a given URL, set outtmpl based on video title, and return video title
def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio/audio',
        'external_downloader': 'aria2c',
        'external_downloader_args': ['-x16', '-k1M'],
        'executable': '/usr/bin/aria2c',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'delete_after': False,
        'extract_audio': True,
        'audio_format': 'mp3',
        'audio_quality': 0,
        'no_check_certificate': True,
    }

    video_title = None
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', None)
        
    return video_title


def transcribe_audio(url):

    # Set the audio and transcription paths
    audio_path = os.path.join(os.getcwd(), 'audio')
    transcription_path = os.path.join(os.getcwd(), 'transcriptions')

    # Create the audio and transcription directories if they don't exist
    # Create the audio and transcription directories if they don't exist
    os.makedirs(audio_path, exist_ok=True)
    os.makedirs(transcription_path, exist_ok=True)

    title = download_audio(url)
    title = "audio"

    # Convert the MP3 to WAV
    input_audiofile = os.path.join(audio_path, title + ".mp3")
    output_audiofile = os.path.join(audio_path, title + ".wav")
    subprocess.call(['ffmpeg', '-i', input_audiofile, '-acodec', 'pcm_s16le', '-ac', '1',
                     '-ar', '16000', output_audiofile])


    # Use OpenAI Whisper to transcribe the wav audio
    model = whisper.load_model("small")
    transcription = model.transcribe(output_audiofile)
    with open(os.path.join(transcription_path, os.path.splitext(title)[0] + ".txt"), 'w') as f:
        f.write(transcription['text'] + ' ')

    # delete audio mp3 and wav files
    os.remove(output_audiofile)
    os.remove(input_audiofile)


if __name__ == '__main__':
    import sys

    dependency_check.check_dependencies()

    # add arguments to the script to run it from the command line
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL (leave blank to test): ")
    # add a test url if no url is provided
    if not url:
        # url = 'https://www.youtube.com/watch?v=wMBHQktcSQ0'  # chris voigt bio talk
        # url = 'https://www.youtube.com/watch?v=2bZi3Xm9tJE'
        # url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        # url = 'https://www.youtube.com/watch?v=XxT-rWMalqM'
        url = 'https://www.youtube.com/watch\?v\=QU6-2MEfVOc'   # neurocomp sci talk
        

    url = url.replace('\\', '')
    print("Downloading audio from {}...".format(url))
    transcribe_audio(str(url))
    print("Done! Transcription saved to {} directory.".format(os.path.join(os.getcwd(), 'transcriptions')))
