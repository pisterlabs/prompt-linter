from moviepy.editor import VideoFileClip
import io
import openai
from io import BytesIO
import json
import base64
import requests

#OpenAI API Key
openai.api_key = "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" 

def get_text(blob_received):
    """
    response = requests.get(url)
    vid = response.content
    # Use BytesIO to convert bytes to a file-like object so it can be used by MoviePy
    video_io = BytesIO(vid)
    print("It worked!")
    video = VideoFileClip(video_io)

    # Extract audio
    audio = video.audio

    # Save audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    audio.write_audiofile(temp_audio_file)

    # Read the audio file into a bytes-like object
    audio_file= open("temp_audio.mp3", "rb")
    """
    print("converting to byte_io")
    try:
        byte_io = BytesIO(blob_received)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
    # Transcribe using Whisper ASR API
    try:
        response_2 = openai.Audio.translate("whisper-1", byte_io)
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)

    # The transcribed text
    print(response_2)
    return response_2.get("text")


