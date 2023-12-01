
import os
import re
import tempfile
import pydub
from pydub.silence import split_on_silence
import openai
import streamlit as st
import yt_dlp as youtube_dl

import tiktoken
from tiktoken import encoding_for_model

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)

def split_audio(audio_file, max_file_size):
    audio = pydub.AudioSegment.from_file(audio_file)
    duration = len(audio)
    max_duration = max_file_size * (duration / os.path.getsize(audio_file))
    chunk_duration_ms = int(max_duration)

    audio_chunks = []

    start_time = 0
    end_time = chunk_duration_ms
    while start_time < len(audio):
        chunk = audio[start_time:end_time]
        audio_chunks.append(chunk)
        start_time += chunk_duration_ms
        end_time += chunk_duration_ms

    return audio_chunks

def transcribe_audio_file(audio_file):
    MAX_FILE_SIZE = 25 * 1024 * 1024
    file_size = os.path.getsize(audio_file)

    if file_size > MAX_FILE_SIZE:
        audio_chunks = split_audio(audio_file, MAX_FILE_SIZE)
        transcript_list = []
        
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(audio_chunks):
            with tempfile.NamedTemporaryFile(suffix=".webm") as temp_audio:
                chunk.export(temp_audio.name, format="webm")
                with open(temp_audio.name, "rb") as temp_audio_file:
                    transcript_chunk = openai.Audio.transcribe("whisper-1", temp_audio_file)
                    transcript_list.append(transcript_chunk["text"])
            
            progress_bar.progress((i + 1) / len(audio_chunks))
        
        transcript = concatenate_transcripts(transcript_list)
    else:
        with open(audio_file, "rb") as audio:
            transcript = openai.Audio.transcribe("whisper-1", audio)
            transcript = transcript["text"]

    return transcript

def concatenate_transcripts(transcript_list):
    transcript = " ".join(transcript_list)
    return transcript.strip()

def embed_youtube_video(video_id: str, width: int = 640, height: int = 360):
    """Embed a YouTube video in a Streamlit app."""
    youtube_embed_url = f"https://www.youtube.com/embed/{video_id}"
    iframe_html = f'<iframe src="{youtube_embed_url}" width="{width}" height="{height}" frameborder="0" allowfullscreen></iframe>'
    st.markdown(iframe_html, unsafe_allow_html=True)

def extract_video_id(youtube_url: str) -> str:
    """Extract the video ID from the YouTube URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/(?:watch\?v=)?(.{11})"
    match = re.match(regex, youtube_url)
    if match:
        return match.group(1)
    return None

@st.cache_data(show_spinner=False)
def get_transcript(video_id):
    # Fetch audio from YouTube URL
    ydl_opts = {
        "format": "bestaudio/best[ext=m4a]",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": f"{video_id}.%(ext)s"
    }
    with st.spinner("Downloading audio from URL..."):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            duration = info["duration"]
            multiplier = duration // (60*5)
            title = info["title"]

    # Perform transcription or subtitle extraction
    with st.spinner('Transcribing audio...'):
        with open(f"{video_id}.webm", "rb") as audio:
            transcript = transcribe_audio_file(f"{video_id}.webm")

    return transcript, multiplier, title

def validate_api_key(api_key: str) -> bool:
    try:
        openai.api_key = api_key
        openai.Engine.list()
        return True
    except:
        return False
    
def display_bmc_button():
    st.write(
        """
        <style>
            .bmc-container {
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
                }
            .github-logo {
                width: 40px;
                height: 40px;
                margin-left: 1rem;
            }
        </style>
        <div class="bmc-container">
            <a href="https://www.buymeacoffee.com/lucharo" target="_blank" title="Donate to support the project">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"  alt="Buy Me A Coffee" width="200">
            </a>
            <a href="https://github.com/lucharo/youtube-ai-summaries" target="_blank" title="Check out the GitHub repository">
                <svg class="github-logo" viewBox="0 0 16 16" version="1.1" width="30" height="30" aria-hidden="true">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>            
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )