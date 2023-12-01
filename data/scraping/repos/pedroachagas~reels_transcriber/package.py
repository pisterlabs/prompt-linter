import openai
import instaloader
from pytube import YouTube
import streamlit as st
import os
import subprocess
import streamlit.components.v1 as components
from instagrapi import Client
import time

def set_api_key():
    openai.api_key = st.secrets.openai_apikey

def tokens_to_brl(tokens):
    # Constants
    USD_TO_BRL = 5.0  # The conversion rate from USD to BRL
    API_COST_PER_TOKEN = 0.002 / 1000  # The OpenAI API cost per token
    return tokens * API_COST_PER_TOKEN * USD_TO_BRL

from yt_dlp import YoutubeDL
from pathlib import Path

def get_video_url(link, max_retries=10):
    retries = 0
    while retries <= max_retries:
        try:
            if "instagram.com" in link:
                cl = Client()
                post = cl.media_pk_from_url(link)
                return cl.media_info(post).video_url, "instagram"
            elif "youtube.com" in link or "youtu.be" in link:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': '%(id)s.%(ext)s',
                    'postprocessors': [{  # Extract audio using ffmpeg
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }]
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(link, download=True)
                    video_path_local = Path(f"{info_dict['id']}.wav")
                    return str(video_path_local), "youtube"
            else:
                return None, None
        except Exception as e:
            retries += 1
            if retries > max_retries:
                st.write("Too many retries. Please check your link or try again later.")
                raise e
            st.write("Error while trying to get the video URL. Retrying...")
            time.sleep(1)

    
def copy_button(text):
    return components.html(
        open("copy_button/index.html").read().replace("{{ text }}", text.replace("\n", "\\n")),
        width=None,
        height=55,  # Set the height to 50 pixels
    )

def apply_prompt(prompt, text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You are a helpful text formatter assistant that answers only using Markdown formatted text."},
            {"role": "user", "content": f"Apply the following prompt to the text:\n\n{prompt}\n\nText:\n{text}"},
        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content, cost_in_brl

def process_transcription(transcription, temp=0.1) :
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = temp,

    messages=[
            {"role": "system", "content": "As a text formatter assistant, your task is to assist users with their queries by responding only in Markdown formatted text. Your primary objective is to categorize the text you receive into chapters, using as many chapters as necessary. However, you must not create a list of topics or chapter names."},
            {"role": "user", "content": f"Text: {transcription}"},
        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content, cost_in_brl

def create_title(transcription):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature = 0.1,

    messages=[
            {"role": "system", "content": "You are a helpful text writing assistant that answers only using Markdown formatted text. You do not say or return anything else other than what was asked by the user"},
            {"role": "user", "content": f"Write a title for the text below. Return only the title, without quotes:\n\n{transcription}"},

        ]
    )
    tokens_used = completion["usage"]["total_tokens"]
    cost_in_brl = tokens_to_brl(tokens_used)
    return completion.choices[0].message.content.replace('"',''), cost_in_brl

def save_uploaded_file(uploaded_file):
    video_path = os.path.join(os.getcwd(), "user_uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path


def convert_video_to_audio(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    with st.spinner('Converting video...'):
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    st.success("Video file uploaded and converted to audio.")
    return audio_path


def display_sidebar():
    with st.sidebar:
        st.subheader("Whisper Model Selection")
        model_options = [
            'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
            'medium.en', 'medium', 'large-v1', 'large-v2', 'large'
        ]
        selected_model = st.selectbox('Select a model size', model_options)
        st.caption('The smaller the model, the faster the transcription.\nThe models with ".en" are smaller, but only work with the english language.')

        st.subheader("Language Selection")
        language_options = [
            'Auto detection', 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese'
        ]
        language_mapping = {
            'Auto detection': None,
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt'
        }
        selected_language = st.selectbox('Select the language of the video', language_options)
        selected_language_code = language_mapping[selected_language]

        if selected_model.endswith('.en') and selected_language_code != 'en' and selected_language_code is not None:
            st.warning("The selected model only works with English. Please choose a different model or select English as the language.")

        st.subheader("Transcription Processing")
        st.caption('Keep this parameter turned off to see the transcription before processing it \n(Recommended)')
        process_transcription_toggle = st.checkbox(
            label="Transcribe and process",
            key="Key2",
            value=False,
        )
    return selected_model, selected_language_code, process_transcription_toggle