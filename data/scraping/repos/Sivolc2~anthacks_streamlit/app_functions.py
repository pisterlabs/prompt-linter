#app_functions.py
import streamlit as st

from llm_functions import *
from st_custom_components import st_audiorec, convert_wav_to_mp3
import os

def chat_page():
    st.header("Chat with Claude")
    messages = []
    
    while True:
        message = st.text_input("Type your message:")
        
        if message:
            messages.append(("You", message))
            response = send_message_to_claude(message)
            messages.append(("Claude", response))
            
            for sender, message in messages:
                st.write(f"{sender}: {message}")


def audio_recording_page():
    st.header("Audio chat with Claude")
    wav_file_path = st_audiorec()

    if wav_file_path is not None:
        # Convert .wav file to .mp3
        mp3_file_path = wav_file_path.replace('.wav', '.mp3')
        convert_wav_to_mp3(wav_file_path, mp3_file_path)

        # Define whisper model
        whisper_model = 'base'

        # Initialize the MediaManager class
        media_manager = MediaManager()

        # Transcribe the audio
        transcript_text = media_manager._transcribe(mp3_file_path, whisper_model)
        # transcript_text = transcript['text']  # Extract the transcription text from the dictionary
        response = send_message_to_claude(transcript_text)  # Implement this function

        st.write(f"You: {transcript_text}")
        st.write(f"Claude: {response}")


def send_message_to_claude(message):
    import anthropic

    # Initialize the Anthropic client
    anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Prepare the prompt for the model
    prompt = anthropic.HUMAN_PROMPT + message + anthropic.AI_PROMPT

    # Send the message to Claude and get the response
    completion = anthropic_client.completion(
        prompt=prompt, model="claude-v1.3-100k", max_tokens_to_sample=1000
    )["completion"]

    return completion
