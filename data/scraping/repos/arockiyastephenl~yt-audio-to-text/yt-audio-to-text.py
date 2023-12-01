import wave
import openai
import pyaudio
from gtts import gTTS
import os
from pytube import YouTube
import streamlit as st

# Set up OpenAI API credentials
openai.api_key = "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"

# Set parameters for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

def download_audio_from_youtube(url: str, video_name: str) -> str:
    """Download the audio from a YouTube video and save it as an MP3 file."""
    video_url= YouTube(url)
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename

def play_audio(filename: str):
    """Play audio using PyAudio."""
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()

    p.terminate()

# Define the Streamlit app
def app():
    st.title("Speech to Text to Speech")

    # Get the YouTube video URL from the user
    url = st.text_input("Enter the YouTube video URL:")

    if url:
        st.write("Downloading audio from YouTube...")

        # Download the audio from the YouTube video and save it as an MP3 file
        VIDEO_NAME = "demo"
        audio_file_path = download_audio_from_youtube(url, VIDEO_NAME)

        st.write("Audio downloaded and saved to file:", audio_file_path)

        # Transcribe audio using Whisper
        audio_file = open(audio_file_path, "rb")
        st.write("Transcribing audio...")
        transcribed_text = openai.Audio.transcribe("whisper-1", audio_file)

        if "text" in transcribed_text:
            transcription = transcribed_text["text"]
            st.write("Transcription:", transcription)
        else:
            st.write("Transcription failed")

        # Send transcribed text as a query to OpenAI GPT-3 API
        st.write("Querying OpenAI GPT-3 API...")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=transcription,
            max_tokens=3000,
            n=1,
            frequency_penalty=1,
            presence_penalty=1,
            stop=None,
            temperature=0.8,
            top_p=1,
            best_of=1
        )

        # Get the response from OpenAI API
        api_response = response.choices[0].text

        # Print the API response
        st.write("API response:", api_response)

        # Generate audio from transcript
        language = 'en'
        tts = gTTS(text=api_response, lang=language)

        # Save audio to file
        filename = "output.mp3"
        tts.save(filename)

        # Play audio using PyAudio
        st.write("Playing audio...")
        play_audio(filename)

# Run the app
if __name__ == "__main__":
    app()
