import streamlit as st
import pyaudio
import wave
import openai
from gtts import gTTS
import os

# Set up OpenAI API credentials
openai.api_key = "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"

# Set parameters for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define a function to record audio and save it to a WAV file
def record_audio():
    # Open audio stream from microphone
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.info("Speak now...")

    # Record audio for specified time
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.info("Done recording. Processing...")

    # Stop audio stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to WAV file
    filename = "audio.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

# Define a function to transcribe audio using OpenAI API
def transcribe_audio(audio_file):
    # Transcribe audio using OpenAI API
    transcribed_text = openai.Audio.transcribe("whisper-1", audio_file)
    if "text" in transcribed_text:
        transcription = transcribed_text["text"]
        st.success("Transcription: " + transcription)
        return transcription
    else:
        st.error("Transcription failed")
        return None

# Define a function to generate audio from transcript
def generate_audio(transcription):
    language = 'en'
    tts = gTTS(text=transcription, lang=language)

    # Save audio to file
    filename = "output.mp3"
    tts.save(filename)

    # Play audio using default media player
    os.startfile(filename)

# Define the Streamlit app
def app():
    st.title("Voice-to-Text using OpenAI")

    # Record audio and transcribe
    if st.button("Record"):
        filename = record_audio()
        with open(filename, "rb") as audio_file:
            transcription = transcribe_audio(audio_file)
            if transcription:
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=transcription,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )

                # Get the response from OpenAI API
                api_response = response.choices[0].text

                # Print the API response
                st.success("OpenAI API response: " + api_response)

                # Generate audio from transcript
                generate_audio(api_response)

# Run the Streamlit app
if __name__ == '__main__':
    app()
