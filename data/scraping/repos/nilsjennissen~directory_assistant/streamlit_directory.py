#% STREAMLIT DOCUMENT HELPER APP

# Importing libraries
import os
import wave
import time
import openai
import pygame
import pyaudio
import requests
import tempfile
import credentials
import pandas as pd
from gtts import gTTS
import streamlit as st



#%%    PATH AND FILE SETTINGS
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "audio/audio_output.wav")

#%%    PROMPT ENGINEERING
preprompt = "You are a csv document reader giving information about the currently loaded csv table."

#%%    DEVICE SETTINGS
# Get channel Macbook Pro Microphone
def get_cur_mic():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'pro microphone' in dev['name'].lower():
            return i
    return None

def get_speaker():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'speaker' in dev['name'].lower():
            return i
    return None


#%%    RECORDING FUNCTIONS
def audio_recording(num_seconds):
    ''' Record audio for num_seconds and save it to a wav file'''

    script_dir = os.path.dirname(os.path.abspath(__file__))

    chunk = 2205
    channels = 1
    fs = 44100
    seconds = max(num_seconds, 0.1)
    sample_format = pyaudio.paInt16
    filename = os.path.join(script_dir, "audio/audio_output.wav")

    print(f'\n... Recording {seconds} seconds of audio initialized ...\n')

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=get_cur_mic(),
                    frames_per_buffer=chunk,
                    input=True)


    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk, False)
        frames.append(data)


    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()



#%%    TRANSCRIPT FUNCTIONS
def get_transcript_whisper():
    '''Get the transcript of the audio file'''
    openai.api_key = credentials.api_key
    file = open(filename, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]

    return transcribed_text

# %%    CHATGPT FUNCTIONS
def run_chatGPT(prompt):
    '''Run chatGPT with the prompt and return the response'''
    openai.api_key = credentials.api_key
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": preprompt + prompt},
        ]
    )
    answer = completion.choices[0].message.content

    return answer


#%%    SPEAK FUNCTIONS
def speak_answer(answer):
    # Initialize pygame mixer
    pygame.mixer.init()
    '''Speak the answer'''
    tts = gTTS(text=answer, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as f:
        tts.save(f.name)
        pygame.mixer.music.load(f.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


#%% STREAMLIT APP
def main():
    st.title("CSV Document Helper")
    st.write("Talk to your CSV-File by uploading a document and start recording or typing your question.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
        except Exception as e:
            st.write("Error reading CSV file:", e)

        user_input = st.text_input("Type your question here:")

        start_recording_pressed = st.button("Start Recording")
        if start_recording_pressed or user_input:
            if start_recording_pressed:
                try:
                    audio_recording(6)
                except KeyboardInterrupt:
                    pass

                transcript = get_transcript_whisper()
            else:
                transcript = user_input

            st.write("Transcript:", transcript)

            if any(word in transcript for word in ['stop', 'Stop', 'exit', 'quit', 'end']):
                st.write("... Script stopped by user")
                exit()

            transcript = f' {transcript}'
            answer = run_chatGPT(transcript)
            st.write("AI Response:", answer)
            st.write("Playing AI response...")
            speak_answer(answer)
            st.write("Finished playing AI response.")
        else:
            st.write("Press 'Start Recording' to begin or type your question and press Enter.")
    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()