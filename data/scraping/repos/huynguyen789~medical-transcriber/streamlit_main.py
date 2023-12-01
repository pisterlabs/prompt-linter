import streamlit as st
from st_audiorec import st_audiorec
import whisper
import tempfile
import time
import os
import ffmpeg
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# TODO:
# 1. Recoding audio: Done
# 2. Transcribe audio file: Done
# 3. Generate summary of transcribed text: Done
# 4. Implement cache

anthropic = Anthropic()
anthropic.api_key = st.secrets["CLAUDE_API_KEY"]
model = whisper.load_model('base.en')


def claude_summary(text):
    completion = anthropic.completions.create(
        model="claude-2.0", #claude-2.0 #claude-instant-1.2
        max_tokens_to_sample=500,
        prompt=f"{HUMAN_PROMPT}Below is a doctor visit note, summarize it in bullet points. High priority on correct information. If the note is empty or too short, say it. DO NOT add extra or remove info.: {text}{AI_PROMPT}",
    )
    return(completion.completion)

def record_audio():
    wav_audio_data = st_audiorec()
    return wav_audio_data

def save_audio(wav_audio_data):
        with open(audio_file_path, 'wb') as f:
            f.write(wav_audio_data)
#######################################################


# Initialize Streamlit app
st.title("Doctor's Assistant")

# RECORD AUDIO:
#######################################################
wav_audio_data = record_audio()

# Create a audio file:
audio_file_path = 'audio.wav'
if wav_audio_data is not None:       
    save_audio(wav_audio_data)
    st.write("Audio file saved successfully.")
else:
    st.write("Please record or upload an audio file first.")    

# Initialize transcribed_text and summary
transcribed_text = ""
summary = ""

# Check if the transcribed text file exists
if os.path.exists('transcribed_text.txt'):
    with open('transcribed_text.txt', 'r') as f:
        transcribed_text = f.read()

# Check if the summary file exists
if os.path.exists('summary.txt'):
    with open('summary.txt', 'r') as f:
        summary = f.read()

#Transcribe
if st.button('Generate Transcription'):
    if "audio.wav" in os.listdir():
        with st.spinner('Generating transcription...'):
                result = model.transcribe(audio_file_path, fp16=False)
                transcribed_text = result['text']
                
                # Save the transcribed text to a file
                with open('transcribed_text.txt', 'w') as f:
                    f.write(transcribed_text)

# GENERATE SUMMARY:
if st.button('Generate Summary'):
    # Check if the transcribed text file exists
    if os.path.exists('transcribed_text.txt'):
        with open('transcribed_text.txt', 'r') as f:
            transcribed_text = f.read()
            
        with st.spinner("Generating summary..."):
            summary = claude_summary(transcribed_text)
            
            # Save the summary to a file
            with open('summary.txt', 'w') as f:
                f.write(summary)

# Display the transcribed text and summary
st.text_area("Transcribed Text:", value=transcribed_text, height=200)
st.text_area("Summary:", value=summary, height=300)