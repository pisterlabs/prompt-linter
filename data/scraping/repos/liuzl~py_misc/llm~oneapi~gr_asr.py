# -*- encoding: utf-8 -*-
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_base = os.getenv("API_BASE")
openai.api_key = os.getenv("API_KEY")

import gradio as gr 
import json

def transcribe_audio(audio_path, prompt: str) -> str:
    with open(audio_path, 'rb') as audio_data:
        transcription = openai.Audio.transcribe("whisper-1", audio_data, prompt=prompt)
        print(json.dumps(transcription, ensure_ascii=False))
        return transcription['text']

def SpeechToText(audio):
    if audio == None : return "" 
    return transcribe_audio(audio, "Transcribe the following audio into Chinese: \n\n")

print("Starting the Gradio Web UI")
gr.Interface(
    title = 'ASR on Gradio Web UI', 
    fn=SpeechToText, 
    
    inputs=[
        gr.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox",
    ],
    live=True
).launch(debug=True)
