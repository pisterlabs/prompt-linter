import gradio as gr
import os
import openai
import config
import subprocess

openai.api_key = config.OPENAI_API_KEY

messages = [{"role": "system", "content": 'You are a therapist. Respond to all input in 25 words or less.'}]

def transcribe(audio):
    global messages

    if audio is None:
        return "Error: No audio file provided."

    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)

    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    subprocess.call(["say", system_message['content']])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")
ui.launch()
