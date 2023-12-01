import gradio
import openai
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

messages = [{"role": "system", "content": "Hello nice to meet you!"}]

def transcribe(audio):
    try: 
        audio_file = open(audio, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file) # use whisper to get audio transcription
    except:
        transcript = "" 

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]  # get response from openai
    messages.append(system_message)

    subprocess.call(["say", system_message['content']]) # speak content of system messages

    chat_transcription = ""
    for message in messages:
        if message['role'] != "system":
            chat_transcription += f"{message['role']}: {message['content']} \n\n"
    
    return chat_transcription

demo = gradio.Interface(fn=transcribe, inputs=gradio.Audio(source="microphone", type="filepath"), outputs="text").launch()
demo.launch()
