import gradio as gr
import openai
from pydub import AudioSegment
import requests
import config
import uuid
import base64
from gradio.outputs import HTML

openai.api_key = config.OPENAI_API_KEY

messages = [
    {"role": "system", "content": "You are a helpful assistant. Keep your responses to all inputs less than 50 words. Do not say you are an AI language model."},
]

def transcribe(audio):
    global messages
    audio_file = AudioSegment.from_file(audio)
    audio_file = audio_file.set_channels(1)  # Set audio to mono
    audio_file = audio_file.set_frame_rate(16000)  # Set frame rate to 16000
    audio_file = audio_file.set_sample_width(2)  # Set sample width to 2 bytes (16-bit)

    audio_file.export("temp_audio.wav", format="wav")

    with open("temp_audio.wav", "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
        print(transcript)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    system_message = response["choices"][0]["message"]["content"]

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ADVISOR_VOICE_ID}/stream"
    data = {
        "text": system_message.replace('"', ''),
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75
        }
    }

    r = requests.post(url, headers={'xi-api-key': config.ELEVEN_LABS_API_KEY}, json=data)

    output_filename = f"reply_{uuid.uuid4()}.mp3"
    with open(output_filename, "wb") as output:
        output.write(r.content)

    chat_transcript = ""
    for i, message in enumerate(messages):
        if i == 0 and message['role'] == 'system':
            continue
        chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    # Convert the mp3 file to base64
    with open(output_filename, "rb") as output_file:
        base64_audio = base64.b64encode(output_file.read()).decode("utf-8")

    # Create an HTML audio element to play the base64-encoded audio
    audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3"></audio>'

    return chat_transcript + "assistant: " + system_message + "\n\n", audio_html

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs=["text", HTML()]).launch(share=True)
