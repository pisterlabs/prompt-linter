from flask import Flask, render_template, request
import openai
from openai.error import ServiceUnavailableError
import threading

import io
from elevenlabs import voices, generate, play
from elevenlabs import set_api_key
set_api_key("da4e1089136abf9605b8ac4fc369f840")


app = Flask(__name__)
openai.api_key = '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'  # Replace with your OpenAI API key

conversation = [
    {"role": "system", "content": "I'm a skincare chatbot. Please ask me skin-related questions. I don't know anything else. I don't answer to anything other than skin-related questions. My answers are precise, small and concise."},
]

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.form['user_query'] + ", be precise, small and concise."
    response = generate_response(user_query)
    return {'response': response}

@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("Transcribing...")
    audio_blob = request.files['audio_blob']
    audio_file =  audio_blob.read()
    buffer = io.BytesIO(audio_file)
    buffer.name = "audio.m4a"
    result = openai.Audio.transcribe("whisper-1", buffer)
    transcript = result["text"]
    print(transcript)
    response = generate_response(transcript)
    return {'prompt': transcript , 'response': response}

def generate_response(user_query):
    global conversation
    conversation.append({"role": "user", "content": user_query})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        reply = response.choices[0].message['content']
        conversation.append({"role": "system", "content": reply})

        # Play audio in the background using threading
        audio_thread = threading.Thread(target=play_audio, args=(reply,))
        audio_thread.start()

        return reply.strip()
    except ServiceUnavailableError:
        return "Sorry, the chatbot is temporarily unavailable. Please try again later."

def play_audio(text):
    audio = generate(
        text=text,
        voice="Bella",
        model="eleven_monolingual_v1"
    )
    play(audio)

if __name__ == '__main__':
    app.run()
