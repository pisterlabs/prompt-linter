from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from flask import Flask, request
import requests
import time
import json
import openai
import os
from dotenv import load_dotenv

import sys
sys.path.append('../')  # Add the parent directory to the system path
from chat import Interaction

interaction = Interaction(task="create a new account", context_directory="../data/ekrem/")
interaction.recipient = "People's Gas"

# Load the variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up Twilio client
account_sid = os.getenv('ACCOUNT_SID')
auth_token = os.getenv('AUTH_TOKEN')
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
recipient_phone_number = os.getenv('RECIPIENT_PHONE_NUMBER')
hume_api_key = os.getenv('HUME_API_KEY')
fliki_api_key = os.getenv('FLIKI_API_KEY')

client = Client(account_sid, auth_token)

# Set up OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# ngrok_url = request.url_root
ngrok_url = "https://3dce-2607-f140-6000-11-7042-9d7-474a-1bff.ngrok-free.app"

# Function to handle incoming call
@app.route('/handle_incoming', methods=['POST'])
def handle_incoming_call():
    print("Handling incoming call...")
    response = VoiceResponse()
    response.say("Hello, how can I assist you?")
    print("Trying to record...", end="")
    response.record(max_length=30, action=f'{ngrok_url}process_recording')
    print("Done.")

def convert_speech_to_text(recording_url):
    url = "https://api.hume.ai/v0/batch/jobs"

    payload = "{\"models\":{\"face\":{\"fps_pred\":3,\"prob_threshold\":0.99,\"identify_faces\":false,\"min_face_size\":60,\"save_faces\":false},\"prosody\":{\"granularity\":\"utterance\",\"identify_speakers\":false,\"window\":{\"length\":4,\"step\":1}},\"language\":{\"granularity\":\"word\",\"identify_speakers\":false},\"ner\":{\"identify_speakers\":false}},\"transcription\":{\"language\":null}," + "\"urls\":[\"" + recording_url + "\"],\"notify\":false}"
    headers = {
        "accept": "application/json; charset=utf-8",
        "content-type": "application/json; charset=utf-8",
        "X-Hume-Api-Key": hume_api_key
    }

    response = requests.post(url, data=payload, headers=headers)
    return response.text

# Function to process the recording and generate a response
def process_recording(recording_url):
    print("Starting: process_recording...")
    # Convert recording to text using speech-to-text API or library
    # Here, let's assume we have a function called `convert_speech_to_text` for this purpose
    recipient_message = convert_speech_to_text(recording_url)

    print("Generating response...", end="")
    # Generate a response using OpenAI
    generated_response = interaction(recipient_message)
    print("Done!")
    print("\t Generated response: ", generated_response)

    print("Saving response as audio...", end="")
    # Save the generated response as an audio url
    audio_url = save_generated_response_as_audio(generated_response)
    print("Done!")

    print("Sending response to recipient...", end="")
    # Respond to the recipient with the generated answer
    response = VoiceResponse()
    response.play(audio_url)
    print("Done!")
    response.record(max_length=30, action='/process_recording')
    return str(response)

# Function to save the generated response as an audio file
def save_generated_response_as_audio(generated_response):
    conversational_style_id = "6434632c9f50eacb088edafd"
    marcus_speaker_id = "643463179f50eacb088edaec"

    url = "https://api.fliki.ai/v1/generate/text-to-speech"
    headers = {
        "Authorization": f"Bearer {fliki_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "content": generated_response,
        "voiceId": marcus_speaker_id,
        "voiceStyleId": conversational_style_id
    }
    
    response = requests.post(url, headers=headers, json=data)

    # Check the response status code
    if response.status_code == 200:
        # Process the response
        audio_data = response.content
        # Do something with the audio data
        response_dict = json.loads(audio_data)

        # Now you can access the dictionary elements
        success = response_dict["success"]
        audio_url = response_dict["data"]["audio"]
        duration = response_dict["data"]["duration"]
        
        return audio_url
    else:
        # Handle the error
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

# Twilio webhook to process the recording and generate a response
@app.route('/process_recording', methods=['POST'])
def process_recording_webhook():
    print("Processing process_recording_webhook...")
    recording_url = request.form['RecordingUrl']
    response = process_recording(recording_url)
    return response

@app.route('/call', methods=['GET'])
def call():
    print("Calling...")
    
    # Create a Twilio call
    twiml = VoiceResponse()
    twiml.say("Hello, how can I assist you?")
    twiml.record(maxLength="30", action="/handle_incoming")
    
    # Create a Twilio call
    call = client.calls.create(
        twiml=str(twiml),
        to=recipient_phone_number,
        from_=twilio_phone_number
    )
    return "Calling..."

# Start the Flask server to listen for incoming requests
if __name__ == '__main__':
    app.run()


