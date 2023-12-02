import json
import tempfile
import requests
import boto3
import requests
import os, sys
from dotenv import load_dotenv
from pydub import AudioSegment
import numpy as np
import  speech_recognition as sr
import noisereduce as nr
import soundfile as sf
import io


import openai

sys.path.append('../')  # Add the parent directory to the system path
from chat import Interaction
from chat_agents import EfficientContextAgent
from agent_helpers import ContextManager

# OLD CODE:
# interaction = Interaction(task="create a new account", context_directory="./")
# interaction.recipient = "People's Gas"
# NEW IMPLEMENTATION:
context_manager = ContextManager()
context_manager.load_from_directory("./")
chat_agent = EfficientContextAgent("create a new account", "People's Gas", context_manager)


load_dotenv()

# access-key= MICPMAWAWF2KI1CRWU0B
# secret-key= DkLNyYz0uP1EJINhAizYIlRLzAgWMZSHzbH11RZY

""" Functions to help out with saving to wasabi/file uploads.
    upload_file_to_wasabi --> Upload mp3 file to wasabi
    get_url_recording --> Download mp3 file from url
    count_files_in_directory --> Count number of files in a directory, to set the name of the new file name.
"""
def save_message(call_id, message, sender):
    print('save_message')
    url = "http://127.0.0.1:8201/save_message"
    data = {"message": message, "sender": sender, "call_id": call_id}
    response_val = requests.post(url, data = json.dumps(data)) 
    print(response_val.text)
    return response_val.text


def upload_file_to_wasabi(file_path, bucket_name):
    s3 = boto3.client('s3',
                      endpoint_url='https://s3.us-west-1.wasabisys.com',  # Use the correct endpoint URL for your Wasabi region
                      aws_access_key_id='6UQ6BKLP89DNA5G37191',  # Replace with your access key
                      aws_secret_access_key='tpkQAodRS6LfjfC33VTF8GzhorewzhzfWuElr8sI')  # Replace with your secret key

    file_name = os.path.basename(file_path)

    try:
        s3.upload_file(file_path, bucket_name, file_name)
        print(f"File uploaded to Wasabi successfully!")
    except Exception as e:
        print("Something went wrong: ", e)

def combine_audios(audio_urls, output_filename):
    # Initialize an empty AudioSegment object
    output_audio = AudioSegment.empty()

    # Iterate over each audio URL and download it to memory
    audio_segments = []
    for audio_url in audio_urls:
        response = requests.get(audio_url)
        audio_bytes = io.BytesIO(response.content)
        audio_segment = AudioSegment.from_file(audio_bytes)
        audio_segments.append(audio_segment)

    # Concatenate the audio segments into a single output AudioSegment
    for audio_segment in audio_segments:
        output_audio += audio_segment

    # Export the output AudioSegment to a new audio file locally
    output_audio.export(output_filename, format="mp3")

    return output_filename

def detect_speech_with_noise_reduction(audio_url):
    # Download the audio file from the URL
    response = requests.get(audio_url)
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(response.content)
        temp_file.flush()

        # Load the downloaded audio file
        audio_data, sample_rate = sf.read(temp_file.name)

        # Apply noise reduction
        # reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)

        # Save the reduced noise audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as noise_reduced_file:
            sf.write(noise_reduced_file.name, audio_data, sample_rate)
            noise_reduced_file.flush()

            # Perform speech recognition on the reduced noise audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(noise_reduced_file.name) as source:
                audio = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio)
                print("Speech detected!")
                print("Transcribed text:", text)
                return True
            except sr.UnknownValueError:
                print("No speech detected.")
                return False

# def combine_audios(audio_urls):
#     combined = AudioSegment.empty()
    
#     for url in audio_urls:
#         # Download the audio file from the URL
#         response = requests.get(url)
#         file_name = "temp_audio.mp3"
        
#         with open(file_name, 'wb') as file:
#             file.write(response.content)
            
#         # Load audio file
#         audio = AudioSegment.from_mp3(file_name)

#         # Append audio file to combined audio
#         combined += audio

#     # Export combined audio file
#     num_files = count_files_in_directory("./outputs")
#     combined.export("outputs/output_{}.mp3".format(num_files), format='mp3')
#     return "outputs/output_{}.mp3".format(num_files)

def get_url_recording(url):
    response = requests.get(url, stream=True)
    print("creating a response")

    # Ensure the request is successful
    if response.status_code == 200:
        # Open the file in write-binary mode and write the response content to it
        with open('output.mp3', 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
    else:
        print('Failed to download the file.')



def count_files_in_directory(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

""" Functions to help out with Humes API calls for speech to text."""
def convert_speech_to_text(recording_locn):
    if os.path.isfile(recording_locn):
        return convert_speech_to_text_whisper_local(recording_locn)
    else:
        raise Exception("File does not exist")
    
def convert_speech_to_text_whisper_local(local_file_path):
    # Transcribe the audio using Whisper API
    with open(local_file_path, "rb") as file:
        transcript = openai.Audio.transcribe("whisper-1", file)
    return transcript.text
    
def convert_speech_to_text_whisper_url(recording_url):
    # Download the audio file from the URL
    response = requests.get(recording_url)
    print(recording_url)
    audio_file = response.content

    # Save the audio data to a file
    with open("temp.wav", "wb") as file:
        file.write(audio_file)

    # Transcribe the audio using Whisper API
    with open("temp.wav", "rb") as file:
        transcript = openai.Audio.transcribe("whisper-1", file)
    
    return transcript.text
    
def convert_speech_to_text_hume(recording_url):
    url = "https://api.hume.ai/v0/batch/jobs"

    payload = "{\"models\":{\"face\":{\"fps_pred\":3,\"prob_threshold\":0.99,\"identify_faces\":false,\"min_face_size\":60,\"save_faces\":false},\"prosody\":{\"granularity\":\"utterance\",\"identify_speakers\":false,\"window\":{\"length\":4,\"step\":1}},\"language\":{\"granularity\":\"word\",\"identify_speakers\":false},\"ner\":{\"identify_speakers\":false}},\"transcription\":{\"language\":null}," + "\"urls\":[\"" + recording_url + "\"],\"notify\":false}"
    headers = {
        "accept": "application/json; charset=utf-8",
        "content-type": "application/json; charset=utf-8",
        "X-Hume-Api-Key": os.getenv("HUME_API_KEY"),
    }

    response = requests.post(url, data=payload, headers=headers)
    return response

# Function to process the recording and generate a response
def process_recording(path, call_id):
    # Convert recording to text using speech-to-text API or library
    # Here, let's assume we have a function called `convert_speech_to_text` for this purpose
    recipient_message = convert_speech_to_text(path)

    generated_response = save_message(call_id, recipient_message)
    print("Done!")
    print("\t Generated response: ", generated_response)

    # Save the generated response as an audio url
    audio_url = save_generated_response_as_audio(generated_response)
    return audio_url, generated_response


# Function to save the generated response as an audio file
def save_generated_response_as_audio(generated_response):
    conversational_style_id = "6434632c9f50eacb088edafd"
    marcus_speaker_id = "643463179f50eacb088edaec"

    url = "https://api.fliki.ai/v1/generate/text-to-speech"
    headers = {
        "Authorization": f"Bearer {os.getenv('FLIKI_API_KEY')}",
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