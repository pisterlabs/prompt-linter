# Import necessary libraries
import sounddevice as sd             # Library to record and play audio
import soundfile as sf               # Library to read and write sound files
import numpy as np                  # Library for numerical operations
import openai                       # OpenAI GPT-3 language model API
import os                           # Library for operating system functionalities
import requests                     # Library to send HTTP requests
import re                           # Library for regular expressions
from colorama import Fore, Style, init  # Library for colored terminal output
import datetime                     # Library for date and time operations
import base64                       # Library for base64 encoding and decoding
from pydub import AudioSegment      # Library to manipulate audio files
from pydub.playback import play     # Library to play audio files

init()  # Initialize colorama for colored terminal output

# Function to open and read a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# API keys are read from external files for security reasons
api_key = open_file('Your OpenAI GPT-3 API KEY HERE')  # OpenAI GPT-3 API key
elapikey = open_file('Your ElevenLabs API key HERE')    # ElevenLabs API key for text-to-speech

conversation1 = []  # List to store conversation history
chatbot1 = open_file('The initial message for the chatbot')  # foexample: Your Name is Ava.
                                                                      # You are an experienced life coach for cream-cheeese.
                                                                      # Your mission is to ask thoughtful questions and empower her to find inner peace and fulfillment.
                                                                      # You will ALWAYS converse in this structure:
                                                                      # Response: Here is where you respond to cream-cheeese. (limit your response to max 3 sentences).
                                                                      # Here is your context:
                                                                      # cream-cheeese has been seeking guidance from you for 5 weeks.
                                                                      # She has faced some challenges but is gradually finding his way to a more positive mindset.
                                                                      # cream-cheeese is feeling anxious about an upcoming important project at his workplace, and your support is crucial in helping him overcome this hurdle and achieve success.

# Function to interact with GPT-3 chat model
def chatgpt(api_key, conversation, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):
    # Set OpenAI API key
    openai.api_key = api_key

    # Append the user input to the conversation list
    conversation.append({"role": "user", "content": user_input})
    messages_input = conversation.copy()

    # Create a prompt message containing the chatbot's message
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])

    # Call OpenAI API to get the chatbot's response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)

    # Extract and return the chatbot's response
    chat_response = completion['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": chat_response})
    return chat_response

# Function to convert text to speech using ElevenLabs API
def text_to_speech(text, voice_id, api_key):
    # ElevenLabs API endpoint for text-to-speech
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}'

    # Headers required for the API request
    headers = {
        'Accept': 'audio/mpeg',
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }

    # Data for the API request, including the text to be converted to speech
    data = {
        'text': text,
        'model_id': 'eleven_monolingual_v1',
        'voice_settings': {
            'stability': 0.6,
            'similarity_boost': 0.85
        }
    }

    # Send a POST request to the ElevenLabs API to get the audio response
    response = requests.post(url, headers=headers, json=data)

    # If the response status code is 200, the request was successful
    # Save the audio response to 'output.mp3' and play it using pydub
    if response.status_code == 200:
        with open('output.mp3', 'wb') as f:
            f.write(response.content)
        audio = AudioSegment.from_mp3('output.mp3')
        play(audio)
    else:
        print('Error:', response.text)

# Function to print colored text in the terminal
def print_colored(agent, text):
    # Dictionary to map agent names to terminal text colors
    agent_colors = {
        "Julie:": Fore.YELLOW,  # Agent name "Julie" will be displayed in yellow
    }
    color = agent_colors.get(agent, "")
    print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")

voice_id1 = 'Your Voice ID'  # Replace this with the actual ElevenLabs voice ID

# Function to record audio and transcribe the speech using GPT-3
def record_and_transcribe(duration=8, fs=44100):
    print('Recording...')

    # Record audio using sounddevice library
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait for the recording to complete

    print('Recording complete.')

    filename = 'myrecording.wav'
    sf.write(filename, myrecording, fs)  # Save the recording to a WAV file

    # Read the saved audio file and transcribe the speech using GPT-3
    with open(filename, "rb") as file:
        openai.api_key = api_key
        result = openai.Audio.transcribe("whisper-1", file)
    transcription = result['text']
    return transcription

# Main loop for the chatbot interaction
while True:
    user_message = record_and_transcribe()  # Record user's audio input and transcribe it
    response = chatgpt(api_key, conversation1, chatbot1, user_message)  # Get chatbot's response
    print_colored("Julie:", f"{response}\n\n")  # Print chatbot's response in yellow
    # Remove any metadata and process text-to-speech for the chatbot's response
    user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
    text_to_speech(user_message_without_generate_image, voice_id1, elapikey)  # Convert response to speech

