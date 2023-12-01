import sounddevice as sd
import soundfile as sf
from pathlib import Path
import openai
import os
import re
from colorama import Fore, Style, init
from pydub import AudioSegment
from dotenv import load_dotenv
import subprocess
from tempfile import NamedTemporaryFile


init()

# Get Open AI API Key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load environment variables from .env file
load_dotenv()

conversation1 = []  

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

chatbot1 = open_file('chatbot1.txt')

def chatgpt(conversation, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):

    conversation.append({"role": "user","content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input)
    chat_response = completion.choices[0].message.content
    #chat_response = completion['choices'][0]['message']['content']
    print(f"Chat GPT Response: {chat_response}\n\n")
    conversation.append({"role": "assistant", "content": chat_response})
    return chat_response

PLAYER = "ffplay"  

def _play_with_ffplay(seg):
    with NamedTemporaryFile("w+b", suffix=".wav", delete=False) as f:
        temp_name = f.name  # Save the name (full path) of the temporary file to the variable temp_name. The name property of the file object f gives us the full path.
    seg.export(temp_name, "wav")  # Outside the with block (meaning after the temporary file has been closed), this line uses pydub's export method to write the audio data from the seg AudioSegment to the temporary file at the path temp_name in WAV format.

    # The next line below uses Python's subprocess.call to execute the ffplay command-line tool and play the temporary WAV file. 
    # PLAYER is a variable that should contain the string "ffplay". The arguments "-nodisp", "-autoexit", and "-hide_banner" are options for ffplay:

    # "-nodisp" tells ffplay not to display the video window (since it's audio only).
    # "-autoexit" makes ffplay close automatically after the audio finishes playing.
    # "-hide_banner" suppresses the printing of the ffplay banner to the console.
        
    subprocess.call([PLAYER, "-nodisp", "-autoexit", "-hide_banner", temp_name])

    os.unlink(temp_name)  # Delete the temporary file after playback

def text_to_speech(text):

    speech_file_path = Path(__file__).parent / "speech.wav"
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    response.stream_to_file(speech_file_path)

    # Load your audio file
    audio = AudioSegment.from_file("C:/Users/juant/Desktop/Projects/Speaking with ChatGPT/speech.wav")

    # Use the custom play function
    _play_with_ffplay(audio)

def print_colored(agent, text):
    agent_colors = {
        "Julie:": Fore.YELLOW,
    }
    color = agent_colors.get(agent, "")
    print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")

def record_and_transcribe(duration=8, fs=44100):
    print('Recording...')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    print('Recording complete.')
    filename = 'myrecording.wav'
    sf.write(filename, myrecording, fs)
    with open(filename, "rb") as file:
        result = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=file
        )
    transcription = result.text
    print(f"Transcription: {transcription}")
    return transcription

while True:
    user_message = record_and_transcribe()
    response = chatgpt(conversation1, chatbot1, user_message)
    print_colored("Chat GPT:", f"{response}\n\n")
    user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
    text_to_speech(user_message_without_generate_image)